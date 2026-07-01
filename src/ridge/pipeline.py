"""
M3 / M2-C Phase 2 编码管线核心 —— 防泄漏的故事级 3 折 CV。

严格按 frozen/analysis_spec.yaml：
  下采样+trim 后的 TR 级特征
    → StandardScaler + PCA-100  (apply_before_fir=true, fit 仅外层训练故事)
    → FIR 延迟 2/4/6/8s        (故事内，不跨故事)
    → himalaya RidgeCV          (per-voxel λ, inner 2-fold, λ∈logspace(-2,4,13))
    → 测试故事每体素 pearson r  (common_scoring_mask: >100s ∩ FIR_valid)
    → 3 外折 effective-TR 加权平均

本模块为**纯数值核心**：输入是已对齐好的 per-story 特征/响应字典，不碰文件、
不做下采样（那是 assemble.py 的事）。solver 可插拔——本地测试用 numpy ridge，
服务器生产用 himalaya GPU。所有 PCA/scaler/CV 仅在外层训练折内 fit，杜绝泄漏。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.fmri.alignment import apply_fir          # noqa: E402
from src.fmri.mask import common_scoring_mask     # noqa: E402
from src.ridge.score import (                      # noqa: E402
    voxelwise_pearson, effective_tr_weighted_mean,
)

LAMBDA_GRID = np.logspace(-2, 4, 13)
DELAYS_S = (2, 4, 6, 8)
TR_SECONDS = 2.0
AFTER_S = 100.0
PCA_K = 100
INNER_FOLDS = 2


@dataclass
class StoryData:
    """单故事已对齐的 TR 级数据（assemble.py 产出）。"""
    X: np.ndarray            # <float>[T, D_feat]  下采样+trim 后的特征（pre-PCA, pre-FIR）
    Y: np.ndarray            # <float>[T, V]       已 trim 的 BOLD 响应
    tr_times: np.ndarray     # <float>[T]          与 Y 对齐的 TR 中心时间（用于 >100s）


@dataclass
class FoldResult:
    test_stories: list[str]
    voxel_r: np.ndarray              # <float>[V]
    valphas: np.ndarray              # <float>[V]
    n_eff_tr: int                    # 测试集有效评分 TR 数（加权用）


@dataclass
class CVResult:
    voxel_r: np.ndarray              # <float>[V]  跨折加权平均
    folds: list[FoldResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 参考 solver（纯 numpy，本地单测用；生产用 himalaya_solver）
# ---------------------------------------------------------------------------

def numpy_ridgecv_solver(
    Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray,
    lambda_grid: np.ndarray, inner_folds: int, seed: int,
):
    """per-voxel λ 选择的纯 numpy ridge（仅供本地测试 / himalaya 不可用时回退）。

    内层 KFold（按行连续切块，时间序列安全），按 validation pearson r 选 λ，
    平手取较大 λ（spec: tie_break=prefer_larger_lambda）。返回 (pred_te, best_lambdas)。
    """
    n, d = Xtr.shape
    v = Ytr.shape[1]
    fold_sizes = np.full(inner_folds, n // inner_folds)
    fold_sizes[: n % inner_folds] += 1
    bounds = np.cumsum(np.concatenate([[0], fold_sizes]))

    # (n_lambda, v) 累积验证 r
    val_r = np.zeros((len(lambda_grid), v))
    for fi in range(inner_folds):
        va = slice(bounds[fi], bounds[fi + 1])
        tr_idx = np.ones(n, dtype=bool)
        tr_idx[va] = False
        Xt, Yt = Xtr[tr_idx], Ytr[tr_idx]
        Xv, Yv = Xtr[va], Ytr[va]
        XtX = Xt.T @ Xt
        XtY = Xt.T @ Yt
        eye = np.eye(d)
        for li, lam in enumerate(lambda_grid):
            W = np.linalg.solve(XtX + lam * eye, XtY)    # (d, v)
            pred = Xv @ W
            val_r[li] += voxelwise_pearson(pred, Yv)

    # prefer_larger_lambda：从大到小找首个达到最大的 λ
    order = np.argsort(lambda_grid)[::-1]                 # 大→小
    best_li = np.zeros(v, dtype=int)
    best_val = np.full(v, -np.inf)
    for li in order:
        better = val_r[li] > best_val + 1e-12
        best_li[better] = li
        best_val[better] = val_r[li][better]
    best_lambdas = lambda_grid[best_li]

    # 用全训练集按所选 λ 拟合并预测测试集
    XtX = Xtr.T @ Xtr
    XtY = Xtr.T @ Ytr
    eye = np.eye(d)
    pred_te = np.zeros((Xte.shape[0], v))
    for lam in np.unique(best_lambdas):
        cols = np.nonzero(best_lambdas == lam)[0]
        W = np.linalg.solve(XtX + lam * eye, XtY[:, cols])
        pred_te[:, cols] = Xte @ W
    return pred_te, best_lambdas


def himalaya_ridgecv_solver(
    Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray,
    lambda_grid: np.ndarray, inner_folds: int, seed: int,
):
    """生产 solver：himalaya RidgeCV（per-target λ via inner CV）。惰性导入。

    显存管理：95556 体素若不分块，CV 内部 (n_samples, n_targets) 预测矩阵会
    一次性摊平实例化，实测单次分配 18.27GB 导致 OOM（24GB 卡上无法容纳）。
    用 solver_params 按体素/alpha 分块 + Y_in_cpu 保留响应在 CPU 按批传输，
    思路与 Phase1 GPU 脚本的 vox_chunk 显存策略一致。
    """
    from himalaya.ridge import RidgeCV
    from himalaya.backend import set_backend
    from himalaya.scoring import correlation_score
    from sklearn.model_selection import KFold

    # set_backend 返回 backend 对象；用其 to_numpy 把 CUDA tensor 转回 numpy，
    # 否则 np.asarray(cuda_tensor) 会抛 "can't convert cuda tensor to numpy"。
    backend = set_backend("torch_cuda", on_error="warn")
    cv = KFold(n_splits=inner_folds)     # 时间序列连续块，不打乱
    model = RidgeCV(
        alphas=lambda_grid, cv=cv, fit_intercept=False,
        solver_params=dict(
            score_func=correlation_score,  # spec: selection_metric=validation_pearson_r
            local_alpha=True,           # per-voxel alpha（spec: alpha_scope=per_voxel）
            n_targets_batch=5000,       # 95556 体素分块，避免全量摊平 OOM
            n_targets_batch_refit=2000, # 全训练集 refit 阶段同样分块
            n_alphas_batch=5,           # 13 个 λ 也分块，进一步控显存
            Y_in_cpu=True,              # 响应矩阵留 CPU，按批传输到 GPU
        ),
    )
    model.fit(Xtr, Ytr)
    pred_te = backend.to_numpy(model.predict(Xte))
    best_lambdas = backend.to_numpy(model.best_alphas_)
    return np.asarray(pred_te), np.asarray(best_lambdas)


# ---------------------------------------------------------------------------
# 单折 + 全 CV
# ---------------------------------------------------------------------------

def _transform_and_fir(story_data: dict[str, StoryData], stories: list[str],
                       scaler, pca, delays_s, tr):
    """对若干故事：scaler→PCA→FIR（逐故事），拼接。返回 (Xfir, Y, valid, tr_times, per_story_T)。"""
    Xfir_parts, Y_parts, valid_parts, trt_parts, lens = [], [], [], [], []
    for s in stories:
        sd = story_data[s]
        Z = pca.transform(scaler.transform(sd.X))           # (T, k)
        Xf, valid = apply_fir(Z, delays_s=delays_s, tr=tr)  # (T, k*ndelays), (T,)
        Xfir_parts.append(Xf)
        Y_parts.append(sd.Y)
        valid_parts.append(valid)
        trt_parts.append(sd.tr_times)
        lens.append(sd.X.shape[0])
    return (np.vstack(Xfir_parts), np.vstack(Y_parts),
            np.concatenate(valid_parts), np.concatenate(trt_parts), lens)


def run_fold(story_data: dict[str, StoryData],
             train_stories: list[str], test_stories: list[str],
             solver, *, pca_k=PCA_K, lambda_grid=LAMBDA_GRID,
             inner_folds=INNER_FOLDS, delays_s=DELAYS_S, tr=TR_SECONDS,
             after_s=AFTER_S, seed=0) -> FoldResult:
    """单外折：训练折 fit scaler/PCA/ridge，测试折打分。全程无泄漏。"""
    # 1) scaler + PCA 仅在训练故事的 pre-FIR 特征上 fit
    Xtr_raw = np.vstack([story_data[s].X for s in train_stories])
    scaler = StandardScaler().fit(Xtr_raw)
    pca = PCA(n_components=pca_k, svd_solver="full", random_state=seed)
    pca.fit(scaler.transform(Xtr_raw))
    del Xtr_raw

    # 2) 训练折：transform→FIR；训练只用 FIR-valid（不施加 >100s 评分 mask）
    Xtr_f, Ytr, vtr, _, _ = _transform_and_fir(
        story_data, train_stories, scaler, pca, delays_s, tr)
    Xtr_f, Ytr = Xtr_f[vtr], Ytr[vtr]

    # 3) 测试折：transform→FIR；评分用 common_scoring_mask（>100s ∩ FIR_valid）
    Xte_f, Yte, vte, trt_te, lens = _transform_and_fir(
        story_data, test_stories, scaler, pca, delays_s, tr)
    # 逐故事构建 >100s∩FIR mask，再拼接（mask 须按故事算 tr_times）
    score_mask_parts, off = [], 0
    for s, L in zip(test_stories, lens):
        seg = slice(off, off + L)
        m = common_scoring_mask(story_data[s].tr_times, vte[seg], after_s=after_s)
        score_mask_parts.append(m)
        off += L
    score_mask = np.concatenate(score_mask_parts)

    # 4) ridge：训练折拟合，测试折全量预测，再按评分 mask 取子集算 r
    pred_te, valphas = solver(Xtr_f, Ytr, Xte_f, lambda_grid, inner_folds, seed)
    r = voxelwise_pearson(pred_te[score_mask], Yte[score_mask])

    return FoldResult(test_stories=list(test_stories), voxel_r=r,
                      valphas=valphas, n_eff_tr=int(score_mask.sum()))


def run_encoding_cv(story_data: dict[str, StoryData],
                    folds: list[tuple[list[str], list[str]]],
                    solver, **kw) -> CVResult:
    """全 3 折 CV：逐折 run_fold，按有效 TR 数加权平均每体素 r。"""
    fold_results = [run_fold(story_data, tr_s, te_s, solver, **kw)
                    for tr_s, te_s in folds]
    voxel_r = effective_tr_weighted_mean(
        [fr.voxel_r for fr in fold_results],
        [fr.n_eff_tr for fr in fold_results],
    )
    return CVResult(voxel_r=voxel_r, folds=fold_results)
