"""
M3 / M2-C Phase 2 编码管线核心 —— 防泄漏的故事级 3 折 CV。

严格按 frozen/analysis_spec.yaml：
  下采样+trim 后的 TR 级特征
    → StandardScaler + PCA-100  (apply_before_fir=true, fit 仅外层训练故事)
    → FIR 延迟 2/4/6/8s        (故事内，不跨故事)
    → himalaya RidgeCV          (per-voxel λ, inner 2-fold, λ∈logspace(-2,7,19))
    → 逐 held-out story 单独算 pearson r (common_scoring_mask: >100s ∩ FIR_valid)
    → story 间 effective-TR 加权平均为 fold 级 r
    → 3 外折再 effective-TR 加权平均为最终 r

story-then-fold 的两层加权平均均调用同一 effective_tr_weighted_mean（见
milestone/里程碑总览_V4.9_最终冻结版.md M3 模块4："先对每个 held-out story
单独计算 voxel r，再形成 fold 级汇总"）。ROI 的 fisher-z 平均只在 roi_mean_r
里做一次（对 voxel），不会和 story/fold 的加权平均混在一起重复变换。

本模块为**纯数值核心**：输入是已对齐好的 per-story 特征/响应字典，不碰文件、
不做下采样（那是 assemble.py 的事）。solver 可插拔——本地测试用 numpy ridge，
服务器生产用 himalaya GPU。所有 PCA/scaler/CV 仅在外层训练折内 fit，杜绝泄漏。
"""

from __future__ import annotations

import time
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
    roi_mean_fisherz, weighted_mean_scalar, fisher_z_inv,
)

LAMBDA_GRID = np.logspace(-2, 7, 19)   # M3a 后扩上界（原 logspace(-2,4,13)），
                                       # 见 frozen/analysis_spec.yaml 变更记录
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
class StoryScore:
    """单个 held-out story 的评分（M4 每 story 独立保存 / M5 story-level bootstrap 单位）。"""
    story: str
    voxel_r: np.ndarray              # <float>[V]  该 story 每体素 r
    roi_z: dict                      # {roi_name: fisher-z 空间 ROI 标量}
    n_eff_tr: int                    # 该 story 有效评分 TR 数（加权用）
    scoring_mask: np.ndarray | None = None  # <bool>[L] 该 story 实际评分 mask（供 normal/shift
                                            # 逐元素对比：n_eff 相等只是必要条件，mask 本身
                                            # 相等才是共同 mask 的充分证据，见 m4_driver 断言）


@dataclass
class FoldResult:
    test_stories: list[str]
    story_scores: list[StoryScore]   # per-story（下沉的 ROI 聚合 + M5 bootstrap 单位）
    voxel_r: np.ndarray              # <float>[V]  fold 级：per-story voxel_r 有效TR加权
    roi_z: dict                      # {roi_name: fold 级 ROI z（per-story roi_z 有效TR加权）}
    valphas: np.ndarray              # <float>[V]
    n_eff_tr: int                    # 测试集有效评分 TR 数
    evr_at_k: float | None = None    # 训练折 PCA 前 pca_k 个成分累计解释方差比（M3 诊断用，
                                     # 从已 fit 的 PCA 直接读取，不改变任何数值计算）


@dataclass
class CVResult:
    voxel_r: np.ndarray              # <float>[V]  跨折加权平均（voxel-level 全脑图用）
    roi_z: dict                      # {roi_name: 跨折 ROI z}
    roi_r: dict                      # {roi_name: tanh(roi_z)，展示用}
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
        Y_in_cpu=True,                  # RidgeCV 顶层参数（不可放 solver_params）：响应留 CPU
        solver_params=dict(
            score_func=correlation_score,  # spec: selection_metric=validation_pearson_r
            local_alpha=True,           # per-voxel alpha（spec: alpha_scope=per_voxel）
            n_targets_batch=5000,       # 95556 体素分块，避免全量摊平 OOM
            n_targets_batch_refit=2000, # 全训练集 refit 阶段同样分块
            n_alphas_batch=5,           # 13 个 λ 也分块，进一步控显存
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
             solver, *, roi_columns=None, pca_k=PCA_K, lambda_grid=LAMBDA_GRID,
             inner_folds=INNER_FOLDS, delays_s=DELAYS_S, tr=TR_SECONDS,
             after_s=AFTER_S, seed=0, verbose=True, tag="",
             shift_valid_by_story: dict[str, np.ndarray] | None = None) -> FoldResult:
    """单外折：训练折 fit scaler/PCA/ridge（一次），测试折逐故事打分再汇总。全程无泄漏。

    拟合/预测不按故事拆分（训练折所有故事拼接后一次 fit，测试折一次 predict）；
    只有**评分**按故事切片。ROI 与 voxel 两条独立聚合路径（对齐冻结文档 M3 模块4）：
      - **每 story**：算 voxel r；再对每个 ROI 做 fisher-z→ROI 内均值 → 该 story 的
        ROI z 标量（roi_columns 提供时）。
      - **fold 级 voxel r**：per-story voxel r 按有效 TR 数加权平均（spec:
        fold_summary=effective_tr_weighted_mean，voxel 空间算术加权，供全脑图）。
      - **fold 级 ROI z**：per-story ROI z 按有效 TR 数在 **z 空间**加权平均（ROI 聚合
        始终在 story 级完成，不从 fold voxel r 反推——两者 fisher-z/加权顺序不等价）。
    per-story 结果全部保留在 FoldResult.story_scores（M4 每 story 保存、M5 bootstrap 单位）。
    roi_columns=None 时只算 voxel（兼容不需要 ROI 的调用/测试）。

    shift_valid_by_story：可选 {story: <bool>[T]}（T=该故事下采样后、FIR前的
    TR 行数），供 M3b/M4 的 40s time-shift 负控制用。若给出，测试故事评分时
    与 FIR-valid、>100s 取交集（common_scoring_mask 的 shift_valid 参数），
    即 normal/shifted 使用共同有效 mask（冻结文档验收标准5）。是否位移特征
    本身（X_shifted(t)=X(t-40s)）由调用方在 story_data 里传入已位移的 X 决定，
    本函数不做位移，只负责把位移导致的边缘无效点纳入评分 mask。

    verbose=True 打印各阶段起止，避免 himalaya fit() 期间日志静默让人误以为卡死。
    """
    roi_columns = roi_columns or {}
    t0 = time.time()
    pfx = f"[fold{tag}]" if tag else "[fold]"
    if verbose:
        print(f"{pfx} 训练={len(train_stories)}故事 测试={test_stories} "
              f"PCA/scaler 拟合中...", flush=True)

    # 1) scaler + PCA 仅在训练故事的 pre-FIR 特征上 fit
    Xtr_raw = np.vstack([story_data[s].X for s in train_stories])
    scaler = StandardScaler().fit(Xtr_raw)
    pca = PCA(n_components=pca_k, svd_solver="full", random_state=seed)
    pca.fit(scaler.transform(Xtr_raw))
    evr_at_k = float(pca.explained_variance_ratio_.sum())  # 诊断量，不参与拟合
    del Xtr_raw

    # 2) 训练折：transform→FIR；训练只用 FIR-valid（不施加 >100s 评分 mask）
    Xtr_f, Ytr, vtr, _, _ = _transform_and_fir(
        story_data, train_stories, scaler, pca, delays_s, tr)
    Xtr_f, Ytr = Xtr_f[vtr], Ytr[vtr]

    # 3) 测试折：transform→FIR（评分 mask 留到第 5 步逐故事构建）
    Xte_f, Yte, vte, trt_te, lens = _transform_and_fir(
        story_data, test_stories, scaler, pca, delays_s, tr)

    if verbose:
        print(f"{pfx} 特征就绪 Xtr={Xtr_f.shape} Xte={Xte_f.shape} "
              f"({time.time()-t0:.1f}s)，调用 solver.fit()（无逐次打印，"
              f"可 nvidia-smi 观察 GPU 是否在动）...", flush=True)

    # 4) ridge：训练折拟合一次，测试折全量预测一次（拟合/预测不按故事拆分）
    t_solver = time.time()
    pred_te, valphas = solver(Xtr_f, Ytr, Xte_f, lambda_grid, inner_folds, seed)

    # 5) story-level 评分：每 story 单独算 voxel r + ROI z（fisher-z 在 story 级完成）
    story_scores, off = [], 0
    for s, L in zip(test_stories, lens):
        seg = slice(off, off + L)
        shift_valid_s = None
        if shift_valid_by_story is not None:
            shift_valid_s = shift_valid_by_story[s]
            if len(shift_valid_s) != L:
                raise ValueError(
                    f"[{s}] shift_valid 长度 {len(shift_valid_s)} != 该故事 TR 数 {L}")
        m = common_scoring_mask(story_data[s].tr_times, vte[seg],
                                shift_valid=shift_valid_s, after_s=after_s)
        off += L
        if m.sum() == 0:
            continue
        v_r = voxelwise_pearson(pred_te[seg][m], Yte[seg][m])
        roi_z = {name: roi_mean_fisherz(v_r, cols)
                 for name, cols in roi_columns.items()}
        story_scores.append(StoryScore(story=s, voxel_r=v_r, roi_z=roi_z,
                                       n_eff_tr=int(m.sum()), scoring_mask=m.copy()))
    if not story_scores:
        raise ValueError(f"折内所有测试故事评分点数为 0: {test_stories}")

    neff = [ss.n_eff_tr for ss in story_scores]
    # fold 级 voxel r：voxel 空间有效TR加权（算术）
    fold_voxel_r = effective_tr_weighted_mean(
        [ss.voxel_r for ss in story_scores], neff)
    # fold 级 ROI z：z 空间有效TR加权（ROI 聚合始终在 story 级，不从 fold voxel r 反推）
    fold_roi_z = {name: weighted_mean_scalar(
                      [ss.roi_z[name] for ss in story_scores], neff)
                  for name in roi_columns}
    n_eff_tr = int(sum(neff))

    if verbose:
        roi_show = "  ".join(f"{n}={fisher_z_inv(z):.4f}"
                             for n, z in fold_roi_z.items())
        print(f"{pfx} 完成 solver={time.time()-t_solver:.1f}s "
              f"总计={time.time()-t0:.1f}s mean_r={fold_voxel_r.mean():.4f} "
              f"({len(story_scores)}/{len(test_stories)} 故事)  {roi_show}", flush=True)

    return FoldResult(test_stories=list(test_stories), story_scores=story_scores,
                      voxel_r=fold_voxel_r, roi_z=fold_roi_z,
                      valphas=valphas, n_eff_tr=n_eff_tr, evr_at_k=evr_at_k)


def run_encoding_cv(story_data: dict[str, StoryData],
                    folds: list[tuple[list[str], list[str]]],
                    solver, *, verbose=True, **kw) -> CVResult:
    """全 3 折 CV：逐折 run_fold，voxel r 与 ROI z 各自按有效 TR 数跨折加权平均。"""
    fold_results = []
    for i, (tr_s, te_s) in enumerate(folds, 1):
        if verbose:
            print(f"[cv] === 折 {i}/{len(folds)} ===", flush=True)
        fold_results.append(run_fold(story_data, tr_s, te_s, solver,
                                     verbose=verbose, tag=f" {i}/{len(folds)}", **kw))
    w = [fr.n_eff_tr for fr in fold_results]
    voxel_r = effective_tr_weighted_mean([fr.voxel_r for fr in fold_results], w)
    roi_names = fold_results[0].roi_z.keys()
    roi_z = {name: weighted_mean_scalar([fr.roi_z[name] for fr in fold_results], w)
             for name in roi_names}
    roi_r = {name: float(fisher_z_inv(z)) for name, z in roi_z.items()}
    return CVResult(voxel_r=voxel_r, roi_z=roi_z, roi_r=roi_r, folds=fold_results)
