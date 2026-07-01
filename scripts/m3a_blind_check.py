"""
M3a 盲态参数核查 —— 只用 outer training stories + inner CV，不碰 held-out r。

按 milestone/里程碑总览_V4.9_最终冻结版.md 里程碑3 M3a 的 P0 纪律：
  - 只在 outer training stories 内运行 scaler / PCA / inner CV；
  - 检查 λ 边界命中率、PCA 维度/解释方差、NaN、零方差、solver 与内存；
  - **不生成、保存或展示任何 outer held-out r**（本脚本从不 predict 测试故事、
    从不 load 测试故事响应做评分）；
  - 不比较不同架构的 inner-validation 性能（本脚本一次只诊断一个模型/H）。

用途：在看到任何正式 held-out 结果之前，确认 λ 网格 logspace(-2,4,13) 是否够宽
（边界命中率高则需依 inner validation 扩网格 → 更新 spec/hash/freeze tag），
以及 PCA/solver 数值是否健康。若边界命中率低、方差保留合理、无 NaN/零方差，
则可冻结进入 M3b。

安全：含 himalaya inner-CV 拟合，仅 AutoDL 服务器运行；比全量 M3 轻（不评分、
不 predict）。默认 float32。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config                 # noqa: E402
from src.ridge.assemble import assemble_all               # noqa: E402
from src.ridge.pipeline import (                          # noqa: E402
    _transform_and_fir, LAMBDA_GRID, PCA_K, INNER_FOLDS, DELAYS_S, TR_SECONDS,
)
from sklearn.preprocessing import StandardScaler          # noqa: E402
from sklearn.decomposition import PCA                     # noqa: E402


def fit_inner_lambdas(Xtr_f, Ytr, lambda_grid, inner_folds):
    """himalaya inner-CV 拟合训练折，返回 per-voxel 选中 λ（不 predict 任何测试数据）。"""
    from himalaya.ridge import RidgeCV
    from himalaya.backend import set_backend
    from himalaya.scoring import correlation_score
    from sklearn.model_selection import KFold

    backend = set_backend("torch_cuda", on_error="warn")
    model = RidgeCV(
        alphas=lambda_grid, cv=KFold(n_splits=inner_folds), fit_intercept=False,
        Y_in_cpu=True,
        solver_params=dict(
            score_func=correlation_score, local_alpha=True,
            n_targets_batch=5000, n_targets_batch_refit=2000, n_alphas_batch=5,
        ),
    )
    model.fit(Xtr_f, Ytr)      # inner CV 全程只在训练折内，不触及 held-out
    lambdas = np.asarray(backend.to_numpy(model.best_alphas_))
    cv_scores = np.asarray(backend.to_numpy(model.cv_scores_))  # 每体素最佳λ的inner-CV分
    return lambdas, cv_scores


def diagnose_fold(fold_name, train_stories, story_data, lambda_grid,
                  pca_k, inner_folds, seed, dtype, signal_threshold):
    """对单个 fold 的训练故事做盲态数值诊断，返回诊断 dict（无 held-out r）。"""
    t0 = time.time()
    # scaler + PCA 仅在训练故事上 fit
    Xtr_raw = np.vstack([story_data[s].X for s in train_stories]).astype(dtype)
    n_nan_X = int(np.isnan(Xtr_raw).sum())
    scaler = StandardScaler().fit(Xtr_raw)
    pca = PCA(n_components=pca_k, svd_solver="full", random_state=seed)
    pca.fit(scaler.transform(Xtr_raw))
    evr = float(pca.explained_variance_ratio_.sum())
    del Xtr_raw

    Xtr_f, Ytr, vtr, _, _ = _transform_and_fir(
        story_data, train_stories, scaler, pca, DELAYS_S, TR_SECONDS)
    Xtr_f, Ytr = Xtr_f[vtr].astype(dtype), Ytr[vtr].astype(dtype)

    # 数值健康检查
    n_nan_Xf = int(np.isnan(Xtr_f).sum())
    y_var = Ytr.var(0)
    n_zerovar_Y = int((y_var == 0).sum())
    n_nan_Y = int(np.isnan(Ytr).sum())

    print(f"[m3a] {fold_name} 训练={len(train_stories)}故事 "
          f"Xtr={Xtr_f.shape} PCA方差={evr:.4f}，inner-CV 拟合中...", flush=True)
    lambdas, cv_scores = fit_inner_lambdas(Xtr_f, Ytr, lambda_grid, inner_folds)

    lam_min, lam_max = float(lambda_grid.min()), float(lambda_grid.max())
    at_min = lambdas <= lam_min * (1 + 1e-6)
    at_max = lambdas >= lam_max * (1 - 1e-6)
    hit_min = float(at_min.mean())
    hit_max = float(at_max.mean())
    # 每档直方图
    hist = {f"{g:.4g}": int((np.abs(lambdas - g) < g * 1e-6).sum())
            for g in lambda_grid}

    # 关键：按 inner-CV score 分层看边界命中率。全脑边界命中率会被大量无信号噪声
    # 体素（正确行为=选最大λ把预测压到0）拉高，不能作为网格够不够的判据；真正
    # 重要的是**有信号体素**（inner-CV score 高）的 λ 是否落在网格内部。
    layers = {}
    for thr in sorted({0.0, 0.05, 0.10, 0.20, round(signal_threshold, 2)}):
        sig = cv_scores > thr
        n = int(sig.sum())
        layers[f"cv>{thr:.2f}"] = {
            "n_voxels": n,
            "frac_of_all": float(n / len(cv_scores)),
            "hit_min_frac": float(at_min[sig].mean()) if n else 0.0,
            "hit_max_frac": float(at_max[sig].mean()) if n else 0.0,
            "boundary_hit_frac": float((at_min[sig] | at_max[sig]).mean()) if n else 0.0,
        }

    diag = {
        "fold": fold_name,
        "n_train_stories": len(train_stories),
        "Xtr_shape": list(Xtr_f.shape),
        "pca_explained_variance_ratio": evr,
        "n_nan_X_preFIR": n_nan_X,
        "n_nan_X_postFIR": n_nan_Xf,
        "n_nan_Y": n_nan_Y,
        "n_zerovar_Y_cols": n_zerovar_Y,
        "lambda_grid_min": lam_min,
        "lambda_grid_max": lam_max,
        "lambda_hit_min_frac": hit_min,
        "lambda_hit_max_frac": hit_max,
        "lambda_boundary_hit_frac": hit_min + hit_max,  # 全脑（含噪声体素）
        "lambda_hist": hist,
        "cv_score_max": float(cv_scores.max()),
        "cv_score_p99": float(np.percentile(cv_scores, 99)),
        "boundary_hit_by_signal_layer": layers,  # 按信号强度分层（真正的判据）
        "seconds": round(time.time() - t0, 1),
    }
    sig_layer = layers[f"cv>{signal_threshold:.2f}"]
    print(f"[m3a] {fold_name} 全脑λ边界命中={hit_min+hit_max:.3f} "
          f"| 信号体素(cv>{signal_threshold:.2f}, n={sig_layer['n_voxels']}) 边界命中="
          f"{sig_layer['boundary_hit_frac']:.3f} "
          f"(上界{sig_layer['hit_max_frac']:.3f}/下界{sig_layer['hit_min_frac']:.3f})",
          flush=True)
    print(f"[m3a] {fold_name} NaN(X/Y)={n_nan_Xf}/{n_nan_Y} 零方差Y={n_zerovar_Y} "
          f"cv_max={cv_scores.max():.3f}  {diag['seconds']}s", flush=True)
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia")
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--layer", default="main", choices=["main", "final"])
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--folds", nargs="+", default=None,
                    help="只诊断指定 fold（如 fold_0）；默认全部")
    ap.add_argument("--boundary-warn", type=float, default=0.05,
                    help="信号体素 λ 边界命中率超过此比例则提示需扩网格")
    ap.add_argument("--signal-cv-threshold", type=float, default=0.10,
                    help="判定'有信号体素'的 inner-CV score 阈值（默认 0.10）；"
                         "只有这些体素的边界命中率才作为网格是否够宽的判据")
    ap.add_argument("--lambda-log-min", type=float, default=-2,
                    help="探索用 λ 网格下界指数（默认 -2，与冻结 spec 一致）")
    ap.add_argument("--lambda-log-max", type=float, default=4,
                    help="探索用 λ 网格上界指数（默认 4，与冻结 spec 一致；"
                         "命中率高时可临时调大，如 7，仅用于诊断探索，"
                         "不会改动 pipeline.py 里真正冻结的 LAMBDA_GRID）")
    ap.add_argument("--lambda-n", type=int, default=13,
                    help="探索用 λ 网格点数（默认 13，与冻结 spec 一致）")
    args = ap.parse_args()

    cfg = load_config()
    paths, ds = cfg["paths"], cfg["datasets"]
    seed = cfg["seeds"]["pca"]
    dt = np.dtype(args.dtype)
    lambda_grid = np.logspace(args.lambda_log_min, args.lambda_log_max, args.lambda_n)
    is_frozen_grid = (lambda_grid.shape == LAMBDA_GRID.shape
                      and np.allclose(lambda_grid, LAMBDA_GRID))
    if not is_frozen_grid:
        print(f"[m3a] ⚠️ 使用探索性网格 logspace({args.lambda_log_min},"
              f"{args.lambda_log_max},{args.lambda_n}) = "
              f"[{lambda_grid.min():.4g},{lambda_grid.max():.4g}]，"
              f"非冻结 spec 网格，仅供 M3a 诊断探索，不影响 pipeline.py", flush=True)

    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        fold_split = json.load(f)
    fold_items = fold_split["folds"]
    fold_names = args.folds or sorted(fold_items)

    # 只需要训练故事的并集（绝不加载/触及各 fold 的 test 故事做评分）
    train_by_fold = {fn: list(fold_items[fn]["train_stories"]) for fn in fold_names}
    train_union = sorted({s for ts in train_by_fold.values() for s in ts})
    print(f"[m3a] 盲态核查 model={args.model} H={args.H} layer={args.layer} "
          f"subject={args.subject} dtype={args.dtype}", flush=True)
    print(f"[m3a] 诊断 {len(fold_names)} 折，训练故事并集 {len(train_union)} 个"
          f"（不碰任何 held-out 评分）", flush=True)

    story_data = assemble_all(
        train_union, args.model, args.H, args.layer, args.subject,
        paths["cache_dir"], ds["data_dir"], ds["respdict"],
        Path(paths["frozen_dir"]) / "word_index.parquet",
    )

    diags = [diagnose_fold(fn, train_by_fold[fn], story_data, lambda_grid,
                           PCA_K, INNER_FOLDS, seed, dt, args.signal_cv_threshold)
             for fn in fold_names]

    # 全脑边界命中率（会被无信号噪声体素拉高，仅供参考）
    max_boundary_all = max(d["lambda_boundary_hit_frac"] for d in diags)
    # 真正的判据：有信号体素（cv>0.1）的边界命中率
    sig_key = f"cv>{args.signal_cv_threshold:.2f}"
    max_boundary_sig = max(
        d["boundary_hit_by_signal_layer"][sig_key]["boundary_hit_frac"] for d in diags)
    min_sig_voxels = min(
        d["boundary_hit_by_signal_layer"][sig_key]["n_voxels"] for d in diags)
    verdict = {
        "criterion": f"signal-voxel ({sig_key}) boundary hit rate <= {args.boundary_warn}",
        "max_boundary_hit_signal_voxels": max_boundary_sig,
        "max_boundary_hit_all_voxels": max_boundary_all,
        "boundary_warn_threshold": args.boundary_warn,
        "min_signal_voxels_per_fold": min_sig_voxels,
        "grid_ok_for_signal_voxels": bool(max_boundary_sig <= args.boundary_warn),
        "any_nan": any(d["n_nan_X_postFIR"] or d["n_nan_Y"] for d in diags),
        "any_zerovar_Y": any(d["n_zerovar_Y_cols"] for d in diags),
    }

    out_dir = Path(paths["results_dir"]) / "m3a_blind_check" / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model, "H": args.H, "layer": args.layer,
        "subject": args.subject, "dtype": args.dtype, "seed": seed,
        "lambda_grid": f"logspace({args.lambda_log_min},{args.lambda_log_max},{args.lambda_n})",
        "lambda_grid_is_frozen_spec": bool(is_frozen_grid),
        "pca_k": PCA_K,
        "inner_folds": INNER_FOLDS,
        "folds": diags, "verdict": verdict,
        "note": "M3a blind check: no held-out r computed or saved.",
    }
    grid_tag = "" if report["lambda_grid_is_frozen_spec"] else \
        f"_grid{args.lambda_log_min}to{args.lambda_log_max}"
    out_path = out_dir / f"m3a_{args.model}_H{args.H}_{args.layer}{grid_tag}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    ok = verdict["grid_ok_for_signal_voxels"]
    print("\n[m3a] === 盲态核查判定 ===", flush=True)
    print(f"[m3a] 全脑 λ 边界命中率 = {max_boundary_all:.4f}（含大量无信号噪声体素，仅参考）",
          flush=True)
    print(f"[m3a] 信号体素({sig_key}, 每折≥{min_sig_voxels}个) 边界命中率 = "
          f"{max_boundary_sig:.4f} "
          f"({'OK ≤' if ok else '⚠️ 超'}{args.boundary_warn}"
          f"，{'网格对信号体素够宽，可冻结进 M3b' if ok else '信号体素也大量撞界，需扩网格并重新冻结'})",
          flush=True)
    print(f"[m3a] NaN: {'有⚠️' if verdict['any_nan'] else '无'}  "
          f"零方差Y: {'有⚠️' if verdict['any_zerovar_Y'] else '无'}", flush=True)
    print(f"[m3a] 报告 → {out_path}", flush=True)


if __name__ == "__main__":
    main()
