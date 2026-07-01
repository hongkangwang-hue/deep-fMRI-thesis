"""
M3 垂直切片 —— 1 模型 × 1 上下文长度 H 的端到端编码（冻结 spec）。

把 M1 特征缓存按 frozen/analysis_spec.yaml 跑完整管线：
  assemble（方案 A 对齐）→ PCA-100 → FIR → himalaya 3 折 CV → 体素 r → ROI fisher-z 均值

证明"特征→对齐→PCA→ridge→打分"整条链在单个模型/H 上端到端跑通，是 M4 全矩阵前的竖切。
同一套 src/ridge/ 核心也供 M2-C Phase 2（输入换成 eng1000）复用。

安全：本脚本含大型 ridge 计算，仅在 AutoDL 服务器运行，需用户确认后手动启动。
默认 solver 为 himalaya GPU；himalaya 不可用时回退 numpy（仅供 smoke）。
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
    run_encoding_cv, numpy_ridgecv_solver, himalaya_ridgecv_solver,
)


def build_folds(fold_split: dict):
    """从 frozen/fold_split.json 构建 [(train_stories, test_stories), ...]。"""
    folds = []
    for name in sorted(fold_split["folds"]):
        f = fold_split["folds"][name]
        folds.append((list(f["train_stories"]), list(f["test_stories"])))
    return folds


def pick_solver(name: str):
    if name == "himalaya":
        return himalaya_ridgecv_solver
    if name == "numpy":
        return numpy_ridgecv_solver
    raise ValueError(f"未知 solver: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia")
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--layer", default="main", choices=["main", "final"])
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--solver", default="himalaya", choices=["himalaya", "numpy"])
    ap.add_argument("--seed", type=int, default=None,
                    help="默认取 config.seeds.pca")
    ap.add_argument("--out-name", default=None,
                    help="默认 m3_<model>_H<H>_<layer>")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                    help="特征/响应精度。M3/M4 描述性实验默认 float32（GPU 快 2-8×、"
                         "内存减半）；Phase1 已用 float64 验证过忠实度，主实验只比模型间 Δr，"
                         "float32 精度差对所有模型一致、不影响相对比较。")
    args = ap.parse_args()

    cfg = load_config()
    paths, ds = cfg["paths"], cfg["datasets"]
    seed = args.seed if args.seed is not None else cfg["seeds"]["pca"]
    out_name = args.out_name or f"m3_{args.model}_H{args.H}_{args.layer}"

    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        fold_split = json.load(f)
    folds = build_folds(fold_split)

    # CV 用到的全部故事（各折 train ∪ test 的并集）
    stories = sorted({s for tr, te in folds for s in (tr + te)})
    print(f"[m3] model={args.model} H={args.H} layer={args.layer} "
          f"seed={seed} solver={args.solver}", flush=True)
    print(f"[m3] {len(folds)} 折，{len(stories)} 个故事", flush=True)

    roi_cols = dict(np.load(
        Path(paths["frozen_dir"]) / f"roi_columns_{args.subject}.npz"))

    print("[m3] 组装特征+响应（方案 A 对齐）...", flush=True)
    t0 = time.time()
    story_data = assemble_all(
        stories, args.model, args.H, args.layer, args.subject,
        paths["cache_dir"], ds["data_dir"], ds["respdict"],
        Path(paths["frozen_dir"]) / "word_index.parquet",
    )
    # 特征/响应转指定精度（float32 加速 GPU ridge + 内存减半）；tr_times 保持 float64
    dt = np.dtype(args.dtype)
    for s in story_data:
        story_data[s].X = story_data[s].X.astype(dt)
        story_data[s].Y = story_data[s].Y.astype(dt)
    print(f"[m3] 组装完成 {time.time()-t0:.1f}s，精度={args.dtype}", flush=True)

    print("[m3] 开始 3 折编码 CV ...", flush=True)
    t0 = time.time()
    result = run_encoding_cv(story_data, folds, pick_solver(args.solver),
                             roi_columns=roi_cols, seed=seed)
    print(f"[m3] CV 完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)

    # ROI r 由 pipeline 按 story 级 fisher-z 聚合得出（result.roi_r），不再对
    # 跨折 voxel_r 重算（那是旧的、与冻结文档 M3 模块4 不符的聚合顺序）
    roi_summary = result.roi_r

    out_dir = Path(paths["results_dir"]) / out_name / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "voxel_r", result.voxel_r)
    np.savez(out_dir / "fold_valphas",
             **{f"fold{i}": fr.valphas for i, fr in enumerate(result.folds)})
    # per-story ROI r（M4 每 story 独立保存、M5 story-level bootstrap 单位）
    per_story = [
        {"fold": i, "story": ss.story, "n_eff_tr": ss.n_eff_tr,
         "roi_r": {n: float(np.tanh(z)) for n, z in ss.roi_z.items()}}
        for i, fr in enumerate(result.folds) for ss in fr.story_scores
    ]
    with open(out_dir / "per_story_roi.json", "w") as f:
        json.dump(per_story, f, indent=2, ensure_ascii=False)
    manifest = {
        "model": args.model, "H": args.H, "layer": args.layer,
        "subject": args.subject, "seed": seed, "solver": args.solver,
        "n_folds": len(folds), "n_stories": len(stories),
        "fold_n_eff_tr": [fr.n_eff_tr for fr in result.folds],
        "voxel_r_mean": float(np.nanmean(result.voxel_r)),
        "voxel_r_max": float(np.nanmax(result.voxel_r)),
        "roi_mean_r": roi_summary,
        "roi_aggregation": "per_story_fisherz_then_effective_tr_weighted (M3 module4)",
        "dtype": args.dtype,
        "spec": "frozen/analysis_spec.yaml",
        "alignment": "plan_A_target_words_only",
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[m3] voxel_r mean={manifest['voxel_r_mean']:.4f} "
          f"max={manifest['voxel_r_max']:.4f}", flush=True)
    for name, r in roi_summary.items():
        print(f"[m3]   ROI {name}: mean_r={r:.4f}", flush=True)
    print(f"[m3] 已保存 → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
