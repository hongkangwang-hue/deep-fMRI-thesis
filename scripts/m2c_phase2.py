"""
M2-C Phase 2 —— 用冻结 spec 跑 eng1000，与 Phase 1 native corrs 描述性对比。

目的：验证主实验（M3/M4）真正使用的那套管线（PCA-100 + FIR + himalaya
per-voxel RidgeCV）在真实 fMRI + eng1000（已知强信号特征）上合理、不漏、
且空间模式与已验证忠实的 native 实现高相关。这是 M3 上模型特征前的"管线体检"。

设计（控制变量，最大化与 native 可比）：
  训练 = 83 个 CV 故事；测试 = wheretheressmoke（与 Phase 1 native **同一测试故事**）。
  差异只剩 PCA + solver + per-voxel alpha 选择，故空间相关对比干净。
  outer 3 折 CV 留给 M3/M4（那里无单一参照故事）。

对比（frozen/m2c_reference_validation.yaml::phase2_frozen_spec）：
  - spatial_pattern_pearson_vs_native  ≥ 0.80（描述性，低于则排查非硬闸门）
  - roi_mean_r_sign                    IFG/PT fisher-z 平均 r 应为正
  - leakage_checks                     硬性：PCA/scaler 仅训练折 fit、测试故事不入训练、FIR 故事内

安全：含大型 ridge 计算，仅 AutoDL 服务器运行，需用户确认后手动启动。
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
from src.ridge.pipeline import (                          # noqa: E402
    StoryData, run_fold, numpy_ridgecv_solver, himalaya_ridgecv_solver,
)
from src.ridge.score import roi_mean_r                    # noqa: E402
from src.fmri.derivatives import load_response            # noqa: E402
from src.fmri.trfile import (                              # noqa: E402
    load_respdict, trimmed_tr_times, TRIM_FIRST, TRIM_LAST,
)


def spatial_pattern_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """两个体素 r 向量 (95556,) 的空间相关（跨体素 Pearson）。"""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def assemble_eng1000(stories, subject, data_dir, respdict_path):
    """组装 eng1000 特征 → {story: StoryData}（Phase 2 自包含，不改 assemble.py）。

    eng1000 对**所有词**都有 985 维查找向量（非目标词子集），下采样与 native 完全相同：
    import 调用原作者 encoding/feature_spaces.get_feature_space('eng1000', ...)（仅调用，
    不修改），逐故事 Lanczos 到 TR；再 trim [10:-5] 对齐已 trim 的响应。
    **不在此 z-score**——pipeline 的 StandardScaler 负责（fit 仅训练折，防泄漏）。
    """
    sys.path.insert(0, str(PROJECT_ROOT / "encoding"))
    from feature_spaces import get_feature_space           # noqa: E402

    respdict = load_respdict(respdict_path)
    feat = get_feature_space("eng1000", stories)           # {story: (resps-pad, 985)}
    out = {}
    for s in stories:
        X_full = np.asarray(feat[s], dtype=np.float64)
        X = X_full[TRIM_FIRST: len(X_full) - TRIM_LAST]    # trim [10:-5]
        Y = load_response(data_dir, subject, s).astype(np.float64)
        trt = trimmed_tr_times(respdict[s])                # 真实 TR 中心时间（>100s 判定）
        if not (X.shape[0] == Y.shape[0] == len(trt)):
            raise ValueError(
                f"[{s}] eng1000 行数不一致 X={X.shape[0]} Y={Y.shape[0]} "
                f"tr_times={len(trt)}")
        out[s] = StoryData(X=X, Y=Y, tr_times=trt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--solver", default="himalaya", choices=["himalaya", "numpy"])
    ap.add_argument("--native", default=None,
                    help="Phase 1 native corrs（与之做空间对比）；缺省时按 "
                         "--subject 取 results/eng1000_gpu_rerun/<subject>/corrs.npz，"
                         "不固定指向某一个被试")
    ap.add_argument("--seed", type=int, default=None, help="默认 config.seeds.pca")
    ap.add_argument("--out-name", default="eng1000_phase2_frozen")
    args = ap.parse_args()

    if args.native is None:
        # 原来固定指向 UTS03，换被试忘了传 --native 会静默拿 UTS03 的 Phase1
        # 结果去和新被试的 Phase2 结果比"空间相关"，数字看似正常实则无意义。
        args.native = f"results/eng1000_gpu_rerun/{args.subject}/corrs.npz"

    cfg = load_config()
    paths, ds = cfg["paths"], cfg["datasets"]
    seed = args.seed if args.seed is not None else cfg["seeds"]["pca"]

    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        fold_split = json.load(f)
    held = fold_split["held_out_test_story"]
    cv_stories = sorted({s for fo in fold_split["folds"].values()
                         for s in (fo["train_stories"] + fo["test_stories"])})
    assert held not in cv_stories, "held-out 测试故事不应在 CV 故事里"

    print(f"[phase2] subject={args.subject} solver={args.solver} seed={seed}",
          flush=True)
    print(f"[phase2] 训练={len(cv_stories)} 故事  测试=[{held}]", flush=True)

    # 组装 eng1000（训练 83 + 测试 1）
    print("[phase2] 组装 eng1000 特征+响应 ...", flush=True)
    t0 = time.time()
    story_data = assemble_eng1000(
        cv_stories + [held], args.subject, ds["data_dir"], ds["respdict"],
    )
    print(f"[phase2] 组装完成 {time.time()-t0:.1f}s", flush=True)

    # 泄漏硬检查：测试故事绝不在训练集
    assert held not in cv_stories
    solver = himalaya_ridgecv_solver if args.solver == "himalaya" else numpy_ridgecv_solver

    print("[phase2] 冻结 spec 编码（PCA-100 + FIR + himalaya）...", flush=True)
    t0 = time.time()
    fr = run_fold(story_data, cv_stories, [held], solver, seed=seed)
    print(f"[phase2] 完成 {(time.time()-t0)/60:.1f} 分钟", flush=True)

    ours = fr.voxel_r
    native = np.load(PROJECT_ROOT / args.native)["arr_0"]
    sp = spatial_pattern_pearson(ours, native)

    roi_cols = dict(np.load(
        Path(paths["frozen_dir"]) / f"roi_columns_{args.subject}.npz"))
    roi_summary = {name: roi_mean_r(ours, cols) for name, cols in roi_cols.items()}

    # 判定（描述性 + 硬性泄漏）
    sp_pass = sp >= 0.80
    roi_pass = all(r > 0 for r in roi_summary.values())
    leakage_pass = held not in cv_stories  # 结构性，pipeline 另由单测保证

    out_dir = Path(paths["results_dir"]) / args.out_name / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "voxel_r", ours)
    np.savez(out_dir / "valphas", fr.valphas)
    manifest = {
        "phase": "M2-C Phase 2 (frozen spec)",
        "subject": args.subject, "seed": seed, "solver": args.solver,
        "train_n_stories": len(cv_stories), "test_story": held,
        "n_eff_tr": fr.n_eff_tr,
        "voxel_r_mean": float(np.nanmean(ours)),
        "voxel_r_max": float(np.nanmax(ours)),
        "spatial_pattern_pearson_vs_native": sp,
        "roi_mean_r": roi_summary,
        "verdict": {
            "spatial_pattern_pearson>=0.80": sp_pass,
            "roi_mean_r_sign_positive": roi_pass,
            "leakage_no_test_in_train": leakage_pass,
        },
        "native_compared": args.native,
        "spec": "frozen/analysis_spec.yaml",
        "alignment": "eng1000_all_words_native_downsample",
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[phase2] voxel_r mean={manifest['voxel_r_mean']:.4f} "
          f"max={manifest['voxel_r_max']:.4f}", flush=True)
    print(f"[phase2] 空间相关 vs native = {sp:.4f} "
          f"({'PASS' if sp_pass else 'CHECK'} ≥0.80)", flush=True)
    for name, r in roi_summary.items():
        print(f"[phase2]   ROI {name}: mean_r={r:.4f} "
              f"({'+' if r > 0 else '-'})", flush=True)
    print(f"[phase2] 泄漏检查(测试故事不入训练): "
          f"{'PASS' if leakage_pass else 'FAIL'}", flush=True)
    print(f"[phase2] 已保存 → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
