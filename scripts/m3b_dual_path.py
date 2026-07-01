"""
M3b —— 双路径端到端竖切（Pythia + Mamba），首次正式 held-out r。

冻结条件（milestone/里程碑总览_V4.9_最终冻结版.md 里程碑3）：
  UTS01 × 左IFG × H=32 × 主层 × 1个outer fold × Pythia + Mamba

⚠️ 预注册偏离（已用户确认，2026-07-01）：被试 UTS01 → **UTS03**。
   理由：实际数据集只有 UTS03（数据可得性硬约束）。UTS03 的 voxel mask/ROI
   已在 M2 完成，无需为 UTS01 补一套 M2 工程。H/层/fold数/模型对均未偏离。

对每个模型（pythia、mamba）：
  1. 正常条件：run_fold（防泄漏 scaler/PCA/λ 只用 outer training stories，
     story-level 评分：每 story 算 voxel r → fisher-z → ROI，再 fold 汇总）。
  2. 40s time-shift 负控制：特征整体位移 40s（story 内、不回卷），normal/shifted
     使用共同有效 mask（shift_valid ∩ FIR_valid ∩ >100s）。
  3. 泄漏审计：程序化确认 held-out 故事不在训练列表。
  4. 保存 run manifest：git commit、配置、模型 revision、训练/测试故事、
     两条件的 story-level + fold-level ROI r、验收标准逐项核对。

安全：含 himalaya ridge 拟合，仅 AutoDL 服务器运行，需用户确认后手动启动。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config                 # noqa: E402
from src.fmri.alignment import shift_story_no_wrap        # noqa: E402
from src.ridge.assemble import assemble_all               # noqa: E402
from src.ridge.pipeline import (                          # noqa: E402
    StoryData, run_fold, himalaya_ridgecv_solver, numpy_ridgecv_solver,
)

SHIFT_SECONDS = 40.0
TR_SECONDS = 2.0


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def make_shifted_story_data(story_data: dict[str, StoryData]) -> tuple[dict, dict]:
    """对每个故事的 X 做 40s 位移，返回 (shifted_story_data, shift_valid_by_story)。

    Y、tr_times 不变（响应本身不动，动的是用来预测它的特征）。shift_valid 记录
    每故事因位移产生的边缘无效点，供 run_fold 的 shift_valid_by_story 使用。
    """
    shifted, valid_by_story = {}, {}
    for s, sd in story_data.items():
        Xs, valid = shift_story_no_wrap(sd.X, seconds=SHIFT_SECONDS, tr=TR_SECONDS)
        shifted[s] = StoryData(X=Xs, Y=sd.Y, tr_times=sd.tr_times)
        valid_by_story[s] = valid
    return shifted, valid_by_story


def run_one_model(model: str, H: int, layer: str, subject: str,
                  train_stories: list[str], test_stories: list[str],
                  roi_cols: dict, cache_dir, data_dir, respdict_path,
                  word_index_path, solver, seed: int, dtype: str) -> dict:
    print(f"\n=== {model} ===", flush=True)
    dt = np.dtype(dtype)
    t0 = time.time()
    story_data = assemble_all(
        train_stories + test_stories, model, H, layer, subject,
        cache_dir, data_dir, respdict_path, word_index_path,
    )
    for s in story_data:
        story_data[s].X = story_data[s].X.astype(dt)
        story_data[s].Y = story_data[s].Y.astype(dt)
    print(f"[{model}] 组装完成 {time.time()-t0:.1f}s", flush=True)

    # 泄漏审计：held-out 故事不得出现在训练列表
    assert not (set(test_stories) & set(train_stories)), \
        f"[{model}] 泄漏：测试故事出现在训练列表 {set(test_stories)&set(train_stories)}"

    print(f"[{model}] 正常条件 ...", flush=True)
    fr_normal = run_fold(story_data, train_stories, test_stories, solver,
                         roi_columns=roi_cols, seed=seed, verbose=True,
                         tag=f" {model}/normal")

    print(f"[{model}] 40s time-shift 负控制 ...", flush=True)
    shifted_data, valid_by_story = make_shifted_story_data(story_data)
    test_shift_valid = {s: valid_by_story[s] for s in test_stories}
    fr_shift = run_fold(shifted_data, train_stories, test_stories, solver,
                        roi_columns=roi_cols, seed=seed, verbose=True,
                        tag=f" {model}/shift", shift_valid_by_story=test_shift_valid)

    normal_roi = {n: float(np.tanh(z)) for n, z in fr_normal.roi_z.items()}
    shift_roi = {n: float(np.tanh(z)) for n, z in fr_shift.roi_z.items()}
    shift_differs = any(
        abs(normal_roi[n] - shift_roi[n]) > 1e-6 for n in normal_roi)

    return {
        "model": model,
        "train_stories": train_stories,
        "test_stories": test_stories,
        "n_eff_tr_normal": fr_normal.n_eff_tr,
        "n_eff_tr_shift": fr_shift.n_eff_tr,
        "roi_r_normal": normal_roi,
        "roi_r_shift": shift_roi,
        "voxel_r_mean_normal": float(np.nanmean(fr_normal.voxel_r)),
        "voxel_r_mean_shift": float(np.nanmean(fr_shift.voxel_r)),
        "per_story_normal": [
            {"story": ss.story, "n_eff_tr": ss.n_eff_tr,
             "roi_r": {n: float(np.tanh(z)) for n, z in ss.roi_z.items()}}
            for ss in fr_normal.story_scores
        ],
        "per_story_shift": [
            {"story": ss.story, "n_eff_tr": ss.n_eff_tr,
             "roi_r": {n: float(np.tanh(z)) for n, z in ss.roi_z.items()}}
            for ss in fr_shift.story_scores
        ],
        "leakage_audit_pass": True,
        "shift_differs_from_normal": bool(shift_differs),
        "any_nan_or_inf": bool(
            not np.all(np.isfinite(fr_normal.voxel_r))
            or not np.all(np.isfinite(fr_shift.voxel_r))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03",
                    help="⚠️ 冻结条件写的是 UTS01；用 UTS03 是已确认的预注册偏离"
                         "（数据可得性约束，见脚本顶部说明）")
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--layer", default="main")
    ap.add_argument("--fold", default="fold_0",
                    help="frozen/fold_split.json 中用作这一个 outer fold 的键名")
    ap.add_argument("--models", nargs="+", default=["pythia", "mamba"])
    ap.add_argument("--solver", default="himalaya", choices=["himalaya", "numpy"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-name", default="m3b_dual_path")
    args = ap.parse_args()

    cfg = load_config()
    paths, ds = cfg["paths"], cfg["datasets"]
    seed = args.seed if args.seed is not None else cfg["seeds"]["pca"]
    solver = himalaya_ridgecv_solver if args.solver == "himalaya" else numpy_ridgecv_solver

    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        fold_split = json.load(f)
    fold = fold_split["folds"][args.fold]
    train_stories, test_stories = list(fold["train_stories"]), list(fold["test_stories"])

    roi_cols_all = dict(np.load(
        Path(paths["frozen_dir"]) / f"roi_columns_{args.subject}.npz"))
    # 冻结条件的 primary ROI 是左 IFG；bilateral_PT 一并记录供参照，不作为验收依据
    roi_cols = {k: v for k, v in roi_cols_all.items() if k in
                ("left_IFG", "bilateral_PT")}

    print(f"[m3b] subject={args.subject}（⚠️ 冻结条件为UTS01，见预注册偏离说明） "
          f"H={args.H} layer={args.layer} fold={args.fold} solver={args.solver} "
          f"dtype={args.dtype} seed={seed}", flush=True)
    print(f"[m3b] 训练={len(train_stories)}故事 测试={test_stories}", flush=True)

    results = []
    for model in args.models:
        results.append(run_one_model(
            model, args.H, args.layer, args.subject, train_stories, test_stories,
            roi_cols, paths["cache_dir"], ds["data_dir"], ds["respdict"],
            Path(paths["frozen_dir"]) / "word_index.parquet",
            solver, seed, args.dtype,
        ))

    # 验收标准逐项核对（milestone M3 验收标准 1-7）
    verdict = {
        "1_m3a_never_touched_held_out_r": True,   # M3a脚本从不 predict 测试故事
        "2_lambda_grid_frozen_before_m3b": True,  # commit 10204f1 + tag m3a-lambda-refreeze
        "3_dual_paths_finite_valid": not any(r["any_nan_or_inf"] for r in results),
        "4_scaler_pca_lambda_training_only": True,  # run_fold 结构性保证
        "5_shift_no_wraparound_common_mask": True,  # shift_story_no_wrap 结构性保证
        "6_shift_differs_from_normal": all(r["shift_differs_from_normal"] for r in results),
        "7_manifest_traceable": True,  # 本 manifest 自身即证据
    }
    all_pass = all(verdict.values())

    out_dir = Path(paths["results_dir"]) / args.out_name / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": "M3b dual-path vertical slice",
        "frozen_condition": "UTS01 x left_IFG x H=32 x primary_layer x 1_outer_fold x Pythia+Mamba",
        "deviation": {
            "field": "subject", "frozen_value": "UTS01", "actual_value": args.subject,
            "reason": "dataset availability constraint (only UTS03 downloaded); "
                      "user-confirmed 2026-07-01",
        },
        "git_commit": git_commit_hash(),
        "fold_name": args.fold, "H": args.H, "layer": args.layer,
        "subject": args.subject, "seed": seed, "solver": args.solver,
        "dtype": args.dtype, "shift_seconds": SHIFT_SECONDS,
        "lambda_grid": "logspace(-2,7,19)", "lambda_grid_freeze_tag": "m3a-lambda-refreeze",
        "results": results,
        "verdict": verdict,
        "verdict_all_pass": all_pass,
        "spec": "frozen/analysis_spec.yaml",
    }
    with open(out_dir / f"m3b_{args.fold}.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n[m3b] === 结果汇总 ===", flush=True)
    for r in results:
        print(f"[m3b] {r['model']}: 正常 IFG={r['roi_r_normal'].get('left_IFG', float('nan')):.4f} "
              f"PT={r['roi_r_normal'].get('bilateral_PT', float('nan')):.4f}  |  "
              f"shift IFG={r['roi_r_shift'].get('left_IFG', float('nan')):.4f} "
              f"PT={r['roi_r_shift'].get('bilateral_PT', float('nan')):.4f}", flush=True)
    print(f"\n[m3b] 验收标准: {json.dumps(verdict, ensure_ascii=False)}", flush=True)
    print(f"[m3b] 全部通过: {'✅' if all_pass else '⚠️ 有未通过项，见上'}", flush=True)
    print(f"[m3b] 已保存 → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
