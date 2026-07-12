"""
M3（三被试扩展）—— 新被试端到端竖切复核。

冻结条件（milestone/里程碑总览_三被试执行计划_V6_3_阶段二_M1至M4逐被试执行.md 里程碑3）：
  某一新被试（建议 UTS01）× 主层 × 1 个 outer fold × 四模型 × H={8,32,128}
  × normal + shift 两条件。不铺全部三折。

⚠️ 与 UTS03 pilot 的 M3b 不同：那次是"UTS01→UTS03"的预注册偏离（当时只有 UTS03
   数据）。本脚本是三被试扩展，UTS01/UTS02 数据已下载，**用 UTS01 就是冻结条件本身，
   不是偏离**，manifest 里不写 deviation 字段。

本阶段是"迁移正确性"闸门，不是架构比较。只回答：新被试数据是否进入正确矩阵、λ 网格
是否仍覆盖合理范围、PCA-100 是否可拟合且 evr@100 可记录、normal/shift 是否用完全相同
的最终评分 mask（逐元素 np.array_equal）、结果是否有限/非全零/两条件不完全相同。
**禁止**据此比较模型排名、修改任何冻结参数、或因某条件 r 低就删除它。

对每个 (model, H)：assemble 一次特征 → normal + shift 各独立 run_fold（各自 fit
training-only scaler/PCA/λ/Ridge，不共享拟合对象，也不只平移 prediction）→ 记录
两条件的 evr@100、λ 边界命中、held-out r、逐元素 mask 一致性。

安全：含 himalaya ridge 拟合，仅 AutoDL 服务器运行，需用户确认后手动启动。
单模型全跑 = 3H × (normal+shift) = 6 次 run_fold；四模型合计 24 次（单折，比 M4 轻）。
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
    StoryData, run_fold, himalaya_ridgecv_solver, numpy_ridgecv_solver, LAMBDA_GRID,
)
from src.models.feature_cache import load_features        # noqa: E402

SHIFT_SECONDS = 40.0
TR_SECONDS = 2.0
ALL_MODELS = ["pythia", "mamba", "rwkv", "awd_lstm"]
DEFAULT_H = [8, 32, 128]


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def make_shifted_story_data(story_data: dict[str, StoryData]) -> tuple[dict, dict]:
    """对每个故事的 X 做 40s 位移，返回 (shifted_story_data, shift_valid_by_story)。同 M4。"""
    shifted, valid_by_story = {}, {}
    for s, sd in story_data.items():
        Xs, valid = shift_story_no_wrap(sd.X, seconds=SHIFT_SECONDS, tr=TR_SECONDS)
        shifted[s] = StoryData(X=Xs, Y=sd.Y, tr_times=sd.tr_times)
        valid_by_story[s] = valid
    return shifted, valid_by_story


def _valphas_stats(valphas: np.ndarray) -> dict:
    """选中 λ 的统计（含网格边界命中率）。全脑含大量无信号噪声体素，正确行为就是
    选最大 λ 把预测压到 0，所以全脑 hit_max_frac 偏高（UTS03 实测约 16-19%）是正常的，
    不是 bug；真正的异常是信号体素也大量撞界（那要单独用 m3a 信号分层脚本查）。"""
    lam_min, lam_max = float(LAMBDA_GRID.min()), float(LAMBDA_GRID.max())
    return {
        "min": float(valphas.min()), "max": float(valphas.max()),
        "median": float(np.median(valphas)),
        "hit_min_frac": float((valphas <= lam_min * (1 + 1e-6)).mean()),
        "hit_max_frac": float((valphas >= lam_max * (1 - 1e-6)).mean()),
    }


def run_one_group(model: str, H: int, layer: str, subject: str,
                  train_stories: list[str], test_stories: list[str],
                  roi_cols: dict, cache_dir, data_dir, respdict_path,
                  word_index_path, solver, seed: int, dtype: str,
                  out_dir: Path) -> dict:
    """一个 (model, H) 竖切：normal + shift 各独立拟合，返回诊断字典。"""
    print(f"\n=== {model} H={H} ===", flush=True)
    dt = np.dtype(dtype)
    t0 = time.time()
    story_data = assemble_all(
        train_stories + test_stories, model, H, layer, subject,
        cache_dir, data_dir, respdict_path, word_index_path,
    )
    for s in story_data:
        story_data[s].X = story_data[s].X.astype(dt)
        story_data[s].Y = story_data[s].Y.astype(dt)
    print(f"[{model}/H{H}] 组装完成 {time.time()-t0:.1f}s", flush=True)

    feat_meta = load_features(cache_dir, model, test_stories[0], H)["meta"]

    # 泄漏审计：held-out 故事不得出现在训练列表
    leak = set(test_stories) & set(train_stories)
    assert not leak, f"[{model}/H{H}] 泄漏：测试故事出现在训练列表 {leak}"

    # 40s 位移及其有效点；normal 与 shift 都施加同一 shift_valid → 共同评分 mask
    shifted_data, valid_by_story = make_shifted_story_data(story_data)
    test_shift_valid = {s: valid_by_story[s] for s in test_stories}

    print(f"[{model}/H{H}] normal（共同 mask）...", flush=True)
    fr_normal = run_fold(story_data, train_stories, test_stories, solver,
                         roi_columns=roi_cols, seed=seed, verbose=True,
                         tag=f" {model}/H{H}/normal", shift_valid_by_story=test_shift_valid)

    print(f"[{model}/H{H}] shift 40s 负控制（同一 mask，独立重新拟合）...", flush=True)
    fr_shift = run_fold(shifted_data, train_stories, test_stories, solver,
                        roi_columns=roi_cols, seed=seed, verbose=True,
                        tag=f" {model}/H{H}/shift", shift_valid_by_story=test_shift_valid)

    # 逐元素 mask 一致性（M3 验收标准4，强验证：n_eff 相等只是必要条件，
    # 逐元素 np.array_equal 才是共同 mask 的充分证据）。不一致直接 raise。
    # 先堵住 zip 静默截断：两条件保留的 story 数必须一致，否则说明某个 story 的评分
    # mask 在一条件下全空被 run_fold 跳过、在另一条件下没被跳过 → 共同 mask 已不成立，
    # 若只 zip 遍历会漏掉这个 story 的不一致（共同 mask 恰恰是这道闸门最要害的校验）。
    if len(fr_normal.story_scores) != len(fr_shift.story_scores):
        raise ValueError(
            f"[{model}/H{H}] normal/shift 保留的 story 数不一致 "
            f"({len(fr_normal.story_scores)} vs {len(fr_shift.story_scores)}) → "
            f"共同 mask 不成立（某 story 在一条件下评分点数为 0 被跳过）")
    mask_bit_identical = True
    for a, b in zip(fr_normal.story_scores, fr_shift.story_scores):
        if a.story != b.story:
            raise ValueError(f"[{model}/H{H}] normal/shift story 顺序不一致")
        if a.scoring_mask is None or b.scoring_mask is None:
            mask_bit_identical = False
            break
        if not np.array_equal(a.scoring_mask, b.scoring_mask):
            raise ValueError(
                f"[{model}/H{H}/{a.story}] normal 与 shift 评分 mask 逐元素不相同 → "
                f"非共同 mask，配对不成立（n_eff {a.n_eff_tr} vs {b.n_eff_tr}）")

    # 选中 λ 落盘（供 M4/审计追溯，两条件各一份）
    np.savez(out_dir / f"valphas_{model}_H{H}.npz",
             normal=fr_normal.valphas, shift=fr_shift.valphas)

    normal_roi = {n: float(np.tanh(z)) for n, z in fr_normal.roi_z.items()}
    shift_roi = {n: float(np.tanh(z)) for n, z in fr_shift.roi_z.items()}
    shift_differs = any(abs(normal_roi[n] - shift_roi[n]) > 1e-6 for n in normal_roi)

    # 独立拟合的**程序化证据**（M3 交付物3：normal/shift 各自 fit 的追溯）：shift 路径
    # 的特征是位移过的，scaler/PCA 在不同数据上 fit → evr@100 必然不同；valphas 也应
    # 不完全相同。若二者都完全一致，反而提示拟合对象被复用（结构上不该发生）。
    evr_independent = fr_normal.evr_at_k != fr_shift.evr_at_k
    valphas_independent = not np.array_equal(fr_normal.valphas, fr_shift.valphas)
    independent_fit_evidence = bool(evr_independent or valphas_independent)

    all_finite = bool(np.all(np.isfinite(fr_normal.voxel_r))
                      and np.all(np.isfinite(fr_shift.voxel_r)))
    not_all_zero = bool(np.any(fr_normal.voxel_r != 0) and np.any(fr_shift.voxel_r != 0))

    return {
        "model": model, "H": H, "layer": layer,
        "model_id": feat_meta.get("model_id"),
        "revision": feat_meta.get("revision"),
        "layer_index": feat_meta.get("layer_main" if layer == "main" else "layer_final"),
        "code_version": feat_meta.get("code_version"),
        "train_n_stories": len(train_stories),
        "test_stories": test_stories,
        "voxel_r_shape": list(fr_normal.voxel_r.shape),
        "common_mask_n_eff_tr": fr_normal.n_eff_tr,
        "scoring_mask_bit_identical": bool(mask_bit_identical),
        "evr_at_100_normal": fr_normal.evr_at_k,
        "evr_at_100_shift": fr_shift.evr_at_k,
        "roi_r_normal": normal_roi,
        "roi_r_shift": shift_roi,
        "voxel_r_mean_normal": float(np.nanmean(fr_normal.voxel_r)),
        "voxel_r_mean_shift": float(np.nanmean(fr_shift.voxel_r)),
        "valphas_stats_normal": _valphas_stats(fr_normal.valphas),
        "valphas_stats_shift": _valphas_stats(fr_shift.valphas),
        "leakage_audit_pass": True,
        "shift_differs_from_normal": bool(shift_differs),
        "independent_fit_evidence": independent_fit_evidence,
        "all_finite": all_finite,
        "not_all_zero": not_all_zero,
        "per_story_normal": [
            {"story": ss.story, "n_eff_tr": ss.n_eff_tr,
             "roi_r": {n: float(np.tanh(z)) for n, z in ss.roi_z.items()}}
            for ss in fr_normal.story_scores
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS01",
                    help="新被试（冻结条件建议 UTS01）；三被试扩展下用 UTS01 非偏离")
    ap.add_argument("--H", type=int, nargs="+", default=DEFAULT_H)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--layer", default="main")
    ap.add_argument("--fold", default="fold_0",
                    help="frozen/fold_split.json 中用作这一个 outer fold 的键名")
    ap.add_argument("--solver", default="himalaya", choices=["himalaya", "numpy"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-name", default="m3_new_subject_slice")
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
    roi_cols = {k: v for k, v in roi_cols_all.items()
                if k in ("left_IFG", "bilateral_PT")}

    print(f"[m3] subject={args.subject} H={args.H} models={args.models} "
          f"layer={args.layer} fold={args.fold} solver={args.solver} "
          f"dtype={args.dtype} seed={seed}", flush=True)
    print(f"[m3] 训练={len(train_stories)}故事 测试={test_stories}", flush=True)

    out_dir = Path(paths["results_dir"]) / args.out_name / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for model in args.models:
        for H in args.H:
            results.append(run_one_group(
                model, H, args.layer, args.subject, train_stories, test_stories,
                roi_cols, paths["cache_dir"], ds["data_dir"], ds["respdict"],
                Path(paths["frozen_dir"]) / "word_index.parquet",
                solver, seed, args.dtype, out_dir,
            ))

    # 维度一致性：同一被试所有 (model,H) 的 held-out voxel 维度应一致（= 该被试 mask 列数）
    expected_V = results[0]["voxel_r_shape"]
    dims_ok = all(r["voxel_r_shape"] == expected_V and r["voxel_r_shape"][0] > 0
                  for r in results)

    # M3 验收标准逐项核对（迁移正确性，非架构结论）
    verdict = {
        "1_dims_correct_no_leakage": bool(
            dims_ok and all(r["leakage_audit_pass"] for r in results)),
        "2_lambda_boundary_recorded": all(  # 记录了两条件边界命中率（判读留给人工，不硬失败）
            "valphas_stats_normal" in r and "valphas_stats_shift" in r for r in results),
        "3_normal_shift_independent_fit": all(  # 程序化证据：evr@100 或 valphas 不同
            r["independent_fit_evidence"] for r in results),
        "4_scoring_mask_bit_identical": all(
            r["scoring_mask_bit_identical"] for r in results),
        "5_heldout_r_finite_nonzero": all(
            r["all_finite"] and r["not_all_zero"] for r in results),
        "6_shift_differs_from_normal": all(
            r["shift_differs_from_normal"] for r in results),
        "7_evr_recorded": all(
            r["evr_at_100_normal"] is not None and r["evr_at_100_shift"] is not None
            for r in results),
        "8_manifest_traceable": all(r.get("revision") for r in results),
    }
    all_pass = all(verdict.values())

    manifest = {
        "phase": "M3 new-subject vertical slice (three-subject extension)",
        "frozen_condition": "new subject x primary layer x 1 outer fold x 4 models "
                            "x H={8,32,128} x normal+shift",
        "note": "UTS01/UTS02 数据已下载，用 UTS01 是冻结条件本身，非预注册偏离",
        "git_commit": git_commit_hash(),
        "subject": args.subject, "models": args.models, "H_list": args.H,
        "layer": args.layer, "fold_name": args.fold, "seed": seed,
        "solver": args.solver, "dtype": args.dtype, "shift_seconds": SHIFT_SECONDS,
        "lambda_grid": "logspace(-2,7,19)", "lambda_grid_freeze_tag": "m3a-lambda-refreeze",
        "uts03_reference_magnitude": {
            "note": "UTS03 pilot 主层 IFG 量级参考（判读用，非通过阈值）：约 0.13-0.14",
        },
        "results": results,
        "verdict": verdict,
        "verdict_all_pass": all_pass,
        "spec": "frozen/analysis_spec.yaml",
    }
    with open(out_dir / f"m3_slice_{args.fold}.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n[m3] === 结果汇总（迁移正确性，非架构比较）===", flush=True)
    for r in results:
        print(f"[m3] {r['model']}/H{r['H']}: "
              f"IFG normal={r['roi_r_normal'].get('left_IFG', float('nan')):.4f} "
              f"shift={r['roi_r_shift'].get('left_IFG', float('nan')):.4f}  "
              f"evr@100={r['evr_at_100_normal']:.3f}  "
              f"λ撞上界={r['valphas_stats_normal']['hit_max_frac']:.3f}", flush=True)
    print(f"\n[m3] 验收标准: {json.dumps(verdict, ensure_ascii=False)}", flush=True)
    print(f"[m3] 全部通过: {'✅' if all_pass else '⚠️ 有未通过项，见上'}", flush=True)
    print(f"[m3] 已保存 → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
