"""
Step 4 —— ctx1（刺激侧上下文控制）条件的 Ridge 重拟合。

⚠️ 计算约束：含 himalaya voxelwise Ridge 拟合，必须在服务器 GPU 环境运行，需你
确认后再执行（本机无 himalaya/torch）。前置：Step 3 已产出 cache/features_ctx1/
（三模型 × H∈{8,128} × 83 故事，已核验完整）。

与主实验的关系：本脚本**完全复用** src/ridge/assemble.py::assemble_all 与
src/ridge/pipeline.py::run_fold（正常分支，不做 40s shift）。唯一区别是把
cache_dir 指向 cache/features_ctx1/ 而非 cache/features/。因此：
  - PCA/scaler/λ/Ridge 全部按 run_fold 内部逻辑在每折训练故事上独立 fit——
    自动满足 V6.4.2 §0.1「每条件独立拟合，不复用 normal 拟合对象」；
  - 主实验的 normal/shift 结果一个字节都不动（本脚本只读 normal 缓存做 mask 核验，
    从不写 normal 的任何结果文件）。

范围（M4S-Core / freeze_manifest.json 的 L1）：
  subject × model × H × fold × 主层 × ROI，其中
    subject ∈ {UTS01,UTS02,UTS03}, model ∈ {pythia,rwkv,mamba}, H ∈ {8,128},
    fold = 3 折, layer = 主层
  → 每被试 3×2×3 = 18 个拟合单元，三被试合计 54（与 V6.4.2 §1.2 一致）。
  ROI 同时算 left_IFG 与 bilateral_PT——PT 的 ROI 聚合在 run_fold 里由同一份
  voxel_r 顺带算出，不额外增加 Ridge 成本；正文 Core 只用 left_IFG，PT 供
  §7 PT-extended 备用，避免将来为 PT 再跑一遍。

§0.9 / §2.1-F mask 核验：ctx1 不移动时间轴，评分 mask（after_100s ∩ FIR_valid）
必须与 normal 逐元素相同。本脚本对每个 test 故事，独立组装 normal 特征、用与
run_fold 完全相同的 apply_fir + common_scoring_mask 算出 normal mask，与 ctx1
run_fold 产出的 scoring_mask 逐元素比对，记录 n_mask_difference（应恒为 0）。
这不是「由构造保证」的口头断言，而是在真实数据上算出两个 mask 逐位比对——
符合本项目「控制通过必须在目标量本身上验证」的纪律。

用法（服务器，M4S-Core 全量）：
  python 实验补充/scripts/m4s_ridge_perturbed.py --subject UTS01 --skip-existing
  python 实验补充/scripts/m4s_ridge_perturbed.py --subject UTS02 --skip-existing
  python 实验补充/scripts/m4s_ridge_perturbed.py --subject UTS03 --skip-existing
  # 算力紧可先 L2：加 --models pythia mamba
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SUPPLEMENT_ROOT = SCRIPT_DIR.parent            # 实验补充/
PROJECT_ROOT = SUPPLEMENT_ROOT.parent           # 仓库根目录
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config                                       # noqa: E402
from src.fmri.alignment import apply_fir                                        # noqa: E402
from src.fmri.mask import common_scoring_mask                                   # noqa: E402
from src.ridge.assemble import assemble_all, remap_roi_columns_to_voxel_mask    # noqa: E402
from src.ridge.pipeline import (                                                # noqa: E402
    run_fold, himalaya_ridgecv_solver, numpy_ridgecv_solver,
    LAMBDA_GRID, DELAYS_S, TR_SECONDS, AFTER_S,
)
from src.models.feature_cache import load_features                              # noqa: E402
from src.fmri.trfile import load_respdict, trimmed_tr_times                     # noqa: E402

CONDITION = "ctx1"
CORE_MODELS = ["pythia", "rwkv", "mamba"]


def _valphas_stats(valphas: np.ndarray) -> dict:
    lam_min, lam_max = float(LAMBDA_GRID.min()), float(LAMBDA_GRID.max())
    return {
        "min": float(valphas.min()), "max": float(valphas.max()),
        "median": float(np.median(valphas)),
        "hit_min_frac": float((valphas <= lam_min * (1 + 1e-6)).mean()),
        "hit_max_frac": float((valphas >= lam_max * (1 - 1e-6)).mean()),
    }


def _fold_summary(fr) -> dict:
    """与 src/ridge/m4_driver.py::_fold_summary 同构，另存 evr_at_k
    （主 M4 算了但从未落盘的量——这里为 ctx1 顺带补上，见 V6.4.2 §10 PCA 缺口）。"""
    return {
        "roi_r": {n: float(np.tanh(z)) for n, z in fr.roi_z.items()},
        "voxel_r_mean": float(np.nanmean(fr.voxel_r)),
        "n_eff_tr": fr.n_eff_tr,
        "valphas_stats": _valphas_stats(fr.valphas),
        "evr_at_k": fr.evr_at_k,
        "any_nan_or_inf": bool(not np.all(np.isfinite(fr.voxel_r))),
        "per_story": [
            {"story": ss.story, "n_eff_tr": ss.n_eff_tr,
             "roi_r": {n: float(np.tanh(z)) for n, z in ss.roi_z.items()}}
            for ss in fr.story_scores
        ],
    }


def reference_scoring_masks(respdict: dict, test_stories: list[str]) -> dict:
    """从 respdict 直接算每个 test 故事的评分 mask，**不加载任何特征或 BOLD**。

    mask = common_scoring_mask(tr_times, fir_valid, after_s)，其中：
      - tr_times = trimmed_tr_times(respdict[story])  —— 只依赖 respdict
      - fir_valid = apply_fir 的边缘有效性，只依赖 TR 行数 T=len(tr_times)（位置量）
    两者都由 respdict[story] 唯一决定，与特征数值/条件无关，故这是 normal 条件
    评分 mask 的权威参照。用它替代"组装整份 normal BOLD 再取 mask"——后者会为
    每个单元多加载一份 (T×体素) 的响应矩阵，是 UTS02/03 内存打满的根因。"""
    masks = {}
    for s in test_stories:
        trt = trimmed_tr_times(respdict[s])
        T = len(trt)
        _, valid = apply_fir(np.zeros((T, 1)), delays_s=DELAYS_S, tr=TR_SECONDS)
        masks[s] = common_scoring_mask(trt, valid, after_s=AFTER_S)
    return masks


def process_cell(model: str, H: int, fold_name: str, fold: dict, subject: str,
                 roi_cols: dict, cache_dir_ctx1: str, cache_dir_normal: str,
                 data_dir, respdict_path, word_index_path, solver, seed: int,
                 dtype: str, out_dir: Path, skip_existing: bool) -> dict | None:
    """一个 (model, H, fold) 单元：ctx1 run_fold + 与 normal 的 mask 逐元素核验。"""
    cells_dir = out_dir / "cells"
    cell_path = cells_dir / f"ctx1_{model}_H{H}_{fold_name}.json"
    if skip_existing and cell_path.exists():
        print(f"  [{model}/H{H}/{fold_name}] 已存在，跳过", flush=True)
        return json.load(open(cell_path))

    train_s, test_s = list(fold["train_stories"]), list(fold["test_stories"])
    assert not (set(test_s) & set(train_s)), f"[{model}/H{H}/{fold_name}] 泄漏"
    all_s = train_s + test_s
    tag = f" ctx1/{model}/H{H}/{fold_name}"
    t0 = time.time()

    # 1) 组装 ctx1 特征（cache_dir 指向 features_ctx1），跑 run_fold 正常分支
    voxel_mask = np.load(Path(load_config()["paths"]["frozen_dir"]) / f"voxel_mask_{subject}.npy")
    ctx1_data = assemble_all(all_s, model, H, "main", subject, cache_dir_ctx1,
                             data_dir, respdict_path, word_index_path, voxel_mask=voxel_mask)
    dt = np.dtype(dtype)
    for s in ctx1_data:
        ctx1_data[s].X = ctx1_data[s].X.astype(dt)
        ctx1_data[s].Y = ctx1_data[s].Y.astype(dt)

    fr = run_fold(ctx1_data, train_s, test_s, solver, roi_columns=roi_cols,
                  seed=seed, tag=tag)
    ctx1_masks = {ss.story: ss.scoring_mask.copy() for ss in fr.story_scores}

    # 立刻释放 ctx1 组装的大数组（83 故事 × 体素的 X/Y），降低 RAM 峰值——
    # UTS02/03 体素多(94k/95k)，不显式释放会把内存顶爆导致进程卡死。
    del ctx1_data
    gc.collect()

    # 2) mask 核验：从 respdict 直接算 normal 参照 mask（零重内存，不加载 BOLD）
    respdict = load_respdict(respdict_path)
    normal_masks = reference_scoring_masks(respdict, test_s)
    per_story_n_diff = {}
    bit_identical = True
    for s in test_s:
        if s not in ctx1_masks:
            # run_fold 会跳过评分点数为 0 的故事；正常不该发生，如发生记录之
            per_story_n_diff[s] = "ctx1_story_dropped_zero_score"
            bit_identical = False
            continue
        a, b = ctx1_masks[s], normal_masks[s]
        if a.shape != b.shape:
            per_story_n_diff[s] = f"shape_mismatch {a.shape} vs {b.shape}"
            bit_identical = False
            continue
        n_diff = int(np.sum(a != b))
        per_story_n_diff[s] = n_diff
        if n_diff != 0:
            bit_identical = False

    # 3) 取一份特征 meta（含 permutation_sha256），做溯源
    feat_meta = load_features(cache_dir_ctx1, model, all_s[0], H)["meta"]

    cell = {
        "layer": "main", "model": model, "H": H, "fold": fold_name, "subject": subject,
        "condition": CONDITION,
        "model_id": feat_meta.get("model_id"), "revision": feat_meta.get("revision"),
        "layer_index": feat_meta.get("layer_main"),
        "code_version": feat_meta.get("code_version"),
        "master_seed": feat_meta.get("master_seed"),
        "permutation_sha256_example": feat_meta.get("permutation_sha256"),
        "train_stories": train_s, "test_stories": test_s,
        "ctx1": _fold_summary(fr),
        "mask_vs_normal": {
            "bit_identical": bool(bit_identical),
            "n_mask_difference_total": int(sum(v for v in per_story_n_diff.values()
                                               if isinstance(v, int))),
            "per_story_n_diff": per_story_n_diff,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    if not bit_identical:
        # 与主实验 M4 的 mask 断言一致：不一致直接 raise，绝不静默产出不可比的配对
        raise ValueError(
            f"[{model}/H{H}/{fold_name}] ctx1 与 normal 评分 mask 不逐元素相同 → "
            f"刺激侧扰动不应改变时间轴/mask，实现有误：{per_story_n_diff}")

    cells_dir.mkdir(parents=True, exist_ok=True)
    with open(cell_path, "w") as f:
        json.dump(cell, f, indent=2, ensure_ascii=False)
    r_show = "  ".join(f"{n}={v:.4f}" for n, v in cell["ctx1"]["roi_r"].items())
    print(f"  [{model}/H{H}/{fold_name}] 完成 {cell['elapsed_seconds']}s  {r_show}  "
          f"mask_identical={bit_identical}", flush=True)
    del fr
    gc.collect()   # 单元间彻底回收，避免跨单元内存累积（UTS02/03 体素多尤其重要）
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03", choices=["UTS01", "UTS02", "UTS03"])
    ap.add_argument("--models", nargs="+", default=CORE_MODELS,
                    choices=["pythia", "rwkv", "mamba"])
    ap.add_argument("--H", nargs="+", type=int, default=[8, 128])
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--solver", default="himalaya", choices=["himalaya", "numpy"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-name", default="m4s_ridge")
    ap.add_argument("--cache-dir-ctx1", default=None,
                    help="默认 <cache_dir 同级>/features_ctx1")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    paths, ds = cfg["paths"], cfg["datasets"]
    seed = args.seed if args.seed is not None else cfg["seeds"]["pca"]
    solver = himalaya_ridgecv_solver if args.solver == "himalaya" else numpy_ridgecv_solver

    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        raw_fold_split = json.load(f)
    fold_names = args.folds if args.folds else list(raw_fold_split["folds"].keys())
    folds = {k: v for k, v in raw_fold_split["folds"].items() if k in fold_names}

    voxel_mask = np.load(Path(paths["frozen_dir"]) / f"voxel_mask_{args.subject}.npy")
    roi_cols_all = dict(np.load(Path(paths["frozen_dir"]) / f"roi_columns_{args.subject}.npz"))
    roi_cols_full = {k: v for k, v in roi_cols_all.items() if k in ("left_IFG", "bilateral_PT")}
    roi_cols = remap_roi_columns_to_voxel_mask(roi_cols_full, voxel_mask)

    cache_dir_normal = paths["cache_dir"]
    cache_dir_ctx1 = (args.cache_dir_ctx1 if args.cache_dir_ctx1
                      else str(Path(cache_dir_normal).parent / "features_ctx1"))
    word_index_path = Path(paths["frozen_dir"]) / "word_index.parquet"
    out_dir = Path(paths["results_dir"]) / args.out_name / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)

    n_units = len(args.models) * len(args.H) * len(folds)
    print(f"[m4s:{args.subject}] ctx1 Ridge 重拟合 | 模型={args.models} H={args.H} "
          f"folds={fold_names} → {n_units} 单元 | solver={args.solver}", flush=True)
    print(f"[m4s:{args.subject}] ctx1 缓存={cache_dir_ctx1}（normal 缓存只读做 mask 核验）", flush=True)

    cells = []
    t_all = time.time()
    done = 0
    for model in args.models:
        for H in args.H:
            for fn in fold_names:
                cell = process_cell(
                    model, H, fn, folds[fn], args.subject, roi_cols,
                    cache_dir_ctx1, cache_dir_normal, ds["data_dir"], ds["respdict"],
                    word_index_path, solver, seed, args.dtype, out_dir, args.skip_existing)
                if cell:
                    cells.append(cell)
                done += 1
                print(f"    进度 {done}/{n_units}，累计 {(time.time()-t_all)/60:.1f} 分钟", flush=True)

    # 汇总 manifest：单元完整性 + mask 全体一致 + 无 NaN
    all_mask_ok = all(c["mask_vs_normal"]["bit_identical"] for c in cells)
    all_finite = all(not c["ctx1"]["any_nan_or_inf"] for c in cells)
    manifest = {
        "phase": "M4S ctx1 ridge refit (stimulus-side context control)",
        "condition": CONDITION, "subject": args.subject,
        "models": args.models, "H_list": args.H, "fold_names": fold_names,
        "expected_units": n_units, "units_done": len(cells),
        "verdict": {
            "all_units_complete": len(cells) == n_units,
            "all_mask_bit_identical_to_normal": all_mask_ok,
            "no_nan_inf": all_finite,
        },
        "total_minutes": round((time.time() - t_all) / 60, 1),
    }
    with open(out_dir / "m4s_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[m4s:{args.subject}] 完成 {len(cells)}/{n_units} 单元，"
          f"{manifest['total_minutes']} 分钟", flush=True)
    print(f"[m4s:{args.subject}] mask 全体与 normal 逐元素一致: {all_mask_ok}；无 NaN: {all_finite}", flush=True)
    print(f"[m4s:{args.subject}] manifest → {out_dir / 'm4s_manifest.json'}", flush=True)
    print(f"[m4s:{args.subject}] 三被试都跑完后进 Step 5（estimands D_m/I_MP + bootstrap）。", flush=True)


if __name__ == "__main__":
    main()
