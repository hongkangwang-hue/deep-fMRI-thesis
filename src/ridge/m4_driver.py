"""
M4 共享驱动逻辑 —— 供 4 个单模型入口脚本
(scripts/m4_{pythia,mamba,rwkv,awd_lstm}.py) 和汇总脚本
(scripts/m4_aggregate.py) 复用，避免 4 个模型各自抄一份。

冻结条件（milestone/里程碑总览_V4.9_最终冻结版.md 里程碑4）：
  主层正常/shifted：3 subjects × 2 ROI(IFG+PT) × 4 models × 3 H × 3 folds
  最终层正常（无 shift）：3 subjects × 1 ROI(左IFG) × 4 models × 3 H × 3 folds

⚠️ 预注册偏离（已用户确认，2026-07-01，与 M3b 同一决策）：3 subjects → UTS03 单被试。

矩阵单元 = (model, H, layer, fold)。每个单元独立算、独立落盘到
results/<out-name>/<subject>/cells/，支持按单元跳过的断点续跑。

主层单元：run_fold 正常 + 40s shift 各一次（normal/shift 共用 mask，同 M3b）。
最终层单元：只有 run_fold 正常一次（IFG only，无 shift——非毕业闭环必需项）。
单模型全跑 = 3H×3folds×(2条件主层+1条件最终层) = 18+9 = 27 次 run_fold；
4 模型合计 108 次，与之前单文件版本矩阵规模一致，只是拆到 4 个进程里分别跑。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fmri.alignment import shift_story_no_wrap        # noqa: E402
from src.ridge.assemble import assemble_all                # noqa: E402
from src.ridge.pipeline import StoryData, run_fold, LAMBDA_GRID  # noqa: E402
from src.models.feature_cache import load_features          # noqa: E402

SHIFT_SECONDS = 40.0
TR_SECONDS = 2.0
ALL_MODELS = ["pythia", "mamba", "rwkv", "awd_lstm"]


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def make_shifted_story_data(story_data: dict[str, StoryData]) -> tuple[dict, dict]:
    """对每个故事的 X 做 40s 位移，返回 (shifted_story_data, shift_valid_by_story)。同 M3b。"""
    shifted, valid_by_story = {}, {}
    for s, sd in story_data.items():
        Xs, valid = shift_story_no_wrap(sd.X, seconds=SHIFT_SECONDS, tr=TR_SECONDS)
        shifted[s] = StoryData(X=Xs, Y=sd.Y, tr_times=sd.tr_times)
        valid_by_story[s] = valid
    return shifted, valid_by_story


def _valphas_stats(valphas: np.ndarray) -> dict:
    lam_min, lam_max = float(LAMBDA_GRID.min()), float(LAMBDA_GRID.max())
    return {
        "min": float(valphas.min()), "max": float(valphas.max()),
        "median": float(np.median(valphas)),
        "hit_min_frac": float((valphas <= lam_min * (1 + 1e-6)).mean()),
        "hit_max_frac": float((valphas >= lam_max * (1 - 1e-6)).mean()),
    }


def _fold_summary(fr) -> dict:
    return {
        "roi_r": {n: float(np.tanh(z)) for n, z in fr.roi_z.items()},
        "voxel_r_mean": float(np.nanmean(fr.voxel_r)),
        "n_eff_tr": fr.n_eff_tr,
        "valphas_stats": _valphas_stats(fr.valphas),
        "any_nan_or_inf": bool(not np.all(np.isfinite(fr.voxel_r))),
        "per_story": [
            {"story": ss.story, "n_eff_tr": ss.n_eff_tr,
             "roi_r": {n: float(np.tanh(z)) for n, z in ss.roi_z.items()}}
            for ss in fr.story_scores
        ],
    }


def process_group(model: str, H: int, layer: str, subject: str, fold_split: dict,
                  roi_cols: dict, cache_dir, data_dir, respdict_path, word_index_path,
                  solver, seed: int, dtype: str, out_dir: Path, skip_existing: bool,
                  progress: dict) -> None:
    """处理一个 (model, H, layer) 组合：组装一次特征，3 折复用。"""
    do_shift = layer == "main"
    prefix = "main" if do_shift else "final"
    fold_names = list(fold_split["folds"].keys())
    cells_dir = out_dir / "cells"

    pending = [fn for fn in fold_names
               if not (skip_existing and (cells_dir / f"{prefix}_{model}_H{H}_{fn}.json").exists())]
    if not pending:
        progress["done"] += len(fold_names)
        print(f"[m4:{model}] {prefix} H={H} 全部 fold 已存在，跳过 "
              f"（进度 {progress['done']}/{progress['total']}）", flush=True)
        return

    print(f"\n[m4:{model}] === {prefix} H={H}（待算 fold: {pending}）===", flush=True)
    t0 = time.time()
    all_stories = sorted({s for fo in fold_split["folds"].values()
                          for s in fo["train_stories"] + fo["test_stories"]})
    story_data = assemble_all(all_stories, model, H, layer, subject, cache_dir,
                              data_dir, respdict_path, word_index_path)
    dt = np.dtype(dtype)
    for s in story_data:
        story_data[s].X = story_data[s].X.astype(dt)
        story_data[s].Y = story_data[s].Y.astype(dt)
    print(f"[m4:{model}] {prefix} H={H} 组装完成 {time.time()-t0:.1f}s", flush=True)

    feat_meta = load_features(cache_dir, model, all_stories[0], H)["meta"]
    shifted_data, valid_by_story = (None, None)
    if do_shift:
        shifted_data, valid_by_story = make_shifted_story_data(story_data)

    cells_dir.mkdir(parents=True, exist_ok=True)
    already_done = len(fold_names) - len(pending)
    progress["done"] += already_done

    for fn in pending:
        fold = fold_split["folds"][fn]
        train_s, test_s = list(fold["train_stories"]), list(fold["test_stories"])
        assert not (set(test_s) & set(train_s)), f"[{model}/H{H}/{fn}] 泄漏"
        tag = f" {prefix}/{model}/H{H}/{fn}"
        shift_valid = {s: valid_by_story[s] for s in test_s} if do_shift else None

        fr_normal = run_fold(story_data, train_s, test_s, solver, roi_columns=roi_cols,
                             seed=seed, tag=tag + "/normal", shift_valid_by_story=shift_valid)
        cell = {
            "layer": layer, "model": model, "H": H, "fold": fn, "subject": subject,
            "model_id": feat_meta.get("model_id"), "revision": feat_meta.get("revision"),
            "layer_index": feat_meta.get("layer_main" if layer == "main" else "layer_final"),
            "code_version": feat_meta.get("code_version"),
            "train_stories": train_s, "test_stories": test_s,
            "normal": _fold_summary(fr_normal),
            "leakage_audit_pass": True,
        }
        valphas = {"normal": fr_normal.valphas}

        if do_shift:
            fr_shift = run_fold(shifted_data, train_s, test_s, solver, roi_columns=roi_cols,
                                seed=seed, tag=tag + "/shift", shift_valid_by_story=shift_valid)
            common_mask_verified = fr_normal.n_eff_tr == fr_shift.n_eff_tr and all(
                a.n_eff_tr == b.n_eff_tr and a.story == b.story
                for a, b in zip(fr_normal.story_scores, fr_shift.story_scores))
            shift_differs = any(
                abs(np.tanh(fr_normal.roi_z[n]) - np.tanh(fr_shift.roi_z[n])) > 1e-6
                for n in fr_normal.roi_z)
            cell["shift"] = _fold_summary(fr_shift)
            cell["common_mask_verified"] = bool(common_mask_verified)
            cell["shift_differs_from_normal"] = bool(shift_differs)
            valphas["shift"] = fr_shift.valphas

        with open(cells_dir / f"{prefix}_{model}_H{H}_{fn}.json", "w") as f:
            json.dump(cell, f, indent=2, ensure_ascii=False)
        np.savez(cells_dir / f"valphas_{prefix}_{model}_H{H}_{fn}.npz", **valphas)

        progress["done"] += 1
        print(f"[m4:{model}] 已保存 {prefix}_{model}_H{H}_{fn}.json "
              f"（进度 {progress['done']}/{progress['total']}）", flush=True)

    del story_data
    if shifted_data is not None:
        del shifted_data


def run_model_matrix(model: str, H_list: list[int], layers: list[str], fold_split: dict,
                     roi_cols_main: dict, roi_cols_final: dict, cache_dir, data_dir,
                     respdict_path, word_index_path, solver, seed: int, dtype: str,
                     out_dir: Path, skip_existing: bool, subject: str) -> None:
    """单模型的完整 H×layer×fold 循环——各模型入口脚本的唯一调用入口。"""
    total = len(H_list) * len(layers) * len(fold_split["folds"])
    progress = {"done": 0, "total": total}
    for H in H_list:
        for layer in layers:
            roi_cols = roi_cols_main if layer == "main" else roi_cols_final
            process_group(model, H, layer, subject, fold_split, roi_cols, cache_dir,
                          data_dir, respdict_path, word_index_path, solver, seed, dtype,
                          out_dir, skip_existing, progress)


def build_manifest(out_dir: Path, models: list[str], H_list: list[int],
                   fold_names: list[str], subject: str) -> dict:
    """扫描 cells/ 下已有单元文件（可能来自 4 个不同模型脚本的独立进程），重建汇总
    manifest + M4 验收标准核对。不跑计算——供 scripts/m4_aggregate.py 调用。"""
    cells_dir = out_dir / "cells"
    main_cells, final_cells = {}, {}
    for model in models:
        for H in H_list:
            for fn in fold_names:
                p = cells_dir / f"main_{model}_H{H}_{fn}.json"
                if p.exists():
                    main_cells[(model, H, fn)] = json.load(open(p))
                pf = cells_dir / f"final_{model}_H{H}_{fn}.json"
                if pf.exists():
                    final_cells[(model, H, fn)] = json.load(open(pf))

    expected = len(models) * len(H_list) * len(fold_names)
    main_missing = [f"{m}/H{H}/{fn}" for m in models for H in H_list for fn in fold_names
                    if (m, H, fn) not in main_cells]
    final_missing = [f"{m}/H{H}/{fn}" for m in models for H in H_list for fn in fold_names
                     if (m, H, fn) not in final_cells]

    verdict = {
        "1_main_matrix_complete": len(main_missing) == 0 and len(main_cells) == expected,
        "2_final_ifg_matrix_complete": len(final_missing) == 0 and len(final_cells) == expected,
        "3_per_story_saved": all(len(c["normal"]["per_story"]) > 0 for c in main_cells.values())
                            and all(len(c["normal"]["per_story"]) > 0 for c in final_cells.values()),
        "4_no_nan_inf": all(
            not c["normal"]["any_nan_or_inf"] and not c.get("shift", {}).get("any_nan_or_inf", False)
            for c in main_cells.values()
        ) and all(not c["normal"]["any_nan_or_inf"] for c in final_cells.values()),
        "5_common_mask_used": all(c.get("common_mask_verified") for c in main_cells.values()),
        "6_manifest_traceable": all(c.get("revision") for c in main_cells.values())
                               and all(c.get("revision") for c in final_cells.values()),
    }
    manifest = {
        "phase": "M4 full matrix (single-subject deviation)",
        "frozen_condition": "3 subjects x 2 ROI(main layer) / 1 ROI(final layer, IFG) "
                            "x 4 models x 3 H x 3 folds",
        "deviation": {
            "field": "subjects", "frozen_value": "UTS01,UTS02,UTS03", "actual_value": subject,
            "reason": "dataset availability constraint (only UTS03 downloaded); "
                      "user-confirmed 2026-07-01, same decision as M3b",
        },
        "git_commit": git_commit_hash(),
        "subject": subject, "models": models, "H_list": H_list, "fold_names": fold_names,
        "expected_cells_per_matrix": expected,
        "main_layer_cells_done": len(main_cells), "main_layer_missing": main_missing,
        "final_layer_cells_done": len(final_cells), "final_layer_missing": final_missing,
        "verdict": verdict,
        "verdict_all_pass": all(verdict.values()),
        "lambda_grid": "logspace(-2,7,19)", "lambda_grid_freeze_tag": "m3a-lambda-refreeze",
        "spec": "frozen/analysis_spec.yaml",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "m4_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n[m4:aggregate] === 汇总 ===", flush=True)
    print(f"[m4:aggregate] main层: {len(main_cells)}/{expected} 完成" +
          (f"  缺: {main_missing[:5]}{'...' if len(main_missing) > 5 else ''}"
           if main_missing else ""), flush=True)
    print(f"[m4:aggregate] final层(IFG): {len(final_cells)}/{expected} 完成" +
          (f"  缺: {final_missing[:5]}{'...' if len(final_missing) > 5 else ''}"
           if final_missing else ""), flush=True)
    print(f"[m4:aggregate] 验收标准: {json.dumps(verdict, ensure_ascii=False)}", flush=True)
    print(f"[m4:aggregate] 全部通过: {'✅' if manifest['verdict_all_pass'] else '⚠️ 未完成/有未通过项'}",
          flush=True)
    print(f"[m4:aggregate] manifest → {out_dir / 'm4_manifest.json'}", flush=True)
    return manifest
