"""
硕士论文核查清单 —— 第10节 PCA / Ridge 部分。

Ridge λ 分布：M4 每个 cell 在拟合时已经把 λ 命中统计（min/max/median/hit_min_frac/
hit_max_frac）和逐体素 valphas 原始数组分别存进了 cell 的 json 和配套的
valphas_*.npz（src/ridge/m4_driver.py::_fold_summary + 同目录 np.savez）——这里只是
读取聚合，不重新拟合任何 Ridge。IQR（四分位范围）cell json 里没有现成字段，从
valphas_*.npz 原始数组现算分位数，同样是读已有数据，不是重算。

PCA evr@100：**已确认的真实缺口**——src/ridge/pipeline.py 的 run_fold 每次都会算
evr_at_k（PCA 前 100 个成分累计解释方差比），但 m4_driver.py::_fold_summary 从未把
它写进 cell json（只在内存里用过就丢了），所以服务器现有的 M4 结果文件里找不到这个
数，不是本脚本能读出来的——需要额外单独补算（见脚本末尾打印的说明，不在本脚本内
静默处理，因为那需要新的计算，按项目规矩必须交给用户确认后再跑，不能我自己代跑）。

用法（需先把 results/m4_full_matrix/<subject>/ 同步到本机同一相对路径）：
  python3 scripts/thesis_supplement_pca_ridge_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config   # noqa: E402

OUT_DIR = PROJECT_ROOT / "thesis_supplement"
SUBJECTS = ("UTS01", "UTS02", "UTS03")
MODELS = ("pythia", "mamba", "rwkv", "awd_lstm")


def _cell_iter(cells_dir: Path):
    for p in sorted(cells_dir.glob("main_*.json")):
        if p.name.startswith("valphas_"):
            continue
        yield p, json.load(open(p)), "main"
    for p in sorted(cells_dir.glob("final_*.json")):
        if p.name.startswith("valphas_"):
            continue
        yield p, json.load(open(p)), "final"


def build_ridge_audit(cfg: dict) -> pd.DataFrame:
    results_dir = Path(cfg["paths"]["results_dir"])
    rows = []
    for subj in SUBJECTS:
        cells_dir = results_dir / "m4_full_matrix" / subj / "cells"
        if not cells_dir.exists():
            print(f"  [跳过] {cells_dir} 不存在")
            continue
        for p, cell, layer in _cell_iter(cells_dir):
            for cond in ("normal", "shift"):
                if cond not in cell:
                    continue
                stats = cell[cond].get("valphas_stats")
                if stats is None:
                    continue
                # 找配套 valphas npz 算 IQR（cell json 命名 main_{model}_H{H}_{fold}.json
                # → npz 是 valphas_main_{model}_H{H}_{fold}.npz，见 m4_driver.py 存盘逻辑）
                npz_path = p.parent / f"valphas_{p.stem}.npz"
                q25 = q75 = None
                if npz_path.exists():
                    arr = np.load(npz_path)[cond]
                    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
                rows.append({
                    "subject": subj, "layer": layer, "model": cell["model"],
                    "H": cell["H"], "fold": cell["fold"], "condition": cond,
                    "lambda_median": stats["median"],
                    "lambda_q25": q25, "lambda_q75": q75,
                    "lambda_min_observed": stats["min"], "lambda_max_observed": stats["max"],
                    "hit_lower_bound_frac": stats["hit_min_frac"],
                    "hit_upper_bound_frac": stats["hit_max_frac"],
                })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Ridge λ 审计（读 M4 cells 已存的 valphas_stats + valphas npz，不重算）...")
    df = build_ridge_audit(cfg)
    if len(df):
        df.to_csv(OUT_DIR / "pca_ridge_audit.csv", index=False)
        print(f"已写 {OUT_DIR / 'pca_ridge_audit.csv'}（{len(df)} 行 = "
             f"被试×模型×H×fold×条件 的 λ 命中统计）")
        print("\n按模型汇总（跨被试/H/fold/条件）：")
        print(df.groupby("model")[["lambda_median", "hit_lower_bound_frac",
                                   "hit_upper_bound_frac"]].mean().to_string())
    else:
        print("没有任何 M4 cells 可读，pca_ridge_audit.csv 未生成——先同步 "
             "results/m4_full_matrix/<subject>/ 到本机。")

    print("\n" + "=" * 70)
    print("PCA evr@100：确认的真实缺口，不在本脚本产出范围内。")
    print("原因：src/ridge/m4_driver.py::_fold_summary 从未把 run_fold 算出的 "
         "evr_at_k 写进 cell json（只是没持久化，不是没算——PCA 本身在每次 "
         "run_fold 里都正常跑了）。现有服务器结果文件里确实找不到这个数。")
    print("如果需要补齐，最小侵入的做法是新写一个只做「特征组装 → fit "
         "scaler+PCA → 记录 evr_at_k」、跳过 Ridge 拟合本身的轻量脚本，对 "
         "4模型×3H×3折=36个组合各跑一次 PCA fit（CPU、不碰 GPU/Ridge/bootstrap，"
         "读已缓存的语言特征，量级是秒级/个）。这仍然是新计算，按项目规矩需要"
         "你确认后再在服务器上跑，我不会擅自执行。")
    print("=" * 70)


if __name__ == "__main__":
    main()
