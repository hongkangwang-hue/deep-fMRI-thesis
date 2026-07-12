"""
M6 —— 论文表格生成（Table 1 / Table 2 / QC 表）。只读现有产物，不重算。

Table 1  四 checkpoint 审计（model_id / revision / 主层·最终层索引 / code_version）
Table 2  Model × ROI 完整数值（r8/r32/r128 + 三类 Δr + RQ1 差值, 均带 95% CI）
QC 表    故事/有效 TR/voxel/ROI 列数/λ 边界命中 等质量与 attrition 指标

数据来源：Table1←M4 cells 的 meta + config；Table2←m5_results.json；QC←frozen 产物 +
m4_manifest + M4 cells。输出 CSV(机读) + Markdown(贴论文) 到 figures/<subject>/tables/。

用法：python3 scripts/m6_tables.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config      # noqa: E402
from src.viz.m6_data import (                   # noqa: E402
    load_results, r_curve, context_gain, rq1,
    MODELS, HS, ROIS, CORE_VS_PYTHIA,
)


def _ci(pt, lo, hi) -> str:
    if pt is None:
        return "—"
    return f"{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def _df_to_markdown(df: pd.DataFrame) -> str:
    """手写 markdown 表（不依赖 tabulate，服务器无需额外装包）。"""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _write(df: pd.DataFrame, outdir: Path, name: str, index=False):
    outdir.mkdir(parents=True, exist_ok=True)
    df = df.fillna("—")                          # 缺列（如非核心模型无 RQ1）显示 — 而非 nan
    df.to_csv(outdir / f"{name}.csv", index=index)
    with open(outdir / f"{name}.md", "w") as f:
        f.write(_df_to_markdown(df))
    print(f"[m6] 已写 {name}.csv / .md", flush=True)


def table1_audit(cells_dir: Path, cfg: dict, outdir: Path):
    """四 checkpoint 审计：从 M4 cells 的 meta 读实际用到的 model_id/revision/层索引。"""
    rows = []
    for m in MODELS:
        main = next(cells_dir.glob(f"main_{m}_H*_*.json"), None)
        final = next(cells_dir.glob(f"final_{m}_H*_*.json"), None)
        meta_main = json.load(open(main)) if main else {}
        meta_final = json.load(open(final)) if final else {}
        rows.append({
            "model": m,
            "model_id": meta_main.get("model_id", "—"),
            "revision": meta_main.get("revision", "—"),
            "primary_layer_idx": cfg["models"]["primary_layers"].get(m),
            "robustness_layer_idx": cfg["models"]["robustness_layers"].get(m),
            "layer_idx_in_main_cell": meta_main.get("layer_index"),
            "layer_idx_in_final_cell": meta_final.get("layer_index"),
            "code_version": meta_main.get("code_version", "—"),
            "role": "historical reference" if m == "awd_lstm" else "core",
        })
    _write(pd.DataFrame(rows), outdir, "table1_checkpoint_audit")


def table2_numbers(results: dict, outdir: Path):
    """Model × ROI 完整数值表（r + 三类 Δr + RQ1，均带 CI）。"""
    est = results["estimands"]
    rows = []
    for m in MODELS:
        for roi in ROIS:
            _, pts, los, his = r_curve(est, m, roi)
            row = {"model": m, "roi": roi}
            for H, p, lo, hi in zip(HS, pts, los, his):
                row[f"r_H{H}"] = _ci(p, lo, hi)
            for kind in ("local", "long", "total"):
                row[f"delta_{kind}"] = _ci(*context_gain(est, m, roi, kind))
            if roi == "left_IFG" and m in CORE_VS_PYTHIA:
                for H in HS:
                    row[f"rq1_vs_pythia_H{H}"] = _ci(*rq1(est, m, H))
            rows.append(row)
    _write(pd.DataFrame(rows), outdir, "table2_full_numbers")


def qc_table(cfg: dict, subject: str, cells_dir: Path, results: dict, outdir: Path):
    """QC/attrition：故事数、有效 TR、voxel/ROI 列数、λ 边界命中等。

    voxel_mask / roi_columns 按 subject 取该被试自己的冻结产物（三被试各不相同：
    UTS03 恒等、UTS01 排除 10、UTS02 排除 32），不得写死某一名被试。
    """
    frozen = Path(cfg["paths"]["frozen_dir"])
    fold_split = json.load(open(frozen / "fold_split.json"))
    vmask = json.load(open(frozen / f"voxel_mask_{subject}.json"))
    roi_cols = dict(np.load(frozen / f"roi_columns_{subject}.npz"))

    n_test = sum(len(v["test_stories"]) for v in fold_split["folds"].values())
    # 每折有效 TR（主层正常）：从任一模型的 main cell 读 per-story n_eff 汇总
    eff_tr = {}
    for fn in fold_split["folds"]:
        c = next(cells_dir.glob(f"main_pythia_H8_{fn}.json"), None)
        if c:
            cell = json.load(open(c))
            eff_tr[fn] = sum(ps["n_eff_tr"] for ps in cell["normal"]["per_story"])

    rows = [
        ("被试", f"{subject}（三被试扩展，逐被试独立建模）"),
        ("outer folds", fold_split.get("n_folds")),
        ("每折测试故事数", ", ".join(f"{k}={len(v['test_stories'])}"
                                     for k, v in fold_split["folds"].items())),
        ("CV 故事去重总数", n_test),
        ("BOLD voxel 保留数", f"{vmask['n_kept']}/{vmask['n_voxels']} "
                              f"(NaN排除{vmask['n_excluded_nan']}, 零方差排除{vmask['n_excluded_zero_var']})"),
        ("left_IFG 列数", len(roi_cols["left_IFG"])),
        ("bilateral_PT 列数", len(roi_cols["bilateral_PT"])),
        ("主层正常有效TR/折", ", ".join(f"{k}={v}" for k, v in eff_tr.items())),
        ("bootstrap 次数", results.get("n_boot")),
        ("bootstrap 种子", results.get("seed")),
        ("repeatability", "未计算（wheretheressmoke 无逐次响应，见 known_gaps）"),
    ]
    _write(pd.DataFrame(rows, columns=["指标", "值"]), outdir, "qc_table")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--m4-name", default="m4_full_matrix")
    ap.add_argument("--m5-name", default="m5_stats")
    args = ap.parse_args()

    cfg = load_config()
    cells_dir = Path(cfg["paths"]["results_dir"]) / args.m4_name / args.subject / "cells"
    m5_path = Path(cfg["paths"]["results_dir"]) / args.m5_name / args.subject / "m5_results.json"
    if not m5_path.exists():
        raise SystemExit(f"未找到 M5 结果：{m5_path}")
    if not cells_dir.exists():
        raise SystemExit(f"未找到 M4 cells：{cells_dir}")
    results = load_results(m5_path)
    outdir = Path(cfg["paths"]["figures_dir"]) / args.subject / "tables"

    print(f"[m6] 生成表格 → {outdir}", flush=True)
    table1_audit(cells_dir, cfg, outdir)
    table2_numbers(results, outdir)
    qc_table(cfg, args.subject, cells_dir, results, outdir)
    print(f"[m6] 完成。表格在 {outdir}", flush=True)


if __name__ == "__main__":
    main()
