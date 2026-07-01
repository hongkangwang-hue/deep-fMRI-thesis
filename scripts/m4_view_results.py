"""
M4 —— 结果查看。把 results/<out-name>/<subject>/cells/ 下逐单元的 json 汇总成
可读表格打印到终端（不重新计算任何东西，纯读文件）。

用法：
  python3 scripts/m4_view_results.py                 # 打印验收状态+全表+model×H均值
  python3 scripts/m4_view_results.py --csv out.csv   # 顺便存一份 csv 方便本地下载分析
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config  # noqa: E402


def load_rows(cells_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(cells_dir.glob("main_*.json")):
        c = json.load(open(p))
        for cond in ("normal", "shift"):
            if cond not in c:
                continue
            r = c[cond]["roi_r"]
            rows.append({
                "layer": "main", "model": c["model"], "H": c["H"], "fold": c["fold"],
                "condition": cond, "left_IFG": r.get("left_IFG"),
                "bilateral_PT": r.get("bilateral_PT"),
                "voxel_r_mean": c[cond]["voxel_r_mean"], "n_eff_tr": c[cond]["n_eff_tr"],
            })
    for p in sorted(cells_dir.glob("final_*.json")):
        c = json.load(open(p))
        r = c["normal"]["roi_r"]
        rows.append({
            "layer": "final", "model": c["model"], "H": c["H"], "fold": c["fold"],
            "condition": "normal", "left_IFG": r.get("left_IFG"), "bilateral_PT": None,
            "voxel_r_mean": c["normal"]["voxel_r_mean"], "n_eff_tr": c["normal"]["n_eff_tr"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--out-name", default="m4_full_matrix")
    ap.add_argument("--csv", default=None, help="可选：把全表存成 csv")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(cfg["paths"]["results_dir"]) / args.out_name / args.subject
    cells_dir = out_dir / "cells"

    manifest_path = out_dir / "m4_manifest.json"
    if manifest_path.exists():
        m = json.load(open(manifest_path))
        print(f"[m4:view] 验收: {json.dumps(m['verdict'], ensure_ascii=False)}")
        print(f"[m4:view] 全部通过: {'✅' if m['verdict_all_pass'] else '⚠️'}")
        if m.get("timing"):
            print(f"[m4:view] 计时: {m['timing']}")
        print()
    else:
        print(f"[m4:view] 未找到 {manifest_path}，先跑 scripts/m4_aggregate.py 生成。\n")

    if not cells_dir.exists():
        print(f"[m4:view] {cells_dir} 不存在。")
        return
    rows = load_rows(cells_dir)
    if not rows:
        print(f"[m4:view] {cells_dir} 下没有找到结果文件。")
        return

    df = pd.DataFrame(rows).sort_values(["layer", "model", "H", "fold", "condition"])
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False))

    main_normal = df[(df.layer == "main") & (df.condition == "normal")]
    if not main_normal.empty:
        print("\n[m4:view] === 主层正常条件：model x H 跨fold平均 ===")
        pivot = main_normal.groupby(["model", "H"])[["left_IFG", "bilateral_PT"]].mean()
        print(pivot.to_string())

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n[m4:view] 已存 → {args.csv}")


if __name__ == "__main__":
    main()
