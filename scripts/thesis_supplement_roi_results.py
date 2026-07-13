"""
硕士论文核查清单 —— 第5+7节：roi_results_full.csv（IFG 主层完整结果 + PT 完整结果）。

数据来源：不重算——figures/<subject>/tables/table2_full_numbers.csv 已经是
scripts/m6_tables.py 在服务器用真实 m5_results.json（1000 次 bootstrap，Fisher-z
ROI 聚合）算好、已 rsync 到本地的正式产物。这里只做两件事：①把"点估计 [CI]"的
展示字符串拆回结构化的数值列；②把三名被试的三份文件合并成一张跨被试长表，补上
delta_local/long/total 各自的 95% CI（table2 已有，只是要对齐清单要求的列名）。

用法：python3 scripts/thesis_supplement_roi_results.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config   # noqa: E402

OUT_DIR = PROJECT_ROOT / "thesis_supplement"
SUBJECTS = ("UTS01", "UTS02", "UTS03")

_CI_RE = re.compile(r"^([+-]?\d+\.\d+)\s*\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]$")


def _parse_ci(cell: str):
    """把 table2 里 '+0.1064 [+0.1010, +0.1116]' 这样的展示字符串拆成 (point, lo, hi)。
    缺项（非核心模型的 RQ1 列）显示为 '—'，原样返回 (None, None, None)。"""
    if not isinstance(cell, str) or cell.strip() == "—":
        return None, None, None
    m = _CI_RE.match(cell.strip().strip('"'))
    if not m:
        raise ValueError(f"无法解析的 CI 字符串：{cell!r}")
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def main():
    cfg = load_config()
    figs = Path(cfg["paths"]["figures_dir"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for subj in SUBJECTS:
        p = figs / subj / "tables" / "table2_full_numbers.csv"
        if not p.exists():
            print(f"  [跳过] {p} 不存在（该被试还没同步 table2 到本地）")
            continue
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            row = {"subject": subj, "model": r["model"], "roi": r["roi"]}
            for H in (8, 32, 128):
                pt, lo, hi = _parse_ci(r[f"r_H{H}"])
                row[f"r{H}"] = pt
                row[f"r{H}_ci_lo"] = lo
                row[f"r{H}_ci_hi"] = hi
            for kind in ("local", "long", "total"):
                pt, lo, hi = _parse_ci(r[f"delta_{kind}"])
                row[f"delta_{kind}"] = pt
                row[f"delta_{kind}_ci_lo"] = lo
                row[f"delta_{kind}_ci_hi"] = hi
            all_rows.append(row)

    out = pd.DataFrame(all_rows)
    out = out.sort_values(["subject", "roi", "model"]).reset_index(drop=True)
    out.to_csv(OUT_DIR / "roi_results_full.csv", index=False)
    print(f"已写 {OUT_DIR / 'roi_results_full.csv'}（{len(out)} 行 = "
         f"{len(SUBJECTS)}被试 × 4模型 × 2ROI）")

    n_expected = len(SUBJECTS) * 4 * 2
    if len(out) != n_expected:
        print(f"  [警告] 行数 {len(out)} != 预期 {n_expected}，检查上面是否有被试被跳过")

    # 与 table2 数值回填核对（浮点误差应为 0——同一份数字，只是格式变了）
    check = out[(out.subject == "UTS03") & (out.model == "pythia")
               & (out.roi == "left_IFG")]
    print("\n核对样例（UTS03 pythia left_IFG）：")
    print(check[["r8", "r32", "r128", "delta_total", "delta_total_ci_lo",
                "delta_total_ci_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
