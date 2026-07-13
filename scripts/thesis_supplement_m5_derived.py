"""
硕士论文核查清单 —— 第6/8/9节（依赖 m5_results.json 的三张表）。

**只读聚合，不重算任何统计**：直接从 scripts/m5_analysis.py 已经跑好、写入
results/m5_stats/<subject>/m5_results.json 的正式产物里取数（confirmatory/
estimands/layer_flip/shifted_diagnostic 字段），跟 scripts/m6_tables.py 是
同一类"只读表格生成"脚本。

前置条件：results/m5_stats/<subject>/m5_results.json 必须先从服务器同步到本机
同一相对路径（该文件当前只在服务器本地生成，是 M5 里程碑的正式产物，本仓库
results/ 整体 gitignore，需要单独 scp/rsync 一次，例如：
  rsync -avz -e "ssh -p <端口>" root@<主机>:~/autodl-tmp/deep-fMRI-dataset/results/m5_stats/ \
        results/m5_stats/
三名被试各自独立跑过 M5，三份 json 互不覆盖。

输出：
  第6节 → thesis_supplement/contrasts_bootstrap.csv
  第8节 → thesis_supplement/final_layer_robustness.csv
  第9节 → thesis_supplement/shift_results.csv（r 部分；mask 部分见
          thesis_supplement_mask_audit.py，那部分本机可独立算，不依赖本文件）

用法：python3 scripts/thesis_supplement_m5_derived.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config   # noqa: E402

OUT_DIR = PROJECT_ROOT / "thesis_supplement"
SUBJECTS = ("UTS01", "UTS02", "UTS03")
MODELS = ("pythia", "mamba", "rwkv", "awd_lstm")
CORE_VS_PYTHIA = ("rwkv", "mamba")


def _e(est: dict, name: str):
    v = est.get(name)
    if v is None:
        return None, None, None
    return v["point"], v["ci_lo"], v["ci_hi"]


def build_contrasts(all_results: dict) -> pd.DataFrame:
    """第6节：两项确认性架构差值，每被试独立 Holm（frozen 口径：
    confirmatory_family = per_subject，不跨被试合并多重比较）。"""
    rows = []
    for subj, res in all_results.items():
        for name, e in res["confirmatory"].items():
            rows.append({
                "subject": subj, "contrast": name,
                "point": e["point"], "ci_lo": e["ci_lo"], "ci_hi": e["ci_hi"],
                "p_raw_bootstrap": e["p"],
                "holm_threshold": e["holm_threshold"],
                "reject_h0": e["reject"],
            })
    return pd.DataFrame(rows)


def build_final_layer(all_results: dict) -> pd.DataFrame:
    """第8节：最终层 r8/r32/r128、delta_total、两项架构差值+层位翻转判定。"""
    rows = []
    for subj, res in all_results.items():
        est = res["estimands"]
        for m in MODELS:
            row = {"subject": subj, "model": m}
            for H in (8, 32, 128):
                pt, lo, hi = _e(est, f"r_{m}_H{H}_left_IFG_final_normal")
                row[f"r{H}_final"] = pt
                row[f"r{H}_final_ci_lo"] = lo
                row[f"r{H}_final_ci_hi"] = hi
            pt, lo, hi = _e(est, f"delta_total_{m}_ifg_final")
            row["delta_total_final"] = pt
            row["delta_total_final_ci_lo"] = lo
            row["delta_total_final_ci_hi"] = hi
            rows.append(row)
        for arch in CORE_VS_PYTHIA:
            pt, lo, hi = _e(est, f"{arch}_minus_pythia_delta_total_ifg_final")
            fl = res["layer_flip"][f"{arch}_minus_pythia_delta_total"]
            rows.append({
                "subject": subj, "model": f"{arch}_minus_pythia (架构差值，非单模型)",
                "delta_total_final": pt, "delta_total_final_ci_lo": lo,
                "delta_total_final_ci_hi": hi,
                "main_layer_ci_side": fl["main_ci_side"],
                "final_layer_ci_side": fl["final_ci_side"],
                "substantive_flip": fl["substantive_flip"],
            })
    return pd.DataFrame(rows)


def build_shift(all_results: dict) -> pd.DataFrame:
    """第9节 r 部分：shift 条件 r8/r32/r128/delta_total（每模型）+ shift 架构差值
    + 负控制关键诊断量（normal−shift 配对差值，判断该模型自身 gain 是否被显著削弱，
    与架构差值收缩是不同层面的问题，见 estimands.py 里的说明）。"""
    rows = []
    for subj, res in all_results.items():
        est = res["estimands"]
        for m in MODELS:
            row = {"subject": subj, "model": m}
            for H in (8, 32, 128):
                pt, lo, hi = _e(est, f"r_{m}_H{H}_left_IFG_main_shift")
                row[f"shift_r{H}"] = pt
                row[f"shift_r{H}_ci_lo"] = lo
                row[f"shift_r{H}_ci_hi"] = hi
            pt, lo, hi = _e(est, f"shifted_delta_total_{m}_ifg_main")
            row["shift_delta_total"] = pt
            row["shift_delta_total_ci_lo"] = lo
            row["shift_delta_total_ci_hi"] = hi
            pt, lo, hi = _e(est, f"delta_total_normal_minus_shift_{m}_ifg_main")
            row["gain_reduction_normal_minus_shift"] = pt
            row["gain_reduction_ci_lo"] = lo
            row["gain_reduction_ci_hi"] = hi
            row["gain_significantly_reduced"] = (
                res["shifted_diagnostic"]["per_model_gain_reduction_normal_minus_shift"]
                [m]["shift_significantly_reduced_gain"])
            rows.append(row)
        for arch in CORE_VS_PYTHIA:
            pt, lo, hi = _e(est, f"shifted_{arch}_minus_pythia_delta_total_ifg_main")
            rows.append({
                "subject": subj,
                "model": f"{arch}_minus_pythia (shift架构差值，非单模型)",
                "shift_delta_total": pt, "shift_delta_total_ci_lo": lo,
                "shift_delta_total_ci_hi": hi,
            })
        rows.append({
            "subject": subj,
            "model": "__diagnostic__",
            "gain_significantly_reduced": (
                f"{res['shifted_diagnostic']['n_models_with_gain_significantly_reduced']}/4 模型"),
            "gain_reduction_normal_minus_shift":
                f"架构差值平移后仍排除0(层面①受威胁)="
                f"{res['shifted_diagnostic']['shifted_reproduces_architecture_effect']}",
        })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for subj in SUBJECTS:
        p = results_dir / "m5_stats" / subj / "m5_results.json"
        if not p.exists():
            print(f"  [跳过] {p} 不存在——需要先从服务器同步该文件，见脚本顶部说明")
            continue
        all_results[subj] = json.load(open(p))

    if not all_results:
        print("\n没有任何 m5_results.json 可用，三张表全部无法生成。先同步文件再跑本脚本。")
        return

    build_contrasts(all_results).to_csv(OUT_DIR / "contrasts_bootstrap.csv", index=False)
    print(f"已写 {OUT_DIR / 'contrasts_bootstrap.csv'}（{len(all_results)}被试 × 2项确认性对比）")

    build_final_layer(all_results).to_csv(OUT_DIR / "final_layer_robustness.csv", index=False)
    print(f"已写 {OUT_DIR / 'final_layer_robustness.csv'}")

    build_shift(all_results).to_csv(OUT_DIR / "shift_results.csv", index=False)
    print(f"已写 {OUT_DIR / 'shift_results.csv'}")

    missing = set(SUBJECTS) - set(all_results)
    if missing:
        print(f"\n[警告] 缺 {sorted(missing)} 的 m5_results.json，以上三份表只含"
             f"{sorted(all_results)}，不是完整三被试版本。")


if __name__ == "__main__":
    main()
