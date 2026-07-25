"""
Step 6 —— 刺激侧上下文控制的论文图表：Table 9 + Figure 17。

⚠️ 纯出图/出表，零算力：只读 Step 5 的 m5s_results.json（每被试一份），不重算
任何统计。本地或服务器均可运行。

对应 V6.4.2 §6（执行版校正后）：
  - 论文现用 表 1–8 / 图 1–16 → 新增 **Table 9**（追加）、**Figure 17**（追加）。
  - Table 9 正文只放核心诊断量（D_pythia, D_mamba, I_MP），完整量入 supplement。
  - Figure 17 四面板，全部标注 diagnostic / uncorrected / non-confirmatory /
    story-paired bootstrap，不进确认性家族。

风格与主实验 M6 对齐：模型配色同 src/viz/m6_data.py（pythia 蓝 / mamba 红 /
rwkv 绿），CI 文本格式同 scripts/m6_tables.py 的 "+0.1064 [+0.1010, +0.1116]"，
PNG(150dpi 速览)+PDF(矢量) 双输出。

用法（Step 5 三被试都跑完后）：
  python 实验补充/scripts/m6s_context_control_figure.py
输出：
  实验补充/results/tables/table9_context_control_core.{csv,md}
  实验补充/results/tables/table9_context_control_supplement.{csv,md}
  实验补充/results/figures/figure17_context_control.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.patches import Patch               # noqa: E402
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SUPPLEMENT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SUPPLEMENT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config           # noqa: E402

SUBJECTS = ["UTS01", "UTS02", "UTS03"]
CORE_MODELS = ["pythia", "mamba", "rwkv"]
MODEL_COLOR = {"pythia": "#1f77b4", "mamba": "#d62728", "rwkv": "#2ca02c"}
MODEL_LABEL = {"pythia": "Pythia", "mamba": "Mamba", "rwkv": "RWKV"}
DRT = r"$\Delta r_{\mathrm{total}}$"


# ── 表格助手（自包含，复刻 scripts/m6_tables.py 的格式，不引入 viz 依赖）──────

def _ci_str(e: dict | None) -> str:
    if e is None or e.get("point") is None or not np.isfinite(e.get("point", np.nan)):
        return "—"
    return f"{e['point']:+.4f} [{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}]"


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _write_table(df: pd.DataFrame, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    df = df.fillna("—")
    df.to_csv(outdir / f"{name}.csv", index=False)
    with open(outdir / f"{name}.md", "w") as f:
        f.write(_df_to_markdown(df))
    print(f"[m6s] 已写 {name}.csv / .md", flush=True)


# ── 数据加载 ──────────────────────────────────────────────────────────────

def load_all(results_dir: Path, m5s_name: str) -> dict[str, dict]:
    """{subject: estimands dict}，只收真实存在的被试。"""
    out = {}
    for subj in SUBJECTS:
        p = results_dir / m5s_name / subj / "m5s_results.json"
        if p.exists():
            out[subj] = json.load(open(p))["estimands"]
        else:
            print(f"[m6s] 跳过 {subj}：未找到 {p}", flush=True)
    if not out:
        raise SystemExit("没有任何被试的 m5s_results.json，先跑 Step 5")
    return out


def _e(est: dict, name: str) -> dict | None:
    return est.get(name)


# ── Table 9 ───────────────────────────────────────────────────────────────

def build_table9(all_est: dict[str, dict], outdir: Path):
    core_rows, supp_rows = [], []
    for subj, est in all_est.items():
        core_rows.append({
            "subject": subj,
            "D_pythia": _ci_str(_e(est, "D_pythia_ifg")),
            "D_mamba": _ci_str(_e(est, "D_mamba_ifg")),
            "I_MP (Mamba−Pythia DiD)": _ci_str(_e(est, "I_MP_ifg")),
            "I_MP CI width": f"{_e(est, 'I_MP_ifg')['ci_width']:.4f}" if _e(est, "I_MP_ifg") else "—",
            "I_MP boot SD": f"{_e(est, 'I_MP_ifg')['bootstrap_sd']:.4f}" if _e(est, "I_MP_ifg") else "—",
            "type": "diagnostic, uncorrected",
        })
        supp_rows.append({
            "subject": subj,
            "D_rwkv": _ci_str(_e(est, "D_rwkv_ifg")),
            "I_RP": _ci_str(_e(est, "I_RP_ifg")),
            "Δr_total^ctx1 pythia": _ci_str(_e(est, "ctx1_delta_total_pythia_ifg")),
            "Δr_total^ctx1 mamba": _ci_str(_e(est, "ctx1_delta_total_mamba_ifg")),
            "Δr_total^ctx1 rwkv": _ci_str(_e(est, "ctx1_delta_total_rwkv_ifg")),
            "A_MP normal": _ci_str(_e(est, "A_MP_normal_ifg")),
            "A_MP ctx1": _ci_str(_e(est, "A_MP_ctx1_ifg")),
        })
    _write_table(pd.DataFrame(core_rows), outdir, "table9_context_control_core")
    _write_table(pd.DataFrame(supp_rows), outdir, "table9_context_control_supplement")


# ── Figure 17 ─────────────────────────────────────────────────────────────

def _bar_ci(ax, x, e, color, width, hatch=None, alpha=0.9, label=None):
    """一根带 95% CI 的柱；e 为 estimand dict（含 point/ci_lo/ci_hi）。"""
    if e is None or not np.isfinite(e.get("point", np.nan)):
        return
    ax.bar(x, e["point"], width, color=color, alpha=alpha, hatch=hatch,
           edgecolor="k", linewidth=0.5, label=label)
    ax.plot([x, x], [e["ci_lo"], e["ci_hi"]], color="k", linewidth=1.0)


def build_figure17(all_est: dict[str, dict], outdir: Path):
    subjects = list(all_est.keys())
    nsub = len(subjects)
    x = np.arange(nsub)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axA, axB, axC, axD = axes.ravel()

    # Panel A：Δr_total，normal(实心) vs ctx1(斜纹)，按模型分组
    wa = 0.13
    for mi, m in enumerate(CORE_MODELS):
        base = (mi - 1) * (2.2 * wa)
        for si, subj in enumerate(subjects):
            est = all_est[subj]
            _bar_ci(axA, x[si] + base - wa/2, _e(est, f"normal_delta_total_{m}_ifg"),
                    MODEL_COLOR[m], wa, alpha=0.9,
                    label=f"{MODEL_LABEL[m]} normal" if si == 0 else None)
            _bar_ci(axA, x[si] + base + wa/2, _e(est, f"ctx1_delta_total_{m}_ifg"),
                    MODEL_COLOR[m], wa, hatch="////", alpha=0.55,
                    label=f"{MODEL_LABEL[m]} ctx1" if si == 0 else None)
    axA.axhline(0, color="k", linewidth=0.8, linestyle=":")
    axA.set_title("A. Context Gain " + DRT + " : normal vs shuffled-context (ctx1)", fontsize=10)
    axA.set_ylabel(DRT + " (95% CI)")
    axA.set_xticks(x); axA.set_xticklabels(subjects)
    axA.legend(fontsize=6.5, ncol=3, loc="upper right")
    axA.grid(True, axis="y", alpha=0.3)

    # Panel B：D_m = normal − ctx1 gain，按模型分组
    wb = 0.22
    for mi, m in enumerate(CORE_MODELS):
        for si, subj in enumerate(subjects):
            _bar_ci(axB, x[si] + (mi - 1) * wb, _e(all_est[subj], f"D_{m}_ifg"),
                    MODEL_COLOR[m], wb, alpha=0.9,
                    label=MODEL_LABEL[m] if si == 0 else None)
    axB.axhline(0, color="k", linewidth=0.8, linestyle=":")
    axB.set_title(r"B. $D_m = \Delta r_{\mathrm{total}}^{\mathrm{normal}} - "
                  r"\Delta r_{\mathrm{total}}^{\mathrm{ctx1}}$  (gain lost when context shuffled)",
                  fontsize=10)
    axB.set_ylabel(r"$D_m$ (95% CI)")
    axB.set_xticks(x); axB.set_xticklabels(subjects)
    axB.legend(fontsize=7); axB.grid(True, axis="y", alpha=0.3)

    # Panel C：核心交互 I_MP（+ I_RP 若有），按被试
    wc = 0.28
    for si, subj in enumerate(subjects):
        est = all_est[subj]
        _bar_ci(axC, x[si] - wc/2, _e(est, "I_MP_ifg"), "#6a3d9a", wc, alpha=0.9,
                label="I_MP (Mamba−Pythia)" if si == 0 else None)
        _bar_ci(axC, x[si] + wc/2, _e(est, "I_RP_ifg"), "#ff7f00", wc, alpha=0.7,
                label="I_RP (RWKV−Pythia)" if si == 0 else None)
    axC.axhline(0, color="k", linewidth=0.8, linestyle=":")
    axC.set_title("C. Difference-in-differences interaction (three-level nested; low power)",
                  fontsize=10)
    axC.set_ylabel("interaction (95% CI)")
    axC.set_xticks(x); axC.set_xticklabels(subjects)
    axC.legend(fontsize=7); axC.grid(True, axis="y", alpha=0.3)

    # Panel D：Mamba−Pythia 架构优势 A_MP，normal vs ctx1
    wd = 0.32
    for si, subj in enumerate(subjects):
        est = all_est[subj]
        _bar_ci(axD, x[si] - wd/2, _e(est, "A_MP_normal_ifg"), "#d62728", wd, alpha=0.9,
                label="A_MP normal" if si == 0 else None)
        _bar_ci(axD, x[si] + wd/2, _e(est, "A_MP_ctx1_ifg"), "#d62728", wd, hatch="////",
                alpha=0.5, label="A_MP ctx1" if si == 0 else None)
    axD.axhline(0, color="k", linewidth=0.8, linestyle=":")
    axD.set_title("D. Mamba−Pythia advantage: normal vs shuffled context", fontsize=10)
    axD.set_ylabel(r"$A_{MP}$ (95% CI)")
    axD.set_xticks(x); axD.set_xticklabels(subjects)
    axD.legend(fontsize=7); axD.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Figure 17. Stimulus-side context control (C1: same-story shuffled context)\n"
                 "diagnostic · uncorrected · story-paired bootstrap · NOT in the confirmatory family",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"figure17_context_control.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[m6s] 已保存 figure17_context_control.png / .pdf", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m5s-name", default="m5s_stats")
    args = ap.parse_args()

    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    all_est = load_all(results_dir, args.m5s_name)
    print(f"[m6s] 载入 {len(all_est)} 个被试：{list(all_est.keys())}", flush=True)

    tables_dir = SUPPLEMENT_ROOT / "results" / "tables"
    figures_dir = SUPPLEMENT_ROOT / "results" / "figures"
    build_table9(all_est, tables_dir)
    build_figure17(all_est, figures_dir)

    print("\n[m6s] Step 6 完成。Table 9（正文 core + supplement）与 Figure 17 已生成。", flush=True)
    print("[m6s] 论文对接：新表=Table 9、新图=Figure 17（追加，不改动现有 1–8/1–16 编号）；"
          "§4.6 拆『控制分析』下设 4.6.1 时间平移 / 4.6.2 刺激侧上下文控制。", flush=True)


if __name__ == "__main__":
    main()
