"""
Figure 18 —— 表示退化诊断图（§2.2 lag-1 + §2.3 OOD）。纯出图，零算力。

## 为什么必须有这张图

Figure 17 只画基于 r 的结果（D_m > 0 三被试显著）。**只看那张图，读者会得出
"控制通过了、Context Gain 确实依赖语言结构"——而这个结论是错的**，纠正它的证据
（打乱诱发 H 依赖的表示塌缩）全部不在 Figure 17 里。

因此这张图不是锦上添花，是防止 Figure 17 被误读的必要配套。

四面板：
  A  有效维度（participation ratio）随 H 的走向 —— **核心：符号反转**
     真实上下文越长表示越丰富；打乱上下文越长表示越塌缩。
  B  lag-1 余弦随 H 的走向 —— 打乱后表示更"呆"（与规范预期方向相反）
  C  有效维度 DiD（ctx1 − normal 的 H 依赖变化）+ 95% CI —— 三模型全部显著
  D  秩序一致性：PR 塌缩幅度 vs D_m —— 三个模型排序相同 = I_MP 被混淆的直接理由

风格与 Figure 17 / 主实验 M6 一致（同配色、PNG 150dpi + PDF 矢量）。

用法（需 Step5/lag1/ood 三份结果都在）：
  python 实验补充/scripts/m6s_degradation_figure.py
输出：实验补充/results/figures/figure18_representation_degradation.{png,pdf}
      实验补充/results/tables/table10_degradation_diagnostics.{csv,md}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.lines import Line2D          # noqa: E402
import numpy as np                           # noqa: E402
import pandas as pd                          # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SUPPLEMENT_ROOT = SCRIPT_DIR.parent
RESULTS = SUPPLEMENT_ROOT / "results"

SUBJECTS = ["UTS01", "UTS02", "UTS03"]
MODELS = ["pythia", "mamba", "rwkv"]
COLOR = {"pythia": "#1f77b4", "mamba": "#d62728", "rwkv": "#2ca02c"}
LABEL = {"pythia": "Pythia", "mamba": "Mamba", "rwkv": "RWKV"}
HS = [8, 128]


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def load_all():
    ood = json.load(open(RESULTS / "ood_diagnostic.json"))
    lag1 = json.load(open(RESULTS / "lag1_diagnostic.json"))
    dm = {}
    for s in SUBJECTS:
        p = RESULTS / "m5s_stats" / s / "m5s_results.json"
        if p.exists():
            dm[s] = json.load(open(p))["estimands"]
    return ood, lag1, dm


def panel_metric_lines(ax, get_value, title, subtitle, ylabel):
    """A/B 通用：每模型两条线（normal 实线 / C1 虚线），横轴为上下文长度 H。"""
    x = [0, 1]
    for m in MODELS:
        n = [get_value(m, "normal", h) for h in HS]
        c = [get_value(m, "ctx1", h) for h in HS]
        if any(v is None for v in n + c):
            continue
        ax.plot(x, n, color=COLOR[m], linestyle="-", marker="o",
                linewidth=2.0, markersize=7)
        ax.plot(x, c, color=COLOR[m], linestyle="--", marker="s",
                linewidth=2.0, markersize=7, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in HS])
    ax.set_xlabel("Context Length H (raw words)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    # 主标题只说测了什么；副标题只说指标怎么读，都不含结果判断
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.margins(x=0.12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-stem", default="figure18_representation_degradation")
    args = ap.parse_args()

    ood, lag1, dm = load_all()

    def ood_val(model, cond, H, metric="participation_ratio"):
        try:
            return ood["by_model"][model]["metrics"][metric]["absolute_medians"][f"{cond}_H{H}"]
        except KeyError:
            return None

    def lag1_val(model, cond, H):
        blk = lag1["by_model_H"].get(f"{model}_H{H}")
        if blk is None:
            return None
        return (blk["normal_tolerance_band"]["median"] if cond == "normal"
                else blk["ctx1_median"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    axA, axB, axC, axD = axes.ravel()

    # ── Panel A：有效维度 ──────────────────────────────────────────────
    panel_metric_lines(
        axA, lambda m, c, h: ood_val(m, c, h, "participation_ratio"),
        "A. Effective dimensionality",
        "Higher participation ratio indicates more distributed representations.",
        "Participation ratio (median over stories)")
    axA.legend(handles=[
        Line2D([0], [0], color="k", linestyle="-", marker="o", label="normal"),
        Line2D([0], [0], color="k", linestyle="--", marker="s", label="C1 shuffled context"),
    ] + [Line2D([0], [0], color=COLOR[m], lw=3, label=LABEL[m]) for m in MODELS],
        fontsize=7.5, loc="upper left", ncol=2)

    # ── Panel B：lag-1 余弦 ────────────────────────────────────────────
    panel_metric_lines(
        axB, lag1_val,
        "B. Temporal smoothness",
        "Higher lag-1 cosine indicates slower-changing features.",
        "Lag-1 cosine similarity between adjacent TRs")

    # ── Panel C：有效维度 DiD + 95% CI ─────────────────────────────────
    xs = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        d = ood["by_model"][m]["metrics"]["participation_ratio"]["ood_h_dependence_did"]
        axC.bar(i, d["median"], 0.55, color=COLOR[m], alpha=0.9,
                edgecolor="k", linewidth=0.5)
        axC.plot([i, i], [d["p2_5"], d["p97_5"]], color="k", linewidth=1.3)
        # 数值标注：RWKV 的柱极小(−0.03)但同样显著，不标会被误看成"无数据"
        star = "*" if d["ci_excludes_zero"] else ""
        axC.annotate(f"{d['median']:+.3f}{star}", (i, d["p2_5"]),
                     textcoords="offset points", xytext=(0, -13),
                     ha="center", fontsize=8.5)
    axC.axhline(0, color="k", linewidth=0.8, linestyle=":")
    axC.margins(y=0.18)
    axC.set_xticks(xs)
    axC.set_xticklabels([LABEL[m] for m in MODELS])
    axC.set_xlabel("Model", fontsize=9)
    axC.set_ylabel(r"PR interaction: $\Delta PR_{\mathrm{C1}} - \Delta PR_{\mathrm{normal}}$"
                   " (95% CI)", fontsize=9)
    axC.set_title("C. H-dependent shift in effective dimensionality\n"
                  r"$\Delta PR = PR_{H=128} - PR_{H=8}$", fontsize=10)
    axC.grid(True, axis="y", alpha=0.3)

    # ── Panel D：秩序一致性（PR 塌缩 vs D_m）──────────────────────────
    pr_did, dm_mean, dm_err = [], [], []
    for m in MODELS:
        pr_did.append(
            ood["by_model"][m]["metrics"]["participation_ratio"]["ood_h_dependence_did"]["median"])
        vals = [dm[s][f"D_{m}_ifg"]["point"] for s in dm if f"D_{m}_ifg" in dm[s]]
        dm_mean.append(float(np.mean(vals)) if vals else np.nan)
        dm_err.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
    for i, m in enumerate(MODELS):
        axD.errorbar(pr_did[i], dm_mean[i], yerr=dm_err[i], fmt="o",
                     color=COLOR[m], markersize=11, capsize=4, linewidth=1.5)
        axD.annotate(LABEL[m], (pr_did[i], dm_mean[i]),
                     textcoords="offset points", xytext=(9, 6), fontsize=9)
    axD.axhline(0, color="k", linewidth=0.8, linestyle=":")
    # 与 Panel C 保持同一符号约定（原始 DiD，与 Table 10 及正文数字一致），
    # 只在轴标签里说明方向，避免同一量在一张图里出现两种符号。
    axD.set_xlabel(r"PR interaction, $\Delta PR_{\mathrm{C1}} - \Delta PR_{\mathrm{normal}}$"
                   "  (more negative = larger dimensionality shift)", fontsize=9)
    axD.set_ylabel(r"Mean context-gain reduction, $D_m$ (mean ± SD over participants)",
                   fontsize=9)
    axD.set_title("D. Descriptive relation between dimensionality shift\nand gain reduction",
                  fontsize=10)
    axD.grid(True, alpha=0.3)
    # 只有 3 个模型 → 单调关系有 1/6 概率纯属偶然。必须标明是描述性比较、
    # 非统计检验，否则读者会把它当成"已证明混淆"的证据。
    # 放左下空白区：左上会盖住 Pythia 的点标签（实测渲染后发现并修正）
    axD.text(0.03, 0.05, "n = 3 models; descriptive comparison only; no statistical test",
             transform=axD.transAxes, fontsize=7.5, style="italic", color="#444444",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#cccccc", alpha=0.9))

    # 主标题只说"测了什么"，不含 degradation 这类结论词；结论留给图注与正文。
    fig.suptitle(
        "Figure 18. Representation-Statistic Diagnostics under "
        "Stimulus-Side Context Perturbation",
        fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{args.out_stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[m6s] 已保存 {args.out_stem}.png / .pdf", flush=True)

    # ── 配套 Table 10：三指标的绝对值与 DiD ────────────────────────────
    rows = []
    for m in MODELS:
        met = ood["by_model"][m]["metrics"]
        row = {"model": LABEL[m]}
        for metric, short in [("participation_ratio", "PR"),
                              ("l2_norm_median", "L2"),
                              ("evr_at_100", "evr@100")]:
            a = met[metric]["absolute_medians"]
            d = met[metric]["ood_h_dependence_did"]
            row[f"{short} normal H8→H128"] = f"{a['normal_H8']:.4g} → {a['normal_H128']:.4g}"
            row[f"{short} ctx1 H8→H128"] = f"{a['ctx1_H8']:.4g} → {a['ctx1_H128']:.4g}"
            row[f"{short} DiD [95% CI]"] = (
                f"{d['median']:+.4g} [{d['p2_5']:+.4g}, {d['p97_5']:+.4g}]"
                + ("*" if d["ci_excludes_zero"] else ""))
        for H in HS:
            b = lag1["by_model_H"][f"{m}_H{H}"]
            row[f"lag-1 H{H} (normal→ctx1)"] = (
                f"{b['normal_tolerance_band']['median']:.4f} → {b['ctx1_median']:.4f}")
        row["verdict"] = ood["by_model"][m]["verdict"]["conclusion"]
        rows.append(row)
    df = pd.DataFrame(rows)
    tab_dir = RESULTS / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tab_dir / "table10_degradation_diagnostics.csv", index=False)
    with open(tab_dir / "table10_degradation_diagnostics.md", "w") as f:
        f.write(_md(df))
    print("[m6s] 已写 table10_degradation_diagnostics.csv / .md", flush=True)
    print("\n[m6s] 论文对接：Figure 18 与 Table 10 追加在 Figure 17 / Table 9 之后；"
          "Results 4.6.2 须先报 Figure 17 的 r 结果，紧接着用 Figure 18 说明该结果"
          "不能归因于语言结构。", flush=True)


if __name__ == "__main__":
    main()
