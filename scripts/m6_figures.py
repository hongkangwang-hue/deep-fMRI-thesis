"""
M6 —— 论文图表生成（Figure 1–5）。只读 M5 结果 (m5_results.json)，不重算任何统计
（冻结文档里程碑6"明确不做"）。

Figure 1  流程与泄漏控制示意（统一索引→状态重置→training-only 变换→story-level CV→shift）
Figure 2  IFG/PT 中四模型 r8/r32/r128 曲线 + 95% CI            —— RQ 主结果
Figure 3  RQ1 H-specific 架构差值 + 三类 Δr + 确认性 Δr_total  —— **主图**
Figure 4  正常 vs 40s shifted 的 r 与 Context Gain             —— 负控制
Figure 5  IFG 主层 vs 最终层同一预注册对比                     —— 层位稳健性

红线（里程碑6）：AWD-LSTM 灰色虚线视觉隔离，图注注明不可比、不参与核心排名；确认性
（仅 2 项 Δr_total 架构差值）与探索性严格分开标注。

用法（服务器或本地，需 m5_results.json 可达）：
  python3 scripts/m6_figures.py                       # 生成全部 5 图
  python3 scripts/m6_figures.py --only fig3           # 只重画某图
输出 PNG(150dpi 速览) + PDF(矢量, 论文用) 到 figures/。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # 服务器无显示，非交互后端
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config    # noqa: E402
from src.viz.m6_data import (                # noqa: E402
    load_results, r_curve, context_gain, rq1, arch_delta_total, gain_reduction,
    confirmatory_rows, ci_excludes_zero,
    CORE_MODELS, REFERENCE_MODEL, MODEL_STYLE, HS, CORE_VS_PYTHIA,
)

ALL_MODELS = CORE_MODELS + [REFERENCE_MODEL]
HPOS = list(range(len(HS)))                  # 均匀间距的 H 轴位置（8/32/128 三点，非真数轴）


def _save(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[m6] 已保存 {name}.png / .pdf", flush=True)


def _errbar(ax, x, pt, lo, hi, style, label=None, dx=0.0):
    """在 x+dx 处画点 + 非对称 CI 误差棒；缺项跳过。"""
    if pt is None:
        return
    yerr = [[pt - lo], [hi - pt]] if (lo is not None and hi is not None) else None
    ax.errorbar(x + dx, pt, yerr=yerr, fmt=style.get("marker", "o"),
                color=style["color"], ecolor=style["color"], capsize=3,
                markersize=6, label=label, linewidth=1.5)


# ---------------------------------------------------------------------------
# Figure 1：流程与泄漏控制示意
# ---------------------------------------------------------------------------

def fig1(results, est, outdir):
    steps = [
        "Unified word index\n(story/word/onset/TR uniquely linked)",
        "Per-word model features\n(state reset per window; H history + target)",
        "Lanczos downsample (word->TR)\n+ within-story FIR (2/4/6/8 s)",
        "Training-only transforms\nStandardScaler + PCA-100 (fit on train fold only)",
        "Per-voxel RidgeCV\n(inner 2-fold, lambda in logspace(-2,7,19))",
        "Story-level scoring\n(per-story voxel r -> Fisher-z -> ROI, >100s common mask)",
        "3-fold story CV summary\n(effective-TR weighted; 40s time-shift control)",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 9.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 2 + 1)
    ax.axis("off")
    y = len(steps) * 2
    for i, s in enumerate(steps):
        box = FancyBboxPatch((1.2, y - 0.8), 7.6, 1.4, boxstyle="round,pad=0.1",
                             fc="#eaf2fb" if i % 2 == 0 else "#f3f0fa",
                             ec="#4a6fa5", linewidth=1.4)
        ax.add_patch(box)
        ax.text(5.0, y - 0.1, s, ha="center", va="center", fontsize=10)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((5.0, y - 0.85), (5.0, y - 1.25),
                         arrowstyle="-|>", mutation_scale=16, color="#4a6fa5"))
        y -= 2
    ax.text(5.0, len(steps) * 2 + 0.4,
            "Figure 1  Encoding pipeline and leakage control", ha="center",
            fontsize=12, fontweight="bold")
    _save(fig, outdir, "fig1_pipeline")


# ---------------------------------------------------------------------------
# Figure 2：IFG/PT 主结果曲线
# ---------------------------------------------------------------------------

def fig2(results, est, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, roi, title in zip(axes, ("left_IFG", "bilateral_PT"),
                              ("Left IFG (primary)", "Bilateral PT (reference)")):
        for m in ALL_MODELS:
            hs, pts, los, his = r_curve(est, m, roi)
            st = MODEL_STYLE[m]
            xs = [HPOS[HS.index(h)] for h in hs]
            ax.plot(xs, pts, color=st["color"], linestyle=st["linestyle"],
                    marker=st["marker"], label=st["label"], linewidth=1.8, markersize=6)
            for x, p, lo, hi in zip(xs, pts, los, his):
                if p is not None and lo is not None:
                    ax.plot([x, x], [lo, hi], color=st["color"], alpha=0.5, linewidth=1.2)
        ax.set_xticks(HPOS)
        ax.set_xticklabels([f"H={h}" for h in HS])
        ax.set_title(title)
        ax.set_xlabel("Context length")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Encoding brain score r (95% CI)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Figure 2  Four-model r8/r32/r128 in IFG/PT "
                 "(AWD-LSTM: gray dashed historical reference, not in core ranking)",
                 fontsize=11, fontweight="bold")
    _save(fig, outdir, "fig2_main_curves")


# ---------------------------------------------------------------------------
# Figure 3：RQ1 + Context Gain + 确认性（主图）
# ---------------------------------------------------------------------------

def fig3(results, est, outdir):
    # 高度加大 + 顶部预留空间，避免 (c) 面板自己的两行标题跟 suptitle 撞在一起
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    fig.subplots_adjust(top=0.78, wspace=0.32)

    # (a) RQ1 H-specific 架构差值 arch−pythia（IFG 主层）
    ax = axes[0]
    ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
    for arch in CORE_VS_PYTHIA:
        st = MODEL_STYLE[arch]
        xs = [HPOS[HS.index(h)] for h in HS]
        pts = [rq1(est, arch, h)[0] for h in HS]
        ax.plot(xs, pts, color=st["color"], marker=st["marker"],
                label=f"{arch}−pythia", linewidth=1.8)
        for h, x in zip(HS, xs):
            p, lo, hi = rq1(est, arch, h)
            if p is not None:
                ax.plot([x, x], [lo, hi], color=st["color"], alpha=0.5, linewidth=1.2)
    ax.set_xticks(HPOS); ax.set_xticklabels([f"H={h}" for h in HS])
    ax.set_title("(a) RQ1: same-H architecture diff (IFG main, exploratory)")
    ax.set_ylabel("r diff vs Pythia (95% CI)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # (b) 三类 Context Gain（每模型，IFG 主层）
    ax = axes[1]
    ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
    kinds = ["local", "long", "total"]
    width = 0.2
    for j, m in enumerate(ALL_MODELS):
        st = MODEL_STYLE[m]
        for k, kind in enumerate(kinds):
            pt, lo, hi = context_gain(est, m, "left_IFG", kind)
            x = k + (j - 1.5) * width
            if pt is not None:
                ax.bar(x, pt, width, color=st["color"],
                       hatch="//" if m == REFERENCE_MODEL else None,
                       alpha=0.85, edgecolor="k", linewidth=0.5,
                       label=st["label"] if k == 0 else None)
                ax.plot([x, x], [lo, hi], color="k", linewidth=1.0)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(["dr_local\n(32-8)", "dr_long\n(128-32)", "dr_total\n(128-8)"])
    ax.set_title("(b) Context Gain (IFG main, descriptive)")
    ax.legend(fontsize=7); ax.grid(True, axis="y", alpha=0.3)

    # (c) 确认性 Δr_total 架构差值（唯一可下确认性结论）
    ax = axes[2]
    ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
    rows = confirmatory_rows(results)
    for i, row in enumerate(rows):
        arch = "rwkv" if "rwkv" in row["name"] else "mamba"
        st = MODEL_STYLE[arch]
        _errbar(ax, i, row["point"], row["ci_lo"], row["ci_hi"], st)
        star = "*" if row.get("reject") else "n.s."
        ax.annotate(f"{star}\np={row['p']:.4f}",
                    (i, row["ci_hi"] if row["ci_hi"] else row["point"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(["RWKV-Pythia", "Mamba-Pythia"])
    ax.set_title("(c) CONFIRMATORY: dr_total arch diff\n(IFG main, Holm alpha=0.05)")
    ax.set_ylabel("dr_total diff (95% CI)")
    ax.grid(True, axis="y", alpha=0.3)
    ylo, yhi = ax.get_ylim()                    # 给 p 值标注留headroom，避免被顶边裁掉
    ax.set_ylim(ylo, yhi + 0.18 * (yhi - ylo))

    fig.suptitle("Figure 3  Context length x architecture "
                 "((c) is the only confirmatory family; rest exploratory/descriptive)",
                 fontsize=11, fontweight="bold", y=0.97)
    _save(fig, outdir, "fig3_rq1_context_gain")


# ---------------------------------------------------------------------------
# Figure 4：负控制（正常 vs 40s shifted）
# ---------------------------------------------------------------------------

def fig4(results, est, outdir):
    from matplotlib.lines import Line2D
    DRT = r"$\Delta r_{\mathrm{total}}$"                     # Δr_total 的 mathtext
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.subplots_adjust(top=0.82, wspace=0.30)

    # (a) IFG r 曲线：normal(实线) vs shifted(点线)——绝对 r 是否被平移摧毁
    ax = axes[0]
    for m in ALL_MODELS:
        st = MODEL_STYLE[m]
        xs = [HPOS[HS.index(h)] for h in HS]
        pn = r_curve(est, m, "left_IFG", "main", "normal")[1]
        ps = r_curve(est, m, "left_IFG", "main", "shift")[1]
        ax.plot(xs, pn, color=st["color"], linestyle="-", marker=st["marker"], linewidth=1.8)
        ax.plot(xs, ps, color=st["color"], linestyle=":", marker=st["marker"],
                linewidth=1.3, alpha=0.8)
    ax.set_xticks(HPOS); ax.set_xticklabels([f"H={h}" for h in HS])
    ax.set_title("(a) IFG brain score r")
    ax.set_ylabel("brain score r"); ax.grid(True, axis="y", alpha=0.3)
    # 图例：明确 实线=normal / 点线=shifted + 模型颜色
    style_handles = [Line2D([0], [0], color="k", linestyle="-", label="normal"),
                     Line2D([0], [0], color="k", linestyle=":", label="shifted (40s)")]
    model_handles = [Line2D([0], [0], color=MODEL_STYLE[m]["color"], marker=MODEL_STYLE[m]["marker"],
                            linestyle="-", label=m) for m in ALL_MODELS]
    ax.legend(handles=style_handles + model_handles, fontsize=7, loc="upper left", ncol=2)

    # (b) Δr_total：normal vs shifted 并排——注意 shifted 并未归零
    ax = axes[1]
    ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
    width = 0.35
    for j, m in enumerate(ALL_MODELS):
        st = MODEL_STYLE[m]
        n_pt, n_lo, n_hi = context_gain(est, m, "left_IFG", "total", shifted=False)
        s_pt, s_lo, s_hi = context_gain(est, m, "left_IFG", "total", shifted=True)
        if n_pt is not None:
            ax.bar(j - width/2, n_pt, width, color=st["color"], alpha=0.85,
                   edgecolor="k", linewidth=0.5, label="normal" if j == 0 else None)
            ax.plot([j - width/2]*2, [n_lo, n_hi], color="k", linewidth=1.0)
        if s_pt is not None:
            ax.bar(j + width/2, s_pt, width, color=st["color"], alpha=0.4, hatch="//",
                   edgecolor="k", linewidth=0.5, label="shifted" if j == 0 else None)
            ax.plot([j + width/2]*2, [s_lo, s_hi], color="k", linewidth=1.0)
    ax.set_xticks(range(len(ALL_MODELS)))
    ax.set_xticklabels(ALL_MODELS, rotation=15)
    ax.set_title(f"(b) {DRT} (IFG main): normal vs shifted")
    ax.set_ylabel(f"{DRT} (95% CI)"); ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # (c) 配对差值 normal−shifted Context Gain（关键统计量）：CI 排除0且正=平移显著削弱
    ax = axes[2]
    ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
    for j, m in enumerate(ALL_MODELS):
        st = MODEL_STYLE[m]
        pt, lo, hi = gain_reduction(est, m)
        _errbar(ax, j, pt, lo, hi, st)
        if pt is not None and lo is not None:
            # 客观对称描述：panel 画 normal − shifted，CI 位于 0 上方=normal>shifted，
            # 下方=shifted>normal，跨0=CI includes 0。不用"reduced/increased"等确认性措辞。
            if lo > 0 and hi > 0:
                txt, col = "normal > shifted", "green"
            elif lo < 0 and hi < 0:
                txt, col = "shifted > normal", "red"
            else:
                txt, col = "CI includes 0", "gray"
            ax.annotate(txt, (j, hi if hi is not None else pt),
                        textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=8, color=col)
    ax.set_xticks(range(len(ALL_MODELS)))
    ax.set_xticklabels(ALL_MODELS, rotation=15)
    ax.set_title(f"(c) normal − shifted {DRT}, paired 95% CI\n"
                 "[diagnostic; unadjusted, not multiplicity-corrected; not confirmatory]",
                 fontsize=9)
    ax.set_ylabel(f"normal − shifted {DRT}"); ax.grid(True, axis="y", alpha=0.3)
    ylo, yhi = ax.get_ylim(); ax.set_ylim(ylo, yhi + 0.15 * (yhi - ylo))

    # 中性标题：纯描述，不预判方向（Pythia 未降、RWKV 反升，"reduced"会误导）
    fig.suptitle("Figure 4. Effects of a 40-s time shift on encoding performance and Context Gain",
                 fontsize=12, fontweight="bold", y=0.97)
    _save(fig, outdir, "fig4_negative_control")


# ---------------------------------------------------------------------------
# Figure 5：层位稳健性（主层 vs 最终层）
# ---------------------------------------------------------------------------

def fig5(results, est, outdir):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
    flips = results.get("layer_flip", {})
    width = 0.35
    for i, arch in enumerate(CORE_VS_PYTHIA):
        st = MODEL_STYLE[arch]
        m_pt, m_lo, m_hi = arch_delta_total(est, arch, "main")
        f_pt, f_lo, f_hi = arch_delta_total(est, arch, "final")
        _errbar(ax, i, m_pt, m_lo, m_hi, st, dx=-width/2)
        _errbar(ax, i, f_pt, f_lo, f_hi, {**st, "marker": "D"}, dx=+width/2)
        base = f"{arch}_minus_pythia_delta_total"
        fl = flips.get(base, {})
        ax.annotate("FLIP" if fl.get("substantive_flip") else "no flip",
                    (i, max(m_hi or 0, f_hi or 0)), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9,
                    color="red" if fl.get("substantive_flip") else "gray")
    ax.set_xticks(range(len(CORE_VS_PYTHIA)))
    ax.set_xticklabels(["RWKV-Pythia", "Mamba-Pythia"])
    ax.set_ylabel("dr_total arch diff (95% CI)")
    ax.set_title("Figure 5  Layer robustness: main (circle) vs final (diamond)\n"
                 "substantive flip = both CIs nonzero with opposite signs")
    ax.scatter([], [], marker="o", color="k", label="main layer")
    ax.scatter([], [], marker="D", color="k", label="final layer")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    ylo, yhi = ax.get_ylim()                    # 给"FLIP/no flip"标注留headroom，避免被顶边裁掉
    ax.set_ylim(ylo, yhi + 0.15 * (yhi - ylo))
    _save(fig, outdir, "fig5_layer_robustness")


FIGURES = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--m5-name", default="m5_stats")
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), default=None,
                    help="只画指定图，默认全画")
    args = ap.parse_args()

    cfg = load_config()
    results_path = Path(cfg["paths"]["results_dir"]) / args.m5_name / args.subject / "m5_results.json"
    if not results_path.exists():
        raise SystemExit(f"未找到 M5 结果：{results_path}（先跑 scripts/m5_analysis.py）")
    results = load_results(results_path)
    est = results["estimands"]
    outdir = Path(cfg["paths"]["figures_dir"]) / args.subject

    which = args.only if args.only else list(FIGURES)
    print(f"[m6] 读 {results_path} → 生成 {which} → {outdir}", flush=True)
    for name in which:
        FIGURES[name](results, est, outdir)
    print(f"[m6] 完成。图表在 {outdir}", flush=True)


if __name__ == "__main__":
    main()
