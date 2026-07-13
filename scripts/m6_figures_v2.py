"""
M6 —— 论文图表生成 v2（对应 milestone/实验结果图统一命名与排版优化方案.md）。

与 v1（scripts/m6_figures.py）关系：**v1 完全不改动**，继续产出到
figures/<subject>/；本脚本产出到 figures/<subject>/v2/，两套图片同时存在、
互不覆盖，方便直接对照新旧排版。数据访问层（src/viz/m6_data.py 的纯函数）
两者完全共用，只读 M5 结果，不重算任何统计。

相对 v1 的主要改动（方案第13节优先级）：
  1. 三被试坐标范围统一（各图固定 ylim/xlim 常量），但接入 safe_lim 安全网：
     真实数据若超出预设范围绝不静默裁掉，自动扩展并打印警告（本地合成数据
     冒烟测试时实测抓到过 Fig4(c) 的显著性星号被固定 xlim 裁没的真实 bug）。
  2. 图内大标题只留一句话，方法/统计细节移入图注（存成 .caption.txt）。
  3. Figure 2 拆出 AWD-LSTM 独立面板，避免它拉宽核心三模型的纵轴。
  4. Figure 3/4 改两行布局，(c) 改横向森林图。
  5. Figure 4(c) 用星号替代长文字标注（方案6.6）——不只是把文字变短，而是彻底
     不在图内写"normal > shifted"这类长句，方向解释移入图注。这同时解决了
     实测发现的标注顶穿图框问题。
  6. Figure 5 改横向配对森林图。
  7. p 值统一为 p<0.001（对齐 n_boot=1000 的 bootstrap 分辨率），不用 p<0.0001。
  8. Panel (c) 的"CONFIRMATORY"标签保留在图内（不完全移入图注）——对方案的一处
     修正：确认性/探索性区分是本项目核心纪律，图注容易被略过。
  9. 所有图例移到图外部（图底部整体一行），不再猜"哪个角落大概率没数据"——
     本地渲染实测发现过图例压在数据线/误差棒上的真实问题，挪出轴外后从
     结构上不可能再撞车。

用法：
  python3 scripts/m6_figures_v2.py --subject UTS01
  python3 scripts/m6_figures_v2.py --subject UTS01 --only fig3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.lines import Line2D           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config     # noqa: E402
from src.viz.m6_data import (                 # noqa: E402
    load_results, r_curve, context_gain, rq1, arch_delta_total, gain_reduction,
    confirmatory_rows, HS, CORE_VS_PYTHIA, CONFIRMATORY,
)
from src.viz.m6_style_v2 import (             # noqa: E402
    MODEL_STYLE, CORE_MODELS, REFERENCE_MODEL, HPOS,
    FS_BIG_TITLE, FS_PANEL_TITLE, FS_AXIS, FS_LEGEND,
    LW_DATA, LW_ERRBAR, LW_ZERO,
    style_axes, subject_tag, fmt_p, safe_lim,
    annotate_with_headroom_h, save_with_caption,
)


def _errbar_v(ax, x, pt, lo, hi, style, label=None, dx=0.0):
    """纵向点+CI误差棒（Figure 2/3a 曲线用）；缺项跳过。"""
    if pt is None:
        return
    yerr = [[pt - lo], [hi - pt]] if (lo is not None and hi is not None) else None
    ax.errorbar(x + dx, pt, yerr=yerr, fmt=style.get("marker", "o"),
               color=style["color"], ecolor=style["color"], capsize=3,
               markersize=6, label=label, linewidth=LW_DATA, elinewidth=LW_ERRBAR)


def _forest_row(ax, y, pt, lo, hi, style, marker=None):
    """横向森林图的一行：点 + 水平 CI。"""
    if pt is None:
        return
    xerr = [[pt - lo], [hi - pt]] if (lo is not None and hi is not None) else None
    ax.errorbar(pt, y, xerr=xerr, fmt=marker or style.get("marker", "o"),
               color=style["color"], ecolor=style["color"], capsize=3,
               markersize=7, linewidth=LW_DATA, elinewidth=LW_ERRBAR)


def _bottom_legend(fig, handles, *, ncol=None, y=-0.02):
    """图例整体放在图外部最下方——结构上不可能和任何面板的数据重叠，不需要
    猜'哪个角落大概率是空的'（本地渲染实测过靠猜的做法会撞车）。"""
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=ncol or len(handles), fontsize=FS_LEGEND, frameon=False)


# ---------------------------------------------------------------------------
# Figure 2 — Encoding Performance Across Context Lengths
# ---------------------------------------------------------------------------

# 三被试固定坐标范围（方案11.2：不逐被试自动缩放；safe_lim 是安全网，不是常规
# 路径）。以下数值已用 UTS01/UTS02/UTS03 三人的真实 M5 结果核对过（服务器实测，
# 2026-07-13）：每个范围都比三人实际需要的最大跨度再留一点余量，三人都不应
# 再触发 safe_lim 的自动扩展警告；若后续重跑 M5（换种子/改故事划分等）导致
# 数值变化到触发警告，把新的警告数字加入下面注释、按同样方法重新收紧范围。
FIG2_YLIM_CORE = (0.0, 0.17)
FIG2_YLIM_AWD = (0.0, 0.11)     # 实测 hi 需要到 0.0958/0.1021/0.1026


def fig2(results, est, subject, outdir):
    fig = plt.figure(figsize=(10.5, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.3], hspace=0.5, wspace=0.25)
    ax_ifg = fig.add_subplot(gs[0, 0])
    ax_pt = fig.add_subplot(gs[0, 1], sharey=ax_ifg)
    ax_awd = fig.add_subplot(gs[1, :])

    core_vals = []
    for ax, roi, title in ((ax_ifg, "left_IFG", "(a) Left IFG"),
                           (ax_pt, "bilateral_PT", "(b) Bilateral PT")):
        for m in CORE_MODELS:
            st = MODEL_STYLE[m]
            hs, pts, los, his = r_curve(est, m, roi)
            xs = [HPOS[h] for h in hs]
            ax.plot(xs, pts, color=st["color"], linestyle=st["linestyle"],
                    marker=st["marker"], linewidth=LW_DATA, markersize=6)
            for x, p, lo, hi in zip(xs, pts, los, his):
                if p is not None and lo is not None:
                    ax.plot([x, x], [lo, hi], color=st["color"], alpha=0.5, linewidth=LW_ERRBAR)
                    core_vals += [lo, hi]
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["8", "32", "128"])
        ax.set_title(title, fontsize=FS_PANEL_TITLE)
        ax.set_xlabel("Context Length (tokens)", fontsize=FS_AXIS)
        style_axes(ax)
    safe_lim(ax_ifg, FIG2_YLIM_CORE, core_vals, context="fig2(a,b)")
    ax_ifg.set_ylabel("Encoding Score, $r$", fontsize=FS_AXIS)

    awd_vals = []
    for roi, ls in (("left_IFG", "-"), ("bilateral_PT", "--")):
        st = MODEL_STYLE[REFERENCE_MODEL]
        hs, pts, los, his = r_curve(est, REFERENCE_MODEL, roi)
        xs = [HPOS[h] for h in hs]
        label = "Left IFG" if roi == "left_IFG" else "Bilateral PT"
        ax_awd.plot(xs, pts, color=st["color"], linestyle=ls, marker=st["marker"],
                    label=label, linewidth=LW_DATA, markersize=6)
        for x, p, lo, hi in zip(xs, pts, los, his):
            if p is not None and lo is not None:
                ax_awd.plot([x, x], [lo, hi], color=st["color"], alpha=0.5, linewidth=LW_ERRBAR)
                awd_vals += [lo, hi]
    ax_awd.set_xticks([0, 1, 2]); ax_awd.set_xticklabels(["8", "32", "128"])
    ax_awd.set_title("(c) AWD-LSTM Historical Reference — separate y-scale",
                     fontsize=FS_PANEL_TITLE)
    ax_awd.set_xlabel("Context Length (tokens)", fontsize=FS_AXIS)
    ax_awd.set_ylabel("Encoding Score, $r$", fontsize=FS_AXIS)
    safe_lim(ax_awd, FIG2_YLIM_AWD, awd_vals, context="fig2(c)")
    ax_awd.legend(fontsize=FS_LEGEND, loc="lower right", frameon=True,
                 facecolor="white", edgecolor="none", framealpha=0.9)
    style_axes(ax_awd)

    fig.suptitle("Encoding Performance Across Context Lengths", fontsize=FS_BIG_TITLE,
                fontweight="bold", y=0.99)
    subject_tag(fig, subject, x=0.985, y=0.99)
    _bottom_legend(fig, [Line2D([0], [0], color=MODEL_STYLE[m]["color"],
                                marker=MODEL_STYLE[m]["marker"], linestyle="-",
                                linewidth=LW_DATA, label=MODEL_STYLE[m]["label"])
                        for m in CORE_MODELS])

    caption = f"""
Figure 2. Encoding performance across context lengths, subject {subject}.
(a-b) Core three models (Pythia, Mamba, RWKV) only; AWD-LSTM shown separately
in (c) on its own y-scale because its absolute encoding score is substantially
lower and would otherwise compress the visible differences among the core
models. Left IFG is the primary ROI; bilateral PT is a reference ROI. Error
bars are 95% bootstrap CI (paired story resampling, method B aggregation).
H denotes the number of preceding words available to the model (not including
the target word). Primary analyses use a predefined layer at approximately
2/3 relative depth for each model (see Methods); AWD-LSTM is a historical
reference and is not part of the core three-model architecture ranking.
""".strip()
    save_with_caption(fig, outdir, "fig2_main_curves", caption)


# ---------------------------------------------------------------------------
# Figure 3 — Architecture-Specific Context Effects
# ---------------------------------------------------------------------------

FIG3A_YLIM = (-0.008, 0.012)    # 实测 lo 需要到 -0.0067，hi 需要到 0.0108
FIG3B_YLIM = (-0.003, 0.010)    # 实测 lo 需要到 -0.0024，hi 需要到 0.0094
FIG3C_XLIM = (-0.005, 0.007)    # 三人实测均已覆盖，无需调整


def fig3(results, est, subject, outdir):
    fig = plt.figure(figsize=(11, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # (a) Matched-Context Architecture Contrasts
    ax_a.axhline(0, color="k", linewidth=LW_ZERO, linestyle=":")
    a_vals = []
    for arch in CORE_VS_PYTHIA:
        st = MODEL_STYLE[arch]
        xs = [HPOS[h] for h in HS]
        pts = [rq1(est, arch, h)[0] for h in HS]
        ax_a.plot(xs, pts, color=st["color"], marker=st["marker"],
                 linewidth=LW_DATA, markersize=6)
        for h, x in zip(HS, xs):
            p, lo, hi = rq1(est, arch, h)
            if p is not None:
                ax_a.plot([x, x], [lo, hi], color=st["color"], alpha=0.5, linewidth=LW_ERRBAR)
                a_vals += [lo, hi]
    ax_a.set_xticks([0, 1, 2]); ax_a.set_xticklabels(["8", "32", "128"])
    ax_a.set_xlabel("Context Length (tokens)", fontsize=FS_AXIS)
    ax_a.set_ylabel("Encoding-Score Difference vs Pythia", fontsize=FS_AXIS)
    ax_a.set_title("(a) Matched-Context Architecture Contrasts", fontsize=FS_PANEL_TITLE)
    safe_lim(ax_a, FIG3A_YLIM, a_vals, context="fig3(a)")
    style_axes(ax_a)

    # (b) Context Gain by Model — 点/误差棒替代柱状图（方案5.6）
    ax_b.axhline(0, color="k", linewidth=LW_ZERO, linestyle=":")
    kinds = ["local", "long", "total"]
    width = 0.18
    b_vals = []
    for j, m in enumerate(CORE_MODELS + [REFERENCE_MODEL]):
        st = MODEL_STYLE[m]
        for k, kind in enumerate(kinds):
            pt, lo, hi = context_gain(est, m, "left_IFG", kind)
            x = k + (j - 1.5) * width
            if pt is not None:
                _errbar_v(ax_b, x, pt, lo, hi, st)
                if lo is not None:
                    b_vals += [lo, hi]
    ax_b.set_xticks(range(len(kinds)))
    ax_b.set_xticklabels(["Local Gain", "Long-Range Gain", "Total Gain"])
    ax_b.set_ylabel(r"Context Gain, $\Delta r$", fontsize=FS_AXIS)
    ax_b.set_title("(b) Context Gain by Model", fontsize=FS_PANEL_TITLE)
    safe_lim(ax_b, FIG3B_YLIM, b_vals, context="fig3(b)")
    style_axes(ax_b)

    # (c) CONFIRMATORY — 横向森林图（方案5.7 + 保留"CONFIRMATORY"标签，见文件头说明）
    ax_c.axvline(0, color="k", linewidth=LW_ZERO, linestyle=":")
    rows = confirmatory_rows(results)
    order = {"rwkv_minus_pythia_delta_total_ifg_main": 1,
            "mamba_minus_pythia_delta_total_ifg_main": 0}
    rows = sorted(rows, key=lambda r: order.get(r["name"], 99))
    labels, annots, c_vals = [], [], []
    for i, row in enumerate(rows):
        arch = "rwkv" if "rwkv" in row["name"] else "mamba"
        st = MODEL_STYLE[arch]
        y = len(rows) - 1 - i
        _forest_row(ax_c, y, row["point"], row["ci_lo"], row["ci_hi"], st)
        if row["ci_lo"] is not None:
            c_vals += [row["ci_lo"], row["ci_hi"]]
        labels.append((y, f"{st['label']} − Pythia"))
        star = "*" if row.get("reject") else ""
        txt = f"{star} {fmt_p(row['p'], holm=True)}".strip()
        annots.append((row["ci_hi"] if row["ci_hi"] is not None else row["point"], y, txt, "black"))
    ax_c.set_yticks([y for y, _ in labels])
    ax_c.set_yticklabels([lab for _, lab in labels])
    ax_c.set_xlabel(r"Difference in Total Context Gain, $\Delta r_{\mathrm{total}}$", fontsize=FS_AXIS)
    ax_c.set_title("(c) CONFIRMATORY: Total Context-Gain Contrasts", fontsize=FS_PANEL_TITLE,
                  fontweight="bold")
    ax_c.set_ylim(-0.7, len(rows) - 0.3)
    safe_lim(ax_c, FIG3C_XLIM, c_vals, axis="x", context="fig3(c)")
    style_axes(ax_c, grid_axis="x")
    annotate_with_headroom_h(ax_c, annots)

    fig.suptitle("Architecture-Specific Context Effects", fontsize=FS_BIG_TITLE,
                fontweight="bold", y=0.99)
    subject_tag(fig, subject, x=0.985, y=0.99)
    _bottom_legend(fig, [Line2D([0], [0], color=MODEL_STYLE[a]["color"],
                                marker=MODEL_STYLE[a]["marker"], linestyle="-",
                                linewidth=LW_DATA, label=f"{MODEL_STYLE[a]['label']} − Pythia")
                        for a in CORE_VS_PYTHIA] +
                  [Line2D([0], [0], color=MODEL_STYLE[m]["color"], marker=MODEL_STYLE[m]["marker"],
                          linestyle="", markersize=7, label=MODEL_STYLE[m]["label"])
                   for m in CORE_MODELS + [REFERENCE_MODEL]], ncol=6)

    caption = f"""
Figure 3. Architecture-specific context effects, subject {subject}.
(a) Matched-context architecture contrasts (RWKV/Mamba minus Pythia encoding
score at the same H), IFG primary layer, exploratory. (b) Three descriptive
Context Gain measures per model, IFG primary layer: Delta r_local = r32-r8,
Delta r_long = r128-r32, Delta r_total = r128-r8. (c) The two CONFIRMATORY
Delta r_total architecture contrasts (IFG primary layer), the only estimands
in this figure carrying a confirmatory conclusion; Holm-Bonferroni corrected
at family-wise alpha=0.05 across these two comparisons. * = rejects H0 after
Holm correction. Error bars/whiskers are 95% bootstrap CI (paired story
resampling, 1000 draws).
""".strip()
    save_with_caption(fig, outdir, "fig3_rq1_context_gain", caption)


# ---------------------------------------------------------------------------
# Figure 4 — Temporal-Shift Control
# ---------------------------------------------------------------------------

FIG4A_YLIM = (-0.012, 0.17)     # 实测 lo 需要到 -0.0102
FIG4B_YLIM = (-0.003, 0.010)    # 实测 hi 需要到 0.0094
FIG4C_XLIM = (-0.008, 0.008)    # 实测 lo 需要到 -0.0070


def fig4(results, est, subject, outdir):
    fig = plt.figure(figsize=(11, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # (a) Encoding Performance: normal(实线) vs shifted(点线)
    a_vals = []
    for m in CORE_MODELS + [REFERENCE_MODEL]:
        st = MODEL_STYLE[m]
        xs = [HPOS[h] for h in HS]
        pn = r_curve(est, m, "left_IFG", "main", "normal")[1]
        ps = r_curve(est, m, "left_IFG", "main", "shift")[1]
        ax_a.plot(xs, pn, color=st["color"], linestyle="-", marker=st["marker"], linewidth=LW_DATA)
        ax_a.plot(xs, ps, color=st["color"], linestyle=":", marker=st["marker"],
                 linewidth=LW_DATA * 0.75, alpha=0.8)
        a_vals += [v for v in pn + ps if v is not None]
    ax_a.set_xticks([0, 1, 2]); ax_a.set_xticklabels(["8", "32", "128"])
    ax_a.set_xlabel("Context Length (tokens)", fontsize=FS_AXIS)
    ax_a.set_ylabel("Encoding Score, $r$", fontsize=FS_AXIS)
    ax_a.set_title("(a) Encoding Performance", fontsize=FS_PANEL_TITLE)
    safe_lim(ax_a, FIG4A_YLIM, a_vals, context="fig4(a)")
    style_axes(ax_a)

    # (b) Total Context Gain: paired points (normal-shifted connected), 方案6.5
    ax_b.axhline(0, color="k", linewidth=LW_ZERO, linestyle=":")
    b_vals = []
    for j, m in enumerate(CORE_MODELS + [REFERENCE_MODEL]):
        st = MODEL_STYLE[m]
        n_pt, n_lo, n_hi = context_gain(est, m, "left_IFG", "total", shifted=False)
        s_pt, s_lo, s_hi = context_gain(est, m, "left_IFG", "total", shifted=True)
        if n_pt is not None and s_pt is not None:
            ax_b.plot([j, j], [n_pt, s_pt], color="#bbbbbb", linewidth=1.2, zorder=1)
        if n_pt is not None:
            ax_b.errorbar(j, n_pt, yerr=[[n_pt - n_lo], [n_hi - n_pt]], fmt="o",
                         color=st["color"], ecolor=st["color"], capsize=3, markersize=6,
                         linewidth=LW_DATA, elinewidth=LW_ERRBAR, zorder=2)
            b_vals += [n_lo, n_hi]
        if s_pt is not None:
            ax_b.errorbar(j, s_pt, yerr=[[s_pt - s_lo], [s_hi - s_pt]], fmt="o", mfc="white",
                         color=st["color"], ecolor=st["color"], capsize=3, markersize=6,
                         linewidth=LW_DATA, elinewidth=LW_ERRBAR, zorder=2)
            b_vals += [s_lo, s_hi]
    ax_b.set_xticks(range(4))
    ax_b.set_xticklabels([MODEL_STYLE[m]["label"] for m in CORE_MODELS + [REFERENCE_MODEL]],
                         rotation=12)
    ax_b.set_ylabel(r"Total Context Gain, $\Delta r_{\mathrm{total}}$", fontsize=FS_AXIS)
    ax_b.set_title("(b) Total Context Gain", fontsize=FS_PANEL_TITLE)
    safe_lim(ax_b, FIG4B_YLIM, b_vals, context="fig4(b)")
    style_axes(ax_b)

    # (c) Normal-Shifted Difference — 横向森林图，星号替代长文字（方案6.6）
    ax_c.axvline(0, color="k", linewidth=LW_ZERO, linestyle=":")
    models_c = CORE_MODELS + [REFERENCE_MODEL]
    annots, c_vals = [], []
    for i, m in enumerate(models_c):
        st = MODEL_STYLE[m]
        y = len(models_c) - 1 - i
        pt, lo, hi = gain_reduction(est, m)
        _forest_row(ax_c, y, pt, lo, hi, st)
        if pt is not None and lo is not None:
            c_vals += [lo, hi]
            sig = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
            star = "*" if sig else ""
            annots.append((hi if hi is not None else pt, y, star, "black"))
    ax_c.set_yticks(range(len(models_c)))
    ax_c.set_yticklabels([MODEL_STYLE[m]["label"] for m in reversed(models_c)])
    ax_c.set_xlabel(r"Normal $-$ Shifted Total Context Gain", fontsize=FS_AXIS)
    ax_c.set_title("(c) Normal−Shifted Difference (diagnostic, unadjusted)",
                  fontsize=FS_PANEL_TITLE)
    ax_c.set_ylim(-0.6, len(models_c) - 0.4)
    safe_lim(ax_c, FIG4C_XLIM, c_vals, axis="x", context="fig4(c)")
    style_axes(ax_c, grid_axis="x")
    annotate_with_headroom_h(ax_c, annots)

    fig.suptitle("Temporal-Shift Control", fontsize=FS_BIG_TITLE, fontweight="bold", y=0.99)
    subject_tag(fig, subject, x=0.985, y=0.99)
    style_handles = [Line2D([0], [0], color="k", linestyle="-", linewidth=LW_DATA, label="Normal"),
                     Line2D([0], [0], color="k", linestyle=":", linewidth=LW_DATA, label="Shifted (40s)")]
    model_handles = [Line2D([0], [0], color=MODEL_STYLE[m]["color"], marker=MODEL_STYLE[m]["marker"],
                            linestyle="-", linewidth=LW_DATA, label=MODEL_STYLE[m]["label"])
                     for m in CORE_MODELS + [REFERENCE_MODEL]]
    _bottom_legend(fig, style_handles + model_handles, ncol=6)

    caption = f"""
Figure 4. Effect of a 40-second non-circular time shift on encoding
performance and Context Gain, subject {subject}. (a) IFG primary-layer
encoding score, normal (solid) vs. shifted (dotted). (b) Total Context Gain
(Delta r_total), normal vs. shifted, paired per model; gray lines connect
the same model's two conditions (filled marker = normal, open marker =
shifted). (c) Paired normal-minus-shifted difference in Total Context Gain
per model — diagnostic only, unadjusted for multiple comparisons, not a
confirmatory result. * marks 95% CI excluding zero; direction (normal>shifted
vs. shifted>normal) can be read directly from which side of the zero line
the point and CI fall on. Error bars are 95% bootstrap CI (paired story
resampling).
""".strip()
    save_with_caption(fig, outdir, "fig4_negative_control", caption)


# ---------------------------------------------------------------------------
# Figure 5 — Layer Sensitivity of Context-Gain Contrasts
# ---------------------------------------------------------------------------

FIG5_XLIM = (-0.006, 0.009)     # 实测 lo 需要到 -0.0050


def fig5(results, est, subject, outdir):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.axvline(0, color="k", linewidth=LW_ZERO, linestyle=":")
    contrasts = list(reversed(CORE_VS_PYTHIA))
    vals = []
    for i, arch in enumerate(contrasts):
        st = MODEL_STYLE[arch]
        y = i
        m_pt, m_lo, m_hi = arch_delta_total(est, arch, "main")
        f_pt, f_lo, f_hi = arch_delta_total(est, arch, "final")
        if m_pt is not None and f_pt is not None:
            ax.plot([m_pt, f_pt], [y, y], color="#bbbbbb", linewidth=1.2, zorder=1)
        _forest_row(ax, y, m_pt, m_lo, m_hi, st, marker="o")
        _forest_row(ax, y, f_pt, f_lo, f_hi, st, marker="D")
        for lo, hi in ((m_lo, m_hi), (f_lo, f_hi)):
            if lo is not None:
                vals += [lo, hi]
    ax.set_yticks(range(len(contrasts)))
    ax.set_yticklabels([f"{MODEL_STYLE[a]['label']} − Pythia" for a in contrasts])
    ax.set_xlabel("Total Context-Gain Difference vs Pythia", fontsize=FS_AXIS)
    ax.set_ylim(-0.6, len(contrasts) - 0.4)
    safe_lim(ax, FIG5_XLIM, vals, axis="x", context="fig5")
    style_axes(ax, grid_axis="x")

    fig.suptitle("Layer Sensitivity of Context-Gain Contrasts", fontsize=FS_BIG_TITLE,
                fontweight="bold", y=1.06)
    subject_tag(fig, subject, x=0.99, y=1.1)
    _bottom_legend(fig, [Line2D([0], [0], marker="o", color="k", linestyle="", markersize=7, label="Main Layer"),
                       Line2D([0], [0], marker="D", color="k", linestyle="", markersize=7, label="Final Layer")],
                  ncol=2, y=-0.05)

    caption = f"""
Figure 5. Layer sensitivity of the two architecture Total Context-Gain
contrasts, subject {subject}. Main Layer (circle) is the predefined
approximately 2/3-relative-depth layer used for the primary/confirmatory
analyses (Pythia layer 8/12, RWKV layer 8/12, Mamba layer 16/24). Final
Layer (diamond) is each model's last layer (Pythia 11, RWKV 11, Mamba 23),
used only as a robustness check, not for confirmatory conclusions. A
substantive layer flip is defined as the two layers' 95% CIs both excluding
zero with opposite signs. Error bars are 95% bootstrap CI.
""".strip()
    save_with_caption(fig, outdir, "fig5_layer_robustness", caption)


FIGURES = {"fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--m5-name", default="m5_stats")
    ap.add_argument("--out-subdir", default="v2",
                    help="v1 图片(figures/<subject>/*.png)保持不动，v2 写到 "
                         "figures/<subject>/<out-subdir>/，两套并存")
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), default=None)
    args = ap.parse_args()

    cfg = load_config()
    results_path = Path(cfg["paths"]["results_dir"]) / args.m5_name / args.subject / "m5_results.json"
    if not results_path.exists():
        raise SystemExit(f"未找到 M5 结果：{results_path}（先跑 scripts/m5_analysis.py）")
    results = load_results(results_path)
    est = results["estimands"]
    outdir = Path(cfg["paths"]["figures_dir"]) / args.subject / args.out_subdir

    which = args.only if args.only else list(FIGURES)
    print(f"[m6v2] 读 {results_path} → 生成 {which} → {outdir}", flush=True)
    for name in which:
        FIGURES[name](results, est, args.subject, outdir)
    print(f"[m6v2] 完成。v2 图表在 {outdir}（v1 图表原样保留在上一级目录）", flush=True)


if __name__ == "__main__":
    main()
