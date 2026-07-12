"""
M6 跨被试图表 —— 三被试并排主曲线 + 跨被试方向一致性图。只读 M5 结果，不重算。

对应冻结文档里程碑6设计思路："五图两表……全部逐被试并排呈现，不给合并组水平数值"，
"跨被试判读用『该效应在三名被试中方向一致 / 部分一致 / 不一致』这类描述性语言，主证据
是被试内效应的跨被试重复性而非合并显著性"。

生成两张跨被试图（补逐被试 m6_figures.py 之外的三被试并排视角）：
  figX_cross_main         三被试 × IFG/PT 的四模型 r8/r32/r128 曲线，并排呈现（每被试一列）
  figX_cross_confirmatory 两项确认性 Δr_total 架构差值：各被试 point+95%CI 并排 + 方向一致性判读

红线（里程碑6）：
  - 不给任何合并组水平数值/组水平CI —— 各被试独立并排，一致性只用描述性标签；
  - AWD-LSTM 灰色虚线视觉隔离，不参与核心排名（复用 MODEL_STYLE）。

数据来源：
  - 主曲线：每被试各自 results/<m5-name>/<subject>/m5_results.json；
  - 一致性判读：results/<cross-name>/m5_cross_subject.json（由 m5_cross_subject.py 生成，
    是 M5 的结构化产物；M6 只读，不在此重算判读）。

用法（服务器或本地，需三被试 M5 结果可达）：
  python3 scripts/m6_cross_subject_figures.py
  python3 scripts/m6_cross_subject_figures.py --only cross_confirmatory
输出 PNG(150dpi) + PDF(矢量) 到 figures/cross_subject/。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                        # 服务器无显示，非交互后端
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.lines import Line2D          # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config     # noqa: E402
from src.viz.m6_data import (                 # noqa: E402
    load_results, r_curve,
    CORE_MODELS, REFERENCE_MODEL, MODEL_STYLE, HS, CONFIRMATORY,
)

ALL_MODELS = CORE_MODELS + [REFERENCE_MODEL]
HPOS = list(range(len(HS)))                   # 8/32/128 三点均匀间距（非真数轴）

# 确认性对比的英文短名（图内不用中文，避免默认字体渲染成方块）
CONTRAST_LABEL = {
    "rwkv_minus_pythia_delta_total_ifg_main": "RWKV - Pythia",
    "mamba_minus_pythia_delta_total_ifg_main": "Mamba - Pythia",
}
# 一致性判读的英文标签（与 src/stats/cross_subject.py 的四类一一对应）
CONSISTENCY_LABEL_EN = {
    "consistent_strong": "Consistent\n(strong within-subject replication)",
    "consistent_direction_only": "Directionally consistent\n(not all CIs exclude 0)",
    "heterogeneous": "Heterogeneous across subjects",
    "insufficient_data": "Insufficient data",
}


def _save(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[m6x] 已保存 {name}.png / .pdf", flush=True)


# ---------------------------------------------------------------------------
# 三被试并排主曲线（IFG/PT × 被试）
# ---------------------------------------------------------------------------

def fig_cross_main(subj_results: dict, subjects: list[str], outdir: Path):
    """行=ROI(IFG/PT)，列=被试；每格四模型 r 随 H 曲线 + CI。逐被试并排，不合并。"""
    rois = [("left_IFG", "Left IFG (primary)"),
            ("bilateral_PT", "Bilateral PT (reference)")]
    nrow, ncol = len(rois), len(subjects)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.1 * nrow),
                             sharey="row", squeeze=False)
    for r, (roi, rtitle) in enumerate(rois):
        for c, subj in enumerate(subjects):
            ax = axes[r][c]
            est = subj_results[subj]["estimands"]
            for m in ALL_MODELS:
                hs, pts, los, his = r_curve(est, m, roi)
                st = MODEL_STYLE[m]
                xs = [HPOS[HS.index(h)] for h in hs]
                ax.plot(xs, pts, color=st["color"], linestyle=st["linestyle"],
                        marker=st["marker"], label=st["label"], linewidth=1.6,
                        markersize=5)
                for x, p, lo, hi in zip(xs, pts, los, his):
                    if p is not None and lo is not None:
                        ax.plot([x, x], [lo, hi], color=st["color"], alpha=0.5,
                                linewidth=1.0)
            ax.set_xticks(HPOS)
            ax.set_xticklabels([f"H={h}" for h in HS])
            ax.grid(True, axis="y", alpha=0.3)
            if r == 0:
                ax.set_title(subj, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{rtitle}\nbrain score r (95% CI)")
    axes[0][ncol - 1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Cross-subject main curves (r8/r32/r128), subjects side by side; "
                 "no pooled/group-level values. AWD-LSTM: gray dashed historical "
                 "reference, not in core ranking.", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, outdir, "figX_cross_main")


# ---------------------------------------------------------------------------
# 跨被试方向一致性（两项确认性 Δr_total 架构差值）
# ---------------------------------------------------------------------------

def fig_cross_confirmatory(cross: dict, subjects: list[str], outdir: Path):
    """每项确认性对比一个子图：各被试 point+95%CI 并排 + 方向一致性判读标签。

    实心点 = 该被试 95%CI 排除 0；空心点 = CI 跨 0。判读标签直接取自 M5 跨被试
    结构化产物（consistency 字段），M6 不在此重算。
    """
    groups = cross["confirmatory_architecture_consistency"]
    names = [n for n in CONFIRMATORY if n in groups]  # 固定顺序（rwkv, mamba）
    fig, axes = plt.subplots(1, len(names), figsize=(5.6 * len(names), 5.0),
                             squeeze=False)
    for i, name in enumerate(names):
        ax = axes[0][i]
        ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
        res = groups[name]
        for j, subj in enumerate(subjects):
            ps = res["per_subject"][subj]
            pt, lo, hi = ps["point"], ps["ci_lo"], ps["ci_hi"]
            if pt is None or (isinstance(pt, float) and pt != pt):   # None/NaN 跳过
                continue
            filled = ps["ci_excludes_zero"]
            yerr = ([[pt - lo], [hi - pt]]
                    if lo is not None and hi is not None else None)
            ax.errorbar(j, pt, yerr=yerr, fmt="o", markersize=8,
                        mfc=("k" if filled else "white"), mec="k",
                        ecolor="k", capsize=4, linewidth=1.4)
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects)
        ax.set_xlim(-0.6, len(subjects) - 0.4)
        verdict = CONSISTENCY_LABEL_EN.get(res["consistency"], res["consistency"])
        ax.set_title(f"{CONTRAST_LABEL.get(name, name)}\n[{verdict}]", fontsize=10)
        if i == 0:
            ax.set_ylabel(r"$\Delta r_{\mathrm{total}}$ arch diff vs Pythia (95% CI)")
        ax.grid(True, axis="y", alpha=0.3)

    legend = [Line2D([0], [0], marker="o", color="k", mfc="k", linestyle="",
                     markersize=8, label="95% CI excludes 0"),
              Line2D([0], [0], marker="o", color="k", mfc="white", linestyle="",
                     markersize=8, label="95% CI includes 0")]
    axes[0][-1].legend(handles=legend, fontsize=8, loc="best")
    fig.suptitle("Cross-subject direction consistency of the two CONFIRMATORY "
                 "architecture contrasts (IFG main, per-subject; descriptive "
                 "replication, NOT pooled significance)", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, outdir, "figX_cross_confirmatory")


FIGURES = {"cross_main": "main", "cross_confirmatory": "confirmatory"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["UTS01", "UTS02", "UTS03"])
    ap.add_argument("--m5-name", default="m5_stats",
                    help="逐被试 M5 结果目录名")
    ap.add_argument("--cross-name", default="m5_cross_subject",
                    help="跨被试综合结果目录名（m5_cross_subject.py 的 --out-name）")
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), default=None,
                    help="只画指定图，默认全画")
    args = ap.parse_args()

    if len(set(args.subjects)) != len(args.subjects):
        raise SystemExit(f"--subjects 存在重复：{args.subjects}；跨被试图要求各被试唯一")

    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    outdir = Path(cfg["paths"]["figures_dir"]) / "cross_subject"
    which = args.only if args.only else list(FIGURES)

    # 主曲线需各被试逐被试结果；一致性图需跨被试产物。按需加载，缺失给明确指引。
    subj_results = None
    if "cross_main" in which:
        subj_results = {}
        for s in args.subjects:
            p = results_dir / args.m5_name / s / "m5_results.json"
            if not p.exists():
                raise SystemExit(
                    f"未找到 {s} 的 M5 结果：{p}\n"
                    f"  → 先跑：python scripts/m5_analysis.py --subject {s}")
            subj_results[s] = load_results(p)

    cross = None
    if "cross_confirmatory" in which:
        cp = results_dir / args.cross_name / "m5_cross_subject.json"
        if not cp.exists():
            raise SystemExit(
                f"未找到跨被试综合结果：{cp}\n"
                f"  → 先跑：python scripts/m5_cross_subject.py "
                f"--subjects {' '.join(args.subjects)}")
        cross = load_results(cp)
        # 图脚本的 --subjects 必须是跨被试产物覆盖的子集，否则按被试取 per_subject
        # 会抛难懂的 KeyError；此处显式挡下并指明该产物实际是用哪些被试生成的。
        missing = [s for s in args.subjects if s not in cross.get("subjects", [])]
        if missing:
            raise SystemExit(
                f"跨被试结果 {cp} 不含被试 {missing}（它是用 "
                f"{cross.get('subjects')} 生成的）；请用一致的 --subjects 重跑 "
                f"m5_cross_subject.py，或把本脚本 --subjects 对齐后再画。")

    print(f"[m6x] 被试={args.subjects} 生成 {which} → {outdir}", flush=True)
    if "cross_main" in which:
        fig_cross_main(subj_results, args.subjects, outdir)
    if "cross_confirmatory" in which:
        fig_cross_confirmatory(cross, args.subjects, outdir)
    print(f"[m6x] 完成。跨被试图在 {outdir}", flush=True)


if __name__ == "__main__":
    main()
