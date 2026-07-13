"""
M6 图表 v2 —— 统一样式常量与排版工具（对应 milestone/实验结果图统一命名与排版
优化方案.md）。

**独立于 src/viz/m6_data.py 与 scripts/m6_figures.py**：v1 完全不改动，v1 的既有
产物（figures/<subject>/fig*.png）继续存在，v2 出图写到单独的子目录，两者互不
覆盖，随时可对照。数据访问层（r_curve/context_gain/rq1/... 等纯函数）与 v1
完全共用，本模块只负责"画得好看"，不重算/不改任何统计口径。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# 字体：优先 Arial/Helvetica（方案 11.3），服务器上多半没装，回退到 matplotlib
# 自带的 DejaVu Sans，避免因缺字体而报警告/渲染失败。
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# 字号（V2 §16 最终统一参数）
FS_BIG_TITLE = 15
FS_PANEL_TITLE = 11
FS_AXIS = 10
FS_TICK = 9
FS_LEGEND = 9
FS_SUBJECT_TAG = 10
FS_ANNOT = 9

# 线宽 / 标记（V2 §16）
LW_DATA = 1.7
LW_ERRBAR = 1.2
CAPSIZE = 3
MARKER_SIZE = 7

# 零线（V2 §16）
LW_ZERO = 1.0
ZERO_LINE_COLOR = "0.25"
ZERO_LINE_STYLE = ":"

# 配对连接线（V2 §16：浅灰细线，不用架构色，避免与 CI 混淆）
CONNECTOR_COLOR = "0.75"
CONNECTOR_LW = 0.8

# Figure 5 主层/最终层上下错开量（V2 §15.5：避免同一 y 上两组 CI 重叠）
LAYER_OFFSET = 0.12

# 模型配色/点形/线型（方案 11.1）。RWKV 原用纯绿色（v1 的 #2ca02c），方案明确
# 建议避免纯绿以提高色觉友好性，v2 改用蓝绿色；其余沿用 v1 已验证的选择。
CORE_MODELS = ["pythia", "mamba", "rwkv"]
REFERENCE_MODEL = "awd_lstm"

MODEL_STYLE = {
    "pythia":   {"color": "#1f77b4", "linestyle": "-",  "marker": "o", "label": "Pythia"},
    "mamba":    {"color": "#E4572E", "linestyle": "-",  "marker": "s", "label": "Mamba"},
    "rwkv":     {"color": "#2A9D8F", "linestyle": "-",  "marker": "^", "label": "RWKV"},
    "awd_lstm": {"color": "#888888", "linestyle": "--", "marker": "x", "label": "AWD-LSTM"},
}

HPOS = {8: 0, 32: 1, 128: 2}   # 均匀间距的 H 轴位置（8/32/128 三点，非真数轴）


def style_axes(ax, *, grid_axis="y"):
    """统一坐标轴外观：仅保留浅灰水平网格，白底，无过粗边框。"""
    ax.set_facecolor("white")
    ax.grid(True, axis=grid_axis, alpha=0.25, linewidth=0.6, color="#999999")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=FS_TICK)


def safe_lim(ax, configured, data_values, *, axis="y", context=""):
    """跨被试固定坐标范围（方案11.2）的安全网。

    优先使用 configured 范围以保证跨被试可比（不逐被试自动缩放）；但如果当前
    被试的真实数据（点估计或 CI 边界）超出这个预先猜测的范围，**绝不静默裁掉
    数据**——那样会让数据点、误差棒甚至显著性标注直接从图上消失且没有任何
    提示（实测发现过：固定 ylim 偏窄时，Mamba 的误差棒被整段裁掉，连显著性
    星号都跟着消失，图看起来像"没有这个模型的数据"）。这里改为自动扩展并打印
    醒目警告，提示需要回头核对、统一调整其余被试的固定范围常量，而不是留着
    一个会吃掉数据的隐患。
    """
    vals = [v for v in data_values if v is not None]
    lo, hi = configured
    if not vals:
        (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)
        return (lo, hi)
    dmin, dmax = min(vals), max(vals)
    span = hi - lo
    margin = 0.06 * span
    new_lo = min(lo, dmin - margin) if dmin - margin < lo else lo
    new_hi = max(hi, dmax + margin) if dmax + margin > hi else hi
    if (new_lo, new_hi) != (lo, hi):
        print(f"[m6v2] ⚠️ {context} 数据超出预设固定坐标范围 ({lo:.4f},{hi:.4f})，"
              f"已自动扩展为 ({new_lo:.4f},{new_hi:.4f}) 以避免裁掉数据/标注——"
              f"建议核对是否需要同步调整其余被试的固定范围常量，保持跨被试可比",
              flush=True)
    (ax.set_ylim if axis == "y" else ax.set_xlim)(new_lo, new_hi)
    return (new_lo, new_hi)


def subject_tag(ax_or_fig, subject: str, *, x=0.99, y=0.99):
    """右上角小号粗体被试编号（方案 2.3：被试编号不进大标题，单独显示）。"""
    ax_or_fig.text(x, y, subject, transform=(
        ax_or_fig.transAxes if hasattr(ax_or_fig, "transAxes") else ax_or_fig.transFigure),
        ha="right", va="top", fontsize=FS_SUBJECT_TAG, fontweight="bold")


def fmt_p(p: float | None) -> str:
    """原始 bootstrap p 值格式化。

    与本项目 bootstrap 分辨率（n_boot=1000 → 最小 1/1000）对齐，写 p<0.001 而不是
    p<0.0001（后者暗示了 1000 次重抽样给不出的精度）。

    **刻意不加 "Holm-adjusted" 前缀**：src/stats/bootstrap.py::holm_bonferroni 返回的
    `p` 是原始双尾 bootstrap p 值，不是 Holm 校正后的 p；Holm 的多重比较结论由
    `reject` 字段（对应图上的星号）承载。把原始 p 标成 "Holm-adjusted" 是错误的，
    会误导读者。图上如实标原始 p，星号单独表示"经 Holm α=0.05 校正后拒绝 H0"，
    两者不冗余（Holm 是 step-down，光看原始 p 推不出 reject）。"""
    if p is None:
        return ""
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def annotate_with_headroom(ax, items, *, dy_points=6, fontsize=FS_ANNOT):
    """在 items=[(x,y,text,color), ...] 处标注文字，并按标注**渲染后的真实像素
    范围**扩展 ylim，避免文字被图框/画布边界裁掉。

    不是凭经验猜一个 headroom 比例（如 '+15%'）——那种做法在文字更长、字号更大
    或多条标注堆叠时仍可能不够，且没有任何机制会提醒你它不够。这里在
    fig.canvas.draw() 之后用 get_window_extent() 量出每条标注实际占多少像素，
    转换回数据坐标，只在真的超出当前 ylim 时才扩，扩多少由实际文字高度决定。
    """
    texts = []
    for x, y, txt, color in items:
        if not txt:
            continue
        t = ax.annotate(txt, (x, y), textcoords="offset points",
                        xytext=(0, dy_points), ha="center",
                        fontsize=fontsize, color=color, clip_on=False)
        texts.append(t)
    if not texts:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ylo, yhi = ax.get_ylim()
    max_top = yhi
    inv = ax.transData.inverted()
    for t in texts:
        bbox = t.get_window_extent(renderer=renderer)
        _, top_data = inv.transform((bbox.x0, bbox.y1))
        max_top = max(max_top, top_data)
    if max_top > yhi:
        pad = 0.04 * (yhi - ylo if yhi > ylo else abs(yhi) or 1.0)
        ax.set_ylim(ylo, max_top + pad)


def annotate_with_headroom_h(ax, items, *, dx_points=6, fontsize=FS_ANNOT):
    """同 annotate_with_headroom，但用于横向森林图：标注加在数据点右侧，按需
    扩展 xlim（而不是 ylim）。items=[(x,y,text,color), ...]。"""
    texts = []
    for x, y, txt, color in items:
        if not txt:
            continue
        t = ax.annotate(txt, (x, y), textcoords="offset points",
                        xytext=(dx_points, 0), ha="left", va="center",
                        fontsize=fontsize, color=color, clip_on=False)
        texts.append(t)
    if not texts:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlo, xhi = ax.get_xlim()
    max_right = xhi
    inv = ax.transData.inverted()
    for t in texts:
        bbox = t.get_window_extent(renderer=renderer)
        right_data, _ = inv.transform((bbox.x1, bbox.y0))
        max_right = max(max_right, right_data)
    if max_right > xhi:
        pad = 0.04 * (xhi - xlo if xhi > xlo else abs(xhi) or 1.0)
        ax.set_xlim(xlo, max_right + pad)


def save_with_caption(fig, outdir, name: str, caption: str):
    """保存 PNG(150dpi速览)+PDF(矢量)，并把图注文字单独写一份 .caption.txt——
    图内只留必要文字，完整方法/统计说明进图注（方案第10节），且图注由生成图片
    的同一段代码产出，不会和实际画的内容脱节。"""
    from pathlib import Path
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300 if ext == "png" else None,
                   bbox_inches="tight", facecolor="white")
    with open(outdir / f"{name}.caption.txt", "w") as f:
        f.write(caption.strip() + "\n")
    plt.close(fig)
    print(f"[m6v2] 已保存 {name}.png / .pdf / .caption.txt", flush=True)
