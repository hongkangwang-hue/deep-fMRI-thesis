"""
M6 补充图 —— ROI 解剖位置 flatmap（Figure 6）。

把冻结的 left_IFG / bilateral_PT 列（frozen/roi_columns_UTS03.npz）画回 UTS03 皮层
展开图，让读者/答辩看到两个 ROI 在脑上的实际位置。**只呈现、不重算**：用的就是编码
分析里实际使用的那 768 / 341 个体素编号，换一种可视化方式，不引入任何新对比/统计
（符合里程碑6"明确不做"）。

空间索引链（与 M2 src/fmri/roi.py 完全一致，反向用）：
  BOLD 列空间(95556) --col_to_full--> 全 volume (54,84,84) C-order
  col_to_full = np.flatnonzero(thick_mask.ravel('C'))，即 roi.py::full_to_column_map 的逆。
  全 volume -> 展开皮层表面：交给 pycortex quickflat（其看家本领，不自行计算）。

依赖：pycortex 1.3.2 + 本地 pycortex-db（flatmask/flatverts/surface-info 缓存需已
`git annex get`；本机已补齐）。纯 CPU 渲染，不用 GPU/服务器。

用法：python3 scripts/m6_roi_location.py
输出：figures/<subject>/fig6_roi_location.png / .pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")                         # 无显示环境，非交互后端
import matplotlib.pyplot as plt               # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch          # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config     # noqa: E402

# ROI 展示配色（与模型配色区分开——这里是脑区不是模型）
ROI_STYLE = {
    "left_IFG":     {"value": 1.0, "color": "#1f77b4", "label": "Left IFG (primary)"},
    "bilateral_PT": {"value": 2.0, "color": "#e8710a", "label": "Bilateral PT (reference)"},
}


def build_roi_volume(roi_cols: dict, subject: str, xfm: str):
    """把 ROI 的 BOLD 列填回全 volume（IFG=1, PT=2, 其余 nan）。返回 (data3d, counts)。"""
    import cortex
    mask = np.asarray(cortex.db.get_mask(subject, xfm, "thick"))     # (54,84,84) bool
    col_to_full = np.flatnonzero(mask.ravel(order="C"))              # 列号 -> 全 volume 平铺idx
    if len(col_to_full) != 95556:
        raise ValueError(f"thick mask True 体素数 {len(col_to_full)} != 95556（与 M2 冻结不符）")

    flat = np.full(mask.size, np.nan)
    counts = {}
    for name, st in ROI_STYLE.items():
        cols = roi_cols[name]
        flat[col_to_full[cols]] = st["value"]
        counts[name] = len(cols)
    data3d = flat.reshape(mask.shape)

    # 健全性闸门：渲染用的体素数必须精确等于冻结列数，否则索引链接错了
    for name, st in ROI_STYLE.items():
        n_set = int(np.sum(data3d == st["value"]))
        if n_set != counts[name]:
            raise ValueError(f"[{name}] volume 内标记体素 {n_set} != 冻结列数 {counts[name]}"
                             "（col->full 索引链错位）")
    return data3d, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--xfm", default="UTS03_auto")
    args = ap.parse_args()

    import cortex
    cfg = load_config()
    roi_path = Path(cfg["paths"]["frozen_dir"]) / f"roi_columns_{args.subject}.npz"
    roi_cols = dict(np.load(roi_path))
    for name in ROI_STYLE:
        if name not in roi_cols:
            raise SystemExit(f"{roi_path} 缺少 ROI {name}")

    data3d, counts = build_roi_volume(roi_cols, args.subject, args.xfm)
    print(f"[m6] ROI 体素数校验通过：left_IFG={counts['left_IFG']} "
          f"bilateral_PT={counts['bilateral_PT']}", flush=True)

    cmap = ListedColormap([ROI_STYLE["left_IFG"]["color"], ROI_STYLE["bilateral_PT"]["color"]])
    vol = cortex.Volume(data3d, args.subject, args.xfm, vmin=1, vmax=2, cmap=cmap)
    # with_rois=False：只画我们自己(主导顶点规则)的 ROI，不叠加 pycortex 内置解剖轮廓
    fig = cortex.quickflat.make_figure(vol, with_curvature=True, with_colorbar=False,
                                       with_rois=False)

    ax = fig.axes[0]
    handles = [Patch(facecolor=st["color"], edgecolor="k",
                     label=f"{st['label']} — {counts[name]} voxels")
               for name, st in ROI_STYLE.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)
    fig.suptitle(f"Figure 6  ROI locations on {args.subject} cortical flatmap "
                 "(dominant-vertex assignment)", fontsize=12, fontweight="bold")

    outdir = Path(cfg["paths"]["figures_dir"]) / args.subject
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig6_roi_location.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[m6] 已保存 fig6_roi_location.png / .pdf → {outdir}", flush=True)


if __name__ == "__main__":
    main()
