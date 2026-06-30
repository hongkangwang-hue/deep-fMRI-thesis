"""
ROI 提取：aparc.a2009s 表面分区 → BOLD 95556 列空间（主导顶点归属，已冻结规则）。

索引链（已在 M2-B 验证）：
  1) BOLD 95556 列 = thick mask（cortex.db.get_mask(subj,xfm,'thick')）的 True 体素，
     列顺序 = thick mask 在 (54,84,84) 上的 C-order；
  2) aparc.a2009s annot 顶点数 == pycortex surface 顶点数 → label 按顶点索引直接对应；
  3) pycortex line_nearest mapper 的 masks[hemi] 形状 (n_vertices, 381024=54·84·84)，
     把顶点映到全 volume 体素（列空间与 thick mask 同为 (54,84,84) C-order）。

归属规则（冻结＝B 主导顶点）：每个 thick 体素沿 line_nearest 贡献最大的顶点若属于
某 ROI 的 aparc label 集合，则该体素归入该 ROI。再与统一 BOLD-only voxel mask 求交，
要求每 ROI 体素数 ≥ min_voxels。

本模块核心为纯函数（依赖注入 mapper masks 与 annot labels），便于单测；pycortex 实际
加载在 scripts/m2_validate_pipeline.py 的 roi step。
"""

from __future__ import annotations

import numpy as np


def per_voxel_dominant_label(mask_csc, vertex_labels: np.ndarray):
    """单半球：每个全 volume 体素（列）沿 line 贡献最大的顶点的 aparc label。

    Args:
        mask_csc:       scipy.csc_matrix (n_vertices, n_full_voxels)，mapper.masks[hemi]。
        vertex_labels:  <int>[n_vertices] 每顶点 aparc label id（annot 值）。

    Returns:
        (best_weight <float>[n_full_voxels], best_label <int>[n_full_voxels])
        无任何顶点贡献的体素 best_weight=0, best_label=-1。
    """
    nvox = mask_csc.shape[1]
    indptr, indices, data = mask_csc.indptr, mask_csc.indices, mask_csc.data
    best_w = np.zeros(nvox, dtype=np.float64)
    best_lab = np.full(nvox, -1, dtype=np.int64)
    for c in range(nvox):
        s, e = indptr[c], indptr[c + 1]
        if e > s:
            k = s + int(np.argmax(data[s:e]))
            best_w[c] = data[k]
            best_lab[c] = vertex_labels[indices[k]]
    return best_w, best_lab


def combine_hemis_dominant(w_lh, lab_lh, names_lh, w_rh, lab_rh, names_rh):
    """跨左右半球取主导：每个全 volume 体素选 line 贡献最大的那一侧的顶点 label 名。

    Returns:
        dom_name <object>[n_full_voxels]：体素主导 aparc label 名（None 表示无贡献），
        dom_hemi <object>[n_full_voxels]：'lh'/'rh'/None。
    """
    nvox = len(w_lh)
    dom_name = np.full(nvox, None, dtype=object)
    dom_hemi = np.full(nvox, None, dtype=object)
    lh_wins = (w_lh >= w_rh) & (w_lh > 0)
    rh_wins = (w_rh > w_lh) & (w_rh > 0)
    for c in np.flatnonzero(lh_wins):
        dom_name[c] = names_lh[lab_lh[c]] if lab_lh[c] >= 0 else None
        dom_hemi[c] = "lh"
    for c in np.flatnonzero(rh_wins):
        dom_name[c] = names_rh[lab_rh[c]] if lab_rh[c] >= 0 else None
        dom_hemi[c] = "rh"
    return dom_name, dom_hemi


def roi_full_voxels(dom_name, dom_hemi, label_names: list[str], hemi: str):
    """按归属规则选出属于某 ROI 的全 volume 体素索引。

    Args:
        label_names: 该 ROI 的 aparc label 名集合。
        hemi:        'L' / 'R' / 'LR'，限定主导半球（双侧 PT 用 'LR'）。

    Returns:
        <int> 全 volume 体素索引数组。
    """
    want = set(label_names)
    hemi_ok = {"L": {"lh"}, "R": {"rh"}, "LR": {"lh", "rh"}}[hemi]
    sel = [c for c in range(len(dom_name))
           if dom_name[c] in want and dom_hemi[c] in hemi_ok]
    return np.array(sel, dtype=np.int64)


def full_to_column_map(thick_mask_flat: np.ndarray) -> np.ndarray:
    """thick mask（C-order 展平 bool，长度 381024）→ 全 volume idx 到 BOLD 列号的映射。

    非 thick 体素映射为 -1。BOLD 列号 = thick True 的 C-order 次序。
    """
    full_to_col = np.full(thick_mask_flat.size, -1, dtype=np.int64)
    full_to_col[thick_mask_flat] = np.arange(int(thick_mask_flat.sum()))
    return full_to_col


def columns_for_roi(full_voxels: np.ndarray, full_to_col: np.ndarray,
                    voxel_mask_keep: np.ndarray | None = None) -> np.ndarray:
    """全 volume 体素 → BOLD 列号，剔除非 thick，并与统一 voxel mask 求交。

    Args:
        full_voxels:     ROI 的全 volume 体素索引。
        full_to_col:     full_to_column_map 输出。
        voxel_mask_keep: 可选统一 BOLD-only 保留列索引（frozen voxel mask）。

    Returns:
        排序去重后的 BOLD 列号。
    """
    cols = full_to_col[full_voxels]
    cols = cols[cols >= 0]                       # 仅 thick 内
    cols = np.unique(cols)
    if voxel_mask_keep is not None:
        cols = np.intersect1d(cols, voxel_mask_keep, assume_unique=True)
    return cols