"""
M2-B ROI 归属纯逻辑单测（合成 mapper + labels，不依赖 pycortex/数据）。

验证主导顶点规则、跨半球取主导、thick→列映射与 voxel mask 求交。
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.fmri.roi import (
    per_voxel_dominant_label, combine_hemis_dominant, roi_full_voxels,
    full_to_column_map, columns_for_roi)


def test_per_voxel_dominant_picks_max_weight_vertex():
    # 3 顶点 × 4 体素；体素0 顶点1(权重0.9)主导，体素1 顶点0(0.5)，体素2 无贡献，体素3 顶点2(0.7)
    M = sp.csc_matrix(np.array([
        [0.1, 0.5, 0.0, 0.0],   # vertex 0
        [0.9, 0.2, 0.0, 0.0],   # vertex 1
        [0.0, 0.0, 0.0, 0.7],   # vertex 2
    ]))
    labels = np.array([10, 11, 12])
    w, lab = per_voxel_dominant_label(M, labels)
    assert lab.tolist() == [11, 10, -1, 12]
    assert np.isclose(w[0], 0.9) and w[2] == 0.0


def test_combine_hemis_dominant_prefers_larger_weight():
    w_lh = np.array([0.8, 0.1, 0.0])
    lab_lh = np.array([0, 1, -1])
    w_rh = np.array([0.2, 0.9, 0.0])
    lab_rh = np.array([0, 0, -1])
    names_lh = ["IFG_a", "OTHER"]
    names_rh = ["PT"]
    dom_name, dom_hemi = combine_hemis_dominant(
        w_lh, lab_lh, names_lh, w_rh, lab_rh, names_rh)
    # 体素0 lh 赢(0.8>0.2) → IFG_a; 体素1 rh 赢(0.9>0.1) → PT; 体素2 无
    assert dom_name.tolist() == ["IFG_a", "PT", None]
    assert dom_hemi.tolist() == ["lh", "rh", None]


def test_roi_full_voxels_filters_label_and_hemi():
    dom_name = np.array(["IFG_a", "IFG_b", "PT", "OTHER", "IFG_a"], dtype=object)
    dom_hemi = np.array(["lh", "lh", "rh", "lh", "rh"], dtype=object)
    # IFG = {IFG_a, IFG_b}，限定左半球 → 体素 0,1（体素4 是 IFG_a 但在 rh，排除）
    ifg = roi_full_voxels(dom_name, dom_hemi, ["IFG_a", "IFG_b"], "L")
    assert ifg.tolist() == [0, 1]
    # PT 双侧
    pt = roi_full_voxels(dom_name, dom_hemi, ["PT"], "LR")
    assert pt.tolist() == [2]


def test_full_to_column_map_c_order():
    # thick mask 长度 6，True 在位置 1,3,4 → 列号 0,1,2
    flat = np.array([0, 1, 0, 1, 1, 0], dtype=bool)
    f2c = full_to_column_map(flat)
    assert f2c.tolist() == [-1, 0, -1, 1, 2, -1]


def test_columns_for_roi_intersects_voxel_mask():
    flat = np.array([1, 1, 1, 1, 1, 1], dtype=bool)  # 全 thick → 列 0..5
    f2c = full_to_column_map(flat)
    full_voxels = np.array([0, 2, 4, 4])             # 含重复
    keep = np.array([0, 4])                          # 统一 mask 只留列 0,4
    cols = columns_for_roi(full_voxels, f2c, voxel_mask_keep=keep)
    assert cols.tolist() == [0, 4]


def test_columns_for_roi_drops_non_thick():
    flat = np.array([1, 0, 1, 0, 1], dtype=bool)     # 列: idx0→0, idx2→1, idx4→2
    f2c = full_to_column_map(flat)
    full_voxels = np.array([0, 1, 2])                # idx1 非 thick，丢弃
    cols = columns_for_roi(full_voxels, f2c)
    assert cols.tolist() == [0, 1]