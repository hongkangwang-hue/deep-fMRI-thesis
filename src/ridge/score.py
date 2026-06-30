"""
M3/M2-C 评分工具 —— 纯函数，无重计算、无文件 IO，可本地单测。

对应 frozen/analysis_spec.yaml::scoring：
  voxel_metric: pearson_r
  transform_before_roi_average: fisher_z
  fold_summary: effective_tr_weighted_mean
"""

from __future__ import annotations

import numpy as np


def voxelwise_pearson(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """逐体素（逐列）Pearson r。

    Args:
        pred:   <float>[T, V] 预测响应。
        actual: <float>[T, V] 实际响应。

    Returns:
        <float>[V] 每列的 Pearson r；零方差列返回 0（nan→0）。
    """
    if pred.shape != actual.shape:
        raise ValueError(f"形状不一致 pred{pred.shape} != actual{actual.shape}")
    p = pred - pred.mean(0)
    a = actual - actual.mean(0)
    num = (p * a).sum(0)
    denom = np.sqrt((p ** 2).sum(0) * (a ** 2).sum(0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / denom
    return np.nan_to_num(r)


def fisher_z(r: np.ndarray) -> np.ndarray:
    """Fisher z = arctanh(r)，数值安全地裁剪 |r|→1 边界。"""
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def fisher_z_inv(z: np.ndarray) -> np.ndarray:
    """Fisher 逆变换 tanh(z)。"""
    return np.tanh(z)


def roi_mean_r(voxel_r: np.ndarray, roi_columns: np.ndarray) -> float:
    """ROI 平均 r：先 fisher-z 再均值再逆变换（spec: transform_before_roi_average=fisher_z）。

    Args:
        voxel_r:     <float>[V] 全体素 r。
        roi_columns: <int>[k] 该 ROI 的列索引（进入统一 voxel mask 后的列号）。

    Returns:
        ROI 的代表 r（标量）。
    """
    if len(roi_columns) == 0:
        return float("nan")
    z = fisher_z(voxel_r[roi_columns])
    return float(fisher_z_inv(z.mean()))


def effective_tr_weighted_mean(
    fold_voxel_r: list[np.ndarray], fold_weights: list[int]
) -> np.ndarray:
    """跨折按有效 TR 数加权平均每体素 r（spec: fold_summary=effective_tr_weighted_mean）。

    Args:
        fold_voxel_r: 每折的 <float>[V] 体素 r 列表。
        fold_weights: 每折测试集的有效评分 TR 数（权重）。

    Returns:
        <float>[V] 加权平均 r。
    """
    if len(fold_voxel_r) != len(fold_weights):
        raise ValueError("折数与权重数不一致")
    R = np.stack(fold_voxel_r, axis=0)            # (n_folds, V)
    w = np.asarray(fold_weights, dtype=np.float64)
    if w.sum() == 0:
        raise ValueError("权重和为 0")
    return (R * w[:, None]).sum(0) / w.sum()
