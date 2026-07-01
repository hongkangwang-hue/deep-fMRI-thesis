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


def roi_mean_fisherz(voxel_r: np.ndarray, roi_columns: np.ndarray) -> float:
    """ROI 内体素 r 的 fisher-z 平均，返回 **z 空间**标量。

    所有跨 story/fold 的 ROI 加权都在 z 空间做（相关系数平均的标准做法），最后
    才 tanh 回 r 展示。故这里返回 z，而非 r。空 ROI 返回 nan。
    """
    if len(roi_columns) == 0:
        return float("nan")
    return float(fisher_z(voxel_r[roi_columns]).mean())


def roi_mean_r(voxel_r: np.ndarray, roi_columns: np.ndarray) -> float:
    """ROI 平均 r：fisher-z→均值→逆变换（spec: transform_before_roi_average=fisher_z）。

    = tanh(roi_mean_fisherz(...))。用于直接展示单个 voxel_r 向量的 ROI 标量。
    """
    return float(fisher_z_inv(roi_mean_fisherz(voxel_r, roi_columns)))


def weighted_mean_scalar(values: list[float], weights: list[float]) -> float:
    """按权重对标量列表加权平均（忽略 nan 值及其权重）。用于跨 story/fold 汇总 ROI z。"""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(v)
    if not ok.any() or w[ok].sum() == 0:
        return float("nan")
    return float((v[ok] * w[ok]).sum() / w[ok].sum())


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
