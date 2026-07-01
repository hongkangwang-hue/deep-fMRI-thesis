"""
M5 —— 配对 story bootstrap 的统计核心（纯 numpy，无文件 IO，可本地单测）。

对应 frozen/config.yaml::statistics 与冻结文档里程碑5"设计思路"：
  - bootstrap = paired_story_within_fold，1000 次，种子 20260701。
  - 每次重抽：每个 outer fold 内对 story 有放回抽样，**同一次重抽的 story 索引
    复用到所有 model/H/layer/condition/ROI**（配对，保证差值 CI 正确）。
  - 聚合链：先 fold 内按有效 TR 数加权 Fisher-z（z 空间），再跨 fold 按 fold 有效
    TR 数加权（z 空间），最后 tanh 回 r。差值（Δr、架构差值）在 r 空间取。
  - 点估计 = 全样本（每 story 恰好一次）聚合，应精确等于 M4 的跨折 r。
  - CI = 1000 次 bootstrap 的 2.5/97.5 百分位（探索性未校正）。
  - 确认性家族：百分位法双尾 bootstrap p 值 + Holm 家族校正（α=0.05）。

本模块只管重抽机制与通用统计（聚合、CI、p 值、Holm）；具体算哪些估计量（registry
对比）在 estimands.py，两者由 scripts/m5_analysis.py 组装。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BootstrapData:
    """配对 bootstrap 的输入（story 级 z 标量 + 权重，已按 fold 对齐）。

    z 和 weights 的每个 per-fold 数组都按 fold_stories[fold] 的顺序对齐，故一次
    重抽产生的整数索引可同时索引所有 key 的 z 与共享的 weights。
    """
    folds: list[str]                              # canonical fold 顺序
    fold_stories: dict[str, list[str]]            # fold -> canonical story 顺序
    weights: dict[str, np.ndarray]                # fold -> <float>[n_story] 有效 TR 权重
    z: dict[tuple, dict[str, np.ndarray]]         # key -> fold -> <float>[n_story] z 标量

    def keys(self):
        return self.z.keys()


def aggregate_to_r(z_by_fold: dict[str, np.ndarray],
                   weights: dict[str, np.ndarray],
                   idx_by_fold: dict[str, np.ndarray]) -> float:
    """一个 (model,H,layer,condition,ROI) 的聚合 r：fold 内加权 z → 跨 fold 加权 z → tanh。

    z_by_fold / weights 每个数组按同一 story 顺序对齐；idx_by_fold 给出该次重抽在每个
    fold 的 story 行索引（点估计传 arange 即不重抽）。nan（空 ROI/无有效点）被跳过。
    """
    fold_z, fold_w = [], []
    for fold, idx in idx_by_fold.items():
        zz = z_by_fold[fold][idx]
        ww = weights[fold][idx]
        m = np.isfinite(zz) & np.isfinite(ww)
        wsum = ww[m].sum()
        if not m.any() or wsum == 0:
            continue
        fold_z.append((zz[m] * ww[m]).sum() / wsum)      # fold 内 z 空间加权
        fold_w.append(wsum)                               # 该 fold 本次重抽的有效 TR 和
    if not fold_z:
        return float("nan")
    fz = np.asarray(fold_z)
    fw = np.asarray(fold_w)
    agg_z = (fz * fw).sum() / fw.sum()                    # 跨 fold z 空间加权
    return float(np.tanh(agg_z))


def _identity_idx(fold_stories: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """点估计用：每 fold 每 story 恰好一次（arange），复现冻结聚合规则。"""
    return {f: np.arange(len(ss)) for f, ss in fold_stories.items()}


def paired_bootstrap(
    data: BootstrapData,
    estimand_fn: Callable[[dict[tuple, float]], dict[str, float]],
    n_boot: int, seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """配对 story bootstrap。返回 (点估计 estimands, 每次重抽 estimands 列表)。

    每次重抽：每 fold 抽一组共享 story 索引 → 对所有 key 聚合成 r 表 → estimand_fn 从
    r 表算出该次的全部命名估计量。所有 key 共用同一组索引即"配对"。
    """
    rng = np.random.default_rng(seed)

    # 点估计：不重抽
    id_idx = _identity_idx(data.fold_stories)
    point_r = {k: aggregate_to_r(data.z[k], data.weights, id_idx) for k in data.keys()}
    point_est = estimand_fn(point_r)

    draws = []
    for _ in range(n_boot):
        idx = {f: rng.integers(0, len(data.fold_stories[f]), len(data.fold_stories[f]))
               for f in data.folds}
        r_tab = {k: aggregate_to_r(data.z[k], data.weights, idx) for k in data.keys()}
        draws.append(estimand_fn(r_tab))
    return point_est, draws


def draws_to_arrays(draws: list[dict[str, float]]) -> dict[str, np.ndarray]:
    """把每次重抽的 estimand dict 列表转成 {name: <float>[n_boot]}。"""
    names = draws[0].keys()
    return {n: np.asarray([d[n] for d in draws], dtype=np.float64) for n in names}


def percentile_ci(values: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    """百分位 CI（忽略 nan）。"""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)))


def bootstrap_two_sided_p(values: np.ndarray) -> float:
    """百分位法双尾 bootstrap p 值：p = 2·min(P(θ*≤0), P(θ*≥0))，截断到 [0,1]。

    直接数 bootstrap 分布落在 0 两侧的比例，与百分位 CI 同源、无额外分布假设
    （冻结文档 registry 只写 holm_bootstrap_pvalues，此为已确认的具体定义）。
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")
    p_le = float((v <= 0).mean())
    p_ge = float((v >= 0).mean())
    return min(1.0, 2.0 * min(p_le, p_ge))


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm 逐步降家族校正。返回 {name: {p, holm_threshold, reject}}。

    按 p 升序，第 i 小（0-based）阈值 = alpha/(m-i)；一旦某项不拒绝，其后全部不拒绝
    （step-down）。忽略 nan p（记为不拒绝）。
    """
    items = sorted(pvals.items(), key=lambda kv: (np.inf if not np.isfinite(kv[1]) else kv[1]))
    m = len(items)
    out = {}
    failed = False
    for i, (name, p) in enumerate(items):
        thresh = alpha / (m - i)
        rej = np.isfinite(p) and (p <= thresh) and not failed
        if not rej:
            failed = True
        out[name] = {"p": float(p) if np.isfinite(p) else float("nan"),
                     "holm_threshold": float(thresh), "reject": bool(rej)}
    return out
