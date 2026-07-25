"""§2.3 OOD 退化的 H 依赖性诊断——纯函数（零算力，可本地单测）。

## 这个诊断要回答什么

lag-1 已经显示「打乱后表示变呆」，但那是间接证据。§2.3 直接测**表示本身的几何
性质**是否随 H 塌缩，因为规范里有一条关键论断：

  若 OOD 退化程度本身随 H 累积（128 个乱序词比 8 个"更不像自然语言"），则退化是
  H 依赖的，会直接混入 Δr_total^ctx1，**配对差值 D_m 与交互量 I_MP 都消除不掉它**。

核心比较不是绝对水平，而是**变化幅度之差**：
    ctx1 在 H8→H128 的变化   vs   normal 在 H8→H128 的变化

## 指标主次（V6.4.2 §2.3 冻结，不得据结果调换）

主指标（表示层退化）：
  1. 目标位置隐表示的 L2 范数中位数；
  2. PCA 前特征矩阵的有效维度（participation ratio），同时记录 evr@100。

辅助指标：目标词 NLL——**只作辅证**，因为它同时受「上下文信息量下降」（本控制
想要的效果）与「输入 OOD 导致模型退化」（混淆项）影响，无法区分二者。本模块
不含 NLL（需 GPU 重跑模型），若不执行须在 Limitations 注明。

## 判读规则（V6.4.2 §2.3 冻结）

  - ctx1 的 H 依赖变化幅度与 normal 相当（尤其两项主指标）→ 可写"未发现 OOD
    退化随 H 系统性加剧的证据"；
  - ctx1 下主指标变化明显更大 → 必须写明 D_m 与 I_MP 受此混淆影响，结论下调为
    "方向性提示"；
  - 仅 NLL 更大而两主指标相当 → 弱化版处理；
  - **无论哪种结果都必须报告**，不得只在有利时呈现。
"""

from __future__ import annotations

import numpy as np


# ── 单故事指标 ────────────────────────────────────────────────────────────

def l2_norm_median(H_word: np.ndarray) -> float:
    """主指标1：目标位置隐表示的 L2 范数中位数。

    H_word: <float>[n_targets, hidden]，特征缓存里的逐目标词隐表示（Lanczos 之前）。
    §2.3 说的「目标位置隐表示」就是这个，不是 TR 级下采样后的。
    """
    if H_word.size == 0:
        return float("nan")
    return float(np.median(np.linalg.norm(H_word, axis=1)))


def participation_ratio(X: np.ndarray) -> float:
    """主指标2：有效维度（participation ratio）。

    PR = (Σλ)² / Σλ²，λ 为特征协方差的特征值。直观含义：方差均匀摊在 d 个方向上
    时 PR≈d；方差全挤进 1 个方向时 PR≈1。因此 PR 下降 = 表示挤进更少方向、
    信息容量下降，正是「退化」的几何刻画。

    X: <float>[T, hidden]，**PCA 前**的特征矩阵（与 pipeline 一致：Lanczos 之后、
    FIR 之前的 TR 级特征）。
    """
    if X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    # 用奇异值平方代替显式协方差特征值（数值更稳，且避免 768×768 矩阵）
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    denom = float(np.sum(lam ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(lam) ** 2 / denom)


def evr_at_k(X: np.ndarray, k: int = 100) -> float:
    """同时记录的 evr@k：前 k 个主成分累计解释方差比。

    与 pipeline 的 PCA(K=100) 对应。注意主 M4 算了 evr_at_k 但从未落盘
    （V6.4.2 §10 记录的 PCA 缺口），这里对两条件一致地补上。
    """
    if X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    total = float(np.sum(lam))
    if total <= 0:
        return float("nan")
    return float(np.sum(lam[:k]) / total)


# ── H 依赖性比较（本诊断的核心） ──────────────────────────────────────────

def h_change_by_story(metric_h8: dict[str, float],
                      metric_h128: dict[str, float]) -> dict:
    """逐故事配对的 H 依赖变化：metric(H128) − metric(H8)。

    配对到故事级（两个 H 看的是同一批故事）可消除故事长度/语速/体裁异质性，
    与 D_m、Δlag1 的配对逻辑一致。
    """
    stories = sorted(set(metric_h8) & set(metric_h128))
    pairs = {s: metric_h128[s] - metric_h8[s] for s in stories}
    d = np.array([v for v in pairs.values() if np.isfinite(v)], dtype=float)
    if d.size == 0:
        return {"median": float("nan"), "p2_5": float("nan"), "p97_5": float("nan"),
                "n_stories": 0, "by_story": {}}
    return {
        "median": float(np.median(d)),
        "p2_5": float(np.percentile(d, 2.5)),
        "p97_5": float(np.percentile(d, 97.5)),
        "n_stories": int(d.size),
        "by_story": pairs,
    }


def ood_h_dependence(change_normal: dict, change_ctx1: dict) -> dict:
    """§2.3 的核心量：ctx1 的 H 依赖变化幅度 **减去** normal 的对应变化。

    逐故事配对相减（同一故事在两条件下都有 H8→H128 的变化），因此这是一个
    difference-in-differences：
        [metric_ctx1(H128) − metric_ctx1(H8)] − [metric_normal(H128) − metric_normal(H8)]

    该量显著偏离 0，即「ctx1 下退化随 H 加剧的幅度与 normal 不同」→ 退化是
    H 依赖的 → D_m / I_MP 受其混淆且配对无法消除。
    """
    p_n = change_normal.get("by_story", {})
    p_c = change_ctx1.get("by_story", {})
    stories = sorted(set(p_n) & set(p_c))
    diff = np.array([p_c[s] - p_n[s] for s in stories], dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {"median": float("nan"), "p2_5": float("nan"), "p97_5": float("nan"),
                "n_stories": 0, "ci_excludes_zero": False}
    lo = float(np.percentile(diff, 2.5))
    hi = float(np.percentile(diff, 97.5))
    return {
        "median": float(np.median(diff)),
        "p2_5": lo, "p97_5": hi,
        "n_stories": int(diff.size),
        "ci_excludes_zero": bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0)),
        "reading": ("偏离0 = ctx1 下该指标随 H 的变化幅度与 normal 不同 → "
                    "退化具 H 依赖性 → D_m / I_MP 受混淆，配对不能消除"),
    }


def verdict(ood_by_metric: dict) -> dict:
    """按 §2.3 冻结判读规则给出结论（只看两项主指标；NLL 本轮未做）。"""
    main = ["l2_norm_median", "participation_ratio"]
    flagged = [m for m in main
               if ood_by_metric.get(m, {}).get("ci_excludes_zero", False)]
    if not flagged:
        concl = "no_evidence_of_H_dependent_degradation"
        text = ("两项主指标的 H 依赖变化在 ctx1 与 normal 之间无显著差异 → "
                "可写明「未发现 OOD 退化随 H 系统性加剧的证据」。")
    else:
        concl = "H_dependent_degradation_detected"
        text = (f"主指标 {flagged} 显示 ctx1 下的 H 依赖变化显著不同于 normal → "
                "必须在 Results 与 Limitations 写明 D_m 与 I_MP 受此混淆影响，"
                "并把相关结论下调为「方向性提示」。")
    return {
        "conclusion": concl,
        "flagged_main_metrics": flagged,
        "required_action": text,
        "nll_auxiliary_status": "not_computed_this_round（需 GPU；未做须在 Limitations 注明）",
    }
