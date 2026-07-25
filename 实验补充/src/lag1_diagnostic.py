"""§2.2 lag-1 平滑度诊断（纯函数，零算力，可本地单测）。

## 这个诊断在本项目里要回答什么

原规范 V6.4.2 §2.2 设计 lag-1 时，目的偏向「查实现对不对」。但 Step 5 的真实
结果把问题变了：

  - Pythia 的 Δr_total 在 ctx1 下**翻成负数**（三被试均显著）；
  - I_MP = D_mamba − D_pythia **三被试均为负**（UTS01/02 显著）。

于是最要紧的问题不再是「实现对不对」（那已由 §2.1 的 443,286 对窗口断言 +
§2.4 与理论量级吻合证明了），而是：

  **打乱上下文后掉的特征平滑度，Pythia 是不是比 Mamba 掉得多？**

若是，则「Pythia 摔得更狠」可能只是特征时间统计变化更大所致，与「Pythia 更
依赖语言结构」无关——那 I_MP 为负就是被混淆的。这是本模块存在的真正理由。

## 哪些照搬、哪些按实况调整（诚实分界）

本模块实现于 D_m/I_MP 结果**已知之后**。因此凡是「可能被结果影响的选择」一律
照搬 V6.4.2 §2.2 冻结原文，不留可调余地：

  - 主指标 = 相邻 TR 表示的余弦 cos(x_t, x_{t-1})，每故事取中位数，计算位置固定
    在 Lanczos 之后、FIR 之前；featurewise 相关仅作补充，**不得**据结果择优；
  - 容差带（判异常）= normal story-level 值自身分布的 median/SD/P2.5/P97.5；
  - 精度区间（仅描述，**不得**当阈值）= normal 中位数的 story bootstrap；
  - 主报告量 = 逐故事配对差 Δlag1 = lag1_normal − lag1_ctx1；
  - 审计触发三规则与 k = 1.0。

按项目实况调整的部分（与「结果好不好看」无关，只是对齐已跑过什么）：

  - 只覆盖 H ∈ {8, 128} 与主层——ctx1 实际只跑了这些；
  - 删去 C2 相关内容——C2 从未执行；
  - 新增 `compare_across_models` / `h_dependence`：规范里没有，是上面那个新问题
    逼出来的**事后描述性**分析，输出里显式标记 post_hoc=True，论文须如实说明。
"""

from __future__ import annotations

import numpy as np

K_AUDIT = 1.0            # V6.4.2 §2.2 冻结阈值（约定值，非统计判据）
OUTLIER_SD = 3.0         # 规则3：偏离自身中位数 3 SD
N_BOOT_PRECISION = 1000  # 精度区间 bootstrap 次数（仅描述用）


# ── 单故事指标 ────────────────────────────────────────────────────────────

def lag1_cosine_median(X: np.ndarray) -> float:
    """主指标：相邻 TR 表示余弦相似度的故事内中位数。

    X: <float>[T, hidden]，Lanczos 下采样后、FIR 展开前的 TR 级特征。
    零范数行跳过而非计 0——余弦对 0 向量无定义，计 0 会人为压低中位数。
    """
    if X.shape[0] < 2:
        return float("nan")
    a, b = X[1:], X[:-1]
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 0
    if not ok.any():
        return float("nan")
    return float(np.median(np.sum(a * b, axis=1)[ok] / den[ok]))


def lag1_featurewise_corr_mean(X: np.ndarray) -> float:
    """补充指标：逐特征维 lag-1 自相关后跨维取均值。

    仅用于确认主指标结论不是余弦这一种度量的假象；**主指标恒为余弦**
    （V6.4.2 §2.2：不得根据结果在两者间择优）。
    """
    if X.shape[0] < 3:
        return float("nan")
    a, b = X[1:], X[:-1]
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    den = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    ok = den > 0
    if not ok.any():
        return float("nan")
    return float(np.mean(np.sum(a * b, axis=0)[ok] / den[ok]))


# ── 规范冻结的参考量与审计 ────────────────────────────────────────────────

def tolerance_band(normal_values: np.ndarray) -> dict:
    """容差带：normal story-level 值自身分布（**判异常用**）。"""
    v = np.asarray(normal_values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"median": float("nan"), "sd": float("nan"),
                "p2_5": float("nan"), "p97_5": float("nan"), "n_stories": 0}
    return {
        "median": float(np.median(v)),
        "sd": float(np.std(v, ddof=1)) if v.size > 1 else float("nan"),
        "p2_5": float(np.percentile(v, 2.5)),
        "p97_5": float(np.percentile(v, 97.5)),
        "n_stories": int(v.size),
    }


def precision_interval(normal_values: np.ndarray, seed: int,
                       n_boot: int = N_BOOT_PRECISION) -> dict:
    """精度区间：normal 中位数的 story bootstrap P2.5/P97.5。

    **仅说明 normal 中位数估得多准，绝不可当异常判定阈值**——V6.4.2 §2.2 专门
    纠正过 V6.4.1 的这个误用（该区间随故事数收窄，与预期中的真实位移尺度不匹配，
    会让审计标记必然触发进而形同虚设）。
    """
    v = np.asarray(normal_values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"median_ci_lo": float("nan"), "median_ci_hi": float("nan"),
                "n_boot": n_boot, "usage": "descriptive_only_never_a_threshold"}
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(v, size=v.size, replace=True)) for _ in range(n_boot)]
    return {
        "median_ci_lo": float(np.percentile(meds, 2.5)),
        "median_ci_hi": float(np.percentile(meds, 97.5)),
        "n_boot": n_boot,
        "usage": "descriptive_only_never_a_threshold",
    }


def paired_delta(normal_by_story: dict[str, float],
                 ctx1_by_story: dict[str, float]) -> dict:
    """主报告量：Δlag1(story) = lag1_normal − lag1_ctx1（>0 表示打乱后变不平滑）。"""
    stories = sorted(set(normal_by_story) & set(ctx1_by_story))
    pairs = {s: normal_by_story[s] - ctx1_by_story[s] for s in stories}
    d = np.array([v for v in pairs.values() if np.isfinite(v)], dtype=float)
    if d.size == 0:
        return {"median": float("nan"), "p2_5": float("nan"), "p97_5": float("nan"),
                "sd": float("nan"), "n_stories": 0, "by_story": {}}
    return {
        "median": float(np.median(d)),
        "p2_5": float(np.percentile(d, 2.5)),
        "p97_5": float(np.percentile(d, 97.5)),
        "sd": float(np.std(d, ddof=1)) if d.size > 1 else float("nan"),
        "n_stories": int(d.size),
        "by_story": pairs,
    }


def audit_triggers(delta: dict, band: dict, ctx1_values: np.ndarray) -> dict:
    """V6.4.2 §2.2 冻结的三条审计触发规则。

    仅触发「缓存/索引/resampling 审计」，**不得**据此删条件、换 seed 或改扰动
    规范；审计结论以 §2.1 的确定性断言为准。
    """
    pairs = delta.get("by_story", {})
    d = np.array([v for v in pairs.values() if np.isfinite(v)], dtype=float)

    # 规则1：median(Δlag1) > k × SD_story(lag1_normal)
    sd_normal = band.get("sd", float("nan"))
    med_delta = delta.get("median", float("nan"))
    r1 = bool(np.isfinite(med_delta) and np.isfinite(sd_normal)
              and med_delta > K_AUDIT * sd_normal)

    # 规则2：ctx1 的 story-level 中位数落在 normal 容差带之外
    c = np.asarray(ctx1_values, dtype=float)
    c = c[np.isfinite(c)]
    ctx1_med = float(np.median(c)) if c.size else float("nan")
    lo, hi = band.get("p2_5", np.nan), band.get("p97_5", np.nan)
    r2 = bool(np.isfinite(ctx1_med) and np.isfinite(lo) and np.isfinite(hi)
              and (ctx1_med < lo or ctx1_med > hi))

    # 规则3：存在偏离自身中位数 3SD 的离群故事（「明显双峰」§2.2 未定检验方法，
    # 不在此事后发明阈值——分布形状留给人工看逐故事表判断）
    outliers = []
    if d.size > 1:
        med, sd_d = np.median(d), np.std(d, ddof=1)
        if sd_d > 0:
            outliers = [s for s, v in pairs.items()
                        if np.isfinite(v) and abs(v - med) > OUTLIER_SD * sd_d]
    r3 = bool(outliers)

    return {
        "rule1_median_delta_exceeds_k_sd": {
            "triggered": r1, "k": K_AUDIT, "median_delta": med_delta,
            "sd_story_normal": sd_normal,
            "threshold": K_AUDIT * sd_normal if np.isfinite(sd_normal) else float("nan"),
        },
        "rule2_ctx1_median_outside_normal_band": {
            "triggered": r2, "ctx1_median": ctx1_med,
            "normal_band_p2_5": lo, "normal_band_p97_5": hi,
        },
        "rule3_outlier_stories_3sd": {
            "triggered": r3, "outlier_stories": outliers,
            "note": "「明显双峰」§2.2 未指定检验方法，未事后发明阈值；"
                    "自动判定仅依据 3SD 离群故事，分布形状见逐故事表。",
        },
        "any_triggered": bool(r1 or r2 or r3),
    }


# ── 事后新增：由 Step 5 实际结果逼出的两个问题（显式标记 post_hoc）──────────

def compare_across_models(delta_by_model: dict[str, dict], reference: str = "pythia") -> dict:
    """**事后描述性**：各模型的 Δlag1 与参照模型（默认 pythia）相比差多少。

    动机：I_MP = D_mamba − D_pythia 三被试为负。若 Pythia 的 Δlag1 明显大于
    Mamba（即打乱让 Pythia 的特征掉更多平滑度），则 I_MP 为负可能来自平滑度
    差异而非语言结构敏感性差异——这是 I_MP 的直接混淆项。

    配对方式：**同一故事内**相减（两模型看的是同一批故事、同一置换），因此
    可消除故事长度/语速/体裁的异质性，与 D_m 的配对逻辑一致。
    """
    out = {"post_hoc": True, "reference_model": reference,
           "motivation": "check whether differential smoothness loss confounds I_MP"}
    if reference not in delta_by_model:
        out["error"] = f"参照模型 {reference} 缺失"
        return out
    ref_pairs = delta_by_model[reference].get("by_story", {})
    for model, dm in delta_by_model.items():
        if model == reference:
            continue
        pairs = dm.get("by_story", {})
        common = sorted(set(pairs) & set(ref_pairs))
        diff = np.array([pairs[s] - ref_pairs[s] for s in common], dtype=float)
        diff = diff[np.isfinite(diff)]
        if diff.size == 0:
            continue
        out[f"{model}_minus_{reference}"] = {
            "median": float(np.median(diff)),
            "p2_5": float(np.percentile(diff, 2.5)),
            "p97_5": float(np.percentile(diff, 97.5)),
            "n_stories": int(diff.size),
            "reading": ("<0 表示该模型比参照模型掉的平滑度更少，"
                        "即参照模型(pythia)特征统计受打乱影响更大"),
        }
    return out


def h_dependence(delta_by_H: dict[int, dict]) -> dict:
    """**事后描述性**：Δlag1 是否随 H 增大而加剧（H=128 vs H=8，逐故事配对）。

    动机：若打乱在长上下文下破坏特征结构更严重，则「H=128 掉得更多」本身就能
    压低 Δr_total^ctx1，是 D_m 与 §2.3 OOD-随-H-累积那条风险的直接观测证据。
    """
    out = {"post_hoc": True,
           "motivation": "does shuffling disrupt feature structure more at larger H"}
    if 8 not in delta_by_H or 128 not in delta_by_H:
        out["error"] = "需要同时有 H=8 与 H=128"
        return out
    p8 = delta_by_H[8].get("by_story", {})
    p128 = delta_by_H[128].get("by_story", {})
    common = sorted(set(p8) & set(p128))
    diff = np.array([p128[s] - p8[s] for s in common], dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        out["error"] = "无共同故事"
        return out
    out["delta_H128_minus_H8"] = {
        "median": float(np.median(diff)),
        "p2_5": float(np.percentile(diff, 2.5)),
        "p97_5": float(np.percentile(diff, 97.5)),
        "n_stories": int(diff.size),
        "reading": ">0 表示 H=128 下打乱造成的平滑度损失比 H=8 更大（OOD 随 H 累积的迹象）",
    }
    return out
