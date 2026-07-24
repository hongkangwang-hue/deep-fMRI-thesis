"""刺激侧上下文控制 Step 5 的估计量（纯函数，可本地单测，零算力）。

输入是一张 r 表 dict[(layer, model, H, cond, roi)] -> r（由 paired_bootstrap 每次
重抽或点估计生成，与主 M5 的 estimands.py 完全同构的接口），输出 dict[name] -> 标量。

对应 V6.4.2 §7.3：
  Δr_total^cond(m) = r128^cond(m) − r8^cond(m)                （主层 left_IFG）
  D_m              = Δr_total^normal(m) − Δr_total^ctx1(m)     （模型内配对差值）
  A_MP^cond        = Δr_total^cond(mamba) − Δr_total^cond(pythia)
  I_MP             = D_mamba − D_pythia                        （difference-in-differences，核心）
  I_RP             = D_rwkv − D_pythia

全部为诊断性 / 未校正 / 非确认性——不做 Holm，不进确认性家族（V6.4.2 §7.1/§7.5）。
"""

from __future__ import annotations

NAN = float("nan")
CORE_MODELS = ["pythia", "rwkv", "mamba"]


def _r(rt: dict, model, H, cond, roi="left_IFG") -> float:
    return rt.get(("main", model, H, cond, roi), NAN)


def _delta_total(rt, model, cond, roi="left_IFG") -> float:
    return _r(rt, model, 128, cond, roi) - _r(rt, model, 8, cond, roi)


def ctx1_estimands(rt: dict, roi: str = "left_IFG") -> dict[str, float]:
    """从 r 表算 ctx1 全部诊断量。roi 默认 left_IFG（Core）；传 bilateral_PT 可复用出 PT 版。"""
    tag = "ifg" if roi == "left_IFG" else ("pt" if roi == "bilateral_PT" else roi)
    out: dict[str, float] = {}

    # 每模型：normal / ctx1 各自的 Δr_total，以及配对差值 D_m
    for m in CORE_MODELS:
        out[f"ctx1_delta_total_{m}_{tag}"] = _delta_total(rt, m, "ctx1", roi)
        out[f"normal_delta_total_{m}_{tag}"] = _delta_total(rt, m, "normal", roi)
        out[f"D_{m}_{tag}"] = (out[f"normal_delta_total_{m}_{tag}"]
                               - out[f"ctx1_delta_total_{m}_{tag}"])

    # 架构差值：normal 与 ctx1 条件下各自的 Mamba−Pythia / RWKV−Pythia
    out[f"A_MP_normal_{tag}"] = (out[f"normal_delta_total_mamba_{tag}"]
                                 - out[f"normal_delta_total_pythia_{tag}"])
    out[f"A_RP_normal_{tag}"] = (out[f"normal_delta_total_rwkv_{tag}"]
                                 - out[f"normal_delta_total_pythia_{tag}"])
    out[f"A_MP_ctx1_{tag}"] = (out[f"ctx1_delta_total_mamba_{tag}"]
                               - out[f"ctx1_delta_total_pythia_{tag}"])
    out[f"A_RP_ctx1_{tag}"] = (out[f"ctx1_delta_total_rwkv_{tag}"]
                               - out[f"ctx1_delta_total_pythia_{tag}"])

    # 核心交互量 difference-in-differences：I_MP = D_mamba − D_pythia
    # （等价于 A_MP_normal − A_MP_ctx1，即 Mamba 相对 Pythia 的优势在打乱上下文后减弱多少）
    out[f"I_MP_{tag}"] = out[f"D_mamba_{tag}"] - out[f"D_pythia_{tag}"]
    out[f"I_RP_{tag}"] = out[f"D_rwkv_{tag}"] - out[f"D_pythia_{tag}"]

    return out


# 报告时哪些量是"核心诊断量"（进正文 Table 9），哪些是辅助（进 supplement）
CORE_REPORT_NAMES = [
    "D_pythia_ifg", "D_mamba_ifg", "I_MP_ifg",
]
SUPPLEMENT_REPORT_NAMES = [
    "D_rwkv_ifg", "I_RP_ifg",
    "ctx1_delta_total_pythia_ifg", "ctx1_delta_total_rwkv_ifg", "ctx1_delta_total_mamba_ifg",
    "A_MP_ctx1_ifg", "A_RP_ctx1_ifg",
    "A_MP_normal_ifg", "A_RP_normal_ifg",
    "normal_delta_total_pythia_ifg", "normal_delta_total_rwkv_ifg", "normal_delta_total_mamba_ifg",
]
