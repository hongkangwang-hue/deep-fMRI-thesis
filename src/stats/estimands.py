"""
M5 —— contrast_registry 估计量计算（纯函数，可本地单测）。

输入是一张 r 表 dict[(layer, model, H, condition, roi)] -> r（由 bootstrap 每次重抽或
点估计生成），输出一张 dict[name] -> 标量估计量。同一函数被点估计和每次重抽复用，故
所有 registry 对比共用一次遍历。命名前缀与 frozen/contrast_registry.yaml 对齐，方便
scripts/m5_analysis.py 按类别（确认性/探索/次要/稳健）分组与打 Holm。

严格分层（冻结文档里程碑5）：只有 CONFIRMATORY 里的两项 Δr_total 架构差值可下确认性
结论；RQ1 H-specific、Δr_local/long、PT、AWD-LSTM、最终层、shifted 全为探索/描述/稳健。
"""

from __future__ import annotations

import numpy as np

MODELS = ["pythia", "mamba", "rwkv", "awd_lstm"]
CORE_VS_PYTHIA = ["rwkv", "mamba"]     # RQ1 / 确认性里与 pythia 比的核心状态模型
HS = [8, 32, 128]
ROIS = ["left_IFG", "bilateral_PT"]

# 唯一可下确认性结论的家族（frozen/contrast_registry.yaml::confirmatory_primary）
CONFIRMATORY = [
    "rwkv_minus_pythia_delta_total_ifg_main",
    "mamba_minus_pythia_delta_total_ifg_main",
]


def _r(rt: dict, layer, model, H, cond, roi) -> float:
    return rt.get((layer, model, H, cond, roi), float("nan"))


def _delta_total(rt, layer, model, cond, roi):
    return _r(rt, layer, model, 128, cond, roi) - _r(rt, layer, model, 8, cond, roi)


def _delta_local(rt, layer, model, cond, roi):
    return _r(rt, layer, model, 32, cond, roi) - _r(rt, layer, model, 8, cond, roi)


def _delta_long(rt, layer, model, cond, roi):
    return _r(rt, layer, model, 128, cond, roi) - _r(rt, layer, model, 32, cond, roi)


def compute_estimands(rt: dict) -> dict[str, float]:
    """从 r 表算出全部 registry 估计量。key 命名规则见各段注释。"""
    out: dict[str, float] = {}

    # ── 交付物1：每 model×H×ROI 的原始 r（主层正常）。含 awd_lstm 上下文曲线所需。 ──
    for m in MODELS:
        for H in HS:
            for roi in ROIS:
                out[f"r_{m}_H{H}_{roi}_main_normal"] = _r(rt, "main", m, H, "normal", roi)

    # ── 主层正常 IFG 的三类 Context Gain（每模型；delta_total 属描述性主估计量） ──
    for m in MODELS:
        out[f"delta_total_{m}_ifg_main"] = _delta_total(rt, "main", m, "normal", "left_IFG")
        out[f"delta_local_{m}_ifg_main"] = _delta_local(rt, "main", m, "normal", "left_IFG")
        out[f"delta_long_{m}_ifg_main"] = _delta_long(rt, "main", m, "normal", "left_IFG")

    # ── 确认性家族：两项 Δr_total 架构差值（IFG 主层） ──
    out["rwkv_minus_pythia_delta_total_ifg_main"] = (
        out["delta_total_rwkv_ifg_main"] - out["delta_total_pythia_ifg_main"])
    out["mamba_minus_pythia_delta_total_ifg_main"] = (
        out["delta_total_mamba_ifg_main"] - out["delta_total_pythia_ifg_main"])

    # ── RQ1 探索：每 H 下 rwkv/mamba − pythia 的直接 r 差值（IFG 主层正常） ──
    for arch in CORE_VS_PYTHIA:
        for H in HS:
            out[f"{arch}_minus_pythia_r{H}"] = (
                _r(rt, "main", arch, H, "normal", "left_IFG")
                - _r(rt, "main", "pythia", H, "normal", "left_IFG"))

    # ── 次要探索：PT 上的 Context Gain + IFG vs PT 描述性差异（每模型 delta_total 之差） ──
    for m in MODELS:
        out[f"delta_total_{m}_pt_main"] = _delta_total(rt, "main", m, "normal", "bilateral_PT")
        out[f"delta_local_{m}_pt_main"] = _delta_local(rt, "main", m, "normal", "bilateral_PT")
        out[f"delta_long_{m}_pt_main"] = _delta_long(rt, "main", m, "normal", "bilateral_PT")
        out[f"ifg_minus_pt_delta_total_{m}"] = (
            out[f"delta_total_{m}_ifg_main"] - out[f"delta_total_{m}_pt_main"])

    # ── 稳健性A：最终层 IFG 上的同一批注册对比（供层位翻转判定，与主层配对比较 CI） ──
    for m in MODELS:
        out[f"delta_total_{m}_ifg_final"] = _delta_total(rt, "final", m, "normal", "left_IFG")
        for H in HS:
            out[f"r_{m}_H{H}_left_IFG_final_normal"] = _r(rt, "final", m, H, "normal", "left_IFG")
    out["rwkv_minus_pythia_delta_total_ifg_final"] = (
        out["delta_total_rwkv_ifg_final"] - out["delta_total_pythia_ifg_final"])
    out["mamba_minus_pythia_delta_total_ifg_final"] = (
        out["delta_total_mamba_ifg_final"] - out["delta_total_pythia_ifg_final"])

    # ── 稳健性B：shifted（40s 负控制）r 与三类 shifted Δr（IFG 主层 shift 条件） ──
    for m in MODELS:
        for H in HS:
            out[f"r_{m}_H{H}_left_IFG_main_shift"] = _r(rt, "main", m, H, "shift", "left_IFG")
        out[f"shifted_delta_total_{m}_ifg_main"] = _delta_total(rt, "main", m, "shift", "left_IFG")
        out[f"shifted_delta_local_{m}_ifg_main"] = _delta_local(rt, "main", m, "shift", "left_IFG")
        out[f"shifted_delta_long_{m}_ifg_main"] = _delta_long(rt, "main", m, "shift", "left_IFG")
    out["shifted_rwkv_minus_pythia_delta_total_ifg_main"] = (
        out["shifted_delta_total_rwkv_ifg_main"] - out["shifted_delta_total_pythia_ifg_main"])
    out["shifted_mamba_minus_pythia_delta_total_ifg_main"] = (
        out["shifted_delta_total_mamba_ifg_main"] - out["shifted_delta_total_pythia_ifg_main"])

    # ── 稳健性B（负控制的正确统计量）：每模型 normal − shifted Context Gain 的**配对**
    # 差值（同一次 bootstrap 抽样内相减 → 正确配对 CI）。这是判断"平移是否显著降低了
    # Context Gain"的关键量：CI 排除 0 且为正 = 平移显著削弱了该模型的上下文增益（负
    # 控制对该模型有效）；CI 跨 0 = 平移未显著降低（该模型的 Δr_total 大部分非词序特异，
    # 需谨慎解读）。注意：这检验的是每模型自身的 gain，与架构差值收缩（上面 shifted_*_
    # minus_pythia）是不同层面——后者才对应确认性家族。
    for m in MODELS:
        out[f"delta_total_normal_minus_shift_{m}_ifg_main"] = (
            out[f"delta_total_{m}_ifg_main"] - out[f"shifted_delta_total_{m}_ifg_main"])

    return out
