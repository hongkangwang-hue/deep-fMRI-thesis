"""ctx1 估计量纯函数单测（Step 5，零算力）。

用手工构造的 r 表验证 Δr_total / D_m / I_MP 的代数正确性——尤其是
I_MP = D_mamba − D_pythia 必须恒等于 A_MP_normal − A_MP_ctx1（两种算法给同一个数），
这是 difference-in-differences 的核心恒等式，写错方向会让结论反过来。
"""

import math
import sys
from pathlib import Path

SUPPLEMENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SUPPLEMENT_ROOT / "src"))

from ctx1_estimands import ctx1_estimands, _delta_total  # noqa: E402


def _make_rt(vals: dict) -> dict:
    """vals: {(model, H, cond): r} → 展开成完整 r 表 key (main, model, H, cond, left_IFG)。"""
    return {("main", m, H, cond, "left_IFG"): r for (m, H, cond), r in vals.items()}


def test_delta_total_basic():
    rt = _make_rt({("mamba", 8, "normal"): 0.10, ("mamba", 128, "normal"): 0.17})
    assert math.isclose(_delta_total(rt, "mamba", "normal"), 0.07)


def test_D_m_is_normal_minus_ctx1_gain():
    # normal: Δr_total = 0.17-0.10 = 0.07；ctx1: 0.12-0.105 = 0.015 → D = 0.055
    rt = _make_rt({
        ("mamba", 8, "normal"): 0.10, ("mamba", 128, "normal"): 0.17,
        ("mamba", 8, "ctx1"): 0.105, ("mamba", 128, "ctx1"): 0.12,
    })
    out = ctx1_estimands(rt)
    assert math.isclose(out["normal_delta_total_mamba_ifg"], 0.07)
    assert math.isclose(out["ctx1_delta_total_mamba_ifg"], 0.015)
    assert math.isclose(out["D_mamba_ifg"], 0.055)


def test_I_MP_equals_did_two_ways():
    """I_MP = D_mamba − D_pythia 必须恒等于 A_MP_normal − A_MP_ctx1。"""
    rt = _make_rt({
        # pythia
        ("pythia", 8, "normal"): 0.10, ("pythia", 128, "normal"): 0.133,
        ("pythia", 8, "ctx1"):   0.10, ("pythia", 128, "ctx1"):   0.128,
        # mamba
        ("mamba", 8, "normal"): 0.10, ("mamba", 128, "normal"): 0.169,
        ("mamba", 8, "ctx1"):   0.10, ("mamba", 128, "ctx1"):   0.121,
        # rwkv (随便给，检验不干扰)
        ("rwkv", 8, "normal"): 0.09, ("rwkv", 128, "normal"): 0.096,
        ("rwkv", 8, "ctx1"):   0.09, ("rwkv", 128, "ctx1"):   0.094,
    })
    out = ctx1_estimands(rt)
    # 方法一：D_mamba − D_pythia
    i_mp_a = out["D_mamba_ifg"] - out["D_pythia_ifg"]
    # 方法二：A_MP_normal − A_MP_ctx1
    i_mp_b = out["A_MP_normal_ifg"] - out["A_MP_ctx1_ifg"]
    assert math.isclose(out["I_MP_ifg"], i_mp_a)
    assert math.isclose(i_mp_a, i_mp_b), f"DiD 两种算法不一致: {i_mp_a} vs {i_mp_b}"


def test_missing_model_yields_nan_not_crash():
    # 只有 pythia+mamba（L2 scope），rwkv 缺失 → 与 rwkv 相关的量为 nan，但不报错
    rt = _make_rt({
        ("pythia", 8, "normal"): 0.10, ("pythia", 128, "normal"): 0.13,
        ("pythia", 8, "ctx1"):   0.10, ("pythia", 128, "ctx1"):   0.12,
        ("mamba", 8, "normal"): 0.10, ("mamba", 128, "normal"): 0.17,
        ("mamba", 8, "ctx1"):   0.10, ("mamba", 128, "ctx1"):   0.12,
    })
    out = ctx1_estimands(rt)
    assert math.isfinite(out["I_MP_ifg"])          # pythia+mamba 齐 → I_MP 有值
    assert math.isnan(out["D_rwkv_ifg"])           # rwkv 缺 → nan
    assert math.isnan(out["I_RP_ifg"])


def test_sign_convention_positive_D_means_gain_dropped():
    """D_m > 0 语义 = 打乱上下文后 Context Gain 下降（控制生效方向）。"""
    rt = _make_rt({
        ("mamba", 8, "normal"): 0.10, ("mamba", 128, "normal"): 0.17,  # normal gain 0.07
        ("mamba", 8, "ctx1"):   0.10, ("mamba", 128, "ctx1"):   0.10,  # ctx1 gain 0.00
    })
    out = ctx1_estimands(rt)
    assert out["D_mamba_ifg"] > 0  # gain 从 0.07 掉到 0 → D>0
