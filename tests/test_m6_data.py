"""M6 数据访问层单测：命名解析、CI 抽取、缺项降级、AWD 隔离常量。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.viz.m6_data import (
    r_curve, context_gain, rq1, arch_delta_total, confirmatory_rows,
    ci_excludes_zero, CORE_MODELS, REFERENCE_MODEL, HS,
)


def _est():
    """构造一小份 estimands，覆盖图表要访问的命名。"""
    e = {}
    for H in HS:
        e[f"r_pythia_H{H}_left_IFG_main_normal"] = {"point": 0.13 + H*1e-4, "ci_lo": 0.12, "ci_hi": 0.14}
        e[f"r_pythia_H{H}_left_IFG_main_shift"] = {"point": 0.01, "ci_lo": 0.005, "ci_hi": 0.015}
    e["delta_total_pythia_ifg_main"] = {"point": 0.0035, "ci_lo": 0.0020, "ci_hi": 0.0051}
    e["delta_local_pythia_ifg_main"] = {"point": 0.0015, "ci_lo": 0.0, "ci_hi": 0.003}
    e["delta_long_pythia_ifg_main"] = {"point": 0.0020, "ci_lo": 0.0, "ci_hi": 0.004}
    e["shifted_delta_total_pythia_ifg_main"] = {"point": 0.0001, "ci_lo": -0.001, "ci_hi": 0.001}
    e["rwkv_minus_pythia_r8"] = {"point": -0.0013, "ci_lo": -0.0028, "ci_hi": 0.0002}
    e["rwkv_minus_pythia_delta_total_ifg_main"] = {"point": -0.0029, "ci_lo": -0.0043, "ci_hi": -0.0015}
    e["rwkv_minus_pythia_delta_total_ifg_final"] = {"point": -0.001, "ci_lo": -0.003, "ci_hi": 0.001}
    return e


def test_r_curve_aligned_to_HS():
    hs, pts, los, his = r_curve(_est(), "pythia", "left_IFG")
    assert hs == HS
    assert pts[0] == 0.13 + 8e-4
    assert los[0] == 0.12 and his[0] == 0.14


def test_r_curve_missing_returns_none():
    # bilateral_PT 未在 _est 里 → 全 None，但长度与 HS 对齐
    hs, pts, los, his = r_curve(_est(), "pythia", "bilateral_PT")
    assert hs == HS and pts == [None, None, None]


def test_context_gain_normal_and_shifted():
    assert context_gain(_est(), "pythia", "left_IFG", "total")[0] == 0.0035
    assert context_gain(_est(), "pythia", "left_IFG", "total", shifted=True)[0] == 0.0001


def test_rq1_and_arch_delta_layers():
    assert rq1(_est(), "rwkv", 8)[0] == -0.0013
    assert arch_delta_total(_est(), "rwkv", "main")[0] == -0.0029
    assert arch_delta_total(_est(), "rwkv", "final")[0] == -0.001


def test_confirmatory_rows():
    results = {"confirmatory": {
        "rwkv_minus_pythia_delta_total_ifg_main":
            {"point": -0.0029, "ci_lo": -0.0043, "ci_hi": -0.0015,
             "p": 0.0, "holm_threshold": 0.025, "reject": True},
    }}
    rows = confirmatory_rows(results)
    assert len(rows) == 1 and rows[0]["reject"] is True and rows[0]["p"] == 0.0


def test_ci_excludes_zero():
    assert ci_excludes_zero(-0.003, -0.0043, -0.0015) is True   # 全负
    assert ci_excludes_zero(0.003, 0.001, 0.005) is True         # 全正
    assert ci_excludes_zero(-0.001, -0.003, 0.001) is False      # 跨0
    assert ci_excludes_zero(None, None, None) is None


def test_awd_isolation_constants():
    assert REFERENCE_MODEL == "awd_lstm"
    assert "awd_lstm" not in CORE_MODELS
    assert CORE_MODELS == ["pythia", "mamba", "rwkv"]
