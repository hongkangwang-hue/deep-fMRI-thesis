"""§2.2 lag-1 诊断纯函数单测（零算力）。

重点验证「度量确实在度量平滑度」这件事本身——用构造的平滑序列 vs 打乱序列，
断言前者 lag-1 明显更高。若这条不成立，后面所有解读都无意义。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SUPPLEMENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SUPPLEMENT_ROOT / "src"))

from lag1_diagnostic import (  # noqa: E402
    lag1_cosine_median, lag1_featurewise_corr_mean, tolerance_band,
    precision_interval, paired_delta, audit_triggers,
    compare_across_models, h_dependence, K_AUDIT,
)


def _smooth_series(T=200, d=64, seed=0):
    """平滑序列：随机游走（相邻行高度相似）——模拟真实上下文下的 TR 特征。"""
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((T, d)) * 0.05
    return np.cumsum(steps, axis=0) + 1.0


def _rough_series(T=200, d=64, seed=1):
    """不平滑序列：独立同分布（相邻行无关）——模拟被彻底打乱后的极端情形。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, d))


# ── 度量本身是否真的在测平滑度 ────────────────────────────────────────────

def test_cosine_higher_for_smooth_than_rough():
    smooth = lag1_cosine_median(_smooth_series())
    rough = lag1_cosine_median(_rough_series())
    assert smooth > rough, f"平滑序列 lag-1({smooth}) 应显著高于粗糙序列({rough})"
    assert smooth > 0.9, "随机游走的相邻余弦应接近 1"
    assert abs(rough) < 0.3, "独立同分布序列的相邻余弦应接近 0"


def test_featurewise_corr_same_direction():
    """补充指标应与主指标同向（不同向说明其中一个实现有误）。"""
    assert (lag1_featurewise_corr_mean(_smooth_series())
            > lag1_featurewise_corr_mean(_rough_series()))


def test_short_or_degenerate_input_returns_nan_not_crash():
    assert np.isnan(lag1_cosine_median(np.zeros((1, 8))))       # 只有一行
    assert np.isnan(lag1_cosine_median(np.zeros((10, 8))))      # 全零 → 范数为0
    assert np.isnan(lag1_featurewise_corr_mean(np.ones((10, 8))))  # 零方差


# ── 容差带 / 精度区间：必须是两个不同的东西 ───────────────────────────────

def test_tolerance_band_is_wider_than_precision_interval():
    """核心区分：容差带描述故事间离散度，精度区间描述中位数估计精度。
    后者随故事数收窄，必然远窄于前者——这正是 V6.4.2 §2.2 纠正 V6.4.1 的点。"""
    rng = np.random.default_rng(0)
    vals = rng.normal(0.8, 0.05, size=83)
    band = tolerance_band(vals)
    prec = precision_interval(vals, seed=1, n_boot=200)
    band_w = band["p97_5"] - band["p2_5"]
    prec_w = prec["median_ci_hi"] - prec["median_ci_lo"]
    assert band_w > prec_w * 3, "容差带应远宽于精度区间"
    assert prec["usage"] == "descriptive_only_never_a_threshold"


# ── 配对差值与审计规则 ────────────────────────────────────────────────────

def test_paired_delta_sign_convention():
    """Δlag1 = normal − ctx1；>0 表示打乱后变不平滑。"""
    nrm = {"a": 0.90, "b": 0.88, "c": 0.92}
    ctx = {"a": 0.70, "b": 0.68, "c": 0.72}
    d = paired_delta(nrm, ctx)
    assert d["median"] == pytest.approx(0.20, abs=1e-9)
    assert d["n_stories"] == 3


def test_paired_delta_only_uses_common_stories():
    d = paired_delta({"a": 0.9, "b": 0.8}, {"a": 0.7, "z": 0.5})
    assert d["n_stories"] == 1


def test_audit_rule1_triggers_when_shift_exceeds_k_sd():
    # normal 分布很窄(SD小)，但 Δlag1 很大 → 规则1 应触发
    nrm_vals = np.full(20, 0.90) + np.random.default_rng(0).normal(0, 0.001, 20)
    band = tolerance_band(nrm_vals)
    delta = {"median": 0.5, "by_story": {f"s{i}": 0.5 for i in range(20)}}
    audit = audit_triggers(delta, band, np.full(20, 0.40))
    assert audit["rule1_median_delta_exceeds_k_sd"]["triggered"]
    assert audit["rule1_median_delta_exceeds_k_sd"]["k"] == K_AUDIT
    assert audit["any_triggered"]


def test_audit_rule2_triggers_when_ctx1_outside_band():
    nrm_vals = np.random.default_rng(0).normal(0.90, 0.01, 50)
    band = tolerance_band(nrm_vals)
    delta = paired_delta({f"s{i}": v for i, v in enumerate(nrm_vals)},
                         {f"s{i}": 0.30 for i in range(len(nrm_vals))})
    audit = audit_triggers(delta, band, np.full(50, 0.30))  # 远低于容差带下界
    assert audit["rule2_ctx1_median_outside_normal_band"]["triggered"]


def test_audit_no_trigger_when_conditions_nearly_identical():
    """两条件几乎相同 → 三条规则都不该触发（避免闸门永远亮红灯）。"""
    rng = np.random.default_rng(0)
    nrm_vals = rng.normal(0.90, 0.02, 50)
    ctx_vals = nrm_vals + rng.normal(0, 0.0005, 50)
    band = tolerance_band(nrm_vals)
    delta = paired_delta({f"s{i}": v for i, v in enumerate(nrm_vals)},
                         {f"s{i}": v for i, v in enumerate(ctx_vals)})
    audit = audit_triggers(delta, band, ctx_vals)
    assert not audit["any_triggered"]


def test_audit_rule3_flags_outlier_story():
    rng = np.random.default_rng(0)
    pairs = {f"s{i}": float(v) for i, v in enumerate(rng.normal(0.1, 0.01, 40))}
    pairs["weird"] = 0.9          # 极端离群
    delta = {"median": 0.1, "by_story": pairs}
    band = tolerance_band(rng.normal(0.9, 0.02, 40))
    audit = audit_triggers(delta, band, rng.normal(0.8, 0.02, 40))
    assert "weird" in audit["rule3_outlier_stories_3sd"]["outlier_stories"]


# ── 事后新增的两个对比 ────────────────────────────────────────────────────

def test_compare_across_models_detects_pythia_losing_more():
    """构造 pythia 掉更多平滑度的情形，mamba−pythia 应为负。"""
    stories = [f"s{i}" for i in range(20)]
    dbm = {
        "pythia": {"by_story": {s: 0.30 for s in stories}},   # 掉很多
        "mamba": {"by_story": {s: 0.10 for s in stories}},    # 掉较少
    }
    out = compare_across_models(dbm, reference="pythia")
    assert out["post_hoc"] is True
    assert out["mamba_minus_pythia"]["median"] == pytest.approx(-0.20, abs=1e-9)


def test_h_dependence_detects_larger_disruption_at_h128():
    stories = [f"s{i}" for i in range(20)]
    dbh = {8: {"by_story": {s: 0.05 for s in stories}},
           128: {"by_story": {s: 0.25 for s in stories}}}
    out = h_dependence(dbh)
    assert out["post_hoc"] is True
    assert out["delta_H128_minus_H8"]["median"] == pytest.approx(0.20, abs=1e-9)


def test_h_dependence_requires_both_H():
    out = h_dependence({8: {"by_story": {"a": 0.1}}})
    assert "error" in out
