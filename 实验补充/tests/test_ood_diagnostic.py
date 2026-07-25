"""§2.3 OOD 退化诊断纯函数单测（零算力）。

重点：participation_ratio 必须真的在测「有效维度」——用已知有效维度的构造矩阵
验证。若这条不成立，后面「表示是否塌缩」的全部解读都不成立。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SUPPLEMENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SUPPLEMENT_ROOT / "src"))

from ood_diagnostic import (  # noqa: E402
    l2_norm_median, participation_ratio, evr_at_k,
    h_change_by_story, ood_h_dependence, verdict,
)


# ── L2 范数 ───────────────────────────────────────────────────────────────

def test_l2_norm_median_known_value():
    X = np.array([[3.0, 4.0], [6.0, 8.0], [0.0, 5.0]])  # 范数 5, 10, 5
    assert l2_norm_median(X) == pytest.approx(5.0)


def test_l2_norm_empty_returns_nan():
    assert np.isnan(l2_norm_median(np.zeros((0, 4))))


# ── participation ratio：核心，必须真的在测有效维度 ────────────────────────

def test_pr_isotropic_equals_dimensionality():
    """各向同性（方差均摊在 d 维）→ PR ≈ d。"""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 10))
    assert participation_ratio(X) == pytest.approx(10, rel=0.15)


def test_pr_rank_one_is_about_one():
    """方差全挤进一个方向 → PR ≈ 1（最极端的塌缩）。"""
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(20)
    X = np.outer(rng.standard_normal(500), direction)
    assert participation_ratio(X) == pytest.approx(1.0, abs=0.05)


def test_pr_drops_when_variance_concentrates():
    """把方差逐步挤进少数方向，PR 必须单调下降——这正是「退化」的操作定义。"""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((2000, 20))
    prs = []
    for decay in (0.0, 0.5, 1.5):          # 谱衰减越强 → 有效维度越低
        scale = np.exp(-decay * np.arange(20))
        prs.append(participation_ratio(base * scale))
    assert prs[0] > prs[1] > prs[2], f"PR 应随谱衰减单调下降，实得 {prs}"


def test_pr_degenerate_input():
    assert np.isnan(participation_ratio(np.zeros((1, 5))))   # 只有一行
    assert np.isnan(participation_ratio(np.ones((10, 5))))   # 零方差


# ── evr@k ─────────────────────────────────────────────────────────────────

def test_evr_at_k_monotone_and_bounded():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 50))
    e10, e50 = evr_at_k(X, 10), evr_at_k(X, 50)
    assert 0 < e10 < e50
    assert e50 == pytest.approx(1.0, abs=1e-6)   # k≥维度 → 解释全部方差


def test_evr_rank_one_near_one_at_k1():
    rng = np.random.default_rng(0)
    X = np.outer(rng.standard_normal(300), rng.standard_normal(10))
    assert evr_at_k(X, 1) == pytest.approx(1.0, abs=1e-6)


# ── H 依赖变化与 DiD ──────────────────────────────────────────────────────

def test_h_change_paired_by_story():
    ch = h_change_by_story({"a": 1.0, "b": 2.0}, {"a": 1.5, "b": 2.5})
    assert ch["median"] == pytest.approx(0.5)
    assert ch["n_stories"] == 2


def test_h_change_only_common_stories():
    ch = h_change_by_story({"a": 1.0, "b": 2.0}, {"a": 1.5, "z": 9.0})
    assert ch["n_stories"] == 1


def test_ood_did_detects_larger_ctx1_change():
    """ctx1 随 H 掉得比 normal 多 → DiD 为负且 CI 排除 0。"""
    stories = [f"s{i}" for i in range(30)]
    ch_n = {"by_story": {s: -0.1 for s in stories}}   # normal 小幅下降
    ch_c = {"by_story": {s: -0.9 for s in stories}}   # ctx1 大幅下降
    out = ood_h_dependence(ch_n, ch_c)
    assert out["median"] == pytest.approx(-0.8)
    assert out["ci_excludes_zero"]


def test_ood_did_no_difference_gives_zero():
    stories = [f"s{i}" for i in range(30)]
    ch = {"by_story": {s: -0.4 for s in stories}}
    out = ood_h_dependence(ch, ch)
    assert out["median"] == pytest.approx(0.0)


# ── 判读规则 ──────────────────────────────────────────────────────────────

def test_verdict_flags_when_main_metric_significant():
    ood = {"l2_norm_median": {"ci_excludes_zero": True},
           "participation_ratio": {"ci_excludes_zero": False}}
    v = verdict(ood)
    assert v["conclusion"] == "H_dependent_degradation_detected"
    assert "l2_norm_median" in v["flagged_main_metrics"]
    assert "方向性提示" in v["required_action"]


def test_verdict_clean_when_no_main_metric_flagged():
    ood = {"l2_norm_median": {"ci_excludes_zero": False},
           "participation_ratio": {"ci_excludes_zero": False}}
    v = verdict(ood)
    assert v["conclusion"] == "no_evidence_of_H_dependent_degradation"
    assert v["flagged_main_metrics"] == []


def test_verdict_ignores_evr_which_is_not_a_main_metric():
    """evr@100 是「同时记录」项，不是主指标——不应单独触发判读。"""
    ood = {"l2_norm_median": {"ci_excludes_zero": False},
           "participation_ratio": {"ci_excludes_zero": False},
           "evr_at_100": {"ci_excludes_zero": True}}
    assert verdict(ood)["conclusion"] == "no_evidence_of_H_dependent_degradation"
