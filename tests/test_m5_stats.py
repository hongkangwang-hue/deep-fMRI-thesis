"""M5 统计核心单测：配对结构、聚合闭式解、bootstrap 可复现、p 值、Holm。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats.bootstrap import (
    BootstrapData, aggregate_to_r, paired_bootstrap, draws_to_arrays,
    percentile_ci, bootstrap_two_sided_p, holm_bonferroni, _identity_idx,
)
from src.stats.estimands import compute_estimands


def test_context_gain_normal_minus_shift_paired():
    """负控制配对差值 = normal Δr_total − shifted Δr_total（逐 draw 相减）。"""
    rt = {}
    # 构造 pythia IFG 主层 normal/shift 的 H8/H128 r，使 Δr_total 已知
    rt[("main", "pythia", 8, "normal", "left_IFG")] = 0.10
    rt[("main", "pythia", 128, "normal", "left_IFG")] = 0.14   # normal Δ=+0.04
    rt[("main", "pythia", 8, "shift", "left_IFG")] = 0.01
    rt[("main", "pythia", 128, "shift", "left_IFG")] = 0.04    # shift Δ=+0.03
    out = compute_estimands(rt)
    assert out["delta_total_pythia_ifg_main"] == pytest.approx(0.04)
    assert out["shifted_delta_total_pythia_ifg_main"] == pytest.approx(0.03)
    # 配对差值 = 0.04 - 0.03 = 0.01（shift 削弱了 0.01 的 gain）
    assert out["delta_total_normal_minus_shift_pythia_ifg_main"] == pytest.approx(0.01)


def _toy_data():
    """2 fold、每 fold 3 story。两个 key：A、B（B = A 每值 +0.5 z），共用同一权重。"""
    fold_stories = {"f0": ["s0", "s1", "s2"], "f1": ["t0", "t1", "t2"]}
    wv = {"f0": np.array([10.0, 20.0, 30.0]), "f1": np.array([5.0, 5.0, 10.0])}
    zA = {"f0": np.array([0.1, 0.2, 0.3]), "f1": np.array([0.0, 0.4, 0.2])}
    zB = {"f0": zA["f0"] + 0.5, "f1": zA["f1"] + 0.5}
    keyA = ("main", "A", 8, "normal", "roi")
    keyB = ("main", "B", 8, "normal", "roi")
    z = {keyA: zA, keyB: zB}
    w = {keyA: {f: v.copy() for f, v in wv.items()},
         keyB: {f: v.copy() for f, v in wv.items()}}
    return BootstrapData(folds=["f0", "f1"], fold_stories=fold_stories, z=z, w=w)


def test_aggregate_point_matches_manual():
    d = _toy_data()
    keyA = ("main", "A", 8, "normal", "roi")
    idx = _identity_idx(d.fold_stories)
    # fold z: f0 = (0.1*10+0.2*20+0.3*30)/60 = (1+4+9)/60 = 14/60
    f0z = (0.1 * 10 + 0.2 * 20 + 0.3 * 30) / 60
    f1z = (0.0 * 5 + 0.4 * 5 + 0.2 * 10) / 20
    # cross-fold weights = fold weight sums = 60, 20
    agg = (f0z * 60 + f1z * 20) / 80
    assert aggregate_to_r(d.z[keyA], d.w[keyA], idx) == pytest.approx(np.tanh(agg))


def test_paired_self_difference_is_exactly_zero():
    """同一 key 相减：每次重抽都精确为 0（配对结构的强证据）。"""
    d = _toy_data()
    keyA = ("main", "A", 8, "normal", "roi")

    def est(rt):
        return {"selfdiff": rt[keyA] - rt[keyA]}

    point, draws = paired_bootstrap(d, est, n_boot=50, seed=1)
    arr = draws_to_arrays(draws)["selfdiff"]
    assert point["selfdiff"] == 0.0
    assert np.all(arr == 0.0)
    assert percentile_ci(arr) == (0.0, 0.0)


def test_bootstrap_reproducible_with_seed():
    d = _toy_data()
    keyA, keyB = ("main", "A", 8, "normal", "roi"), ("main", "B", 8, "normal", "roi")

    def est(rt):
        return {"diff": rt[keyB] - rt[keyA]}

    _, d1 = paired_bootstrap(d, est, n_boot=100, seed=20260701)
    _, d2 = paired_bootstrap(d, est, n_boot=100, seed=20260701)
    _, d3 = paired_bootstrap(d, est, n_boot=100, seed=999)
    a1 = draws_to_arrays(d1)["diff"]
    a2 = draws_to_arrays(d2)["diff"]
    a3 = draws_to_arrays(d3)["diff"]
    assert np.array_equal(a1, a2)          # 同种子完全一致
    assert not np.array_equal(a1, a3)      # 不同种子不同


def test_bootstrap_p_all_positive_is_small():
    """B−A 恒正（B 每值 +0.5 z → r 更大），双尾 p 应为 0。"""
    d = _toy_data()
    keyA, keyB = ("main", "A", 8, "normal", "roi"), ("main", "B", 8, "normal", "roi")

    def est(rt):
        return {"diff": rt[keyB] - rt[keyA]}

    _, draws = paired_bootstrap(d, est, n_boot=200, seed=7)
    arr = draws_to_arrays(draws)["diff"]
    assert np.all(arr > 0)
    assert bootstrap_two_sided_p(arr) == 0.0


def test_bootstrap_two_sided_p_symmetric_around_zero():
    vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    # p_le = 3/5 (含0), p_ge = 3/5 → 2*min=2*0.6=1.2→截断1.0
    assert bootstrap_two_sided_p(vals) == pytest.approx(1.0)
    vals2 = np.array([-1.0] + [1.0] * 99)   # 1% 在 <=0 侧
    assert bootstrap_two_sided_p(vals2) == pytest.approx(0.02)


def test_holm_stepdown():
    p = {"a": 0.001, "b": 0.02, "c": 0.60}
    out = holm_bonferroni(p, alpha=0.05)
    # 排序 a(0.001)@0.05/3=0.0167 拒绝; b(0.02)@0.05/2=0.025 拒绝; c(0.6)@0.05 不拒绝
    assert out["a"]["reject"] and out["b"]["reject"] and not out["c"]["reject"]


def test_holm_stepdown_stops_after_first_failure():
    # 排序后 a(0.001) < b(0.30) < c(0.40)：a@0.05/3=0.0167 拒绝；b@0.025 失败 →
    # step-down 后即使 c 单独看 <0.05 也因排在 b 之后而不拒绝。
    p = {"a": 0.001, "b": 0.30, "c": 0.40}
    out = holm_bonferroni(p, alpha=0.05)
    assert out["a"]["reject"]
    assert not out["b"]["reject"]
    assert not out["c"]["reject"]


def test_holm_two_contrasts_confirmatory_like():
    p = {"rwkv": 0.03, "mamba": 0.20}
    out = holm_bonferroni(p, alpha=0.05)
    # rwkv(0.03)@0.05/2=0.025 → 0.03>0.025 不拒绝 → mamba 也不拒绝
    assert not out["rwkv"]["reject"]
    assert not out["mamba"]["reject"]
