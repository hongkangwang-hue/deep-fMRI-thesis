"""
M2-A 对齐与 mask 单元测试（纯 numpy，可本地运行，不碰模型/ridge）。

覆盖 M2 验收标准：
  - synthetic alignment：已知时延能被正确 FIR 恢复，错误时移得分更低；
  - FIR 边缘失效：前 max_shift 个 TR 标记为无效；
  - 故事安全：每个故事独立 FIR/重采样，边缘无效不跨故事传播；
  - word→TR 与参考 lanczosinterp2D 数值一致；
  - 共同评价 mask 的 >100s 与交集逻辑。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "encoding"))

from src.fmri.alignment import apply_fir, word_to_tr, _delays_to_shifts
from src.fmri.mask import after_time_mask, common_scoring_mask
from src.fmri.trfile import story_tr_times, expected_response_rows, trimmed_tr_times
from ridge_utils.interpdata import lanczosinterp2D


# ----------------------------- FIR --------------------------------------

def test_delays_to_shifts():
    assert _delays_to_shifts((2, 4, 6, 8), 2.0) == [1, 2, 3, 4]
    assert _delays_to_shifts((2, 4, 6, 8), 1.0) == [2, 4, 6, 8]


def test_fir_shape_and_edge_invalidation():
    rng = np.random.default_rng(0)
    nt, dim = 50, 7
    X = rng.standard_normal((nt, dim))
    Xf, valid = apply_fir(X, delays_s=(2, 4, 6, 8), tr=2.0)
    # 4 个延迟 → 列数 = dim * 4
    assert Xf.shape == (nt, dim * 4)
    # 最大 shift = 4 → 前 4 行无效，其余有效
    assert not valid[:4].any()
    assert valid[4:].all()


def test_fir_delayed_column_content():
    # 延迟 d 的那块应等于把 X 下移 d 行、前 d 行补零
    nt, dim = 20, 3
    X = np.arange(nt * dim, dtype=float).reshape(nt, dim)
    Xf, _ = apply_fir(X, delays_s=(2,), tr=2.0)  # shift = 1
    assert np.allclose(Xf[1:], X[:-1])
    assert np.allclose(Xf[0], 0.0)


def test_fir_respects_valid_in():
    nt, dim = 30, 4
    X = np.ones((nt, dim))
    valid_in = np.ones(nt, dtype=bool)
    valid_in[:3] = False  # 模拟下采样本身前 3 行无效
    _, valid = apply_fir(X, delays_s=(2, 4, 6, 8), tr=2.0, valid_in=valid_in)
    # 无效 = max(前3的 valid_in, 前4的 FIR) → 前 4 行无效
    assert not valid[:4].any()
    assert valid[4:].all()


# --------------------- synthetic alignment ------------------------------

def test_synthetic_alignment_recovers_known_delay():
    """构造 BOLD = 把刺激按已知 TR 延迟搬移，正确 FIR 列与之相关最高。"""
    rng = np.random.default_rng(42)
    nt = 400
    true_shift = 2                       # 真实延迟 2 TR（= 4s @TR2）
    stim = rng.standard_normal((nt, 1))
    bold = np.zeros((nt, 1))
    bold[true_shift:] = stim[:-true_shift]   # BOLD 落后刺激 2 TR
    bold += 0.01 * rng.standard_normal((nt, 1))  # 轻噪声

    Xf, valid = apply_fir(stim, delays_s=(2, 4, 6, 8), tr=2.0)  # 列 = 延迟1,2,3,4
    # 在有效点上，逐延迟列与 BOLD 的相关，应在「延迟2（= true_shift）」处最大
    b = bold[valid, 0]
    corrs = [np.corrcoef(Xf[valid, c], b)[0, 1] for c in range(Xf.shape[1])]
    assert int(np.argmax(corrs)) == true_shift - 1  # 列 index = shift-1
    # 正确延迟相关显著高于错误延迟
    best = corrs[true_shift - 1]
    others = [corrs[c] for c in range(len(corrs)) if c != true_shift - 1]
    assert best > max(others) + 0.3


def test_wrong_shift_scores_lower():
    """人为错配延迟（把 BOLD 多移一格）会降低最佳相关。"""
    rng = np.random.default_rng(7)
    nt = 400
    stim = rng.standard_normal((nt, 1))
    bold = np.zeros((nt, 1))
    bold[2:] = stim[:-2]
    Xf, valid = apply_fir(stim, delays_s=(2, 4, 6, 8), tr=2.0)
    b_correct = bold[valid, 0]
    # 错误：把 BOLD 再下移 5 格，与设计矩阵失配
    bold_wrong = np.zeros((nt, 1))
    bold_wrong[7:] = stim[:-7]
    b_wrong = bold_wrong[valid, 0]
    best_correct = max(np.corrcoef(Xf[valid, c], b_correct)[0, 1]
                       for c in range(Xf.shape[1]))
    best_wrong = max(np.corrcoef(Xf[valid, c], b_wrong)[0, 1]
                     for c in range(Xf.shape[1]))
    assert best_correct > best_wrong + 0.3


# ---------------------- story-safe（无跨故事传播）-----------------------

def test_fir_no_cross_story_leakage():
    """两个故事各自 FIR，每个故事自己的前 max_shift 行都无效（不会因拼接而只在
    全局开头无效）。"""
    rng = np.random.default_rng(1)
    story_a = rng.standard_normal((30, 5))
    story_b = rng.standard_normal((25, 5))
    _, va = apply_fir(story_a, delays_s=(2, 4, 6, 8), tr=2.0)
    _, vb = apply_fir(story_b, delays_s=(2, 4, 6, 8), tr=2.0)
    assert not va[:4].any() and va[4:].all()
    assert not vb[:4].any() and vb[4:].all()  # 故事 B 自己的开头也无效
    # 若错误地先拼接再 FIR，故事 B 开头会“借用”故事 A 末尾→这里独立处理则不会
    Xb, _ = apply_fir(story_b, delays_s=(2,), tr=2.0)
    assert np.allclose(Xb[0], 0.0)  # B 第一行仍是零填充，未泄漏 A 的数据


# ------------------------- word→TR 与参考一致 ---------------------------

def test_word_to_tr_matches_reference():
    rng = np.random.default_rng(3)
    n_words, dim, n_tr = 120, 10, 40
    wv = rng.standard_normal((n_words, dim))
    data_times = np.sort(rng.uniform(0, 80, n_words))
    tr_times = np.arange(n_tr) * 2.0 + 1.0
    out = word_to_tr(wv, data_times, tr_times, window=3)
    ref = lanczosinterp2D(wv, data_times, tr_times, window=3)
    assert out.shape == (n_tr, dim)
    assert np.allclose(out, ref)


def test_word_to_tr_length_mismatch_raises():
    with pytest.raises(ValueError):
        word_to_tr(np.zeros((10, 3)), np.zeros(9), np.zeros(5))


# ----------------------------- mask -------------------------------------

def test_after_time_mask():
    tr_times = np.array([-9.0, 1.0, 99.0, 100.0, 101.0])
    m = after_time_mask(tr_times, after_s=100.0)
    assert m.tolist() == [False, False, False, False, True]


def test_common_scoring_mask_intersection():
    n = 10
    tr_times = np.arange(n) * 20.0          # 0,20,...,180 → >100 即 idx>=6（120..）
    fir_valid = np.ones(n, dtype=bool); fir_valid[:4] = False
    shift_valid = np.ones(n, dtype=bool); shift_valid[-1] = False
    m = common_scoring_mask(tr_times, fir_valid, shift_valid=shift_valid)
    # >100s: idx 6,7,8,9；fir 去掉前4（无影响）；shift 去掉 idx9 → {6,7,8}
    assert np.flatnonzero(m).tolist() == [6, 7, 8]


def test_common_scoring_mask_length_check():
    with pytest.raises(ValueError):
        common_scoring_mask(np.zeros(10), np.zeros(9, dtype=bool))


# ----------------- trfile 行数契约（与真实响应对齐）---------------------

def test_trfile_axis_lengths_consistent():
    # tr_times 长度 = resps - pad；trim 后 = resps - pad - 15
    resps = 375
    assert len(story_tr_times(resps)) == resps - 5
    assert expected_response_rows(resps) == resps - 5 - 15
    assert len(trimmed_tr_times(resps)) == expected_response_rows(resps)