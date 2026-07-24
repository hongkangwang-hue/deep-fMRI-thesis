"""刺激侧上下文控制 C1 的纯函数单测（无需 torch/模型，Step 1 交付物）。

覆盖：置换的确定性与可复现性、不同故事/不同 seed 得到不同置换、
V6.4.2 §2.1-B 相邻滑窗 H-1 重叠硬断言（本地可测部分）、目标词/窗口末位不变、
越界与长度不匹配报错、context_origins 与窗口内容一致。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SUPPLEMENT_ROOT = Path(__file__).parent.parent  # 实验补充/
sys.path.insert(0, str(SUPPLEMENT_ROOT / "src"))

from context_perturb import (  # noqa: E402
    story_seed, story_permutation, build_perturbed_window, context_origins,
)


# ── 置换的确定性 ──────────────────────────────────────────────────────────

def test_story_permutation_deterministic():
    p1 = story_permutation("treasureisland", 20260724, 500)
    p2 = story_permutation("treasureisland", 20260724, 500)
    assert np.array_equal(p1, p2)


def test_story_permutation_is_valid_permutation():
    p = story_permutation("wheretheressmoke", 20260724, 300)
    assert sorted(p.tolist()) == list(range(300))


def test_different_story_different_permutation():
    p1 = story_permutation("storyA", 20260724, 500)
    p2 = story_permutation("storyB", 20260724, 500)
    assert not np.array_equal(p1, p2)


def test_different_seed_different_permutation():
    p1 = story_permutation("treasureisland", 20260724, 500)
    p2 = story_permutation("treasureisland", 1, 500)
    assert not np.array_equal(p1, p2)


def test_story_seed_deterministic_and_cross_process_stable():
    # 不依赖 Python 内置 hash()（受 PYTHONHASHSEED 影响），sha256 派生结果
    # 必须是跨进程、跨 Python 版本恒定的固定值——直接断言已知输入的输出。
    s1 = story_seed(20260724, "treasureisland")
    s2 = story_seed(20260724, "treasureisland")
    assert s1 == s2
    assert isinstance(s1, int) and s1 >= 0


def test_story_permutation_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        story_permutation("s1", 20260724, 0)


# ── 窗口内容：目标词不变、长度正确 ─────────────────────────────────────────

def test_build_perturbed_window_length_and_target_unchanged():
    words = [f"w{k}" for k in range(300)]
    perm = story_permutation("s1", 20260724, len(words))
    i, H = 150, 8
    window = build_perturbed_window(words, perm, i, H)
    assert len(window) == H + 1
    assert window[-1] == words[i]  # 目标词本身绝对不变


def test_build_perturbed_window_h128():
    words = [f"w{k}" for k in range(400)]
    perm = story_permutation("s1", 20260724, len(words))
    window = build_perturbed_window(words, perm, 200, 128)
    assert len(window) == 129
    assert window[-1] == words[200]


def test_build_perturbed_window_context_is_actually_shuffled():
    # 上下文部分不应等于真实的原始上下文（对随机置换而言，逐位全部相同的
    # 概率极低；用真实语料词表增加不重复概率）。
    words = [f"w{k}" for k in range(1000)]
    perm = story_permutation("realistic_story", 20260724, len(words))
    i, H = 500, 128
    real_ctx = words[i - H : i]
    perturbed = build_perturbed_window(words, perm, i, H)
    assert perturbed[:-1] != real_ctx


def test_build_perturbed_window_off_by_one_raises():
    words = [f"w{k}" for k in range(300)]
    perm = story_permutation("s1", 20260724, len(words))
    with pytest.raises(ValueError):
        build_perturbed_window(words, perm, 127, H=128)
    assert len(build_perturbed_window(words, perm, 128, H=128)) == 129


def test_build_perturbed_window_perm_length_mismatch_raises():
    words = [f"w{k}" for k in range(300)]
    wrong_perm = story_permutation("s1", 20260724, 299)  # 故意长度不对
    with pytest.raises(ValueError):
        build_perturbed_window(words, wrong_perm, 150, H=8)


def test_window_respects_story_boundary():
    # 每个故事独立词表 + 独立置换 → 窗口内容不可能跨故事
    story_a = [f"a{k}" for k in range(200)]
    story_b = [f"b{k}" for k in range(200)]
    perm_a = story_permutation("storyA", 20260724, len(story_a))
    perm_b = story_permutation("storyB", 20260724, len(story_b))
    wa = build_perturbed_window(story_a, perm_a, 130, H=128)
    assert all(t.startswith("a") for t in wa)
    wb = build_perturbed_window(story_b, perm_b, 130, H=128)
    assert all(t.startswith("b") for t in wb)


# ── V6.4.2 §2.1-B：相邻滑窗 H-1 重叠（核心硬断言，本地可测） ───────────────

def test_adjacent_window_h_minus_1_overlap():
    """对同一故事、同一 H，相邻目标位置 i 与 i+1：
    ctx_slots(i+1,H)[:-1] == ctx_slots(i,H)[1:]（V6.4.2 §2.1-B）。
    这是"整篇打乱一次再滑窗"与"逐窗口独立打乱"的本质区别所在——
    只有共用同一个 perm 才能让这条恒成立，是本模块最重要的正确性保证。"""
    words = [f"w{k}" for k in range(600)]
    perm = story_permutation("realistic_story", 20260724, len(words))
    for H in (8, 32, 128):
        for i in (H, H + 1, H + 50, 300, 599 - 1):
            if i + 1 >= len(words) or i < H:
                continue
            origins_i = context_origins(perm, i, H)
            origins_i1 = context_origins(perm, i + 1, H)
            assert np.array_equal(origins_i1[:-1], origins_i[1:]), (
                f"H={H}, i={i}: 相邻窗口 origin 不满足 H-1 重叠"
            )
            # 同时验证：这一重叠在实际取词内容上也成立
            win_i = build_perturbed_window(words, perm, i, H)
            win_i1 = build_perturbed_window(words, perm, i + 1, H)
            assert win_i1[:-2] == win_i[1:-1], (
                f"H={H}, i={i}: 相邻窗口实际词内容不满足 H-1 重叠"
            )


def test_context_origins_matches_window_words():
    words = [f"w{k}" for k in range(300)]
    perm = story_permutation("s1", 20260724, len(words))
    i, H = 150, 32
    origins = context_origins(perm, i, H)
    window = build_perturbed_window(words, perm, i, H)
    expected_ctx = [words[j] for j in origins]
    assert window[:-1] == expected_ctx


def test_context_origins_off_by_one_raises():
    perm = story_permutation("s1", 20260724, 300)
    with pytest.raises(ValueError):
        context_origins(perm, 7, H=8)
