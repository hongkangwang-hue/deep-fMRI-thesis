"""M1 模型无关逻辑的本地单元测试（无需 torch/模型）。

覆盖：窗口 off-by-one、首个有效目标、故事边界、W_common 构建、
缓存 round-trip + hash 校验、token map 校验、以及用假适配器验证 extract 的
last-subtoken pooling 与状态重置语义。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.windowing import (
    build_window, first_eligible_index, build_w_common, iter_story_targets,
)
from src.models.base import LayerSpec, ModelAdapter
from src.models.feature_cache import save_features, load_features
from src.models.token_map import make_token_map, validate_token_map


# ── 窗口语义 ────────────────────────────────────────────────────────────────

def test_build_window_h8():
    words = [f"w{k}" for k in range(300)]
    i = 128
    w8 = build_window(words, i, H=8)
    assert len(w8) == 9
    assert w8[0] == words[i - 8]
    assert w8[-1] == words[i]
    assert words[i - 9] not in w8


def test_build_window_h128_first_eligible():
    words = [f"w{k}" for k in range(300)]
    assert first_eligible_index(128) == 128
    w = build_window(words, 128, H=128)
    assert len(w) == 129
    assert w[0] == words[0]
    assert w[-1] == words[128]


def test_build_window_off_by_one_raises():
    words = [f"w{k}" for k in range(300)]
    # i == H-1 没有完整历史
    with pytest.raises(ValueError):
        build_window(words, 127, H=128)
    # i == H 恰好可行
    assert len(build_window(words, 128, H=128)) == 129


def test_window_respects_story_boundary():
    # 每个故事独立词表 → 窗口不可能跨故事
    story_a = [f"a{k}" for k in range(200)]
    story_b = [f"b{k}" for k in range(200)]
    wa = build_window(story_a, 130, H=128)
    assert all(t.startswith("a") for t in wa)
    wb = build_window(story_b, 130, H=128)
    assert all(t.startswith("b") for t in wb)


# ── W_common / 目标枚举 ──────────────────────────────────────────────────────

def _fake_word_index():
    rows = []
    wid = 0
    for story, n in [("s1", 200), ("s2", 150)]:
        for local in range(n):
            rows.append({
                "word_id": wid, "story": story, "word_local_id": local,
                "word": f"{story}_{local}",
                "onset_s": local * 0.4, "offset_s": local * 0.4 + 0.3,
                "eligible_h128": local >= 128,
                "onset_after_100s": local * 0.4 > 100.0,
            })
            wid += 1
    return pd.DataFrame(rows)


def test_w_common_eligible_only():
    wi = _fake_word_index()
    wc = build_w_common(wi)
    # s1: 200-128=72, s2: 150-128=22 → 94
    assert len(wc) == 72 + 22
    assert wc["is_w_common"].all()
    assert (wc["word_local_id"] >= 128).all()


def test_w_common_intersects_model_flags():
    wi = _fake_word_index()
    eligible_ids = set(wi[wi["eligible_h128"]]["word_id"])
    # 假设某模型只能输出其中一半
    half = set(list(eligible_ids)[:40])
    flags = {"pythia": eligible_ids, "awd_lstm": half}
    wc = build_w_common(wi, model_output_flags=flags)
    assert len(wc) == len(half)
    assert "output_pythia" in wc.columns
    assert "output_awd_lstm" in wc.columns


def test_iter_story_targets():
    wi = _fake_word_index()
    out = dict((s, (w, e)) for s, w, e in iter_story_targets(wi))
    assert set(out) == {"s1", "s2"}
    words, elig = out["s1"]
    assert len(words) == 200
    assert elig[0] == 128 and elig[-1] == 199


# ── 假适配器：验证 extract 的 pooling 与状态重置语义 ─────────────────────────

class FakeAdapter(ModelAdapter):
    """词级假模型：每词 1 token；hidden = one-hot 风格的可预测向量。
    支持「故意继承上一窗状态」用于状态重置反向测试。"""
    is_recurrent = True
    model_id = "fake"
    revision = "0"
    hidden_width = 4
    n_layers = 3

    def __init__(self):
        self._carry = 0.0  # 模拟跨窗状态

    def load(self): pass

    def reset_state(self):
        self._carry = 0.0

    def tokenize_with_spans(self, words):
        ids = list(range(len(words)))
        spans = [(k, k + 1) for k in range(len(words))]
        is_unk = [False] * len(words)
        return ids, spans, is_unk

    def forward_hidden(self, token_ids, layers):
        n = len(token_ids)
        out = {}
        for layer in {layers.main, layers.final}:
            arr = np.zeros((n, self.hidden_width), dtype=np.float32)
            # 每个 token 的向量编码其位置 + 跨窗状态 + 层号
            for t in range(n):
                arr[t, 0] = t
                arr[t, 1] = layer
                arr[t, 2] = self._carry
            out[layer] = arr
        return out

    def inherit_then_extract(self, words, i, H, layers):
        """不重置、注入非零状态 → 用于反向测试。"""
        from src.models.windowing import build_window
        window = build_window(words, i, H)
        self._carry = 99.0  # 故意继承
        token_ids, spans, is_unk = self.tokenize_with_spans(window)
        hidden = self.forward_hidden(token_ids, layers)
        ts, te = spans[-1]
        return hidden[layers.main][te - 1]


def test_extract_last_subtoken_pooling():
    words = [f"w{k}" for k in range(200)]
    a = FakeAdapter()
    layers = LayerSpec(main=1, final=2)
    rep = a.extract(words, i=150, H=128, layers=layers)
    # 窗口长 129，目标是最后一个 token → index 128
    assert rep.n_tokens == 129
    assert rep.target_token_index == 128
    assert rep.main[0] == 128  # 位置编码
    assert rep.main[1] == 1    # 主层号
    assert rep.final[1] == 2   # 最终层号


def test_state_reset_positive_and_negative():
    words = [f"w{k}" for k in range(200)]
    a = FakeAdapter()
    layers = LayerSpec(main=1, final=2)
    # 正向：两次冷启动一致
    r1 = a.extract(words, 150, 128, layers)
    r2 = a.extract(words, 150, 128, layers)
    assert np.allclose(r1.main, r2.main)
    # 反向：故意继承上一窗状态 → 不同
    inherited = a.inherit_then_extract(words, 150, 128, layers)
    assert not np.allclose(r1.main, inherited)


# ── 缓存 round-trip 与 hash 校验 ─────────────────────────────────────────────

def test_feature_cache_roundtrip(tmp_path):
    n, d = 50, 8
    wid = np.arange(n)
    main = np.random.randn(n, d).astype(np.float32)
    final = np.random.randn(n, d).astype(np.float32)
    is_unk = np.zeros(n, dtype=bool)
    meta = {"model_id": "fake", "revision": "0", "layer_main": 1,
            "layer_final": 2, "code_version": "test"}
    save_features(tmp_path, "fake", "s1", 128, wid, main, final, is_unk, meta)
    loaded = load_features(tmp_path, "fake", "s1", 128)
    assert np.array_equal(loaded["word_ids"], wid)
    assert np.allclose(loaded["main"], main)
    assert loaded["meta"]["model_id"] == "fake"


def test_feature_cache_detects_tampering(tmp_path):
    import numpy as np
    n, d = 10, 4
    save_features(tmp_path, "fake", "s1", 8, np.arange(n),
                  np.zeros((n, d), np.float32), np.zeros((n, d), np.float32),
                  np.zeros(n, bool),
                  {"model_id": "f", "revision": "0", "layer_main": 0,
                   "layer_final": 1, "code_version": "t"})
    # 篡改 npz 中的 main 数组
    from src.models.feature_cache import cache_path
    p = cache_path(tmp_path, "fake", "s1", 8)
    with np.load(p, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["main"] = data["main"] + 1.0
    np.savez_compressed(p, **data)
    with pytest.raises(ValueError, match="hash 不匹配"):
        load_features(tmp_path, "fake", "s1", 8)


# ── token map 校验 ───────────────────────────────────────────────────────────

def test_token_map_validate_ok():
    wi = _fake_word_index()
    rows = []
    for _, r in wi[wi["eligible_h128"]].head(10).iterrows():
        rows.append({
            "word_id": r["word_id"], "story": r["story"],
            "word_local_id": r["word_local_id"], "H": 128,
            "target_token_index": 128, "n_tokens": 129,
            "n_target_subtokens": 1, "is_unk": False,
        })
    tm = make_token_map(rows)
    validate_token_map(tm, wi)  # 不应抛错


def test_token_map_detects_misalignment():
    wi = _fake_word_index()
    rows = [{
        "word_id": 130, "story": "WRONG", "word_local_id": 130, "H": 128,
        "target_token_index": 128, "n_tokens": 129,
        "n_target_subtokens": 1, "is_unk": False,
    }]
    tm = make_token_map(rows)
    with pytest.raises(AssertionError, match="错位"):
        validate_token_map(tm, wi)
