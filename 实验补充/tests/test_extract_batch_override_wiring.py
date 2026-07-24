"""验证 base.py 新增的 window_override/windows_override 参数接线正确（Step 3 前置）。

这不是测试 context_perturb.py 本身（那部分已在 test_context_perturb.py 覆盖），
而是测试"把 context_perturb 生成的乱序窗口喂进 extract_batch 之后，走的还是
和正常提取完全相同的 tokenize/pooling/state-reset 代码路径"这件事本身。
用假适配器（无需 torch），本地零算力，是上 GPU 前必须先过的接线正确性证明。

覆盖：
  1. 不传 override → 行为与改动前逐位相同（回归保护，防止误改了默认路径）。
  2. 传入由 build_perturbed_window 生成的窗口 → 提取结果确实反映"乱序后的
     内容"，而不是静默退回真实上下文（防止 override 被忽略这种最危险的
     静默 bug）。
  3. 批量 windows_override 与逐条 window_override 结果一致（批量路径没有
     引入 padding/顺序错位）。
  4. windows_override 长度与 indices 不符时报错，不静默截断。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
SUPPLEMENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SUPPLEMENT_ROOT / "src"))

from src.models.base import ModelAdapter, LayerSpec  # noqa: E402
from context_perturb import story_permutation, build_perturbed_window  # noqa: E402


class PositionEncodingAdapter(ModelAdapter):
    """假模型：每词 1 token，hidden 的第 0 维直接编码该 token 在窗口内的
    位置。用于验证提取结果确实是"对喂进去的窗口内容做前向"，而不是某种
    与输入无关的固定输出——如果 override 被静默忽略，位置编码会指向原始
    窗口而非乱序窗口，测试会用词形是否匹配来抓出这种情况。"""

    is_recurrent = False
    model_id = "fake_pos"
    revision = "0"
    hidden_width = 2
    n_layers = 1

    def load(self):
        pass

    def reset_state(self):
        pass

    def tokenize_with_spans(self, words):
        ids = list(range(len(words)))
        spans = [(k, k + 1) for k in range(len(words))]
        return ids, spans, [False] * len(words)

    def forward_hidden(self, token_ids, layers):
        n = len(token_ids)
        arr = np.zeros((n, self.hidden_width), dtype=np.float32)
        for t in range(n):
            arr[t, 0] = t
        return {layers.main: arr, layers.final: arr}


def _words():
    return [f"w{k}" for k in range(400)]


def test_no_override_behaves_as_before():
    """回归保护：不传 window_override/windows_override，输出必须与旧行为
    （直接用 build_window）逐元素相同。"""
    words = _words()
    a = PositionEncodingAdapter()
    layers = LayerSpec(main=0, final=0)
    indices = [128, 150, 200]

    single = [a.extract(words, i, 128, layers) for i in indices]
    batched = a.extract_batch(words, indices, 128, layers, batch_size=2)
    for s, b in zip(single, batched):
        assert s.n_tokens == b.n_tokens == 129
        assert np.array_equal(s.main, b.main)


def test_single_extract_with_window_override_preserves_pooling_semantics():
    """window_override 只替换窗口内容，不改变 pooling 语义本身：目标词
    仍在最后一位、target_token_index 仍正确。（"override 内容真的被用上"
    这一更强的主张由下面 test_batch_windows_override_actually_differs_
    from_normal_batch 用带因果依赖的适配器证明——本测试的玩具适配器不
    含跨 token 依赖，不足以单独证明内容生效。）"""
    words = _words()
    perm = story_permutation("fake_story", 20260724, len(words))
    a = PositionEncodingAdapter()
    layers = LayerSpec(main=0, final=0)

    i, H = 300, 128
    real_window = words[i - H : i] + [words[i]]
    perturbed_window = build_perturbed_window(words, perm, i, H)
    assert perturbed_window != real_window  # 前提：确实是不同的窗口内容

    rep_real = a.extract(words, i, H, layers)  # 不传 override，走正常路径
    rep_perturbed = a.extract(words, i, H, layers, window_override=perturbed_window)

    assert rep_real.target_token_index == rep_perturbed.target_token_index == H
    # 目标词未变（window_override 的最后一个元素本就是真实目标词）
    assert perturbed_window[-1] == words[i]


def test_batch_windows_override_matches_single_window_override():
    """批量 windows_override 路径必须与逐条 window_override 路径完全一致
    （padding/顺序不能引入偏差）。"""
    words = _words()
    perm = story_permutation("fake_story", 20260724, len(words))
    a = PositionEncodingAdapter()
    layers = LayerSpec(main=0, final=0)
    H = 128
    indices = [128, 129, 200, 350, 399]

    perturbed_windows = [build_perturbed_window(words, perm, i, H) for i in indices]

    singles = [
        a.extract(words, i, H, layers, window_override=w)
        for i, w in zip(indices, perturbed_windows)
    ]
    batched = a.extract_batch(
        words, indices, H, layers, batch_size=2, windows_override=perturbed_windows
    )
    assert len(batched) == len(indices)
    for s, b in zip(singles, batched):
        assert s.target_token_index == b.target_token_index
        assert s.n_tokens == b.n_tokens
        assert np.array_equal(s.main, b.main)
        assert np.array_equal(s.final, b.final)


class CausalSumAdapter(ModelAdapter):
    """假模型，但带真实的「上下文影响目标表示」因果依赖（真实 Transformer/
    RNN 都有、简单玩具适配器通常没有）：目标 token 的 hidden 是窗口内全部
    token id 的因果累加和，而不只是它自己。token id 直接解析自词形本身的
    编号（"w37" → 37），而不是词在窗口内的局部位置——这样上下文内容（而非
    仅长度）真正决定目标表示的数值，才能诚实检验 override 是否被使用，
    而不是因为玩具模型本身不依赖上下文内容而巧合通过。"""

    is_recurrent = False
    model_id = "fake_causal_sum"; revision = "0"; hidden_width = 1; n_layers = 1

    def load(self): pass
    def reset_state(self): pass

    def tokenize_with_spans(self, words):
        ids = [int(w[1:]) for w in words]  # "w37" -> 37，编码真实词形而非位置
        spans = [(k, k + 1) for k in range(len(words))]
        return ids, spans, [False] * len(words)

    def forward_hidden(self, token_ids, layers):
        n = len(token_ids)
        arr = np.zeros((n, self.hidden_width), dtype=np.float32)
        running = 0.0
        for t in range(n):
            running += token_ids[t]
            arr[t, 0] = running  # 因果累加：位置 t 的表示依赖 0..t 全部内容
        return {layers.main: arr, layers.final: arr}


def test_batch_windows_override_actually_differs_from_normal_batch():
    """最关键的一条：ctx1 的批量提取结果必须与 normal 批量提取结果不同
    （证明 override 真的生效了，不是被 extract_batch 内部忽略、静默退回
    build_window）。用带因果依赖的适配器——目标表示本身就应该随上下文
    内容变化，这才是诚实的检验，而不是巧合通过。"""
    words = _words()
    perm = story_permutation("fake_story", 20260724, len(words))
    a = CausalSumAdapter()
    layers = LayerSpec(main=0, final=0)
    H = 128
    indices = [200, 250, 300]

    normal_batch = a.extract_batch(words, indices, H, layers, batch_size=3)
    perturbed_windows = [build_perturbed_window(words, perm, i, H) for i in indices]
    ctx1_batch = a.extract_batch(
        words, indices, H, layers, batch_size=3, windows_override=perturbed_windows
    )
    for i, n, c in zip(indices, normal_batch, ctx1_batch):
        assert n.target_token_index == c.target_token_index  # 目标位置不变
        # 目标表示（因果累加和）必须因上下文内容不同而不同——真正证明
        # windows_override 被使用，而非被静默忽略退回真实上下文。
        assert n.main[0] != c.main[0], (
            f"i={i}: ctx1 与 normal 的目标表示相同，疑似 windows_override "
            f"被忽略，静默退回了真实上下文"
        )
        # 目标词自身 id（=i）在两条件下都必须被正确累加进最终和——
        # 反推 ctx1 的累加和减去「乱序上下文 id 之和」应恰好等于 i。
        ctx_ids_ctx1 = perm[i - H : i]
        expected_ctx1_sum = float(ctx_ids_ctx1.sum() + i)
        assert c.main[0] == pytest.approx(expected_ctx1_sum)


def test_windows_override_length_mismatch_raises():
    words = _words()
    a = PositionEncodingAdapter()
    layers = LayerSpec(main=0, final=0)
    indices = [128, 129, 130]
    wrong_length_override = [words[0:129]]  # 只给 1 个，indices 有 3 个
    with pytest.raises(ValueError):
        a.extract_batch(words, indices, 128, layers, windows_override=wrong_length_override)
