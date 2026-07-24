"""补充实验（回应导师建议）——刺激侧上下文控制 C1（ctx_shuffle_story）。

对应 `../../milestone/里程碑增补老师修改建议上下文控制.md`（V6.4.2）§0.3/§0.6/§0.7。
纯逻辑，无 torch/模型依赖，本地可测。

核心语义（冻结，见 ../config/freeze_manifest.json）：
    对每篇故事的完整词序列生成一次固定置换 π（种子由 master_seed + story_id
    通过 sha256 稳定派生，不用 Python 内置 hash()）。目标词 w_i 本身、时间戳、
    在窗口末尾的位置完全不变；只有它前面 H 个上下文槽位改放置换后序列
    s[i-H:i] 对应的词。

    绝对禁止对每个目标窗口独立重新随机打乱——那样会破坏相邻窗口的 H-1 滑窗
    重叠，使 encoding score 的下降无法区分"语言结构被破坏"还是"特征时间统计
    结构被额外破坏"（V6.4.2 §0.6）。本模块的 API 强制同一故事只调用一次
    story_permutation，随后所有目标位置的窗口都在同一个 perm 上滑窗取值。
"""

from __future__ import annotations

from hashlib import sha256

import numpy as np


def story_seed(master_seed: int, story_id: str) -> int:
    """稳定派生子 seed。V6.4.2 §0.7：sha256 而非 Python hash()（后者受
    PYTHONHASHSEED 影响，跨进程不可复现）。"""
    digest = sha256(f"{master_seed}|{story_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def story_permutation(story_id: str, master_seed: int, n_words: int) -> np.ndarray:
    """对整篇故事生成一次固定置换 π，长度 n_words。

    记号（V6.4.2 §0.3）：s[j] = words[π[j]]，即置换后序列第 j 位放原始第
    π[j] 个词。同一 (story_id, master_seed) 永远得到同一个 π（确定性、可复现）。
    """
    if n_words <= 0:
        raise ValueError(f"n_words 必须为正数，got {n_words}")
    seed = story_seed(master_seed, story_id)
    rng = np.random.default_rng(seed)
    return rng.permutation(n_words)


def context_origins(perm: np.ndarray, i: int, H: int) -> np.ndarray:
    """C1_context_origins(i,H)：上下文槽位 {i-H,...,i-1} 承载的原始位置集合。

    等价于 V6.4.2 §0.3 的 C1_context_origins(i,H) = {π(j) : j ∈ {i-H,...,i-1}}。
    """
    if i < H:
        raise ValueError(f"target index {i} 在 H={H} 下没有完整历史（需要 i >= H）")
    return perm[i - H : i]


def build_perturbed_window(words: list[str], perm: np.ndarray, i: int, H: int) -> list[str]:
    """C1 窗口：乱序上下文 + 真实目标词，共 H+1 个词。

    input_C1(i, H) = s[i-H:i] + w_i，s[j] = words[perm[j]]。

    Args:
        words: 该故事按 word_local_id 升序排列的完整词表（与
            src.models.windowing.build_window 的入参语义一致）。
        perm:  由 story_permutation(story_id, master_seed, len(words)) 生成的
            该故事固定置换——调用方必须只生成一次、传给该故事所有目标位置
            共用，不得每次调用本函数时重新生成（见模块文档的"绝对禁止"）。
        i: 故事内 0-based 目标位置。
        H: 历史词数。

    Returns:
        长度 H+1 的词列表，最后一个元素恒等于真实目标词 words[i]。
    """
    if i < H:
        raise ValueError(f"target index {i} 在 H={H} 下没有完整历史（需要 i >= H）")
    if len(perm) != len(words):
        raise ValueError(f"perm 长度 {len(perm)} != words 长度 {len(words)}")
    origins = context_origins(perm, i, H)
    window = [words[j] for j in origins] + [words[i]]
    assert len(window) == H + 1, f"window 长度 {len(window)} != {H + 1}"
    assert window[-1] == words[i], "window 最后一个词必须是真实目标词"
    return window
