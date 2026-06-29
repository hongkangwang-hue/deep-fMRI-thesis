"""
M1 — 开窗与 W_common 构建（模型无关，纯逻辑，本地可测）

窗口语义（冻结）：
    W_i(H) = [w_{i-H}, ..., w_{i-1}, w_i]
    H 只数目标词之前的历史词数，实际输入 H+1 个词。
    窗口不跨故事边界：调用方按故事分组传入该故事的有序词表，i 为故事内
    word_local_id。

W_common：在所有「H=128 有完整历史」且「四模型均可技术性输出」的共同目标
位置上建立的正式目标集合。AWD-LSTM 的 xxunk 只记录不删除。
"""

from __future__ import annotations

import pandas as pd


def build_window(words: list[str], i: int, H: int) -> list[str]:
    """返回 H 个历史词 + 当前目标词，共 H+1 个。

    words 必须是「单个故事」内按 word_local_id 升序排列的词表，i 为故事内
    0-based 位置。i < H 时目标词没有完整历史，抛 ValueError（绝不静默截断）。
    """
    if i < H:
        raise ValueError(
            f"target index {i} 在 H={H} 下没有完整历史（需要 i >= H）"
        )
    window = words[i - H : i + 1]
    assert len(window) == H + 1, f"window 长度 {len(window)} != {H + 1}"
    assert window[-1] == words[i], "window 最后一个词必须是目标词"
    return window


def first_eligible_index(H: int) -> int:
    """给定 H，故事内首个有完整历史的 0-based 目标位置。"""
    return H


def build_w_common(
    word_index: pd.DataFrame,
    model_output_flags: dict[str, set[int]] | None = None,
    binding_H: int = 128,
) -> pd.DataFrame:
    """构建 W_common 正式目标集合。

    Args:
        word_index: M0 冻结的 word_index（含 word_id/story/word_local_id/
            eligible_h128 等列）。
        model_output_flags: 可选 {model_name: set(可输出的 word_id)}。来自
            M1 在 AutoDL 上的实测——某模型对某目标位置能否技术性输出。
            缺省 None 表示「假定四模型在所有 H=128 合格位置均可输出」，便于
            本地先行构建底表；AutoDL 跑通后再用实测 flag 收紧。
        binding_H: 约束性上下文长度。H=128 是最严格约束（历史词最多），故
            一个位置只要在 H=128 合格，就自动在 H=8/32 合格。

    Returns:
        DataFrame：在 word_index 基础上加 per-model 输出 flag 列与
        is_w_common 列，仅保留 is_w_common==True 的行。
    """
    if binding_H != 128:
        eligible_col = word_index["word_local_id"] >= binding_H
    else:
        eligible_col = word_index["eligible_h128"]

    base = word_index[eligible_col].copy()

    model_names = list(model_output_flags) if model_output_flags else []
    w_common_mask = pd.Series(True, index=base.index)
    for name in model_names:
        ids = model_output_flags[name]
        col = f"output_{name}"
        base[col] = base["word_id"].isin(ids)
        w_common_mask &= base[col]

    base["is_w_common"] = w_common_mask
    result = base[base["is_w_common"]].reset_index(drop=True)
    return result


def iter_story_targets(
    word_index: pd.DataFrame, binding_H: int = 128
) -> "list[tuple[str, list[str], list[int]]]":
    """按故事产出 (story, words_in_order, eligible_local_ids)。

    供特征提取主循环使用：每个故事一份有序词表，以及该故事内所有满足
    binding_H 完整历史的目标 word_local_id 列表。
    """
    out = []
    for story, grp in word_index.groupby("story", sort=True):
        grp = grp.sort_values("word_local_id")
        words = grp["word"].tolist()
        eligible = grp.loc[
            grp["word_local_id"] >= binding_H, "word_local_id"
        ].tolist()
        out.append((story, words, eligible))
    return out
