"""
M1 — 模型级 token map（word_id → subtoken span → target token）

每个 (story, model) 产出一张 token map，把 M0 统一 word_index 的每个目标
word_id 关联到该模型 tokenizer 下的 subtoken 跨度与目标 token 位置，并记录
是否 unk。用于证明「四模型 token map 与统一 word_index 一一关联、无静默错位」。

注意：这里只定义结构与读写；真实 token 跨度由各适配器的 tokenize_with_spans
产生（需模型库，在 AutoDL 上运行）。本模块可用假 tokenizer 在本地测试结构。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TOKEN_MAP_COLUMNS = [
    "word_id",          # 关联 M0 word_index 的全局 word_id
    "story",
    "word_local_id",
    "H",                # 该映射对应的上下文长度（不同 H 窗口 token 数不同）
    "target_token_index",   # 目标词最后一个 subtoken 在窗口序列中的位置
    "n_tokens",             # 该窗口 token 总数
    "n_target_subtokens",   # 目标词被切成几个 subtoken
    "is_unk",               # 目标词是否被映射为 unk
]


def make_token_map(rows: list[dict]) -> pd.DataFrame:
    """从提取过程收集的行构建 token map DataFrame 并校验列完整。"""
    df = pd.DataFrame(rows)
    missing = set(TOKEN_MAP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"token map 缺少列: {sorted(missing)}")
    return df[TOKEN_MAP_COLUMNS]


def validate_token_map(token_map: pd.DataFrame, word_index: pd.DataFrame) -> None:
    """校验 token map 与统一 word_index 一一关联，无静默错位。"""
    # 每个 (word_id, H) 唯一
    dup = token_map.duplicated(subset=["word_id", "H"]).sum()
    assert dup == 0, f"token map 存在 {dup} 个重复 (word_id, H)"

    # 所有 word_id 必须存在于 word_index
    known = set(word_index["word_id"])
    unknown = set(token_map["word_id"]) - known
    assert not unknown, f"token map 含 word_index 中不存在的 word_id: {list(unknown)[:5]}"

    # story / word_local_id 必须与 word_index 一致（无错位）
    ref = word_index.set_index("word_id")[["story", "word_local_id"]]
    joined = token_map.join(ref, on="word_id", rsuffix="_ref")
    mismatch = (
        (joined["story"] != joined["story_ref"])
        | (joined["word_local_id"] != joined["word_local_id_ref"])
    ).sum()
    assert mismatch == 0, f"token map 与 word_index 有 {mismatch} 处 story/local_id 错位"


def save_token_map(token_map: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token_map.to_parquet(path, index=False)


def load_token_map(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
