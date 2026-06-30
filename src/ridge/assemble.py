"""
M3 数据组装 —— 把 M1 特征缓存对齐到 TR 级响应空间（方案 A）。

方案 A：模型特征只对合格目标词存在（故事开头若干词无特征）。Lanczos 下采样
**只在目标词的 data_times 子集上**进行，故事开头无支撑的 TR 由 >100s 评分 mask
自然丢弃，不补零、不污染插值。

流程（单故事，复刻 encoding_utils 的契约）：
  特征缓存 word_ids(全局) → join word_index.parquet → data_times=(onset+offset)/2
  word_to_tr(main, data_times, story_tr_times(n_resps))  → (resps-pad, hidden)
  trim [10:-5]                                            → (T, hidden) 对齐已 trim 的 .hf5
  BOLD 响应 load_response(已 trim)                         → (T, V)

本模块**仅做对齐与加载**，不做 PCA/FIR/ridge（那是 pipeline.py），但会读取文件。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.fmri.alignment import word_to_tr                 # noqa: E402
from src.fmri.derivatives import load_response            # noqa: E402
from src.fmri.trfile import (                              # noqa: E402
    load_respdict, story_tr_times, trimmed_tr_times,
    TRIM_FIRST, TRIM_LAST,
)
from src.models.feature_cache import load_features        # noqa: E402
from src.ridge.pipeline import StoryData                  # noqa: E402


def _word_times(word_index: pd.DataFrame, word_ids: np.ndarray) -> np.ndarray:
    """按 word_ids（全局）取中心时刻 (onset+offset)/2，顺序与 word_ids 一致。"""
    idx = word_index.set_index("word_id")
    rows = idx.loc[word_ids]
    return ((rows["onset_s"].to_numpy() + rows["offset_s"].to_numpy()) / 2.0)


def assemble_story(
    story: str, model: str, H: int, layer: str,
    subject: str, cache_dir, data_dir, respdict: dict,
    word_index: pd.DataFrame,
) -> StoryData:
    """组装单故事的 (X 特征, Y 响应, tr_times)。layer ∈ {'main','final'}。"""
    feat = load_features(cache_dir, model, story, H)
    word_ids = feat["word_ids"]
    vecs = feat[layer].astype(np.float64)                 # (n_targets, hidden)

    data_times = _word_times(word_index, word_ids)
    order = np.argsort(data_times)                        # Lanczos 要求 data_times 单调
    data_times, vecs = data_times[order], vecs[order]

    n_resps = respdict[story]
    tr_full = story_tr_times(n_resps)                     # (resps - pad)
    X_full = word_to_tr(vecs, data_times, tr_full)        # (resps - pad, hidden)
    X = X_full[TRIM_FIRST: len(X_full) - TRIM_LAST]       # trim [10:-5]

    Y = load_response(data_dir, subject, story).astype(np.float64)  # (T, V) 已 trim
    trt = trimmed_tr_times(n_resps)

    if not (X.shape[0] == Y.shape[0] == len(trt)):
        raise ValueError(
            f"[{story}] 行数不一致 X={X.shape[0]} Y={Y.shape[0]} tr_times={len(trt)}")
    return StoryData(X=X, Y=Y, tr_times=trt)


def assemble_all(
    stories: list[str], model: str, H: int, layer: str,
    subject: str, cache_dir, data_dir, respdict_path, word_index_path,
) -> dict[str, StoryData]:
    """组装多故事 → {story: StoryData}。"""
    respdict = load_respdict(respdict_path)
    word_index = pd.read_parquet(word_index_path)
    return {
        s: assemble_story(s, model, H, layer, subject, cache_dir, data_dir,
                          respdict, word_index)
        for s in stories
    }
