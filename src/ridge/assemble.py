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
    word_index: pd.DataFrame, voxel_mask: np.ndarray | None = None,
) -> StoryData:
    """组装单故事的 (X 特征, Y 响应, tr_times)。layer ∈ {'main','final'}。

    voxel_mask：M1 冻结的 BOLD-only 保留列索引（frozen/voxel_mask_{subject}.npy）。
    给定时传给 load_response(columns=...)，Y 被压缩为仅保留列（M1 已剔除的
    NaN/零方差体素不会进入 Y，从源头避免它们污染 ridge 拟合）。为 None 时保持
    历史行为（返回全量列），仅供不需要该保护的旧调用点兼容。UTS03 的
    voxel_mask 是恒等映射（0 个被排除），传入与否结果完全一致。
    """
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

    Y = load_response(data_dir, subject, story,
                      columns=voxel_mask).astype(np.float64)  # (T, V) 已 trim
    trt = trimmed_tr_times(n_resps)

    if not (X.shape[0] == Y.shape[0] == len(trt)):
        raise ValueError(
            f"[{story}] 行数不一致 X={X.shape[0]} Y={Y.shape[0]} tr_times={len(trt)}")
    return StoryData(X=X, Y=Y, tr_times=trt)


def assemble_all(
    stories: list[str], model: str, H: int, layer: str,
    subject: str, cache_dir, data_dir, respdict_path, word_index_path,
    voxel_mask: np.ndarray | None = None,
) -> dict[str, StoryData]:
    """组装多故事 → {story: StoryData}。voxel_mask 见 assemble_story。"""
    respdict = load_respdict(respdict_path)
    word_index = pd.read_parquet(word_index_path)
    return {
        s: assemble_story(s, model, H, layer, subject, cache_dir, data_dir,
                          respdict, word_index, voxel_mask=voxel_mask)
        for s in stories
    }


def remap_roi_columns_to_voxel_mask(
    roi_cols: dict[str, np.ndarray], voxel_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """把 ROI 列索引从"全量 BOLD 列空间"重映射到 voxel_mask 压缩后的空间。

    frozen/roi_columns_{subject}.npz 里的索引是 M1 阶段在全量 BOLD 列空间（thick
    mask C-order）里算出来的；构建时已与 voxel_mask 求过交（保证不含被排除体
    素），但数值仍是全量空间的原始编号。assemble_all(voxel_mask=...) 把 Y 压缩成
    只剩 voxel_mask 保留列后，同一体素的列号会整体前移——用 searchsorted 在有序
    的 voxel_mask 里定位每个 ROI 列在压缩后数组中的新位置。

    显式核验每个 ROI 列确实落在 voxel_mask 里（M1 的交集步骤保证这一点成立），
    不满足则说明 roi_columns 与 voxel_mask 版本不一致，直接 raise 而不是让
    searchsorted 静默返回错误位置。UTS03 的 voxel_mask 是恒等映射，重映射前后
    索引数值不变。
    """
    remapped = {}
    for name, cols in roi_cols.items():
        pos = np.searchsorted(voxel_mask, cols)
        if not np.array_equal(voxel_mask[pos], cols):
            raise ValueError(
                f"ROI '{name}' 存在不属于 voxel_mask 保留列的列号——"
                f"roi_columns 与 voxel_mask 版本不一致，需重新核对两者是否来自"
                f"同一次 M1 构建")
        remapped[name] = pos
    return remapped
