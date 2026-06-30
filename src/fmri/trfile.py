"""
按故事重建 TR 时间轴（与 LeBel 参考实现数值一致）。

参考路径（encoding/feature_spaces.py → ridge_utils）中，TR 时间轴由
load_simulated_trfiles(respdict, tr=2.0, start_time=10.0, pad=5) 生成：

    trtimes      = arange(resps - pad) * tr           # 模拟触发时刻
    reltrig      = trtimes - start_time               # 相对 sound-start
    tr_times     = reltrig + tr/2                      # TR 中心（DataSequence.from_grid 里 +tr/2）
               = arange(resps - pad) * tr - start_time + tr/2

其中 resps = respdict[story]（该故事原始扫描 TR 数）。下采样特征行数 = len(tr_times)
= resps - pad。编码阶段再对特征施加首尾 trim [5+trim : -trim]（trim=5，去首 10 尾 5），
最终与已 trim 的响应 .hf5 行数对齐：response_rows = (resps - pad) - (10 + 5)。

本模块只负责时间轴与行数契约，不加载 BOLD，不做重计算。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 与参考实现一致的常量（load_simulated_trfiles 默认值 + 编码阶段 trim）
TR_SECONDS = 2.0
SOUND_START_SECONDS = 10.0
SIMULATE_PAD = 5          # load_simulated_trfiles 的 pad
TRIM_FIRST = 10           # encoding_utils: 5 + trim，trim=5
TRIM_LAST = 5             # encoding_utils: trim


def load_respdict(respdict_path: str | Path) -> dict[str, int]:
    """读取 respdict.json：{story: 原始扫描 TR 数}。"""
    with open(respdict_path, "r") as f:
        return json.load(f)


def story_tr_times(n_resps: int, tr: float = TR_SECONDS,
                   start_time: float = SOUND_START_SECONDS,
                   pad: int = SIMULATE_PAD) -> np.ndarray:
    """该故事的 TR 中心时间轴（相对 sound-start），长度 = n_resps - pad。

    精确复刻 load_simulated_trfiles + DataSequence.from_grid 的 tr_times。
    """
    n = n_resps - pad
    return np.arange(n) * tr - start_time + tr / 2.0


def expected_response_rows(n_resps: int, pad: int = SIMULATE_PAD,
                           trim_first: int = TRIM_FIRST,
                           trim_last: int = TRIM_LAST) -> int:
    """该故事 trim 后应有的响应/特征行数 = (n_resps - pad) - trim_first - trim_last。"""
    return (n_resps - pad) - trim_first - trim_last


def trimmed_tr_times(n_resps: int, tr: float = TR_SECONDS,
                     start_time: float = SOUND_START_SECONDS,
                     pad: int = SIMULATE_PAD,
                     trim_first: int = TRIM_FIRST,
                     trim_last: int = TRIM_LAST) -> np.ndarray:
    """trim 后、与响应行对齐的 TR 中心时间轴（用于 >100s mask 的时间判定）。"""
    t = story_tr_times(n_resps, tr, start_time, pad)
    return t[trim_first: len(t) - trim_last]