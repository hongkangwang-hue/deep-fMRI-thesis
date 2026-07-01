"""
按故事独立的时间对齐：word→TR（Lanczos 下采样）与 FIR 延迟（含边缘失效标记）。

⚠️ 故事安全（story-safe）是本模块的硬约束：所有函数只接收并处理「单个故事」的
数组。重采样和 FIR 绝不跨故事边界传播——跨故事拼接只在得到各故事独立结果后，
在更上层（评分阶段）按行 vstack，且每个故事的边缘无效点已显式标记并最终被
评价 mask 排除。

数值上复用 LeBel 参考实现的算子（lanczosinterp2D、make_delayed 同构），以便 M2-C
能与参考 corrs.npz 对照。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 复用参考实现的 Lanczos 算子，保证与 eng1000 参考路径数值一致
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "encoding"))
from ridge_utils.interpdata import lanczosinterp2D  # noqa: E402


def word_to_tr(word_vectors: np.ndarray, data_times: np.ndarray,
               tr_times: np.ndarray, window: int = 3) -> np.ndarray:
    """单故事：把逐词特征 Lanczos 下采样到 TR 中心时间轴。

    Args:
        word_vectors: <float>[n_words, dim] 该故事逐词特征。
        data_times:   <float>[n_words] 每个词的中心时刻 (onset+offset)/2。
        tr_times:     <float>[n_trs] TR 中心时刻（见 trfile.story_tr_times）。
        window:       Lanczos 窗叶数（参考实现默认 3）。

    Returns:
        <float>[n_trs, dim] 下采样后的 TR 级特征。
    """
    if word_vectors.shape[0] != len(data_times):
        raise ValueError(
            f"word_vectors 行数 {word_vectors.shape[0]} != data_times {len(data_times)}")
    return lanczosinterp2D(word_vectors, data_times, tr_times, window=window)


def _delays_to_shifts(delays_s, tr: float) -> list[int]:
    """FIR 延迟（秒）→ TR 位移（样本数），四舍五入。delays 2/4/6/8s @TR2 → 1/2/3/4。"""
    return [int(round(d / tr)) for d in delays_s]


def apply_fir(X: np.ndarray, delays_s=(2, 4, 6, 8), tr: float = 2.0,
              valid_in: np.ndarray | None = None):
    """单故事 FIR：对 X 施加各延迟并显式标记边缘无效点，不循环、不跨故事补值。

    每个延迟 d>0 把信号下移 d（dstim[d:] = X[:-d]），前 d 行为零填充→无效；
    d<0 上移，末 |d| 行无效。合并后的 valid = 各延迟有效点的逻辑与，再与传入的
    valid_in（如下采样本身的有效范围）取与。

    Args:
        X:        <float>[n_trs, dim] 单故事 TR 级特征。
        delays_s: FIR 延迟（秒）。
        tr:       TR 秒数。
        valid_in: 可选 <bool>[n_trs] 输入有效掩码。

    Returns:
        (X_fir <float>[n_trs, dim*n_delays], valid <bool>[n_trs])
    """
    nt, ndim = X.shape
    shifts = _delays_to_shifts(delays_s, tr)
    parts = []
    valid = np.ones(nt, dtype=bool) if valid_in is None else valid_in.astype(bool).copy()
    for d in shifts:
        dpart = np.zeros((nt, ndim), dtype=X.dtype)
        vmask = np.zeros(nt, dtype=bool)
        if d > 0:
            dpart[d:, :] = X[:-d, :]
            vmask[d:] = True
        elif d < 0:
            dpart[:d, :] = X[-d:, :]
            vmask[:d] = True
        else:
            dpart = X.copy()
            vmask[:] = True
        parts.append(dpart)
        valid &= vmask
    return np.hstack(parts), valid


def shift_story_no_wrap(X: np.ndarray, seconds: float = 40.0, tr: float = 2.0):
    """单故事：X_shifted(t) = X(t - seconds)，用于 40s time-shift 负控制。

    与 apply_fir 同一位移约定（正 shift 下移、零填充、故事内不回卷），但只有
    一档、不做多档拼接。用于 M3b/M4 的 time-shift 技术诊断：把特征相对响应
    整体错开 40 秒（远超 2-8s 的 FIR 延迟窗），若模型仍给出与正常条件相近的
    相关，说明存在与语义无关的伪相关（如慢漂移/自相关），而非真实编码信号。

    Args:
        X:       <float>[n_trs, dim] 单故事 TR 级特征（下采样后、PCA/FIR 前）。
        seconds: 位移秒数，正数=特征相对响应延后（用更早的特征预测当前响应）。
        tr:      TR 秒数。

    Returns:
        (X_shifted <float>[n_trs, dim], valid <bool>[n_trs])  故事内不回卷，
        边缘无效点显式标记为 False（供 common_scoring_mask 的 shift_valid 使用）。
    """
    d = int(round(seconds / tr))
    nt, ndim = X.shape
    out = np.zeros_like(X)
    valid = np.zeros(nt, dtype=bool)
    if d > 0:
        out[d:, :] = X[:-d, :]
        valid[d:] = True
    elif d < 0:
        out[:d, :] = X[-d:, :]
        valid[:d] = True
    else:
        out[:] = X
        valid[:] = True
    return out, valid