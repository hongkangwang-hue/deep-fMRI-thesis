"""
共同评价 mask：>100s ∩ FIR 有效 ∩（可选）shift 有效。

冻结规则（analysis_spec / 里程碑 M2）：
  - 正常条件：mask = (tr_time > 100s) ∩ FIR_valid
  - shifted 比较：再 ∩ normal_valid ∩ shift_valid，保证 normal 与 shifted 在
    完全相同的有效点集合上比较，避免有效点差异污染 Context Gain 差值。

>100s 规则按「每个故事 trim 后的 TR 中心时间」判定（见 trfile.trimmed_tr_times），
只保留故事开始 100 秒之后的 TR，丢弃早期未进入稳定状态的响应。
"""

from __future__ import annotations

import numpy as np

HELDOUT_SCORE_AFTER_SECONDS = 100.0


def after_time_mask(tr_times: np.ndarray,
                    after_s: float = HELDOUT_SCORE_AFTER_SECONDS) -> np.ndarray:
    """单故事：TR 中心时间严格大于 after_s 的布尔掩码。"""
    return tr_times > after_s


def common_scoring_mask(tr_times: np.ndarray,
                        fir_valid: np.ndarray,
                        shift_valid: np.ndarray | None = None,
                        normal_valid: np.ndarray | None = None,
                        after_s: float = HELDOUT_SCORE_AFTER_SECONDS) -> np.ndarray:
    """构建单故事共同评价 mask。

    Args:
        tr_times:     <float>[n_trs] trim 后 TR 中心时间。
        fir_valid:    <bool>[n_trs] FIR 边缘有效掩码。
        shift_valid:  可选 <bool>[n_trs] time-shift 负控制下的有效掩码。
        normal_valid: 可选 <bool>[n_trs] 正常条件 FIR 有效掩码（shifted 比较时传入，
                      与 shift 条件取交集）。
        after_s:      >100s 阈值。

    Returns:
        <bool>[n_trs] 共同评价 mask。
    """
    n = len(tr_times)
    for name, m in [("fir_valid", fir_valid), ("shift_valid", shift_valid),
                    ("normal_valid", normal_valid)]:
        if m is not None and len(m) != n:
            raise ValueError(f"{name} 长度 {len(m)} != tr_times {n}")

    mask = after_time_mask(tr_times, after_s) & fir_valid.astype(bool)
    if normal_valid is not None:
        mask &= normal_valid.astype(bool)
    if shift_valid is not None:
        mask &= shift_valid.astype(bool)
    return mask