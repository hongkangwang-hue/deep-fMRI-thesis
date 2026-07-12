"""
M5 跨被试综合 —— 方向一致性判读（纯函数，可本地单测）。

严格遵循冻结文档里程碑5"明确不做"：**不把三名被试合并成人群样本做池化/组水平推断，
不生成合并后的组水平CI**。本模块只对每名被试各自已算好的点估计与95%CI，做描述性的
方向一致性判读：

  - 一致（consistent_strong）：各被试方向相同，且各自95%CI均排除0 —— 强的被试内重复；
  - 部分一致（consistent_direction_only）：方向相同，但并非每名被试CI都排除0；
  - 不一致（heterogeneous）：被试间方向异质 —— 如实报告为被试间异质性；
  - 数据不足（insufficient_data）：存在缺失/NaN 估计，无法判读。

point_min / point_max 仅为描述性量级范围，**不是组水平合并估计，不构成组水平CI**。
"""

from __future__ import annotations

import math


def _is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _direction(point) -> str:
    """点估计的方向标签：'+'/'-'/'0'/'nan'。"""
    if _is_nan(point):
        return "nan"
    if point > 0:
        return "+"
    if point < 0:
        return "-"
    return "0"


def _ci_excludes_zero(lo, hi) -> bool:
    """95%CI 是否整体在 0 的同一侧（严格排除 0）。缺失值视为不排除。"""
    if _is_nan(lo) or _is_nan(hi):
        return False
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


CONSISTENCY_LABELS_ZH = {
    "consistent_strong": "一致（强被试内重复：各被试方向相同且各自95%CI均排除0）",
    "consistent_direction_only": "部分一致（方向相同，但并非每名被试CI都排除0）",
    "heterogeneous": "不一致（被试间方向异质）",
    "insufficient_data": "数据不足（存在缺失/NaN估计，无法判读）",
}


def direction_consistency(
    per_subject: dict[str, dict], subject_order: list[str] | None = None,
) -> dict:
    """对**单个**估计量，判读其方向/量级在各被试身上是否一致重现。

    Args:
        per_subject: {subject: {"point":, "ci_lo":, "ci_hi":}}，每名被试对该估计量
            已算好的点估计与95%CI（由逐被试 M5 结果读入）。
        subject_order: 输出中被试的固定顺序；默认按 per_subject 的键序。

    Returns:
        判读结果 dict。**不含任何池化/组水平合并量**——point_min/max 只是描述性
        量级范围，pooling 字段恒为 "none"。
    """
    subjects = subject_order or list(per_subject.keys())
    rows: dict[str, dict] = {}
    directions: list[str] = []
    n_ci = 0
    points: list[float] = []
    any_missing = False

    for s in subjects:
        e = per_subject[s]
        pt, lo, hi = e.get("point"), e.get("ci_lo"), e.get("ci_hi")
        d = _direction(pt)
        ci0 = _ci_excludes_zero(lo, hi)
        if d == "nan":
            any_missing = True
        else:
            points.append(pt)
        if ci0:
            n_ci += 1
        directions.append(d)
        rows[s] = {"point": pt, "ci_lo": lo, "ci_hi": hi,
                   "direction": d, "ci_excludes_zero": ci0}

    nonnan = [d for d in directions if d != "nan"]
    # 全部被试方向相同、且都是明确的正/负（不含 '0'、不含缺失）
    all_same = (
        len(nonnan) == len(directions) and len(set(nonnan)) == 1
        and nonnan[0] in ("+", "-"))

    if any_missing or not nonnan:
        consistency = "insufficient_data"
    elif all_same:
        consistency = ("consistent_strong" if n_ci == len(subjects)
                       else "consistent_direction_only")
    else:
        consistency = "heterogeneous"

    return {
        "per_subject": rows,
        "subject_order": subjects,
        "directions": directions,
        "all_same_direction": bool(all_same),
        "n_ci_excludes_zero": n_ci,
        "n_subjects": len(subjects),
        # 描述性量级范围（非组水平合并估计、非组水平CI）
        "point_min": min(points) if points else float("nan"),
        "point_max": max(points) if points else float("nan"),
        "consistency": consistency,
        "consistency_label_zh": CONSISTENCY_LABELS_ZH[consistency],
        "pooling": "none",
        "note": ("仅方向一致性描述性判读；point_min/max 为量级范围，"
                 "非组水平合并估计，不构成组水平CI（里程碑5明确不做1-2）。"),
    }
