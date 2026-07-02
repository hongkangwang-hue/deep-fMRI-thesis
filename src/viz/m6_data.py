"""
M6 —— 从 M5 结果 (m5_results.json) 抽取图表所需的数值访问层（纯函数，可单测）。

图表 (scripts/m6_figures.py) 与表格 (scripts/m6_tables.py) 都通过本模块取数，把
estimand 命名规则集中在一处——命名与 src/stats/estimands.py 完全对应，模型/H/ROI
常量直接从那里 import，保证统计层改名时这里同步、不会静默错位。

M6 只读、不重算（冻结文档里程碑6"明确不做"：不重新计算统计）。
"""

from __future__ import annotations

import json
from pathlib import Path

# 与统计层同源，避免命名漂移
from src.stats.estimands import MODELS, CORE_VS_PYTHIA, HS, ROIS, CONFIRMATORY  # noqa: F401

# AWD-LSTM 是历史参照，图表须视觉隔离（虚线/灰度），绝不参与核心排名（里程碑6红线）
CORE_MODELS = ["pythia", "mamba", "rwkv"]
REFERENCE_MODEL = "awd_lstm"

# 展示用配色/线型（核心三模型实线，AWD-LSTM 灰色虚线）
MODEL_STYLE = {
    "pythia":   {"color": "#1f77b4", "linestyle": "-",  "marker": "o", "label": "Pythia (Transformer)"},
    "mamba":    {"color": "#d62728", "linestyle": "-",  "marker": "s", "label": "Mamba (SSM)"},
    "rwkv":     {"color": "#2ca02c", "linestyle": "-",  "marker": "^", "label": "RWKV (RNN/linear-attn)"},
    "awd_lstm": {"color": "#888888", "linestyle": "--", "marker": "x", "label": "AWD-LSTM (historical ref.)"},
}


def load_results(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _get(est: dict, name: str) -> dict | None:
    """取一个 estimand 的 {point, ci_lo, ci_hi}；不存在返回 None（供缺项优雅降级）。"""
    return est.get(name)


def r_curve(est: dict, model: str, roi: str, layer: str = "main", cond: str = "normal"):
    """返回某 (model, roi, layer, cond) 的 r 随 H 曲线：(Hs, points, ci_los, ci_his)。

    对应 estimands 命名 r_{model}_H{H}_{roi}_{layer}_{cond}。缺项以 None 占位（对齐 Hs）。
    """
    hs, pts, los, his = [], [], [], []
    for H in HS:
        e = _get(est, f"r_{model}_H{H}_{roi}_{layer}_{cond}")
        hs.append(H)
        pts.append(e["point"] if e else None)
        los.append(e["ci_lo"] if e else None)
        his.append(e["ci_hi"] if e else None)
    return hs, pts, los, his


def context_gain(est: dict, model: str, roi: str = "left_IFG",
                 kind: str = "total", shifted: bool = False):
    """三类 Context Gain 之一：(point, ci_lo, ci_hi)。

    kind ∈ {total, local, long}；shifted=True 取 40s 位移条件（仅 IFG 主层有）。
    命名：delta_{kind}_{model}_{roi_tag}_main（shifted 前缀 shifted_，仅 ifg）。
    """
    roi_tag = "ifg" if roi == "left_IFG" else "pt"
    if shifted:
        name = f"shifted_delta_{kind}_{model}_ifg_main"
    else:
        name = f"delta_{kind}_{model}_{roi_tag}_main"
    e = _get(est, name)
    return (e["point"], e["ci_lo"], e["ci_hi"]) if e else (None, None, None)


def rq1(est: dict, arch: str, H: int):
    """RQ1 H-specific 架构差值 arch−pythia（IFG 主层正常）：(point, ci_lo, ci_hi)。"""
    e = _get(est, f"{arch}_minus_pythia_r{H}")
    return (e["point"], e["ci_lo"], e["ci_hi"]) if e else (None, None, None)


def arch_delta_total(est: dict, arch: str, layer: str = "main"):
    """架构 Δr_total 差值 arch−pythia（IFG，layer∈{main,final}）：(point, ci_lo, ci_hi)。"""
    e = _get(est, f"{arch}_minus_pythia_delta_total_ifg_{layer}")
    return (e["point"], e["ci_lo"], e["ci_hi"]) if e else (None, None, None)


def confirmatory_rows(results: dict) -> list[dict]:
    """确认性家族逐项：名字/点估计/CI/p/Holm阈/是否拒绝（直接读 results['confirmatory']）。"""
    rows = []
    for name, e in results.get("confirmatory", {}).items():
        rows.append({
            "name": name, "point": e["point"], "ci_lo": e["ci_lo"], "ci_hi": e["ci_hi"],
            "p": e.get("p"), "holm_threshold": e.get("holm_threshold"),
            "reject": e.get("reject"),
        })
    return rows


def ci_excludes_zero(pt, lo, hi) -> bool | None:
    """CI 是否完全在 0 一侧（点估计显著方向）；缺项返回 None。"""
    if lo is None or hi is None:
        return None
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
