"""
硕士论文核查清单 —— 第10节 English1000 部分。

真值来源分两类：

1. 全体素相关 + voxel_mask 限定后相关 + 原始/补充判据判定——这几个数字已经在
   M2 阶段真实跑过、写进了 `milestone/工作留痕_三被试扩展_全部M.md`（"M2 —
   UTS01/UTS02 的 English1000 参照验证硬闸门"一节，2026-07-12）。**这些数字
   不是靠脚本重新算出来的**：m2c_compare.py 只打印到 stdout、不落盘 json，
   所以历史数字唯一的机读来源就是那份工作留痕；本函数直接把它转成结构化表格，
   不重新发明。UTS03 的 Phase 1 数字来自 [[m2-pipeline-state]] 记忆记录
   （voxel_r=0.9962，2026-06-30 跑通，原判据直接 PASS，未触发补充判据）。

2. **逐 ROI（left_IFG / bilateral_PT）限定的相关系数**——这是清单新提的、
   之前诊断链里只抽查过 2 个 UTS02 体素、从未对全 ROI 批量算过的数字，属实
   缺口。下面的 compute_roi_correlations() 已经写好、可读 corrs.npz 真算，
   但需要 results/eng1000/<subject>/corrs.npz（native 参照）与
   results/eng1000_gpu_rerun/<subject>/corrs.npz（GPU 重跑）都同步到本机才能
   跑——本机目前只有 results/eng1000/UTS03/corrs.npz（native 参照，无重跑
   对照组），三名被试都跑不了，运行时会如实打印缺什么文件。

用法：python3 scripts/thesis_supplement_english1000.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config   # noqa: E402

OUT_DIR = PROJECT_ROOT / "thesis_supplement"
SUBJECTS = ("UTS01", "UTS02", "UTS03")

# 来源：milestone/工作留痕_三被试扩展_全部M.md「M2 — UTS01/UTS02 的 English1000
# 参照验证硬闸门」一节（UTS01/UTS02，2026-07-12 真实运行）+ 记忆
# [[m2-pipeline-state]] 记录的 UTS03 Phase 1 结果（2026-06-30 真实运行，直接
# 满足原判据，未触发补充判据、无需诊断链）。
DOCUMENTED_GATE_RESULTS = {
    "UTS01": {
        "voxel_r_all": 0.9867,
        "voxel_r_mask_restricted": 0.995028,
        "n_voxels_total": 81126,
        "n_voxels_excluded_by_voxel_mask": 10,
        "original_threshold": 0.995,
        "original_criterion_pass": False,
        "supplementary_criterion_pass": True,
        "supplementary_criterion_basis": (
            "第8条补充判据（frozen 计划书 2026-07-12 追加）：差异体素100%被M1 "
            "voxel_mask排除（数值退化，原生实现Rsq分母近零病态放大，GPU版有 "
            "nan_to_num 保险丝行为不同）；voxel_mask限定后=0.995028，+0.000028 "
            "过线。"),
        "fail_reason_if_any": (
            "原始全体素判据FAIL的根因：10个体素在原生CPU实现下Rsq分母近零，"
            "数值病态放大出|corr|>1的非法值，GPU实现对同样病态输入返回0（保险丝"
            "行为不同，非同一bug）；这10个体素0个落在M1冻结的voxel_mask保留列表"
            "里，即本来就会被BOLD-only QC排除，不进入任何ROI打分。"),
    },
    "UTS02": {
        "voxel_r_all": 0.9713,
        "voxel_r_mask_restricted": 0.994712,
        "n_voxels_total": 94251,
        "n_voxels_excluded_by_voxel_mask": 32,
        "original_threshold": 0.995,
        "original_criterion_pass": False,
        "supplementary_criterion_pass": True,
        "supplementary_criterion_basis": (
            "同UTS01第8条补充判据；voxel_mask限定后=0.994712，仍差0.000288未过原"
            "始阈值，但补充判据看的是「差异定位到具体机制+与ROI重叠可忽略」而非"
            "严格0.995——差异最大的100个体素里，0个落bilateral_PT，仅2个落"
            "left_IFG（占比0.26%），且这2个体素逐一核对确认是Rsq≈0的符号翻转"
            "（数值噪声范围内摆动，非实现错误）。"),
        "fail_reason_if_any": (
            "在UTS01的数值退化根因（10→32个体素，同机制）之外，还有一批Rsq极度"
            "接近0的体素在CPU/GPU不同线性代数库的浮点误差下符号翻转（corr= "
            "sqrt(|Rsq|)·sign(Rsq)对Rsq≈0极敏感）；voxel_mask限定后仍差0.000288"
            "未过原始阈值，走补充判据通过。"),
    },
    "UTS03": {
        "voxel_r_all": 0.9962,
        "voxel_r_mask_restricted": 0.9962,   # voxel_mask 对 UTS03 是恒等映射，0排除
        "n_voxels_total": 95556,
        "n_voxels_excluded_by_voxel_mask": 0,
        "original_threshold": 0.995,
        "original_criterion_pass": True,
        "supplementary_criterion_pass": None,   # 未触发，原判据直接过
        "supplementary_criterion_basis": "N/A——原始全体素判据直接PASS，未触发补充判据。",
        "fail_reason_if_any": "",
    },
}


def compute_roi_correlations(cfg: dict) -> pd.DataFrame:
    """逐 ROI 相关系数：需要 native 参照 + GPU 重跑两份 corrs.npz 都在本机。"""
    results_dir = Path(cfg["paths"]["results_dir"])
    frozen = Path(cfg["paths"]["frozen_dir"])
    rows = []
    for subj in SUBJECTS:
        ref_p = results_dir / "eng1000" / subj / "corrs.npz"
        gpu_p = results_dir / "eng1000_gpu_rerun" / subj / "corrs.npz"
        roi_p = frozen / f"roi_columns_{subj}.npz"
        missing = [str(p) for p in (ref_p, gpu_p, roi_p) if not p.exists()]
        if missing:
            print(f"  [跳过 {subj}] 缺文件：{missing}")
            continue
        ref = np.load(ref_p)
        ref = np.asarray(ref[ref.files[0]]).ravel()
        gpu = np.load(gpu_p)
        gpu = np.asarray(gpu[gpu.files[0]]).ravel()
        roi_cols = dict(np.load(roi_p))
        valid = np.isfinite(ref) & np.isfinite(gpu)
        for roi_name, cols in roi_cols.items():
            cols = np.asarray(cols)
            m = valid.copy()
            sel = np.zeros_like(valid)
            sel[cols] = True
            m &= sel
            r = float(np.corrcoef(ref[m], gpu[m])[0, 1]) if m.sum() > 1 else None
            rows.append({"subject": subj, "roi": roi_name, "n_voxels": int(m.sum()),
                        "ref_vs_gpu_pearson_r": r})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for subj, d in DOCUMENTED_GATE_RESULTS.items():
        rows.append({"subject": subj, **d})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "english1000_validation.csv", index=False)
    print(f"已写 {OUT_DIR / 'english1000_validation.csv'}（全体素/mask限定相关 + "
         "判据通过情况，来源见脚本内注释）")

    print("\n尝试计算逐 ROI（IFG/PT）相关系数（需要 native+GPU 两份 corrs.npz 都在本机）...")
    roi_df = compute_roi_correlations(cfg)
    if len(roi_df):
        roi_df.to_csv(OUT_DIR / "english1000_roi_correlations.csv", index=False)
        print(f"已写 {OUT_DIR / 'english1000_roi_correlations.csv'}")
    else:
        print("  没有任何被试凑齐所需文件，english1000_roi_correlations.csv 未生成——"
             "这是清单里「三名被试IFG相关/PT相关」这两项目前唯一还没有真实数字的"
             "地方，需要把 results/eng1000/ 和 results/eng1000_gpu_rerun/ 同步到"
             "本机（或在服务器上直接跑本脚本）才能补齐。")


if __name__ == "__main__":
    main()
