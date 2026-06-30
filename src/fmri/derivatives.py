"""
Derivatives 加载与 BOLD-only voxel QC（统一 voxel mask 构建）。

设计契约（已核对，见 M2-A derivatives 核对）：
  - 响应 .hf5 形状 (n_trs, n_voxels=95556)，预处理时已按体素 z-score（训练故事
    体素 std=1）；测试故事 wheretheressmoke 为跨重复平均（std<1）。loader 一律
    不再 z-score（与参考 get_response 一致）。
  - BOLD-only QC 与模型无关：在所有目标故事的并集上，剔除任一故事中含 NaN 或
    零方差的体素列。最终评分列 = ROI(label) ∩ 本统一 mask。
  - mask 以「显式保留列索引 + 来源说明」持久化，所有模型/条件复用同一列集合，
    禁止各自重算导致的静默列错位。

QC 规则来自 frozen/roi_spec.json: exclude_nan, exclude_zero_variance。
min_voxels_per_roi 为 ROI 级检查，不在本全局 mask 内实施。
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

N_VOXELS_EXPECTED = 95556
ZERO_VAR_EPS = 0.0  # 严格零方差（std==0）才剔除；预处理已 z-score，正常体素 std≈1


def response_path(data_dir: str | Path, subject: str, story: str) -> Path:
    # data_dir 即 config 的 datasets.data_dir，已指向 .../preprocessed_data
    return Path(data_dir) / subject / f"{story}.hf5"


def load_response(data_dir: str | Path, subject: str, story: str,
                  columns: np.ndarray | None = None) -> np.ndarray:
    """加载单故事响应 (n_trs, n_voxels)，不再 z-score。

    Args:
        columns: 可选保留列索引；给定则只返回这些列（用于复用统一 mask）。
    """
    with h5py.File(response_path(data_dir, subject, story), "r") as f:
        data = f["data"][:]
    if columns is not None:
        data = data[:, columns]
    return np.asarray(data)


def list_subject_stories(data_dir: str | Path, subject: str) -> list[str]:
    base = Path(data_dir) / subject
    return sorted(p.stem for p in base.glob("*.hf5"))


def compute_bold_only_mask(data_dir: str | Path, subject: str,
                           stories: list[str]):
    """在所有给定故事并集上构建 BOLD-only 有效 voxel mask。

    逐故事流式处理（每次仅驻留一个故事），累积：
      - has_nan[v]:  该体素在任一故事中出现 NaN；
      - zero_var[v]: 该体素在任一故事中 std==0。
    有效体素 = 非 has_nan 且 非 zero_var。

    Returns:
        dict 含 keep_index(保留列索引), valid_mask(bool[n_voxels]),
        n_voxels, n_kept, excluded 统计与逐故事形状，用于持久化与审计。
    """
    if not stories:
        raise ValueError("stories 为空，无法构建 voxel mask（检查 data_dir/subject 路径）")
    n_vox = None
    has_nan = None
    zero_var = None
    per_story_shape = {}

    for story in stories:
        data = load_response(data_dir, subject, story)
        if n_vox is None:
            n_vox = data.shape[1]
            has_nan = np.zeros(n_vox, dtype=bool)
            zero_var = np.zeros(n_vox, dtype=bool)
        elif data.shape[1] != n_vox:
            raise ValueError(
                f"{story} voxel 数 {data.shape[1]} != {n_vox}，列空间不一致")
        per_story_shape[story] = list(data.shape)
        has_nan |= np.isnan(data).any(axis=0)
        # 对含 NaN 的列，std 会是 NaN；单独判零方差时忽略已标 NaN 的列
        std = np.nanstd(data, axis=0)
        zero_var |= (std <= ZERO_VAR_EPS)
        del data

    valid_mask = ~(has_nan | zero_var)
    keep_index = np.flatnonzero(valid_mask).astype(np.int64)
    return {
        "subject": subject,
        "n_voxels": int(n_vox),
        "n_kept": int(valid_mask.sum()),
        "n_excluded_nan": int(has_nan.sum()),
        "n_excluded_zero_var": int((zero_var & ~has_nan).sum()),
        "keep_index": keep_index,
        "valid_mask": valid_mask,
        "stories": list(stories),
        "per_story_shape": per_story_shape,
    }


def save_voxel_mask(out_dir: str | Path, result: dict,
                    provenance: dict | None = None) -> dict:
    """持久化统一 voxel mask：显式保留列索引 (.npy) + 来源说明 (.json)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = out_dir / f"voxel_mask_{result['subject']}.npy"
    meta_path = out_dir / f"voxel_mask_{result['subject']}.json"
    np.save(idx_path, result["keep_index"])
    meta = {
        "subject": result["subject"],
        "n_voxels": result["n_voxels"],
        "n_kept": result["n_kept"],
        "n_excluded_nan": result["n_excluded_nan"],
        "n_excluded_zero_var": result["n_excluded_zero_var"],
        "n_stories": len(result["stories"]),
        "stories": result["stories"],
        "qc_rule": {"exclude_nan": True, "exclude_zero_variance": True,
                    "zero_var_eps": ZERO_VAR_EPS},
        "keep_index_file": idx_path.name,
        "note": ("统一 BOLD-only voxel mask，在全部故事并集上构建；所有模型/条件"
                 "复用同一保留列集合。最终评分列 = ROI(label) ∩ 本 keep_index。"),
    }
    if provenance:
        meta["provenance"] = provenance
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"index_path": str(idx_path), "meta_path": str(meta_path)}