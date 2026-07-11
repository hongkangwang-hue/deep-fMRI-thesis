"""
M2 — fMRI 对齐与参考验证驱动。

按 --subject 逐被试独立执行（UTS03 已跑过并冻结，UTS01/UTS02 复用同一套规则）。

分步执行（--step）：
  voxel_mask : 在该被试全部故事并集上构建统一 BOLD-only voxel mask 并持久化到
               frozen/voxel_mask_{subject}.*（该被试的所有模型/条件复用同一保留
               列集合，体素总数因人而异，95556 只是 UTS03 的观测值）。
  roi        : aparc.a2009s → BOLD 列（主导顶点归属，冻结规则 B），与该被试的统一
               voxel mask 求交，校验 min_voxels，持久化到
               frozen/roi_columns_{subject}.*。pycortex transform 名默认从
               derivatives/subject_xfms.json 按 --subject 查找（可用 --xfm 覆盖）。

⚠️ voxel_mask / roi 均为轻量本地步骤（.hf5 流式统计、稀疏矩阵映射），无模型推理、
无 ridge。roi 依赖 pycortex + nibabel + 已 annex get 的 surface/transform/annot。
重计算（eng1000 ridge 参照）在后续 --step 单独标注。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.fmri.derivatives import (
    compute_bold_only_mask, list_subject_stories, save_voxel_mask)
from src.fmri import roi as roi_mod


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _dataset_version(cfg: dict) -> dict:
    """ds003020 数据版本（git HEAD + describe），用于 frozen 产物溯源。不含绝对路径。"""
    ds_root = Path(cfg["datasets"]["data_dir"]).parent.parent  # .../ds003020
    info = {"dataset": ds_root.name}
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ds_root, text=True).strip()
        info["describe"] = subprocess.check_output(
            ["git", "describe", "--always", "--tags"], cwd=ds_root, text=True).strip()
    except Exception:
        info["git_commit"] = "unknown"
    return info


def _provenance(cfg: dict) -> dict:
    """统一 provenance：代码 commit + 配置版本 + 数据版本。刻意不写本地绝对路径。"""
    return {
        "code_git_commit": _git_commit(),
        "config_version": cfg["version"],
        "dataset_version": _dataset_version(cfg),
    }


def step_voxel_mask(cfg: dict, subject: str) -> None:
    data_dir = cfg["datasets"]["data_dir"]
    stories = list_subject_stories(data_dir, subject)
    print(f"[voxel_mask] {subject}: {len(stories)} 个故事，逐故事流式 QC...")
    res = compute_bold_only_mask(data_dir, subject, stories)
    print(f"  voxel 总数: {res['n_voxels']}")
    print(f"  保留: {res['n_kept']}  "
          f"剔除(NaN): {res['n_excluded_nan']}  "
          f"剔除(零方差): {res['n_excluded_zero_var']}")
    out = save_voxel_mask(
        Path(cfg["paths"]["frozen_dir"]), res, provenance=_provenance(cfg))
    print(f"  保留列索引 → {out['index_path']}")
    print(f"  来源说明   → {out['meta_path']}")


def _derivatives_dir(cfg: dict) -> Path:
    # datasets.data_dir 指向 .../preprocessed_data，其父即 derivatives
    return Path(cfg["datasets"]["data_dir"]).parent


def _set_pycortex_filestore(store: Path) -> None:
    """把 pycortex filestore 写入用户 config（幂等），使 get_mapper 找到该被试。"""
    import configparser
    import os
    cfgpath = os.path.expanduser("~/.config/pycortex/options.cfg")
    cp = configparser.ConfigParser()
    cp.read(cfgpath)
    if "basic" not in cp:
        cp["basic"] = {}
    cp["basic"]["filestore"] = str(store)
    os.makedirs(os.path.dirname(cfgpath), exist_ok=True)
    with open(cfgpath, "w") as f:
        cp.write(f)


def step_roi(cfg: dict, subject: str, xfm: str | None = None) -> None:
    import nibabel as nib
    from scipy import ndimage

    deriv = _derivatives_dir(cfg)
    store = deriv / "pycortex-db"
    fs_dir = deriv / "freesurfer_subjdir"
    _set_pycortex_filestore(store)
    import cortex

    if xfm is None:
        # 逐被试 pycortex transform 名不同（UTS01_auto/UTS02_auto/...），从数据集
        # derivatives/subject_xfms.json 里查，不硬编码某个被试的名字。
        xfm = json.loads((deriv / "subject_xfms.json").read_text())[subject]

    roi_spec = json.loads(
        (Path(cfg["paths"]["frozen_dir"]) / "roi_spec.json").read_text())
    min_vox = roi_spec["bold_only_voxel_rule"]["min_voxels_per_roi"]

    # 1) thick mask → 列映射
    mask = np.asarray(cortex.db.get_mask(subject, xfm, "thick"))  # (54,84,84)
    SH = mask.shape
    flat = mask.ravel(order="C")
    full_to_col = roi_mod.full_to_column_map(flat)
    col_to_full = np.flatnonzero(flat)
    print(f"[roi] thick mask True = {int(flat.sum())} (= BOLD 列数)")

    # 2) 统一 voxel mask（求交用）
    keep = np.load(Path(cfg["paths"]["frozen_dir"]) / f"voxel_mask_{subject}.npy")

    # 3) annot + mapper → 每全volume体素主导 label
    def annot(hemi):
        lab, _, names = nib.freesurfer.read_annot(
            fs_dir / subject / "label" / f"{hemi}.aparc.a2009s.annot")
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        return lab, names
    lab_lh, names_lh = annot("lh")
    lab_rh, names_rh = annot("rh")
    m = cortex.get_mapper(subject, xfm, "line_nearest")
    M_lh, M_rh = m.masks[0].tocsc(), m.masks[1].tocsc()
    print("[roi] 计算各全volume体素主导顶点 label ...")
    w_lh, dl_lh = roi_mod.per_voxel_dominant_label(M_lh, lab_lh)
    w_rh, dl_rh = roi_mod.per_voxel_dominant_label(M_rh, lab_rh)
    dom_name, dom_hemi = roi_mod.combine_hemis_dominant(
        w_lh, dl_lh, names_lh, w_rh, dl_rh, names_rh)

    # 4) 逐 ROI 提列 + 求交 + 校验 + 空间终检
    out_cols, report = {}, {}
    for r in roi_spec["rois"]:
        full_vox = roi_mod.roi_full_voxels(
            dom_name, dom_hemi, r["labels"], r["hemi"])
        cols = roi_mod.columns_for_roi(full_vox, full_to_col, voxel_mask_keep=keep)
        # 空间终检：连通块
        ijk = np.array(np.unravel_index(col_to_full[cols], SH)).T
        vol = np.zeros(SH, bool); vol[tuple(ijk.T)] = True
        _, ncomp = ndimage.label(vol)
        sizes = np.bincount(ndimage.label(vol)[0].ravel())[1:]
        frac = float(sizes.max() / len(cols)) if len(cols) else 0.0
        out_cols[r["name"]] = cols
        report[r["name"]] = {
            "labels": r["labels"], "hemi": r["hemi"],
            "n_columns": int(len(cols)),
            "n_components": int(ncomp),
            "largest_component_frac": round(frac, 3),
            "centroid_zyx": [round(float(x), 1) for x in ijk.mean(0)],
            "passes_min_voxels": bool(len(cols) >= min_vox),
        }
        status = "OK" if len(cols) >= min_vox else "不足"
        print(f"  [{r['name']}] 列={len(cols)} ({status}≥{min_vox}) "
              f"连通块={ncomp} 最大块占比={frac:.2f}")
        if len(cols) < min_vox:
            raise SystemExit(f"ROI {r['name']} 体素数 {len(cols)} < {min_vox}")

    # 5) 持久化
    out_dir = Path(cfg["paths"]["frozen_dir"])
    npz_path = out_dir / f"roi_columns_{subject}.npz"
    np.savez(npz_path, **out_cols)
    meta_path = out_dir / f"roi_columns_{subject}.json"
    meta = {
        "subject": subject, "xfm": xfm,
        "atlas": roi_spec["atlas"],
        "assignment_rule": "dominant_vertex_line_nearest",
        # thick mask 体素数因被试而异（UTS03=95556），用实测值而非硬编码常数
        "column_space": f"thick_mask_{int(flat.sum())}_C_order",
        "intersected_with": f"voxel_mask_{subject}.npy",
        "min_voxels_per_roi": min_vox,
        "rois": report,
        "provenance": _provenance(cfg),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  ROI 列索引 → {npz_path}")
    print(f"  验证报告   → {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["voxel_mask", "roi"])
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--xfm", default=None,
                     help="pycortex transform 名；缺省时从 "
                          "derivatives/subject_xfms.json 按 --subject 查找")
    args = ap.parse_args()
    cfg = load_config()
    if args.step == "voxel_mask":
        step_voxel_mask(cfg, args.subject)
    elif args.step == "roi":
        step_roi(cfg, args.subject, xfm=args.xfm)


if __name__ == "__main__":
    main()