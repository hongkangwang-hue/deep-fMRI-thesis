"""
M2-C Phase 1 比较：我方 native 重跑的 corrs 对齐历史参考 corrs.npz。

按 frozen/m2c_reference_validation.yaml 的冻结指标判 PASS/FAIL。只读 (95556,) 小
数组，无重计算。Phase 1 实际的 ridge 重跑在服务器，本脚本只做产出后的对齐核验。

用法：
  python scripts/m2c_compare.py \
      --ours results/eng1000_native_rerun/UTS03/corrs.npz \
      --ref  results/eng1000/UTS03/corrs.npz \
      --ours-valphas results/eng1000_native_rerun/UTS03/valphas.npz \
      --ref-valphas  results/eng1000/UTS03/valphas.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent


def _load_vec(path: str) -> np.ndarray:
    d = np.load(path)
    return np.asarray(d[d.files[0]]).ravel()


def _alpha_grid_index(alphas: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """每个选定 alpha 在网格上的档位 index（取最近档）。"""
    return np.abs(np.log(alphas)[:, None] - np.log(grid)[None, :]).argmin(axis=1)


def compare(ours, ref, spec, ours_valphas=None, ref_valphas=None):
    valid = np.isfinite(ours) & np.isfinite(ref)
    o, r = ours[valid], ref[valid]
    delta = np.abs(o - r)
    results = {}
    m = spec["phase1_native"]["metrics"]

    vr = float(np.corrcoef(o, r)[0, 1])
    results["voxel_r_vector_pearson"] = (
        vr, vr >= m["voxel_r_vector_pearson"]["threshold_min"], ">=",
        m["voxel_r_vector_pearson"]["threshold_min"])

    med = float(np.median(delta))
    results["median_abs_delta_r"] = (
        med, med <= m["median_abs_delta_r"]["threshold_max"], "<=",
        m["median_abs_delta_r"]["threshold_max"])

    p95 = float(np.percentile(delta, 95))
    results["p95_abs_delta_r"] = (
        p95, p95 <= m["p95_abs_delta_r"]["threshold_max"], "<=",
        m["p95_abs_delta_r"]["threshold_max"])

    if ours_valphas is not None and ref_valphas is not None:
        grid = np.logspace(1, 3, 10)
        oi = _alpha_grid_index(ours_valphas[valid], grid)
        ri = _alpha_grid_index(ref_valphas[valid], grid)
        frac = float(np.mean(np.abs(oi - ri) <= 1))
        results["valpha_within_1_grid_step_frac"] = (
            frac, frac >= m["valpha_within_1_grid_step_frac"]["threshold_min"],
            ">=", m["valpha_within_1_grid_step_frac"]["threshold_min"])

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", default=None,
                    help="缺省时从 --ours 路径推断被试，用 "
                         "results/eng1000/<subject>/corrs.npz")
    ap.add_argument("--ours-valphas")
    ap.add_argument("--ref-valphas", default=None,
                    help="缺省时同 --ref 的被试推断规则")
    ap.add_argument("--spec",
                    default="frozen/m2c_reference_validation.yaml")
    args = ap.parse_args()

    if args.ref is None or args.ref_valphas is None:
        # --ours 形如 results/<out_name>/<subject>/corrs.npz，从中取 subject，
        # 不硬编码某一个被试（原来固定指向 UTS03，换被试忘了传 --ref 会静默
        # 拿 UTS03 的参照去对比新被试的结果，对比数字看似正常实则无意义）。
        subject = Path(args.ours).parent.name
        if args.ref is None:
            args.ref = f"results/eng1000/{subject}/corrs.npz"
        if args.ref_valphas is None:
            args.ref_valphas = f"results/eng1000/{subject}/valphas.npz"

    spec = yaml.safe_load((PROJECT_ROOT / args.spec).read_text())
    ours, ref = _load_vec(args.ours), _load_vec(args.ref)
    ov = _load_vec(args.ours_valphas) if args.ours_valphas else None
    rv = _load_vec(args.ref_valphas) if (args.ours_valphas and args.ref_valphas) else None

    print(f"我方: {args.ours}  ({ours.shape})")
    print(f"参考: {args.ref}  ({ref.shape})")
    res = compare(ours, ref, spec, ov, rv)
    all_pass = True
    print(f"\n{'指标':32} {'值':>10}  {'判据':>14}  结果")
    for name, (val, ok, op, thr) in res.items():
        all_pass &= ok
        print(f"{name:32} {val:>10.4f}  {op+' '+str(thr):>14}  {'PASS' if ok else 'FAIL'}")
    print(f"\nPhase 1 总判定: {'PASS' if all_pass else 'FAIL（先定位差异来源，不放宽容差）'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
