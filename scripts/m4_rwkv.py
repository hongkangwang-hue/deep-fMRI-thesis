"""
M4 —— rwkv 单模型驱动。跑该模型在给定 H/layers/folds 范围内的全部单元
（主层正常+shift双ROI + 最终层正常IFG），独立进程/独立日志/独立续跑。

共享逻辑见 src/ridge/m4_driver.py 顶部文档（冻结条件、三被试扩展说明、矩阵定义）。
四模型各一份入口脚本（m4_pythia.py/m4_mamba.py/m4_rwkv.py/m4_awd_lstm.py），方便
分别用不同 GPU/终端并行启动，互不阻塞。全部模型跑完后用
scripts/m4_aggregate.py 汇总（不跑计算，只扫已有结果文件）。

安全：含 himalaya ridge 拟合，仅 AutoDL 服务器运行，需用户确认后手动启动。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config                              # noqa: E402
from src.ridge.pipeline import himalaya_ridgecv_solver, numpy_ridgecv_solver  # noqa: E402
from src.ridge.m4_driver import run_model_matrix                        # noqa: E402
from src.ridge.assemble import remap_roi_columns_to_voxel_mask          # noqa: E402

MODEL = "rwkv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03",
                    help="三被试(UTS01/UTS02/UTS03)各跑一次，--subject 指定当前被试；"
                         "UTS03 pilot 期结果已存在可复用，新增的是 UTS01/UTS02")
    ap.add_argument("--H", nargs="+", type=int, default=[8, 32, 128])
    ap.add_argument("--folds", nargs="+", default=None,
                    help="frozen/fold_split.json 的键名子集；默认全部3折")
    ap.add_argument("--layers", nargs="+", default=["main", "final"],
                    choices=["main", "final"],
                    help="main=正常+shift双条件/双ROI；final=正常单条件/IFG only（P1稳健性）")
    ap.add_argument("--solver", default="himalaya", choices=["himalaya", "numpy"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-name", default="m4_full_matrix")
    ap.add_argument("--skip-existing", action="store_true",
                    help="单元结果文件已存在则跳过（断点续跑）")
    args = ap.parse_args()

    cfg = load_config()
    paths, ds = cfg["paths"], cfg["datasets"]
    seed = args.seed if args.seed is not None else cfg["seeds"]["pca"]
    solver = himalaya_ridgecv_solver if args.solver == "himalaya" else numpy_ridgecv_solver

    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        raw_fold_split = json.load(f)
    fold_names = args.folds if args.folds else list(raw_fold_split["folds"].keys())
    fold_split = {"folds": {k: v for k, v in raw_fold_split["folds"].items()
                            if k in fold_names}}

    # M1 冻结的 BOLD-only 保留列（剔除 NaN/零方差体素）；UTS03 是恒等映射，
    # UTS01/UTS02 会真正排除若干体素。喂给 assemble_all 压缩 Y，避免这些体素
    # 的 NaN 直接冲进 ridge 拟合导致 himalaya 报错（M3 竖切已实测触发过）。
    voxel_mask = np.load(Path(paths["frozen_dir"]) / f"voxel_mask_{args.subject}.npy")

    roi_cols_all = dict(np.load(Path(paths["frozen_dir"]) / f"roi_columns_{args.subject}.npz"))
    roi_cols_main_full = {k: v for k, v in roi_cols_all.items() if k in ("left_IFG", "bilateral_PT")}
    roi_cols_final_full = {"left_IFG": roi_cols_all["left_IFG"]}
    # roi_columns 里的列号是全量 BOLD 列空间编号；Y 被压缩到 voxel_mask 后同一
    # 体素的列号会整体前移，必须同步重映射，否则 ROI 打分会取到错误体素。
    roi_cols_main = remap_roi_columns_to_voxel_mask(roi_cols_main_full, voxel_mask)
    roi_cols_final = remap_roi_columns_to_voxel_mask(roi_cols_final_full, voxel_mask)

    out_dir = Path(paths["results_dir"]) / args.out_name / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    word_index_path = Path(paths["frozen_dir"]) / "word_index.parquet"

    print(f"[m4:{MODEL}] subject={args.subject}（三被试逐被试独立跑，见脚本顶部说明） "
          f"H={args.H} folds={fold_names} layers={args.layers} solver={args.solver} "
          f"dtype={args.dtype} seed={seed}", flush=True)

    run_model_matrix(MODEL, args.H, args.layers, fold_split, roi_cols_main, roi_cols_final,
                     paths["cache_dir"], ds["data_dir"], ds["respdict"], word_index_path,
                     solver, seed, args.dtype, out_dir, args.skip_existing, args.subject,
                     voxel_mask)

    print(f"[m4:{MODEL}] 完成。四模型都跑完后用 scripts/m4_aggregate.py 汇总。", flush=True)


if __name__ == "__main__":
    main()
