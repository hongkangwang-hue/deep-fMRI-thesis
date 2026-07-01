"""
M4 —— 跨模型结果汇总。不跑任何计算，只扫描 4 个模型脚本
（scripts/m4_{pythia,mamba,rwkv,awd_lstm}.py）各自独立写下的单元结果文件
（results/<out-name>/<subject>/cells/），重建总表并核对 M4 验收标准 1-6。

可在任何模型未跑完时运行，用于查看进度（缺失单元会列在 manifest 里）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config          # noqa: E402
from src.ridge.m4_driver import build_manifest, ALL_MODELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--H", nargs="+", type=int, default=[8, 32, 128])
    ap.add_argument("--folds", nargs="+", default=None,
                    help="frozen/fold_split.json 的键名子集；默认全部3折")
    ap.add_argument("--out-name", default="m4_full_matrix")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        raw_fold_split = json.load(f)
    fold_names = args.folds if args.folds else list(raw_fold_split["folds"].keys())

    out_dir = Path(paths["results_dir"]) / args.out_name / args.subject
    build_manifest(out_dir, args.models, args.H, fold_names, args.subject)


if __name__ == "__main__":
    main()
