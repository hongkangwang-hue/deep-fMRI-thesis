"""
诊断脚本 —— AWD-LSTM 上下文记忆衰减曲线（细粒度 H 网格，非冻结矩阵）。

背景：冻结矩阵只在 H∈{8,32,128} 三点评分，之前的核实（corr(X8,X32)=0.99987,
corr(X32,X128)=1.000000）只能证明"两个端点间有/无差异"，不能证明这是平滑的
真实记忆衰减，还是某种台阶式实现问题。本脚本在更细的 H 网格上抽取同一批
目标词的 AWD-LSTM 主层特征，逐一与 H=128（渐近参考）比较，输出衰减曲线。

**不写入 frozen/、不写入任何正式结果目录、不影响 M1-M6 任何已冻结产物**——
纯诊断，读 word_index.parquet + 直接调用 adapter，输出只打印到终端。

用法（服务器，需 fastai+torch）：
  python scripts/diag_awd_lstm_context_decay.py --story souls --n-targets 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config          # noqa: E402
from src.models import get_adapter                  # noqa: E402
from src.models.base import LayerSpec                # noqa: E402
from src.models.windowing import iter_story_targets  # noqa: E402

H_GRID = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", default="souls")
    ap.add_argument("--n-targets", type=int, default=300,
                     help="该故事内取前 N 个 H=128 合格目标（控制成本，非全量）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config()
    word_index = pd.read_parquet(Path(cfg["paths"]["frozen_dir"]) / "word_index.parquet")
    story_targets = {s: (w, e) for s, w, e in iter_story_targets(word_index)}
    if args.story not in story_targets:
        raise SystemExit(f"故事不在 word_index 中: {args.story}")
    words, eligible = story_targets[args.story]
    targets = eligible[: args.n_targets]
    print(f"[diag] story={args.story} 词数={len(words)} 取目标={len(targets)}/{len(eligible)}",
          flush=True)

    layers = LayerSpec(main=cfg["models"]["primary_layers"]["awd_lstm"],
                       final=cfg["models"]["robustness_layers"]["awd_lstm"])
    adapter = get_adapter("awd_lstm", device=args.device)
    adapter.load()

    X = {}
    for H in H_GRID:
        reps = adapter.extract_batch(words, targets, H, layers, batch_size=len(targets))
        X[H] = np.stack([r.main for r in reps]).astype(np.float64)
        print(f"[diag] H={H:>3} 特征就绪 shape={X[H].shape}", flush=True)

    ref = X[128]
    print("\n[diag] === 相对渐近参考 H=128 的衰减曲线 ===", flush=True)
    print(f"{'H':>4} {'corr(X_H,X_128)':>18} {'1-corr^2':>12} "
          f"{'mean|diff|':>12} {'max|diff|':>12}", flush=True)
    for H in H_GRID:
        c = np.corrcoef(X[H].ravel(), ref.ravel())[0, 1]
        d = np.abs(X[H] - ref)
        print(f"{H:>4} {c:>18.6f} {1 - c**2:>12.6f} {d.mean():>12.6e} {d.max():>12.6e}",
              flush=True)

    print("\n[diag] === 相邻网格点差异（看衰减是否平滑单调，非台阶跳变）===", flush=True)
    for a, b in zip(H_GRID[:-1], H_GRID[1:]):
        c = np.corrcoef(X[a].ravel(), X[b].ravel())[0, 1]
        print(f"  corr(X_{a:>3}, X_{b:>3}) = {c:.6f}", flush=True)


if __name__ == "__main__":
    main()
