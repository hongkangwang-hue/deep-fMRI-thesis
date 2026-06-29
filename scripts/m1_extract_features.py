"""
M1 — 单故事 smoke test + 容量试跑（特征提取主脚本）

⚠️ 计算约束：本脚本会加载语言模型并做 GPU 前向，属于重计算，必须在 AutoDL
GPU 环境上运行，且确认显存/磁盘充足后再跑。默认只跑「一个故事、可限制目标
数」的 smoke，不做全量。全量矩阵在 M3/M4 才按 W_common 展开。

用法（在 AutoDL 上）：
  # 单模型 smoke：取一个中等长度故事的前 200 个有效目标
  python scripts/m1_extract_features.py --models pythia --story souls \
      --max-targets 200 --device cuda

  # 容量试跑：四模型 × 3 个 H，完整一个故事，输出容量报告
  python scripts/m1_extract_features.py --models pythia rwkv mamba awd_lstm \
      --story souls --device cuda --all-targets --capacity-report

本地（无 torch）只能验证参数解析与目标枚举，不能实际提取。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "encoding"))

from src.config_loader import load_config
from src.models import get_adapter
from src.models.base import LayerSpec
from src.models.feature_cache import save_features
from src.models.token_map import make_token_map, save_token_map, validate_token_map
from src.models.windowing import iter_story_targets


def layer_spec_for(cfg: dict, model: str) -> LayerSpec:
    return LayerSpec(
        main=cfg["models"]["primary_layers"][model],
        final=cfg["models"]["robustness_layers"][model],
    )


def extract_story_model(
    adapter, model: str, story: str, words: list[str],
    eligible_ids: list[int], word_id_base: pd.DataFrame,
    H_list: list[int], layers: LayerSpec, cfg: dict,
    max_targets: int | None,
):
    """对一个 (story, model)，按各 H 提取所有有效目标的双层表示并写缓存。

    Returns: (dict(H -> 计时与形状信息), token_map 行列表)。
    """
    cache_dir = cfg["paths"]["cache_dir"]
    local_to_global = dict(
        zip(word_id_base["word_local_id"], word_id_base["word_id"])
    )

    targets = eligible_ids if max_targets is None else eligible_ids[:max_targets]
    stats = {}
    token_rows = []

    for H in H_list:
        main_rows, final_rows, wid_rows, unk_rows = [], [], [], []
        t0 = time.perf_counter()
        for local_id in targets:
            rep = adapter.extract(words, local_id, H, layers)
            gid = int(local_to_global[local_id])
            main_rows.append(rep.main)
            final_rows.append(rep.final)
            wid_rows.append(gid)
            unk_rows.append(rep.is_unk)
            token_rows.append({
                "word_id": gid, "story": story,
                "word_local_id": int(local_id), "H": H,
                "target_token_index": rep.target_token_index,
                "n_tokens": rep.n_tokens,
                "n_target_subtokens": 1,
                "is_unk": rep.is_unk,
            })
        elapsed = time.perf_counter() - t0

        main_arr = np.stack(main_rows)
        final_arr = np.stack(final_rows)
        meta = {
            "model_id": adapter.model_id,
            "revision": adapter.revision,
            "layer_main": layers.main,
            "layer_final": layers.final,
            "code_version": cfg["version"],
        }
        save_features(
            cache_dir, model, story, H,
            np.array(wid_rows), main_arr, final_arr, np.array(unk_rows), meta,
        )
        stats[H] = {
            "n_targets": len(targets),
            "seconds": round(elapsed, 2),
            "sec_per_1k": round(elapsed / max(1, len(targets)) * 1000, 2),
            "main_shape": list(main_arr.shape),
            "final_shape": list(final_arr.shape),
            "unk_rate": round(float(np.mean(unk_rows)), 4),
        }
        print(f"    [{model} | {story} | H={H}] {len(targets)} 目标, "
              f"{elapsed:.1f}s ({stats[H]['sec_per_1k']}s/1k), "
              f"main={main_arr.shape}, unk={stats[H]['unk_rate']}")

    return stats, token_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["pythia"],
                    choices=["pythia", "rwkv", "mamba", "awd_lstm"])
    ap.add_argument("--story", default="souls")
    ap.add_argument("--max-targets", type=int, default=200,
                    help="每个 H 最多提取多少目标（smoke 用）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--capacity-report", action="store_true")
    ap.add_argument("--all-targets", action="store_true",
                    help="忽略 max-targets，提取该故事全部有效目标")
    args = ap.parse_args()

    cfg = load_config()
    H_list = cfg["models"]["contexts_preceding_words"]
    word_index = pd.read_parquet(
        Path(cfg["paths"]["frozen_dir"]) / "word_index.parquet"
    )

    story_targets = {s: (w, e) for s, w, e in iter_story_targets(word_index)}
    if args.story not in story_targets:
        raise SystemExit(f"故事 {args.story} 不在 word_index 中")
    words, eligible = story_targets[args.story]
    base = word_index[word_index["story"] == args.story]
    max_t = None if args.all_targets else args.max_targets

    print(f"故事 {args.story}: {len(words)} 词, {len(eligible)} 个 H=128 有效目标")
    print(f"提取目标数: {'全部' if max_t is None else max_t}, 模型: {args.models}")

    report = {"story": args.story, "models": {}}
    for model in args.models:
        print(f"\n=== 加载 {model} ===")
        adapter = get_adapter(model, device=args.device)
        adapter.load()
        print(f"  {adapter.audit_row()}")
        layers = layer_spec_for(cfg, model)

        stats, token_rows = extract_story_model(
            adapter, model, args.story, words, eligible, base,
            H_list, layers, cfg, max_t,
        )

        tm = make_token_map(token_rows)
        validate_token_map(tm, word_index)
        tm_path = (Path(cfg["paths"]["cache_dir"]) / model
                   / f"{args.story}_token_map.parquet")
        save_token_map(tm, tm_path)
        report["models"][model] = {"audit": adapter.audit_row(), "by_H": stats}

    if args.capacity_report:
        _write_capacity_report(cfg, report, word_index, H_list)

    print("\nM1 smoke 完成。")


def _write_capacity_report(cfg, report, word_index, H_list):
    """基于本次实测外推全量 GPU 小时（含 25% 缓冲）。"""
    total_eligible = int(word_index["eligible_h128"].sum())
    out = {"measured": report, "extrapolation": {}}
    for model, m in report["models"].items():
        sec_per_1k = float(np.mean([h["sec_per_1k"] for h in m["by_H"].values()]))
        total_units = total_eligible * len(H_list)
        gpu_hours = sec_per_1k * total_units / 1000 / 3600 * 1.25
        out["extrapolation"][model] = {
            "total_eligible_targets": total_eligible,
            "H_count": len(H_list),
            "est_gpu_hours_with_25pct_buffer": round(gpu_hours, 2),
        }
    path = Path(cfg["paths"]["results_dir"]) / "m1_capacity_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n容量报告写入: {path}")


if __name__ == "__main__":
    main()
