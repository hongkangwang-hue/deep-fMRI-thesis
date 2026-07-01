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
from src.models.feature_cache import save_features, cache_path
from src.models.token_map import make_token_map, save_token_map, validate_token_map
from src.models.windowing import iter_story_targets


def layer_spec_for(cfg: dict, model: str) -> LayerSpec:
    return LayerSpec(
        main=cfg["models"]["primary_layers"][model],
        final=cfg["models"]["robustness_layers"][model],
    )


def verify_batch_matches_single(adapter, words, targets, H, layers, n):
    """对前 n 个目标，断言批量提取与逐窗 extract 逐元素一致（容差内）。

    这是批量路径的正确性闸门：right-padding/mask 的 bug 会让目标 hidden 大幅
    偏离（远超浮点误差），而非 ~1e-4。返回观测到的最大绝对差。
    """
    sample = list(targets[:n])
    if not sample:
        return 0.0
    singles = [adapter.extract(words, i, H, layers) for i in sample]
    batched = adapter.extract_batch(words, sample, H, layers, batch_size=len(sample))
    max_diff = 0.0
    for s, b in zip(singles, batched):
        assert s.target_token_index == b.target_token_index, "目标 token 位置不一致"
        assert s.n_tokens == b.n_tokens, "token 数不一致"
        max_diff = max(
            max_diff,
            float(np.abs(s.main - b.main).max()),
            float(np.abs(s.final - b.final).max()),
        )
    return max_diff


def extract_story_model(
    adapter, model: str, story: str, words: list[str],
    eligible_ids: list[int], word_id_base: pd.DataFrame,
    H_list: list[int], layers: LayerSpec, cfg: dict,
    max_targets: int | None, batch_size: int,
):
    """对一个 (story, model)，按各 H 批量提取所有有效目标的双层表示并写缓存。

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
        reps = adapter.extract_batch(words, targets, H, layers, batch_size)
        for local_id, rep in zip(targets, reps):
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
                "n_target_subtokens": rep.n_target_subtokens,
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


def resolve_stories(args, story_targets, cfg) -> list[str]:
    """按参数决定要提取的故事。优先级：all-stories > from-fold-split > stories > story > 默认 souls。"""
    if args.all_stories:
        return sorted(story_targets)
    if args.from_fold_split:
        with open(Path(cfg["paths"]["frozen_dir"]) / "fold_split.json") as f:
            fs = json.load(f)
        return sorted({s for fo in fs["folds"].values()
                       for s in (fo["train_stories"] + fo["test_stories"])})
    if args.stories:
        return list(args.stories)
    if args.story:
        return [args.story]
    return ["souls"]


def _all_cached(cache_dir, model, story, H_list) -> bool:
    """该故事请求的所有 H 是否都已缓存（断点续跑判据）。"""
    return all(cache_path(cache_dir, model, story, H).exists() for H in H_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["pythia"],
                    choices=["pythia", "rwkv", "mamba", "awd_lstm"])
    # 故事选择（四选一，优先级见 resolve_stories）
    ap.add_argument("--story", default=None, help="单故事（smoke）")
    ap.add_argument("--stories", nargs="+", default=None, help="显式多故事列表")
    ap.add_argument("--all-stories", action="store_true",
                    help="提取 word_index 全部故事（M1b 全量）")
    ap.add_argument("--from-fold-split", action="store_true",
                    help="提取 fold_split.json 的 83 个 CV 故事（M3/M4）")
    ap.add_argument("--H", nargs="+", type=int, default=None,
                    help="只提指定 H（如 --H 128）；默认用 config 全部 [8,32,128]")
    ap.add_argument("--max-targets", type=int, default=200,
                    help="每个 H 最多提取多少目标（smoke 用）")
    ap.add_argument("--all-targets", action="store_true",
                    help="忽略 max-targets，提取该故事全部有效目标")
    ap.add_argument("--skip-existing", action="store_true",
                    help="某故事请求的所有 H 已缓存则跳过（断点续跑）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="批量前向的窗口数（右侧 padding，因果模型结果不变）")
    ap.add_argument("--verify-n", type=int, default=4,
                    help="加载后用前 N 个目标核验「批量==逐窗」，0 跳过")
    ap.add_argument("--capacity-report", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    H_list = args.H if args.H else cfg["models"]["contexts_preceding_words"]
    word_index = pd.read_parquet(
        Path(cfg["paths"]["frozen_dir"]) / "word_index.parquet"
    )

    story_targets = {s: (w, e) for s, w, e in iter_story_targets(word_index)}
    stories = resolve_stories(args, story_targets, cfg)
    missing = [s for s in stories if s not in story_targets]
    if missing:
        raise SystemExit(f"故事不在 word_index 中: {missing}")
    max_t = None if args.all_targets else args.max_targets
    cache_dir = cfg["paths"]["cache_dir"]

    print(f"提取 {len(stories)} 个故事 | H={H_list} | "
          f"目标={'全部' if max_t is None else max_t} | 模型={args.models}")

    report = {"stories": stories, "H_list": H_list, "models": {}}
    for model in args.models:
        print(f"\n=== 加载 {model} ===")
        adapter = get_adapter(model, device=args.device)
        adapter.load()
        print(f"  {adapter.audit_row()}")
        layers = layer_spec_for(cfg, model)

        # verify 只对第一个故事做一次（最长上下文最易暴露 padding/mask bug）
        if args.verify_n > 0:
            w0, e0 = story_targets[stories[0]]
            H_check = max(H_list)
            md = verify_batch_matches_single(
                adapter, w0, e0, H_check, layers, args.verify_n)
            tol = 2e-3
            status = "OK" if md < tol else "失败"
            print(f"  [verify] 批量vs逐窗 (H={H_check}, n={args.verify_n}): "
                  f"max|Δ|={md:.2e} ({status})")
            if md >= tol:
                raise SystemExit(
                    f"批量提取与逐窗结果不一致 (max|Δ|={md:.2e} >= {tol})，"
                    f"疑似 padding/mask bug，已中止。")

        model_stats = {}
        t_model = time.perf_counter()
        for i, story in enumerate(stories, 1):
            if args.skip_existing and _all_cached(cache_dir, model, story, H_list):
                print(f"  [{i}/{len(stories)}] {story} 已缓存，跳过")
                continue
            words, eligible = story_targets[story]
            base = word_index[word_index["story"] == story]
            print(f"  [{i}/{len(stories)}] {story}: {len(words)} 词, "
                  f"{len(eligible)} 有效目标")
            stats, token_rows = extract_story_model(
                adapter, model, story, words, eligible, base,
                H_list, layers, cfg, max_t, args.batch_size,
            )
            tm = make_token_map(token_rows)
            validate_token_map(tm, word_index)
            tm_path = Path(cache_dir) / model / f"{story}_token_map.parquet"
            save_token_map(tm, tm_path)
            model_stats[story] = stats
        print(f"  {model} 完成 {len(stories)} 故事，"
              f"{(time.perf_counter()-t_model)/60:.1f} 分钟")
        report["models"][model] = {"audit": adapter.audit_row(),
                                   "by_story": model_stats}

    if args.capacity_report:
        _write_capacity_report(cfg, report, word_index, H_list)

    print("\nM1 提取完成。")


def _write_capacity_report(cfg, report, word_index, H_list):
    """基于本次实测外推全量 GPU 小时（含 25% 缓冲）。"""
    total_eligible = int(word_index["eligible_h128"].sum())
    out = {"measured": report, "extrapolation": {}}
    for model, m in report["models"].items():
        by_story = m["by_story"]
        if not by_story:
            continue
        all_sec = [h["sec_per_1k"] for st in by_story.values() for h in st.values()]
        sec_per_1k = float(np.mean(all_sec))
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
