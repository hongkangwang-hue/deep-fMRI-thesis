"""
Step 3 —— 刺激侧上下文控制（C1/ctx1）特征重提取。

⚠️ 计算约束：本脚本加载语言模型并做 GPU 前向，属于重计算，必须在服务器 GPU
环境上运行，且需你确认后再执行（本机无 torch/GPU，无法本地跑）。上 GPU 前
必须先过两道零算力闸门：
    1. 实验补充/scripts/m4s_assert_impl.py（已通过，见 results/step2_diagnostics.json）
    2. 实验补充/tests/（15+15+5=35 单测全过，含 test_extract_batch_override_wiring.py
       对 base.py windows_override 接线的因果依赖验证）

与 scripts/m1_extract_features.py 的关系：结构逐行对应（CLI 参数、
resolve_stories、verify batch==single、capacity report 全部复用同一套逻辑），
唯一区别是：
    - 每个目标窗口的上下文改用 build_perturbed_window（乱序），目标词本身、
      时间戳、pooling、state-reset 逻辑完全不变；
    - 写入独立缓存目录 cache/features_ctx1/，不覆盖 cache/features/；
    - meta 里多记 condition/master_seed/permutation_sha256；
    - 额外落盘置换 manifest（每故事一份 .npy + 汇总 SHA-256 表），供审计
      （V6.4.2 §0.7："manifest 至少记录：master seed；story ID；derived
      story seed；condition；permutation length；permutation file；
      permutation SHA-256"）。

用法（在服务器上，M4S-Core 默认档：3 核心模型 × H∈{8,128} × 83 个 CV 故事）：
  python 实验补充/scripts/m4s_extract_perturbed_features.py \
      --models pythia rwkv mamba --H 8 128 --from-fold-split --device cuda

  # 算力紧可先跑 L2（仅 Mamba+Pythia）：
  python 实验补充/scripts/m4s_extract_perturbed_features.py \
      --models pythia mamba --H 8 128 --from-fold-split --device cuda

本地（无 torch）只能验证参数解析与置换/目标枚举，不能实际提取。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SUPPLEMENT_ROOT = SCRIPT_DIR.parent            # 实验补充/
PROJECT_ROOT = SUPPLEMENT_ROOT.parent           # 仓库根目录
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "encoding"))
sys.path.insert(0, str(SUPPLEMENT_ROOT / "src"))

from src.config_loader import load_config                                    # noqa: E402
from src.models import get_adapter                                            # noqa: E402
from src.models.base import LayerSpec                                         # noqa: E402
from src.models.feature_cache import save_features, cache_path                # noqa: E402
from src.models.token_map import make_token_map, save_token_map, validate_token_map  # noqa: E402
from src.models.windowing import iter_story_targets                           # noqa: E402
from context_perturb import story_permutation, build_perturbed_window         # noqa: E402

CONDITION = "ctx1"
MASTER_SEED = 20260724   # 见 ../config/freeze_manifest.json，与 V6.4.2 §0.7 一致


def layer_spec_for(cfg: dict, model: str) -> LayerSpec:
    return LayerSpec(
        main=cfg["models"]["primary_layers"][model],
        final=cfg["models"]["robustness_layers"][model],
    )


def _perm_sha256(perm: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(perm).tobytes()).hexdigest()


def save_permutation(perm_dir: Path, story: str, perm: np.ndarray) -> tuple[Path, str]:
    """置换落盘（独立于任何模型/被试——见 m4s_assert_impl.py 的 C/D 结构性保证），
    返回 (文件路径, SHA-256)。同一故事只应调用一次；重复调用会得到内容相同、
    覆盖写入的同一份文件（story_permutation 本身是确定性函数）。"""
    perm_dir.mkdir(parents=True, exist_ok=True)
    path = perm_dir / f"{story}.npy"
    np.save(path, perm)
    return path, _perm_sha256(perm)


def verify_batch_matches_single_perturbed(adapter, words, perm, targets, H, layers, n):
    """ctx1 版的批量↔逐窗一致性核验（对应 m1 的 verify_batch_matches_single，
    但窗口内容换成乱序上下文）。同样是正确性闸门：padding/mask 的 bug 会让
    目标 hidden 大幅偏离，而非 ~1e-4 浮点误差。"""
    sample = list(targets[:n])
    if not sample:
        return 0.0
    windows = [build_perturbed_window(words, perm, i, H) for i in sample]
    singles = [
        adapter.extract(words, i, H, layers, window_override=w)
        for i, w in zip(sample, windows)
    ]
    batched = adapter.extract_batch(
        words, sample, H, layers, batch_size=len(sample), windows_override=windows
    )
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


def extract_story_model_ctx1(
    adapter, model: str, story: str, words: list[str], perm: np.ndarray,
    eligible_ids: list[int], word_id_base: pd.DataFrame,
    H_list: list[int], layers: LayerSpec, cfg: dict,
    max_targets: int | None, batch_size: int, cache_dir_ctx1: str, perm_sha256: str,
):
    """对一个 (story, model)，按各 H 批量提取 ctx1 条件下所有有效目标的双层
    表示并写缓存。与 m1_extract_features.py::extract_story_model 结构对应，
    唯一区别是窗口来自 build_perturbed_window 而非 build_window。

    Returns: (dict(H -> 计时与形状信息), token_map 行列表)。
    """
    local_to_global = dict(
        zip(word_id_base["word_local_id"], word_id_base["word_id"])
    )

    targets = eligible_ids if max_targets is None else eligible_ids[:max_targets]
    stats = {}
    token_rows = []

    for H in H_list:
        main_rows, final_rows, wid_rows, unk_rows = [], [], [], []
        t0 = time.perf_counter()

        windows = [build_perturbed_window(words, perm, i, H) for i in targets]
        reps = adapter.extract_batch(
            words, targets, H, layers, batch_size, windows_override=windows
        )
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
            "condition": CONDITION,
            "master_seed": MASTER_SEED,
            "permutation_sha256": perm_sha256,
        }
        save_features(
            cache_dir_ctx1, model, story, H,
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
        print(f"    [{model} | {story} | H={H} | {CONDITION}] {len(targets)} 目标, "
              f"{elapsed:.1f}s ({stats[H]['sec_per_1k']}s/1k), "
              f"main={main_arr.shape}, unk={stats[H]['unk_rate']}")

    return stats, token_rows


def resolve_stories(args, story_targets, cfg) -> list[str]:
    """与 m1 一致：all-stories > from-fold-split > stories > story。
    M4S-Core 的 54 拟合单元来自 fold_split 的 83 个 CV 故事，故默认建议用
    --from-fold-split（见文件头用法示例）。"""
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
    return all(cache_path(cache_dir, model, story, H).exists() for H in H_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["pythia", "rwkv", "mamba"],
                    choices=["pythia", "rwkv", "mamba", "awd_lstm"],
                    help="M4S-Core 默认三核心模型（不含 awd_lstm）")
    ap.add_argument("--story", default=None, help="单故事（smoke）")
    ap.add_argument("--stories", nargs="+", default=None, help="显式多故事列表")
    ap.add_argument("--all-stories", action="store_true",
                    help="提取 word_index 全部 84 故事（含 held-out）")
    ap.add_argument("--from-fold-split", action="store_true",
                    help="提取 fold_split.json 的 83 个 CV 故事（M4S-Core 默认范围）")
    ap.add_argument("--H", nargs="+", type=int, default=[8, 128],
                    help="M4S-Core scope 默认 {8,128}；L0/Extended 可传 --H 8 32 128")
    ap.add_argument("--max-targets", type=int, default=None,
                    help="每个 H 最多提取多少目标（smoke 用）；默认 None=全部")
    ap.add_argument("--skip-existing", action="store_true",
                    help="某故事请求的所有 H 已缓存则跳过（断点续跑）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--verify-n", type=int, default=4,
                    help="加载后用前 N 个目标核验「批量==逐窗」（乱序窗口版），0 跳过")
    ap.add_argument("--cache-dir-ctx1", default=None,
                    help="默认 <cache_dir 同级>/features_ctx1（不覆盖 normal 的 features/）")
    args = ap.parse_args()

    cfg = load_config()
    H_list = args.H
    word_index = pd.read_parquet(
        Path(cfg["paths"]["frozen_dir"]) / "word_index.parquet"
    )

    cache_dir_normal = Path(cfg["paths"]["cache_dir"])
    cache_dir_ctx1 = (
        Path(args.cache_dir_ctx1) if args.cache_dir_ctx1
        else cache_dir_normal.parent / "features_ctx1"
    )
    perm_dir = cache_dir_normal.parent / "permutations_ctx1"

    story_targets = {s: (w, e) for s, w, e in iter_story_targets(word_index)}
    stories = resolve_stories(args, story_targets, cfg)
    missing = [s for s in stories if s not in story_targets]
    if missing:
        raise SystemExit(f"故事不在 word_index 中: {missing}")
    cache_dir_ctx1_s = str(cache_dir_ctx1)

    print(f"[ctx1] 提取 {len(stories)} 个故事 | H={H_list} | 模型={args.models}")
    print(f"[ctx1] 特征缓存目录: {cache_dir_ctx1_s}（不碰 normal: {cache_dir_normal}）")
    print(f"[ctx1] 置换缓存目录: {perm_dir}")
    print(f"[ctx1] master_seed={MASTER_SEED}")

    # 每篇故事只生成一次置换，落盘 + 记录 SHA-256（V6.4.2 §0.3/§0.7）。
    # 所有模型、H、被试共用同一份置换（结构性保证，见 m4s_assert_impl.py 的 C/D）。
    perms: dict[str, np.ndarray] = {}
    perm_manifest_rows = []
    for story in stories:
        words, _ = story_targets[story]
        perm = story_permutation(story, MASTER_SEED, len(words))
        path, sha = save_permutation(perm_dir, story, perm)
        perms[story] = perm
        perm_manifest_rows.append({
            "story": story, "n_words": len(words), "master_seed": MASTER_SEED,
            "permutation_file": str(path), "permutation_sha256": sha,
        })
    manifest_path = SUPPLEMENT_ROOT / "results" / "step3_permutation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({"condition": CONDITION, "master_seed": MASTER_SEED,
                   "stories": perm_manifest_rows}, f, indent=2, ensure_ascii=False)
    print(f"[ctx1] 置换 manifest 已写: {manifest_path}")

    report = {"condition": CONDITION, "stories": stories, "H_list": H_list, "models": {}}
    for model in args.models:
        print(f"\n=== 加载 {model} ===")
        adapter = get_adapter(model, device=args.device)
        adapter.load()
        print(f"  {adapter.audit_row()}")
        layers = layer_spec_for(cfg, model)

        if args.verify_n > 0:
            w0, e0 = story_targets[stories[0]]
            H_check = max(H_list)
            md = verify_batch_matches_single_perturbed(
                adapter, w0, perms[stories[0]], e0, H_check, layers, args.verify_n)
            tol = 2e-3
            status = "OK" if md < tol else "失败"
            print(f"  [verify:{CONDITION}] 批量vs逐窗 (H={H_check}, n={args.verify_n}): "
                  f"max|Δ|={md:.2e} ({status})")
            if md >= tol:
                raise SystemExit(
                    f"[{CONDITION}] 批量提取与逐窗结果不一致 (max|Δ|={md:.2e} >= {tol})，"
                    f"疑似 padding/mask bug，已中止。")

        model_stats = {}
        t_model = time.perf_counter()
        for i, story in enumerate(stories, 1):
            if args.skip_existing and _all_cached(cache_dir_ctx1_s, model, story, H_list):
                print(f"  [{i}/{len(stories)}] {story} 已缓存(ctx1)，跳过")
                continue
            words, eligible = story_targets[story]
            base = word_index[word_index["story"] == story]
            perm = perms[story]
            print(f"  [{i}/{len(stories)}] {story}: {len(words)} 词, "
                  f"{len(eligible)} 有效目标 [{CONDITION}]")
            stats, token_rows = extract_story_model_ctx1(
                adapter, model, story, words, perm, eligible, base,
                H_list, layers, cfg, args.max_targets, args.batch_size,
                cache_dir_ctx1_s,
                next(r["permutation_sha256"] for r in perm_manifest_rows if r["story"] == story),
            )
            tm = make_token_map(token_rows)
            validate_token_map(tm, word_index)  # 目标词/index 未变的再一道核验
            tm_path = Path(cache_dir_ctx1_s) / model / f"{story}_token_map.parquet"
            save_token_map(tm, tm_path)
            model_stats[story] = stats
        print(f"  {model} [{CONDITION}] 完成 {len(stories)} 故事，"
              f"{(time.perf_counter()-t_model)/60:.1f} 分钟")
        report["models"][model] = {"audit": adapter.audit_row(),
                                   "by_story": model_stats}

    report_path = SUPPLEMENT_ROOT / "results" / "step3_extraction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[ctx1] 提取报告已写: {report_path}")
    print("Step 3（ctx1 特征重提取）完成。下一步：Step 4 用 cache_dir_ctx1 走 "
         "assemble_all → run_fold（normal 分支）重新拟合。")


if __name__ == "__main__":
    main()
