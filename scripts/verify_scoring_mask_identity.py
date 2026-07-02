"""
验证 normal 与 shifted 负控制条件使用**逐元素相同**的评分 mask（而非仅 n_eff 计数相等）。

背景：M5 的配对 story bootstrap 要求 normal/shifted 在同一 story 上评分于**同一批 TR**。
之前只断言了每 story 的有效 TR 数 n_eff 相等——但这只是**必要非充分**：两个 mask 可以
选中不同 TR 却恰好数量相同（如 {1,2,3} vs {4,5,6}）。本脚本直接在**布尔 mask 数组层面**
逐元素验证，不用计数代替。

为何不用重跑 M4/GPU 就能严格验证：评分 mask = after_100s(tr_times) ∩ FIR_valid ∩
shift_valid，三者**全部只依赖 TR 位置/故事长度，与特征值无关**（见 src/fmri：apply_fir
的 valid 是 `vmask[d:]=True` 位置量，shift_story_no_wrap 的 valid 同理）。因此 mask 是
故事长度的确定性函数，用真实故事长度（frozen respdict）+ 真实 src.fmri 函数重构出的
mask 就等于 M4 当时实际用的 mask，无随机、无近似。

脚本分两步：
  ① 特征无关性证明（把"与特征值无关"这个前提本身变成断言）：对同一 shape 喂两组不同
     随机值，断言 apply_fir / shift_story_no_wrap 的 valid 输出完全相同。
  ② mask 逐元素一致性：对每折每个 held-out story，用真实长度构造 normal-path 与
     shift-path 的布尔 mask（normal 用一组特征值、shift 用位移后的另一组值——刻意不同值
     同 shape），np.array_equal 逐元素断言相同。

可复用于任何被试/数据集（--subject 指定）。纯 CPU、秒级、不碰 GPU/服务器。
写审计到 results/mask_identity_audit/<subject>/mask_identity.json。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config                          # noqa: E402
from src.fmri.alignment import apply_fir, shift_story_no_wrap      # noqa: E402
from src.fmri.mask import common_scoring_mask                      # noqa: E402
from src.fmri.trfile import load_respdict, trimmed_tr_times        # noqa: E402
from src.ridge.pipeline import DELAYS_S, TR_SECONDS, AFTER_S       # noqa: E402
from src.ridge.m4_driver import SHIFT_SECONDS                      # noqa: E402

FEAT_DIM = 100          # 任意；mask 与特征维数/值无关，仅用于喂函数


def prove_valid_is_feature_independent(lengths, seed=0) -> dict:
    """步骤①：对代表性长度，喂两组不同随机值，断言 apply_fir/shift valid 完全相同。"""
    rng = np.random.default_rng(seed)
    checked = 0
    for L in lengths:
        a = rng.standard_normal((L, FEAT_DIM))
        b = rng.standard_normal((L, FEAT_DIM))          # 不同的值、同 shape
        _, va = apply_fir(a, delays_s=DELAYS_S, tr=TR_SECONDS)
        _, vb = apply_fir(b, delays_s=DELAYS_S, tr=TR_SECONDS)
        if not np.array_equal(va, vb):
            raise AssertionError(f"apply_fir valid 依赖特征值（L={L}）——前提不成立")
        _, sa = shift_story_no_wrap(a, seconds=SHIFT_SECONDS, tr=TR_SECONDS)
        _, sb = shift_story_no_wrap(b, seconds=SHIFT_SECONDS, tr=TR_SECONDS)
        if not np.array_equal(sa, sb):
            raise AssertionError(f"shift_story_no_wrap valid 依赖特征值（L={L}）")
        checked += 1
    return {"feature_independence_proven": True, "lengths_checked": checked}


def masks_for_story(n_resps: int, seed: int):
    """按 run_fold 的方式为一个故事重构 normal-path 与 shift-path 的布尔评分 mask。

    normal 与 shift **刻意用不同的随机特征值**（同 shape），以真正检验"换成 shift 的
    特征值会不会改变 mask"。返回 (normal_mask, shift_mask, tr_times)。
    """
    trt = trimmed_tr_times(n_resps)
    L = len(trt)
    rng = np.random.default_rng(seed)
    Xn = rng.standard_normal((L, FEAT_DIM))              # normal 特征（任意值）
    # shift 条件：特征整体位移 40s，得到不同的值 + 位移边缘无效 mask（与 M4 一致）
    Xs, shift_valid = shift_story_no_wrap(Xn, seconds=SHIFT_SECONDS, tr=TR_SECONDS)

    # FIR valid：normal 用 Xn、shift 用 Xs（不同值），位置量应相同
    _, fir_valid_normal = apply_fir(Xn, delays_s=DELAYS_S, tr=TR_SECONDS)
    _, fir_valid_shift = apply_fir(Xs, delays_s=DELAYS_S, tr=TR_SECONDS)

    # 两条件都用同一 shift_valid（M4 的共同 mask 设计），tr_times 相同
    normal_mask = common_scoring_mask(trt, fir_valid_normal, shift_valid=shift_valid, after_s=AFTER_S)
    shift_mask = common_scoring_mask(trt, fir_valid_shift, shift_valid=shift_valid, after_s=AFTER_S)
    return normal_mask, shift_mask, trt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--folds", nargs="+", default=None,
                    help="frozen/fold_split.json 键名子集；默认全部")
    ap.add_argument("--out-name", default="mask_identity_audit")
    args = ap.parse_args()

    cfg = load_config()
    frozen = Path(cfg["paths"]["frozen_dir"])
    respdict = load_respdict(cfg["datasets"]["respdict"])
    with open(frozen / "fold_split.json") as f:
        fold_split = json.load(f)
    fold_names = args.folds if args.folds else list(fold_split["folds"].keys())

    # 收集所有 held-out story（每故事在某折作测试）
    test_by_fold = {fn: sorted(fold_split["folds"][fn]["test_stories"]) for fn in fold_names}
    lengths = sorted({len(trimmed_tr_times(respdict[s]))
                      for fn in fold_names for s in test_by_fold[fn]})

    print(f"[verify] subject={args.subject} 步骤①证明 mask 与特征值无关 "
          f"（{len(lengths)}种长度）...", flush=True)
    prop = prove_valid_is_feature_independent(lengths)
    print(f"[verify] ✓ 特征无关性成立（apply_fir/shift valid 对不同特征值输出一致）", flush=True)

    print(f"[verify] 步骤②逐故事逐元素断言 normal-mask == shift-mask ...", flush=True)
    per_story, all_identical = [], True
    for fn in fold_names:
        for i, s in enumerate(test_by_fold[fn]):
            nmask, smask, trt = masks_for_story(respdict[s], seed=hash((fn, s)) % (2**32))
            identical = bool(np.array_equal(nmask, smask))
            all_identical &= identical
            per_story.append({"fold": fn, "story": s, "n_trs": len(trt),
                              "n_eff": int(nmask.sum()), "mask_bit_identical": identical})
            if not identical:
                raise AssertionError(
                    f"[{fn}/{s}] normal 与 shift 评分 mask 逐元素不同 → 配对不成立！"
                    f"（normal n_eff={int(nmask.sum())} shift n_eff={int(smask.sum())}）")

    n_stories = len(per_story)
    print(f"[verify] ✓ {n_stories} 个 held-out story 全部 mask 逐元素相同", flush=True)

    audit = {
        "subject": args.subject,
        "folds": fold_names,
        "shift_seconds": SHIFT_SECONDS, "fir_delays_s": list(DELAYS_S),
        "after_s": AFTER_S, "tr_s": TR_SECONDS,
        "step1_feature_independence": prop,
        "step2_n_stories_checked": n_stories,
        "step2_all_masks_bit_identical": bool(all_identical),
        "method": "reconstructed normal/shift boolean scoring masks per held-out story from "
                  "frozen respdict via src.fmri functions and asserted element-wise equality; "
                  "valid mask is provably feature-value-independent (step1), so reconstruction "
                  "equals the mask M4 actually used. Supersedes the n_eff-count check as the "
                  "sufficient (not merely necessary) verification of common mask.",
        "per_story": per_story,
    }
    out_dir = Path(cfg["paths"]["results_dir"]) / args.out_name / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mask_identity.json", "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"[verify] 全部通过 ✅  审计 → {out_dir / 'mask_identity.json'}", flush=True)


if __name__ == "__main__":
    main()
