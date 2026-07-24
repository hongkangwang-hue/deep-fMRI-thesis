"""补充实验 Step 2——硬性实现断言 + §2.4 词形重复诊断（零算力，对真实数据跑）。

对应 V6.4.2 §2.1（硬闸门 A-F）+ §2.4（origin inclusion / form repetition，
M4S-Core 必做）。只依赖冻结的 word_index + 置换索引，不涉及任何模型推理，
CPU 上跑全部 84 个真实故事只需数秒。

用法：python3 实验补充/scripts/m4s_assert_impl.py
输出：实验补充/results/step2_diagnostics.json

§2.1 硬闸门覆盖情况（本脚本做哪些、哪些留到有真实模型/被试数据后再做）：
    A 单故事单置换       —— 本脚本验证（确定性：同 story_id+seed 两次生成一致）
    B 相邻滑窗 H-1 重叠   —— 本脚本对全部真实故事、H∈{8,32,128} 全量验证
    C 跨模型一致         —— 结构性保证：story_permutation 的签名里没有 model
                             参数，置换与模型无关，不需要额外验证
    D 跨被试一致         —— 结构性保证：同理，签名里没有 subject 参数
    E 目标词/index 未改  —— 本脚本验证：窗口末位==真实词；且目标集合大小与
                             M0 冻结的 eligible_h128 总数（147,846）核对一致
    F mask 未改          —— 结构性保证：C1 不移动时间轴，只改词内容，因此
                             评分 mask（after_100s ∩ FIR_valid）不可能受影响；
                             将在 Step 4 用真实 per-story TR 数再次实测确认
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SUPPLEMENT_ROOT = SCRIPT_DIR.parent          # 实验补充/
PROJECT_ROOT = SUPPLEMENT_ROOT.parent         # 仓库根目录
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SUPPLEMENT_ROOT / "src"))

from src.config_loader import load_config              # noqa: E402
from src.models.windowing import iter_story_targets     # noqa: E402
from context_perturb import (                            # noqa: E402
    story_permutation, build_perturbed_window, context_origins,
)

MASTER_SEED = 20260724   # 见 ../config/freeze_manifest.json，与 V6.4.2 §0.7 一致
H_LIST = [8, 32, 128]    # 诊断表覆盖全部三档 H（零成本）；Step 3/4 实际建模只用 {8,128}（L1 Core scope）


def run_hard_assertions(word_index: pd.DataFrame) -> dict:
    """§2.1 A/B/E 的本地可测部分，对全部真实故事跑。"""
    report = {"A_single_permutation_per_story": {}, "B_adjacent_overlap": {},
             "E_target_word_and_set_unchanged": {}}

    total_eligible = 0
    b_checked_pairs = 0
    b_failures = []

    for story, words, eligible in iter_story_targets(word_index, binding_H=128):
        n = len(words)
        total_eligible += len(eligible)

        # A: 同一 (story_id, seed) 两次生成必须逐位相同
        perm1 = story_permutation(story, MASTER_SEED, n)
        perm2 = story_permutation(story, MASTER_SEED, n)
        a_ok = np.array_equal(perm1, perm2)
        report["A_single_permutation_per_story"][story] = bool(a_ok)
        if not a_ok:
            raise AssertionError(f"[A] 故事 {story} 两次生成的置换不一致，实现有误")

        perm = perm1
        eligible_sorted = sorted(eligible)

        # E: 目标词与末位在任意 H 下都不能变
        for H in H_LIST:
            for i in eligible_sorted[:3] + eligible_sorted[-3:]:  # 每故事抽样首尾各3个，足够且省时间
                if i < H:
                    continue
                window = build_perturbed_window(words, perm, i, H)
                if window[-1] != words[i]:
                    raise AssertionError(f"[E] 故事{story} i={i} H={H}: 目标词被改动")

        # B: 相邻合法目标位置的 H-1 重叠（eligible 是连续整数区间，逐对检查）
        for H in H_LIST:
            valid_idx = [i for i in eligible_sorted if i >= H]
            for a, b in zip(valid_idx, valid_idx[1:]):
                if b != a + 1:
                    continue  # 非相邻（不应发生，eligible 本身连续，但保险起见跳过）
                origins_a = context_origins(perm, a, H)
                origins_b = context_origins(perm, b, H)
                b_checked_pairs += 1
                if not np.array_equal(origins_b[:-1], origins_a[1:]):
                    b_failures.append((story, H, a, b))

    report["B_adjacent_overlap"] = {
        "n_pairs_checked": b_checked_pairs,
        "n_failures": len(b_failures),
        "failures_sample": b_failures[:5],
        "pass": len(b_failures) == 0,
    }
    if b_failures:
        raise AssertionError(f"[B] {len(b_failures)} 对相邻窗口未满足 H-1 重叠，实现有误")

    report["E_target_word_and_set_unchanged"] = {
        "total_eligible_h128_targets": total_eligible,
        "expected_from_m0_freeze": 147846,
        "match": total_eligible == 147846,
    }
    if total_eligible != 147846:
        raise AssertionError(
            f"[E] eligible_h128 目标总数 {total_eligible} != M0 冻结值 147846，"
            "word_index 或 iter_story_targets 行为已变化，需排查"
        )

    report["C_cross_model_consistency"] = {
        "status": "结构性保证（非空跑验证）",
        "reason": "story_permutation(story_id, master_seed, n_words) 签名不含 model 参数，"
                  "置换生成与模型无关，任何模型用同一 (story_id, seed) 必然拿到同一置换",
    }
    report["D_cross_subject_consistency"] = {
        "status": "结构性保证（非空跑验证）",
        "reason": "同理，签名不含 subject 参数；刺激侧特征本就与被试无关"
                  "（被试差异只在 BOLD/ROI/拟合阶段引入）",
    }
    report["F_mask_unchanged"] = {
        "status": "结构性保证，Step 4 用真实数据复核",
        "reason": "C1 只改变喂给模型的词内容，不改变任何时间戳/TR 对齐，"
                  "故评分 mask（after_100s ∩ FIR_valid）不可能受影响；"
                  "待 Step 4 产出真实 per-story 评分 mask 后按 "
                  "np.array_equal(mask_ctx1, mask_normal) 逐元素复核",
    }
    return report


def compute_origin_inclusion_and_form_repetition(word_index: pd.DataFrame) -> pd.DataFrame:
    """§2.4：origin_inclusion_rate(H) 与 form_repetition_rate(H | story-frequency tier)。

    预期量级（V6.4.2 §2.4，结果产出前写死，用于判断实现是否正常）：
        E[origin_inclusion_rate(H)] ≈ H/N
        E[form_repetition_rate(H) | freq=f] ≈ 1 - (1-H/N)^f
    """
    rows = []
    for story, words, eligible in iter_story_targets(word_index, binding_H=128):
        n = len(words)
        perm = story_permutation(story, MASTER_SEED, n)
        eligible_sorted = sorted(eligible)

        # 该故事内每个词形的出现频次（story-frequency），用于分层
        freq = defaultdict(int)
        for w in words:
            freq[w] += 1

        for H in H_LIST:
            valid_idx = [i for i in eligible_sorted if i >= H]
            if not valid_idx:
                continue
            origin_inclusion_hits = 0
            form_rep_hits = 0
            n_targets = len(valid_idx)
            for i in valid_idx:
                origins = context_origins(perm, i, H)
                origin_inclusion_hits += int(i in origins)
                ctx_words = {words[j] for j in origins}
                form_rep_hits += int(words[i] in ctx_words)

            origin_rate = origin_inclusion_hits / n_targets
            form_rate = form_rep_hits / n_targets
            expected_origin = H / n
            rows.append({
                "story": story, "H": H, "n_words": n, "n_targets": n_targets,
                "origin_inclusion_rate": origin_rate,
                "origin_inclusion_expected": expected_origin,
                "form_repetition_rate": form_rate,
            })
    return pd.DataFrame(rows)


def summarize_by_frequency_tier(word_index: pd.DataFrame) -> pd.DataFrame:
    """form_repetition_rate 按目标词的故事内频次分层（hapax / 2-5 / >5），
    对照 V6.4.2 §2.4 预期表：E[rate|f] ≈ 1-(1-H/N)^f。"""
    rows = []
    for story, words, eligible in iter_story_targets(word_index, binding_H=128):
        n = len(words)
        perm = story_permutation(story, MASTER_SEED, n)
        eligible_sorted = sorted(eligible)
        freq = defaultdict(int)
        for w in words:
            freq[w] += 1

        for H in H_LIST:
            valid_idx = [i for i in eligible_sorted if i >= H]
            tier_hits = defaultdict(lambda: [0, 0])  # tier -> [hits, n]
            for i in valid_idx:
                f = freq[words[i]]
                tier = "hapax(f=1)" if f == 1 else ("f=2-5" if f <= 5 else "f>5")
                origins = context_origins(perm, i, H)
                ctx_words = {words[j] for j in origins}
                hit = int(words[i] in ctx_words)
                tier_hits[tier][0] += hit
                tier_hits[tier][1] += 1
            for tier, (hits, n_t) in tier_hits.items():
                if n_t == 0:
                    continue
                rows.append({
                    "story": story, "H": H, "freq_tier": tier,
                    "form_repetition_rate": hits / n_t, "n_targets_in_tier": n_t,
                })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    word_index_path = Path(cfg["paths"]["frozen_dir"]) / "word_index.parquet"
    word_index = pd.read_parquet(word_index_path)
    print(f"已读取 {word_index_path}，{len(word_index)} 行（{word_index['story'].nunique()} 个故事）")

    print("\n[1/3] 硬性实现断言（§2.1 A/B/E，C/D/F 为结构性保证）...")
    assert_report = run_hard_assertions(word_index)
    print(f"  A: {sum(assert_report['A_single_permutation_per_story'].values())}/"
         f"{len(assert_report['A_single_permutation_per_story'])} 故事通过确定性检查")
    print(f"  B: {assert_report['B_adjacent_overlap']['n_pairs_checked']} 对相邻窗口，"
         f"{assert_report['B_adjacent_overlap']['n_failures']} 处失败")
    print(f"  E: 目标总数 {assert_report['E_target_word_and_set_unchanged']['total_eligible_h128_targets']} "
         f"(期望 147846, match={assert_report['E_target_word_and_set_unchanged']['match']})")
    print("  全部硬闸门通过。")

    print("\n[2/3] §2.4 origin inclusion / form repetition（全量真实故事）...")
    diag_df = compute_origin_inclusion_and_form_repetition(word_index)
    by_H = diag_df.groupby("H").agg(
        mean_origin_inclusion=("origin_inclusion_rate", "mean"),
        mean_origin_inclusion_expected=("origin_inclusion_expected", "mean"),
        mean_form_repetition=("form_repetition_rate", "mean"),
    ).reset_index()
    print(by_H.to_string(index=False))

    print("\n[3/3] 按目标词故事内频次分层的 form_repetition_rate...")
    tier_df = summarize_by_frequency_tier(word_index)
    tier_summary = tier_df.groupby(["H", "freq_tier"]).agg(
        mean_form_repetition_rate=("form_repetition_rate", "mean"),
        n_stories=("story", "nunique"),
        n_targets_total=("n_targets_in_tier", "sum"),
    ).reset_index()
    print(tier_summary.to_string(index=False))

    out_dir = SUPPLEMENT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    diag_df.to_csv(out_dir / "step2_origin_form_by_story.csv", index=False)
    tier_summary.to_csv(out_dir / "step2_form_repetition_by_freq_tier.csv", index=False)

    result = {
        "master_seed": MASTER_SEED,
        "hard_assertions": assert_report,
        "origin_form_summary_by_H": by_H.to_dict(orient="records"),
        "form_repetition_by_freq_tier": tier_summary.to_dict(orient="records"),
        "expected_table_v642": {
            "H=8":   {"origin_inclusion": "≈0.4%", "form_rep_f1": "≈0.4%", "form_rep_f5": "≈2.1%", "form_rep_f50": "≈19%"},
            "H=32":  {"origin_inclusion": "≈1.7%", "form_rep_f1": "≈1.7%", "form_rep_f5": "≈8.2%", "form_rep_f50": "≈58%"},
            "H=128": {"origin_inclusion": "≈6.8%", "form_rep_f1": "≈6.8%", "form_rep_f5": "≈30%",  "form_rep_f50": "≈97%"},
        },
    }
    out_path = out_dir / "step2_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n已写 {out_path}")
    print(f"已写 {out_dir / 'step2_origin_form_by_story.csv'}（逐故事×H明细）")
    print(f"已写 {out_dir / 'step2_form_repetition_by_freq_tier.csv'}（按频次分层）")


if __name__ == "__main__":
    main()
