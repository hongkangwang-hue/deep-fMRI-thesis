"""
M0 — 地基与冻结文件生成

生成 frozen/ 下所有冻结文件：
  story_manifest.csv       — 故事清单
  word_index.parquet       — 词级索引
  fold_split.json          — 三折 CV 划分
  roi_spec.json            — ROI 规则
  analysis_spec.yaml       — 分析参数冻结
  contrast_registry.yaml   — 对比分层

用法：
  python scripts/m0_build_frozen.py
"""

import json
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "encoding"))

from src.config_loader import load_config
from ridge_utils.stimulus_utils import load_textgrids

TR_DURATION = 2.0  # 秒，LeBel 数据集固定 TR


# ── 1. Story manifest ──────────────────────────────────────────────────────────

def build_story_manifest(cfg: dict) -> pd.DataFrame:
    """从 respdict.json 和 UTS03 BOLD 目录构建故事清单。"""
    with open(cfg["datasets"]["respdict"]) as f:
        respdict = json.load(f)

    held_out = cfg["datasets"]["held_out_story"]
    subject  = cfg["datasets"]["subjects"][0]
    bold_dir = Path(cfg["datasets"]["data_dir"]) / subject

    rows = []
    for story, n_trs in sorted(respdict.items()):
        bold_path = bold_dir / f"{story}.hf5"
        rows.append({
            "story":        story,
            "n_trs":        n_trs,
            "duration_s":   n_trs * TR_DURATION,
            "duration_min": round(n_trs * TR_DURATION / 60, 2),
            "is_held_out":  story == held_out,
            "bold_available": bold_path.exists(),
        })

    return pd.DataFrame(rows).sort_values("story").reset_index(drop=True)


# ── 2. Word index ──────────────────────────────────────────────────────────────

def _parse_textgrid_words(grid) -> list[tuple]:
    """从项目自带的 ridge_utils TextGrid 对象的 word tier 返回 [(word, onset_s, offset_s), ...]。"""
    word_tier = next(t for t in grid.tiers if t.nameid == "word")
    words = []
    for start_s, end_s, label in word_tier.simple_transcript:
        w = label.strip()
        # 跳过静音标记和空白
        if w and w not in ("sp", "{B_TRANS}", "{E_TRANS}"):
            words.append((w, float(start_s), float(end_s)))
    return words


def build_word_index(cfg: dict, stories: list[str]) -> pd.DataFrame:
    """从 TextGrids 构建词级索引，记录 TR 映射和有效性标记。"""
    data_dir = str(Path(cfg["datasets"]["textgrid_dir"]).parent.parent.parent)
    grids = load_textgrids(sorted(stories), data_dir)

    rows = []
    global_word_id = 0

    for story in sorted(stories):
        grid = grids.get(story)
        if grid is None:
            print(f"  WARNING: 找不到 TextGrid — {story}")
            continue

        story_words = _parse_textgrid_words(grid)

        for local_id, (word, onset_s, offset_s) in enumerate(story_words):
            rows.append({
                "word_id":        global_word_id,
                "story":          story,
                "word_local_id":  local_id,
                "word":           word,
                "onset_s":        round(onset_s, 4),
                "offset_s":       round(offset_s, 4),
                "tr_index":       int(onset_s / TR_DURATION),
                # H=128：该词有至少 128 个前驱词（0-based 位置 >= 128）
                "eligible_h128":  local_id >= 128,
                # 评分掩码：跳过故事前 100 秒（transient BOLD 响应）
                "score_after_100s": onset_s > 100.0,
            })
            global_word_id += 1

    return pd.DataFrame(rows)


# ── 3. Fold split ──────────────────────────────────────────────────────────────

def build_fold_split(cfg: dict, manifest: pd.DataFrame) -> dict:
    """
    将训练故事均衡划分为 N 折。

    策略：按 n_trs 降序排列后轮转分配，保证每折总 TR 数接近。
    wheretheressmoke 作为固定 held-out 测试集，不进入任何折。
    """
    n_folds  = cfg["datasets"]["outer_folds"]
    held_out = cfg["datasets"]["held_out_story"]

    train_df = (
        manifest[manifest["bold_available"] & ~manifest["is_held_out"]]
        .sort_values("n_trs", ascending=False)
        .reset_index(drop=True)
    )

    # 轮转分配
    train_df["fold"] = train_df.index % n_folds

    folds = {}
    for fold_idx in range(n_folds):
        test_mask  = train_df["fold"] == fold_idx
        test_stories = train_df[test_mask]["story"].tolist()
        train_stories = train_df[~test_mask]["story"].tolist()
        folds[f"fold_{fold_idx}"] = {
            "test_stories":  test_stories,
            "train_stories": train_stories,
            "n_test_stories": len(test_stories),
            "n_train_stories": len(train_stories),
            "test_n_trs":  int(train_df[test_mask]["n_trs"].sum()),
            "train_n_trs": int(train_df[~test_mask]["n_trs"].sum()),
        }

    return {
        "held_out_test_story": held_out,
        "n_folds":             n_folds,
        "split_unit":          cfg["datasets"]["split_unit"],
        "balance_metric":      "n_trs_round_robin_descending",
        "seed":                cfg["seeds"]["fold_split"],
        "folds":               folds,
    }


# ── 4. ROI spec ────────────────────────────────────────────────────────────────

def build_roi_spec(cfg: dict) -> dict:
    return {
        "atlas": "aparc.a2009s",
        "freesurfer_subjdir": "data/ds003020/derivatives/freesurfer_subjdir",
        "rois": cfg["brain_analysis"]["rois"],
        "bold_only_voxel_rule": {
            "exclude_nan": True,
            "exclude_zero_variance": True,
            "min_voxels_per_roi": 10,
        },
        "repeatability": {
            "method": "split_half_correlation",
            "repeated_story": cfg["datasets"]["held_out_story"],
            "note": "wheretheressmoke 在 5 个 session 中重复收听，用于计算 repeatability",
        },
        "extraction_note": (
            "Atlas annotation 文件需在 AutoDL 上下载后才能执行实际 ROI 提取（M2 阶段）"
        ),
    }


# ── 5. Analysis spec ───────────────────────────────────────────────────────────

def build_analysis_spec(cfg: dict) -> dict:
    return {
        "context_lengths_H": cfg["models"]["contexts_preceding_words"],
        "first_eligible_target_index_0based": cfg["models"]["first_eligible_target_index_zero_based"],
        "window_definition": "H preceding words + target word = H+1 tokens total",
        "state_reset_per_window": cfg["models"]["state_reset_per_window"],
        "transformer_reuse_external_kv_cache": cfg["models"]["transformer_reuse_external_kv_cache"],
        "primary_layers":   cfg["models"]["primary_layers"],
        "robustness_layers": cfg["models"]["robustness_layers"],
        "pca": {
            "n_components":   cfg["pca"]["k"],
            "fit_scope":      cfg["pca"]["fit_scope"],
            "apply_before_fir": cfg["pca"]["apply_before_fir"],
            "svd_solver":     cfg["pca"]["svd_solver"],
        },
        "ridge": {
            "solver":           cfg["ridge"]["solver"],
            "alpha_scope":      cfg["ridge"]["alpha_scope"],
            "lambda_grid":      cfg["ridge"]["lambda_grid"],
            "inner_folds":      cfg["ridge"]["inner_folds"],
            "selection_metric": cfg["ridge"]["selection_metric"],
            "tie_break_rule":   cfg["ridge"]["tie_break_rule"],
        },
        "scoring": cfg["scoring"],
        "heldout_score_after_story_seconds": cfg["brain_analysis"]["heldout_score_after_story_seconds"],
        "fir_delays_seconds": cfg["brain_analysis"]["fir_delays_seconds"],
        "negative_control": cfg["negative_control"],
        "seeds": cfg["seeds"],
    }


# ── 6. Contrast registry ───────────────────────────────────────────────────────

def build_contrast_registry() -> dict:
    return {
        "confirmatory_primary": {
            "family_name": "ifg_main_layer_delta_total_architecture",
            "fwer_control": "holm_bootstrap_pvalues_alpha_0.05",
            "note": "唯一允许下确认性结论的家族",
            "contrasts": [
                "rwkv_minus_pythia_delta_total_ifg_main",
                "mamba_minus_pythia_delta_total_ifg_main",
            ],
        },
        "main_estimands_descriptive": [
            "delta_total_pythia_ifg_main",
            "delta_total_rwkv_ifg_main",
            "delta_total_mamba_ifg_main",
        ],
        "exploratory_rq1_same_context_architecture": [
            "rwkv_minus_pythia_r8",   "rwkv_minus_pythia_r32",   "rwkv_minus_pythia_r128",
            "mamba_minus_pythia_r8",  "mamba_minus_pythia_r32",  "mamba_minus_pythia_r128",
        ],
        "secondary_exploratory": [
            "delta_local_by_model",
            "delta_long_by_model",
            "ifg_vs_pt_descriptive_pattern",
            "awd_lstm_within_model_context_curve",
        ],
        "robustness": [
            "registered_contrasts_on_final_layer_ifg",
            "shifted_r_by_h",
            "shifted_delta_local",
            "shifted_delta_long",
            "shifted_delta_total",
        ],
    }


# ── 验收检查 ───────────────────────────────────────────────────────────────────

def validate(manifest: pd.DataFrame, word_index: pd.DataFrame, fold_split: dict) -> None:
    """基本完整性检查，任何失败都会抛出 AssertionError。"""
    held_out = fold_split["held_out_test_story"]

    # wheretheressmoke 不在任何折里
    for fold_name, fold in fold_split["folds"].items():
        assert held_out not in fold["test_stories"],  f"{held_out} 出现在 {fold_name} 的 test 里"
        assert held_out not in fold["train_stories"], f"{held_out} 出现在 {fold_name} 的 train 里"

    # 每个折的 train + test = 所有训练故事
    n_train_total = len(manifest[manifest["bold_available"] & ~manifest["is_held_out"]])
    for fold_name, fold in fold_split["folds"].items():
        n = fold["n_test_stories"] + fold["n_train_stories"]
        assert n == n_train_total, f"{fold_name}: 故事数量不对 ({n} != {n_train_total})"

    # H=128 索引规则
    for story, grp in word_index.groupby("story"):
        grp = grp.sort_values("word_local_id")
        assert grp[grp["word_local_id"] == 128]["eligible_h128"].all(), \
            f"{story}: local_id=128 应该是 eligible_h128"
        if len(grp) > 0:
            assert not grp[grp["word_local_id"] == 127]["eligible_h128"].any(), \
                f"{story}: local_id=127 不应该是 eligible_h128"

    print("  验收检查全部通过 ✓")


# ── 主程序 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    frozen_dir = Path(cfg["paths"]["frozen_dir"])
    frozen_dir.mkdir(exist_ok=True)

    # 1. Story manifest
    print("[1/6] 构建 story_manifest ...")
    manifest = build_story_manifest(cfg)
    manifest.to_csv(frozen_dir / "story_manifest.csv", index=False)
    n_avail = manifest["bold_available"].sum()
    print(f"  总故事: {len(manifest)}, BOLD 可用: {n_avail}, held-out: {cfg['datasets']['held_out_story']}")

    # 2. Word index（所有 84 个故事的 TextGrid 都可读）
    print("[2/6] 构建 word_index ...")
    all_stories = manifest["story"].tolist()
    word_index = build_word_index(cfg, all_stories)
    word_index.to_parquet(frozen_dir / "word_index.parquet", index=False)
    eligible = word_index[word_index["eligible_h128"] & word_index["score_after_100s"]]
    print(f"  总词数: {len(word_index):,}, 有效编码目标 (H=128 + >100s): {len(eligible):,}")

    # 3. Fold split
    print("[3/6] 构建 fold_split ...")
    fold_split = build_fold_split(cfg, manifest)
    with open(frozen_dir / "fold_split.json", "w") as f:
        json.dump(fold_split, f, indent=2, ensure_ascii=False)
    for i in range(cfg["datasets"]["outer_folds"]):
        fi = fold_split["folds"][f"fold_{i}"]
        print(f"  Fold {i}: {fi['n_test_stories']} 测试故事 ({fi['test_n_trs']} TRs) | "
              f"{fi['n_train_stories']} 训练故事 ({fi['train_n_trs']} TRs)")

    # 4. ROI spec
    print("[4/6] 写 roi_spec ...")
    roi_spec = build_roi_spec(cfg)
    with open(frozen_dir / "roi_spec.json", "w") as f:
        json.dump(roi_spec, f, indent=2, ensure_ascii=False)

    # 5. Analysis spec
    print("[5/6] 写 analysis_spec ...")
    analysis_spec = build_analysis_spec(cfg)
    with open(frozen_dir / "analysis_spec.yaml", "w") as f:
        yaml.dump(analysis_spec, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # 6. Contrast registry
    print("[6/6] 写 contrast_registry ...")
    registry = build_contrast_registry()
    with open(frozen_dir / "contrast_registry.yaml", "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # 验收
    print("\n[验收] 运行完整性检查 ...")
    validate(manifest, word_index, fold_split)

    print(f"\nM0 完成。冻结文件位于: {frozen_dir.resolve()}")
    print("下一步: git commit + freeze tag，然后进入 M1。")
