"""
硕士论文核查清单 —— 本地可算部分（第1/2/4节）。

只用 frozen/*、config/*、em_data/*、data/ds003020/derivatives/respdict.json 这些本机
已有的冻结/元数据文件，配合 src/fmri/{trfile,alignment,mask}.py 的纯函数重算每故事
有效评分 TR 数——不需要服务器上的 BOLD 数据或任何模型/Ridge 结果，因为 trim/FIR/>100s
规则只依赖 respdict 里的原始 TR 数，与被试、体素、模型完全无关（这也是 M4 三名被试的
"主层正常有效TR/折"在 figures/*/tables/qc_table.csv 里数值完全相同的原因）。

输出对应 milestone/硕士论文代码核查与结果补充清单_精简终版.md：
  第1节 → thesis_supplement/story_manifest.csv
  第2节 → thesis_supplement/target_audit.csv
  第4节 → thesis_supplement/pipeline_config.txt

用法：python3 scripts/thesis_supplement_local.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config          # noqa: E402
from src.fmri.trfile import (                       # noqa: E402
    load_respdict, trimmed_tr_times, expected_response_rows,
    TR_SECONDS, SOUND_START_SECONDS, SIMULATE_PAD, TRIM_FIRST, TRIM_LAST,
)
from src.fmri.alignment import apply_fir             # noqa: E402
from src.fmri.mask import after_time_mask, HELDOUT_SCORE_AFTER_SECONDS  # noqa: E402

FIR_DELAYS = (2, 4, 6, 8)
OUT_DIR = PROJECT_ROOT / "thesis_supplement"


def n_eff_tr_normal(n_resps: int) -> int:
    """单故事、正常条件下的有效评分 TR 数（>100s ∩ FIR 边缘有效），与模型/被试无关。"""
    t = trimmed_tr_times(n_resps)
    n = len(t)
    dummy = np.zeros((n, 1))
    _, fir_valid = apply_fir(dummy, FIR_DELAYS, TR_SECONDS)
    mask = after_time_mask(t) & fir_valid
    return int(mask.sum())


def build_story_manifest(cfg: dict) -> pd.DataFrame:
    frozen = Path(cfg["paths"]["frozen_dir"])
    fold_split = json.load(open(frozen / "fold_split.json"))
    respdict = load_respdict(cfg["datasets"]["respdict"])
    word_index = pd.read_parquet(frozen / "word_index.parquet")

    story_to_fold = {}
    for fold_id, spec in fold_split["folds"].items():
        for s in spec["test_stories"]:
            story_to_fold[s] = fold_id
    held_out = fold_split["held_out_test_story"]

    n_words = word_index.groupby("story").size().to_dict()

    rows = []
    for story, n_resps in sorted(respdict.items()):
        is_repeated = (story == held_out)
        fold_id = story_to_fold.get(story)
        included = story in story_to_fold
        if is_repeated:
            exclusion_reason = "held_out_repeat_probe(wheretheressmoke，跨5个session重复播放，用于repeatability，非CV)"
        elif not included:
            exclusion_reason = "not_in_fold_split(未出现在frozen/fold_split.json任何fold的test_stories中)"
        else:
            exclusion_reason = ""
        rows.append({
            "story_id": story,
            "fold_id": fold_id if fold_id else ("held_out" if is_repeated else ""),
            "duration_sec": round(n_resps * TR_SECONDS, 1),
            "n_words": int(n_words.get(story, 0)),
            "n_tr_raw": int(n_resps),
            "n_tr_expected_after_trim": expected_response_rows(n_resps),
            "n_tr_scored_normal": n_eff_tr_normal(n_resps),
            "is_repeated": is_repeated,
            "included_in_cv": included,
            "exclusion_reason": exclusion_reason,
        })
    df = pd.DataFrame(rows)
    assert len(df) == len(respdict), "respdict 故事数与输出行数不一致"
    return df


def build_target_audit(cfg: dict) -> pd.DataFrame:
    frozen = Path(cfg["paths"]["frozen_dir"])
    fold_split = json.load(open(frozen / "fold_split.json"))
    word_index = pd.read_parquet(frozen / "word_index.parquet")

    story_to_fold = {}
    for fold_id, spec in fold_split["folds"].items():
        for s in spec["test_stories"]:
            story_to_fold[s] = fold_id

    rows = []
    for story, g in word_index.groupby("story"):
        elig = g[g["eligible_h128"]]
        midpoint = (elig["onset_s"] + elig["offset_s"]) / 2.0
        n_missing = int(g["onset_s"].isna().sum() + g["offset_s"].isna().sum())
        rows.append({
            "story_id": story,
            "fold_id": story_to_fold.get(story, "held_out"),
            "n_words_raw": len(g),
            "n_eligible_h128": int(elig.shape[0]),
            "first_eligible_midpoint_sec": (float(midpoint.min())
                                            if len(midpoint) else None),
            "n_missing_time_labels": n_missing,
            "n_invalid_features": None,   # 需要语言模型特征缓存，本机没有，见下方说明
        })
    df = pd.DataFrame(rows).sort_values("story_id").reset_index(drop=True)
    return df


def print_target_summary(word_index: pd.DataFrame, target_df: pd.DataFrame):
    n_raw = len(word_index)
    n_elig = int(word_index["eligible_h128"].sum())
    print(f"[target_audit] 全部故事原始词数 = {n_raw}")
    print(f"[target_audit] eligible_h128 总数 = {n_elig}")
    print("[target_audit] 每折 eligible_h128 数量：")
    print(target_df.groupby("fold_id")["n_eligible_h128"].sum().to_string())
    fm = target_df["first_eligible_midpoint_sec"].dropna()
    print(f"[target_audit] 首个合格目标词 midpoint：max={fm.max():.2f}s, "
          f"median={fm.median():.2f}s")
    print(f"[target_audit] 缺失时间标注词数合计 = {target_df['n_missing_time_labels'].sum()}")


def build_pipeline_config_txt(cfg: dict) -> str:
    frozen = Path(cfg["paths"]["frozen_dir"])
    spec_text = (frozen / "analysis_spec.yaml").read_text()
    lines = []
    lines.append("=" * 78)
    lines.append("第4节：实际执行的预处理流程（如实核对代码，非计划书推测）")
    lines.append("来源：src/fmri/{trfile,alignment,mask}.py 的实现 + frozen/analysis_spec.yaml"
                 " + config/config.yaml（这两个 yaml 是代码实际读取、跑 M1-M6 时真实生效的配置，"
                 "不是另一份“计划”文档）")
    lines.append("=" * 78)
    lines.append("")
    checks = [
        ("是否排除故事开始前 100 秒",
         f"是。src/fmri/mask.py::after_time_mask，阈值 HELDOUT_SCORE_AFTER_SECONDS="
         f"{HELDOUT_SCORE_AFTER_SECONDS}s，按 trim 后 TR 中心时间判定，只保留 >100s 的 TR。"),
        ("是否排除故事结尾 20 秒",
         "否——代码里没有单独的“结尾20秒”规则。结尾影响来自 src/fmri/trfile.py 的 "
         f"TRIM_LAST={TRIM_LAST}（{TRIM_LAST}个TR×{TR_SECONDS}s={TRIM_LAST*TR_SECONDS}秒，"
         "对齐 LeBel 参考实现的 encoding_utils trim=5 规则），不是独立的20秒规则；"
         "如果论文草稿写了“排除结尾20秒”，需要改成如实描述这个 trim 规则，不能照抄未实现的计划书条款。"),
        ("是否使用 Savitzky–Golay 滤波",
         "否——src/fmri/、src/ridge/、src/models/ 全代码库检索不到 savgol/Savitzky 相关调用，"
         "未实现，未使用。"),
        ("是否对 BOLD 做标准化",
         "是，但不在本项目代码里做——训练故事的 .hf5 在数据集 derivatives 阶段已按体素 "
         "z-score(std=1)（[[m2-pipeline-state]] 记录的核对结论），loader 不重复 z-score；"
         "held-out 故事 wheretheressmoke 是跨重复平均，std≈0.36，不是单位方差。"),
        ("标准化是否只基于训练故事",
         "BOLD 标准化不适用（上条：标准化发生在数据集 derivatives 阶段，非本项目代码步骤）。"
         "本项目代码里真正“只用训练折拟合”的是 StandardScaler(语言特征侧) 与 PCA——"
         "两者 fit_scope 均为 outer_training_stories_only（frozen/analysis_spec.yaml），"
         "程序化保证：src/ridge/pipeline.py 的 run_fold 每折用该折的 training stories 单独 "
         "fit scaler/PCA/ridge，测试故事只 transform。"),
        ("词时间是否使用 midpoint",
         "是。target_audit 的 first_eligible_midpoint_sec 及 Lanczos 重采样都用 "
         "(onset_s+offset_s)/2 作为词的代表时刻（src/fmri/alignment.py::word_to_tr 的 "
         "data_times 参数）。"),
        ("Lanczos 重采样参数",
         "src/fmri/alignment.py::word_to_tr，window=3（Lanczos 窗叶数，等同参考实现 "
         "lanczosinterp2D 默认值），复用 encoding/ridge_utils/interpdata.py 的算子（与 "
         "LeBel 参考实现数值一致，M2-C Phase 1 用此路径核对通过：voxel_r=0.9962）。"),
        ("FIR 是否为 2、4、6、8 秒",
         f"是。src/fmri/alignment.py::apply_fir 默认 delays_s={FIR_DELAYS}，TR="
         f"{TR_SECONDS}s，对应位移 1/2/3/4 个 TR；前 {max(round(d/TR_SECONDS) for d in FIR_DELAYS)} "
         "个 TR 因最大延迟缺乏历史支撑被标记为边缘无效（FIR valid mask）。"),
        ("PCA 是否在 FIR 之前",
         "是。frozen/analysis_spec.yaml::pca.apply_before_fir=true；实现顺序也确认是 "
         "特征下采样到 TR → PCA-100 → FIR 展开（诊断脚本 diag_awd_lstm_context_decay.py "
         "的 [4] 阶段专门核实过这个顺序，因为直觉顺序容易搞反）。"),
        ("PCA 是否只使用 outer-training stories 拟合",
         "是。frozen/analysis_spec.yaml::pca.fit_scope=outer_training_stories_only，"
         "src/ridge/pipeline.py::run_fold 每折只用该折训练故事 fit PCA，测试故事只 "
         "transform，程序化防泄漏（M3a 盲态核查专门验证过不碰 held-out r）。"),
        ("Ridge λ 候选范围",
         "logspace(-2, 7, 19) = [0.01, 1e7]，每 0.5 log10 一个点（19 个候选）。"
         "**注意这不是计划书原始网格**：原网格 logspace(-2,4,13)=[0.01,1e4]"
         "在 M3a 盲态核查（只用 inner-CV，不碰 held-out r）中发现信号体素"
         "（inner-CV score>0.10）100% 撞上界，根因是 PCA-100 输出未标准化、"
         "方差远超 native 参考实现的单位方差特征；已按 inner-validation 结果"
         "扩网格重新冻结（commit 见 git tag m3a-lambda-refreeze），扩网格后"
         "信号体素边界命中率=0.000。"),
        ("λ 是否在训练故事内部选择",
         "是。frozen/analysis_spec.yaml::ridge.inner_folds=2，"
         "selection_metric=validation_pearson_r，在训练折内部再切 2 份做 "
         "inner CV 选 λ，held-out outer test 故事从不参与选择。"),
    ]
    for q, a in checks:
        lines.append(f"- {q}")
        lines.append(f"  {a}")
        lines.append("")
    lines.append("=" * 78)
    lines.append("附：frozen/analysis_spec.yaml 原文（供逐条对照）")
    lines.append("=" * 78)
    lines.append(spec_text)
    return "\n".join(lines)


def main():
    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] 生成 story_manifest.csv ...")
    story_df = build_story_manifest(cfg)
    story_df.to_csv(OUT_DIR / "story_manifest.csv", index=False)
    n_cv = int(story_df["included_in_cv"].sum())
    print(f"  故事总数={len(story_df)}（respdict.json 权威计数），"
          f"included_in_cv={n_cv}，is_repeated={int(story_df['is_repeated'].sum())}")
    for fold_id, g in story_df[story_df.included_in_cv].groupby("fold_id"):
        print(f"  {fold_id}: {len(g)} 故事")

    print("\n[2/3] 生成 target_audit.csv ...")
    frozen = Path(cfg["paths"]["frozen_dir"])
    word_index = pd.read_parquet(frozen / "word_index.parquet")
    target_df = build_target_audit(cfg)
    target_df.to_csv(OUT_DIR / "target_audit.csv", index=False)
    print_target_summary(word_index, target_df)
    print("  n_invalid_features 列留空——需要语言模型特征缓存(cache/features/)逐文件核对"
          "非有限值，本机没有该缓存（只在服务器），已在 README 里如实标注为待服务器补齐项。")

    print("\n[3/3] 生成 pipeline_config.txt ...")
    txt = build_pipeline_config_txt(cfg)
    (OUT_DIR / "pipeline_config.txt").write_text(txt)
    print(f"  已写 {OUT_DIR / 'pipeline_config.txt'}")

    print(f"\n完成。输出目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
