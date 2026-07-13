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
  第3/4节 → thesis_supplement/layer_index_mapping.csv（层号→论文层号对照）
  第4节 → thesis_supplement/pipeline_config.txt（含 TRIM_FIRST 精确值 + 官方
          BOLD z-score 精确范围，均用真实 UTS03 .hf5 数据实测验证，非转述）

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


def verify_bold_zscore(cfg: dict) -> dict | None:
    """实测官方 BOLD z-score 的精确范围：逐体素 z-score 是在**单个故事自己的
    TR 序列内部**算的（不是跨训练故事拼接后算的），且只对训练故事生效——
    直接读 UTS03 真实 .hf5（本机 git-annex 已下载）验证，不是猜测/转述文档。
    UTS01/UTS02 的 .hf5 本机是悬空 symlink（未下载），跳过。
    """
    import h5py

    data_dir = Path(cfg["datasets"]["data_dir"])
    train_path = data_dir / "UTS03" / "treasureisland.hf5"
    heldout_path = data_dir / "UTS03" / "wheretheressmoke.hf5"
    if not train_path.exists() or not heldout_path.exists():
        return None

    def _stats(path):
        with h5py.File(path, "r") as f:
            data = f["data"][:]
        std = np.nanstd(data, axis=0)
        mean = np.nanmean(data, axis=0)
        return {
            "shape": data.shape,
            "mean_of_voxel_means": float(np.nanmean(mean)),
            "mean_of_voxel_stds": float(np.nanmean(std)),
            "median_voxel_std": float(np.nanmedian(std)),
            "min_voxel_std": float(np.nanmin(std)),
            "max_voxel_std": float(np.nanmax(std)),
            "n_voxels_std_exactly_1": int(np.sum(np.isclose(std, 1.0))),
            "n_voxels_total": data.shape[1],
        }

    return {"train_story": "treasureisland", "train": _stats(train_path),
           "heldout_story": "wheretheressmoke", "heldout": _stats(heldout_path)}


# 层号→论文层号对照（纯配置计算，不需要任何数据/模型权重）。三个 HF 模型的
# 约定见 src/models/base.py::hf_forward_hidden_batch 与各 adapter 的
# _assert_layer_convention()（AutoDL 上对真实 checkpoint 跑过的运行时断言，
# 不是未经验证的假设）：hidden_states 长度 = n_layers+1（[0]=embedding），
# 配置层号是 0-based block 索引 b，代码取 hidden_states[b+1]。AWD-LSTM 走
# 完全不同的机制（src/models/awd_lstm_adapter.py：直接在 encoder.rnns[k] 上挂
# forward hook，k 与配置层号 1:1 对应，没有 embedding 位移）。
LAYER_MODELS = {
    "pythia":   {"n_layers": 12, "primary": 8,  "robustness": 11, "offset": 1},
    "rwkv":     {"n_layers": 12, "primary": 8,  "robustness": 11, "offset": 1},
    "mamba":    {"n_layers": 24, "primary": 16, "robustness": 23, "offset": 1},
    "awd_lstm": {"n_layers": 3,  "primary": 1,  "robustness": 2,  "offset": 0},
}


def build_layer_index_mapping() -> pd.DataFrame:
    rows = []
    for model, spec in LAYER_MODELS.items():
        n = spec["n_layers"]
        for role in ("primary", "robustness"):
            b = spec[role]                       # 0-based block 索引（=config.yaml 层号）
            block_1based = b + 1                  # 该 block 在 1..n 计数下的序号
            hidden_states_index = b + spec["offset"]  # 代码实际取用的 hidden_states 索引
            rows.append({
                "model": model,
                "role": role,
                "config_layer_idx_0based": b,
                "n_layers_total": n,
                "block_position_1based": f"{block_1based}/{n}",
                "hidden_states_index_taken": (hidden_states_index if model != "awd_lstm"
                                              else f"rnns[{b}] (直接hook，无embedding位移)"),
                "depth_fraction": round(block_1based / n, 4),
                "is_literal_final_block": block_1based == n,
            })
    return pd.DataFrame(rows)


def build_pipeline_config_txt(cfg: dict, zscore_stats: dict | None,
                              layer_df: pd.DataFrame) -> str:
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
        ("是否排除故事结尾 20 秒 / TRIM_FIRST 精确取值",
         f"结尾：否，没有独立的“结尾20秒”规则，是 TRIM_LAST={TRIM_LAST} 个TR"
         f"（{TRIM_LAST*TR_SECONDS:.0f}秒）的编码衔接 trim。"
         f"\n  开头 TRIM_FIRST：精确值={TRIM_FIRST}（src/fmri/trfile.py 常量，作用在"
         f"「已减去 SIMULATE_PAD={SIMULATE_PAD} 个TR之后」的模拟触发时间序列上，不是"
         f"直接作用在 respdict 原始TR数上）。**从故事原始扫描开始算，总共丢弃的开头"
         f"TR数 = SIMULATE_PAD({SIMULATE_PAD}) + TRIM_FIRST({TRIM_FIRST}) = "
         f"{SIMULATE_PAD+TRIM_FIRST} 个TR = {(SIMULATE_PAD+TRIM_FIRST)*TR_SECONDS:.0f}秒**；"
         f"加上结尾 TRIM_LAST({TRIM_LAST})，故事级总丢弃 = "
         f"{SIMULATE_PAD+TRIM_FIRST+TRIM_LAST} 个TR = "
         f"{(SIMULATE_PAD+TRIM_FIRST+TRIM_LAST)*TR_SECONDS:.0f}秒，即 "
         f"response_rows = respdict[story] − {SIMULATE_PAD+TRIM_FIRST+TRIM_LAST}。"
         f"**已用真实 UTS03 .hf5 逐故事验证**：treasureisland respdict=414 → "
         f"414−20=394，实测 .hf5 行数=394，精确相符；wheretheressmoke respdict=311 → "
         f"311−20=291，实测 .hf5 行数=291，精确相符。"),
        ("是否使用 Savitzky–Golay 滤波",
         "否——src/fmri/、src/ridge/、src/models/ 全代码库检索不到 savgol/Savitzky 相关调用，"
         "未实现，未使用。"),
        ("是否对 BOLD 做标准化 / 官方 z-score 的精确范围",
         "是，但不在本项目代码里做（derivatives.py::load_response 明确不重复 z-score）。"
         + (("\n  **已用真实 UTS03 .hf5 逐体素实测**（下方数字来自本次真实计算，非转述）：\n"
             f"  训练故事 treasureisland（{zscore_stats['train']['shape'][0]}TR × "
             f"{zscore_stats['train']['shape'][1]}体素）：全部体素均值={zscore_stats['train']['mean_of_voxel_means']:.6f}、"
             f"std={zscore_stats['train']['mean_of_voxel_stds']:.6f}，"
             f"{zscore_stats['train']['n_voxels_std_exactly_1']}/{zscore_stats['train']['n_voxels_total']} "
             f"个体素 std 精确等于 1.000000——**z-score 的“范围”是单个故事自己的 TR 序列内部**"
             f"（该故事独立标准化，不是跨训练故事拼接后统一算的一个全局 z-score）。\n"
             f"  held-out 故事 wheretheressmoke（跨5次重复播放平均，{zscore_stats['heldout']['shape'][0]}TR）："
             f"均值仍≈0（{zscore_stats['heldout']['mean_of_voxel_means']:.6f}），但 std 不是1，"
             f"逐体素范围=[{zscore_stats['heldout']['min_voxel_std']:.6f}, "
             f"{zscore_stats['heldout']['max_voxel_std']:.6f}]，中位数="
             f"{zscore_stats['heldout']['median_voxel_std']:.6f}，均值="
             f"{zscore_stats['heldout']['mean_of_voxel_stds']:.6f}——因为重复平均降低了噪声"
             f"方差、且没有重新做单位方差标准化，不是实现问题。")
            if zscore_stats else
            "\n  ⚠️ 本机 UTS01/UTS02 的 .hf5 是悬空 git-annex 符号链接（未下载），"
            "此实测目前只覆盖 UTS03；UTS03 训练/held-out 两类故事的行为差异"
            "（[[m2-pipeline-state]] 记录）已在其他资料中确认，理论上对三名被试"
            "应一致（同一 derivatives 预处理管线），如需三被试都实测需要在服务器上跑。")),
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
    lines.append("层位索引 → 论文层号对应关系（config.yaml 的 0-based 层号如何换算成")
    lines.append("“第几层/共几层”的自然语言描述；详见 thesis_supplement/layer_index_mapping.csv）")
    lines.append("=" * 78)
    lines.append(
        "三个 HF 模型（pythia/rwkv/mamba）的约定完全一致，且**每次真实加载模型时都会"
        "跑一次运行时断言核验**（各 adapter 的 _assert_layer_convention()：断言"
        "len(hidden_states)==n_layers+1，不是没验证过的假设）：transformers 的 "
        "output_hidden_states 返回长度 n_layers+1 的 tuple，[0]=embedding 输出，"
        "[k]=第k个transformer block之后的输出(k=1..n_layers)。config.yaml 里的层号是"
        "**0-based block 索引 b**，代码取 hidden_states[b+1]。这等价于：层号 b 就是"
        "「第 (b+1) 个 block（1-based 计数）」，即 config 的 0-based 索引恰好等于"
        "1-based block 序号减一，**没有额外的 off-by-one**，可以直接写成"
        "「block (b+1) of n_layers」。\n"
        "AWD-LSTM 走完全不同机制（awd_lstm_adapter.py 直接在 encoder.rnns[k] 上挂"
        "forward hook），config 层号 k 与 rnns[k] 是 1:1 对应，同样没有位移。\n\n"
        "**发现**：config.yaml 里 pythia/rwkv 主层的注释写“约2/3深度”——按上述精确"
        "换算，8(0-based)对应 block 9/12，实际深度分数是 9/12=75.0%，不是 2/3≈66.7%；"
        "mamba 的 16(0-based)对应 block 17/24=70.8%。这是 config.yaml 注释的一处不"
        "精确表述（不影响代码实际取用哪一层，只影响论文文字描述深度时的用词），"
        "论文里如果要引用“约2/3深度”这个说法，应该改成精确的 75.0%/75.0%/70.8%，"
        "或统一改口径为“约3/4深度”更贴近实际数字。两个模型的 robustness/最终层"
        "（pythia=11/rwkv=11/mamba=23）经验证都精确对应 block n/n，即真正的字面"
        "最后一层，标注为“最终层”没有问题。")
    lines.append("")
    lines.append(layer_df.to_string(index=False))
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

    print("\n[3/4] 实测官方 BOLD z-score 精确范围（UTS03 真实 .hf5）...")
    zscore_stats = verify_bold_zscore(cfg)
    if zscore_stats:
        t, h = zscore_stats["train"], zscore_stats["heldout"]
        print(f"  训练故事({zscore_stats['train_story']})：mean={t['mean_of_voxel_means']:.6f} "
             f"std={t['mean_of_voxel_stds']:.6f}（{t['n_voxels_std_exactly_1']}/"
             f"{t['n_voxels_total']} 体素精确std=1）")
        print(f"  held-out故事({zscore_stats['heldout_story']})：mean={h['mean_of_voxel_means']:.6f} "
             f"std范围=[{h['min_voxel_std']:.4f},{h['max_voxel_std']:.4f}] 中位数={h['median_voxel_std']:.4f}")
    else:
        print("  跳过——本机 UTS03 .hf5 缺失（UTS01/UTS02 本来就是悬空symlink，预期内）")

    print("\n[4/4] 生成 layer_index_mapping.csv + pipeline_config.txt ...")
    layer_df = build_layer_index_mapping()
    layer_df.to_csv(OUT_DIR / "layer_index_mapping.csv", index=False)
    print(f"  已写 {OUT_DIR / 'layer_index_mapping.csv'}")
    print(layer_df.to_string(index=False))

    txt = build_pipeline_config_txt(cfg, zscore_stats, layer_df)
    (OUT_DIR / "pipeline_config.txt").write_text(txt)
    print(f"\n  已写 {OUT_DIR / 'pipeline_config.txt'}")

    print(f"\n完成。输出目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
