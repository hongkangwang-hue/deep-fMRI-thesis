"""
§2.2 lag-1 平滑度诊断（补漏项）——纯 CPU，只读已有特征缓存，不加载 BOLD。

## 为什么现在才补

写「执行版」清单时我把 §2.2 漏掉了（freeze_manifest.json 里也没冻结它），但它是
V6.4.2 的 §4 交付物#8、§5 验收标准#10，且 §2.2 明写「论文中必须同时报告」。
本脚本补齐这个缺口。

## 它现在要回答的真问题（比原规范更具体）

Step 5 结果显示：Pythia 的 Δr_total 在 ctx1 下翻负（三被试显著）、I_MP 三被试为负。
最有力的替代解释是「打乱改变了特征时间统计，而 Pythia 受影响更大」。本脚本除按
§2.2 出标准诊断外，额外算两个**事后描述性**对比（输出里标 post_hoc=True）：
  - 跨模型：各模型 Δlag1 相对 Pythia 的差 → I_MP 是否被平滑度差异混淆；
  - 跨 H：Δlag1(H=128) − Δlag1(H=8) → 打乱的破坏是否随 H 累积。

## 内存（吸取 Step 4 的教训）

只调用 load_features + Lanczos 重采样得到 TR 级特征，**不加载任何 BOLD**，逐故事
算完标量即丢弃。峰值内存约等于单个故事的特征矩阵，远低于 Step 4。

用法（服务器或本地，需 cache/features 与 cache/features_ctx1 可达）：
  python 实验补充/scripts/m4s_lag1_diagnostic.py
输出：实验补充/results/lag1_diagnostic.json + lag1_by_story.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SUPPLEMENT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SUPPLEMENT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "encoding"))
sys.path.insert(0, str(SUPPLEMENT_ROOT / "src"))

from src.config_loader import load_config                                   # noqa: E402
from src.fmri.alignment import word_to_tr                                   # noqa: E402
from src.fmri.trfile import load_respdict, story_tr_times, TRIM_FIRST, TRIM_LAST  # noqa: E402
from src.models.feature_cache import load_features                          # noqa: E402
from src.ridge.assemble import _word_times                                  # noqa: E402
from lag1_diagnostic import (                                               # noqa: E402
    lag1_cosine_median, lag1_featurewise_corr_mean, tolerance_band,
    precision_interval, paired_delta, audit_triggers,
    compare_across_models, h_dependence,
)

CORE_MODELS = ["pythia", "rwkv", "mamba"]
H_LIST = [8, 128]          # ctx1 实际只跑了这两档（按项目实况，非规范原文的 {8,32,128}）
LAYER = "main"             # ctx1 只跑了主层


def tr_level_features(cache_dir, model: str, story: str, H: int,
                      respdict: dict, word_index: pd.DataFrame) -> np.ndarray:
    """复刻 assemble_story 的**特征侧**：Lanczos 下采样 + trim，但不碰 BOLD。

    与 src/ridge/assemble.py::assemble_story 的特征路径逐行一致（同一 _word_times、
    同一 story_tr_times、同一 TRIM_FIRST/TRIM_LAST），保证这里算 lag-1 的 X 与真正
    喂进 PCA/FIR 的 X 是同一个东西——§2.2 要求的「Lanczos 后、FIR 前」正是此处。
    """
    feat = load_features(cache_dir, model, story, H)
    vecs = feat[LAYER].astype(np.float64)
    data_times = _word_times(word_index, feat["word_ids"])
    order = np.argsort(data_times)
    X_full = word_to_tr(vecs[order], data_times[order], story_tr_times(respdict[story]))
    return X_full[TRIM_FIRST: len(X_full) - TRIM_LAST]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir-ctx1", default=None,
                    help="默认 <cache_dir 同级>/features_ctx1")
    ap.add_argument("--models", nargs="+", default=CORE_MODELS)
    ap.add_argument("--H", nargs="+", type=int, default=H_LIST)
    ap.add_argument("--seed", type=int, default=None,
                    help="精度区间 bootstrap 种子；默认取 config seeds.bootstrap")
    args = ap.parse_args()

    cfg = load_config()
    paths, ds = cfg["paths"], cfg["datasets"]
    seed = args.seed if args.seed is not None else cfg["seeds"]["bootstrap"]
    cache_normal = Path(paths["cache_dir"])
    cache_ctx1 = (Path(args.cache_dir_ctx1) if args.cache_dir_ctx1
                  else cache_normal.parent / "features_ctx1")

    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        fold_split = json.load(f)
    stories = sorted({s for fo in fold_split["folds"].values()
                      for s in fo["train_stories"] + fo["test_stories"]})
    respdict = load_respdict(ds["respdict"])
    word_index = pd.read_parquet(Path(paths["frozen_dir"]) / "word_index.parquet")

    print(f"[lag1] {len(stories)} 故事 × {args.models} × H={args.H} × "
          f"{{normal, ctx1}}，只读特征不加载 BOLD", flush=True)

    # 1) 逐 (model, H, condition, story) 算标量
    rows = []
    for model in args.models:
        for H in args.H:
            for cond, cdir in (("normal", cache_normal), ("ctx1", cache_ctx1)):
                for st in stories:
                    try:
                        X = tr_level_features(cdir, model, st, H, respdict, word_index)
                    except FileNotFoundError:
                        print(f"  [跳过] {model}/H{H}/{cond}/{st} 缺缓存", flush=True)
                        continue
                    rows.append({
                        "model": model, "H": H, "condition": cond, "story": st,
                        "lag1_cosine_median": lag1_cosine_median(X),
                        "lag1_featurewise_corr_mean": lag1_featurewise_corr_mean(X),
                    })
                print(f"  {model}/H{H}/{cond} 完成 {len(stories)} 故事", flush=True)
    df = pd.DataFrame(rows)

    out_dir = SUPPLEMENT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "lag1_by_story.csv", index=False)
    print(f"[lag1] 已写 {out_dir / 'lag1_by_story.csv'}（逐故事明细）", flush=True)

    # 2) 按 (model, H) 出 §2.2 标准诊断
    result = {
        "metric_primary": "lag1_cosine_median (cos of adjacent TR representations, "
                          "median per story, computed after Lanczos and before FIR)",
        "metric_supplementary": "lag1_featurewise_corr_mean",
        "spec_source": "V6.4.2 §2.2（指标/容差带/精度区间/审计三规则与 k=1.0 均照搬冻结原文）",
        "timing_disclosure": (
            "本诊断的实现与执行晚于 Step 5 的 D_m/I_MP 结果产出。指标定义与阈值取自"
            "ctx1 运行前即已冻结的 V6.4.2 §2.2，未做任何事后调参；但论文须如实写明"
            "该诊断为补做，且 compare_across_models / h_dependence 两项为事后新增的"
            "描述性分析。"),
        "scope_note": f"H={args.H}、layer={LAYER}（对齐 ctx1 实际运行范围）；C2 未执行故不涉及",
        "by_model_H": {},
    }
    delta_by_model_H: dict[tuple, dict] = {}

    for model in args.models:
        for H in args.H:
            sub = df[(df["model"] == model) & (df["H"] == H)]
            nrm = sub[sub["condition"] == "normal"].set_index("story")["lag1_cosine_median"]
            ctx = sub[sub["condition"] == "ctx1"].set_index("story")["lag1_cosine_median"]
            if nrm.empty or ctx.empty:
                continue
            band = tolerance_band(nrm.to_numpy())
            prec = precision_interval(nrm.to_numpy(), seed=seed)
            delta = paired_delta(nrm.to_dict(), ctx.to_dict())
            audit = audit_triggers(delta, band, ctx.to_numpy())
            delta_by_model_H[(model, H)] = delta

            key = f"{model}_H{H}"
            result["by_model_H"][key] = {
                "normal_tolerance_band": band,
                "normal_precision_interval": prec,
                "ctx1_median": float(np.median(ctx.to_numpy())),
                "paired_delta_lag1": {k: v for k, v in delta.items() if k != "by_story"},
                "audit_triggers": audit,
            }
            print(f"\n[lag1] {key}", flush=True)
            print(f"  normal 中位数={band['median']:.4f} (SD={band['sd']:.4f}, "
                  f"容差带[{band['p2_5']:.4f}, {band['p97_5']:.4f}])", flush=True)
            print(f"  ctx1   中位数={result['by_model_H'][key]['ctx1_median']:.4f}", flush=True)
            print(f"  Δlag1  中位数={delta['median']:+.4f} "
                  f"[{delta['p2_5']:+.4f}, {delta['p97_5']:+.4f}]  "
                  f"(>0=打乱后更不平滑)", flush=True)
            print(f"  审计触发={audit['any_triggered']}", flush=True)

    # 3) 事后新增：跨模型（I_MP 混淆检查）与跨 H（OOD 随 H 累积）
    result["post_hoc_cross_model"] = {}
    for H in args.H:
        dbm = {m: delta_by_model_H[(m, H)] for m in args.models
               if (m, H) in delta_by_model_H}
        if len(dbm) >= 2:
            result["post_hoc_cross_model"][f"H{H}"] = compare_across_models(dbm, "pythia")

    result["post_hoc_h_dependence"] = {}
    for model in args.models:
        dbh = {H: delta_by_model_H[(model, H)] for H in args.H
               if (model, H) in delta_by_model_H}
        if len(dbh) >= 2:
            result["post_hoc_h_dependence"][model] = h_dependence(dbh)

    print("\n[lag1] === 事后对比：Δlag1 跨模型（检查 I_MP 是否被平滑度差异混淆）===", flush=True)
    for hk, blk in result["post_hoc_cross_model"].items():
        for k, v in blk.items():
            if isinstance(v, dict) and "median" in v:
                print(f"  {hk} {k}: 中位数={v['median']:+.4f} "
                      f"[{v['p2_5']:+.4f}, {v['p97_5']:+.4f}]", flush=True)

    print("\n[lag1] === 事后对比：Δlag1 的 H 依赖（打乱破坏是否随 H 累积）===", flush=True)
    for model, blk in result["post_hoc_h_dependence"].items():
        v = blk.get("delta_H128_minus_H8")
        if v:
            print(f"  {model}: Δlag1(H128)−Δlag1(H8) 中位数={v['median']:+.4f} "
                  f"[{v['p2_5']:+.4f}, {v['p97_5']:+.4f}]", flush=True)

    with open(out_dir / "lag1_diagnostic.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[lag1] 已写 {out_dir / 'lag1_diagnostic.json'}", flush=True)


if __name__ == "__main__":
    main()
