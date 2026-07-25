"""
§2.3 OOD 退化的 H 依赖性诊断（两项主指标，纯 CPU，不需要 GPU）。

## 为什么不需要 GPU（更正此前估计）

我先前告诉用户 §2.3 需要约 1 小时 GPU——那个估计只对**辅助**指标 NLL 成立。
两项**主**指标都能从已有特征缓存直接算：
  - L2 范数：对缓存里的目标位置隐表示求范数；
  - 有效维度 / evr@100：对 TR 级特征做 SVD。
NLL 才需要重跑模型拿 logits，而规范明确把它降级为辅助、并警告不能单独据其判断
（它分不清「上下文信息量下降」这一**期望效果**与「模型 OOD 退化」这一混淆项）。
本脚本只做两项主指标；NLL 未做，须在 Limitations 注明。

## 与 lag-1 的分工

lag-1 测的是「相邻 TR 表示有多像」——间接反映表示是否变呆。
本诊断测的是「表示本身的几何是否塌缩」——L2 范数与有效维度是直接刻画。
两者相互印证：若 lag-1 升高**且**有效维度下降，则「表示退化」的证据链完整。

用法（服务器或本地，需 cache/features 与 cache/features_ctx1 可达）：
  python 实验补充/scripts/m4s_ood_diagnostic.py
输出：实验补充/results/ood_diagnostic.json + ood_by_story.csv
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
from ood_diagnostic import (                                                # noqa: E402
    l2_norm_median, participation_ratio, evr_at_k,
    h_change_by_story, ood_h_dependence, verdict,
)

CORE_MODELS = ["pythia", "rwkv", "mamba"]
H_LIST = [8, 128]
LAYER = "main"
METRICS = ["l2_norm_median", "participation_ratio", "evr_at_100"]


def story_metrics(cache_dir, model: str, story: str, H: int,
                  respdict: dict, word_index: pd.DataFrame) -> dict:
    """一个 (model, story, H, condition) 的三个指标。

    L2 范数用**词级**隐表示（§2.3 说的「目标位置隐表示」）；
    有效维度/evr 用 **TR 级**特征（§2.3 说的「PCA 前特征矩阵」，与 pipeline 里
    真正喂给 PCA 的是同一个东西）。
    """
    feat = load_features(cache_dir, model, story, H)
    H_word = feat[LAYER].astype(np.float64)

    data_times = _word_times(word_index, feat["word_ids"])
    order = np.argsort(data_times)
    X_full = word_to_tr(H_word[order], data_times[order],
                        story_tr_times(respdict[story]))
    X_tr = X_full[TRIM_FIRST: len(X_full) - TRIM_LAST]

    return {
        "l2_norm_median": l2_norm_median(H_word),
        "participation_ratio": participation_ratio(X_tr),
        "evr_at_100": evr_at_k(X_tr, k=100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir-ctx1", default=None)
    ap.add_argument("--models", nargs="+", default=CORE_MODELS)
    ap.add_argument("--H", nargs="+", type=int, default=H_LIST)
    args = ap.parse_args()

    cfg = load_config()
    paths, ds = cfg["paths"], cfg["datasets"]
    cache_normal = Path(paths["cache_dir"])
    cache_ctx1 = (Path(args.cache_dir_ctx1) if args.cache_dir_ctx1
                  else cache_normal.parent / "features_ctx1")

    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        fold_split = json.load(f)
    stories = sorted({s for fo in fold_split["folds"].values()
                      for s in fo["train_stories"] + fo["test_stories"]})
    respdict = load_respdict(ds["respdict"])
    word_index = pd.read_parquet(Path(paths["frozen_dir"]) / "word_index.parquet")

    print(f"[ood] {len(stories)} 故事 × {args.models} × H={args.H} × "
          f"{{normal, ctx1}}（纯 CPU，主指标不需 GPU）", flush=True)

    rows = []
    for model in args.models:
        for H in args.H:
            for cond, cdir in (("normal", cache_normal), ("ctx1", cache_ctx1)):
                for st in stories:
                    try:
                        m = story_metrics(cdir, model, st, H, respdict, word_index)
                    except FileNotFoundError:
                        continue
                    rows.append({"model": model, "H": H, "condition": cond,
                                 "story": st, **m})
                print(f"  {model}/H{H}/{cond} 完成", flush=True)
    df = pd.DataFrame(rows)

    out_dir = SUPPLEMENT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ood_by_story.csv", index=False)
    print(f"[ood] 已写 {out_dir / 'ood_by_story.csv'}", flush=True)

    result = {
        "spec_source": "V6.4.2 §2.3（主指标 L2 范数 + 有效维度；NLL 为辅助，本轮未做）",
        "why_no_gpu": "两项主指标可从既有特征缓存直接计算；只有辅助指标 NLL 需要 "
                      "GPU 重跑模型，且规范明确其不可单独作为退化判据。",
        "core_comparison": "ctx1 在 H8→H128 的变化 减去 normal 的对应变化（逐故事配对 DiD）",
        "by_model": {},
    }

    for model in args.models:
        sub = df[df["model"] == model]
        per_metric = {}
        for metric in METRICS:
            def by_story(cond, H):
                s = sub[(sub["condition"] == cond) & (sub["H"] == H)]
                return s.set_index("story")[metric].to_dict()

            ch_n = h_change_by_story(by_story("normal", 8), by_story("normal", 128))
            ch_c = h_change_by_story(by_story("ctx1", 8), by_story("ctx1", 128))
            ood = ood_h_dependence(ch_n, ch_c)
            per_metric[metric] = {
                "normal_H_change": {k: v for k, v in ch_n.items() if k != "by_story"},
                "ctx1_H_change": {k: v for k, v in ch_c.items() if k != "by_story"},
                "ood_h_dependence_did": ood,
                "absolute_medians": {
                    f"{cond}_H{H}": float(np.median(
                        sub[(sub["condition"] == cond) & (sub["H"] == H)][metric].dropna()))
                    for cond in ("normal", "ctx1") for H in args.H
                },
            }
        v = verdict({m: per_metric[m]["ood_h_dependence_did"] for m in METRICS})
        result["by_model"][model] = {"metrics": per_metric, "verdict": v}

        print(f"\n[ood] === {model} ===", flush=True)
        for metric in METRICS:
            pm = per_metric[metric]
            a = pm["absolute_medians"]
            d = pm["ood_h_dependence_did"]
            print(f"  {metric}", flush=True)
            print(f"    normal: H8={a['normal_H8']:.4g} → H128={a['normal_H128']:.4g}  "
                  f"(Δ={pm['normal_H_change']['median']:+.4g})", flush=True)
            print(f"    ctx1  : H8={a['ctx1_H8']:.4g} → H128={a['ctx1_H128']:.4g}  "
                  f"(Δ={pm['ctx1_H_change']['median']:+.4g})", flush=True)
            star = " *CI≠0" if d["ci_excludes_zero"] else ""
            print(f"    DiD(ctx1−normal) = {d['median']:+.4g} "
                  f"[{d['p2_5']:+.4g}, {d['p97_5']:+.4g}]{star}", flush=True)
        print(f"  判读: {v['conclusion']}", flush=True)

    with open(out_dir / "ood_diagnostic.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[ood] 已写 {out_dir / 'ood_diagnostic.json'}", flush=True)


if __name__ == "__main__":
    main()
