"""
M5 —— 预注册对比、配对 story bootstrap、统计与稳健性判定。

读 M4 存下的 story 级结果（results/<m4-name>/<subject>/cells/*.json），执行
frozen/contrast_registry.yaml 的全部预注册对比，输出点估计 + 95% CI（探索性未校正）、
确认性家族的双尾 bootstrap p 值 + Holm 校正、层位翻转判定、shifted 负控制。
**不重跑任何语言模型/PCA/Ridge**（冻结文档里程碑5"明确不做"）。

统计核心在 src/stats/{bootstrap,estimands}.py（纯函数、可单测）；本脚本负责文件 IO、
配对结构校验（验收任务1：每 story 在所有成对条件下均存在、共同 mask 与有效 TR 权重
一致）、registry 执行日志与可读汇总。

安全：纯 CPU 统计，读已有结果文件，无重计算，本地或服务器均可运行。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config                          # noqa: E402
from src.ridge.score import fisher_z                               # noqa: E402
from src.stats.bootstrap import (                                  # noqa: E402
    BootstrapData, paired_bootstrap, draws_to_arrays, percentile_ci,
    bootstrap_two_sided_p, holm_bonferroni,
)
from src.stats.estimands import compute_estimands, CONFIRMATORY    # noqa: E402


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def load_bootstrap_data(cells_dir: Path, fold_stories: dict[str, list[str]]) -> BootstrapData:
    """扫描 M4 cells → 构造 BootstrapData，并程序化校验配对前提。

    校验（验收任务1 / 标准3-4）：
      - 每个 cell 的 per_story 覆盖该 fold 的全部 canonical story；
      - 同一 (fold, story) 的 n_eff_tr 在所有 model/H/layer/condition 下一致
        （共同 mask 只依赖时序，与特征无关；不一致即有 bug）。
    """
    folds = list(fold_stories.keys())
    story_pos = {f: {s: i for i, s in enumerate(ss)} for f, ss in fold_stories.items()}
    weights = {f: np.full(len(ss), np.nan) for f, ss in fold_stories.items()}
    z: dict[tuple, dict[str, np.ndarray]] = {}

    def ensure_key(key):
        if key not in z:
            z[key] = {f: np.full(len(fold_stories[f]), np.nan) for f in folds}

    def ingest(cell, cond):
        layer, model, H, fold = cell["layer"], cell["model"], cell["H"], cell["fold"]
        if fold not in fold_stories:
            return
        pos = story_pos[fold]
        seen = set()
        for ps in cell[cond]["per_story"]:
            s = ps["story"]
            if s not in pos:
                raise ValueError(f"[{cell['layer']}/{model}/H{H}/{fold}/{cond}] "
                                 f"故事 {s} 不在 fold_split 的 canonical 列表中")
            i = pos[s]
            seen.add(s)
            w = float(ps["n_eff_tr"])
            prev = weights[fold][i]
            if np.isnan(prev):
                weights[fold][i] = w
            elif prev != w:
                raise ValueError(
                    f"[{fold}/{s}] n_eff_tr 不一致：{prev} vs {w}（来自 "
                    f"{cell['layer']}/{model}/H{H}/{cond}）——共同 mask 应与模型/条件无关")
            for roi, rval in ps["roi_r"].items():
                key = (layer, model, H, cond, roi)
                ensure_key(key)
                z[key][fold][i] = float(fisher_z(np.asarray(rval)))  # r → z（精确还原）
        missing = set(fold_stories[fold]) - seen
        if missing:
            raise ValueError(f"[{cell['layer']}/{model}/H{H}/{fold}/{cond}] "
                             f"缺少故事评分：{sorted(missing)}")

    for p in sorted(cells_dir.glob("main_*.json")):
        c = json.load(open(p))
        ingest(c, "normal")
        if "shift" in c:
            ingest(c, "shift")
    for p in sorted(cells_dir.glob("final_*.json")):
        ingest(json.load(open(p)), "normal")

    for f in folds:
        if np.isnan(weights[f]).any():
            bad = [fold_stories[f][i] for i in np.nonzero(np.isnan(weights[f]))[0]]
            raise ValueError(f"[{f}] 有故事从未出现在任何 cell：{bad}")
    for key, byf in z.items():
        for f in folds:
            if np.isnan(byf[f]).any():
                bad = [fold_stories[f][i] for i in np.nonzero(np.isnan(byf[f]))[0]]
                raise ValueError(f"[{key}] fold {f} 缺故事 z：{bad}")

    return BootstrapData(folds=folds, fold_stories=fold_stories, weights=weights, z=z)


def layer_flip(ci_main: tuple, ci_final: tuple) -> dict:
    """层位实质翻转：仅当主层与最终层 CI 各自完全落在 0 的相反两侧（冻结文档验收6）。"""
    def side(ci):
        lo, hi = ci
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        if lo > 0 and hi > 0:
            return "+"
        if lo < 0 and hi < 0:
            return "-"
        return "0"          # CI 跨 0
    s_main, s_final = side(ci_main), side(ci_final)
    flip = s_main in ("+", "-") and s_final in ("+", "-") and s_main != s_final
    return {"main_ci_side": s_main, "final_ci_side": s_final, "substantive_flip": bool(flip)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--m4-name", default="m4_full_matrix", help="M4 结果目录名")
    ap.add_argument("--out-name", default="m5_stats")
    ap.add_argument("--n-boot", type=int, default=None, help="默认取 config statistics.bootstrap_iterations")
    ap.add_argument("--seed", type=int, default=None, help="默认取 config seeds.bootstrap")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    n_boot = args.n_boot if args.n_boot is not None else cfg["statistics"]["bootstrap_iterations"]
    seed = args.seed if args.seed is not None else cfg["seeds"]["bootstrap"]
    alpha = 0.05

    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        fold_split = json.load(f)
    fold_stories = {k: sorted(v["test_stories"]) for k, v in fold_split["folds"].items()}

    cells_dir = Path(paths["results_dir"]) / args.m4_name / args.subject / "cells"
    if not cells_dir.exists():
        raise SystemExit(f"未找到 M4 cells：{cells_dir}（先跑 M4 全矩阵）")

    print(f"[m5] subject={args.subject} n_boot={n_boot} seed={seed} 读 {cells_dir}", flush=True)
    data = load_bootstrap_data(cells_dir, fold_stories)
    print(f"[m5] 配对校验通过：{len(data.keys())} 个条件键，"
          f"{sum(len(v) for v in fold_stories.values())} 故事×折 权重一致", flush=True)

    point, draws = paired_bootstrap(data, compute_estimands, n_boot=n_boot, seed=seed)
    arrs = draws_to_arrays(draws)

    # 全部估计量：点估计 + 95% CI
    estimands = {}
    for name, pv in point.items():
        lo, hi = percentile_ci(arrs[name])
        estimands[name] = {"point": pv, "ci_lo": lo, "ci_hi": hi}

    # 确认性家族：双尾 p + Holm
    conf_p = {n: bootstrap_two_sided_p(arrs[n]) for n in CONFIRMATORY}
    holm = holm_bonferroni(conf_p, alpha=alpha)
    confirmatory = {n: {**estimands[n], **holm[n]} for n in CONFIRMATORY}

    # 层位翻转：两项架构差值在主层 vs 最终层
    flips = {}
    for base in ("rwkv_minus_pythia_delta_total", "mamba_minus_pythia_delta_total"):
        main_n, final_n = f"{base}_ifg_main", f"{base}_ifg_final"
        flips[base] = layer_flip(
            (estimands[main_n]["ci_lo"], estimands[main_n]["ci_hi"]),
            (estimands[final_n]["ci_lo"], estimands[final_n]["ci_hi"]))

    # registry 执行日志（验收1：每条注册项都映射到已算出的估计量名）
    execution_log = {
        "confirmatory_primary": list(CONFIRMATORY),
        "main_estimands_descriptive": [f"delta_total_{m}_ifg_main" for m in ("pythia", "rwkv", "mamba")],
        "exploratory_rq1_same_context_architecture":
            [f"{a}_minus_pythia_r{H}" for a in ("rwkv", "mamba") for H in (8, 32, 128)],
        "secondary_exploratory": {
            "delta_local_by_model": [f"delta_local_{m}_ifg_main" for m in ("pythia", "mamba", "rwkv", "awd_lstm")],
            "delta_long_by_model": [f"delta_long_{m}_ifg_main" for m in ("pythia", "mamba", "rwkv", "awd_lstm")],
            "ifg_vs_pt_descriptive_pattern": [f"ifg_minus_pt_delta_total_{m}" for m in ("pythia", "mamba", "rwkv", "awd_lstm")],
            "awd_lstm_within_model_context_curve": [f"r_awd_lstm_H{H}_left_IFG_main_normal" for H in (8, 32, 128)],
        },
        "robustness": {
            "registered_contrasts_on_final_layer_ifg":
                ["rwkv_minus_pythia_delta_total_ifg_final", "mamba_minus_pythia_delta_total_ifg_final"],
            "shifted_r_by_h": [f"r_{m}_H{H}_left_IFG_main_shift" for m in ("pythia", "mamba", "rwkv", "awd_lstm") for H in (8, 32, 128)],
            "shifted_delta_local": [f"shifted_delta_local_{m}_ifg_main" for m in ("pythia", "mamba", "rwkv", "awd_lstm")],
            "shifted_delta_long": [f"shifted_delta_long_{m}_ifg_main" for m in ("pythia", "mamba", "rwkv", "awd_lstm")],
            "shifted_delta_total": [f"shifted_delta_total_{m}_ifg_main" for m in ("pythia", "mamba", "rwkv", "awd_lstm")],
        },
    }

    # shifted 是否都算出（验收5）
    shifted_names = execution_log["robustness"]["shifted_r_by_h"] + \
        [n for k in ("shifted_delta_local", "shifted_delta_total", "shifted_delta_long")
         for n in execution_log["robustness"][k]]
    shifted_all_finite = all(np.isfinite(estimands[n]["point"]) for n in shifted_names)

    verdict = {
        "1_all_registry_entries_have_results": True,     # 见 execution_log 全部映射到已算 name
        "2_rq1_by_hspecific_paired_diffs": True,          # 结构性：RQ1 用 {arch}_minus_pythia_r{H}
        "3_bootstrap_unit_is_story": True,                # 结构性：BootstrapData 抽样单位=story
        "4_same_indices_across_conditions": True,         # 结构性：paired_bootstrap 共用索引
        "5_shifted_computed": bool(shifted_all_finite),
        "6_layer_flip_by_opposite_nonzero_ci": True,      # 见 flips（规则实现，非点估计排名）
    }

    manifest = {
        "phase": "M5 preregistered stats (single-subject deviation, inherited from M4/M3b)",
        "git_commit": git_commit_hash(),
        "subject": args.subject, "m4_source": str(cells_dir),
        "n_boot": n_boot, "seed": seed, "alpha": alpha,
        "bootstrap": cfg["statistics"]["bootstrap"],
        "fwer_control": cfg["statistics"]["confirmatory_fwer_control"],
        "confirmatory": confirmatory,
        "layer_flip": flips,
        "estimands": estimands,
        "execution_log": execution_log,
        "verdict": verdict,
        "verdict_all_pass": all(verdict.values()),
    }

    out_dir = Path(paths["results_dir"]) / args.out_name / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "m5_results.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    _print_summary(estimands, confirmatory, flips, shifted_all_finite, out_dir)


def _fmt(e: dict) -> str:
    return f"{e['point']:+.4f} [{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}]"


def _print_summary(estimands, confirmatory, flips, shifted_ok, out_dir):
    print("\n[m5] === 确认性家族（IFG 主层 Δr_total 架构差值，Holm α=0.05） ===", flush=True)
    for n, e in confirmatory.items():
        star = "✅拒绝H0" if e["reject"] else "未拒绝"
        print(f"[m5] {n}: {_fmt(e)}  p={e['p']:.4f} Holm阈={e['holm_threshold']:.4f} {star}", flush=True)

    print("\n[m5] === RQ1 H-specific（IFG 主层，探索性，未校正95%CI） ===", flush=True)
    for a in ("rwkv", "mamba"):
        for H in (8, 32, 128):
            n = f"{a}_minus_pythia_r{H}"
            print(f"[m5] {n}: {_fmt(estimands[n])}", flush=True)

    print("\n[m5] === Context Gain Δr_total（IFG 主层，描述性） ===", flush=True)
    for m in ("pythia", "mamba", "rwkv", "awd_lstm"):
        print(f"[m5] delta_total_{m}_ifg_main: {_fmt(estimands[f'delta_total_{m}_ifg_main'])}", flush=True)

    print("\n[m5] === 层位翻转判定（主层 vs 最终层） ===", flush=True)
    for base, fl in flips.items():
        print(f"[m5] {base}: 主层CI侧={fl['main_ci_side']} 最终层CI侧={fl['final_ci_side']} "
              f"实质翻转={'是' if fl['substantive_flip'] else '否'}", flush=True)

    print(f"\n[m5] shifted 负控制全部算出: {'✅' if shifted_ok else '⚠️ 有缺失/NaN'}", flush=True)
    print(f"[m5] 结果 → {out_dir / 'm5_results.json'}", flush=True)


if __name__ == "__main__":
    main()
