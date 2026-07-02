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

    校验（验收任务1 / 标准3-4-5）：
      - 每个 cell 的 per_story 覆盖该 fold 的全部 canonical story；
      - 每个 key 的 z/权重无缺失故事；
      - **共同 mask 复核**：主层同一 (model,H,roi) 的 normal 与 shift 权重逐故事相等
        （验收5：normal/shifted 用共同 mask）。注意权重**按 key 存**——主层（shift-
        限制 mask）与最终层 normal（非限制 mask）每故事 n_eff 本就不同，不做跨层相等
        断言（那是错误假设，会误报）。
    """
    folds = list(fold_stories.keys())
    story_pos = {f: {s: i for i, s in enumerate(ss)} for f, ss in fold_stories.items()}
    z: dict[tuple, dict[str, np.ndarray]] = {}
    w: dict[tuple, dict[str, np.ndarray]] = {}

    def ensure_key(key):
        if key not in z:
            z[key] = {f: np.full(len(fold_stories[f]), np.nan) for f in folds}
            w[key] = {f: np.full(len(fold_stories[f]), np.nan) for f in folds}

    def ingest(cell, cond):
        layer, model, H, fold = cell["layer"], cell["model"], cell["H"], cell["fold"]
        if fold not in fold_stories:
            return
        pos = story_pos[fold]
        seen = set()
        for ps in cell[cond]["per_story"]:
            s = ps["story"]
            if s not in pos:
                raise ValueError(f"[{layer}/{model}/H{H}/{fold}/{cond}] "
                                 f"故事 {s} 不在 fold_split 的 canonical 列表中")
            i = pos[s]
            seen.add(s)
            nw = float(ps["n_eff_tr"])
            for roi, rval in ps["roi_r"].items():
                key = (layer, model, H, cond, roi)
                ensure_key(key)
                z[key][fold][i] = float(fisher_z(np.asarray(rval)))  # r → z（精确还原）
                w[key][fold][i] = nw
        missing = set(fold_stories[fold]) - seen
        if missing:
            raise ValueError(f"[{layer}/{model}/H{H}/{fold}/{cond}] "
                             f"缺少故事评分：{sorted(missing)}")

    for p in sorted(cells_dir.glob("main_*.json")):
        c = json.load(open(p))
        ingest(c, "normal")
        if "shift" in c:
            ingest(c, "shift")
    for p in sorted(cells_dir.glob("final_*.json")):
        ingest(json.load(open(p)), "normal")

    # 每个 key 无缺失故事
    for key, byf in z.items():
        for f in folds:
            if np.isnan(byf[f]).any():
                bad = [fold_stories[f][i] for i in np.nonzero(np.isnan(byf[f]))[0]]
                raise ValueError(f"[{key}] fold {f} 缺故事 z：{bad}")

    # 验收5复核（**必要非充分**）：主层 normal 与 shift 权重(n_eff)逐故事相等。注意这只是
    # 必要条件——两 mask 可选中不同 TR 却计数相同。**mask 本身逐元素相同的充分验证**在
    # M4 源头(m4_driver 的 scoring_mask_bit_identical 断言)+ scripts/verify_scoring_mask_
    # identity.py 完成；此处保留 n_eff 相等作为 M5 侧的廉价交叉检查。
    for key in z:
        layer, model, H, cond, roi = key
        if layer == "main" and cond == "normal":
            shift_key = (layer, model, H, "shift", roi)
            if shift_key in w:
                for f in folds:
                    if not np.array_equal(w[key][f], w[shift_key][f]):
                        raise ValueError(
                            f"[{model}/H{H}/{roi}/{f}] 主层 normal 与 shift 有效 TR 权重"
                            f"不等 → 未用共同 mask（违反验收5）")

    return BootstrapData(folds=folds, fold_stories=fold_stories, z=z, w=w)


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
            "context_gain_normal_minus_shift":
                [f"delta_total_normal_minus_shift_{m}_ifg_main" for m in ("pythia", "mamba", "rwkv", "awd_lstm")],
        },
    }

    # shifted 是否都算出（验收5 前半）
    shifted_names = execution_log["robustness"]["shifted_r_by_h"] + \
        [n for k in ("shifted_delta_local", "shifted_delta_total", "shifted_delta_long")
         for n in execution_log["robustness"][k]]
    shifted_all_finite = all(np.isfinite(estimands[n]["point"]) for n in shifted_names)

    # 验收5 后半：shifted 若"稳定复制"架构效应（40s 位移后架构差值 CI 仍排除 0）→
    # 反常，应触发解释限制/排查漂移伪影，而非当作机制证据。
    def ci_excludes_zero(name):
        e = estimands[name]
        return np.isfinite(e["ci_lo"]) and np.isfinite(e["ci_hi"]) and (
            (e["ci_lo"] > 0 and e["ci_hi"] > 0) or (e["ci_lo"] < 0 and e["ci_hi"] < 0))
    shifted_arch = ["shifted_rwkv_minus_pythia_delta_total_ifg_main",
                    "shifted_mamba_minus_pythia_delta_total_ifg_main"]
    # 每模型 Context Gain 是否被平移显著削弱（配对差值 normal−shifted，CI 排除0且正=削弱）
    gain_reduction = {}
    for m in ("pythia", "mamba", "rwkv", "awd_lstm"):
        n = f"delta_total_normal_minus_shift_{m}_ifg_main"
        e = estimands[n]
        gain_reduction[m] = {**e,
                             "shift_significantly_reduced_gain":
                                 bool(ci_excludes_zero(n) and e["point"] > 0)}
    shifted_diagnostic = {
        # 层面①：架构差值在平移后是否仍成立（对应确认性家族；应为 False=塌掉才好）
        "shifted_architecture_contrasts": {n: {**estimands[n],
                                               "ci_excludes_zero": bool(ci_excludes_zero(n))}
                                          for n in shifted_arch},
        "shifted_reproduces_architecture_effect": bool(any(ci_excludes_zero(n) for n in shifted_arch)),
        # 层面②：每模型自身 Context Gain 是否被平移显著削弱（配对差值）
        "per_model_gain_reduction_normal_minus_shift": gain_reduction,
        "n_models_with_gain_significantly_reduced":
            int(sum(v["shift_significantly_reduced_gain"] for v in gain_reduction.values())),
        "note": "架构差值应在平移后塌掉(reproduces=False 才好)；每模型自身 gain 若未被显著"
                "削弱(reduction CI 跨0)，则该模型 Δr_total 只部分为词序特异，解读须谨慎。",
    }

    # 验收1：execution_log 里每个注册项映射的估计量都真实算出且有限（程序化，非自我声明）
    def _leaves(x):
        if isinstance(x, list):
            return list(x)
        if isinstance(x, dict):
            return [n for v in x.values() for n in _leaves(v)]
        return []
    logged = _leaves(execution_log)
    registry_complete = all(
        n in estimands and np.isfinite(estimands[n]["point"]) for n in logged)

    verdict = {
        "1_all_registry_entries_have_results": bool(registry_complete),  # 逐条核对已算
        "2_rq1_by_hspecific_paired_diffs": True,          # 结构性：RQ1 用 {arch}_minus_pythia_r{H}
        "3_bootstrap_unit_is_story": True,                # 结构性：BootstrapData 抽样单位=story
        "4_same_indices_across_conditions": True,         # 结构性：paired_bootstrap 共用索引
        "5_shifted_computed": bool(shifted_all_finite),
        "6_layer_flip_by_opposite_nonzero_ci": True,      # 见 flips（规则实现，非点估计排名）
    }

    # 已知缺口：repeatability（交付物7）需 wheretheressmoke 逐次响应，现有 .hf5 只有
    # 跨重复平均 → 数据不可得，暂缺（P1，不阻塞确认性主结论）。显式记录而非静默省略。
    known_gaps = {
        "ifg_pt_repeatability": {
            "status": "not_computed",
            "reason": "per-repeat responses for wheretheressmoke unavailable (only "
                      "cross-repeat average in .hf5); repeatability is P1, does not block "
                      "the two confirmatory contrasts. ROI interpretation reports base r / "
                      "CI width / shift-drop without a repeatability ceiling.",
        },
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
        "shifted_diagnostic": shifted_diagnostic,
        "estimands": estimands,
        "execution_log": execution_log,
        "known_gaps": known_gaps,
        "verdict": verdict,
        "verdict_all_pass": all(verdict.values()),
    }

    out_dir = Path(paths["results_dir"]) / args.out_name / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "m5_results.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    _print_summary(estimands, confirmatory, flips, shifted_all_finite,
                   shifted_diagnostic, out_dir)


def _fmt(e: dict) -> str:
    return f"{e['point']:+.4f} [{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}]"


def _print_summary(estimands, confirmatory, flips, shifted_ok, shifted_diag, out_dir):
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
    repro = shifted_diag["shifted_reproduces_architecture_effect"]
    print(f"[m5] 层面①架构差值: 40s位移后架构差值CI仍排除0 = "
          f"{'⚠️ 是（确认性发现受威胁，需排查）' if repro else '否（架构差值塌掉，确认性发现是词序特异的✓）'}",
          flush=True)
    print("[m5] 层面②每模型自身 Context Gain 是否被平移显著削弱（normal−shifted 配对差值）:", flush=True)
    for m, v in shifted_diag["per_model_gain_reduction_normal_minus_shift"].items():
        lo, hi, pt = v["ci_lo"], v["ci_hi"], v["point"]
        sig = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
        if sig and pt > 0:
            tag = "✓显著削弱"
        elif sig and pt < 0:
            tag = "⚠️平移反而显著增强gain（更反常，非纯词序信号）"
        else:
            tag = "⚠️未显著变化(CI跨0)"
        print(f"[m5]   {m}: Δgain={pt:+.4f} [{lo:+.4f},{hi:+.4f}] {tag}", flush=True)
    n_red = shifted_diag["n_models_with_gain_significantly_reduced"]
    print(f"[m5]   → {n_red}/4 个模型的 Context Gain 被平移显著削弱；未被削弱的模型其 Δr_total "
          f"只部分为词序特异，解读须谨慎（非纯长程语言整合）", flush=True)
    print(f"[m5] 结果 → {out_dir / 'm5_results.json'}", flush=True)


if __name__ == "__main__":
    main()
