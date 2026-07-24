"""
Step 5 —— ctx1（刺激侧上下文控制）的配对 story bootstrap 统计。

⚠️ 纯 CPU、无重计算：只读已有的 story 级分数（normal 来自主 M4 cells，ctx1 来自
Step 4 的 m4s_ridge cells），复用 src/stats/bootstrap.py 的 paired_bootstrap。
本地或服务器均可运行。

产出（V6.4.2 §7.3/§7.6，全部诊断性/未校正/非确认性）：
  - 每模型 Δr_total^ctx1 与 95% CI
  - 每模型 D_m = Δr_total^normal − Δr_total^ctx1 与 95% CI
  - ctx1 条件下架构差值 A_MP^ctx1 / A_RP^ctx1
  - 核心交互 I_MP = D_mamba − D_pythia（difference-in-differences）+ CI + CI宽度 + bootstrap SD
  - I_RP = D_rwkv − D_pythia

配对正确性的关键（V6.4.2 §7.2）：normal 与 ctx1 两条件放进**同一个**
BootstrapData，paired_bootstrap 每次重抽对所有 key 用同一批 story 索引 →
D_m 是"同一次抽样内 normal 与 ctx1 相减"，配对 CI 正确。种子沿用主 M5
的 bootstrap seed（config seeds.bootstrap=20260701），与主实验统计口径一致。

用法（本地或服务器，三被试各一次）：
  python 实验补充/scripts/m5s_context_control_stats.py --subject UTS01
  python 实验补充/scripts/m5s_context_control_stats.py --subject UTS02
  python 实验补充/scripts/m5s_context_control_stats.py --subject UTS03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SUPPLEMENT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SUPPLEMENT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SUPPLEMENT_ROOT / "src"))

from src.config_loader import load_config                              # noqa: E402
from src.ridge.score import fisher_z                                   # noqa: E402
from src.stats.bootstrap import (                                      # noqa: E402
    BootstrapData, paired_bootstrap, draws_to_arrays, percentile_ci,
)
from ctx1_estimands import ctx1_estimands, CORE_REPORT_NAMES           # noqa: E402


def _ingest(cell: dict, cond: str, cell_field: str, fold_stories: dict,
            story_pos: dict, z: dict, w: dict, folds: list):
    """把一个 cell 的 per_story（roi_r + n_eff_tr）灌进 z/w 表。
    cell_field 是 per_story 所在字段名：normal cells 用 'normal'，ctx1 cells 用 'ctx1'。"""
    layer, model, H, fold = cell["layer"], cell["model"], cell["H"], cell["fold"]
    if fold not in fold_stories:
        return
    pos = story_pos[fold]
    seen = set()
    for ps in cell[cell_field]["per_story"]:
        s = ps["story"]
        if s not in pos:
            raise ValueError(f"[{model}/H{H}/{fold}/{cond}] 故事 {s} 不在 canonical 列表")
        i = pos[s]
        seen.add(s)
        nw = float(ps["n_eff_tr"])
        for roi, rval in ps["roi_r"].items():
            key = (layer, model, H, cond, roi)
            if key not in z:
                z[key] = {f: np.full(len(fold_stories[f]), np.nan) for f in folds}
                w[key] = {f: np.full(len(fold_stories[f]), np.nan) for f in folds}
            z[key][fold][i] = float(fisher_z(np.asarray(rval)))
            w[key][fold][i] = nw
    missing = set(fold_stories[fold]) - seen
    if missing:
        raise ValueError(f"[{model}/H{H}/{fold}/{cond}] 缺故事评分：{sorted(missing)}")


def load_paired_data(normal_cells_dir: Path, ctx1_cells_dir: Path,
                     fold_stories: dict) -> BootstrapData:
    """构造含 normal + ctx1 两条件的 BootstrapData，并做配对前提校验。"""
    folds = list(fold_stories.keys())
    story_pos = {f: {s: i for i, s in enumerate(ss)} for f, ss in fold_stories.items()}
    z: dict = {}
    w: dict = {}

    # normal：主 M4 的 main_*.json（只取 H∈{8,128}，其余 H/最终层不影响 ctx1 估计量）
    n_norm = 0
    for p in sorted(normal_cells_dir.glob("main_*.json")):
        c = json.load(open(p))
        if c["H"] in (8, 128):
            _ingest(c, "normal", "normal", fold_stories, story_pos, z, w, folds)
            n_norm += 1
    # ctx1：Step 4 的 ctx1_*.json
    n_ctx1 = 0
    for p in sorted(ctx1_cells_dir.glob("ctx1_*.json")):
        c = json.load(open(p))
        _ingest(c, "ctx1", "ctx1", fold_stories, story_pos, z, w, folds)
        n_ctx1 += 1

    if n_ctx1 == 0:
        raise SystemExit(f"未找到 ctx1 cells：{ctx1_cells_dir}（先跑 Step 4）")
    if n_norm == 0:
        raise SystemExit(f"未找到 normal cells：{normal_cells_dir}（主 M4 结果）")

    # 无缺失故事
    for key, byf in z.items():
        for f in folds:
            if np.isnan(byf[f]).any():
                bad = [fold_stories[f][i] for i in np.nonzero(np.isnan(byf[f]))[0]]
                raise ValueError(f"[{key}] fold {f} 缺故事 z：{bad}")

    # 配对前提廉价交叉检查：同一 (model,H,roi) 的 normal 与 ctx1 有效 TR 权重逐故事相等
    # （ctx1 不移时间轴 → mask 逐元素相同 → n_eff 必然相同；bit-level 已在 Step 4 断言，
    #  这里在统计侧再核一次 n_eff 相等，作为独立复核）。
    n_checked = 0
    for key in list(z.keys()):
        layer, model, H, cond, roi = key
        if cond != "normal":
            continue
        ck = (layer, model, H, "ctx1", roi)
        if ck in w:
            for f in folds:
                if not np.array_equal(w[key][f], w[ck][f]):
                    raise ValueError(
                        f"[{model}/H{H}/{roi}/{f}] normal 与 ctx1 有效 TR 权重不等 → "
                        f"mask 不一致（与 Step 4 的 bit-level 断言矛盾，需排查）")
            n_checked += 1
    print(f"  配对校验通过：normal cells={n_norm}, ctx1 cells={n_ctx1}, "
          f"权重一致复核 {n_checked} 个 (model,H,roi) 键", flush=True)

    return BootstrapData(folds=folds, fold_stories=fold_stories, z=z, w=w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03", choices=["UTS01", "UTS02", "UTS03"])
    ap.add_argument("--m4-normal-name", default="m4_full_matrix",
                    help="主 M4 结果目录名（normal cells 来源）")
    ap.add_argument("--m4s-ctx1-name", default="m4s_ridge",
                    help="Step 4 ctx1 结果目录名")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="默认取 config statistics.bootstrap_iterations")
    ap.add_argument("--seed", type=int, default=None,
                    help="默认取 config seeds.bootstrap（与主 M5 同口径）")
    ap.add_argument("--out-name", default="m5s_stats")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    n_boot = args.n_boot if args.n_boot is not None else cfg["statistics"]["bootstrap_iterations"]
    seed = args.seed if args.seed is not None else cfg["seeds"]["bootstrap"]

    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        fold_split = json.load(f)
    fold_stories = {k: sorted(v["test_stories"]) for k, v in fold_split["folds"].items()}

    normal_cells = Path(paths["results_dir"]) / args.m4_normal_name / args.subject / "cells"
    ctx1_cells = Path(paths["results_dir"]) / args.m4s_ctx1_name / args.subject / "cells"

    print(f"[m5s] subject={args.subject} n_boot={n_boot} seed={seed}", flush=True)
    print(f"[m5s] normal cells: {normal_cells}", flush=True)
    print(f"[m5s] ctx1   cells: {ctx1_cells}", flush=True)

    data = load_paired_data(normal_cells, ctx1_cells, fold_stories)

    point, draws = paired_bootstrap(data, ctx1_estimands, n_boot=n_boot, seed=seed)
    arrs = draws_to_arrays(draws)

    estimands = {}
    for name, pv in point.items():
        lo, hi = percentile_ci(arrs[name])
        vals = arrs[name][np.isfinite(arrs[name])]
        estimands[name] = {
            "point": pv, "ci_lo": lo, "ci_hi": hi,
            "ci_width": (hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else float("nan"),
            "bootstrap_sd": float(np.std(vals)) if vals.size else float("nan"),
            "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi)
                                     and ((lo > 0 and hi > 0) or (lo < 0 and hi < 0))),
            "status": "diagnostic_uncorrected_non_confirmatory",
        }

    # 可读小结：核心诊断量（进正文 Table 9）
    print("\n[m5s] === 核心诊断量（diagnostic, uncorrected, 不进确认性家族）===", flush=True)
    for name in CORE_REPORT_NAMES:
        e = estimands.get(name)
        if e is None:
            continue
        star = " *CI≠0" if e["ci_excludes_zero"] else ""
        print(f"  {name:24s} = {e['point']:+.4f}  95%CI[{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}]"
              f"  宽={e['ci_width']:.4f}  SD={e['bootstrap_sd']:.4f}{star}", flush=True)

    manifest = {
        "phase": "M5S stimulus-side context control (ctx1) paired bootstrap",
        "status": "all diagnostic / uncorrected / non-confirmatory (V6.4.2 §7.1/§7.5)",
        "subject": args.subject,
        "n_boot": n_boot, "seed": seed,
        "normal_source": str(normal_cells), "ctx1_source": str(ctx1_cells),
        "estimands": estimands,
        "core_report_names": CORE_REPORT_NAMES,
        "interpretation_note": (
            "D_m>0 且 CI 排除 0 = 该模型 Context Gain 依赖完整语言结构（打乱后下降）；"
            "I_MP 为三级嵌套差值，CI 跨 0 应解读为'精度不足，尚不能判断方向'，非'无交互'"
            "（V6.4.2 §7.3E/§7.4）。§2.4 词形重复偏差系统性压低 D_m，故 D_m 显著为正时结论更稳。"
        ),
    }
    out_dir = Path(paths["results_dir"]) / args.out_name / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "m5s_results.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[m5s] 已写 {out_dir / 'm5s_results.json'}", flush=True)
    print("[m5s] 三被试都跑完后进 Step 6（Figure 17 / Table 9）。", flush=True)


if __name__ == "__main__":
    main()
