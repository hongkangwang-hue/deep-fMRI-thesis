"""
M5 跨被试综合 —— 方向一致性描述性判读（严格不池化、不生成组水平CI）。

读取每名被试各自的 M5 逐被试结果（results/<m5-name>/<subject>/m5_results.json，由
scripts/m5_analysis.py 逐被试生成），对两项确认性 Δr_total 架构差值（RWKV−Pythia、
Mamba−Pythia）与 RQ1 H-specific 差值，判读其方向与量级是否在各被试身上一致重现：

  一致 / 部分一致 / 不一致 / 数据不足（判读规则见 src/stats/cross_subject.py）。

严格遵循冻结文档里程碑5"明确不做"1-2：不把三名被试合并成人群样本做池化/组水平
推断，不生成合并后的组水平CI。本脚本只读逐被试 M5 已算好的点估计与CI，做描述性
方向一致性判读，**不重跑任何 bootstrap / 语言模型 / PCA / Ridge**。

安全：纯读文件 + 纯 Python 判读，无重计算，本地或服务器均可运行。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config                      # noqa: E402
from src.stats.cross_subject import direction_consistency      # noqa: E402
from src.stats.estimands import CONFIRMATORY                    # noqa: E402

# RQ1 H-specific（探索性）：每 H 下 rwkv/mamba − pythia 的 IFG 主层 r 差值
RQ1_HSPECIFIC = [f"{a}_minus_pythia_r{H}"
                 for a in ("rwkv", "mamba") for H in (8, 32, 128)]


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def load_subject_estimands(m5_dir: Path, subject: str) -> dict:
    """读取单名被试 M5 结果的 estimands 表（name -> {point, ci_lo, ci_hi, ...}）。"""
    p = m5_dir / subject / "m5_results.json"
    if not p.exists():
        raise SystemExit(
            f"未找到 {subject} 的 M5 结果：{p}\n"
            f"  → 先对该被试跑：python scripts/m5_analysis.py --subject {subject}")
    with open(p) as f:
        man = json.load(f)
    est = man.get("estimands")
    if not est:
        raise SystemExit(f"{p} 缺少 estimands 字段，M5 结果不完整")
    return est


def _gather(name: str, subj_est: dict[str, dict]) -> dict[str, dict]:
    """把某个估计量在各被试上的 point/CI 收集成 direction_consistency 的输入。

    缺失该估计量的被试记为 NaN（判读时归入 insufficient_data，不静默跳过）。
    """
    per = {}
    for s, est in subj_est.items():
        e = est.get(name)
        if e is None:
            per[s] = {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
        else:
            per[s] = {"point": e.get("point", float("nan")),
                      "ci_lo": e.get("ci_lo", float("nan")),
                      "ci_hi": e.get("ci_hi", float("nan"))}
    return per


def build_group(names: list[str], subj_est: dict[str, dict],
                subjects: list[str]) -> dict[str, dict]:
    return {name: direction_consistency(_gather(name, subj_est), subject_order=subjects)
            for name in names}


def _fmt_ci(e: dict) -> str:
    pt, lo, hi = e["point"], e["ci_lo"], e["ci_hi"]
    star = "★" if e["ci_excludes_zero"] else " "
    return f"{pt:+.4f}[{lo:+.4f},{hi:+.4f}]{star}"


def _print_group(title: str, group: dict[str, dict], subjects: list[str]) -> None:
    print(f"\n[m5x] === {title} ===", flush=True)
    print(f"[m5x] {'contrast':<42} " + " ".join(f"{s:<24}" for s in subjects)
          + " 判读", flush=True)
    for name, res in group.items():
        cells = " ".join(f"{_fmt_ci(res['per_subject'][s]):<24}" for s in subjects)
        print(f"[m5x] {name:<42} {cells} {res['consistency_label_zh']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["UTS01", "UTS02", "UTS03"],
                    help="参与跨被试综合的被试；逐被试各自已跑过 m5_analysis.py")
    ap.add_argument("--m5-name", default="m5_stats",
                    help="逐被试 M5 结果目录名（对应 m5_analysis.py 的 --out-name）")
    ap.add_argument("--out-name", default="m5_cross_subject")
    args = ap.parse_args()

    # 重复被试会让 subj_est 去重成更少的键，而 subject_order 仍含重复 → n_subjects
    # 与 CI 计数被重复计入，静默给出错误的一致性判读。直接挡掉，不容忍。
    if len(set(args.subjects)) != len(args.subjects):
        raise SystemExit(f"--subjects 存在重复：{args.subjects}；跨被试综合要求各被试唯一")

    cfg = load_config()
    paths = cfg["paths"]
    m5_dir = Path(paths["results_dir"]) / args.m5_name

    subj_est = {s: load_subject_estimands(m5_dir, s) for s in args.subjects}
    print(f"[m5x] 已读 {len(args.subjects)} 名被试逐被试 M5 结果："
          f"{', '.join(args.subjects)}", flush=True)

    confirmatory = build_group(list(CONFIRMATORY), subj_est, args.subjects)
    rq1 = build_group(RQ1_HSPECIFIC, subj_est, args.subjects)

    _print_group("确认性架构差值方向一致性（IFG 主层 Δr_total，★=该被试CI排除0）",
                 confirmatory, args.subjects)
    _print_group("RQ1 H-specific 方向一致性（探索性，描述性）", rq1, args.subjects)

    # 汇总：确认性两项各自的一致性判读（不合并、不打组水平结论）
    summary = {name: res["consistency"] for name, res in confirmatory.items()}

    manifest = {
        "phase": "M5 cross-subject direction-consistency synthesis (descriptive, no pooling)",
        "git_commit": git_commit_hash(),
        "subjects": args.subjects,
        "m5_sources": {s: str(m5_dir / s / "m5_results.json") for s in args.subjects},
        "pooling": "none",
        "group_level_ci": None,
        "note": ("跨被试仅方向一致性描述性判读；不池化、不生成组水平CI、不合并成"
                 "单一 Holm 家族（里程碑5明确不做1-2）。确认性结论仍是逐被试各自的。"),
        "confirmatory_architecture_consistency": confirmatory,
        "rq1_hspecific_consistency": rq1,
        "confirmatory_summary": summary,
    }

    out_dir = Path(paths["results_dir"]) / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "m5_cross_subject.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n[m5x] === 确认性两项跨被试一致性小结 ===", flush=True)
    for name, res in confirmatory.items():
        pts = ", ".join(f"{res['per_subject'][s]['point']:+.4f}" for s in args.subjects)
        print(f"[m5x] {name}: {res['consistency_label_zh']}"
              f"（各被试点估计 {pts}；量级范围 "
              f"[{res['point_min']:+.4f},{res['point_max']:+.4f}]，描述性非组水平CI）",
              flush=True)
    print(f"\n[m5x] 结果 → {out_path}", flush=True)


if __name__ == "__main__":
    main()
