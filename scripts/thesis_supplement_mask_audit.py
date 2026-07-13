"""
硕士论文核查清单 —— 第9节 mask_audit.csv 部分。

评分 mask（>100s ∩ FIR边缘有效 ∩ shift边缘有效）只依赖故事长度（respdict 里的原始
TR 数），与被试的 BOLD/体素完全无关（src/fmri 三个模块的纯函数性质）——所以
scripts/verify_scoring_mask_identity.py 可以完全在本机对三名被试各跑一次，不需要
服务器/GPU/任何模型结果。本脚本只是把它已经写好的
results/mask_identity_audit/<subject>/mask_identity.json 三份汇总成清单要求的
mask_audit.csv（每故事一行 + n_mask_difference 列）。

用法：先跑三次
  python3 scripts/verify_scoring_mask_identity.py --subject UTS01
  python3 scripts/verify_scoring_mask_identity.py --subject UTS02
  python3 scripts/verify_scoring_mask_identity.py --subject UTS03
再跑本脚本：python3 scripts/thesis_supplement_mask_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config   # noqa: E402

OUT_DIR = PROJECT_ROOT / "thesis_supplement"
SUBJECTS = ("UTS01", "UTS02", "UTS03")


def main():
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = []
    for subj in SUBJECTS:
        p = results_dir / "mask_identity_audit" / subj / "mask_identity.json"
        if not p.exists():
            print(f"  [跳过] {p} 不存在——先跑 "
                 f"`python3 scripts/verify_scoring_mask_identity.py --subject {subj}`")
            continue
        d = json.load(open(p))
        for ps in d["per_story"]:
            rows.append({
                "subject": subj,
                "fold": ps["fold"],
                "story": ps["story"],
                "n_trs": ps["n_trs"],
                "n_eff_tr": ps["n_eff"],
                "mask_bit_identical": ps["mask_bit_identical"],
                "n_mask_difference": 0 if ps["mask_bit_identical"] else None,
            })
        summary.append({
            "subject": subj,
            "n_stories_checked": d["step2_n_stories_checked"],
            "all_masks_bit_identical": d["step2_all_masks_bit_identical"],
            "feature_independence_proven": d["step1_feature_independence"]["feature_independence_proven"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "mask_audit.csv", index=False)
    print(f"已写 {OUT_DIR / 'mask_audit.csv'}（{len(df)} 行 = "
         f"{len(SUBJECTS)}被试 × 83故事）")

    n_bad = int((~df["mask_bit_identical"]).sum()) if len(df) else None
    print(f"\nn_mask_difference 非0（即 mask 不完全一致）的行数：{n_bad}")
    print("\n各被试汇总：")
    for s in summary:
        print(f"  {s['subject']}: {s['n_stories_checked']} 故事全部一致="
             f"{s['all_masks_bit_identical']}，特征无关性已证明="
             f"{s['feature_independence_proven']}")


if __name__ == "__main__":
    main()
