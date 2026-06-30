"""
M2-C Phase 1 —— 严格复现 LeBel native eng1000 路径（输出到独立目录，不覆盖参考）。

复用 encoding/ 的同一套函数（feature_spaces / encoding_utils / ridge_utils.ridge），
设置与产出 results/eng1000/UTS03/corrs.npz 的 encoding.py 完全一致，仅：
  - 固定随机种子（保证我方可复现；历史参考的原始种子未知）；
  - 输出到 results/eng1000_native_rerun/<subject>/（保护参考产物不被覆盖）。

⚠️ 重计算：bootstrap_ridge 在 95556 体素 × eng1000(985×4 延迟) 上做 50 次 bootstrap，
属 CPU 密集，必须在服务器跑，确认资源后再启动。产出后用 scripts/m2c_compare.py
对照参考 corrs.npz（冻结指标见 frozen/m2c_reference_validation.yaml）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import join

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, join(PROJECT_ROOT, "encoding"))

from encoding_utils import apply_zscore_and_hrf, get_response          # noqa: E402
from feature_spaces import get_feature_space, _FEATURE_CONFIG          # noqa: E402
from ridge_utils.ridge import bootstrap_ridge                          # noqa: E402
from config import EM_DATA_DIR                                         # noqa: E402

# ---- native 设置（与 encoding.py 默认一致；冻结见 m2c_reference_validation.yaml）----
ALPHAS = np.logspace(1, 3, 10)
NBOOTS, CHUNKLEN, NCHUNKS = 50, 40, 125
TRIM, NDELAYS = 5, 4
SINGCUTOFF, SINGLE_ALPHA, USE_CORR = 1e-10, False, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03")
    ap.add_argument("--feature", default="eng1000", choices=list(_FEATURE_CONFIG))
    ap.add_argument("--sessions", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--out-name", default="eng1000_native_rerun",
                    help="输出到 results/<out-name>/<subject>/，避免覆盖参考")
    args = ap.parse_args()

    np.random.seed(args.seed)  # 冻结种子，保证可复现

    with open(join(EM_DATA_DIR, "sess_to_story.json")) as f:
        sess_to_story = json.load(f)
    train_stories, test_stories = [], []
    for sess in map(str, args.sessions):
        stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
        train_stories.extend(stories)
        if tstory not in test_stories:
            test_stories.append(tstory)
    assert not (set(train_stories) & set(test_stories)), "Train-Test overlap!"
    allstories = list(set(train_stories) | set(test_stories))

    save_location = join(PROJECT_ROOT, "results", args.out_name, args.subject)
    os.makedirs(save_location, exist_ok=True)
    print(f"[phase1] 输出: {save_location}  (种子={args.seed})")
    print(f"[phase1] train={len(train_stories)} 故事, test={test_stories}")

    feat = get_feature_space(args.feature, allstories)
    delRstim = apply_zscore_and_hrf(train_stories, feat, TRIM, NDELAYS)
    delPstim = apply_zscore_and_hrf(test_stories, feat, TRIM, NDELAYS)
    zRresp = get_response(train_stories, args.subject)
    zPresp = get_response(test_stories, args.subject)
    print(f"[phase1] delRstim{delRstim.shape} delPstim{delPstim.shape} "
          f"zRresp{zRresp.shape} zPresp{zPresp.shape}")

    wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
        delRstim, zRresp, delPstim, zPresp, ALPHAS, NBOOTS, CHUNKLEN, NCHUNKS,
        singcutoff=SINGCUTOFF, single_alpha=SINGLE_ALPHA, use_corr=USE_CORR)

    np.savez(join(save_location, "corrs"), corrs)
    np.savez(join(save_location, "valphas"), valphas)
    np.savez(join(save_location, "bscorrs"), bscorrs)
    np.savez(join(save_location, "valinds"), np.array(valinds))
    # 不保存 wt（特征×体素，GB 级，易 OOM；Phase 1 对照只需 corrs/valphas）
    manifest = {
        "subject": args.subject, "feature": args.feature, "seed": args.seed,
        "sessions": args.sessions, "test_stories": test_stories,
        "alphas": "logspace(1,3,10)", "nboots": NBOOTS, "chunklen": CHUNKLEN,
        "nchunks": NCHUNKS, "trim": TRIM, "ndelays": NDELAYS,
        "single_alpha": SINGLE_ALPHA, "use_corr": USE_CORR,
        "corrs_shape": list(corrs.shape),
    }
    with open(join(save_location, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[phase1] corrs{corrs.shape} mean={np.nanmean(corrs):.4f} "
          f"max={np.nanmax(corrs):.4f} → 已保存")
    print(f"[phase1] 下一步: python scripts/m2c_compare.py "
          f"--ours {save_location}/corrs.npz "
          f"--ours-valphas {save_location}/valphas.npz")


if __name__ == "__main__":
    main()
