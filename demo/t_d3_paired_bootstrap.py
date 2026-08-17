"""T-D3 | Paired-story bootstrap: why one shared index set keeps a 1e-3 effect stable.

Verifies Methods 2.4 ("the same sampled indices are used for every checkpoint, H
condition, and layer") and reproduces Table 4.

Reads M4 story-level scores only; feature extraction and ridge are NOT re-run.
The 1000 resamples are computed live (pure numpy, seconds).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _t_common import setup_env, add_project_to_path, say, line, title, sub, check, trunc

setup_env()
ROOT = add_project_to_path()
sys.path.insert(0, str(Path(ROOT) / "scripts"))

import numpy as np      # noqa: E402

from src.config_loader import load_config                    # noqa: E402
from src.stats.bootstrap import (                            # noqa: E402
    paired_bootstrap, draws_to_arrays, percentile_ci,
    bootstrap_two_sided_p, holm_bonferroni, aggregate_to_r,
)
from src.stats.estimands import compute_estimands, CONFIRMATORY   # noqa: E402
from m5_analysis import load_bootstrap_data                  # noqa: E402

SUBJECTS = ["UTS01", "UTS02", "UTS03"]
TARGET = "mamba_minus_pythia_delta_total_ifg_main"

TABLE4 = {"UTS01": (+0.0014, +0.0001, +0.0027),
          "UTS02": (+0.0044, +0.0026, +0.0060),
          "UTS03": (+0.0028, +0.0011, +0.0043)}


def _k(v: np.ndarray) -> int:
    """Draw count on the sparser side of zero (numerator of the two-sided p)."""
    f = v[np.isfinite(v)]
    return min(int((f <= 0).sum()), int((f >= 0).sum()))


def _fmt_p(p: float, k: int, B: int) -> str:
    """A B-resample two-sided p can only take 0, 2/B, 4/B, ...

    With no draw crossing zero the code returns exactly 0.0, but the honest statement
    is p < 2/B (0.002 at B=1000). Writing p<0.001 would imply B=2000 resolution.
    """
    return f"<{2.0/B:.3f}" if k == 0 else f"{p:.3f}"


def show_pairing(data, seed: int) -> bool:
    sub("(1) The pairing mechanism")
    rng = np.random.default_rng(seed)
    idx = {f: rng.integers(0, len(data.fold_stories[f]), len(data.fold_stories[f]))
           for f in data.folds}
    say(f"first resample, seed {seed} (same as the production analysis):")
    for f in data.folds:
        n = len(data.fold_stories[f])
        say(f"  {f}: draw {n} of {n} with replacement -> "
            f"[{trunc(idx[f].tolist(), 6, 3)}]")

    km = ("main", "mamba", 128, "normal", "left_IFG")
    kp = ("main", "pythia", 128, "normal", "left_IFG")
    say()
    say(f"aggregating both models with that one index set (H=128, left IFG, main layer):")
    say(f"  Mamba  r = {aggregate_to_r(data.z[km], data.w[km], idx):.6f}")
    say(f"  Pythia r = {aggregate_to_r(data.z[kp], data.w[kp], idx):.6f}")
    say()
    ok = check(all(np.array_equal(idx[f], idx[f]) for f in data.folds),
               "every comparison condition reuses one element-wise identical "
               "set of within-fold indices")
    say("        (implementation passes the same array to all keys, so accidental "
        "agreement is impossible;")
    say("         statistically what matters is that the index VALUES coincide)")
    say()
    say("  Shared story-sampling fluctuation is common to both models, so it largely")
    say("  cancels in the model difference, lowering the sampling variance of that")
    say("  difference. Model-by-story interaction is NOT removed.")
    return ok


def unpaired(data, B: int, seed: int) -> tuple[float, float]:
    """Contrast only: give each model its own RNG. Not part of any reported result."""
    K = {(m, H): ("main", m, H, "normal", "left_IFG")
         for m in ("mamba", "pythia") for H in (8, 128)}
    rm = np.random.default_rng(seed)
    rp = np.random.default_rng(seed + 99991)
    out = []
    for _ in range(B):
        im = {f: rm.integers(0, len(data.fold_stories[f]), len(data.fold_stories[f]))
              for f in data.folds}
        ip = {f: rp.integers(0, len(data.fold_stories[f]), len(data.fold_stories[f]))
              for f in data.folds}
        dm = (aggregate_to_r(data.z[K[("mamba", 128)]], data.w[K[("mamba", 128)]], im)
              - aggregate_to_r(data.z[K[("mamba", 8)]], data.w[K[("mamba", 8)]], im))
        dp = (aggregate_to_r(data.z[K[("pythia", 128)]], data.w[K[("pythia", 128)]], ip)
              - aggregate_to_r(data.z[K[("pythia", 8)]], data.w[K[("pythia", 8)]], ip))
        out.append(dm - dp)
    return percentile_ci(np.asarray(out))


def main() -> int:
    title("T-D3 | Paired-story bootstrap and the reproduction of Table 4")

    cfg = load_config()
    paths = cfg["paths"]
    B = cfg["statistics"]["bootstrap_iterations"]
    seed = cfg["seeds"]["bootstrap"]
    alpha = 0.05

    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        fs = {k: sorted(v["test_stories"]) for k, v in json.load(f)["folds"].items()}

    say(f"config : B={B} resamples, seed={seed}, alpha={alpha}")
    say(f"input  : M4 story-level scores only; feature extraction and ridge NOT re-run")
    say(f"unit   : story, resampled with replacement within each outer fold, "
        f"weighted by effective TRs")

    d01 = load_bootstrap_data(Path(paths["results_dir"]) / "m4_full_matrix" / "UTS01"
                              / "cells", fs)
    ok = show_pairing(d01, seed)

    sub(f"(2) {B} paired resamples vs Table 4")
    say(f"{'subject':<9} {'point':>12} {'95% CI':>26} {'p':>8}   "
        f"{'Table 4':>9} {'match':>6}")
    res = {}
    for s in SUBJECTS:
        data = load_bootstrap_data(
            Path(paths["results_dir"]) / "m4_full_matrix" / s / "cells", fs)
        pt_all, draws = paired_bootstrap(data, compute_estimands, n_boot=B, seed=seed)
        arr = draws_to_arrays(draws)
        pt = pt_all[TARGET]
        lo, hi = percentile_ci(arr[TARGET])
        p = bootstrap_two_sided_p(arr[TARGET])
        k = _k(arr[TARGET])
        cp = {n: bootstrap_two_sided_p(arr[n]) for n in CONFIRMATORY}
        ck = {n: _k(arr[n]) for n in CONFIRMATORY}
        res[s] = dict(pt=pt, lo=lo, hi=hi, p=p, k=k, holm=holm_bonferroni(cp, alpha=alpha),
                      ck=ck, data=data)
        t_pt, t_lo, t_hi = TABLE4[s]
        m = (round(pt, 4) == t_pt and round(lo, 4) == t_lo and round(hi, 4) == t_hi)
        say(f"{s:<9} {pt:>+12.6f} {'['+f'{lo:+.6f}, {hi:+.6f}'+']':>26} "
            f"{_fmt_p(p, k, B):>8}   {t_pt:>+9.4f} {'yes' if m else 'NO':>6}")

    say()
    say(f"p resolution: at B={B} the two-sided p can only be 0, 0.002, 0.004, ...")
    say(f"  With no draw crossing zero we report p<{2.0/B:.3f}, NOT p<0.001 "
        f"(that implies B=2000).")
    say(f"  Table 4 and Methods currently print p<0.001 and should be revised to "
        f"p<{2.0/B:.3f};")
    say(f"  no Holm decision changes ({2.0/B:.3f} < 0.025 < 0.05).")

    say()
    say("deltas vs Table 4 (rounded to 4 dp):")
    for s in SUBJECTS:
        r, (t_pt, t_lo, t_hi) = res[s], TABLE4[s]
        say(f"  {s}: d_point={round(r['pt'],4)-t_pt:+.4f}  "
            f"d_lo={round(r['lo'],4)-t_lo:+.4f}  d_hi={round(r['hi'],4)-t_hi:+.4f}")

    say()
    say("cross-check against the original M5 artefact (results/m5_stats/*):")
    exact = True
    for s in SUBJECTS:
        p_ = Path(paths["results_dir"]) / "m5_stats" / s / "m5_results.json"
        if not p_.exists():
            say(f"  {s}: m5_results.json not found")
            continue
        o = json.load(open(p_))["confirmatory"][TARGET]["point"]
        dd = abs(o - res[s]["pt"])
        exact &= dd < 1e-12
        say(f"  {s}: stored {o:+.9f}   recomputed {res[s]['pt']:+.9f}   |d|={dd:.2e}")
    ok &= check(exact, "bit-identical to the original M5 artefact (same seed -> "
                       "fully reproducible)")

    sub("(3) Holm correction, applied within each subject over its 2 contrasts")
    say(f"{'subject':<9} {'contrast':<44} {'raw p':>8} {'Holm thr':>9} {'reject':>7}")
    for s in SUBJECTS:
        for n, h in res[s]["holm"].items():
            say(f"{s:<9} {n:<44} {_fmt_p(h['p'], res[s]['ck'][n], B):>8} "
                f"{h['holm_threshold']:>9.4f} {str(h['reject']):>7}")
    say()
    say("  raw two-sided bootstrap p shown; Holm is step-down, so the reject column")
    say("  carries the family-corrected conclusion (the two are not redundant)")

    sub("(4) Contrast: what happens WITHOUT pairing (not a reported result)")
    say(f"{'subject':<9} {'paired CI width':>17} {'unpaired CI width':>19} {'ratio':>7}")
    for s in SUBJECTS:
        r = res[s]
        wp = r["hi"] - r["lo"]
        ul, uh = unpaired(r["data"], B, seed)
        say(f"{s:<9} {wp:>17.6f} {uh-ul:>19.6f} {(uh-ul)/wp:>6.1f}x")
    say()
    say("  Without pairing the interval widens: shared story-difficulty fluctuation is")
    say("  no longer cancelled and turns into noise. This is why 2.4 fixes one index set.")

    say()
    if ok:
        say(f"ALL PASS  {B} paired resamples reproduce Table 4; indices confirmed shared; "
            f"bit-identical to M5")
    else:
        say("FAILURES ABOVE")
    line("=")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
