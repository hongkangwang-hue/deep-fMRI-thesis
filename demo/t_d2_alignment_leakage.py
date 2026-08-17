"""T-D2 | Word-to-TR alignment shapes + executable no-leakage checks.

Verifies the Slide-6 claim: "No held-out story contributes to feature transformation,
hyperparameter selection or model fitting."

(a) prints the shape at every alignment step for one real story
(b) checks leakage three ways: frozen fold split, RUNTIME interception of the real
    run_fold's scaler/PCA fits, and the audit fields written during the full M4 run

Reads M1 feature cache; never re-runs a language model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _t_common import setup_env, add_project_to_path, say, line, title, sub, check, trunc

setup_env()
ROOT = add_project_to_path()

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402

from src.config_loader import load_config                              # noqa: E402
from src.models.feature_cache import load_features                     # noqa: E402
from src.ridge.assemble import _word_times, assemble_all               # noqa: E402
from src.fmri.trfile import (                                          # noqa: E402
    story_tr_times, trimmed_tr_times, TRIM_FIRST, TRIM_LAST, SIMULATE_PAD,
)
from src.fmri.alignment import word_to_tr, apply_fir                   # noqa: E402
from src.fmri.mask import common_scoring_mask                          # noqa: E402
from src.ridge.pipeline import (                                       # noqa: E402
    run_fold, numpy_ridgecv_solver, DELAYS_S, TR_SECONDS, AFTER_S, PCA_K,
)

SUBJECT = "UTS03"
STORY = "adollshouse"
MODEL = "pythia"
H = 8
MINI_VOXELS = 150


def part_a(cfg) -> None:
    sub(f"(a) Alignment chain   story={STORY}  model={MODEL}  H={H}  subject={SUBJECT}")

    paths, ds = cfg["paths"], cfg["datasets"]
    wi = pd.read_parquet(Path(paths["frozen_dir"]) / "word_index.parquet")
    with open(ds["respdict"]) as f:
        respdict = json.load(f)

    feat = load_features(paths["cache_dir"], MODEL, STORY, H)
    ids, vecs = feat["word_ids"], feat["main"].astype(np.float64)

    say(f"{'step':<34} {'shape / value':<22} note")
    say(f"{'1. word-level features (cached)':<34} {str(vecs.shape):<22} "
        f"{len(ids)} eligible targets x 768")

    t = _word_times(wi, ids)
    order = np.argsort(t)
    t, vecs = t[order], vecs[order]
    say(f"{'2. word mid-times (on+off)/2':<34} {str(t.shape):<22} "
        f"{t.min():.1f}s..{t.max():.1f}s, irregular spacing")

    n = respdict[STORY]
    tr_full = story_tr_times(n)
    say(f"{'3. TR grid (pad removed)':<34} {str(tr_full.shape):<22} "
        f"respdict={n} - pad {SIMULATE_PAD} ({SIMULATE_PAD*TR_SECONDS:.0f}s) = {len(tr_full)}")

    X_full = word_to_tr(vecs, t, tr_full)
    say(f"{'4. Lanczos resample word->TR':<34} {str(X_full.shape):<22} "
        f"irregular words -> uniform {TR_SECONDS:.0f}s grid")

    X = X_full[TRIM_FIRST: len(X_full) - TRIM_LAST]
    trt = trimmed_tr_times(n)
    say(f"{'5. trim [10:-5]':<34} {str(X.shape):<22} "
        f"drop {TRIM_FIRST} head ({TRIM_FIRST*TR_SECONDS:.0f}s), "
        f"{TRIM_LAST} tail ({TRIM_LAST*TR_SECONDS:.0f}s)")

    head = SIMULATE_PAD + TRIM_FIRST
    say(f"{'   cumulative vs raw response':<34} "
        f"{f'{n}-{SIMULATE_PAD}-{TRIM_FIRST}-{TRIM_LAST}={X.shape[0]}':<22} "
        f"head {head} TRs ({head*TR_SECONDS:.0f}s), tail {TRIM_LAST} TRs "
        f"({TRIM_LAST*TR_SECONDS:.0f}s)")
    say(f"{'   row-count consistency':<34} {f'{X.shape[0]} == {len(trt)}':<22} "
        f"hard assert inside assemble_story")

    say(f"{'6. [in-fold] scaler + PCA':<34} {f'(T, {PCA_K})':<22} "
        f"fit on training stories only -- see (b)")

    Xf, valid = apply_fir(np.zeros((X.shape[0], PCA_K)), delays_s=DELAYS_S, tr=TR_SECONDS)
    say(f"{'7. FIR delays (2/4/6/8s)':<34} {str(Xf.shape):<22} "
        f"width x{len(DELAYS_S)} = {PCA_K}x{len(DELAYS_S)}")
    say(f"{'   FIR-valid TRs':<34} {f'{int(valid.sum())} / {len(valid)}':<22} "
        f"leading frames zero-filled -> invalid")

    mask = common_scoring_mask(trt, valid, after_s=AFTER_S)
    say(f"{'8. scoring mask, story >100s':<34} "
        f"{f'{int(mask.sum())} / {len(mask)}':<22} TRs actually scored")
    say()
    say(f"  {len(ids)} word vectors -> {X.shape[0]} TR rows -> {Xf.shape[1]} design columns "
        f"-> {int(mask.sum())} scored TRs")


def part_b(cfg) -> bool:
    sub("(b) No-leakage checks, three independent lines of evidence")

    paths, ds = cfg["paths"], cfg["datasets"]
    with open(Path(paths["frozen_dir"]) / "fold_split.json") as f:
        folds = json.load(f)["folds"]
    allp = True

    say("[1] Frozen fold split      source: frozen/fold_split.json")
    for fn, fd in folds.items():
        tr_s, te_s = list(fd["train_stories"]), list(fd["test_stories"])
        say(f"    {fn}: train {len(tr_s):>2}, test {len(te_s):>2}   "
            f"first test story = {sorted(te_s)[0]}")
        allp &= check(not (set(tr_s) & set(te_s)), f"{fn}: train n test = empty")
    counts = [len(fd["test_stories"]) for fd in folds.values()]
    allp &= check(sorted(counts) == [27, 28, 28], f"test-story counts {counts} = 28/28/27")
    seen = [s for fd in folds.values() for s in fd["test_stories"]]
    allp &= check(len(seen) == len(set(seen)),
                  f"each story tested in exactly one fold ({len(set(seen))} stories)")

    say()
    say("[2] Runtime interception   monkeypatch sklearn fits, then call the real run_fold")
    f0 = folds[list(folds)[0]]
    mtr, mte = sorted(f0["train_stories"])[:3], sorted(f0["test_stories"])[:2]
    say(f"    reduced fold: train={mtr}")
    say(f"                  test ={mte}   (first {MINI_VOXELS} voxels only)")
    say(f"    real stories, real features, unmodified run_fold -- shrunk only for speed;")
    say(f"    the full 28/28/27 path is covered by [3]")

    sd = assemble_all(mtr + mte, MODEL, H, "main", SUBJECT, paths["cache_dir"],
                      ds["data_dir"], ds["respdict"],
                      str(Path(paths["frozen_dir"]) / "word_index.parquet"),
                      voxel_mask=np.arange(MINI_VOXELS))
    rows = {s: d.X.shape[0] for s, d in sd.items()}
    n_tr = sum(rows[s] for s in mtr)
    n_te = sum(rows[s] for s in mte)
    say(f"    TR rows: train={n_tr}, test={n_te}, total={n_tr+n_te}")

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    log, slog = [], []
    o_sc, o_pca = StandardScaler.fit, PCA.fit

    def sc_fit(self, X, y=None, **k):
        log.append(("StandardScaler.fit", np.asarray(X).shape))
        return o_sc(self, X, y, **k)

    def pca_fit(self, X, y=None, **k):
        log.append(("PCA.fit", np.asarray(X).shape))
        return o_pca(self, X, y, **k)

    def solver(Xtr, Ytr, Xte, grid, inner, seed):
        slog.append((Xtr.shape, Xte.shape, len(grid), inner))
        return numpy_ridgecv_solver(Xtr, Ytr, Xte, grid, inner, seed)

    StandardScaler.fit, PCA.fit = sc_fit, pca_fit
    try:
        run_fold(sd, mtr, mte, solver, roi_columns=None,
                 pca_k=min(PCA_K, n_tr - 1), verbose=False, tag="/t")
    finally:
        StandardScaler.fit, PCA.fit = o_sc, o_pca

    say(f"    intercepted {len(log)} fit call(s):")
    for what, shp in log:
        say(f"      {what:<20} input shape {shp}")
    sc = [s for w, s in log if w == "StandardScaler.fit"]
    pc = [s for w, s in log if w == "PCA.fit"]
    allp &= check(len(sc) == 1 and sc[0][0] == n_tr,
                  f"StandardScaler.fit saw {sc[0][0]} rows = train {n_tr} "
                  f"(not the {n_te} test rows)")
    allp &= check(len(pc) == 1 and pc[0][0] == n_tr,
                  f"PCA.fit saw {pc[0][0]} rows = train {n_tr}")
    allp &= check(all(s[0] != n_tr + n_te for s in sc + pc),
                  f"no fit call ever saw all {n_tr+n_te} rows")
    if slog:
        xtr, xte, nl, ni = slog[0]
        say(f"    ridge solver: Xtr={xtr}  Xte={xte}  lambdas={nl}  inner folds={ni}")
        allp &= check(xtr[0] <= n_tr, f"lambda inner-CV splits train rows only "
                                      f"({xtr[0]} <= {n_tr})")
        allp &= check(xte[0] == n_te, f"test rows used for prediction only "
                                      f"({xte[0]} == {n_te})")

    say()
    say("[3] Full-run audit fields  source: results/m4_full_matrix/*/cells/*.json")
    tot, flags, per = 0, {"leakage_audit_pass": 0, "common_mask_verified": 0,
                          "scoring_mask_bit_identical": 0}, {}
    for subj in ("UTS01", "UTS02", "UTS03"):
        ns, nb = 0, 0
        for p in sorted((Path(paths["results_dir"]) / "m4_full_matrix" / subj /
                         "cells").glob("main_*.json")):
            c = json.load(open(p))
            tot += 1
            ns += 1
            nb += 1 if c.get("scoring_mask_bit_identical") is True else 0
            for k in flags:
                flags[k] += 1 if c.get(k) is True else 0
        per[subj] = (ns, nb)
    say(f"    {tot} main-layer cells (3 subjects x 4 models x 3 H x 3 folds)")
    for k, v in flags.items():
        say(f"      {k:<30} {v} / {tot}")
    allp &= check(flags["leakage_audit_pass"] == tot, f"leakage_audit_pass true in all {tot}")
    allp &= check(flags["common_mask_verified"] == tot,
                  f"common_mask_verified true in all {tot}")

    say(f"    scoring_mask_bit_identical by subject:")
    for s, (ns, nb) in per.items():
        say(f"      {s}: {nb} / {ns}" + ("" if nb == ns else
                                         "   <- pilot run, field not yet emitted"))
    say(f"    supplementary: results/mask_identity_audit/ (independent script, 3 subjects)")
    aok, rows_a = True, 0
    for s in ("UTS01", "UTS02", "UTS03"):
        ap_ = Path(paths["results_dir"]) / "mask_identity_audit" / s / "mask_identity.json"
        if not ap_.exists():
            aok = False
            continue
        a = json.load(open(ap_))
        rows_a += 1
        say(f"      {s}: {a['step2_n_stories_checked']} stories, "
            f"bit-identical = {a['step2_all_masks_bit_identical']}")
        aok &= bool(a["step2_all_masks_bit_identical"])
    allp &= check(aok and rows_a == 3,
                  "normal/shift scoring masks bit-identical for all 3 subjects "
                  "(covers the pilot gap above)")
    return allp


def main() -> int:
    title("T-D2 | Word-to-TR alignment shapes and executable no-leakage checks")
    say("features read from cache/features (M1); no language model is re-run")
    cfg = load_config()
    part_a(cfg)
    ok = part_b(cfg)
    say()
    if ok:
        say("ALL PASS  no held-out story entered feature transformation, "
            "hyperparameter choice, or fitting")
    else:
        say("FAILURES ABOVE")
    line("=")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
