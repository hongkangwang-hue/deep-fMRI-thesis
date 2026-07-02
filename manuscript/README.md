# Reproducibility README

*M6 deliverable 12: from which config/tag/commands each result and figure can
be rebuilt. Written honestly — includes known gaps, not just what's done.*

## Repository and remotes

- Working repo: `mine` = `git@github.com:hongkangwang-hue/deep-fMRI-thesis.git`
  (all project commits/tags live here; local ↔ AutoDL server sync goes
  through this remote only — there is no direct link between them).
- `origin` = `https://github.com/HuthLab/deep-fMRI-dataset.git`, **read-only**,
  used only to diff against the original LeBel reference implementation
  (never pushed to).
- Manuscript draft written at commit `fdf0c1c` (after M6 figures/tables code
  landed and was validated on the server).

## Data provenance

- LeBel et al. naturalistic-listening fMRI dataset, OpenNeuro `ds003020`,
  version v3.1.1, commit `cb7c536d471d48ff95a90f0531d5e2bdb910a77a` (recorded
  in `frozen/voxel_mask_UTS03.json` / `frozen/roi_columns_UTS03.json`
  provenance fields).
- Subject: UTS03 only (preregistered deviation from 3 subjects — see
  `results.md`).

## Frozen configuration and tags

| Tag | Commit | What it froze |
|---|---|---|
| `m0-freeze` | `7665abb` | Initial `frozen/` artifacts: word index, fold split, ROI spec, analysis spec, contrast registry |
| `m0-freeze-v2` | `b54eb5d` | Word-filter fix to match canonical `make_word_ds` (index bug) |
| `m3a-lambda-refreeze` | `10204f1` | λ grid re-frozen to `logspace(-2,7,19)` after the M3a blind parameter check — done **before** any held-out r was viewed |

`1.0.0`/`1.0.1`/`1.0.2` are **upstream HuthLab tags** from the `origin`
remote (2023), unrelated to this project's own freeze points — do not cite
them as this project's provenance.

Governing spec files (all in `frozen/`, all git-tracked):
`analysis_spec.yaml` (PCA/Ridge/FIR/scoring rules), `contrast_registry.yaml`
(confirmatory vs. exploratory contrast list), `fold_split.json` (CV split),
`roi_spec.json` + `roi_columns_UTS03.{npz,json}` (ROI definition/columns),
`voxel_mask_UTS03.{npy,json}` (BOLD-only voxel inclusion), `word_index.parquet`
(unified word/TR index).

## Commands to rebuild each stage

Run from the AutoDL server (`~/autodl-tmp/deep-fMRI-dataset`) after
`git pull mine master`. `HF_HUB_OFFLINE=1` is required for Pythia/RWKV once
their weights are cached.

```bash
# M1b — feature extraction (12 model×H combinations, ~hours; only needed once per container)
python3 scripts/m1_extract_features.py --models pythia mamba rwkv awd_lstm \
    --from-fold-split --skip-existing

# M4 — full matrix (one process per model; each ~35-40 min)
python3 scripts/m4_pythia.py   --skip-existing
python3 scripts/m4_mamba.py    --skip-existing
python3 scripts/m4_rwkv.py     --skip-existing
python3 scripts/m4_awd_lstm.py --skip-existing
python3 scripts/m4_aggregate.py                    # rebuild manifest only, no recompute

# M5 — preregistered statistics (~1-2 min, CPU only)
python3 scripts/m5_analysis.py

# M6 — figures and tables (read-only over M5 results)
python3 scripts/m6_figures.py
python3 scripts/m6_tables.py
# M6 supplementary — ROI location flatmap (reads frozen ROI columns + pycortex-db;
# needs the pycortex-db flatmask/flatverts/surface-info annex caches fetched)
python3 scripts/m6_roi_location.py
```

Every script accepts `--subject`/`--out-name` to point at a different results
directory; defaults match what produced the numbers in `results.md`.

## Where each deliverable's numbers come from

| Deliverable | Source file |
|---|---|
| Figures 1–5 | `figures/UTS03/fig{1..5}_*.{png,pdf}` ← `scripts/m6_figures.py` ← `results/m5_stats/UTS03/m5_results.json` |
| Figure 6 (ROI locations, supplementary) | `figures/UTS03/fig6_roi_location.{png,pdf}` ← `scripts/m6_roi_location.py` ← `frozen/roi_columns_UTS03.npz` + pycortex-db |
| Table 1 (checkpoint audit) | `figures/UTS03/tables/table1_checkpoint_audit.{csv,md}` ← M4 cell meta + `config/config.yaml` |
| Table 2 (full numbers) | `figures/UTS03/tables/table2_full_numbers.{csv,md}` ← `m5_results.json::estimands` |
| QC table | `figures/UTS03/tables/qc_table.{csv,md}` ← `frozen/` + M4 cells |
| Confirmatory / Holm results | `m5_results.json::confirmatory` |
| Layer-flip verdicts | `m5_results.json::layer_flip` |
| Shifted diagnostic | `m5_results.json::shifted_diagnostic` |
| Known statistical gaps | `m5_results.json::known_gaps` |

## Storage caveats

- `cache/features/` (model feature cache) and `results/` (all M2–M5 outputs,
  including `m5_results.json`) are **gitignored** and exist only on the
  AutoDL server's local disk. If that container is destroyed, these must be
  regenerated from scratch (feature extraction alone is on the order of an
  hour+ across all models/H; M4 full matrix ≈2.5 h; M5 ≈1-2 min).
- `figures/` is **not** gitignored, but is only ever generated on the server
  (which never pushes to git) — figures/tables must be downloaded manually
  via the AutoDL file browser and added to this repo (or the manuscript
  build) by hand.

## Environment

`environment/requirements-lock.txt` (263 lines, captured 2026-07-02 via
`pip freeze` on the AutoDL GPU server, committed) is the actual environment
used to produce every M4/M5 result in this draft — key versions: `torch
2.3.0+cu121`, `himalaya 0.4.11`, `transformers 4.46.3`, `fastai 2.8.7`,
`numpy 2.5.0`, `pandas 3.0.3`, `scikit-learn 1.9.0`. The root-level
`requirements.txt` is a stale 4-line stub inherited from the upstream
template (`datalad`/`pathlib`/`h5py`/`tables`) — **do not use it**; it does
not reflect this project's actual environment.

## Known reproducibility gaps (honest, not yet closed)

- **No single frozen-config hash.** `config/config.yaml` declares
  `hash_algorithm: sha256` but no script actually computes one combined hash
  over the `frozen/` artifacts. In practice, provenance is tracked via the
  git commit/tag table above plus the individual `frozen/*.json` files'
  own `provenance` fields (dataset commit, subject, etc.) — this is a P2
  simplification per the milestone's tiering, not a blocking gap, but should
  be named explicitly rather than silently assumed done.
