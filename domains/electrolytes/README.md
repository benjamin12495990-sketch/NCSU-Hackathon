# Solid lithium electrolytes

**Status: exploratory grouped-CV baseline only. Not in the product, not a design tool.**
OBELiX provides experimental room-temperature lithium-ion conductivity for 599 materials. The raw table, the
source's fixed train/test split and the 321-file randomized CIF archive are described (with SHA-256) in
`work/data/README.md`. The formula-only baseline is `work/baseline.py`; its report is
`work/reproductions/obelix_group_cv.json`.

## What the result says

- Data used: the source-provided **training partition only** (`train_idx.csv`, 478 rows).
- Target: log10 of ionic conductivity in S/cm.
- Features: element fractions parsed from the `Composition` string (55 elements seen in the training rows).
  DOI, space group, lattice parameters and CIFs are **not** used.
- Groups: rows that share a composition string or any DOI token are joined into one connected component:
  **124 groups** (the largest holds 106 rows; 78 are single rows). This matches the 124 groups recorded in the
  documentation snapshot.
- Folds: 5-fold `GroupKFold` (no shuffle), so no composition or paper is split across folds.
- Models: mean; Ridge at alpha 100, 10, 1, 0.1, 0.01 (standardised features); ExtraTrees (500 trees, leaf 1,
  all features) and ExtraTrees (500 trees, leaf 3, sqrt features). Selection by pooled out-of-fold MAE; any model
  within 0.01 log units of the best goes to the simpler one.

| Model | Pooled OOF MAE (log10 S/cm) | R² |
| --- | --- | --- |
| mean | 2.55 | -0.05 |
| best Ridge (alpha 100) | 2.72 | -0.42 |
| ExtraTrees leaf 3, sqrt | 2.45 | 0.15 |
| **ExtraTrees leaf 1 (selected)** | **1.52** | 0.57 |

A 1.52-log-unit MAE is a typical multiplicative error of about 33x. Ridge is worse than the mean: a linear model
on element fractions extrapolates badly to element combinations it has not seen.

**Comparison with the documentation snapshot.** The snapshot (whose code is not available here) reported the
same 124 groups, 2.55 for the mean and 1.77 for its selected ExtraTrees. This run reproduces the mean exactly and
the direction of the result (ExtraTrees well ahead of the mean, Ridge not useful), but its ExtraTrees MAE is lower.
The snapshot's tree settings and fold assignment are unknown. Re-running the same models on five reshuffled group
fold assignments (reported in the JSON, **not** used for selection) gives 1.59 to 1.71 for the selected ExtraTrees
and 2.58 to 2.96 for the mean, so the snapshot's 1.77 sits at the edge of the fold-to-fold spread. Treat the
honest summary as "about 1.5 to 1.8 log units, against about 2.6 for the mean".

**Error by family** (out-of-fold, selected model; families come from a composition-pattern heuristic in
`baseline.py`, not crystallography): garnet-like 0.38 (71 rows, 16 groups), other sulfides 1.05 (67, 21),
NASICON-like 1.17 (121, 10), perovskite-like 1.21 (23, 6), argyrodite-like 1.47 (50, 3), other oxides 2.72
(120, 55), halides 2.62 (15, 12). The model is far better on families with many well-measured fast conductors and
poor on the diverse, mostly slow "other oxide" group. Families with only 3 to 10 groups give unstable estimates.

**CIF archive.** All 321 files in `all_cifs.zip` parse with a minimal reader (six cell parameters and at least
one fractional site), and all 321 IDs are in the table. For 245 of them the CIF's `a` length matches the table
within 0.001 A; the rest are presumably different cell settings or the randomization. They are not used yet.

**The 121-row test partition was not scored.** Its labels are in the public table, so it is not a blind test.

## Limits, in plain language

- This predicts room-temperature ionic conductivity only. It says nothing about full-cell capacity, cycle life,
  voltage window, safety, cost or whether the material can be made.
- Conductivity depends on phase, defects, synthesis, density, grain boundaries and the measurement method. A
  formula alone cannot capture these, so two samples with the same formula can differ by orders of magnitude.
- An error of ~1.5 log units (~30x) is too large to rank candidate electrolytes. Use it only as a baseline that a
  better model must beat on the same folds.
- 478 rows in 124 groups is small; one large group of 106 rows dominates a fold.

## Reproduce

From the repository root (CPU only, ~30 s):

```
pip install -r domains/electrolytes/work/requirements.txt   # if needed
python3 domains/electrolytes/work/baseline.py
```

The script reads only `work/data/`, never scores `test_idx.csv` rows, and writes a new JSON file each run
(`obelix_group_cv.json` first, then time-stamped names); it never overwrites an earlier report. The JSON
records data SHA-256s, the command, library versions, fold sizes, per-model and per-fold metrics, the selected
model and the elapsed time.

## Next technical step

Audit the randomized CIFs against their IDs, then compare a structure-aware model with this formula-only result
on the same group folds. A positive result still needs newer, paper-held-out measurements before design use.
