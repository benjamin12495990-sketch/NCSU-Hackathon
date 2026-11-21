# General chemistry (molecules & mixtures)

> **Status: an offline water-solubility (LogS) model, trained and independently checked; not on the site.**
> The AqSolDB CSV is now in `work/data/` (CC0, pinned upstream commit and SHA-256 in `work/data/README.md`).
> The whole pipeline was re-run on CPU on 2026-11-21 and reproduced every reported number exactly
> (`work/reproductions/20261121_cpu.json`). **That reproduction is not new evidence**: the fixed 2,000-row test
> split had already been scored in the original report, so re-scoring it only shows the code and data are
> reproducible. **A fairly tuned tree model ties the network**, so the network should not be described as
> beating standard methods.

## What the numbers say

| Model (LogS, log10 mol/L) | Test MAE, 2,000 rows (already reported) | 3-fold grouped CV on training rows |
| --- | --- | --- |
| always predict the training mean | 1.922 | |
| gradient-boosted trees, one fixed untuned setting | 0.834 | 0.824 |
| gradient-boosted trees, tuned by grouped CV (600 trees, depth 6, lr 0.05) | 0.777 | 0.784 |
| network, 5-seed ensemble (settings chosen by grouped CV) | 0.761 | 0.759 |

- Network minus tuned trees on the test split: -0.016 log units; the independent check's bootstrap interval
  (-0.035 to +0.003, `work/skeptic_check.json`) includes zero. This is a tie.
- On a Bemis-Murcko scaffold split (ring systems unseen in training) both models get about 1.00 (network 1.003,
  trees 1.008): expect that error, roughly a factor of 10 in concentration, on new chemical families.
- Eight extra grouped 80/20 resamples average 0.745 (network) and 0.819 (untuned trees); they overlap heavily and
  include main-test rows, so they are not independent confirmation.
- Groups: the first block of the InChIKey (9,671 groups over 9,982 rows); 7,982 training and 2,000 test rows.

## The reproduction

`python3 domains/chemistry/work/reproduce.py` (about 8 minutes on 4 shared CPUs) runs `prepare.py`, `nn_run.js`
(cv, final and splits modes), `trees.py`, the scaffold split and evaluation, `make_notes.py`, `evaluate.py`, and
`trees_tuned.py` (a re-run of the independent check's 8-setting tree grid, chosen by grouped CV on training rows).
Everything regenerated goes to `work/runs/<run_id>/`, which is gitignored; the committed `work/result.json` is not
touched. The compact report records the data SHA-256, command, library and Node versions, fold definitions, both
sets of metrics, the per-step timing and whether each headline number matches. Every metric matched the original
to the last reported digit; only training times differed. (`extra_cv_outside_grid_not_used` is absent from the
regenerated result because its input, `nn_cv_extra.js`, is not in this repository.)

`nn_data.json` (2 MB, the network's training rows) is **not committed**: `prepare.py` is deterministic (two runs
gave byte-identical files, SHA-256 `7426a5c0466e...`, full hash in the report) and rebuilds it in about two
seconds:

```
python3 -I domains/chemistry/work/prepare.py domains/chemistry/work/data/data_curated.csv domains/chemistry/work
```

That command writes `nn_data.json`, `all_rows.json`, `splits.json` and `prepare_info.json` into `work/`; all four
are gitignored.

## Limits

17 bulk 2D descriptors cannot see crystal packing, melting point, polymorphs, pH or temperature. Labels are noisy
(78% are single measurements). An error of 0.76 log units is about a factor of 6 in concentration: a rough
screen, not a measurement, and not able to rank close analogues. A real generalization claim needs solubility
data that was never used for any choice here.

## Original plan

Molecular and mixture properties as the surrogate target — the non-biological, physico-chemical
end of chemistry.

- **Outcome:** solubility, boiling point, viscosity, reaction yield.
- **Datasets (confirm via MCPs):** QM9 (quantum properties), ESOL / Delaney (aqueous solubility),
  the **physico-chemical** subsets of MoleculeNet. Avoid its toxicity/bio tasks — out of scope.
- **Candidate space:** perturb molecular structure within the dataset's chemical space.
- **Models:** either model family, expected + cautious.
- **Cost:** synthesis difficulty, cost — cited in `../../research/references/chemistry.md`.

**Out of scope:** anything biological (binding, toxicity, metabolism). Keep to physics-and-chemistry
properties of molecules and mixtures.
