# Cleaning products & surfactants

> **Status: first surfactant-property baselines with paper-held-out cross-validation. No product model.**
> **These are surfactant physical properties, not cleaning efficacy.** SURFPRO measures the critical
> micelle concentration (as pCMC = -log10 CMC in mol/L) and the air-water surface tension at the CMC
> (AW_ST_CMC, mN/m) of single, pure surfactants. It contains no detergent formulations, soils, substrates,
> water hardness or washing tests, so nothing here says how well anything cleans.

Propose detergent / cleaner formulations that clean well at the lowest cost and environmental load (the
long-term goal; see the original plan below).

## Data and preparation

`work/data/S129_SURFPRO.csv` (Zenodo 15861809, CC BY 4.0; checksum and attribution in `work/data/README.md`)
has 1,624 surfactants, one row each (no duplicate InChIKeys). `work/prepare_surfpro.py` writes
`work/cleaned/surfpro_clean.csv` with SMILES, surfactant type, CMC temperature, the two labels, their reference
tokens and group ids, plus `surfpro_clean_info.json` with counts.

**Groups (paper-held-out).** `Ref_<property>` is where SURFPRO took the value from; it is often a compilation or
review, and 49 pCMC / 22 surface-tension rows list several DOIs. `SubRef_<property>` is the original measurement
paper cited by that compilation, when given. The primary group is the *measurement reference* (SubRef if present,
otherwise the Ref DOI(s)), with rows that share any DOI joined by connected components:

| Task | Rows | Groups | Largest group | Rows grouped via Ref_CMC fallback |
| --- | --- | --- | --- | --- |
| pCMC | 1,395 | 115 | 150 | 0 |
| AW_ST_CMC | 972 | 190 | 70 | 43 (no surface-tension reference given) |

A stricter grouping that also joins every row of a compilation (all Ref and SubRef DOIs) is written too
(`group_union_*`), but it collapses to 89 groups with one of 649 rows for pCMC and 29 groups with one of 719 rows
for surface tension, which leaves no usable 5-fold split. It was not used for scoring.

## First result (`work/reproductions/surfpro_group_cv.json`)

Features: 217 RDKit 2D descriptors and a 512-bit Morgan count fingerprint (radius 2) from the SMILES (counterions
included; all 1,624 SMILES parse), the surfactant type one-hot (8 types) and the CMC temperature. Seven fixed
settings were compared with 5-fold `GroupKFold` over the measurement-reference groups (imputation and scaling
fitted inside each training fold). Selection: lowest pooled out-of-fold MAE, simpler model within 0.01 (pCMC) or
0.1 mN/m (surface tension).

| Model | pCMC MAE (log units) | pCMC R² | AW_ST_CMC MAE (mN/m) | AW_ST_CMC R² |
| --- | --- | --- | --- | --- |
| mean | 0.899 | -0.02 | 5.55 | -0.03 |
| best Ridge (alpha 100) | 0.829 | -0.12 | 5.39 | -0.03 |
| ExtraTrees 500, leaf 1, sqrt | 0.542 | 0.55 | **4.75 (selected)** | 0.14 |
| ExtraTrees 500, leaf 3, 30% features | **0.530 (selected)** | 0.56 | 4.77 | 0.13 |
| HistGradientBoosting (500, lr 0.05) | 0.542 | 0.57 | 4.76 | 0.16 |

- **pCMC:** the selected model cuts the mean-baseline error by 0.37 log units (paper-bootstrap 95% interval
  0.25 to 0.45). A 0.53 MAE is about a factor of 3.4 in CMC. Best on anionic (0.39) and non-ionic (0.51) types;
  the 3 gemini anionic and 17 gemini zwitterionic rows each come from one paper and are poorly predicted.
- **Surface tension at the CMC:** weak. The trees improve on the mean by only 0.8 mN/m (interval 0.4 to 1.2) and
  explain about 14% of the variance; gemini cationics (416 rows) are no better than the mean. One fold is much
  worse than the others (7.1 vs about 4 mN/m), so the estimate depends on which papers are held out.
- The three tree models are within 0.012 of each other on pCMC and 0.02 mN/m on surface tension: none is
  meaningfully better. The selected score was chosen on the same folds it is reported on, so it is slightly
  optimistic. There is no separate test set.

## Limits, in plain language

- **Not cleaning performance.** CMC and surface tension are physical-chemistry handles that relate to detergency,
  but cleaning also depends on the formulation, builders, soil, surface, temperature, water hardness and the test
  method. None of that is in this data.
- Paper-held-out is not structure-held-out: homologues from the same series can appear in different papers. A
  surfactant-family or scaffold split is the next, harder check before claiming transfer to new chemistries.
- Measurement temperature is mostly 25 C; the model has almost no information about temperature effects.
- The labels come from many labs and methods (tensiometry, conductivity, fluorescence); method is not recorded.
- Out-of-fold error is an estimate for papers like these, not for new surfactant classes.

## Reproduce

From the repository root (RDKit, scikit-learn, pandas, numpy):

```
python3 -I domains/cleaning/work/prepare_surfpro.py   # cleaned table; never replaces a different existing file
python3 -I domains/cleaning/work/train_surfpro.py     # about 1-7 min on 4 shared CPUs
```

Each run of `train_surfpro.py` writes a new report (data SHA-256, command, versions, folds, per-model and per-fold
metrics, per-type errors, the selected model and timing) and never overwrites an earlier one.

## Original plan

## 1. The expensive outcome
Formulating and bench-testing a cleaner (soil-removal assays, foam, stability) takes days.
Predictable targets:
- **Critical micelle concentration (CMC)** and **surface tension** — the physical-chemistry
  handles on cleaning power, well studied for ML.
- **Cleaning efficacy** score where a dataset provides one.

## 2. Dataset (confirm current access via the research MCPs)
- Published **CMC / surface-tension datasets** for surfactants (find via OpenAlex; several ML-CMC
  papers release their tables).
- Surfactant property compilations (cite the source).
Features: surfactant structure (head/tail descriptors), blend ratios, builder & solvent content,
temperature, water hardness.

## 3. Candidate space
Blend known surfactants at new ratios, add/scale builders and solvents, stay within tested ranges.

## 4. Models
Expected CMC / efficacy + calibrated cautious bound. Reuse either model family. Split by surfactant
family to avoid leakage.

## 5. Cost objective
Cost per wash, **biodegradability** (physico-chemical proxies, cite them), and irritancy class —
treated as sourced reference data, **not** modelled biology. Factors cited in
`../../research/references/cleaning.md`.

## 6. Neighbour check
Flag only when nearest measured surfactant systems agree.

---
**Ground rules:** predictions as predictions; every non-learned number cited; chemistry & materials
only — biodegradability/irritancy enter only as cited reference classes, never as biological
models. See `../../docs/AGENTS.md`.
