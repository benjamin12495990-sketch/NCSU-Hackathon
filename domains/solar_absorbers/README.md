# Solar absorbers

> **Status: plan only (2026-11-27). No model trained, nothing on the site.** A DFT-calculated family (crystals
> already on disk) and a measured band-gap family, never pooled; perovskite cell efficiency comes later, if at all.
> Facts come from the category research of 2026-11-27 (materials), the data research on crystals, and their
> independent checks ("the verifier"); where a verifier corrected the research, this page uses the correction.
> "Counted here" means counted for this plan from `gpu/data_cache/jarvis/jdft_3d-9-24-2025.json.zip`.

## 1. The expensive outcome
- **Predicted:** (a) SLME, the spectroscopic limited maximum efficiency, and the TBmBJ band gap of a crystal, both
  DFT-calculated. SLME is a calculated efficiency limit, not a device result. (b) The measured band gap of a
  composition. (c) Later: the measured power-conversion efficiency (PCE) of perovskite cells.
- **Why it is slow:** fabricating and testing full cells (current-voltage curves under AM1.5 light) and stability
  tests over hundreds of hours. General background in the category research, not from a cited source.

## 2. Dataset
No `../../research/references/solar_absorbers.md` exists yet; create it before adding any source.

| Source | Measured or calculated | Size | Licence as found | Source ID per row | Still unchecked |
| --- | --- | --- | --- | --- | --- |
| JARVIS-DFT `dft_3d` (9-24-2025), cached | DFT-calculated | `slme` on 10,022 crystals in 5,634 chemical systems, values 0 to 33.99; `mbj_bandgap` (TBmBJ) 21,559; both on 9,377; `optb88vdw_bandgap` 93,902; `hse_gap` 56 (counted here; single-field counts match the research and verifier) | CC BY 4.0, from the provenance file (figshare 10.6084/m9.figshare.6815699) | JARVIS id and `reference` | the unit of `slme` |
| JARVIS-Leaderboard measured gaps, `dft_3d_bandgap.json.zip` in github.com/usnistgov/jarvis_leaderboard; paper doi 10.1038/s41524-024-01259-w | measured | 54 gaps keyed to JARVIS ids | NIST terms in its LICENSE.rst: US government work, free to use, copy and redistribute | JARVIS id | none |
| Zhuo, Mansouri Tehrani, Brgoch 2018, *J. Phys. Chem. Lett.* 9(7):1668-1673, doi 10.1021/acs.jpclett.8b00124; matbench_expt_gap; github.com/yzhuo33/BandGapPredictor | measured, composition only | matbench_expt_gap 4,604 (de-duplicated from matminer's expt_gap, 6,354). The GitHub file's 3,896 non-zero gaps cover 2,470 compositions; 456 have more than one value (median spread 0.18 eV, 10.3% spread more than 0.5 eV) | no licence file on GitHub; matbench not checked | composition only, no paper ID | licence |
| RealMat-BaG, arXiv 2604.25568; github.com/Shef-AIRE/bandgap-benchmark | measured, matched to Materials Project ids | 1,705 gaps (1,183 from Dong and Cole, 185 from a set called DS2, 337 from Matbench-Expt) | MIT on the repository; upstream terms may differ | Materials Project id | the polymorph is the lowest-energy one, not the measured phase |
| Borlido et al. 2019, *J. Chem. Theory Comput.* 15, 5069-5079, doi 10.1021/acs.jctc.9b00322 | measured gaps with experimental structures | 472 nonmagnetic solids | not checked | not checked | whether the table can be downloaded |
| Dong and Cole 2022, *Sci. Data* 9:193, doi 10.1038/s41597-022-01294-6; data doi 10.17863/CAM.85234 | text-mined | 100,236 records from 128,776 articles; weighted precision 84%, recall 65% | not checked | not checked | it mixes bulk, thin-film and nano samples, and optical and electronic gaps |
| Perovskite Database, Jacobsson et al., *Nature Energy* 7:107-115, doi 10.1038/s41560-021-00941-3; mirror hf.co/datasets/jablonkagroup/perovskite_db | measured devices | over 42,400 devices, up to 100 parameters each; the mirror's `raw_data` has 20.7K / 4.4K / 4.4K rows | database: not checked. Mirror: its card text says CC BY 4.0 (no licence tag) | none in the mirror; its `__index_level_0__` column may map back to the database (unconfirmed) | the `jsc` unit (A/m² fits one preview row; inferred) |

Traps:
- The DFT gaps here are underestimates. On the 54 measured gaps the OptB88vdW MAE is 1.355 eV and the TBmBJ MAE
  0.476 eV on 53 (the leaderboard's own TBmBJ figure is 0.4981 on 54; data research on crystals, reproduced by its
  verifier). [Likely] TBmBJ is the better label to train on; `../../research/perplexity/ROUND1_DIGEST.md` says the same.
- Matching a measured composition to a crystal picks a polymorph by energy, which can be the wrong phase. Use such
  matches for checking only, and state the matching rule.
- Nothing stops the crystal network from predicting a negative gap or SLME: its output layer is a plain linear layer
  (physics-code research). Clip at zero and count the clipped predictions.

## 3. Candidate space
- **DFT family:** first the 12,182 JARVIS crystals with a TBmBJ gap but no SLME (21,559 minus 9,377, counted here),
  then those with only an OptB88vdW gap. All are relaxed, known crystals; this is screening, not design.
- **Measured gaps:** composition changes within element sets that have measured gaps, inside the measured ranges.
- **Cell efficiency:** no candidates in the first version.

## 4. Models
- **Expected value:** the crystal graph network with new `TARGETS` entries `slme` and `mbj_bandgap`
  (`gpu/structure/train_structure.py`), against the strongest trees (DEC-004, DEC-014). Measured gaps: composition
  trees against the table network (`gpu/train_tabular.py`).
- **Cautious estimate:** conformal interval (DEC-017); its lower edge for SLME.
- **Held out by entity:** whole chemical systems and whole elements for the DFT family (DEC-014, DEC-016); for
  measured gaps, every report of one composition on one side, then whole chemical systems.
- **Metrics:** MAE in eV (and in SLME's own unit), with the paired-bootstrap interval over whole groups (DEC-012).
  The DFT-trained models are also scored on the 54 measured gaps and on Zhuo matches, reported as a check against
  measurement, never as training data. All 54 crystals are in the cached file (data research on crystals), so for
  that check the 54, the matched crystals and their chemical systems are kept out of training and validation.
- **Prior evidence:** on the OptB88vdW band gap with whole chemical systems held out, the network average was clearly
  better than the strongest trees, -0.116 [-0.134, -0.0951] eV, and each of 2 runs was clearly better (DFT-calculated
  labels; exploratory, because these test systems were scored before under the original neighbour rule;
  `docs/results/20261122T014943Z_overnight_REPORT_20261126T2001.md`). 70% of those test crystals had a gap of
  exactly 0.

## 5. Cost objective
No element price, abundance or scarcity number is sourced in this repository. The first version ranks by the
cautious SLME. A filter that removes chosen elements needs no number; an abundance term waits for a cited table.

## 6. Neighbour check
- **DFT family:** the 3 nearest training crystals (L1 on element fractions, same space group where possible). The
  page says "calculated neighbours agree", never "neighbours passed".
- **Measured gaps:** the 3 nearest measured compositions; "neighbours passed" only when at least 2 reached the target.

## 7. First step, and what would stop it
- **First step:** add `slme` and `mbj_bandgap` to `TARGETS`, run the held-out-systems split against the strongest
  trees, then score the TBmBJ model on the 54 measured gaps.
- **Stop the SLME part** if neither model is clearly better than guessing the average on held-out systems.
- **Stop the cell-efficiency part** unless a paper ID per row can be recovered: without it no grouped split exists.

## Conclusions
- [Certain] 10,022 SLME values and 21,559 TBmBJ gaps (DFT-calculated) are on disk, with 54 measured gaps under NIST
  terms to check against.
- [Likely] A model trained on TBmBJ gaps will sit closer to measurement than one trained on OptB88vdW gaps, because
  the labels do (0.476 against 1.355 eV MAE on the measured set).
- [Speculative] A high calculated SLME picks materials that make efficient cells. Cell efficiency depends on
  processing that no input here describes.

---
**Ground rules:** predictions labelled as predictions; every non-learned number cited; the tool
narrows the experiment list, it does not replace the experiment; chemistry & materials only, never
biology. See `../../docs/AGENTS.md`.
