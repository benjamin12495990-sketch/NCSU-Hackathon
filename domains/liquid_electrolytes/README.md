# Liquid battery electrolytes

> **Status: plan only (2026-11-27). Nothing downloaded, no model trained, nothing on the site.** Lithium salts in
> organic solvent blends. A new family, separate from the solid-electrolyte family (`../electrolytes/`, OBELiX); the
> two are never pooled. Facts come from the category research of 2026-11-27 (chemistry) and its independent check
> ("the verifier"); where the verifier corrected the research, this page uses the correction.

## 1. The expensive outcome
- **Predicted:** ionic conductivity (log10 scale) from the salt, the solvent fractions, the salt concentration and the
  temperature. Later, Coulombic efficiency.
- **Why it is slow:** mixing in a dry glovebox, then impedance (EIS) runs across temperature; for Coulombic
  efficiency, weeks of Li‖Cu cell cycling. General background in the category research, not from a cited source.

## 2. Dataset
No `../../research/references/liquid_electrolytes.md` exists yet; create it before the first download.

| Source | Measured or calculated | Size | Licence as found | Source ID per row | Still unchecked |
| --- | --- | --- | --- | --- | --- |
| CALiSol-23, de Blasio, Elsborg, Vegge, Flores, Bhowmik, *Sci. Data* 2024, doi 10.1038/s41597-024-03575-8; https://data.dtu.dk/articles/dataset/CALiSol-23/24559960 | measured | 13,825 points from 27 articles, 38 solvents, 14 lithium salts (abstract; Perplexity round 2 read the same on the DTU record) | CC BY 4.0 according to the Hugging Face mirror card, which cites DTU. The DTU page was not reachable for the research, and Perplexity saw no licence line in its page text | a `doi` column (in the mirror) | empty `doi` cells; the 2.74 MB size; the version (a 2025 paper, arXiv 2510.15403, lists 13,269 samples and 13 salts); the authors' reported warning that conductivity depends on the measurement method |
| Mirror: hf.co/datasets/test-authoring/calisol-electrolyte-conductivity-new-paper | the same data | 13.8K rows, 46 columns: `doi`, `k`, `T`, `c`, `salt`, `c units`, `solvent ratio type`, then one fraction column per solvent (EC, PC, DMC, EMC, DEC, DME, FEC, DOL and others) | as above | `doi` | the unit of `k` |
| Rahmanian et al. 2022, doi 10.5281/zenodo.7244939, "Dataset of 5035 Conductivity Experiments for Lithium-Ion Battery Electrolyte Formulations at Various Temperatures" | not checked | one 39.2 MB CSV, rows not counted | CC BY 4.0 on the record | not checked | whether the values are measured; overlap with CALiSol-23. Source: Perplexity round 2 (`../../research/perplexity/round2/computer_mode_answers.md`), an unverified lead |
| Bradford et al., *ACS Cent. Sci.* 2023, 9:206-216, doi 10.1021/acscentsci.2c01123; mirror hf.co/datasets/liuganghuggingface/moldata-chemixhub_polymer_electrolyte | measured; solid polymer electrolytes, a separate sub-family | 11,350 points, 1,749 mixtures (CheMixHub, arXiv 2506.12231); the mirror has 11.0K rows | MIT (mirror tag, and CheMixHub appendix A.1) | not checked | none beyond that |
| Kim et al., *PNAS* 2023, doi 10.1073/pnas.2214357120 | measured Li‖Cu Coulombic efficiency | 147 formulations, a count from papers that reuse it (not checked) | not checked | not checked | whether it can be downloaded |

Traps: concentrations come in more than one unit (the `c units` column), so convert before training. Only 27 papers
means wide intervals.

## 3. Candidate space
Change the solvent fractions (summing to 1) and the salt concentration only within the ranges seen for that salt and
solvent set, and the temperature only within the range measured for that system. Only salts and solvents present in
the data.

## 4. Models
- **Expected value:** tuned trees and the table network (`gpu/train_tabular.py`) on log10 conductivity. Inputs: salt
  as a one-hot column, solvent fractions, concentration, temperature. Later, solvent descriptors from SMILES with
  RDKit, as in `../polymers/prepare.py`, so that unseen solvents can be handled.
- **Cautious estimate:** conformal lower edge (DEC-017).
- **Held out by entity:** whole papers (the `doi` column, DEC-009); then a harder test that holds out whole solvent
  systems.
- **Metrics:** MAE in log10 units with the paired-bootstrap interval over papers (DEC-012).
- **Temperature behaviour:** nothing in the current models makes conductivity follow a temperature law (physics-code
  research). Check that each candidate's conductivity rises with temperature and report the exceptions.
- **Prior evidence, solid electrolytes:** on OBELiX grouped folds, the trees were clearly better than the network
  average, +0.587 [+0.199, +1.08] log10 S/cm, and clearly better than each of the 5 single runs
  (`docs/results/20261122T014943Z_overnight_REPORT_20261126T2001.md`). [Likely] on small grouped tables, trees are
  the opponent to beat.

## 5. Cost objective
No salt or solvent price is sourced in this repository. The first version ranks by the cautious conductivity at the
chosen temperature. A cost term waits for cited prices.

## 6. Neighbour check
The 3 nearest measured formulations with the same salt (L1 on solvent fractions, plus concentration and temperature
on a fixed scale); "neighbours passed" only when at least 2 of them reached the target.

## 7. First step, and what would stop it
- **First step:** download CALiSol-23 from DTU and read the licence on the record; count rows, papers, empty `doi`
  cells and concentration units; run paper-held-out cross-validation, trees against the mean.
- **Stop** if the DTU terms do not allow a derived table (then keep the data local), or if the trees are not clearly
  better than the mean on held-out papers.

## Conclusions
- [Certain] The mirror carries a `doi` column, so whole papers can be held out (DEC-009) once its empty cells are
  counted.
- [Likely] This is the cleanest measured fit of the new categories: measured values, a table as simple as concrete's,
  and CC BY 4.0 according to a third-party mirror card; the DTU record's own page text showed no licence line
  (Perplexity round 2), so the licence is not settled.
- [Speculative] 27 papers are enough to separate a model from the mean. The solid-electrolyte family had 124 groups
  (`../README.md`).

---
**Ground rules:** predictions labelled as predictions; every non-learned number cited; the tool
narrows the experiment list, it does not replace the experiment; chemistry & materials only, never
biology. See `../../docs/AGENTS.md`.
