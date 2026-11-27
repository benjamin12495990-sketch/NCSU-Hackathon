# Dyes: absorption and emission wavelength

> **Status: plan only (2026-11-27). Nothing downloaded, no model trained, nothing on the site.** Organic chromophores
> only; no open dataset for inorganic pigments was found. Facts come from the category research of 2026-11-27
> (chemistry) and its independent check ("the verifier"); where the verifier corrected the research, this page uses
> the correction.

## 1. The expensive outcome
- **Predicted:** the absorption maximum and the emission maximum (nm) of a chromophore in a given solvent; later the
  extinction coefficient and the photoluminescence quantum yield.
- **Why it is slow:** a new chromophore needs multi-step synthesis and purification (weeks). The UV-vis measurement
  itself is cheap. General background in the category research, not from a cited source.

## 2. Dataset
No `../../research/references/dyes.md` exists yet; create it before the first download.

| Source | Measured or calculated | Size | Licence as found | Source ID per row | Still unchecked |
| --- | --- | --- | --- | --- | --- |
| Joung et al. 2020, *Sci. Data* 7:295, doi 10.1038/s41597-020-00634-8; figshare "DB for chromophore" (version 3, 2024-10-21, from a search snippet) | measured, curated by hand: two people extracted each value independently and a third cross-checked | 20,236 points from 1,358 articles: absorption and emission maxima, bandwidths, extinction coefficient, quantum yield, lifetime; 7,016 chromophores in 365 solvents or the solid state (paper abstract; the figshare listing says 7,018 and 366) | not checked (figshare returned 403) | not checked | licence; an article ID per row |
| Beard et al. 2019, *Sci. Data* 6:307, doi 10.1038/s41597-019-0306-0; Cambridge Apollo handle 1810/314763 | measured absorption maxima, text-mined (extinction coefficient where available); also sTDA/TD-DFT **calculated** values for 5,380 compounds | 18,309 records, 8,488 compounds, from 402,034 documents | not checked | not checked | licence |

Beard's calculated values go in their own column and are never used as measurements (DEC-002). Text mining makes the
Beard set noisier than the Joung set (category research).

## 3. Candidate space
- Known chromophores in solvents not yet measured for them: no synthesis, so a cheap first check.
- Substituent swaps on tested scaffolds, kept inside the descriptor ranges of the data. This needs a molecule editor;
  RDKit is already used in `../polymers/prepare.py`.

## 4. Models
- **Expected value:** Morgan fingerprints and RDKit descriptors of the chromophore and of the solvent, with tuned
  trees and the table network (`gpu/train_tabular.py`).
- **Cautious estimate:** a conformal interval (DEC-017); a candidate counts only if the whole interval lies inside the
  target wavelength window.
- **Held out by entity:** whole chromophore scaffolds, all solvents of one chromophore on one side; then whole
  articles, if the rows carry an article ID (DEC-009).
- **Metrics:** MAE in nm with the paired-bootstrap interval over scaffolds (DEC-012). On the 5,380 Beard compounds
  with a TD-DFT value, that value is a physics baseline the model has to beat.

## 5. Cost objective
No synthesis-difficulty score or reagent price is sourced in this repository. Rank by how far inside the target
window the cautious interval sits.

## 6. Neighbour check
The 3 nearest measured chromophores by Tanimoto similarity on the Morgan bits, in the same solvent; "neighbours
passed" only when at least 2 of them were measured inside the window.

## 7. First step, and what would stop it
- **First step:** get the Joung file, read its licence, and check whether each row carries an article ID; then a
  scaffold-held-out baseline, trees against the mean.
- **Stop** if the licence forbids a derived table (then keep the data local), or if the scaffold-held-out error is not
  clearly below that of the mean.

## Conclusions
- [Certain] A hand-curated measured set exists: 20,236 points from 1,358 articles.
- [Likely] Errors with whole scaffolds held out will be higher than on a random split, which puts the same
  chromophore, measured in another solvent, on both sides. LES-001 records the same kind of near-copy leak for
  concrete.
- [Speculative] Adding the TD-DFT value as an input improves held-out scaffolds. Not tested.

---
**Ground rules:** predictions labelled as predictions; every non-learned number cited; the tool
narrows the experiment list, it does not replace the experiment; chemistry & materials only, never
biology. See `../../docs/AGENTS.md`.
