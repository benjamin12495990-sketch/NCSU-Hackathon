# Polymer gas permeability (membranes)

> **Status: plan only (2026-11-27). Nothing downloaded, no model trained, nothing on the site.** A second polymer
> target next to the live Tg family (`../polymers/`). Facts come from the category research of 2026-11-27
> (chemistry) and its independent check ("the verifier"); where the verifier corrected the research, this page uses
> the correction.

## 1. The expensive outcome
- **Predicted:** log permeability of a polymer film to CO2, CH4, H2, N2 and O2. Selectivity follows from pairs of
  predictions.
- **Why it is slow:** synthesis, film casting, then time-lag constant-volume permeation for each gas: days to weeks
  per polymer. General background in the category research, not from a cited source.

## 2. Dataset
No `../../research/references/polymer_membranes.md` exists yet; create it before the first download.

| Source | Measured or calculated | Size | Licence as found | Source ID per row | Still unchecked |
| --- | --- | --- | --- | --- | --- |
| hf.co/datasets/liuganghuggingface/moldata-polymer_gas_permeability, released with torch-molecule (github.com/liugangcode/torch-molecule) | measured, per the research | 786 polymers, columns CH4, CO2, H2, N2, O2 | MIT tag on the mirror | not checked | the upstream source and its terms; the unit |
| Kumar, Shastry, Carbone 2024, "Data from: Machine learning-based discovery of molecular descriptors that control polymer gas permeation", Dryad doi 10.5061/dryad.5x69p8dbm | measured | "~500 polymers with experimental gas permeation data" (abstract) | not checked | the research reports a companion file listing source publications (unverified) | the research's 605 CO2 entries and 6 gases; the venue |
| Membrane Society of Australasia database | measured | about 1,500 polymers, 1950 to 2018; one study cleaned it to 836 homopolymers (arXiv 2404.10903) | not checked | not checked | access. Machine-learning-filled values for its gaps (Yuan et al. 2021, not checked) are imputed, not measured, and stay out |

Rows that trace to PoLyInfo stay out: Perplexity reports that its terms forbid reproduction or distribution without
written permission from NIMS (`../../research/perplexity/ROUND1_DIGEST.md`, item 1; a lead, not checked by the project).

Traps: [Likely] the three sources overlap. De-duplicate on the canonical repeat-unit SMILES before splitting. Values
span orders of magnitude, so model log10.

## 3. Candidate space
As in the polymer plan (`../polymers/README.md`, section 3): swap monomers or substituents of tested polymers and vary
copolymer ratios, staying inside the descriptor ranges the data cover.

## 4. Models
- **Expected value:** the polymer featuriser in `../polymers/prepare.py` (RDKit descriptors and Morgan fingerprints
  of the repeat unit), with tuned trees and the table network (`gpu/train_tabular.py`); one model per gas first.
- **Cautious estimate:** conformal interval (DEC-017).
- **Held out by entity:** whole backbone families (a scaffold label still to be defined), or whole source papers
  where the Dryad file gives them (DEC-009).
- **Metrics:** MAE in log10 permeability with the paired-bootstrap interval over groups (DEC-012).

## 5. Cost objective
Rank by the cautious selectivity for a gas pair at a minimum permeability: both are learned, so no outside number is
needed. The category research notes a known trade-off between permeability and selectivity (no source recorded).
Monomer prices: none sourced.

## 6. Neighbour check
The 3 nearest measured polymers by Tanimoto similarity on the Morgan bits; "neighbours passed" only when at least 2
of them reached the target for that gas.

## 7. First step, and what would stop it
- **First step:** download the 786-row mirror, find its upstream source and terms, and match it against the Dryad
  file on canonical SMILES.
- **Stop** if no grouping larger than one polymer can be built (no family or paper label), or if the MIT tag cannot be
  traced to the original data's terms.

## Conclusions
- [Likely] The data are small (786 polymers) but the polymer featuriser and grouped evaluation can be reused as they
  are.
- [Speculative] Selectivity from two separate predictions is accurate enough to rank membranes: the errors of the two
  gases add.

---
**Ground rules:** predictions labelled as predictions; every non-learned number cited; the tool
narrows the experiment list, it does not replace the experiment; chemistry & materials only, never
biology. See `../../docs/AGENTS.md`.
