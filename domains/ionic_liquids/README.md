# Ionic liquids

> **Status: plan only (2026-11-27). Nothing downloaded, no model trained, nothing on the site.** Facts come from the
> category research of 2026-11-27 (chemistry) and its independent check ("the verifier"); where the verifier
> corrected the research, this page uses the correction.

## 1. The expensive outcome
- **Predicted:** viscosity and ionic conductivity (log10 scale), later density and melting point, from the cation,
  the anion, the mixture composition, temperature and pressure.
- **Why it is slow:** synthesis, purification and drying (water content shifts viscosity strongly), then viscometry
  and conductivity; ionic liquids are costly per gram. General background in the category research, not from a
  cited source.

## 2. Dataset
No `../../research/references/ionic_liquids.md` exists yet; create it before the first download.

| Source | Measured or calculated | Size | Licence as found | Source ID per row | Still unchecked |
| --- | --- | --- | --- | --- | --- |
| NIST ILThermo v2.0 (SRD 147), https://ilthermo.boulder.nist.gov/ | measured, compiled from papers | not counted here | **Conflict.** data.gov lists the NIST open licence (https://www.nist.gov/open/license), while the ILThermo v2.0 home page carries "(c) US Secretary of Commerce ... All rights reserved". Cite v2.0 as Kazakov et al., NIST SRD 147; Dong et al. 2007 (doi 10.1021/je700171f) describes v1 | not checked | the terms; the site was unreachable for the verifier |
| CheMixHub subsets of ILThermo, arXiv 2506.12231; github.com/chemcognition-lab/chemixhub | as ILThermo | viscosity 75,992 points, 15,878 mixtures, 699 molecules; conductivity 40,904 points, 14,438 mixtures, 479 molecules | CheMixHub appendix A.1 lists "IlThermo: CC BY 4.0" | see the mirror | none beyond that |
| Mirror: hf.co/datasets/liuganghuggingface/moldata-chemixhub_ionic_liquids | as ILThermo | 116.8K rows, 28 columns: T, P, value, error, property, unit, component SMILES, mole fractions, `source_dataset`, `source_table`, `source_row_first` | cc-by-4.0 tag | **none**: no source-paper column | none |
| ILThermoPy, github.com/IvanChernyshov/ILThermoPy | an unofficial loader | none | MIT | its README does not say whether it keeps literature references | whether it does |

**Not used:** the Ionic Liquid Property Explorer (Zenodo, doi 10.5281/zenodo.3251660 according to OpenAlex; the
research's ...3251661 was wrong). It is a virtual library of over 8 million ionic liquids with machine-learning and
COSMO-RS predictions, not measurements.

Traps: drop rows that break a plain physical trend, such as viscosity rising with temperature (category research),
and count them. Holding out whole papers (DEC-009) needs a fresh pull from ILThermo that keeps the reference fields.

## 3. Candidate space
New pairings of a cation and an anion that each appear in the data but not together; mixture fractions within the
tested range of that system; temperature and pressure within the measured range.

## 4. Models
- **Expected value:** descriptors or fingerprints of each ion, mole fractions, temperature and pressure, with tuned
  trees and the table network (`gpu/train_tabular.py`).
- **Cautious estimate:** conformal interval (DEC-017).
- **Held out by entity:** whole ion pairs, then whole cation families; whole source papers only after the fresh pull.
- **Metrics:** MAE in log10 units with the paired-bootstrap interval over ion pairs (DEC-012).
- **Temperature behaviour:** not enforced by any current model (physics-code research); check each candidate along
  temperature and report the exceptions.

## 5. Cost objective
No ion price is sourced in this repository. Rank by the cautious conductivity (or viscosity) at the chosen
temperature. A cost term waits for cited prices.

## 6. Neighbour check
The 3 nearest measured ion pairs that share the cation or the anion, at the nearest measured temperature;
"neighbours passed" only when at least 2 of them reached the target.

## 7. First step, and what would stop it
- **First step:** settle the ILThermo terms (read its terms page, or ask NIST) before committing any rows. Until then
  work from the CheMixHub mirror in a local folder, with whole ion pairs held out.
- **Stop** committing data if the terms forbid redistribution; publish metrics only. If no reference fields can be
  pulled, results stay at ion-pair level and say that whole papers were not held out.

## Conclusions
- [Certain] Large measured sets exist: 75,992 viscosity and 40,904 conductivity points from ILThermo.
- [Likely] The terms conflict and the missing paper column, not the amount of data, are what block this category.
- [Speculative] Ion-pair-held-out errors carry over to new cation families. Not tested.

---
**Ground rules:** predictions labelled as predictions; every non-learned number cited; the tool
narrows the experiment list, it does not replace the experiment; chemistry & materials only, never
biology. See `../../docs/AGENTS.md`.
