# Magnetic ordering temperature (Curie and Néel)

> **Status: plan only (2026-11-27). Nothing downloaded, no model trained, nothing on the site.** This plan covers
> the Curie and Néel temperatures of magnetic materials only. Facts come from the category research of 2026-11-27
> (materials) and its independent check ("the verifier"); where the verifier corrected the research, this page uses
> the correction. "Counted here" means counted for this plan from a file in this repository.

## 1. The expensive outcome
- **Predicted:** Curie temperature T_C and Néel temperature T_N (K) of a compound from its composition, learned from
  measured values text-mined from papers.
- **Not planned:** coercivity. About 300 measured materials exist (Bhandari et al., arXiv 2312.02475, where some
  coercivities were estimated as Ha/4 rather than measured), and coercivity is controlled by microstructure, which
  composition cannot describe (general background in the category research, not from a cited source).
- **Why it is slow:** synthesis, then SQUID or VSM magnetometry versus temperature. General background in the
  category research, not from a cited source.

## 2. Dataset
No `../../research/references/magnetic.md` exists yet; create it before any download.

| Source | Measured or calculated | Size | Licence as found | Source ID per row | Still unchecked |
| --- | --- | --- | --- | --- | --- |
| Court and Cole 2018, *Sci. Data* 5:180111, doi 10.1038/sdata.2018.111; figshare collection 10.6084/m9.figshare.c.3954418 | measured values, text-mined | 39,822 records (MongoDB, CSV, JSON); precision 73% overall, 82% with the modified Snowball algorithm | not checked | not checked | licence; a DOI per row |
| Kelly and Cole 2025, *Sci. Data* 12:2000, doi 10.1038/s41597-025-06244-6 | measured values, text-mined | 56,037 records from 108,181 articles; precision 72%, recall 61% | article CC BY 4.0; data not checked | not checked | overlap with Court and Cole |
| NEMAD, Itani, Zhang, Zang, *Nat. Commun.* 16:9415, doi 10.1038/s41467-025-64458-z; arXiv 2409.15675; nemad.org | measured values, text-mined (per the category research) | 67,573 entries with Curie and Néel temperatures, coercivity, magnetization, moment, remanence and susceptibility (an older version: 26,706, from a search snippet) | none in the paper's data-availability section | not checked | licence |
| JARVIS-DFT `magmom_oszicar`, cached at `gpu/data_cache/jarvis/jdft_3d-9-24-2025.json.zip` | DFT-calculated magnetic moment | 88,811 crystals, 17,303 above 0.5 μB (counted here and by the verifier) | CC BY 4.0, from the provenance file | JARVIS id | none |
| Heusler magnetization, hf.co/datasets/foundry-ml/heusler_magnetization_v1-1 | very likely DFT-calculated: its source is named as the University of Alabama Heusler database, which a search describes as VASP/PBE | 1,153 compounds | CC BY 4.0 | not checked | whether any value is measured |

**The loader reads only composition, the Curie or Néel temperature, its unit and the source DOI.** No other field of
these databases is used.

Traps:
- At the 72 to 73% precision reported for the two Cole databases (NEMAD reports none), about a quarter of their
  records are wrong. [Likely] label noise, not the model, sets the first error floor.
- One compound appears in many papers, and Kelly and Cole extends Court and Cole, so [Likely] the two overlap
  heavily. De-duplicate on composition, value and DOI, and keep every report of one composition on one side of the
  split.
- A DFT moment is not an ordering temperature, and not the saturation magnetization of a bulk magnet. JARVIS moments
  are at most an input feature.
- The NEMAD authors report a Curie MAE of 56 K (R² 0.87) and a Néel MAE of 38 K (R² 0.83) with XGBoost on a balanced
  set. That is their split; it is comparable with ours only if re-run on our grouped split.

## 3. Candidate space
Composition changes within element sets already reported, each fraction inside the reported range, with the distance
tiers of `src/lib/alloyGen.js`. No phase is modelled: [Speculative] a compound moved off its stoichiometry may not
keep its structure.

## 4. Models
- **Expected value:** composition trees and the table network (`gpu/train_tabular.py`); T_C and T_N as separate
  targets.
- **Cautious estimate:** conformal lower edge (DEC-017), for questions such as "stays magnetically ordered above X".
- **Held out by entity:** whole compositions (all reports together) and whole chemical systems (DEC-009, DEC-014).
- **Metrics:** MAE in K with the paired-bootstrap interval over systems (DEC-012).

## 5. Cost objective
The category research names rare-earth-free permanent magnets as the valuable screen. Excluding chosen elements, for
example the rare earths, is a set filter and needs no number. A price or supply-risk term waits for cited values;
none is sourced in this repository.

## 6. Neighbour check
The 3 nearest reported compositions (L1 on element fractions). For each, the median of all its reports and their
spread are shown. "Neighbours passed" only when at least 2 of the 3 medians reach the target.

## 7. First step, and what would stop it
- **First step:** read the Court and Cole data licence at figshare, download the CSV, and count records with a DOI,
  unique compositions, and the spread between reports of the same composition.
- **Stop** if reports of one composition disagree by as much as the model error we would aim for: then clean or drop
  the labels before any model. If no licence allows a derived table, keep the data local and publish metrics only.

## Conclusions
- [Certain] Three large text-mined sources exist (39,822, 56,037 and 67,573 entries); none has a data licence anyone
  has read.
- [Likely] Composition-only models of T_C will be limited by label noise before model choice matters.
- [Speculative] JARVIS moments as an extra input improve T_C predictions. Not tested.

---
**Ground rules:** predictions labelled as predictions; every non-learned number cited; the tool
narrows the experiment list, it does not replace the experiment; chemistry & materials only, never
biology. See `../../docs/AGENTS.md`.
