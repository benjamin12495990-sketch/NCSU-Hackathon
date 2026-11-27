# High-entropy and refractory alloys: yield strength at temperature

> **Status: plan only (2026-11-27). No model trained, nothing on the site.** A new target for the alloy data the
> repository already holds. It would be a separate task and never changes the live hardness model (`../materials/`).
> Facts come from the category research of 2026-11-27 (materials), the data research on measured alloys, and their
> independent checks ("the verifier"); where a verifier corrected the research, this page uses the correction.
> "Counted here" means counted for this plan from `../materials/work/data/MPEA_dataset.csv`.

## 1. The expensive outcome
- **Predicted:** yield strength (MPa) from composition, processing route, test type (tension or compression) and
  test temperature.
- **Why it is slow:** arc melting and homogenisation, then hot compression or tension tests in vacuum or inert gas,
  one test per temperature. This is general background in the category research, not from a cited source.

## 2. Dataset
No `../../research/references/alloys_hot.md` exists yet; create it before adding any source.

| Source | Measured or calculated | Size | Licence as found | Source ID per row | Still unchecked |
| --- | --- | --- | --- | --- | --- |
| MPEA, Borg et al. 2020, *Sci. Data* 7, 430, doi 10.1038/s41597-020-00768-9; in the repository as `../materials/work/data/MPEA_dataset.csv` | measured, compiled from papers | 1,545 rows; 1,067 with yield strength, every one with a test temperature (743 compression, 324 tension); 395 formulas, 150 element systems, 191 DOIs. At 500 °C and above: 275 rows, 81 formulas, 55 element systems, 36 DOIs, 253 compression and 22 tension, 500 to 1,600 °C, 24 to 1,597 MPa (counted here; 1,067, 275, 81 and 395 match the research and the verifier) | Apache-2.0 (GitHub CitrineInformatics/MPEA_dataset) | a DOI on every yield-strength row (counted here: none missing) | how the "Calculated Density" column was calculated |
| Couzinié, Senkov, Miracle, Dirras 2018, *Data in Brief* 21:1622-1641, doi 10.1016/j.dib.2018.10.071 | measured | 122 refractory high-entropy and complex concentrated alloys (2010 to January 2018): yield stress versus test temperature, loading type, microstructure, computed density | not checked | not checked | licence; overlap with MPEA |
| Gorsse, Nguyen, Senkov, Miracle 2018, *Data in Brief* 21:2664-2678, doi 10.1016/j.dib.2018.11.111 | measured | 370 alloys (2004 to 2016), plus 27 refractory alloys with yield stress versus temperature | not checked | not checked | Borg 2020 says it is "built upon recently published MPEA databases" (quote confirmed by the data verifier), so Gorsse is a cross-check, not new rows. The research's claim that its density and modulus are rule-of-mixtures estimates was not confirmed |
| Yadav 2026, doi 10.5281/zenodo.23491016 | derived from MPEA | a de-duplicated yield-strength table of 1,007 records, each with source DOI, provenance cluster and cluster-grouped folds | CC BY 4.0 | yes | none; it is a de-duplication aid, not new data |
| ULTERA, Zenodo 7566416 and 15716432 | measured, creep-focused | 7566416 (July 2022): over 6,200 points on 2,485 alloys from 455 DOIs. 15716432 (May 2025 alpha): over 12,540 points on over 4,270 alloys | restricted: neither record has files or a licence | not usable now | access |

Traps, counted here:
- 72 of the 81 formulas tested at 500 °C or above were tested at 3 or more temperatures, and 80 of them also have a
  room-temperature row. One alloy's temperature series is one entity: its whole element system goes to one side.
- 253 of the 275 hot rows are compression tests, so a tension prediction at temperature rests on 22 rows. Test type
  is an input and is never pooled silently.
- Processing route: the alloy generator once counted neighbours made by other routes as support (LES-004). Here the
  route must match.

Perplexity round 1 found no open hot-hardness data (`../../research/perplexity/ROUND1_DIGEST.md`; a lead, unverified).

## 3. Candidate space
The rules of `src/lib/alloyGen.js`: only element sets already tested, each fraction inside the range seen in that
set, fractions in steps of 0.001, and its L1 distance tiers (under 0.10 not new; 0.10 to 0.40 kept in three tiers;
over 0.40 dropped). Added here: the test temperature stays within the range measured for that element set, and the
user fixes the test type.

## 4. Models
- **Expected value:** tuned trees and the table network (`gpu/train_tabular.py`) on the alloy family's features
  (element fractions plus Hume-Rothery-style descriptors, per the physics-code research), with test temperature, test
  type and route added.
- **Cautious estimate:** conformal lower edge (DEC-017).
- **Held out by entity:** whole element systems (the alloy family already has
  `../materials/work/cv_folds_system.json`) and whole source papers (DEC-009). Rows at 500 °C and above are reported
  on their own as well.
- **Metrics:** MAE in MPa with the 95% paired-bootstrap interval over element systems (DEC-012). Opponents: tuned
  trees (LES-003, LES-011) and the same model without the temperature input, which shows whether temperature is
  learned at all.
- **Temperature behaviour:** nothing in the current models makes strength fall as temperature rises (physics-code
  research). Check each candidate's predictions along temperature and report how many rise. scikit-learn's
  `monotonic_cst` (its support was confirmed by the physics-code verifier) is one option for the trees. Perplexity's
  P1 answer says such constraints hold only near the training data (a lead, unverified).
- **Prior evidence:** on hardness, the network average showed no clear difference from tuned trees on grouped folds,
  +5.05 [-1.39, +12] HV (`docs/results/20261122T014943Z_overnight_REPORT_20261126T2001.md`).

## 5. Cost objective
- **Specific strength at temperature:** yield strength divided by density. All 1,067 yield-strength rows carry the
  dataset's own "Calculated Density"; 102 also have a measured density (counted here). A new composition needs
  element densities with a citation, and none is sourced yet, so the first version ranks by the cautious yield
  strength.
- **Element price or scarcity:** none sourced (`research/references/materials.md`: "No entries yet").

## 6. Neighbour check
The 3 nearest distinct tested compositions (L1 on element fractions) made by the same route and tested in the same
mode, read at the nearest tested temperature. "Neighbours passed" only when at least 2 of the 3 reached the target
there (the rule in `src/lib/alloyGen.js`, with the route fix of LES-004).

## 7. First step, and what would stop it
- **First step:** a loader for the MPEA yield-strength rows (formula, route, test type, temperature, DOI), checked row
  by row against Yadav's 1,007-record table; then grouped cross-validation by element system, trees against the mean
  and against the no-temperature model.
- **Stop** if, on held-out element systems at 500 °C and above, no model is clearly better than the no-temperature
  model, or if 55 systems and 36 papers give intervals too wide to separate any model from the mean.

## Conclusions
- [Certain] The data are already in the repository, with a DOI on every yield-strength row and an Apache-2.0 licence.
- [Likely] The hot subset (81 formulas, 36 papers) is the binding limit. [Speculative] Couzinié's 122 alloys could
  grow it, but MPEA was "built upon recently published MPEA databases" (Borg 2020), the category research says
  Couzinié overlaps MPEA, and the overlap is not counted, so the number of new rows may be small.
- [Speculative] Training on hardness and yield strength together helps both. Not tested.

---
**Ground rules:** predictions labelled as predictions; every non-learned number cited; the tool
narrows the experiment list, it does not replace the experiment; chemistry & materials only, never
biology. See `../../docs/AGENTS.md`.
