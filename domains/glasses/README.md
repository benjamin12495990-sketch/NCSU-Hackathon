# Oxide glasses

> **Status: plan only (2026-11-27). Nothing downloaded, no model trained, nothing on the site.** Facts come from the
> category research of 2026-11-27 (materials and chemistry) and their independent checks ("the verifier"); where a
> verifier corrected the research, this page uses the correction.

## 1. The expensive outcome
- **Predicted:** glass transition temperature Tg, density, refractive index, Young's modulus, thermal expansion,
  liquidus temperature and viscosity, from the oxide mole fractions. Start with one property, Tg or density.
- **Why it is slow:** melt-quench at roughly 1,200 to 1,600 °C, annealing, then DSC, refractometry and ultrasonic
  modulus: days per composition. General background in the category research, not from a cited source.

## 2. Dataset
No `../../research/references/glasses.md` exists yet; create it, with the licence terms below, before any download.

| Source | Measured or calculated | Size | Licence as found | Source ID per row | Still unchecked |
| --- | --- | --- | --- | --- | --- |
| SciGlass, github.com/epam/SciGlass | measured values from papers and patents | about 422,000 glasses and melts: over 268,000 oxide, about 18,500 halide, about 38,500 chalcogenide; over 40,000 sources, over 19,700 of them patents; about 35,000 chemical-durability entries; over 15,000 optical spectra (confirmed by both verifiers). The SciGlass site lists 422,949 (Perplexity round 2, an unverified lead) | **Treat as ODbL share-alike.** The LICENSE file is headed "ODC Open Database License (ODbL)" but its body is MIT text ("Copyright (c) 2019 EPAM Systems"); GitHub's sidebar shows MIT; the SciGlass site and Wikipedia say the data were released under the ODbL in 2019 | not checked | rows per property; whether each row carries a source ID |
| GlassPy, github.com/drcassar/GlassPy | a loader for SciGlass | none | code GPL-3.0 (PyPI: GPLv3+) | none | none |
| GlassNet, Cassar 2023, *Ceramics International* 49(22):36013-36024, doi 10.1016/j.ceramint.2023.08.281; arXiv 2303.15538 | built from SciGlass through GlassPy | 218,533 unique compositions (the full curated set, its 10% holdout included), 85 properties | not checked | none | rows per property; a Tg subset of 45,302 reported in a paper seen only in search results |
| Metallic glasses, a separate family for later: Foundry-ML `dataset_metallicglass_dmax` and `dataset_metallicglass_rc` (Hugging Face); matbench_glass | measured | maximum casting diameter 998; critical cooling rate 2,125 (its columns include "Prediction" and "quality", so filter); matbench_glass 5,680 labels (glass or not, classification) | Foundry-ML: "other"; matbench: not checked | not checked | source terms |

Format: Microsoft Access files (`select.mdb`, `property.mdb`) in Access 2.0 format for 32-bit Windows, so reading
them is real work. GlassPy's code is GPL-3.0: read that licence before copying any of it into this repository, or
write a separate reader.

Traps: the same or nearly the same composition appears in many papers and patents (LES-001, LES-011). Group by glass
system, the set of components present, which is the analogue of holding out chemical systems (DEC-014).

## 3. Candidate space
Change the mole fractions of tested glasses only within the range seen in their own system, with fractions summing
to 1 and only the components present in that system. Distance tiers as in the alloy generator (`src/lib/alloyGen.js`).

## 4. Models
- **Expected value:** tuned trees and the table network (`gpu/train_tabular.py`) on mole fractions, optionally with
  element-property features.
- **Cautious estimate:** conformal interval (DEC-017).
- **Held out by entity:** whole glass systems; whole sources too, if rows carry a source ID (DEC-009).
- **Metrics:** MAE in K for Tg, g/cm³ for density, with the paired-bootstrap interval over systems (DEC-012).

## 5. Cost objective
Raw-material price per kg of batch and melting energy are the natural costs; neither has a sourced number here. The
first version ranks by the cautious estimate. [Speculative] the same model's predicted liquidus could stand in for
melting energy, since it is learned, not looked up.

## 6. Neighbour check
The 3 nearest measured glasses in the same system (L1 on mole fractions); "neighbours passed" only when at least 2 of
them reached the target.

## 7. First step, and what would stop it
- **First step:** read `property.mdb` in a scratch folder outside the repository; count rows per property, glass
  systems, and whether a source ID exists per row. Record the ODbL obligations in
  `../../research/references/glasses.md`. Commit code and metrics only, no derived table, until those obligations
  are agreed.
- **Stop** if Tg rows carry no source ID and grouping by system shows no clear gain over the mean, or if the
  share-alike terms cannot be met for anything the project would publish; then keep the data local.

## Conclusions
- [Certain] SciGlass is the largest open measured composition-to-property source the research found, about 422,000
  glasses and melts.
- [Likely] Composition-only inputs fit the existing table pipeline; the new work is the Access-file reader and the
  grouping by system.
- [Speculative] The share-alike terms allow publishing model metrics without publishing a derived table. Nobody has
  checked this against the ODbL text.

---
**Ground rules:** predictions labelled as predictions; every non-learned number cited; the tool
narrows the experiment list, it does not replace the experiment; chemistry & materials only, never
biology. See `../../docs/AGENTS.md`.
