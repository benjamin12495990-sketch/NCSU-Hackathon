# References — Carbon-based polymers

Sourced numbers for the polymers domain. Use `../templates/reference-entry.md`. Retrieve and
cross-check via the research MCPs (`../connectors.md`).

First sources to confirm (licence + fields) before use:
- **PoLyInfo** (NIMS) — measured polymer properties.
- **Polymer Genome** / Ramprasad-group datasets — Tg and more, with structure.
- Property tables from polymer-informatics reviews (via OpenAlex).
- Monomer embodied-CO2 / cost factors for the cost objective.

## Entries

### Measured Tg training data (in use)
- **What:** `domains/polymers/work/data/experimental_dataset.csv`, 666 rows: 315 homopolymers and 351 binary random
  copolymers, glass-transition temperature in K.
- **Source:** Bhati, Afzal, Chew, Browning and Halls (Schrodinger), "Glass Transition Prediction of Binary Copolymers
  Across Large Chemical Spaces Using Machine Learning and Physics-Based Modeling", Zenodo, 2026,
  https://doi.org/10.5281/zenodo.19242815 (file experimental_dataset.csv). The paper is reported as Polymers 18(14):1727
  (not opened by the project).
- **Licence:** "Creative Commons Attribution 4.0 International", as written on the Zenodo record (read by Perplexity on
  2026-11-27, `../perplexity/round2/computer_mode_answers.md`).
- **Identity check:** the record lists md5 11eb3b96be3116907d87038d9b74399f; the project's copy has the same MD5
  (checked 2026-11-27). SHA-256 of the project's copy:
  e4ce0b9b645230885ca60672d19e8d6c8121558e938cd91a3837bf2e75c5c939.
- **Origin of the values (reported, not checked):** 301 from Bicerano's handbook (Prediction of Polymer Properties,
  2002) and 365 from Penzel, Rieger and Schneider, Polymer 38:325 (1997). The measurement method (DSC or DMA) and heating
  rate are not stated on the record.
- **Rights:** CC BY 4.0 requires attribution: cite the Zenodo record wherever the data or a model trained on it is
  shown.

### Not usable without permission
- **PoLyInfo (NIMS):** its page says "No reproduction, republication or distribution to third parties of any content is
  permitted without written permission of NIMS" and that bulk acquisition "is prohibited by the MatNavi Service Terms of
  Use" (https://polymer.nims.go.jp, read 2026-11-27). The 7,170 GREA rows of PolyMetriX trace to PoLyInfo
  (`../data_sources.md`).
