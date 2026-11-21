> Reconstructed from the documentation snapshot of commit eff7c37 and checked against this repository on 2026-11-21.

# References — Solid lithium electrolytes

### OBELiX conductivity data

- **Source:** Therrien, F. et al. "OBELiX: A Curated Dataset of Crystal Structures and Experimentally
  Measured Ionic Conductivities for Lithium Solid-State Electrolytes." DOI: https://doi.org/10.1039/D5DD00441A.
- **Dataset repository:** https://github.com/NRC-Mila/OBELiX
- **License:** CC BY 4.0 in the upstream repository.
- 599 synthesized solid electrolytes with experimental room-temperature ionic conductivity; the
  source split has 478 training and 121 public holdout rows. The split groups both source papers
  and compositions.
- 321 CIF files are included in a randomized archive. Some structural entries are close matches, so
  verify exact ID/provenance before using CIF features.
- Local source files, split hashes, and provenance: `domains/electrolytes/work/data/README.md`.

This supports a conductivity-screening task only. It does not measure whole-cell battery
performance, and the public official split is not a new blind test.
