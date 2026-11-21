# OBELiX solid-electrolyte data

> Reconstructed in this repository from the documentation snapshot of commit eff7c37 (branch experiment/physics-capped-age, another workspace, not pushed). The file was downloaded again here and its SHA-256 matches the value recorded in that snapshot.

- Source: NRC-Mila OBELiX, "A Curated Dataset of Crystal Structures and Experimentally Measured Ionic Conductivities for Lithium Solid-State Electrolytes".
- Upstream repository: https://github.com/NRC-Mila/OBELiX
- Pinned source revision: `64c795b0b4c0c47237b1aad3eaa2539a72a3e23b`
- Files: `processed.csv`, and the author-provided `train_idx.csv` / `test_idx.csv`, from upstream `data/`; `all_cifs.zip` from upstream `data/downloads/`; `LICENSE.txt` from the repository root.
- Licence: CC BY 4.0, as supplied by the upstream `LICENSE.txt`. Retain attribution and cite the dataset paper.
- Dataset paper: Therrien et al., "OBELiX: A Curated Dataset of Crystal Structures and Experimentally Measured Ionic Conductivities for Lithium Solid-State Electrolytes", DOI 10.1039/D5DD00441A.

The processed CSV has 599 rows and 12 columns (ID, Composition, Space group number, a, b, c, alpha, beta, gamma, Ionic conductivity (S cm-1), DOI, CIF). The target is experimental room-temperature ionic conductivity in S/cm, from 4.2e-18 to 2.5e-2. The source split assigns 478 rows to train and 121 to test. The snapshot's validation found complete ID coverage, no duplicate IDs, and zero train/test overlap in composition or source DOI.

The authors' split procedure groups entries by composition and source paper before assigning splits. The table has 223 distinct DOI field values; some rows cite several DOI tokens, and 51 compositions have repeated entries. CIF provenance is mixed: 260 entries are marked Match, 61 Close Match and 278 No Match. `all_cifs.zip` holds 321 randomized CIF files; they are not guaranteed exact experimental geometries.

| File | Bytes | SHA-256 |
| --- | --- | --- |
| processed.csv | 62,547 | `2a57b4b4287bd56e774402c8aec313f44c33df74d26a32e4bc3196e40b051474` |
| train_idx.csv | 1,914 | `6bd506b34d431ff233880147f934369e3d7c7f739299e10284e32927f502775a` |
| test_idx.csv | 486 | `0ad4cfaf5a61e85e228b2e0782a97b01469d08ac481b553622b91035a8ec719d` |
| all_cifs.zip | 292,449 | `90f54f781fe6c4ba014636046556ec66c3bfa909b617e7258ff50ac3dcfa027c` |

The archive passes ZIP integrity checks and contains 321 files. The official holdout labels are in the public table, so the holdout is not blind prospective evidence. A generalization claim needs newer measurements sealed before model selection.
