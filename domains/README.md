# domains/

> **Status: concrete is built in the main app. `materials` (alloy hardness), `insulators` (dielectric
> constant) and `polymers` (glass transition) are live on the site; their folders hold the agent's
> `prepare.py` and `nn_generic.js`, copied byte-identical, plus `result.json` and `skeptic_check.json`.
> `chemistry` (solubility) has its AqSolDB source data, preparation/training code, and a model trained
> and checked offline; it is not on the site. `cleaning` has a licensed surfactant-property source but
> no trained model yet. `corrosives` has a public NIST source archive but no parsed table or model yet.
> `electrolytes` has OBELiX data; its first grouped-CV formula baseline is being rebuilt here. It is
> not in the product.** Sources were not checked again through MCPs (see docs/AGENTS.md), and the
> insulator element table was typed from memory by its agent and is uncited.

Scope beyond concrete. Each subfolder is one domain that reuses the project's method
(`../docs/SCOPE.md`): a surrogate model, a cautious lower estimate (on the live concrete page it covers
87.7% of test results, short of the 90% aim), candidate generation by perturbing known-good examples,
ranking by a cost objective, and a "neighbours passed" check.

**Chemistry and materials only. No biology**: it is a complete subject of its own and is
deliberately excluded.

- `_TEMPLATE/`: copy this to start a new domain.

## Domains
- **polymers/**: carbon-based polymers (Tg, strength, thermal stability)
- **corrosives/**: corrosive solutions and material compatibility
- **cleaning/**: cleaning products, surfactants, detergents
- **insulators/**: thermal and electrical insulators, dielectrics
- **materials/**: general materials properties (concrete is the first worked example)
- **chemistry/**: general molecular and mixture properties
- **electrolytes/**: lithium solid-electrolyte ionic conductivity

### Reach domains (same engine, later)
coatings · adhesives · lubricants · battery chemistries beyond solid electrolytes · refrigerants ·
dyes and pigments · catalysts · propellants · sealants · composites · glasses and ceramics.

The shared tree and network scripts in `train/` and `pipeline/` were written for concrete; they are
not automatically valid for molecular inputs or different measurement conditions. Each trained
family keeps its own feature preparation and evaluation code. Literature grounding for every domain
lives in `../research/references/<domain>.md`.

## Current family status

| Family | Where it stands | Next useful evidence |
| --- | --- | --- |
| Concrete | Main app; the pre-registered far-future test showed the evolved model lost to a log-age straight-line baseline. | New sealed long-age measurements; keep capped-age work exploratory. |
| Alloys | Live hardness model on measured literature data; rough screening performance is similar to tuned tree baselines. | More alloys with source/processing context and a new whole-system holdout. |
| Dielectrics | Live dielectric-constant model, trained on calculated DFT labels. | Broaden with separately labelled JARVIS data; validate against measured dielectric properties. |
| Polymers | Live Tg model on experimental data; licensing/source audit remains open. | Resolve data rights and add processing-aware, polymer-family-held-out evidence. |
| Chemistry | Offline AqSolDB solubility model; the fixed test split is historical, not new evidence. | Preserve as a low-cost benchmark; do not claim it beats tuned trees. |
| Cleaning | SURFPRO raw data acquired. The labels measure surfactant properties, not cleaning efficacy. | Paper-held-out pCMC / surface-tension pipeline. |
| Corrosion | NIST raw archive acquired. | Parse units and ratings, retain test conditions, hold out source studies. |
| Solid electrolytes | OBELiX data and split acquired; the public holdout is not blind. | Formula-only grouped-CV baseline, then structure-aware models; then seek new measurements. |

## Work folders: everything needed to rebuild and re-check the trained families

Each trained family has a `work/` folder with the files its training agent produced: the results
(`result.json`), the independent re-run (`skeptic_check.json`), the data preparation and trainer
(`prepare.py`, `nn_generic.js`), the training and evaluation scripts, the splits, and the dataset
where its licence allows it. `npm run family-files` rebuilds `public/model/<family>/` from these
folders alone, and `npm run alloy-check` and `npm run family-parity` run from them too. The
network's own training rows (`nn_data.json`) are not duplicated here; they are read from
`public/model/<family>/`.

| Family | Dataset in `work/data/` | Source and licence |
| --- | --- | --- |
| materials | `MPEA_dataset.csv` | Borg et al. (2020), *Scientific Data* 7, 430; CitrineInformatics/MPEA_dataset on GitHub, Apache License 2.0 |
| insulators | `dielectric_table.csv` (columns taken from the full records) | Petousis et al., DOI 10.18126/racd-go9m; Hugging Face foundry-ml/dielectric_constant_v1-1, CC BY 4.0. Values are calculated (DFT), not measured |
| polymers | `experimental_dataset.csv` | Zenodo record 19242815 (Bhati et al.), CC BY 4.0 |
| polymers | `HT_MD_polymer_properties.csv` (checked, not used for training) | Hugging Face AdrianM0/bicerano_polymers, MIT |
| chemistry | `data_curated.csv` | AqSolDB, Sorkun, Khetan and Er (2019), *Scientific Data* 6, 143; CC0 data. Pinned upstream commit and SHA-256 are in `chemistry/work/data/README.md`. Run `prepare.py` to rebuild its 2 MB `nn_data.json`. |
| cleaning | `S129_SURFPRO.csv`, `S129_SURFPRO_notes.txt` | SURFPRO, Zenodo record 15861809; CC BY 4.0. Attribution and SHA-256 in `cleaning/work/data/README.md`. Surfactant-property data, not cleaning-efficacy data. |
| electrolytes | `processed.csv`, `train_idx.csv`, `test_idx.csv`, `all_cifs.zip`, `LICENSE.txt` | OBELiX, Therrien et al.; CC BY 4.0. Source revision and SHA-256 values in `electrolytes/work/data/README.md`. Experimental room-temperature conductivity; the public split is not blind. |
| corrosives | `CORR-DATA_Database.zip` | NIST CORR-DATA; NIST Open Data terms. Checksum and parsing cautions in `corrosives/work/data/README.md`. |

Not included: the polymer agent's `tasks.json` (3.4 MB of cross-validation fold rows) and the raw
Hugging Face responses for the insulators (6.4 MB); both can be rebuilt from the files above.
