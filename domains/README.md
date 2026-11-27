# domains/

> **Status: concrete is built in the main app. `materials` (alloy hardness), `insulators` (dielectric
> constant) and `polymers` (glass transition) are live on the site; their folders hold the agent's
> `prepare.py` and `nn_generic.js`, copied byte-identical, plus `result.json` and `skeptic_check.json`.
> `chemistry` (solubility) has its AqSolDB source data, preparation/training code, and a model trained
> and checked offline (reproduced exactly on 2026-11-21); it is not on the site. `cleaning` (SURFPRO),
> `corrosives` (NIST CORR-DATA) and `electrolytes` (OBELiX) have source data, a cleaned table and a first
> grouped-cross-validation baseline each; none of them is in the product. Since 2026-11-26 all three also have loaders in `gpu/data.py`
> (`cleaning_pcmc`, `cleaning_tension`, `corrosion`, `electrolytes`), so the GPU pipeline trains network sweeps for them.** Sources were not checked again through MCPs (see docs/AGENTS.md), and the
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
dyes and pigments · catalysts · sealants · composites · glasses and ceramics. Oxide glasses, dyes and
liquid battery electrolytes now have written plans, as do six categories not on this list; see
"Planned categories" below. Propellants and other energetic materials are out of scope (removed from
this list on 2026-11-27; every prompt in `../research/perplexity_prompts.md` already excludes them).

The shared tree and network scripts in `train/` and `pipeline/` were written for concrete; they are
not automatically valid for molecular inputs or different measurement conditions. Each trained
family keeps its own feature preparation and evaluation code. Literature grounding for every domain
lives in `../research/references/<domain>.md`.

## Planned categories (plans only, 2026-11-27)

Nine new categories have a written plan in `<name>/README.md`: the template's six sections plus a first step and a
stop rule. Nothing was downloaded and no model trained for any of them, and none has a
`../research/references/<name>.md` file yet. The plans come from the category research of 2026-11-27 and its
independent check ("the verifier"), and use the checked values. Calculated and measured labels stay in separate
families (`../docs/DATA_ROADMAP.md`). Ranked by feasibility, meaning how soon a held-out, grouped result could exist,
not by value. [Certain]: strong evidence; [Likely]: evidence that is not proof; [Speculative]: an educated guess.

| Rank | Plan | Data, as checked | Why this rank |
| --- | --- | --- | --- |
| 1 | `thermoelectrics/` | 23,218 crystals with DFT-calculated Seebeck coefficient and power factor, on disk (CC BY 4.0 per the provenance file); measured: ESTM, 5,205 observations of 880 materials | [Certain] no download; the crystal training and tree scripts take a new target through `TARGETS` (the nightly pipeline also needs per-target entries in `gpu/pipeline/plan.py`). The measured part waits on unclear reuse terms |
| 2 | `alloys_hot/` | MPEA, already in `materials/work/data/`: 1,067 yield strengths with test temperature, 275 at 500 °C or above (81 formulas, 36 papers); Apache-2.0; a DOI on every row | [Certain] the data are in hand. [Likely] the hot subset is too small for tight intervals |
| 3 | `solar_absorbers/` | 10,022 SLME values and 21,559 TBmBJ gaps, DFT-calculated, on disk; 54 measured gaps under NIST terms to check against | [Certain] the DFT part is ready. [Speculative] that a calculated SLME ranks real cells |
| 4 | `liquid_electrolytes/` | CALiSol-23: 13,825 measured conductivities from 27 papers, a `doi` column, CC BY 4.0 per the mirror card | [Likely] the cleanest measured fit, but it needs a download and a licence check first. The licence on the DTU record and empty DOI cells are not checked; 27 papers give wide intervals |
| 5 | `glasses/` | SciGlass: about 422,000 measured glasses and melts, in Access files | [Likely] the largest measured set, but the file reader and the ODbL share-alike terms come first; source IDs per row not checked |
| 6 | `polymer_membranes/` | 786 polymers, 5 gases, MIT tag on a mirror | [Likely] small, but it reuses the polymer featuriser; upstream terms not checked |
| 7 | `dyes/` | 20,236 hand-curated measured points from 1,358 articles | [Likely] good data; licence and article IDs not checked |
| 8 | `ionic_liquids/` | ILThermo subsets: 75,992 viscosity and 40,904 conductivity points | [Likely] blocked by a terms conflict and by a mirror with no source-paper column |
| 9 | `magnetic/` | text-mined Curie and Néel temperatures: 39,822, 56,037 and 67,573 entries; precision 73% and 72% for the two Cole databases, none reported for NEMAD | [Likely] label noise and unread data licences come before any model |

### Not planned now

Scores (1 to 5) are the category research's; the corrections are the verifier's. Piezoelectrics and solvents scored
4, as high as some planned categories, and are left out for the reasons given.

| Category | Score | Why not now |
| --- | --- | --- |
| Piezoelectrics | 4, calculated only | JARVIS holds 3,347 `dfpt_piezo_max_dij` and 4,799 `dfpt_piezo_max_eij` values; Foundry-ML `piezoelectric_tensor_v1-1` has 941 DFT tensors (CC BY 4.0). No measured set was searched, and the target is heavy-tailed with many zeros. A later DFT family |
| Hydrogen-storage alloys | 3 | ML-HydPARK v0.0.5 (Zenodo 10680097) is CC BY 4.0 with 772 rows; 741 have a wt% H, 412 a temperature and only 373 a reference (counted by the verifier). HyStor (doi 10.1016/j.ijhydene.2024.09.390): 1,280 alloys, licence not checked. Small, and few rows can be grouped by paper |
| Catalysts | 3 | OC20 and OC22 are CC BY 4.0 but large and GPU-heavy. Catalysis-Hub (doi 10.1038/s41597-019-0081-y): CC BY 4.0 is the article's licence; the data's appears only in a third-party catalogue. OCx24 (arXiv 2411.11783): its 441 electrodes include replicates, leaving 230 distinct preparations and 179 modelling targets; CC BY 4.0. Slab-plus-adsorbate inputs need new code |
| Porous materials for CO2 (MOFs) | 3 calculated, 2 measured | CoRE MOF 2019: the research's DOI 10.5281/zenodo.7691378 is v1.1.4; v2.0.0 is 10.5281/zenodo.14132134 and the latest, v2.0.1, is 10.5281/zenodo.14184621 (10,367 structures, CC BY 4.0, derived from the CSD). NIST ISODB (37,703 isotherms) reads "All rights reserved": not an open licence, so ask before redistributing. ODAC23 (CC BY 4.0) has a 232 GB training split and an 809 MB subset. QMOF is CC BY 4.0; hMOF's count and terms are not checked. Cells of hundreds of atoms, and uptake depends on conditions |
| Battery electrodes | 3 calculated voltage, 2 measured capacity | Materials Project insertion electrodes need a free API key; count and reachability not checked. Huang and Cole 2020 (doi 10.1038/s41597-020-00602-2, 292,313 records): C-rate "cannot be standardised". Capacity depends on rate, cycle number, voltage window and loading |
| Fuel-cell perovskites | 3 | Foundry-ML sets are small and licensed "other": area-specific resistance 289, conductivity 7,230, thermal expansion 137 (from experiment), H absorption 795 |
| Thermal-barrier coatings | 2 | No open measured coating dataset was found. OSTI 988711 is a 2010 article (doi 10.1007/s11666-010-9486-z), not a dataset. The AFLOW sets (thermal conductivity 4,887, licence "other") are calculated bulk values, not coatings |
| Composites | 1 to 2 | The 2026 *Sci. Data* tensile set (doi 10.1038/s41597-026-07333-w) covers one system (polysulfone with T700SC fibre). The DukeSpace link is a dissertation, not a dataset. The Mendeley CFRP set (fspdwb4mst) is CC BY-NC 3.0, non-commercial. Fibre, layup and cure are the hardest inputs to represent |
| Solvents | 4 | Large NIST data (239,201 binary-mixture viscosities, per the CheMixHub table; a separate, smaller public extract, mds2-2962, at 293.15 to 323.15 K, includes `refs.dat`, so that extract can be grouped by paper) and a flash-point set of 17,333 entries (licence not checked). Each measurement is cheap, so a surrogate saves little. From OPERA, only its physical-property endpoints would be used |
| Deep eutectic solvents | 3 | DeepDESmelt (Zenodo 20274752): 2,349 systems with measured melting points, licence not checked. Labels are noisy (many reported mixtures are not deep eutectics) and the experiments are cheap |
| Fuels | 3 | Combustion-engine fuels only, never propellants. Motor octane: 684 points (CC BY 4.0 per CheMixHub); NREL cetane compendium: 496 compounds, PDF tables only; sooting index: 400 or more compounds, licence not checked. One heating-value set (Mendeley 10.17632/5xswr8yp34) has gaps filled by nearest neighbours, so it is not purely measured |
| Refrigerants | 2 | About 300 known refrigerants; most open values come from equations of state, so they are calculated. The research's 56,000-to-62 screening funnel is not confirmed for the DOI it gave |
| Lubricants | 2 viscosity index, 1 friction | No open viscosity-index set; the alkane compilation (arXiv 1908.02067) defines one only for eicosane. The friction data (410 coefficients) cover one coating type |
| Coatings | 2 | One lacquer set (Zenodo 13742098) whose contents and licence are not checked, and a 32-sample study |
| Adhesives and sealants | 1 | About 32 samples and patent tables |
| Polymer tensile strength | 2 | PoLyInfo is restricted. PolyBench26 has no tensile data, and its bundled data carry a GTRC licence and per-source terms, not MIT |

## Current family status

| Family | Where it stands | Next useful evidence |
| --- | --- | --- |
| Concrete | Main app; the pre-registered far-future test showed the evolved model lost to a log-age straight-line baseline. | New sealed long-age measurements; keep capped-age work exploratory. |
| Alloys | Live hardness model on measured literature data; rough screening performance is similar to tuned tree baselines. | More alloys with source/processing context and a new whole-system holdout. |
| Dielectrics | Live dielectric-constant model, trained on calculated DFT labels. | Broaden with separately labelled JARVIS data; validate against measured dielectric properties. |
| Polymers | Live Tg model on experimental data; licensing/source audit remains open. | Resolve data rights and add processing-aware, polymer-family-held-out evidence. |
| Chemistry | Offline AqSolDB solubility model, reproduced exactly on CPU (network 0.761 against tuned trees 0.777 LogS MAE: a tie). The fixed test split is historical, not new evidence. | Preserve as a low-cost benchmark; do not claim it beats tuned trees. |
| Cleaning | SURFPRO cleaned; paper-held-out grouped CV: pCMC 0.53 against 0.90 log units for the mean; surface tension at the CMC 4.75 against 5.55 mN/m (weak). Surfactant properties, not cleaning efficacy. | Surfactant-family and scaffold holdouts; formulation-level cleaning tests for any efficacy claim. |
| Corrosion | NIST CORR-DATA parsed into a normalised table (24,721 rows; 7,718 numeric rates, 16,648 letter ratings). Exploratory source-held-out baseline: log10 mm/yr MAE 0.78 against 1.03 for the mean. Not safety advice. | Harmonise material names; model the letter ratings and bounds (77% of rows); audit extreme rates. |
| Solid electrolytes | OBELiX formula-only baseline, 124 composition/paper groups: ExtraTrees 1.52 (1.59-1.71 over reshuffled folds) against 2.55 log10 S/cm for the mean, about a 33x typical error. Too imprecise to rank candidates; the public holdout is not blind and was not scored. | Structure-aware model on the audited CIFs (all 321 parse), compared on the same folds; then new measurements. |

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
