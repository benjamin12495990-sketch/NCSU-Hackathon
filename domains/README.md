# domains/

> **Status: concrete is built in the main app. `materials` (alloy hardness), `insulators` (dielectric
> constant) and `polymers` (glass transition) are live on the site; their folders hold the agent's
> `prepare.py` and `nn_generic.js`, copied byte-identical, plus `result.json` and `skeptic_check.json`.
> `chemistry` (solubility) has the same four files for a model trained and checked offline; its data
> file and weights are not in the repo and it is not on the site yet. The other folders are plans.** Sources were
> not checked again through MCPs (see docs/AGENTS.md), and the insulator element table was typed from
> memory by its agent and is uncited.

Scope beyond concrete. Each subfolder is one domain that reuses the project's method
(`../docs/SCOPE.md`): a surrogate model, a cautious lower estimate (on the live concrete page it covers 87.7% of test
results, short of the 90% aim), candidate generation by
perturbing known-good examples, ranking by a cost objective, and a "neighbours passed" check.

**Chemistry and materials only. No biology** — it is a complete subject of its own and is
deliberately excluded.

- `_TEMPLATE/` — copy this to start a new domain.

## Domains
- **polymers/** — carbon-based polymers (Tg, strength, thermal stability)
- **corrosives/** — corrosive solutions & material compatibility
- **cleaning/** — cleaning products, surfactants, detergents
- **insulators/** — thermal & electrical insulators / dielectrics
- **materials/** — general materials properties (concrete is the first worked example)
- **chemistry/** — general molecular / mixture properties

### Reach domains (same engine, later)
coatings · adhesives · lubricants · electrolytes & battery chemistries · refrigerants · dyes &
pigments · catalysts · propellants · sealants · composites · glasses & ceramics.

Both model scripts (`train/train.py` trees, `pipeline/train_nn.py` neural net) were written for
concrete and could be pointed at another tabular dataset. None of these domains has one yet. Literature grounding for every domain lives in `../research/references/<domain>.md`.

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
| chemistry | not included (3.8 MB) | AqSolDB, Sorkun, Khetan and Er (2019), *Scientific Data* 6, 143. Download `https://raw.githubusercontent.com/mcsorkun/AqSolDB/master/results/data_curated.csv` (repository MIT, data folder CC0) into `chemistry/work/data/` and run `prepare.py` to rebuild its 2 MB `nn_data.json` |

Not included: the polymer agent's `tasks.json` (3.4 MB of cross-validation fold rows) and the raw
Hugging Face responses for the insulators (6.4 MB); both can be rebuilt from the files above.
