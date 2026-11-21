---
license: cc-by-4.0
task_categories:
  - tabular-regression
  - tabular-classification
tags:
  - materials-science
  - chemistry
  - foundry-ml
  - scientific-data
size_categories:
  - 1K<n<10K
---

# High-throughput screening of inorganic compounds for the discovery of novel dielectric and optical materials

Dataset containing DFT-calculated dielectric properties for 1056 materials

## Dataset Information

- **Source**: [Foundry-ML](https://github.com/MLMI2-CSSI/foundry)
- **DOI**: [10.18126/racd-go9m](https://doi.org/10.18126/racd-go9m)
- **Year**: 2022
- **Authors**: Petousis, Ioannis, Mrdjenovich, David, Ballouz, Eric, Liu, Miao, Winston, Donald, Chen, Wei, Graf, Tanja, Schladt, Thomas D., Persson, Kristin A., Prinz, Fritz B.
- **Data Type**: tabular

### Fields

| Field | Role | Description | Units |
|-------|------|-------------|-------|
| material_id | input | Materials Project ID |  |
| formula | input | Material composition |  |
| nsites | input | Number of sites in the unit cell |  |
| space_group | input | Space group number |  |
| volume | input | Volume of relaxed structure | Cubic Angstroms |
| structure | input | Pymatgen structure representation of material |  |
| band_gap | input | Bandgap of material from Materials Project | eV |
| e_electronic | target | Electronic portion of the dielectric constant tens |  |
| e_total | target | Total dielectic constant tensor |  |
| n | target | Index of refraction |  |
| poly_electronic | target | Polycrystal estimate of electronic part of dielect |  |
| poly_total | target | Polycrystal estimate of total dielectric constant  |  |
| log(poly_total) | target | log10 of poly total |  |
| pot_ferroelectric | target | Whether the material is potentially a ferroelectri |  |
| cif | input | Material structure in CIF format |  |
| meta | input | DFT calculation metadata |  |
| poscar | input | Material structure in POSCAR format |  |


### Splits

- **train**: train


## Usage

### With Foundry-ML (recommended for materials science workflows)

```python
from foundry import Foundry

f = Foundry()
dataset = f.get_dataset("10.18126/racd-go9m")
X, y = dataset.get_as_dict()['train']
```

### With HuggingFace Datasets

```python
from datasets import load_dataset

dataset = load_dataset("dielectric_constant_v1.1")
```

## Citation

```bibtex
@misc{https://doi.org/10.18126/racd-go9m
doi = {10.18126/racd-go9m}
url = {https://doi.org/10.18126/racd-go9m}
author = {Petousis, Ioannis and Mrdjenovich, David and Ballouz, Eric and Liu, Miao and Winston, Donald and Chen, Wei and Graf, Tanja and Schladt, Thomas D. and Persson, Kristin A. and Prinz, Fritz B.}
title = {High-throughput screening of inorganic compounds for the discovery of novel dielectric and optical materials}
keywords = {machine learning, foundry}
publisher = {Materials Data Facility}
year = {root=2022}}
```

## License

CC-BY 4.0

---

*This dataset was exported from [Foundry-ML](https://github.com/MLMI2-CSSI/foundry), a platform for materials science datasets.*
