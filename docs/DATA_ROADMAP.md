> Reconstructed from the documentation snapshot of commit eff7c37 and checked against this repository on 2026-11-21.

# Data roadmap for the material simulations

This is a shortlist for building reproducible training sets, not a claim that every source is ready
to train immediately. Before use, download the source metadata and licence, preserve raw files
unchanged, record checksums, and build a documented cleaned table with units, citations, duplicate
handling, and group-aware splits. Keep computed values separate from experimental measurements.

## What the repository has now

| Simulation | Current data | Practical state |
| --- | --- | --- |
| Concrete | UCI concrete strength; BOxCrete pre-training rows (`lab/external/`) | Ready to run locally. UCI contains a small set of strength measurements; BOxCrete currently contributes early-age data only (ages 1 to 28 days). |
| Alloys | MPEA hardness data | Ready for repeatable CPU training. Keep alloy systems grouped when measuring generalization. |
| Dielectric materials | Materials Project-derived DFT dielectric values | Ready for computational-property prediction; these are calculated labels, not lab measurements. |
| Polymers | Experimental Tg dataset | Ready, but source identity, processing, and measurement context should be audited before combining other sources. |
| General chemistry | AqSolDB CSV, preparation/training code, and historical model results | Source CSV restored and pinned in `domains/chemistry/work/data/`. The CPU rerun was regenerated here and every metric reproduced exactly (`domains/chemistry/work/reproductions/20261121_cpu.json`). It reproduces the prior split and is not new test evidence. |
| Corrosion | NIST CORR-DATA archive | Raw archive acquired (`domains/corrosives/work/data/`) and parsed into `work/cleaned/corrdata_normalized.csv` (24,721 rows, source IDs kept). An exploratory source-held-out baseline exists; letter ratings and bounds are not modelled yet. |
| Cleaning / surfactants | SURFPRO CSV and source metadata | Data acquired (`domains/cleaning/work/data/`), cleaned, and given paper-held-out pCMC and surface-tension baselines (`work/reproductions/`). This is not cleaning-efficacy data. |
| Solid lithium electrolytes | OBELiX table, official train/test IDs, and randomized CIF archive | Data acquired (`domains/electrolytes/work/data/`); source split verified. The formula-only grouped-CV baseline was rebuilt here (`work/reproductions/obelix_group_cv.json`): about 1.5-1.8 against 2.55 log10 S/cm for the mean, too weak to rank candidates. No holdout score or product model. |

The family status and existing preparation scripts are documented in `domains/README.md`. Corrosion
and cleaning should not be shown as trained simulations until each has domain-specific features, a
held-out protocol, and a model evaluation.

## Best next data targets

### Corrosion compatibility: NIST CORR-DATA

NIST's CORR-DATA records published observations of samples in corrosive environments, including
material, environment, concentration, temperature, and observed rate. The catalog describes over
24,000 records from more than 250 source documents. It is historical (the program ended in 1997),
and the catalog explicitly warns that an individual observation does not establish service safety.
This looks like the strongest near-term source for a corrosion-rate or compatibility prototype.
Keep source document and test condition as grouping/provenance fields; do not randomly split rows
from the same study across train and test.

- Dataset page: https://catalog.data.gov/dataset/corr-data
- NIST data landing page: https://data.nist.gov/od/id/54AE54FB37AC022DE0531A570681D4291851
- Data archive: https://opendata.nist.gov/1851/CORR-DATA_Database.zip
- License information: https://www.nist.gov/open/license
- Local copy, size and checksum: `domains/corrosives/work/data/README.md`

### Surfactants: SURFPRO and the NIST CMC collection

SURFPRO lists 1,624 surfactants, including 1,395 experimental pCMC and 972
surface-tension-at-CMC records curated from literature. Its CSV has been acquired into
`domains/cleaning/work/data/`; the source record lists CC BY 4.0 and the local README records
attribution and checksum. The PSDI CMC collection is another machine-readable source derived from a
NIST reference publication and identifies a CC BY 4.0 license. These sources support a first
surfactant-property model (CMC or surface tension), not a direct claim about cleaning efficacy. For
a cleaning-performance model, add formulation-level cleaning tests with substrate, stain,
concentration, temperature, and test protocol.

- SURFPRO: https://zenodo.org/records/15861809
- PSDI/NIST CMC collection: https://resources.psdi.ac.uk/data/a7c82670-d2e2-46c6-920a-74294289aa34
- NIST Chemistry WebBook: https://webbook.nist.gov/chemistry/

### More materials and dielectric tasks: JARVIS and Matbench

NIST JARVIS provides curated computational materials data and an OPTIMADE API. Materials Project's
Matbench provides property-prediction benchmark datasets. They can broaden composition/property
experiments and provide standard benchmark comparisons. Many labels are DFT-computed; preserve the
calculation provenance and do not mix them with measured material results without an explicit
target definition and source flag.

- JARVIS: https://jarvis.nist.gov/
- JARVIS-OPTIMADE: https://jarvis.nist.gov/optimade/jarvisdft/
- Matbench docs: https://docs.materialsproject.org/services/ml-and-ai-applications/matbench
- Materials Project access and terms: https://docs.materialsproject.org/downloading-data/using-the-api

### Battery materials: solid lithium electrolytes

OBELiX adds a distinct family: experimental room-temperature ionic conductivity for 599 synthesized
Li solid electrolytes, with composition, crystal descriptors, source DOI, and CIF availability
metadata. The published split has 478 training and 121 public holdout rows; the authors' procedure
separates source papers and compositions. In the snapshot, the formula-only baseline used grouped
CV only and was too imprecise for candidate ranking. That baseline is being rebuilt in this
repository. This target does not measure full-cell capacity, cycle life, safety, or
manufacturability.

- OBELiX repository: https://github.com/NRC-Mila/OBELiX
- Dataset paper: https://doi.org/10.1039/D5DD00441A
- License, file hashes, and randomized-structure caveat: `domains/electrolytes/work/data/README.md`

### GPU-relevant materials data: NIST JARVIS

For a later structure-aware model, NIST JARVIS is a much larger and more GPU-relevant source than
the current tabular families. Its 3D DFT collection includes roughly 77,000 materials and computed
properties such as formation energy, band gap, elastic tensors, dielectric response,
piezoelectricity, and Seebeck coefficients. Start with one narrow target (elasticity or
thermoelectrics) and its benchmark splits; do not mix these calculated labels with experimental
measurements. The full 3D archive is about 1.42 GB, so it is not copied into this repository. Use
the NIST data page or jarvis-tools named-dataset loader to retrieve only what the chosen task needs,
and record the exact version and checksum.

- NIST JARVIS overview and property counts: https://jarvis.nist.gov/
- JARVIS-Tools dataset API: https://jarvis-tools.readthedocs.io/en/master/databases.html
- JARVIS-DFT 3D dataset record, CC BY 4.0: https://doi.org/10.6084/m9.figshare.6815699

Matbench offers useful benchmark tasks such as experimental band gaps, steel yield strength,
exfoliation energy, and phonon targets. It mixes Materials Project data with external datasets, so
check the license and provenance for the specific task before copying its raw data into this repo.

- Matbench overview: https://docs.materialsproject.org/services/ml-and-ai-applications/matbench

### Longer-term GPU family: catalyst adsorption and electrocatalysis

Open Catalyst 2020/2022 provides structure-level DFT tasks for adsorption energies, total energies,
forces, and relaxations. OC20 contains about 460,000 adsorbate-catalyst systems and more than 133M
single-point calculations; OC22 focuses on 62,331 oxide electrocatalyst relaxations. These are a
natural graph-neural-network workload for the 5080, but their full datasets are large. Start with
an official small task/split after the JARVIS pipeline works; do not download or train the full
corpus as the first GPU run. Predictions approximate DFT labels and still need DFT or lab
confirmation.

- OC20 official dataset documentation and CC BY 4.0 license: https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc20.md
- OC22 official dataset documentation: https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc22.md
- Dataset papers: https://doi.org/10.1021/acscatal.0c04525 and https://doi.org/10.1021/acscatal.2c05426

### Polymer properties

PoLyInfo records polymer structures, properties, measurement conditions, processing, and monomer
information. Some dataset exports are available publicly, while broader datasets may require paid
access. Obtain and review the exact terms before adding records. Start with measured Tg and keep
polymer identity, source paper, method, and sample conditions; group by polymer family or structure
when testing generalization.

- PoLyInfo: https://polymer.nims.go.jp/en/
- Existing open experimental dataset used by this repo: see `domains/polymers/README.md` and its
  `work/data/` files.

### General chemistry

AqSolDB contains 9,982 curated aqueous-solubility records and is CC0. The source CSV is now
restored in `domains/chemistry/work/data/`; its pinned upstream commit and checksum are documented
beside it. The snapshot recorded a rerun of the preparation and evaluation pipeline. That rerun is
being regenerated here; its compact report goes in `domains/chemistry/work/reproductions/`.

- Harvard Dataverse record: https://doi.org/10.7910/DVN/OVHAW8

## Practical order

1. Run the fixed OBELiX grouped-CV baseline (its script is being rebuilt here), inspect error by
   material family, and treat its public holdout as a benchmark only. Seek newer paper-held-out data before claiming generalization.
2. Build the SURFPRO pCMC/surface-tension task with paper-held-out evaluation; do not call it a
   cleaning-efficacy model.
3. Parse NIST CORR-DATA into source-linked rate/condition tables, normalize units, and hold out
   studies before training.
4. Keep AqSolDB as a reproducible chemistry task. Its CPU rerun goes in
   `domains/chemistry/work/reproductions/`; it is not new test evidence.
5. Add a narrow JARVIS elastic or thermoelectric task for structure-aware GPU work, with a pinned
   dataset version and benchmark split. Keep computed and measured labels distinct.
6. Expand polymer data only after licensing, units, duplicate identity, and measurement conditions
   have been reviewed.
7. For tasks without suitable historical data, collect lab measurements and seal a new test set
   before training decisions.

## How to scale local training

The current models are small tabular networks and trees. Six parallel CPU workers
(`npm run lab:evolve2 -- --workers 6`) can increase throughput if the laptop has available cores,
but do not make the predictions more accurate by themselves. The current JavaScript trainer does
not use the RTX 5080. Profile the actual laptop before porting a real training loop to
PyTorch/CUDA: `lab/GPU_LAPTOP.md` gives the reasoning and `gpu/README.md` the commands. This
workspace has no NVIDIA device or PyTorch installation. A GPU port must preserve the CPU reference,
fixed splits, and reported metrics.
