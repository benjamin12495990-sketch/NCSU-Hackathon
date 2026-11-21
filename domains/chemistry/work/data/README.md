# AqSolDB source data

> Reconstructed in this repository from the documentation snapshot of commit eff7c37 (branch experiment/physics-capped-age, another workspace, not pushed). The file was downloaded again here and its SHA-256 matches the value recorded in that snapshot.

`data_curated.csv` is the curated AqSolDB CSV used by the chemistry preparation and training scripts.

- Source record: Sorkun, Khetan and Er (2019), Harvard Dataverse, DOI 10.7910/DVN/OVHAW8.
- Upstream source file: `mcsorkun/AqSolDB/results/data_curated.csv` at commit `8e02b548fd9a78778ff89a5aa9a460d1a289cc3a`.
- Download URL: https://raw.githubusercontent.com/mcsorkun/AqSolDB/8e02b548fd9a78778ff89a5aa9a460d1a289cc3a/results/data_curated.csv
- Data licence: CC0 1.0, as listed by Harvard Dataverse. The upstream code repository has a separate licence; this CSV's licence is the one that applies to the data.
- Size: 3,750,252 bytes. SHA-256: `363a42a4e4ea8e6039758146d34efdd95cde2da81c665316cacc1869e0a805ed`.

The file is kept unchanged. Rebuild derived rows with:

```
python3 -I domains/chemistry/work/prepare.py domains/chemistry/work/data/data_curated.csv domains/chemistry/work
```

The preparation script groups rows by the first InChIKey block and creates fixed grouped splits. Its output is a reproducibility artefact: the chemistry test split was already scored in the project's earlier report and is not a fresh sealed test.
