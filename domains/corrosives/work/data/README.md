# NIST CORR-DATA source archive

> Reconstructed in this repository from the documentation snapshot of commit eff7c37 (branch experiment/physics-capped-age, another workspace, not pushed). The file was downloaded again here and its SHA-256 matches the value recorded in that snapshot.

`CORR-DATA_Database.zip` is the unchanged archive from the NIST Open Data record.

- Dataset: NIST CORR-DATA, Data.gov / NIST identifier 54AE54FB37AC022DE0531A570681D4291851 (https://catalog.data.gov/dataset/corr-data).
- Download: https://opendata.nist.gov/1851/CORR-DATA_Database.zip
- Licence: NIST Open Data (https://www.nist.gov/open/license).
- Archive size: 354,591 bytes; SHA-256 `013d9c5d7d4b9433d8a679c8b793d17f487463e0472b466d289450290d0da95c`. ZIP integrity check passes.
- Contents: `CORR-DATA_Database.csv` (5,506,050 bytes; over 24,000 historical observations), `columnDefinitions.txt` and `about-CORR-DATA_Database.txt`. The records draw on more than 250 source documents; the programme ended in 1997.

Keep the raw archive unchanged. The records mix numeric rates with categorical ratings, concentration units and conditions vary, temperatures and exposure durations can be ranges or text, and "Reference #" links rows to source documents. Parse these fields explicitly and group train/test by source document (and material where feasible). The NIST description says an observation is indicative of one material under one set of conditions, not advice or a guarantee of service safety. A corrosion model may be presented only as a screening aid for material selection and protection.
