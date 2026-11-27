# Perplexity round 2: what it settled

Round 2 ran the two Computer-mode prompts (version 2) on 2026-11-27 until the account's Computer credits ran out. Its
sandbox could read web pages but could not download files, so it read record pages and licence lines only. The raw
answers are in `round2/computer_mode_answers.md`. Below: what is now settled, what the project checked itself, and what
is still open.

## Settled

| Item | Finding | How we know |
| --- | --- | --- |
| Polymer training file | Our `experimental_dataset.csv` is the file in Zenodo 10.5281/zenodo.19242815 (Schrodinger, 2026), CC BY 4.0 | Perplexity read the record's MD5 (11eb3b96be3116907d87038d9b74399f); the project computed the same MD5 on its copy. [Certain] |
| PoLyInfo terms | "No reproduction, republication or distribution to third parties of any content is permitted without written permission of NIMS"; bulk acquisition is prohibited | Quoted from https://polymer.nims.go.jp. The separate MatNavi terms page was not read. [Certain] for the quotes |
| GREA withdrawal | "We delegated the polymer datasets for GlassTemp, MeltingTemp, and PolyDensit as requested by the NIMS Materials Database, MatNavi." | Quoted from the GREA README. [Certain] |
| PolyMetriX Tg | Zenodo record licence CC BY 4.0; the record says nothing about PoLyInfo rights | Record page. The 7,170 GREA rows stay out (see above). [Likely] |
| SciGlass | The LICENSE file is headed ODbL with MIT-style text; the SciGlass site says ODbL (2019) | Agrees with our own verifier. Treat as ODbL share-alike. [Likely] |
| Liquid electrolytes, second source | Zenodo 10.5281/zenodo.7244939: 5,035 conductivity experiments on Li-ion electrolyte formulations at several temperatures (Rahmanian et al., 2022), CC BY 4.0, one 39.2 MB CSV | Record page. Measured or not, and overlap with CALiSol-23, not checked |
| CALiSol-23 | "13,825 individual data points from 27 different experimental articles" | DTU record page; its licence line was not found in the page text |
| Hydrogen storage | ML-HydPARK (Zenodo 10680097) is CC BY 4.0 | Record page |
| OBELiX | CC BY 4.0; its README lists "Laskowski ID" and "Liion ID" columns, which our processed table lacks | Repository README. So the upstream release can be de-duplicated against LiIon and Laskowski. |
| Liverpool LiIon | Commercial extraction or re-use needs a licence from the University of Liverpool | Quoted from the database page |
| Starrydata2 | Releases from 2026-04-01 to 2026-09-08 have duplicated sample_id values; join on (SID, sample_id); no licence line in the README | README |
| Polymer Tg by DMA, noise | Between-laboratory reproducibility limits 15.85 to 25.69 °C over 5 to 7 sites (a lower 4.4 to 7.1 °C in a second table); within-laboratory 2.92 to 5.94 °C | NPL good-practice guide, secondary. Compare: our polymer models miss by about 15 to 16 K on average |
| Concrete cylinders, ASTM C39 (2005 text) | Single-operator CV 2.4% (laboratory), 2.9% (field); multi-laboratory CV 5.0%, for 150 x 300 mm cylinders | Unofficial copy, secondary |

## New leads (not checked)

- The McHaffie et al. 571-compound lithium-conductor set: https://data.caltech.edu/records/23mvv-6gk43 (snippet).
- A 2026 paper with 712 Li and 259 Na room-temperature conductivity records:
  https://www.sciencedirect.com/science/article/pii/S2352492826007063 (snippet; data availability unknown).
- LiCondAI supplement with temperature columns: https://pubs.acs.org/doi/10.1021/acs.iecr.5c01315 (snippet).

## Still open

Everything that needs a file opened: row counts, overlaps, headers, and every V4 concrete record (which stays sealed).
On 2026-11-27 the project's own cloud environment was given access to the data hosts these checks need (Zenodo,
figshare, Mendeley Data, DTU Data, Dryad, Dataverse, Caltech Data, DigBat), so the file-level checks can now be run by
the project directly instead of through Perplexity.
