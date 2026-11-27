# Thermoelectrics

> **Status: plan only (2026-11-27). Nothing downloaded, no model trained, nothing on the site.** Two families,
> never pooled: DFT-calculated transport values for crystals already on disk, and measured values versus
> temperature that still have to be fetched. Facts come from the category research of 2026-11-27 (materials) and
> its independent check, called "the verifier" here; where the verifier corrected the research, this page uses the
> correction. "Counted here" means counted for this plan from a file in this repository.

## 1. The expensive outcome
- **Predicted:** Seebeck coefficient S and power factor of a crystal (DFT-calculated). For the measured family: S,
  electrical conductivity, thermal conductivity and zT versus temperature.
- **Why it is slow:** each sample is synthesised and densified (for example by hot pressing), then S and electrical
  conductivity are measured over temperature and thermal conductivity by laser flash. A transport calculation per
  crystal is costly too. This is general background in the category research, not from a cited source.

## 2. Dataset
No `../../research/references/thermoelectrics.md` exists yet; create it with these entries before the first download.

| Source | Measured or calculated | Size | Licence as found | Source ID per row | Still unchecked |
| --- | --- | --- | --- | --- | --- |
| JARVIS-DFT `dft_3d` (9-24-2025), cached at `gpu/data_cache/jarvis/jdft_3d-9-24-2025.json.zip` | DFT-calculated; BoltzTraP with a constant relaxation time (a search confirmed JARVIS uses it; no field definition was found) | 23,218 of 93,902 crystals have all six fields `n-Seebeck`, `p-Seebeck`, `n-powerfact`, `p-powerfact`, `ncond`, `pcond`, in 12,632 chemical systems (counted here; 23,218 matches the research and the verifier) | CC BY 4.0, from the repository's provenance file only (figshare 10.6084/m9.figshare.6815699; figshare itself was not reached) | JARVIS id; the `reference` field names the source structure: a Materials Project, AFLOW or Alexandria id for 72,560 of the 93,902 records; the rest carry JQE- (11,279), JVASP- (4,569), bare-number (3,629, origin unknown), mvc- (1,565, origin not checked) or 300 other labels (counted here; data research on crystals and its verifier) | units of each field; what `nkappa` and `pkappa` hold. They are believed to be the electronic part only `[UNVERIFIED - from memory]`, so never use them as thermal conductivity |
| Ricci et al. 2017, *Sci. Data*, doi 10.1038/sdata.2017.85 | DFT-calculated, same method | about 48,000 Materials Project compounds | not checked; the data are in Dryad | Materials Project id | licence; overlap with JARVIS |
| ESTM, Na and Chang 2022, *npj Comput. Mater.* 8:214, doi 10.1038/s41524-022-00897-2; files in github.com/KRICT-DATA/SIMD | measured | 5,205 observations of 880 materials (S, electrical and thermal conductivity, power factor, zT) | none stated. Its README says the repository holds only part of the Starrydata data because of Starrydata's licence, so reuse terms are unclear. Only the article is CC BY 4.0 | not checked | licence; a paper ID per row |
| Sierepeklis and Cole 2022, *Sci. Data* 9:648, doi 10.1038/s41597-022-01752-1; figshare 10.6084/m9.figshare.19658787 | measured values text-mined from papers, precision 82.25% | 22,805 records, 10,641 chemical names, temperature in K | not stated in the article | not checked | licence |
| Starrydata2, figshare record 30747185; github.com/starrydata/starrydata_datasets | digitised measured curves | not settled. The research's 53,000 samples was not confirmed; the NIMS page (by search) lists 104,823 samples from 17,326 papers; the 2025 *STAM Methods* paper (doi 10.1080/27660400.2025.2506976) reports over 82,000 samples from over 13,000 publications | unresolved: the NIMS page footer says "All rights reserved", the paper record says open for commercial and non-commercial use | not checked | licence. Perplexity round 2 (`../../research/perplexity/round2/computer_mode_answers.md`, an unverified lead) reports duplicated `sample_id` values in files published 2026-04-01 to 2026-09-08, to be joined on (SID, sample_id) |

Traps, counted here unless noted:
- The sign is not fixed by the field name: 6,713 of the 23,218 `n-Seebeck` values are zero or positive, and 6,596
  `p-Seebeck` values are zero or negative. Settle the sign convention before choosing a target.
- No power factor is negative, but 11 `n-powerfact` and 10 `p-powerfact` values are exactly zero. A log10 target
  drops them and says so.
- 57,345 JARVIS records cite a Materials Project id (data research on crystals, reproduced by its verifier). Any
  added DFT set, Ricci's included, is matched on `reference` and kept on the same side of the split.
- [Speculative] ESTM rows include dopant series of one host from one paper, which would be near-copies; nobody has
  opened the files yet. Group by host system.

## 3. Candidate space
- **DFT family:** the 70,684 JARVIS crystals with no transport values (93,902 minus 23,218, counted here). They are
  already relaxed, so no new structure has to be built. This screens known crystals; it does not design new ones.
- **Measured family:** vary dopant level and composition only within the range ESTM covers for that host, and
  temperature only within the range measured for that host.

## 4. Models
- **Expected value:** the crystal graph network in `gpu/structure/train_structure.py`, with new `TARGETS` entries
  for `n-Seebeck`, `p-Seebeck` and log10 power factor, against the strongest tree model chosen on validation rows
  (DEC-004, DEC-014). Measured family: trees and the table network (`gpu/train_tabular.py`) on composition, dopant
  level and temperature.
- **Cautious estimate:** conformal interval (DEC-017); its lower edge for power factor or for the size of S.
- **Held out by entity:** whole chemical systems (DEC-014) and whole elements (DEC-016) for the DFT family; whole
  host systems and whole source papers for the measured family (DEC-009).
- **Metrics:** MAE in the field's unit (log10 for power factor) with the 95% paired-bootstrap interval over whole
  groups (DEC-012).
- **Prior evidence, for another target:** on the DFT-calculated OptB88vdW band gap with whole chemical systems held
  out, the network average was clearly better than the strongest trees, -0.116 [-0.134, -0.0951] eV, and each of the
  2 runs was clearly better (exploratory: these test systems were scored before under the original neighbour rule;
  `docs/results/20261122T014943Z_overnight_REPORT_20261126T2001.md`). It says nothing yet about Seebeck coefficient.

## 5. Cost objective
No element price or scarcity number is sourced in this repository (`research/references/materials.md`: "No entries
yet"). The first version ranks by the cautious estimate, as the live alloy page does (`src/lib/alloyGen.js`). A user
filter that removes chosen elements needs no number. A price or supply-risk term waits for cited values.

## 6. Neighbour check
- **DFT family:** the 3 nearest training crystals by L1 distance on element fractions, same space group where
  possible. These are calculated values, so the page says "calculated neighbours agree", never "neighbours passed".
- **Measured family:** the 3 nearest measured samples of the same host at the nearest measured temperature.
  "Neighbours passed" only when at least 2 of them reached the target (the rule in `src/lib/alloyGen.js`).

## 7. First step, and what would stop it
- **First step:** add the three targets to `TARGETS`, read only those fields, and run the held-out-systems and
  held-out-elements splits against the strongest trees. In parallel, open the ESTM files and count rows, hosts and
  papers.
- **Stop the DFT family** if, on held-out systems, neither the network nor the trees is clearly better than guessing
  the average, or if the units and sign convention cannot be confirmed from JARVIS documentation.
- **Keep the measured family out of the repository** until a licence allows it: ESTM carries Starrydata rows, and
  Starrydata's terms are unresolved.

## Conclusions
- [Certain] 23,218 crystals with DFT-calculated Seebeck coefficient and power factor are on disk, CC BY 4.0 according
  to the provenance file, and the crystal training and tree scripts take new targets through one table (`TARGETS`);
  the nightly pipeline also needs per-target entries in `gpu/pipeline/plan.py`.
- [Likely] This is the cheapest new family to start: no download, and the split and baseline code exist.
- [Speculative] A constant-relaxation-time Seebeck screen picks crystals that also measure well. No comparison with
  measured values has been made.

---
**Ground rules:** predictions labelled as predictions; every non-learned number cited; the tool
narrows the experiment list, it does not replace the experiment; chemistry & materials only, never
biology. See `../../docs/AGENTS.md`.
