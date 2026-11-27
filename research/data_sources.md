# More training data: sources for each trained family

Written 2026-11-27. Question: where can each family the project already trains get more data, and which of it is
safe to use?

How this was found: research agents searched free tools on 2026-11-27, opened the data records they could reach, and
measured overlap against this repository's files. A second agent checked every finding the same day. Where the check
corrected a finding, the corrected version is used here. The reports are kept outside the repository; their names are
given in each section (`research_data_measured_1`, `research_data_measured_2`, `research_data_dft`,
`research_categories_materials`, `research_physics_lit`, each with its `verify_*` check). Perplexity's round-1 answers are
in `research/perplexity/round1/` and summarised in `research/perplexity/ROUND1_DIGEST.md`; every Perplexity item below is
marked "Perplexity lead, not checked". Nothing in this file has been trained on. `docs/DATA_ROADMAP.md` describes the
sources already in use.

## The hidden assumption

"More data is always better" holds only if every new row measures the same quantity, the same way, on a sample that is
independent of the rows we have, inside the range we predict, and if we may use it. Each part fails for some source in
this file. New data hurts in five ways.

1. **Duplicates.** The same measurement enters twice, often once on each side of a split, so the test score rises while
   the model does not improve (LES-001; MD-HIT, https://doi.org/10.1038/s41524-024-01426-z). Found here: 122 rows of the
   SCM concrete set cite Yeh 1998, the UCI source; Tan's concrete set contains the 1,030 UCI rows and the 103 UCI slump
   rows (by its abstract); every one of the 109,254 PSDI LogS records comes from SolCuration or ESOL; the Toni Tg set
   holds all 7,367 PolyMetriX structures; 57,345 of the 93,902 JARVIS records are Materials Project structures relaxed
   again. Literature compilations also share source papers, and in literature-mined tasks a model given only the
   author, journal and year predicted from the descriptors matched the full model on one task (accuracy 0.900 against
   0.899) and came close on another (0.901 against 0.923), though not on battery capacity ("Clever Materials", arXiv
   2602.17730), so splits must hold out whole papers.
2. **Same name, different measurement.** Cube and cylinder strength differ by a factor of 0.65 to 0.90 that changes with
   strength (Perplexity X3, not checked). Nanoindentation hardness read 10 to 30% above microhardness in one study
   (Perplexity X3, not checked). One surfactant's CMC was 1.8 mM by surface tension and 2.7 mM by fluorescence (Perplexity
   X3, not checked). Computed on our cached JARVIS file: its shear moduli are Voigt averages, a median 1.072 times the
   Voigt-Reuss-Hill value that Materials Project and AFLOW publish; on 645 shared crystals the JARVIS and Materials
   Project dielectric constants differ by more than 25% for 27.6% (crystal section). Kinetic and thermodynamic
   solubility, bulk and total ionic conductivity, and immersion and atmospheric corrosion rates are also different
   quantities under one name.
3. **Computed labels sold as data.** The Open Polymer Challenge Tg labels are molecular-dynamics results (arXiv
   2512.08896, full text checked); DieDatabase permittivities are model predictions (Perplexity F8, not checked);
   AlloyManufacturingNet holds network predictions; PolyAskInG holds over 6 million QSPR-computed Tg values beside 214
   measured ones. Values a language model extracted from papers (the PSDI CMC table, the precipitate alloy set, the
   5,597-paper inhibitor set) are measurements copied by a machine and need spot checks. DFT values are calculations,
   not measurements (DEC-002).
4. **Rights.** A permissive licence on a compilation does not clear the rows it copied. PolyMetriX is CC BY 4.0, but
   7,170 of its 7,367 Tg rows come from GREA, whose README says its Tg data were withdrawn at the request of the NIMS
   materials database (PoLyInfo). Others found here: LiIon (academic use only), AFLOW and GNoME (non-commercial), the
   Blended Cement Concrete Database (CC BY-NC, Perplexity lead), BASF-owned CMC sets, CORDATA (contributors only), ULTERA
   (no licence). The team spends no money on data (DEC-005), and the site is public.
5. **Extrapolation.** Rows from another regime (dilute Ti alloys, ultra-high-performance concrete, alkali-activated
   binders, Cu alloys) change what the model is asked to do, and pooled without a flag they may pull predictions away
   for the regime the family serves. For truly new chemistry, more data did not help: in one study the error with all
   hydrogen compounds held out rose 5x as the training set grew from 10^4 to 10^6 entries (ALIGNN; arXiv 2406.06489,
   checked in the full text). Forest models average training values, so they cannot predict above the largest one, and
   in one benchmark no baseline, random forest included, went beyond the training range (arXiv 2502.05970).

Where more rows of the same kind buy least:

- **Crystals.** Against 54 measured band gaps, the JARVIS OptB88vdW gap is 1.355 eV too low on average, every value an
  underestimate (computed on the cached file and checked twice; crystal section). The crystal network misses the DFT
  label by about 0.20 eV (held-out systems, fixed neighbour rule, mean of single runs, 0.2038 eV;
  `docs/results/20261122T014943Z_overnight_REPORT_20261126T2001.md`, "the report" below). [Certain] More OptB88vdW gap
  rows cannot remove a label error of about 1.4 eV on these materials.
- **Water solubility.** Typical laboratories differ by 0.5 to 0.6 log units (Llompart et al. 2024,
  https://doi.org/10.1038/s41597-024-03105-6); the network average misses by 0.745 (the report). Cleaning AqSolDB
  lowered RMSE by about 0.10 log in that paper (10-fold cross-validation within each set: random forest 0.86 to 0.78,
  so part of the drop may come from cleaner test labels). [Likely] Cleaner labels help this family more than more rows.
- For concrete and solubility, our errors are above the spread between laboratories: concrete cylinder strength about
  5% coefficient of variation (Perplexity, from unofficial copies of ASTM C39, not checked) against 9.07 to 11.50 MPa
  RMSE on the spent 56-to-365-day test (LES-006); solubility 0.5 to 0.6 log (Llompart et al. 2024) against 0.745 MAE.
  For Tg, Perplexity gives only DSC repeatability on one sample, 2 to 5 °C (not checked), far below our 14.56 K
  (`research/perplexity/ROUND1_DIGEST.md`, point 6). [Likely] Clean, independent rows should still help concrete and
  solubility. [Speculative] The same for Tg: Perplexity says the spread between sources is far larger than the
  instrument's, and that spread was not measured here.

## Merge rules

Every new source passes all six before it trains anything.

1. **Open the record.** Download the data file itself, not a landing page. Write its URL, version or DOI, date, size and
   SHA-256 in a README next to the file, as `domains/*/work/data/README.md` do. Count the rows yourself. A count from an
   abstract, a search snippet or Perplexity is a lead.
2. **Quote the licence.** Copy the licence sentence from the record into that README, with where it was read and who read
   it. Non-commercial, academic-only, "on request" or no licence: hold the source until the person decides. Check what
   the record says about the sources it copied.
3. **Remove overlaps.** Before any split, match the new rows against every row the family has, training and test, by the
   family's key (table below). Drop exact copies. Put near-copies and rows from a paper we already hold in the existing
   group, so a mix, alloy, compound or paper never sits on both sides (DEC-003, DEC-009). Report the counts.
4. **Harmonise definitions.** One target is one quantity: same unit, specimen, method, averaging, functional and
   conditions. Add a source column and a flag for measured, DFT-calculated or machine-extracted. Where a conversion
   ratio varies (cube to cylinder), keep two targets instead of converting with one factor. Never fill a missing amount
   with 0 (Nyby et al. say missing major elements "should not be replaced with zeros"). Read named fields only: take
   JARVIS labels by an explicit list of field names, never "every label in the file", because the file carries a field
   outside this project's scope (`verify_data_dft`).
5. **Re-score with grouped cross-validation.** Use the family's existing grouped folds (DEC-009) with the new source's
   papers as extra groups, the tuned tree opponent (DEC-004, DEC-012) and the paired bootstrap over whole groups. Report
   the error on the original family's held-out groups with and without the new source, and the error on the new source
   with it held out. Write the keep-or-drop condition down before the run (DEC-019). Never decide on a fixed test split.
6. **Keep sealed test candidates unseen.** If a source could give a fresh test, the first script that opens it splits
   those rows off by a rule written down before opening (concrete: every row older than 28 days; alloys: a whole source),
   writes them to a sealed file with a SHA-256 and prints counts only, as `lab/make_splits.mjs` did for the vault
   (DEC-006). They are scored once, against models named in a pre-registration (DEC-008). Concrete late-age rows matter
   most: the 271-row vault was opened on 2026-11-21 and is spent (LES-006), so new late-age measurements are the only way
   to test long-age predictions again. No agent report shows target values from the Tan, Ponnu Durai, GE or Cu files
   (never opened), and Perplexity opened only the landing pages of the Pavement and Blended Cement records; for the
   alkali-activated set only counts per age were looked at (`research_data_measured_1`, `ROUND1_DIGEST.md`).

Adding a source changes a family's rows and folds, which is a change of method: ask the person first and add a row to
`docs/DECISIONS.md`. A new data file also changes the task's hash, so the nightly run retrains that task (DEC-018).

| Family | Match new rows on |
| --- | --- |
| Concrete | the 7 ingredient amounts and age, with the 5 kg rule (DEC-003); source DOI |
| Alloys | composition in atomic % (convert mass %), processing route, source DOI |
| Polymers | RDKit canonical SMILES with end markers as `*`; copolymer pair and fraction; DOI |
| Water solubility | first block of the InChIKey (the family's groups) |
| Surfactants | first block of the InChIKey, temperature, additive |
| Corrosion | material (UNS number where given), environment, concentration, temperature, reference |
| Solid electrolytes | composition string and DOI (our `processed.csv` has no LiIon ID column; Perplexity reports one in the upstream OBELiX release, not checked) |
| Insulators and crystals | JARVIS `reference` field (mp-, auid-, agm ids); structure matching where a file has no id |

## How to read the tables

- **Licence (who):** as found, then who read it. R = our research agent, V = our verifier, P = only Perplexity, repo = a
  file in this repository.
- **Verdict:** add; test set only (used only to score models, never trained on and never copied into the repository;
  where its licence is unread or absent, read it or ask the person before any score is published); hold for rights (a
  source we would train on whose licence is unread, or forbids our use); exclude.
- "Not checked" means no report behind this file opened that part. Several hosts (data.mendeley.com, figshare,
  nature.com, mdpi.com) were blocked for the agents, so many licences are still unread.

## Concrete

Sources: `research_data_measured_1` (R), `verify_data_measured_1` (V), Perplexity F1.

**Trains on now.** UCI Concrete Compressive Strength (Yeh 1998): 1,030 rows, 25 exact repeats dropped, 1,005 left; the
live page trains on 786 and tests on 219 from 62 mixes (`README.md`). The lab and GPU kit train on 253 rows at 1 to 7 days
and select on 481 rows at 14 and 28 days, split into REWARD 321 and MONITOR 160 (`lab/README.md`, DEC-007). The 271 rows
at 56 to 365 days were the sealed test and are spent (DEC-006, LES-006). Pre-training uses the 395 BOxCrete concrete rows
cured at 22 C, ages 1 to 28 days, MIT (`lab/external/LICENCE.md`; recounted for this file).

| Name | Link | Licence (who) | Size | Measured or computed | Adds | Overlap with ours | Verdict | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCMs Dataset, cylinder strength (Ünverdi 2026) | https://doi.org/10.5281/zenodo.24108549 | CC BY 4.0 (R, V) | 1,006 rows, 67 reference strings | measured, 28-day cylinders | silica fume column; mixes from 65 studies by its description, Yeh 1998 among them | 122 rows cite Yeh 1998; 4 cite Tan & Pu 1998, not Yeh; 5 exact duplicates; about 71 rows repeat a mix with another strength (V) | add | 28-day only, so no seal issue. Fill the reference down before grouping. Score the current 28-day models on its non-Yeh rows once, as a cross-lab check, then train on it. |
| Multi-source Modified Concrete (Tan 2026) | https://doi.org/10.17632/3ph7nk4jzz | not checked (R, V) | 2,578 mixes with curing age (abstract) | measured, 41 papers plus own work | silica fume, desulfurised gypsum, ages, durability targets | contains Yeh 1998 (1,030) and Yeh 2007 slump (103) by its abstract; about 1,445 left by subtraction, before other overlaps | hold for rights | Best candidate for a new late-age test. If it holds all 1,030 UCI rows, as its abstract says, it holds the spent vault rows too, so the opening script drops Ref.3 and Ref.11 before anything is printed. Its "identical test protocols" claim is implausible for 41 papers (V). |
| Alkali-Activated Concrete Dataset (2026) | https://doi.org/10.5281/zenodo.24089584 | CC BY 4.0 (R, V) | 1,630 mixes; 3,154 strengths, 330 at over 28 days | measured; its CO2 column is estimated | a different binder family | none expected with UCI or BOxCrete; 67 mixes contain Portland cement (V) | exclude | Different chemistry: a family of its own, if ever. Cube and cylinder columns are separate; oven curing is not age. |
| Non-destructive estimation databases (Matthews et al. 2026) | https://doi.org/10.5281/zenodo.23988578 | CC BY 4.0 (R, V) | 16,531 tests, 115 studies | measured, digitised from figures | age and w/c trends; size normalisation | none | exclude | No kg/m3 mix amounts, so it cannot feed the mix model. Perplexity saw an older version, 10.5281/zenodo.14921019. |
| UCI Concrete Slump Test (Yeh 2007) | https://archive.ics.uci.edu/dataset/182/concrete+slump+test (DOI 10.24432/C5FG7D) | CC BY 4.0 stated on the UCI page (V; R had "none") | 103 rows | measured, 28-day | same 7 inputs | inside Tan's set | add | Small and clean. Never add both it and Tan's Ref.11 rows. |
| UHPC dataset (Kashem 2023) | https://doi.org/10.17632/85r7bh4zsz | not checked (R, V) | 810 samples, 14 attributes | measured, compiled | ultra-high-performance regime | other UHPC sets reuse papers | hold for rights | A separate family; check shared papers with the UHPC lead below. |
| Normal concrete, plant ledger (Qiu 2024) | https://doi.org/10.17632/zrsbhndz9f | not checked (R, V) | row count not stated | measured, production records | accelerator, silica fume, curing time | not checked | hold for rights | One design may repeat many times: group by design. |
| GGBFS, Alccofine and metakaolin study (Ponnu Durai 2024) | https://doi.org/10.17632/m8557m359k | not checked (R, V) | not checked | measured, one lab | ages 7, 28, 56 and 90 days | none expected | hold for rights | Seal its 56 and 90 day rows on opening. One study is one group. |
| ConcreteXAI (Guzmán-Torres et al. 2024) | https://github.com/JaGuzmanT/ConcreteXAI | MIT (R, V) | 4,420 rows, 12 mixtures | measured; augmentation unresolved | ages 3 to 120 days | none | exclude | No kg/m3 amounts; only 12 groups; its README names an "artificial data" file. |
| BOxCrete rows not used yet | `lab/external/boxcrete_data.csv` | MIT (repo) | 5 concrete rows at 10 C; 270 mortar rows | measured | cold curing | already in the repo | exclude | Mortar is not concrete; the 5 cold rows need a temperature input first. |
| CO2-cured concrete | https://doi.org/10.17632/2myr3k8n4g | not checked (R, V) | 198 samples (abstract) | measured; concrete and mortar mixed (V) | carbonation curing | not checked | exclude | A different curing process. |
| CatBoost-NSGA-II artifacts | https://doi.org/10.5281/zenodo.22746748 | CC BY 4.0 (R, V) | 1,867 rows | sources called "proprietary" | nothing usable | 563 duplicates | exclude | Total binder never equals the sum of the binders; silica fume and slag are 0 in every row; placeholder repository link (V). |
| Hugging Face re-uploads | https://hf.co/datasets/foundry-ml/dataset_concrete_compressive_strength and electricsheepafrica/africa-synth-concrete-compressive-strength-comoros | "other"; cc-by-4.0 (R, V) | 1,030; 10K to 100K | UCI copy; synthetic | nothing | the first is UCI | exclude | Do not open the UCI copy: it repeats the late-age rows that `CLAUDE.md` keeps agents from reading. |
| Perplexity lead, not checked: Blended Cement Concrete Database (2026) | https://doi.org/10.5281/zenodo.18080351 | CC BY-NC 4.0 (P) | 8,979 records, 260 papers (P) | extracted by a language model, over 80% checked by hand (P) | curing temperature and humidity; a DOI per row | may re-publish UCI (P) | hold for rights | Non-commercial; ages never seen; seal its late-age rows on opening. |
| Perplexity lead, not checked: Pavement Concrete Datasets (2026) | https://doi.org/10.17632/pz38ys9pdx.1 | CC BY 4.0 (P) | 1,430 strength records (P) | measured, agency reports (P) | transport-agency mixes | not stated | hold for rights | Licence not read by us; ages never seen; seal its late-age rows on opening. |
| Perplexity lead, not checked: UHPC mix-design database (2026) | https://doi.org/10.17632/7wk5nd9jz8.1 | CC BY 4.0 (P) | 1,809 formulations, about 68% with a strength (P) | extracted from 361 papers (P) | specimen geometry per row; curing regime | same class as Kashem | hold for rights | Separate regime. |

Also seen by Perplexity only: a Zenodo "Concrete Strength Dataset (2013-2025)" (catalogue snippet) and a curing-temperature
set on Taylor & Francis figshare (page did not load). Both are unopened leads.

## Alloys (Vickers hardness)

Sources: `research_data_measured_1` (R), `verify_data_measured_1` (V), `research_categories_materials` and its check,
Perplexity F2.

**Trains on now.** `domains/materials/work/data/MPEA_dataset.csv` (Borg et al. 2020): 1,545 rows, 530 with a Vickers
value (recounted), 377 distinct alloys; 426 training and 104 held-out rows (`README.md`). Licence: Apache 2.0 by
`domains/README.md`; `README.md` says "not confirmed"; the Yadav record below states that the upstream figshare
10.6084/m9.figshare.12642953 is CC BY 4.0 (V). Cross-fitted: trees 70.9 HV, network average 75.9 HV, no clear difference
(+5.05 [-1.39, +12]; report).

| Name | Link | Licence (who) | Size | Measured or computed | Adds | Overlap with ours | Verdict | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dax-ti, Ti-alloy compilation (Salvador et al. 2022) | https://doi.org/10.5281/zenodo.6089749 | CC BY 4.0 (R, V) | 283 rows; 152 with HV (134 to 560 HV) from 47 papers | measured; a third blind-reviewed | Ti alloys with heat treatment | 1 DOI on HV rows, 3 over all rows (V) | add | As its own group: dilute Ti (about 73 to 86 at.% Ti in the first rows read) is outside the high-entropy range. "ST-*" needs a new route or a recorded mapping. |
| GE refractory alloy screening (Detor et al. 2022) | https://doi.org/10.6084/m9.figshare.17032760 | not checked (R, V) | 82 arc-melted alloys (86 rows by a snippet, not checked) | measured hardness, one lab; solidus computed | a blind test | published after Borg; compositions may repeat | hold for rights | Then test set only: one lab and one protocol. The Vickers scale is not confirmed. |
| Cu-based alloys (Gorsse et al. 2023) | https://doi.org/10.1038/s41597-023-02411-9 | not checked (R, V) | about 150 alloys, about 1,830 records; hardness count not checked | measured, literature | heat-treatment times and temperatures | the Mendeley Cu-Ni-Si set 10.17632/7tdd6vngzm is a subset (V) | hold for rights | Mass % to atomic %; dilute alloys, own group. |
| Laser powder bed fusion alloys (Dovgyy et al.) | https://doi.org/10.5281/zenodo.6514897 | CC BY 4.0 (R, V) | 2,187 rows; 398 with HV, 50 high-entropy type | measured | printed route and parameters | none of its 48 DOIs on HV rows is in our file (R, V) | add | The 50 high-entropy rows only, for now. Commercial alloys are named, not given as compositions, and would need a cited composition table. |
| Precipitate-containing MPEA set (Raj et al. 2026) | https://doi.org/10.5281/zenodo.18021833 | Zenodo CC BY 4.0 (R, V); article CC BY-NC by a search, not checked | 396 rows; 107 hardness cells: 33 bare numbers, most others text such as "216 ± 2 HV", 4 in GPa (V) | extracted by a language model, reviewed by experts | numeric processing history | with Borg not checked | add | Only hand-parsed HV values; drop the GPa values. |
| ULTERA (Krajewski et al.) | https://doi.org/10.5281/zenodo.15716432 | no licence, no files (R, V) | 12,540+ points, 4,270+ alloys (May 2025 alpha); an older record 7566416 says 6,200 points and is restricted | measured, extracted | the largest pool | probably large with Borg | hold for rights | Nothing to download. |
| As-cast HEA Vickers set (Yang et al., Acta Mater. 222, 117431, 2022) | https://github.com/dovahkiin0022/representations | not checked | 370 alloys, as-cast HV (V, via arXiv 2301.00179) | measured | the same quantity as ours | not checked | hold for rights | Possibly the closest match to our target; read the repository licence first. |
| Gorsse et al. 2018 | https://doi.org/10.1016/j.dib.2018.11.111 | not checked | 370 alloys | measured | nothing new | Borg 2020 is built on it (V) | exclude | Use only to cross-check entries. |
| MPEA leakage-aware tables (Yadav 2026) | https://doi.org/10.5281/zenodo.23491016 | CC BY 4.0 (R, V) | yield 1,007, phase 812; no hardness | derived from Borg | a duplicate-finding method | Borg | exclude | Not new data; reuse its clustering to find near-duplicates among our 530 HV rows. |
| Off-target records | AlloyManufacturingNet 10.5281/zenodo.10851804; steel grades 10.5281/zenodo.17704410; Al-Si 10.5281/zenodo.14249076; polyGNN 10.5281/zenodo.7975920; EN AW-2618A 10.5281/zenodo.10396823; foundry-ml steel_strength_v1-1, dataset_rpv_tts, dataset_mg_alloy | various (R, V) | various | predictions; specification ranges; tensile strength; load curves; Brinell; yield; embrittlement shift | nothing for HV | none | exclude | Not Vickers hardness of a stated composition. The Mg set is also outside scope (DEC-001), as the check flagged. |
| Unopened leads | Wen et al. 2019, 10.1016/j.actamat.2019.03.010; github.com/Mat-Design-Yu/SREF | not checked; SREF shows no licence | not checked | not checked | not checked | not checked | hold for rights | Open before judging. |
| Perplexity lead, not checked: steel tempering data (Raiipa, Kaggle) | Kaggle page, no DOI (Perplexity F2) | CC BY 4.0 (P) | 1,466 tests (P) | measured, old literature (P) | steels, tempering time and temperature | none | exclude | Rockwell C, not Vickers; 82 rows were converted outside the conversion table's valid range (P). |
| Perplexity leads, not checked: tempering record, steel set, PMD records | 10.5281/zenodo.23215479 (snippet only); 10.5281/zenodo.18914884 (17.5 GB of arrays); Mendeley 10.17632/jmwb9ddd43.1 (yield, not hardness); PMD Vickers records (a Cu alloy) | CC BY 4.0 for 18914884 and jmwb9ddd43.1; others not stated (P) | not stated | not stated | not stated | not stated | hold for rights | Unopened; most look off-target. |

Perplexity found no hot-hardness dataset.

## Polymers (glass transition)

Sources: `research_data_measured_1` (R), `verify_data_measured_1` (V), Perplexity F3.

**Trains on now.** `domains/polymers/work/data/experimental_dataset.csv`: 666 rows, 64,139 bytes, SHA-256 starting
`e4ce0b9b` (rechecked); Zenodo 10.5281/zenodo.19242815 (Bhati et al.), CC BY 4.0 by `domains/README.md`. By the paper, as
Perplexity reports it, 315 homopolymers and 351 binary copolymers (F3). Cross-fitted MAE: trees 16.07 K, one network
15.0 K, average of 5 networks 14.56 K. The average of 5 networks is clearly better than one tree model (-1.51 [-2.89,
-0.0985] K); one network at a time is clearly better in 0 of 5; against the average of the 3 best tree models, no clear
difference (-0.473 [-1.8, +0.886] K) (report). The error is about 6 K on copolymers and 30 K on homopolymers
(`README.md`).

| Name | Link | Licence (who) | Size | Measured or computed | Adds | Overlap with ours | Verdict | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PolyMetriX curated Tg, non-GREA rows (Kunchapu & Jablonka 2025) | https://doi.org/10.5281/zenodo.15210035 | CC BY 4.0 (R, V) | 197 rows: Qiu 152, Xie 21, Schrodinger 15, Nguyen 7, Uchicago 2 | measured, compiled | 182 rows new to us (R; 183 by V) | the 15 Schrodinger rows are in the Bicerano file | add | Their original sources were not traced. Homopolymers only. |
| PolyMetriX, GREA rows | same record | CC BY 4.0 on the record, but traced to PoLyInfo (R, V) | 7,170 rows | measured, compiled | about 23x our 315 homopolymer rows (about 11x our whole 666-row file) | 191 to 203 of our 315 homopolymers appear, median difference 1.0 K, 15 to 17 differ by over 20 K (R, V) | hold for rights | The GREA README says its Tg data were withdrawn at the request of NIMS MatNavi. |
| OpenPoly benchmark (2025) | https://github.com/WangGroupFDU/Openpoly_benchmark | MIT (R, V) | 443 Tg rows, only 304 unique structures (V) | measured, each value a mean of many reports | about 146 unique structures new to us and PolyMetriX (V) | about 89 in our files (V) | add | Weight it below single measurements; merge name variants first. |
| CoarsePGNN frozen datasets (Wang 2026) | https://doi.org/10.5281/zenodo.22741277 | CC BY 4.0 data, MIT code; "do not replace rights held by third parties" (R, V) | 803 polyurethane Tg (in C), plus Tm and tensile data | measured; sources "not all independently revalidated" | the only formulation-aware Tg found | its homopolymer part comes from OpenPoly | hold for rights | Third-party notice; a new sub-family that needs formulation and method columns. |
| Toni, Mahmud, Karim 2026 | https://doi.org/10.5281/zenodo.23304692 | CC BY 4.0 (R, V) | 7,769 rows marked experimental | measured, compiled | about 400 rows beyond PolyMetriX | contains all 7,367 PolyMetriX structures, 7,126 with identical Tg (V) | exclude | Inherits the GREA lineage; the extra rows are untraced. |
| Uddin & Fan 2024 | https://doi.org/10.3390/polym16081049 | not checked | 7,174 samples | not checked | not checked | probably PoLyInfo lineage | hold for rights | Not traced. |
| Open Polymer Challenge (NeurIPS 2025) | https://arxiv.org/abs/2512.08896 | not found | 511 Tg labels (V) | computed (molecular dynamics) | nothing measured | not checked | exclude | Simulated. |
| Simulated or untraced Tg sets | https://doi.org/10.5281/zenodo.17401439; https://hf.co/datasets/OsBaran/polymer_tg_dataset | -; none | -; 7.8K rows | simulated (extracted by a language model); no provenance | nothing | not checked | exclude | Computed labels; no licence or source. |
| Unopened copolymer leads | PHB/PHBV copolyesters, 10.3390/polym18131559; PolyAskInG polyimides, 10.1021/acsomega.2c04649; solution SBR, 10.1002/mgea.78; Tao 2022, 10.1016/j.isci.2022.104585 | not checked | 118 Tg points; 214 measured Tg; 68 measured Tg | measured parts; PolyAskInG's 6 million QSPR values and the SBR paper's GAN rows are computed | copolymer composition | not checked | hold for rights | Abstracts checked (V); data access not checked. |
| Perplexity lead, not checked: computed files in our own source record | https://doi.org/10.5281/zenodo.19242815 (md_tg_data.csv, enumerated predictions) | CC BY 4.0 (P) | 50 MD values; about 148,365 predictions (P) | computed | nothing measured | same record | exclude | Keep apart from the measured file. |
| Perplexity leads, not checked: other Tg records | figshare 24219958 (Seok et al. 2024); a GitHub course compilation | not stated; CC BY 4.0 (P) | 29 kB; 7,372 rows (P) | not stated; mostly PolyMetriX again (P) | not stated | PolyMetriX | exclude | Re-publication, or unknown content. |
| Perplexity lead, not checked: PoLyInfo terms | https://polymer.nims.go.jp/en/ | no reproduction or distribution without NIMS's written permission; no bulk download (P) | not stated | measured | the large polymer source | lineage of the 7k-row sets | hold for rights | Ask NIMS in writing. |

Perplexity found no open copolymer or blend Tg record beyond the 351 copolymer rows we already have (F3).

Also in the repo, checked but not used: `HT_MD_polymer_properties.csv` (Bicerano, MIT; `domains/README.md`). It holds
computed and experimental columns side by side (R).

## Water solubility (LogS)

Sources: `research_data_measured_2` (R), `verify_data_measured_2` (V), Perplexity F4.

**Trains on now.** AqSolDB `domains/chemistry/work/data/data_curated.csv`: 9,982 rows (recounted), CC0; 7,982 training and
2,000 test rows, grouped by the first block of the InChIKey (9,671 groups; `domains/chemistry/README.md`). Cross-fitted
MAE: trees 0.774, one network 0.777, average of 5 networks 0.745 log units. The average of 5 networks is clearly better
than one tree model (-0.0284 [-0.0398, -0.017]) and than the average of the 3 best tree models (-0.026 [-0.0372,
-0.0151]); one network at a time is clearly better in 0 of 5 (report).

| Name | Link | Licence (who) | Size | Measured or computed | Adds | Overlap with ours | Verdict | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AqSolDBc and the OChem sets (Llompart et al. 2024) | https://doi.org/10.57745/CZVZIA (paper https://doi.org/10.1038/s41597-024-03105-6) | not checked: host blocked (R, V) | AqSolDBc count not found; OChemUnseen.csv holds 2,251 compounds (V) | measured | cleaner labels; a never-used external set | AqSolDBc is a subset of AqSolDB; 5,212 OChem compounds are shared | hold for rights | Then AqSolDBc as a cleaner variant, and OChemUnseen as test set only. Redo its pH 7 ionisation so the InChIKeys match ours. |
| BigSolDB 2.2 (Krasnov et al.) | https://doi.org/10.5281/zenodo.22648301 | CC BY 4.0 (V; v2.0 also R) | 125,752 values, 1,627 solutes, 221 solvents; the v2.0 water subset is 6,814 rows for 711 solutes (V) | measured; mol/L recalculated from mole fraction | temperature, 243 to 425 K | with AqSolDB not computed | add | Water rows with temperature as an input. Replicates from different papers differ by RMSE 0.39 logS. Leave out the rows our verifier flagged as borderline for scope (DEC-001); a column of the file marks them. |
| SOMAS (Gao et al. 2022) | https://doi.org/10.6084/m9.figshare.14552697 | article CC BY 4.0; data licence not checked (R, V) | 11,696 compounds | measured labels; computed descriptors | grade-1 sets 3 (149 from NIST SRD 106) and 5 (1,743 from eChemPortal) | expected to be large; not checked | hold for rights | Sets 2 and 6 are outside scope (DEC-001, V). Sets 1 and 2 may be EPI Suite values. |
| SolCuration (Meng et al. 2022) | https://github.com/Mengjintao/SolCuration | code BSD 3-Clause by the paper; no LICENSE file; data terms follow each source (R, V) | repo files: AQUA 1,310, PHYS 2,001, ESOL 1,116, OCHEM 4,218, AQSOL 8,701, CHEMBL 31,099, KINECT 82,057 (V) | measured | PHYS and OCHEM after removing overlap | AQSOL is AqSolDB; AQUA and ESOL are inside it | hold for rights | Exclude KINECT (kinetic solubility, another quantity) and CHEMBL (outside scope, DEC-001). |
| PSDI Physical Chemistry Properties, LogS subset | https://github.com/PSDI-UK/psdi-datasets/tree/main/Data_Collection_DataSets/LogS | "To be confirmed (TBC)" (R, V) | 109,254 water records | measured, re-published | nothing | every record cites SolCuration (108,121) or ESOL (1,133) (V) | exclude | A repackaging; it adds no new data. |
| IUPAC-NIST Solubility Database (NIST SRD 106) | https://srdata.nist.gov/solubility (DOI 10.18434/T4QC79) | NIST open licence by data.gov (R; not re-read by V) | over 67,500 measurements | measured, critically evaluated | high-quality spot checks | SOMAS used 149 | test set only | No bulk download; leans to salts and gases. |
| Solubility Challenge 2 (Llinas & Avdeef 2019) | https://doi.org/10.1021/acs.jcim.9b00345 | not checked | 100 and 32 compounds | measured intrinsic solubility | nothing | may overlap | exclude | Intrinsic solubility of the neutral form, not AqSolDB's quantity; outside scope (DEC-001, V). |
| EUOS/SLAS solubility challenge | EU-OPENSCREEN news page (research_data_measured_2) | not checked | about 100,000 compounds | kinetic, three classes | nothing | none | exclude | Kinetic; outside scope (DEC-001, V). |
| Perplexity lead, not checked: SolProp (Vermeire, Chung, Green 2022) | https://zenodo.org/records/5970538 | CC BY 4.0 (P) | aqueous solubility at 298 K for 11,804 solutes (P) | compiled; some files computed or fitted (P) | many solutes | with AqSolDB not checked | hold for rights | Take the aqueous logSaq file only; never use its fitted Abraham parameters as labels. |
| Perplexity lead, not checked: Bradley open melting points | https://figshare.com/articles/dataset/Jean_Claude_Bradley_Open_Melting_Point_Datset/1031637 | CC0 (P) | 28,645 measurements, some marked "do not use" (P) | measured | melting point as an input | none | hold for rights | An input, not a label. The general solubility equation did far worse than trees on AqSolDBc (RMSE 1.86 against 0.78; Llompart et al. 2024, checked in `verify_physics_lit`), so the gain is unclear. |

Unopened lead: MixtureSolDB (10.5281/zenodo.18660057; organic solvent mixtures, not water, by
Perplexity F4). Perplexity found no open aqueous set with several temperatures per compound.

## Surfactants (CMC and surface tension)

Sources: `research_data_measured_2` (R), `verify_data_measured_2` (V), Perplexity F5.

**Trains on now.** SURFPRO `domains/cleaning/work/data/S129_SURFPRO.csv` (Zenodo 15861809, CC BY 4.0): 1,624 surfactants
(recounted); pCMC 1,395 rows in 115 paper groups, surface tension at the CMC 972 rows in 190 groups
(`domains/cleaning/README.md`). 1,311 of the 1,395 pCMC rows are at 25 C (R, V). Cross-fitted trees: 0.527 pCMC, 4.84 mN/m
(report). These are surfactant properties, not cleaning performance.

| Name | Link | Licence (who) | Size | Measured or computed | Adds | Overlap with ours | Verdict | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PSDI NIST CMC collection (NSRDS-NBS 36, 1971) | https://resources.psdi.ac.uk/data/a7c82670-d2e2-46c6-920a-74294289aa34 | CC BY 4.0 (R, V) | 719 compounds; 4,990 value rows, 4,559 numeric, 2,390 with additives; 1,429 at 25 C, 397 with unknown temperature (V) | measured (1971 compilation); its SMILES and values were extracted by a language model or an AI service | CMC at several temperatures and with additives | 168 of 604 parsable compounds match SURFPRO; about 436 may be new (R; V reproduced it) | add | After re-curation: re-assign every SMILES from the name (four checked examples are wrong), spot-check values against the 1971 PDF, start with no-additive rows. |
| Temperature-dependent CMC (Brozos et al. 2024) | https://arxiv.org/abs/2403.03767 | training data "property of BASF" (R, V) | 1,377 points, 492 surfactants, 0 to 90 C | measured | the public test sets: 227 points at new temperatures, 218 points of 50 unseen surfactants | many points from the 1971 table | test set only | Use the 218-point unseen-surfactant set after removing overlaps; the repository URL is not confirmed. |
| Multi-property surfactant repository (Brozos et al. 2024) | https://github.com/brozosc/Graph-Neural-Networks-for-Surfactant-Multi-Property-Prediction | MIT, a software licence (R, V) | test set only (22,710 bytes); the paper used 429 CMC and 164 Γmax molecules | measured | a small external check | likely with SURFPRO | test set only | Training data on request. |
| Mixture CMC (Brozos et al. 2025) | https://arxiv.org/abs/2411.02224 | training data BASF property (R, V) | 108 binary mixtures, 515 mixture points | measured | the only mixture-CMC source found | not checked | hold for rights | Could be rebuilt from the 11 primary papers it lists. Which two test sets were released is not stated (V). |
| Complementary 43-surfactant set (Moriarty et al. 2023) | https://doi.org/10.1021/acs.jctc.3c00868 | not checked | 43 systems | measured | nothing new | a subset of the 1971 table; the Qin source appears in 114 SURFPRO rows (V) | exclude | Already covered; a worked example of filtering the 1971 table. |
| Temperature-dependent CMC (Adygamov et al. 2026) | https://www.mdpi.com/2624-8549/8/2/26 | preprint CC BY 4.0 (V) | 1,829 values for 1,707 surfactants (V) | measured, merged | temperature annotations | very high: SURFPRO, Qin, the 1971 table, Chen 2024, the Brozos test set | exclude | Mostly what we have. Its Chen et al. 2024 source (Colloids Surf. A 703, 135276; 779 surfactants) is a new, unopened lead. |
| Perplexity lead, not checked: P2SURF (Sandia) | https://github.com/snlai4chembiogroup/P2SURF | LICENSE file not read (P) | not stated | experimental CMC (P) | single and binary CMC | builds on Qin (P) | hold for rights | Values in log µM; convert to pCMC. |
| Perplexity lead, not checked: PendantProp (2025) | https://doi.org/10.5281/zenodo.17232763 | CC BY 4.0 (P) | 11 surfactants plus mixtures (P) | measured by one robot | full isotherms from one protocol | authors overlap the SURFPRO group | hold for rights | Small; a one-protocol test. |
| Perplexity leads, not checked: Kleinheins et al. 2024; SurfPro2 | https://doi.org/10.5281/zenodo.10550607; ChemRxiv 10.26434/chemrxiv.15006392 | none named; data link not verified (P) | not stated | not stated | maybe added salt | not stated | hold for rights | Unopened. |

Perplexity found no open dataset of cleaning performance with formulation composition.

## Corrosion (rate)

Sources: `research_data_measured_2` (R), `verify_data_measured_2` (V), Perplexity F6.

**Trains on now.** NIST CORR-DATA (NIST open data terms), parsed into 24,721 rows; the target is log10 mm/yr on 5,761 point
values from 207 reference groups; letter ratings are not modelled (`domains/corrosives/README.md`). Cross-fitted: network
average 0.768, trees 0.772 log10 mm/yr, no clear difference (-0.00428 [-0.0221, +0.0154]; report).

No checked source adds rows of the same quantity, an immersion rate in a named environment; the one lead, Zenodo
23752330 (359 immersion weight-loss coupons in water and sediment; Perplexity F6, search snippet only), is unopened.
Perplexity found no immersion dataset that states ASTM G31 or NACE TM0169. Everything else below measures something
else, so it would be a new target, not more data for this one.

| Name | Link | Licence (who) | Size | Measured or computed | Adds | Overlap with ours | Verdict | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Electrochemical metrics for corrosion-resistant alloys (Nyby et al. 2021) | https://doi.org/10.1038/s41597-021-00840-y | not checked: figshare (R, V) | 1,274 records, 85 papers; 8 metrics (icorr, ipass, Ecorr, Epit, Erp, Ecrev, Tpit, Tcrev), no corrosion rate (V) | measured; some digitised | electrochemical targets for alloy compositions | essentially none | hold for rights | Many Ni-Cr-Mo Erp values are lower bounds; potentials vs SCE. |
| MPEA corrosion database (2024) | https://doi.org/10.17632/nskb9khgsm.1 | CC BY 4.0 (V; R had not checked) | 619 alloys; Ecorr 604, icorr 580, Epit 329 | measured; partly digitised | alloy-electrolyte pairs | partial with Nyby's HEA tab, not checked | add | As a new target (log icorr). Drop the 13 PBS and 5 Hanks rows (scope). Check icorr units in the file: the review labels them A/cm2 in one table and µA/cm2 in another. |
| Additively manufactured MPEA corrosion database | https://doi.org/10.17632/cfz68tkwfk.3 | CC BY 4.0 (V) | not checked | measured | processing variables | check against the database above | add | With the database above, after removing overlaps; recompute its entropy and VEC features. |
| CORDATA inhibitor database (Galvao et al. 2022) | https://datacor.shinyapps.io/cordata/ | full set to contributors or on request (R, V) | 4,934 efficiencies, 374 compounds, 39 metals | measured | inhibitor efficiency | none | hold for rights | Efficiency is relative to each paper's blank: group by paper. |
| DATACOR Dataset V2 | https://doi.org/10.17632/v5p322m2t8.3 | CC BY 4.0 (R, V) | row count not checked | measured labels, computed descriptors | an open seed for an Al-alloy inhibitor target | probably CORDATA's Al part | add | As a new target, after removing overlaps by SMILES, alloy and pH. |
| AA2024-T3 inhibitor library (Özkan et al. 2024) | https://doi.org/10.1038/s41529-024-00435-z | article CC BY 4.0; supplement not checked | 78 molecules; 59 fully dissolved and used (V) | measured, one lab | the most uniform inhibitor data found | none | test set only | For an inhibitor model; use the 59. |
| Inhibitor set from 5,597 papers (2026) | https://doi.org/10.17632/wm65mb6p29.2 | CC BY 4.0 (V; R had not checked) | not checked | measured, extracted by a language model | possibly the largest inhibitor set (row count not checked) | remove overlap with CORDATA | add | As a new target, after checking extractions against its raw JSON. |
| Al/Mg/Cu seawater compilation (Kaboudvand et al.) | https://arxiv.org/abs/2508.11685 | no release | 331 samples | mostly CORR-DATA, plus commercial databases and computed sources | nothing | heavy with CORR-DATA | exclude | Converts letter ratings to numbers, which the family deliberately does not do. |
| Perplexity lead, not checked: first-year atmospheric corrosion of carbon steel (2026) | https://doi.org/10.17632/ysr7wy3y7z.1 | CC BY 4.0 (P) | 617 programme records at 170 sites, plus 106 (P) | measured rates; computed weather covariates | atmospheric exposure | re-publishes the ICP, ISOCORRAG and MICAT records (P) | hold for rights | Atmospheric, not immersion: a new target. |
| Perplexity leads, not checked: NIMS CoDS, ICP Materials supplement, salt-spray and weight-loss sets | https://smds.nims.go.jp/corrosion/index_en.html; https://doi.org/10.3390/ma10080969; plos figshare 28952334; https://zenodo.org/records/23752330 | not stated (P) | not stated | measured (P) | atmospheric or salt-spray tests; immersion in water and sediment (23752330) | not stated | hold for rights | Unopened. Only 23752330 may measure our quantity; its standard and temperature are not stated (P). |

Also unopened: a CO2-corrosion dataset of 1,152 entries (10.13374/j.issn2095-9389.2025.10.21.001) and a 1978 PNNL catalogue
of aluminium in sea water (R). Perplexity excluded a Mendeley "machine learning for corrosion" set: it lists papers, not
rates.

## Solid lithium electrolytes (conductivity)

Sources: `research_data_measured_2` (R), `verify_data_measured_2` (V), Perplexity F7.

**Trains on now.** OBELiX `domains/electrolytes/work/data/processed.csv`: 599 rows (recounted), CC BY 4.0; the training
partition of 478 rows in 124 composition-or-paper groups, formula only; the 121-row test partition is public and was not
scored (`domains/electrolytes/README.md`). On 5-fold grouped cross-validation, trees (1.55 log10 S/cm) are clearly better
than the average of 5 networks (2.14): +0.587 [+0.199, +1.08], and than each single network (the report). One study
reports 0.56 log units between pellet densities of the same material, so a formula-only model has an error floor
(Perplexity, `ROUND1_DIGEST.md` point 3, not checked).

| Name | Link | Licence (who) | Size | Measured or computed | Adds | Overlap with ours | Verdict | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Liverpool Ionics, LiIon (Hargreaves et al. 2023) | http://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.html | "freely available for academic use"; commercial enquiries to the authors (R, V; P agrees) | 820 entries, 214 sources, 5 to 873 C; 465 at room temperature | measured by impedance; 31 room-temperature values extrapolated | about 355 entries at other temperatures | OBELiX was built from its room-temperature part | hold for rights | Academic-only terms may not allow a public site. Bulk and total conductivity are mixed; de-duplicate by composition and DOI (Perplexity reports a LiIon ID column in the upstream OBELiX release, not checked; our `processed.csv` has none). |
| DigBat, formerly DDSE | https://www.digbat.org (paper 10.1016/j.nanoms.2026.07.001) | not found (R, V, P) | 3,816 electrolytes, 27,645 conductivity entries, 852 computed (press release of 25 August 2026, counts as of June 2026) | mostly measured | many temperatures; possibly 5 to 10x our rows (R's estimate) | not checked | hold for rights | Ask for data and terms in writing. |
| Laskowski, McHaffie & See 2023 | https://doi.org/10.1039/d2ee03499a | unclear; ICSD structures are proprietary (R, V) | 1,346 entries, 344 with ICSD ids | measured | compositions OBELiX dropped; activation energies (not checked) | high | hold for rights | Supporting PDF only. |
| Text-mined conductivity database (Shon & Min 2023) | https://doi.org/10.1021/acsomega.3c01424 | not checked | 4,032 entries, 1,457 papers (by OBELiX) | extracted by machine | nothing reliable | not checked | exclude | 24.3% of records cannot tell ionic from electronic conductivity; no temperature. |
| LiIon plus OBELiX structure set (Kim et al. 2026) | https://doi.org/10.5281/zenodo.17157647 | CC BY 4.0 (V) | 499 electrolytes at 20 to 34 C | measured labels; 152 generated and 26 database structures are computed | candidate structures, no new labels | nearly complete | exclude | No new labels; its split is not group-safe. |
| DFT migration barriers (Devi et al. 2025) | https://arxiv.org/abs/2508.06459 | not checked | 619 barriers; 91 electrolytes; Li 28.3% | computed, mixed functionals | an auxiliary target | none | exclude | Not a conductivity; keep apart from measured labels. |
| OBELiX test partition | `domains/electrolytes/work/data/test_idx.csv` | CC BY 4.0 (repo) | 121 rows | measured | a held-out check | it is ours | test set only | Its labels are public, so it is not a blind test (`domains/electrolytes/README.md`). |
| Perplexity leads, not checked: stability windows, Na and liquid sets | figshare 25416247; github.com/Jeff-oakley/EchemWindow; 10.1038/s41524-024-01392-6; Zenodo 7244939 | not stated (P) | 1,501 compounds; 34 measured Na conductivities; 5,035 liquid experiments (P) | computed; Na, not Li; liquid | other quantities | none | exclude | Not lithium solid-electrolyte conductivity. |

Perplexity found no open data with conductivity at several temperatures, activation energies, or bulk and grain-boundary
values kept apart. LiIon and DigBat have several temperatures, but neither has an open licence.

## Insulators (dielectric constant, DFT-calculated)

Sources: `research_data_dft` (R), `verify_data_dft` (V), Perplexity F8.

**Trains on now.** `domains/insulators/work/data/dielectric_table.csv`: 1,056 rows, 964 compounds; Materials Project DFPT
values (Petousis et al.), CC BY 4.0 via Foundry-ML 10.18126/racd-go9m; DFT-calculated. Target log10(poly_total).
Cross-fitted: network average 0.118, trees 0.122 log10, no clear difference (-0.00458 [-0.00994, +0.000734]; report). The
90% interval covers only 39% of the 18 held-out compounds above 30 (`README.md`).

| Name | Link | Licence (who) | Size | Measured or computed | Adds | Overlap with ours | Verdict | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| JARVIS DFPT dielectric, already on disk | `gpu/data_cache/jarvis/jdft_3d-9-24-2025.json.zip` (https://doi.org/10.6084/m9.figshare.6815699) | CC BY 4.0 (repo provenance file) | 4,706 totals; 4,809 electronic and ionic parts (recounted) | DFT-calculated, OptB88vdW | 4,033 totals outside our 1,056 (recounted) | 986 of our 1,056 MP ids are in JARVIS; on 645 pairs the median JARVIS/MP ratio is 1.06 and 27.6% differ by over 25% (R, V, recounted) | add | As a separately labelled target or with a source flag, never pooled silently. Whether `dfpt_piezo_max_dielectric` is the largest component or an average is not checked; MP's poly_total is an average. |
| JARVIS optical dielectric, already on disk | same file | CC BY 4.0 (repo) | epsx 60,354 (OptB88vdW); mepsx 19,834 (TBmBJ) | DFT-calculated | electronic (high-frequency) values | 911 of ours have an epsx value (R) | exclude | Not the static constant we predict; only for a separate electronic-constant task. |
| Current Materials Project dielectric data | MP API, free key; 7,113 materials extracted by Kim, Kim and Min 2022 (https://doi.org/10.1039/d1cp04702g) | CC BY 4.0 on the legacy terms page (V via a search); a third-party claim of a commercial restriction is unverified | about 7,113 | DFT-calculated, same source as ours (R) | about 6.7x our rows on the same definition | should contain our 1,056 (not checked) | add | After reading the current terms: the cleanest way to grow this family. |
| Phonon dielectric set (Petretto et al. 2018) | https://doi.org/10.1038/sdata.2018.65 | not checked | 1,521 compounds; 1,296 in matminer | DFT-calculated (ABINIT) | a third functional | MP ids | hold for rights | Needs a functional tag. |
| Perplexity lead, not checked: DFPT screening set (Riebesell et al. 2024) | https://doi.org/10.5281/zenodo.10456384 | not stated for data; code MIT (P) | 2,552 DFPT values (P) | DFT-calculated | pre-training | partly MP structures | hold for rights | Unopened. |
| Perplexity leads, not checked: Dielectric_ML (Morita 2020), DieDatabase | https://github.com/WMD-group/Dielectric_ML; https://tmm-group.github.io/diedatabase | not read; not stated (P) | 1,364; page showed 0 compounds (P) | DFT from MP 2020; model predictions and a formula | nothing | MP | exclude | A copy of MP; predicted values. |
| matbench_dielectric | Matbench | not checked | 4,764 | DFT-calculated | nothing | MP | exclude | Its target is refractive index, not the dielectric constant (R, V). |
| Measured check: JARVIS-Leaderboard dielectric | https://github.com/usnistgov/jarvis_leaderboard | NIST terms, free to use (V) | 16 values on exact JARVIS ids | measured | a reality check | none | test set only | For electronic values only: they agree with them (12.6% error) but not with DFPT totals (30.5%, biased +3.2), so they cannot check poly_total (R, V). |
| Measured check: Petousis et al. 2016 | Phys. Rev. B 93, 115151 | not checked | 88 compounds | measured values, compared with DFPT | a static-constant check | MP | test set only | PBE mean absolute relative deviation 16.2% (V). |
| Measured check: Young & Frederikse 1973 | https://doi.org/10.1063/1.3253121 | NIST PDF, not checked | over 300 inorganic solids (not checked) | measured static values with temperature and frequency | the best static check found | not checked | test set only | Must be typed in by hand. |
| Measured: text-mined dielectric records (Zhao & Cole 2022) | https://doi.org/10.1038/s41597-022-01295-5 | CC BY 4.0 (R; not confirmed by V) | 60,804 records, 11,054 chemicals | extracted by machine | broad coverage | not checked | test set only | After filtering: liquids and films included, frequency often missing. |
| Measured: Starrydata dielectric subset | https://github.com/starrydata/starrydata_datasets (paper 10.1080/27660400.2025.2485018) | not stated (R, P) | over 20,000 samples (not checked) | measured, digitised from papers | temperature and frequency | not checked | hold for rights | Public release not confirmed. Some releases repeat sample ids (Perplexity lead, not checked). |

Perplexity found no open measured breakdown-strength dataset.

## Crystals (JARVIS, DFT-calculated)

Sources: `research_data_dft` (R), `verify_data_dft` (V), Perplexity F9. "Recounted" means recomputed for this file from the
cached zip with a local script.

**Trains on now.** JARVIS `dft_3d` of 2025-09-24: 93,902 records, all OptB88vdW, CC BY 4.0, SHA-256 starting `f9e0a330`
(`gpu/data_cache/jarvis/jdft_3d-9-24-2025.json.zip.provenance.json`). Four targets (`gpu/structure/train_structure.py`):
Voigt bulk and shear modulus in log10 GPa (32,433 records each), formation energy per atom and the OptB88vdW band gap.
Held-out chemical systems, fixed neighbour rule, mean of single runs: bulk 0.101 and shear 0.181 log10 GPa, formation
energy 0.052 eV/atom, band gap 0.204 eV (report).

### Labels already in the cached file

| Field | Records (recounted) | What it is | Use |
| --- | --- | --- | --- |
| `mbj_bandgap` | 21,559 | TBmBJ band gap | a second band-gap target, much closer to experiment (below) |
| `hse_gap` | 56 | HSE band gap | too few to train |
| `epsx` (and y, z) | 60,354 | OptB88vdW optical dielectric constant | an electronic-constant task |
| `mepsx` | 19,834 | TBmBJ optical dielectric constant | the same |
| `dfpt_piezo_max_dielectric` | 4,706 | DFPT total dielectric constant | a second insulator target |
| DFPT electronic and ionic parts | 4,809 each | parts of the above | the same |
| `elastic_tensor` | 34,502 | 6x6 elastic tensor | Voigt-Reuss-Hill moduli can be computed |
| `icsd` | 20,812 | ICSD id present | flags experimentally known structures |

Traps (R, V): Kv is 0 or below in 536 records and Gv in 2,667 (recounted), so log10 targets must drop them; 8,635
formula-and-space-group groups hold more than one record (20,604 records); 4,994 records are marked 2D-bulk. 1,290
elastic records carry bare numeric references of unknown origin (V corrected R's "no reference").

### Where the structures come from

The `reference` field gives each structure's origin (R, V; first three recounted): mp- 57,345 records (55,231 distinct
Materials Project ids), auid- (AFLOW) 10,723, agm (Alexandria) 4,492. In the elastic subset: mp- 16,566, auid- 3,337, agm
3,069. More than half of JARVIS is other databases' crystals relaxed again, so any MP, AFLOW or Alexandria data contains
the same crystals under another functional. Join on this field, or match structures, and keep each crystal and chemical
system on one side of the split.

### Label shifts computed on the cached file

- **Voigt against Voigt-Reuss-Hill.** On 23,426 mechanically sensible tensors, Kv/K_VRH has median 1.003 (90th
  percentile 1.11, 99th 1.525) and Gv/G_VRH median 1.072 (1.398, 1.89); Gv/G_VRH exceeds 1.10 for 40.9% of crystals, Kv
  for 11.0% (R, V). Materials Project, Matbench and AFLOW publish VRH. log10(1.072) = 0.030, against a shear test error
  of about 0.17 to 0.18 log10 GPa. A recount for this file with a slightly different stability filter (22,842 tensors)
  gave median Gv/G_VRH 1.070, 39.8% above 1.10. Recompute Voigt from the tensors, or VRH for everything, before
  merging.
- **Materials Project dielectric against JARVIS DFPT.** On 645 crystals in both, the median JARVIS/MP ratio is 1.06 and
  27.6% differ by more than 25% (R, V, recounted). Using every matched record gives 673 pairs and 27.9% (V).
- **Formation energy.** Materials Project's MP2020 corrections and OQMD's chemical potentials are fitted to experiment;
  JARVIS OptB88vdW is not. Databases differ by up to 0.105 eV/atom (Hegde et al., arXiv 2007.01988, V), about 2.2 times
  our 0.048 eV/atom formula-split error (V, report).
- **Band gap.** OptB88vdW gaps sit about 1 eV below measured ones (next table). PBE gaps from MP or Alexandria are similar
  in kind; TBmBJ, HSE and experiment are a different level.

### Measured benchmark sets (test set only)

| Name | Link | Licence (who) | Size | Result against the cached labels, or note |
| --- | --- | --- | --- | --- |
| JARVIS-Leaderboard experimental sets | https://github.com/usnistgov/jarvis_leaderboard (benchmarks/ES/SinglePropertyPrediction) | NIST terms, free to use (V) | 54 band gaps, 21 bulk moduli, 16 dielectric constants, all on exact JARVIS ids | OptB88vdW gap MAE 1.355 eV, every value too low; TBmBJ 0.476 eV on 53 (the leaderboard lists 0.4981 on 54); HSE 0.268 eV on 6; Kv MAE 5.73 GPa, 6.4% low on average (Si 87.3 against 99.2 GPa) (R, V) |
| Zhuo et al. 2018 experimental gaps | https://github.com/yzhuo33/BandGapPredictor (paper 10.1021/acs.jpclett.8b00124) | no LICENSE file (V) | 3,896 nonzero gaps, 2,470 compositions; 744 matched to an ICSD-linked JARVIS polymorph | OptB88vdW MAE 1.06 eV (bias -0.98); TBmBJ 0.74 eV on 506. The polymorph is picked by lowest energy, not by the measured phase (R, V) |
| RealMat-BaG | https://github.com/Shef-AIRE/bandgap-benchmark | MIT (V; R had "not stated") | 1,705 gaps with MP ids | the same polymorph guess; structures fetched from MP |
| Borlido et al. 2019 | https://doi.org/10.1021/acs.jctc.9b00322 | not checked | 472 solids with experimental structures | download not checked |
| BandgapDatabase1 (Dong & Cole 2022) | https://doi.org/10.1038/s41597-022-01294-6 | not checked | 100,236 records | text-mined, precision 84%: filter to bulk, single phase, room temperature first |
| foundry_ml_exp_bandgaps | https://doi.org/10.6084/m9.figshare.22814318 | not checked | 2,069 entries | not opened |
| Kim et al. 2017 formation enthalpies | https://doi.org/10.1038/sdata.2017.162 | not checked | 1,276 calorimetric values with MP and OQMD ids | mostly binary intermetallics, borides, carbides, silicides |
| Wang et al. 2021 compilation (matminer expt_formation_enthalpy_kingsbury) | https://doi.org/10.1038/s41598-021-94550-5 | not checked | 2,135 values | MP2020 corrections were fitted to it (R): independent for JARVIS, not for MP energies |
| SSUB | https://doi.org/10.6084/m9.figshare.22583677 | not checked | 1,726 entries | not opened |
| Perplexity lead, not checked: MinX in UniFFBench | https://github.com/M3RG-IITD/UniFFBench | code MIT; data not stated (P) | about 1,500 minerals with experimental structures and elastic properties (P) | the only structure-linked measured elastic set found; value types not checked |
| Perplexity lead, not checked: NIST-JANAF tables | https://doi.org/10.18434/T42S31 | Standard Reference Data terms (P) | PDF per element | evaluated formation enthalpies; typed in by hand |
| Perplexity lead, not checked: JARVIS paper's Table 7 | as reported in `ROUND1_DIGEST.md` | - | formation energy MAE 0.128 eV/atom on 1,317 materials | not checked; its band-gap and bulk-modulus figures (1.33 eV, 5.75 GPa) agree with what we computed on the leaderboard sets (1.355 eV, 5.73 GPa) |

Perplexity found no measured band-gap set that gives a structure ID, method and temperature together (F9).

No large open set of measured single-crystal elastic moduli was found (a negative search, R, V). The known pieces are the
21 leaderboard bulk moduli, AFLOW's 74 benchmark systems and about 150 systems compiled by de Jong (not checked). The
720,308-record literature set of Kumar, Kabra and Cole (https://doi.org/10.1038/s41597-024-03979-6) holds engineering,
polycrystalline values and cannot check Voigt moduli.

### Larger DFT sources

| Name | Link | Licence (who) | Size | Measured or computed | Adds | Overlap with ours | Verdict | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Materials Project elasticity tensors (MatTen) | https://doi.org/10.5281/zenodo.8190849 | CC BY 4.0 (V; R had not checked) | 10,276 tensors | DFT-calculated, PBE family | a second-fidelity modulus label | likely most of JARVIS's 16,566 MP-sourced elastic records | add | Recompute Voigt from the tensors. The file has no MP id, so it needs structure matching (V). |
| Matbench log_kvrh / log_gvrh; de Jong 2015 | Matbench docs; https://doi.org/10.1038/sdata.2015.9 | not checked | 10,987 VRH entries; 1,181 compounds | DFT-calculated | nothing beyond MatTen | MP | exclude | VRH and filtered; subsets of MP. |
| Alexandria | https://archive.materialscloud.org/record/2023.71 | CC BY 4.0 (R, V) | 4,489,295 3D PBE entries; 422,098 PBEsol and SCAN | DFT-calculated | formation energy and band-gap pre-training | 4,492 JARVIS records cite Alexandria | add | For pre-training only, with a functional tag; no elastic or dielectric labels. |
| LeMat-Bulk | https://huggingface.co/datasets/LeMaterial/LeMat-Bulk | CC BY 4.0 (R, V) | 5.3 million PBE rows, plus PBEsol and SCAN | DFT-calculated | one fingerprint across MP, OQMD and Alexandria | MP, OQMD, Alexandria | add | As a de-duplication tool and for energies; no gaps or moduli. Its newer features are marked "not yet released" (V). |
| OQMD | https://doi.org/10.1038/npjcompumats.2015.10 | "without restrictions" by the 2015 paper; CC BY 4.0 by a secondary listing, not checked | over 1.2 million (not checked) | DFT-calculated, fitted chemical potentials | formation-energy pre-training | MP-like | hold for rights | A different quantity from JARVIS formation energy; never pooled into its test target. |
| AFLOW AEL-AGL | https://arxiv.org/abs/1611.05714 | "free for scientific, academic and non-commercial purposes. Any other use is prohibited" (V) | over 3,000 elastic entries | DFT-calculated | a second elastic source | 10,723 JARVIS records cite AFLOW | hold for rights | Non-commercial; C44 and C66 differ from JARVIS by a median 13% and 19% on 18 matched structures, consistent with an axis convention (V). |
| SNUMAT hybrid-functional gaps | https://doi.org/10.1038/s41597-020-00723-8 | not checked | 10,481 materials | DFT-calculated, hybrid | a high-fidelity gap label | ICSD-based | hold for rights | RMSE 0.36 eV against experiment (V: it is an RMSE). |
| GNoME | https://github.com/google-deepmind/materials_discovery | data CC BY-NC 4.0 (R, V) | about 381,000 structures | DFT-calculated | hypothetical crystals | far from JARVIS's space | exclude | Non-commercial. |
| C2DB, MC3D, NOMAD | https://doi.org/10.1088/2053-1583/ac1059; arXiv 2508.19223; NOMAD API | not checked | C2DB over 15,000 monolayers (V); MC3D 32,013 PBEsol structures | DFT-calculated | little for these tasks | not checked | exclude | 2D layers; mixed settings per upload. |
| OMat24, MatPES | https://huggingface.co/datasets/facebook/OMAT24; materialyze/matpes | CC BY 4.0; BSD-3-clause (R, V) | 100K to 1M structures (MatPES) | DFT-calculated forces and energies | interatomic-potential data | not checked | exclude | Not property labels. |

## Not checked

- Licences still unread because hosts were blocked: the Tan, Kashem, Qiu, Ponnu Durai and CO2-cured concrete records; GE,
  Cu and ULTERA alloys; AqSolDBc, SOMAS and Nyby; LiIon's CSV page; OQMD, SNUMAT, Zhao & Cole, Dong & Cole, NOMAD, MC3D
  and C2DB.
- Every overlap count marked "expected" or "probably" was reasoned from papers, not computed. The counts that were
  computed are: SCM against Yeh by reference DOI (not row by row), dax-ti and LPBF DOIs against our alloy file,
  PolyMetriX and OpenPoly against our polymers, Toni against PolyMetriX, the PSDI LogS source column, PSDI CMC against
  SURFPRO, MP against JARVIS dielectric, and the JARVIS reference prefixes.
- Polymer overlaps are lower bounds: stereochemistry, end groups and repeat-unit choice can hide true matches (R).
- No source in this file has been merged, trained on or scored. Every "adds" column is a prediction about value, not a
  result.
- Hours in the list below are our guesses, not measurements.

## The ten additions with the best expected value per hour

1. **Score the crystal models on the measured sets already keyed to JARVIS** (leaderboard: 54 gaps, 21 bulk moduli, 16
   dielectric constants; Zhuo: 744 matched compositions). Test set only; the downloads are three files under 1 kB and
   the 8 MB Zhuo workbook; about 2 hours. The Zhuo repository has no licence file (V): keep the workbook outside the
   repository and ask the person before publishing its scores. [Certain] The OptB88vdW gap is about 1 eV below measured values on both sets
   (MAE 1.355 eV on 54, 1.06 eV on the Zhuo match), far above the network's 0.20 eV error against DFT, so this number
   belongs next to every band-gap result.
2. **Add a TBmBJ band-gap task** from the 21,559 labels on disk. About 3 hours. [Likely] Its labels are much closer to
   experiment (TBmBJ 0.476 eV on 53 leaderboard gaps against OptB88vdW 1.355 eV on all 54; TBmBJ 0.74 eV on 506 Zhuo
   matches against OptB88vdW 1.06 eV on all 744), but it has a quarter of the rows, and the network's error on it is
   unknown.
3. **Seal new late-age concrete rows before anyone looks at them**: Tan's set, the Ponnu Durai study, then the Pavement
   lead. About 4 hours for a sealing script on the pattern of `lab/make_splits.mjs`, after the licences are read. [Likely]
   This is the only route to a fresh long-age test now that the vault is spent (LES-006). [Speculative] How many
   late-age rows they hold: nobody has seen their ages.
4. **AqSolDBc as a cleaner training variant, and its 2,251-compound OChemUnseen file as a sealed external test.** About 3
   hours once the licence is read. [Likely] The source paper reports about 0.10 log lower RMSE from the cleaning (V),
   though that is cross-validation within each set, so part of it may come from cleaner test labels; no other
   never-used LogS test was found.
5. **The SCM concrete set (28-day)**, after removing its 122 Yeh rows and matching against `lab/data/select.json`, which
   holds every UCI row at 14 and 28 days (481 rows; `lab/README.md`), so no blocked file is opened. About 3 hours. [Likely]
   Its CC BY 4.0 licence is confirmed and it brings mixes from about 64 studies besides Yeh 1998; scoring the current models on it
   once first measures the cross-lab drop that `README.md` warns about.
6. **The 182 to 183 non-GREA PolyMetriX rows and about 146 new OpenPoly structures** for homopolymers. About 4 hours.
   [Likely] They target the largest error (about 30 K on homopolymers). [Speculative] How much they help: OpenPoly values
   are averages over many reports.
7. **Grow the insulator family on its own definition**: the current Materials Project dielectric collection (about 7,113
   materials) first, JARVIS DFPT (4,033 new totals on disk) second, as a labelled second target. About 4 hours. [Likely]
   for the MP rows, which share source and functional with the 1,056 we train on. [Speculative] for JARVIS DFPT until its
   field definition is checked, given the 27.6% of shared crystals that differ by more than 25%.
8. **The GE refractory alloys as a blind test, and dax-ti as an extrapolation group.** About 3 hours once the GE licence
   is read. [Likely] 82 alloys from one lab, published after Borg, make a clean whole-source holdout. [Speculative] That
   dilute Ti alloys improve high-entropy predictions.
9. **Ask in writing for terms**: NIMS (the 7,170 PoLyInfo-lineage Tg rows), the LiIon authors (about 355 entries at other
   temperatures), DigBat (27,645 conductivity entries) and the BASF/RWTH group (CMC training sets). About 1 hour of
   writing. [Speculative] Whether anyone says yes; the possible gain is the largest in this file, and it costs nothing
   (DEC-005).
10. **BigSolDB 2.2 water rows with temperature as an input** (6,814 rows for 711 solutes in v2.0). About 3 hours.
    [Likely] It adds the temperature axis AqSolDB lacks. [Speculative] Whether it lowers error at 25 C.

Next after these: the PSDI NIST CMC table (about 436 possibly new surfactants with temperatures and additives, but every
SMILES and many values need re-curation), the MPEA corrosion database as a new log icorr target, and the MatTen moduli
recomputed as Voigt.
