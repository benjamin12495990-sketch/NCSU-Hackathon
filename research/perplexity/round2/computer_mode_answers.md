# Perplexity Computer mode, round 2 (partial: credits ran out)

Pasted by the project owner on 2026-11-27. Perplexity ran `computer_prompt_1_rest.txt` and `computer_prompt_2_verify.txt`
(version 2) until the account ran out of Computer credits. Its sandbox could read web pages but could not download files,
so nothing here rests on an opened data file. Tags: [record] = read on the data record page, [snippet] = search result
only, [secondary] = a document quoting a standard. Two attached documents (A_new_categories, SUMMARY) did not come
through; the text below includes their summaries. The project's own checks of these leads are in `ROUND2_NOTES.md`.

---

Both record pages now answer the licence question. I still haven't opened either data file, so no row counts exist.
Zenodo 7244939 (A1)

* What it is: "Dataset of 5035 Conductivity Experiments for Lithium-Ion Battery Electrolyte Formulations at Various Temperatures" [record]. The authors are Rahmanian, Vogler, Wölke, Yan, Fuchs, Winter, Cekic-Laskovic and Stein, published 2022-10-24. The record is at https://doi.org/10.5281/zenodo.7244939.
* Licence: "Creative Commons Attribution 4.0 International" [record], written at https://zenodo.org/records/7244939.
* File: Conductivtiy_experiment.csv, 39.2 MB [record]. It is not downloaded, so I haven't counted rows.
* Open questions: The record says it accompanies a data descriptor with "Automated Analysis" in its title. It doesn't say whether the values are measured, and it doesn't say whether they overlap CALiSol-23. I haven't checked either.

ML-HydPARK (A7)

* Licence: "Creative Commons Attribution 4.0 International" [record], written at https://zenodo.org/records/10680097.
* File: ML-HYDPARK_v0.0.5.csv, 100.5 kB, version 0.0.5, published 2024-02-19 [record]. I haven't counted rows with an enthalpy, an entropy or a capacity.
* Contents: The record says it is cleaned from the DOE's HydPARK database. Duplicates and entries missing critical data were removed, and "Most entries contain data on the enthalpy and entropy" plus the maximum hydrogen capacity [record]. The page mentions no per-row source column.

Not done

* Bradford 2023 DOI link: https://doi.org/10.1021/acscentsci.2c01123 failed to fetch.
* CALiSol-23 licence: The DTU page text still showed no licence line.
* DOE HydPARK, HyStor and DigHyd: Not attempted.

Each run starts with a fresh sandbox, so the files I wrote earlier may not carry over. The output files only hold this run's entries, so keep the findings in this reply.
I can't download or count rows in this setup. If I can't, the pass-1 items will stay at "partly done". For them, "done" would mean reading the file itself.
Summarize findings for Section A so far
Section A is mostly not started. Only five items have any findings, and none rests on an opened data file, so nothing can be ranked yet.
Findings so far
A1, liquid electrolytes

* CALiSol-23: The DTU record says "13,825 individual data points from 27 different experimental articles", covering 38 solvents and 14 lithium salts [record]. The 13,269 figure is unchecked. The licence is not found on the page text I retrieved, and the file was not opened. Record: https://data.dtu.dk/articles/dataset/CALiSol-23/24559960
* Zenodo 7244939: "Dataset of 5035 Conductivity Experiments for Lithium-Ion Battery Electrolyte Formulations at Various Temperatures", published 2022-10-24 [record]. The licence is "Creative Commons Attribution 4.0 International" [record], written at https://zenodo.org/records/7244939. The file is Conductivtiy_experiment.csv, 39.2 MB. It was not opened, so rows are not counted. Whether the values are measured, and whether it overlaps CALiSol-23, is unchecked.
* Bradford 2023: The DOI link failed to fetch, so the data record, licence and row count are not found.

A2, SciGlass

* The LICENSE file at https://github.com/epam/SciGlass/blob/master/LICENSE starts with "ODC Open Database License (ODbL)" and "Copyright (c) 2019 EPAM Systems". The text after that is MIT-style [file].
* https://sciglass.uni-jena.de says the data is "released under the ODC Open Database License (ODbL) in 2019" [record]. It lists 422,949 glasses and melts.
* The source-reference ID and the Tg, density and Young's modulus counts are not done.

A3, Starrydata2

* Your earlier claim is confirmed [record]. Datasets published from 2026-04-01 to 2026-09-08 contain duplicated `sample_id` values. The README says the fix came on 2026-09-08.
* GitHub Releases and Google Drive are corrected. Figshare archives are corrected from 2026-10-01. The workaround is to join on (SID, sample_id). Source: https://github.com/starrydata/starrydata_datasets
* No licence line appeared in the README text I retrieved. ESTM and Sierepeklis and Cole are not started.

A7, hydrogen storage

* ML-HydPARK: The licence is "Creative Commons Attribution 4.0 International" [record], written at https://zenodo.org/records/10680097. The file is ML-HYDPARK_v0.0.5.csv, 100.5 kB. It was not opened, so the enthalpy, entropy and capacity counts are not done.
* DOE HydPARK, HyStor and DigHyd were not attempted.

Status

* Not started: A4, A5, A6, A8, A9, A10, A11, A12 and A13.
* Licence conflicts: Only A2, ODbL against MIT-style text in the same file.
* Could not open: The Zenodo and DTU API and download routes failed earlier. The Bradford DOI link failed too.
* Ranking: Every item is unranked, because the rules require a counted [file] row total.

List what still needs to be checked for Section A
Open the Zenodo file to count the rows
Check the remaining items in Section A
Draft the report file update for Section A
Investigate the failed Bradford DOI link

Only V1b is done in this run. Everything else is "not started", and V1a could not be completed. My tools here can read web pages, but the code sandbox has no internet. That means I can't download data files, count rows, compute SHA-256 hashes or read headers. R3 and R6 require file-level reads for most items, so I haven't guessed any of those numbers. I also haven't created VERIFICATION.md, which breaks R7. The content below is laid out so you can paste it into that file.

* # VERIFICATION.md
* Status summary: V1b partly done. V1a, V1c, V1d, V2 to V9 not started.
* ## V1. POLYMERS
* V1a status: not started. Not a single route has been tried on https://zenodo.org/records/19242815. I can't tell you whether your 64,139-byte copy is the same file.
* V1b status: partly done. I read the PoLyInfo landing page. I did not open the separate MatNavi Service Terms of Use page, which is where the exact reproduction, redistribution and derived-data wording would be.
* V1b record URL: https://polymer.nims.go.jp [record]
* V1b disclaimer, exact quote 1: "National Institute for Materials Science (NIMS) holds the copyright of this database system." [record] (page: https://polymer.nims.go.jp)
* V1b disclaimer, exact quote 2: "No reproduction, republication or distribution to third parties of any content is permitted without written permission of NIMS." [record] (page: https://polymer.nims.go.jp)
* V1b bulk-download notice, exact quote: "The acquisition of large amounts of data, whether by manual or mechanical means, is prohibited by the MatNavi Service Terms of Use." [record] (page: https://polymer.nims.go.jp)
* V1b access note: the page says use requires DICE account registration and a MatNavi usage application. Per R1 I did not go further. [record]
* V1b dataset note: the page says "Data including unpublished data are also available in dataset format for a fee." [record]
* V1b public counts on the page, "As of 30 September 2026": homopolymers 16,064; copolymers 5,638; polymer blends 1,978; composites 2,153; polymer samples 112,771; data points 357,812; references 15,365. [record]
* V1b derived-data wording: not stated on the landing page. The Terms of Use page is still to be checked.
* V1c status: not started. URL to open: https://github.com/liugangcode/GREA
* V1d status: not started. Record to open: https://zenodo.org/records/15210035
* ## V2 to V9
* V2a to V2h: not started.
* V3a to V3g: not started.
* V4a to V4g: not started. R6 stays in force: no strength values or statistics will be shown.
* V5a to V5e: not started.
* V6a to V6d: not started.
* V7a to V7e: not started.
* V8a to V8f: not started.
* V9a to V9g: not started.
* ## Could not open
* V1a: the Zenodo record, the API and the file download were not attempted.
* V1b: the MatNavi Service Terms of Use page was not attempted.
* V2 to V9: no routes attempted.
* Next step: run this in an environment with internet-enabled code execution. The record pages, Zenodo and figshare APIs, GitHub raw files and Mendeley can then be downloaded and counted as R3 requires. Or tell me to continue with web-page-only items, such as the licence quotes, the V3 literature look-ups and the record-page checks. The file-level counts would stay "could not open".

Continue with the web-page-only items
I finished the record-page parts of V1a, V1c, V1d and V5d. I did not run V2 to V9, and the file-level items (hashes, row counts, headers) are still "could not open". Dropbox-style routes were not needed. The Zenodo API URLs failed in my fetch tool, but the normal record pages opened.

* ## V1. POLYMERS (updates)
* V1a record URL: https://zenodo.org/records/19242815 [record]
* V1a title as shown: "Glass Transition Prediction of Binary Copolymers Across Large Chemical Spaces Using Machine Learning and Physics-Based Modeling", published March 27, 2026, version v1, authors Bhati, Afzal, Chew, Browning, Halls (Schrodinger Inc). [record]
* V1a licence line: "Creative Commons Attribution 4.0 International", at https://zenodo.org/records/19242815 [record]
* V1a file: experimental_dataset.csv, shown as 64.1 kB, md5:11eb3b96be3116907d87038d9b74399f. [record] The page gives only a rounded size, not exact bytes. It shows no SHA-256, so I cannot say whether your 64,139-byte copy is identical. To check, compute the MD5 of your copy and compare it with 11eb3b96be3116907d87038d9b74399f.
* V1a description of the file: "Experimental Tg dataset for homo- and copolymers used for training ML model". [record]
* V1a Tg measurement method and heating rate: not stated on the record page. The paper (Polymers 18(14):1727) was not opened. Could not open: paper not attempted.
* V1b: see the previous message. The MatNavi Terms of Use page is still unread.
* V1c record URL: https://github.com/liugangcode/GREA [record]
* V1c README sentence, exact quote: "Update March 26, 2025: We delegated the polymer datasets for GlassTemp, MeltingTemp, and PolyDensit as requested by the NIMS Materials Database, MatNavi." [record]
* V1c licence: a LICENSE file is listed in the repository. I did not open it. [record]
* V1d record URL: https://zenodo.org/records/15210035 [record]
* V1d title: "Curated Glass Transition Temperature for Polymers", version v4, published April 14, 2025. The page says a newer version exists. The authors are Kunchapu and Jablonka (Friedrich Schiller University Jena). [record]
* V1d licence line: "Creative Commons Attribution 4.0 International", at https://zenodo.org/records/15210035 [record]
* V1d file: LAMALAB_CURATED_Tg_structured_polymerclass.csv, 5.3 MB, md5:833ca70192089b0613b3292394b91c57. [record]
* V1d description says it includes polymer identity, PSMILES, data source, Tg range, number of data points, Tg values, reliability and standard deviation. [record]
* V1d PoLyInfo or NIMS rights: nothing is stated on the record page. [record]
* V1d meaning of Qiu, Xie, Nguyen and Uchicago, and the reliability colours (black, gold, yellow, red): not stated on the record page. The PolyMetriX repository is at https://github.com/lamalab-org/PolyMetriX.git and was not opened. Could not open: documentation not attempted.
* ## V5. CRYSTALS AND DIELECTRICS (update)
* V5d record URL: https://github.com/starrydata/starrydata_datasets [record]
* V5d where the data is: the README says the data is split into per-project files, such as ThermoelectricMaterials, BatteryMaterials and MagneticMaterials. A dielectric project is not named in the README text I saw. The project list is "dynamic" and is given in https://starrydata.github.io/starrydata_datasets/manifest.json, which I did not open. [record]
* V5d licence or terms: not stated in the README text I read. Could not open: no licence page attempted.
* V5d rows with a permittivity: not found. Whole-database totals from the README's manifest example are 17,381 papers, 105,260 samples and 234,029 curves. These are example values dated 2026-07-17, not dielectric counts. [record]
* V5d warning: the README says datasets published between 2026-04-01 and 2026-09-08 have duplicated sample_id values. It says to join on (SID, sample_id). It also says the GitHub Releases and Google Drive copies are corrected, and the monthly figshare archives are fixed from 2026-10-01. [record]
* ## Still not done
* V1a: the SHA-256, header line, exact byte count, and the Tg method and heating rate from the paper.
* V1b: the MatNavi Terms of Use wording.
* V1c: the LICENSE file text.
* V1d: the source-column meanings and reliability labels.
* V2a to V2h, V3a to V3g, V4a to V4g, V5a to V5c and V5e, V6a to V6d, V7a to V7e, V8a to V8f, V9a to V9g: not started.

I can carry on with V2 next, then V3.
Summarize all V records done

* # VERIFICATION.md (status summary)
* Not done: V4 to V9, and V5a, V5b, V5c and V5e. I have not opened any of them. V4 stays sealed under R6.
* Not possible with my tools: downloading files, hashing them, counting rows or reading headers. Every "rows" line is therefore not found, unless a page stated the count. I could not create VERIFICATION.md. This summary is the whole report.
* ## V1. POLYMERS
* V1a: partly done. Record https://zenodo.org/records/19242815 [record]. Licence line: "Creative Commons Attribution 4.0 International" [record]. The file experimental_dataset.csv is shown as 64.1 kB with md5:11eb3b96be3116907d87038d9b74399f [record]. I have no exact byte count and no SHA-256, so I can't say whether your copy is identical. The Tg method and heating rate are not stated on the record page. The paper was not opened.
* V1b: partly done. Source https://polymer.nims.go.jp [record]. Quote: "No reproduction, republication or distribution to third parties of any content is permitted without written permission of NIMS." [record]. Quote: "The acquisition of large amounts of data, whether by manual or mechanical means, is prohibited by the MatNavi Service Terms of Use." [record]. The Terms of Use page itself was not read.
* V1c: done for the README line. Source https://github.com/liugangcode/GREA [record]. Quote: "Update March 26, 2025: We delegated the polymer datasets for GlassTemp, MeltingTemp, and PolyDensit as requested by the NIMS Materials Database, MatNavi." [record]
* V1d: partly done. Record https://zenodo.org/records/15210035 [record]. Licence: "Creative Commons Attribution 4.0 International" [record]. The file is LAMALAB_CURATED_Tg_structured_polymerclass.csv, 5.3 MB. The record says nothing about PoLyInfo or NIMS. The source-column meanings and the reliability labels are not found.
* ## V2. LITHIUM SOLID ELECTROLYTES
* V2a: could not open https://www.digbat.org. The fetch failed and no other route was tried.
* V2b: partly done. Source http://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.html [record]. Quote: "Where a user wishes to use this Database for the purposes of any extraction or re-utilisation for commercial use, then such use may only be granted under licence by the University." [record]. Entries at 15 to 35 C could not be counted. Three sources give three figures (434, 403 and 465), so a count from the table is needed.
* V2c: partly done. Source https://github.com/NRC-Mila/OBELiX [record]. Licence shown as "CC-BY-4.0 license". The README lists "Laskowski ID" and "Liion ID" columns. The row counts that have each were not obtained.
* V2d: could not open https://doi.org/10.1039/d2ee03499a. The fetch failed.
* V2e: found at https://data.caltech.edu/records/23mvv-6gk43, with 571 Li-containing compounds [snippet]. The licence and row count were not read.
* V2f: probable paper at https://www.sciencedirect.com/science/article/pii/S2352492826007063, with 712 Li-based entries [snippet]. The sodium figure, data availability and licence are not confirmed.
* V2g: done for the licence. Record https://zenodo.org/records/17157647 [record]. Licence: "Creative Commons Attribution 4.0 International" [record]. Structures: OBELiX 321 entries, USPEX computer-generated 156, Materials Project and OQMD 26 [record]. The record holds structures, not conductivity values.
* V2h: partly done. No fully open multi-temperature dataset was confirmed. A LiCondAI supplement at https://pubs.acs.org/doi/10.1021/acs.iecr.5c01315 has temperature columns [snippet]; I have not checked its licence or size.
* ## V3. MEASUREMENT NOISE
* V3a: not found.
* V3b: not found. The free ASTM E384 preview at https://store.astm.org/e0384-26.html shows no precision numbers.
* V3c: partly done. The 2005 edition copy at https://dl.azmanco.com/standards/ASTM-C/ASTM-C-Series-Full/C/C39.pdf gives, for 150 by 300 mm cylinders, a single-operator coefficient of variation of 2.4% in the laboratory and 2.9% in the field [secondary]. Multi-laboratory coefficient of variation is 5.0% [secondary]. The full current table was not found.
* V3d: done. Source https://eprintspublications.npl.co.uk/2422/1/mgpg62.pdf [secondary]. Repeatability limits range from 2.92 to 5.94 °C. Reproducibility limits range from 15.85 to 25.69 °C, over 5 to 7 sites. The guide gives a lower reproducibility limit of 4.4 to 7.1 °C in its second table.
* V3e, V3f, V3g: not found or not searched.
* ## V5. CRYSTALS AND DIELECTRICS
* V5d: partly done. Source https://github.com/starrydata/starrydata_datasets [record]. No dielectric subset is named in the README. No licence is stated there. The dielectric row count was not found.
* ## Could not open
* V1a: exact byte count, SHA-256, header line, and the paper's Tg method.
* V1b: the MatNavi Terms of Use page.
* V1d: source-column and reliability-label meanings.
* V2a: https://www.digbat.org.
* V2c: Laskowski and Liion ID row counts.
* V2d: https://doi.org/10.1039/d2ee03499a.
* V2e and V2f: the licence and data pages.
* V3a, V3b, V3e, V3f, V3g: no published number retrieved.
