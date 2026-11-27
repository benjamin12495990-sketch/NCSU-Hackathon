# Perplexity prompts: finding more training data and better methods

Written 2026-11-27. These are for a person to paste into Perplexity, one prompt at a time. Use Research
(deep research) mode if your plan has it, with both Web and Academic sources on. Each prompt stands
alone: it names what the project already has, so Perplexity looks past it. It also asks for the facts
that decide whether a dataset helps.

**Out of scope in every prompt:** superconductors (a separate project), biology and medicine, and
explosives or energetic materials.

## Status (2026-11-27)

- **Run:** F1 to F9, X1 to X3 and P1. The raw answers are in `research/perplexity/round1/`, and
  `research/perplexity/ROUND1_DIGEST.md` says what they change. Perplexity opened landing pages, not
  data files, so every item there is a lead until it is checked.
- **Not run as single prompts:** P2, M1, M2 and N1 to N8. They are replaced by one prompt for
  Perplexity's Computer mode: `research/perplexity/computer_prompt_1_rest.txt`.
- **Added:** `research/perplexity/computer_prompt_2_verify.txt`, also for Computer mode. It asks
  Perplexity to open the data files that round 1 only saw from the outside.

Each of the two `.txt` files holds one prompt and nothing else: select all, copy, paste.

## Read this first: "more data is better" is only true under conditions

The hidden assumption behind this file is that more rows always make the models more accurate. Our
own research this week found five ways new data makes a model worse or makes its score dishonest:

1. **Duplicates of data we already have.** Compilations re-publish the same rows. Examples:
   - The 2026 Zenodo SCM concrete set (1,006 rows) contains 122 rows from the UCI (Yeh) set.
   - Tan's 2026 Mendeley set lists the whole UCI set (1,030) and the UCI slump set (103) among its
     sources. This comes from its metadata; the file itself was not opened.

   If duplicates land on both sides of a train/test split, the test score goes up and the model has
   learned nothing.
2. **The same name for a different measurement.** Examples:
   - Concrete strength measured on a cube vs on a cylinder.
   - Voigt vs Voigt-Reuss-Hill elastic moduli. In our JARVIS file the ratio between the two
     shear-modulus averages is 1.072 at the median and 1.89 at the 99th percentile.
   - Glass-transition temperature measured by DSC vs DMA.
   - Corrosion rate from weight loss vs from electrochemistry.
   - Kinetic vs thermodynamic solubility.

   Merging these without a flag adds a systematic error.
3. **Computed labels sold as data.** Examples:
   - The NeurIPS 2025 Open Polymer Challenge Tg values come from molecular-dynamics simulation, not
     experiment.
   - The 100,236-record text-mined band-gap database reports 84% weighted precision.
   - One 2026 corrosion-inhibitor set was extracted by a language model.
4. **Rights.** PolyMetriX's 7,367 Tg rows are CC BY 4.0 on Zenodo. But 7,170 of them come from
   GREA, which traces to PoLyInfo (NIMS). GREA withdrew those files at NIMS's request, so the Zenodo
   label does not settle the rights.
5. **More of the same doesn't fix extrapolation.** Our crystal model's formation-energy error is
   0.048 eV/atom on held-out crystal systems but 0.48 on held-out elements. Rows like the ones we
   have will not close that gap. Rows with new elements, or physics in the inputs, might.

So every prompt below asks Perplexity for these items:

- the licence as written;
- measured or computed;
- the test method;
- what was actually opened;
- the overlap with what we already have.

[Likely] Perplexity will still get some licences and row counts wrong. Our own agents found
conflicting licences for SciGlass (MIT vs ODbL) and for the Raj 2026 alloy set (CC BY vs CC BY-NC).
Nothing from these answers goes into training until someone has opened the data record, checked the
licence, removed overlaps, and re-scored with grouped cross-validation.

## How to report back

Save each answer as `research/perplexity/<prompt id>.md` (Perplexity can export Markdown), or paste
it into the chat. Keep the prompt id (for example `F3`) at the top.

## Suggested order

1. **X2, noise floors.** This tells us where more data cannot help. If a model's error is already
   near the measurement repeatability, stop collecting for that property.
2. **F7, solid electrolytes.** This is the most data-starved family: 478 training rows and a 90%
   range of about ±4 log units.
3. **F3, polymers.** This is the largest possible gain if the rights check passes.
4. **F1, concrete long-age.** This is the data the hardest concrete test needs.
5. **F9, crystals.** Experimental test values for models trained only on DFT.
6. Then the rest, in any order.

---

## The standard ending

Every prompt below already includes this block, so you never need to add it yourself. It is shown
here once so you can see what Perplexity is asked for.

```
Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

---

## Part F: more data for the families the project already trains

### F1. Concrete: compressive strength from mix and age

**Keep late-age values unseen.** If this finds a dataset with late-age strengths, nobody opens its
values. It may become the next sealed far-future test, because the old one is spent (LES-006). The
same applies to Tan's set: seal its non-UCI rows older than 28 days before anyone looks at them.

```
I am building a model that predicts concrete compressive strength (MPa) from the mix proportions (cement, blast-furnace slag, fly ash, silica fume, water, superplasticizer, coarse and fine aggregate, kg per m3) and the curing age in days.

Already have (do not list again): the UCI Concrete Compressive Strength data (Yeh 1998, 1,030 rows) and the BOxCrete rows from github.com/facebookresearch/SustainableConcrete (MIT licence, ages 1 to 28 days).
Already found (do not list again): Zenodo 10.5281/zenodo.24108549 (Unverdi 2026, SCM cylinders); Mendeley 10.17632/3ph7nk4jzz (Tan 2026, multi-source); Zenodo 10.5281/zenodo.24089584 (Saif 2026, alkali-activated); Zenodo 10.5281/zenodo.23988578 (Matthews 2026, non-destructive tests); UCI Concrete Slump Test; Mendeley 10.17632/85r7bh4zsz (Kashem UHPC); Mendeley 10.17632/zrsbhndz9f (Qiu 2024); Mendeley 10.17632/m8557m359k; github.com/JaGuzmanT/ConcreteXAI; Mendeley 2myr3k8n4g (CO2-cured). Excluded on purpose: Zenodo 10.5281/zenodo.22746748 and the Hugging Face re-uploads of the UCI data.

Find, in this order of importance:
1. Mix-design datasets that report strength at late ages: 90, 180, 365 days or longer. This matters most.
2. Datasets that report curing temperature or humidity, or maturity (temperature-time) data.
3. Open data from transport departments, national laboratories, standards bodies or ready-mix producers' quality-control records.
4. For each dataset, whether strength was measured on cylinders or cubes, and their size.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### F2. Alloys: Vickers hardness from composition and processing

```
I am building a model that predicts the Vickers hardness (HV) of metallic alloys, especially high-entropy and multi-principal-element alloys, from composition (atomic or weight fractions) and processing (as-cast, homogenized, annealed, wrought, additively manufactured, heat-treatment temperature and time).

Already have (do not list again): Borg et al. 2020, "Expanded dataset of mechanical properties and observed phases of multi-principal element alloys", Scientific Data 7, 430.
Already found (do not list again): dax-ti Zenodo 10.5281/zenodo.6089749 (Ti alloys); GE refractory alloy screening figshare 10.6084/m9.figshare.17032760; Gorsse et al. 2023 copper alloys (Scientific Data, 10.1038/s41597-023-02411-9); LPBF alloys Zenodo 10.5281/zenodo.6514897; Raj et al. 2026 precipitate MPEAs Zenodo 10.5281/zenodo.18021833; ULTERA Zenodo 10.5281/zenodo.15716432; Gorsse et al. 2018 HEA database (10.1016/j.dib.2018.11.111); Yadav 2026 MPEA benchmark Zenodo 10.5281/zenodo.23491016; matbench_steels / Foundry-ML steel_strength (312 steels). Excluded: AlloyManufacturingNet (mostly model predictions).

Find:
1. Measured hardness datasets for steels, aluminium, titanium, nickel and cobalt alloys and HEAs that give processing history.
2. Hardness measured at elevated temperature (hot hardness).
3. For each, whether it is Vickers, Brinell, Rockwell or nanoindentation, the load, and whether any values were converted between scales.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### F3. Polymers: glass-transition temperature

```
I am building a model that predicts the glass-transition temperature Tg (K) of polymers from the repeat-unit structure (SMILES with polymerization points).

Already have (do not list again): a file named experimental_dataset.csv of measured Tg values whose original source I have not confirmed. If you can identify which public repository or paper distributes a measured-Tg file with that name, tell me its source and licence; that is a separate, valuable answer.
Already found (do not list again): PolyMetriX curated Tg, Zenodo 10.5281/zenodo.15210035 (Kunchapu and Jablonka 2025); OpenPoly benchmark, github.com/WangGroupFDU/Openpoly_benchmark; CoarsePGNN datasets, Zenodo 10.5281/zenodo.22741277. Excluded: the NeurIPS 2025 Open Polymer Challenge (labels from molecular dynamics, arXiv 2512.08896).

Find:
1. Measured Tg datasets, with the method (DSC, DMA, dilatometry), the heating rate and the molecular weight where given.
2. The current access and redistribution terms of PoLyInfo (NIMS) and of the GREA data used by PolyMetriX. Can data derived from them be redistributed or used to train a model that is shown on a public website?
3. Measured Tg for copolymers and polymer blends, with compositions.
4. Polymer Genome / Khazana: what experimental Tg data is downloadable and under what licence.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### F4. Water solubility

```
I am building a model that predicts the aqueous solubility of organic compounds, LogS (log10 mol/L, near 25 C), from the molecular structure.

Already have (do not list again): AqSolDB (Sorkun, Khetan and Er 2019).
Already found (do not list again): AqSolDBc and the OChem external test sets (Llompart et al. 2024, Recherche Data Gouv 10.57745/CZVZIA); BigSolDB 2.0 (Zenodo 10.5281/zenodo.15094979); SOMAS (figshare 10.6084/m9.figshare.14552697); SolCuration (github.com/Mengjintao/SolCuration); Solubility Challenge 2019 test sets (Llinas and Avdeef); the PSDI LogS collection (github.com/PSDI-UK/psdi-datasets); IUPAC-NIST Solubility Database (NIST SRD 106); the EUOS/SLAS Kaggle solubility challenge.

Find:
1. Measured aqueous solubility sets that state the method (shake-flask, CheqSol, potentiometric, nephelometry) and whether the value is thermodynamic (equilibrium) or kinetic.
2. Aqueous solubility at several temperatures for the same compound.
3. Open, measured melting-point datasets for organic compounds. Melting point is an input to the general solubility equation, so it is wanted as a feature.
4. Which public solubility sets are re-compilations of AqSolDB's own sources, so I can avoid double counting.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### F5. Surfactants (cleaning products)

```
I am building a model that predicts two properties of single surfactants from their structure (SMILES) and temperature: pCMC (minus log10 of the critical micelle concentration in mol/L) and the surface tension at the CMC (mN/m).

Already have (do not list again): SURFPRO.
Already found (do not list again): the PSDI machine-readable NIST CMC collection (NSRDS-NBS 36, Mukerjee and Mysels 1971); Brozos et al. 2024 temperature-dependent CMC (JCTC, arXiv 2403.03767); Brozos et al. 2024 multi-property repository (github.com/brozosc/Graph-Neural-Networks-for-Surfactant-Multi-Property-Prediction); Brozos et al. 2025 surfactant mixtures (arXiv 2411.02224); Moriarty et al. 2023 complementary 43-surfactant set; Adygamov et al. 2026 (Chemistry 8(2):26).

Find:
1. Measured CMC or surface-tension datasets that state the method (tensiometry, conductivity, fluorescence) and the added salt concentration.
2. Related measured properties: maximum surface excess (Gamma_max), area per molecule, Krafft point, pC20, cloud point of nonionic surfactants.
3. Any open dataset that measures the cleaning performance of formulations (detergency, soil removal by a standard test) together with the formulation's composition. Say plainly if none exists.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### F6. Corrosion rates

```
I am building a model that predicts the corrosion rate (mm/yr, as log10) of a named metal or alloy in a named chemical environment, given concentration, temperature and exposure time.

Already have (do not list again): NIST CORR-DATA.
Already found (do not list again): Nyby et al. 2021 electrochemical metrics for corrosion-resistant alloys (Scientific Data 8:58); Birbilis et al. 2024 MPEA corrosion database (Mendeley 10.17632/nskb9khgsm); Ghorbani and Birbilis additively-manufactured MPEA corrosion (Mendeley 10.17632/cfz68tkwfk); CORDATA (datacor.shinyapps.io/cordata); DATACOR V2 (Mendeley 10.17632/v5p322m2t8); Ozkan et al. 2024 AA2024-T3 inhibitor library (npj Mater. Degrad. 8:21); the 2026 LLM-extracted inhibitor set (Mendeley 10.17632/wm65mb6p29); Kaboudvand et al. seawater compilation (arXiv 2508.11685).

Find:
1. Immersion (weight-loss) corrosion-rate datasets that state the standard (for example ASTM G31, NACE TM0169) with material, environment, concentration, temperature and duration.
2. Open data from atmospheric and marine exposure programmes (for example ISO CORRAG, MICAT, UNECE ICP Materials) with numeric first-year or long-term rates.
3. Chemical-compatibility tables that give numeric rates rather than only A/B/C/D ratings, and whose licence allows reuse.
4. For each, whether the rate comes from weight loss, electrochemistry (polarisation resistance, Tafel) or thickness loss.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### F7. Solid lithium electrolytes: ionic conductivity

```
I am building a model that predicts the room-temperature ionic conductivity (S/cm, as log10) of lithium solid electrolytes from their composition, and later from their crystal structure. My training set is small (about 480 rows), so every measured row matters.

Already have (do not list again): OBELiX.
Already found (do not list again): the Liverpool Ionics LiIon database (Hargreaves et al. 2023, npj Comput. Mater. 9:9); DigBat, formerly DDSE (Tohoku WPI-AIMR); Laskowski, McHaffie and See 2023 (Energy Environ. Sci. 16:1264); Shon and Min 2023 text-mined database (ACS Omega 8:18122); Kim et al. 2026 LiIon + OBELiX structure set (Zenodo 10.5281/zenodo.17157647); Devi et al. 2025 DFT migration barriers (arXiv 2508.06459).

Find:
1. Measured conductivity datasets that give the temperature of each value, or the activation energy, so conductivity can be modelled versus temperature.
2. Whether each value is total, bulk or grain-boundary conductivity, and the pellet density or sintering conditions.
3. Measured electrochemical stability windows of solid electrolytes.
4. Sodium, magnesium or potassium solid-electrolyte conductivity datasets (as separate tasks).
5. The current access terms for DigBat and for the LiIon database (can the data be downloaded and redistributed?).

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### F8. Insulators: dielectric constant

```
I am building a model that predicts the static dielectric constant of inorganic crystals from composition and structure. My current labels are DFT-calculated (density-functional perturbation theory), and I need measured values as well.

Already have (do not list again): the Materials Project dielectric dataset (Petousis et al. 2017, 1,056 compounds, via Foundry-ML) and the JARVIS-DFT DFPT dielectric tensors.
Already found (do not list again): Zhao and Cole 2022 text-mined dielectric constants and refractive indices (Scientific Data 9:192); the JARVIS-Leaderboard experimental dielectric benchmark (16 values).

Find:
1. Measured static or low-frequency dielectric constants of inorganic crystals or ceramics, with frequency and temperature, ideally matched to a crystal-structure ID (ICSD, Materials Project, COD).
2. Open datasets of microwave dielectric ceramics: relative permittivity, quality factor (Q x f) and temperature coefficient of resonant frequency.
3. Measured dielectric breakdown strength of insulators.
4. Measured refractive indices of inorganic crystals matched to structures.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### F9. Crystals: measured values for models trained on DFT

```
I train crystal graph neural networks on DFT-calculated properties from JARVIS-DFT (OptB88vdW functional): formation energy per atom, band gap, and bulk and shear moduli (Voigt averages). I need MEASURED values for crystals whose structure is known, to test how far DFT-trained predictions are from experiment.

Already have (do not list again): JARVIS-DFT dft_3d (2025-09-24 release).
Already found (do not list again): Kim, Meschel, Nash and Chen 2017 calorimetric formation enthalpies (Scientific Data, 10.1038/sdata.2017.162); the Wang et al. 2021 / Kingsbury experimental formation enthalpies (2,135); SSUB via OQMD; Zhuo, Mansouri Tehrani and Brgoch 2018 band gaps / matbench_expt_gap; RealMat-BaG (github.com/Shef-AIRE/bandgap-benchmark); Dong and Cole 2022 BandgapDatabase1; Borlido et al. 2019 (472 solids); the JARVIS-Leaderboard experimental benchmarks; de Jong et al. 2015 elastic constants (DFT). Also known: Materials Project, OQMD, AFLOW, Alexandria, LeMat-Bulk, SNUMAT, GNoME, OMat24, MatPES.

Find:
1. Open compilations of measured single-crystal elastic constants or bulk and shear moduli, matched to a structure ID. Say plainly if only paywalled handbooks (for example Landolt-Bornstein) hold them.
2. Measured standard formation enthalpies of inorganic solids beyond the sets above, for example open thermochemical tables (NIST-JANAF, NBS tables): what is downloadable and under what terms.
3. Measured band gaps with a structure ID and the measurement method (optical absorption, photoluminescence, ellipsometry) and temperature.
4. Published systematic offsets between OptB88vdW (or PBE) values and experiment for each of these four properties, with the number of compounds compared.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

---

## Part X: data quality, across all families

### X1. Which public datasets are copies of each other

```
In materials and chemistry machine learning, public datasets are often re-compiled from the same original sources. For each of these widely used datasets, list the later public datasets that include all or part of it, with how many rows overlap if the authors state it:
- UCI Concrete Compressive Strength (Yeh 1998)
- Borg et al. 2020 multi-principal-element-alloy mechanical properties (Scientific Data 7, 430)
- AqSolDB (Sorkun et al. 2019)
- the Materials Project dielectric dataset (Petousis et al. 2017)
- OBELiX and the Liverpool LiIon database (lithium solid electrolytes)
- the NIST CMC compilation (Mukerjee and Mysels 1971)

Also list published papers that measured how much duplication or train/test leakage inflated reported accuracy on any of these, and by how much.

Rules: give a link for every claim (DOI preferred), write "not stated" when a source does not say, and do not include superconductors, biology or medicine, or explosives. Answer as a table: original dataset | later dataset (link) | overlap as stated | how you know (quote or section).
```

### X2. The measurement-noise floor for each property

```
For each property below, find the published repeatability and reproducibility of the measurement: how far apart two measurements of the same material are, within one laboratory and between laboratories. Use standards' precision statements (ASTM, ISO, EN) and inter-laboratory round-robin studies.
- Concrete compressive strength of cylinders (for example the ASTM C39 precision statement) and of cubes (EN 12390-3)
- Vickers hardness of metals (ASTM E384 / E92, ISO 6507)
- Polymer glass-transition temperature by DSC (for example ASTM E1356, ISO 11357-2) and by DMA
- Aqueous solubility of organic compounds (log10 units), shake-flask and other methods
- Critical micelle concentration of surfactants, and surface tension at the CMC
- Corrosion rate by immersion weight loss (ASTM G31) and by electrochemical methods
- Ionic conductivity of solid electrolytes by impedance spectroscopy (inter-laboratory studies)
- Static dielectric constant of ceramics
- Formation enthalpy by calorimetry; optical band gap; elastic moduli by ultrasound or resonance

Give each value as a standard deviation or a 95% limit, with its unit, the number of laboratories and materials it is based on, and a link. If a value is a rule of thumb rather than a measured study, say so. Write "not found" rather than estimating. Do not include superconductors, biology or medicine, or explosives.
```

### X3. Same name, different measurement

```
I am merging property data from many sources. For each property below, list the different measurement or calculation definitions that are commonly reported under the same name, and the published size of the systematic difference between them, with links:
- concrete compressive strength: cylinder vs cube, specimen size, curing regime
- hardness: Vickers vs nanoindentation vs converted values
- glass-transition temperature: DSC midpoint vs onset vs DMA tan-delta peak vs loss-modulus peak, heating rate
- aqueous solubility: thermodynamic vs kinetic, intrinsic vs at a given pH, temperature
- CMC: tensiometry vs conductivity vs fluorescence; with and without added salt
- corrosion rate: weight loss vs polarisation resistance vs Tafel extrapolation; short vs long exposure
- ionic conductivity: total vs bulk vs grain boundary; pellet density
- DFT elastic moduli: Voigt vs Reuss vs Voigt-Reuss-Hill averages; functional (PBE, PBEsol, OptB88vdW)
- DFT band gap: PBE or OptB88vdW vs TBmBJ vs HSE vs measured optical gap

Give numbers only where a source states them. Do not include superconductors, biology or medicine, or explosives.
```

---

## Part P: physics the models should obey

### P1. Laws and constraints, per family

```
I build data-driven models (gradient-boosted trees and neural networks) for the properties below. For each one, list the established physical laws, limits and constraints that any correct model must obey: monotonic trends, bounds, limiting cases, invariances, conservation laws and known functional forms. For each, give the equation or statement, its range of validity, and a textbook or review reference with a link:
- concrete strength vs water-to-cement ratio and age (for example Abrams' law, Powers' gel-space ratio, ACI 209 or fib Model Code strength-development curves)
- alloy hardness and strength vs composition and grain size (for example rule of mixtures, Hall-Petch, solid-solution strengthening models for high-entropy alloys such as Varvenne-Curtin)
- polymer Tg vs structure and molecular weight (for example Flory-Fox, Fox equation for copolymers and blends)
- aqueous solubility (for example the general solubility equation using melting point and logP)
- CMC vs alkyl chain length, head group, temperature and salt (for example the Klevens relation, Corrin-Harkins)
- corrosion rate vs temperature, concentration and time (for example Arrhenius behaviour, parabolic or power-law kinetics)
- ionic conductivity vs temperature (Arrhenius, Vogel-Tammann-Fulcher) and vs composition
- crystal properties: elastic stability (Born criteria), positive moduli, formation energy and the convex hull, band gap of zero or more, rotational and translational invariance, extensive vs intensive quantities

Then list published studies that built such constraints into machine-learning models (as inputs, as loss terms, as monotonic constraints, or as a physics baseline whose error the model learns) and reported numbers on how accuracy changed, especially when extrapolating. Give the numbers as reported. Do not include superconductors, biology or medicine, or explosives.
```

### P2. Physics baselines: how good is the formula alone?

```
For each property below, find published results that compare a physics-based or semi-empirical formula with machine learning on the same public data, with the error of each (MAE or RMSE, in the property's unit) and the dataset used:
- concrete strength: Abrams' law or Powers' model vs ML on the UCI concrete data or similar
- aqueous solubility: the general solubility equation vs ML on AqSolDB, ESOL or the Solubility Challenge sets
- polymer Tg: group-contribution methods (for example van Krevelen) vs ML
- CMC: group-contribution or Klevens-type correlations vs ML
- formation enthalpy: the Miedema model vs DFT vs ML
- alloy strength: solid-solution strengthening models vs ML for high-entropy alloys
- ionic conductivity: bond-valence or descriptor baselines vs ML

Report the numbers exactly as stated, with links. If a comparison does not exist, say so. Do not include superconductors, biology or medicine, or explosives.
```

---

## Part M: other ways to get more information

### M1. Learning from cheap computed data plus scarce measurements

```
I have a few hundred to a few thousand MEASURED values per property (concrete strength, alloy hardness, polymer Tg, aqueous solubility, surfactant CMC, corrosion rate, solid-electrolyte ionic conductivity), and for some properties many more COMPUTED values (DFT, molecular dynamics, machine-learned interatomic potentials).

Find published, quantified evidence on which of these methods improve accuracy on small measured tabular or molecular datasets, and by how much (error before and after, same test split):
- multi-fidelity learning (combining computed and measured labels)
- pretraining on large computed datasets, then fine-tuning on measured ones
- using a computed property as an input feature (for example a DFT band gap to predict a measured gap)
- physics-generated synthetic training data
- active learning (choosing which new measurement would help most)
- data augmentation for tabular materials data

For each study give the dataset, how many measured rows, the improvement as reported, and a link. Note any studies where these methods made results worse. Do not include superconductors, biology or medicine, or explosives.
```

### M2. Making our own computed data on one laptop GPU

```
I have one laptop with an 8 GB NVIDIA GPU. Which free, openly licensed tools can compute useful labels or features for these properties in a reasonable time, and how accurate are they against experiment?
- universal machine-learned interatomic potentials (for example MACE-MP, CHGNet, MatterSim, SevenNet, ORB) for formation energies, elastic constants and relaxed structures of inorganic crystals
- semi-empirical quantum chemistry (for example GFN2-xTB) for molecular descriptors useful for solubility and surfactant properties
- molecular dynamics of polymers for Tg (and how far simulated Tg is from measured)

For each tool: licence, typical time per structure or molecule on a consumer GPU or CPU, the published error against experiment or DFT (with the benchmark name), and a link. Say what is known to fail. Do not include superconductors, biology or medicine, or explosives.
```

---

## Part N: new categories (data to judge whether to start them)

The project's candidate list (from this week's research) is:

- thermoelectrics;
- high-entropy and refractory alloy strength at temperature;
- solar absorbers;
- glasses;
- magnetic ordering temperatures;
- piezoelectrics;
- hydrogen-storage alloys;
- catalysts;
- porous materials for CO2 capture;
- battery electrodes;
- fuel-cell perovskites;
- thermal-barrier coatings;
- composites.

The cached JARVIS file already holds:

- 23,218 Seebeck and power-factor labels;
- 10,022 SLME solar-efficiency labels;
- 21,559 TBmBJ band gaps;
- 3,347 DFPT piezoelectric labels.

All of these are DFT-calculated. The prompts below look for measured data.

### N1. Thermoelectrics

```
I want to start a model for thermoelectric materials: measured Seebeck coefficient, electrical conductivity, thermal conductivity and figure of merit zT versus temperature, from composition (and structure where known).
Already found (do not list again): the JARVIS-DFT computed Seebeck and power factor; Ricci et al. 2017 computed transport (10.1038/sdata.2017.85); ESTM (Na and Chang 2022, github.com/KRICT-DATA/SIMD); Sierepeklis and Cole 2022 text-mined thermoelectric database (10.1038/s41597-022-01752-1); Starrydata2.
Find other measured, openly downloadable thermoelectric datasets, and state for each whether values are digitised from figures, at which temperatures, and the doping level. Also find and quote the licence or terms of use of Starrydata2 (I have seen both "All rights reserved" and "free for commercial and non-commercial use") and of ESTM.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### N2. Alloy yield strength at temperature

```
I want to model the yield strength (MPa) of high-entropy, refractory and nickel-based alloys as a function of composition, processing, test type (tension or compression) and test temperature.
Already have (do not list again): Borg et al. 2020 MPEA dataset (Scientific Data 7, 430); Couzinie, Senkov, Miracle and Dirras 2018 (10.1016/j.dib.2018.10.071); Gorsse, Nguyen, Senkov and Miracle 2018 (10.1016/j.dib.2018.11.111). Known but restricted: ULTERA.
Find other measured, openly downloadable datasets of strength versus test temperature for these alloys, including creep or high-temperature compression data. State the strain rate where given. Also say whether ULTERA can now be downloaded, and under what terms.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### N3. Glasses

```
I want to model oxide glass properties from composition: glass-transition temperature, density, Young's modulus, hardness and refractive index.
Already found (do not list again): SciGlass (github.com/epam/SciGlass) and GlassNet (Cassar 2023); for metallic glasses, the Foundry-ML maximum-casting-diameter and critical-cooling-rate sets and matbench_glass. The SciGlass licence is unclear to me: one source says MIT, another says ODbL. Please find the licence stated in the repository's own LICENSE file and quote it.
Find other measured, openly downloadable glass-property datasets (oxide, chalcogenide, metallic glasses), and any published leakage-safe benchmark splits for SciGlass.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### N4. Magnetic ordering temperature

```
I want to model the measured Curie and Neel temperatures of magnetic materials from composition (and structure where known).
Already found (do not list again): the JARVIS-DFT computed magnetic moments; NEMAD (Itani, Zhang and Zang 2025); the text-mined Curie/Neel databases of Court and Cole 2018 and Kelly and Cole 2025; the Foundry-ML Heusler magnetisation set.
Find other measured, openly downloadable Curie and Neel temperature datasets, and any measured datasets of saturation magnetisation or coercivity for permanent-magnet materials. Also quote the licence or terms of NEMAD and of the Court and Cole database.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### N5. Hydrogen-storage alloys

```
I want to model metal-hydride properties from alloy composition: hydride formation enthalpy and entropy, plateau pressure and maximum hydrogen capacity (wt%).
Already found (do not list again): ML-HydPARK (Witman, Allendorf and Stavila, Zenodo 10680097), derived from the DOE HydPARK database; HyStor (Int. J. Hydrogen Energy 90, 10.1016/j.ijhydene.2024.09.390); DigHyd (reported to hold about 5,089 pressure-composition-temperature entries).
Find other measured, openly downloadable metal-hydride datasets. State the current licence and status of the original DOE HydPARK data, and whether DigHyd can be downloaded and under what terms.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### N6. Solar absorbers and perovskite solar cells

```
I want to model solar-absorber materials: measured band gap and absorption, and measured device efficiency of perovskite and other thin-film solar cells from composition and processing.
Already found (do not list again): the JARVIS-DFT computed SLME and TBmBJ band gaps; the Perovskite Database Project (Jacobsson et al. 2022, perovskitedatabase.com) and its Hugging Face mirror jablonkagroup/perovskite_db; matbench_expt_gap.
Find other measured, openly downloadable datasets of absorber band gaps or solar-cell device performance, with the licence and how device-to-device variation is reported. Also quote the data licence of the Perovskite Database Project itself (not the paper's licence).

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### N7. The other material candidates, in one sweep

```
For each of these material categories, find the single best measured (not computed), openly downloadable dataset, if one exists, and say plainly when none does:
- piezoelectric coefficients (d33) of ceramics
- electrocatalyst activity (overpotential, Tafel slope) with catalyst composition
- CO2 uptake of porous materials (MOFs, zeolites) at stated pressure and temperature
- battery electrode measured capacity and average voltage
- solid-oxide fuel-cell cathode area-specific resistance
- thermal conductivity of thermal-barrier coating ceramics
- tensile strength and stiffness of fibre-reinforced composites with layup and fibre fraction

Already found (do not list again): OC20/OC22, OCx24 and ODAC23 (Open Catalyst / fair-chem); Catalysis-Hub; QMOF, CoRE MOF 2019, hMOF; the NIST/ARPA-E ISODB adsorption database (SRD-205); Materials Project insertion electrodes; the Foundry-ML perovskite area-specific-resistance, conductivity and thermal-expansion sets; the JARVIS-DFT piezoelectric tensors.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```

### N8. New chemistry properties

```
I want to add measured chemistry properties of organic compounds, predicted from molecular structure. For each property below, find the best measured, openly downloadable datasets (licence, size, method, temperature):
- octanol-water partition coefficient (logP)
- acid dissociation constant (pKa) in water
- melting point and boiling point
- vapour pressure and enthalpy of vaporisation
- liquid viscosity versus temperature
- flash point
- Henry's law constants
- ionic conductivity of liquid battery electrolytes (salt, solvents, concentration, temperature)

Exclude biological activity, toxicity, drug metabolism and anything medical. Also exclude explosives and energetic materials.

Rules for your answer:
- Only list a dataset if you found its actual data record: a DOI, or a Zenodo, figshare, Mendeley Data, Dryad, GitHub, Hugging Face or government database page. Give that link. Data "available on request" does not count; list those papers separately at the end, one line each.
- Write "not stated" when the page does not say. Do not guess a licence or a row count.
- Do not include superconductors or superconducting properties, biology or medicine, or explosives/energetic materials.
- Answer with one table, one row per dataset, with these columns:
  1. Name, authors, year
  2. Direct link to the data (DOI preferred)
  3. Licence exactly as written on the data page (for example CC BY 4.0, CC0, MIT, ODbL, "no licence stated")
  4. Rows with a usable value of the target, and how many distinct materials or compounds
  5. Measured in a lab, or computed (DFT, molecular dynamics, ML-predicted), or text-mined/LLM-extracted (give the reported precision)
  6. Test method and conditions (standard, specimen, temperature, time, ...)
  7. Input columns given (list them)
  8. Overlap: does it include or re-publish any dataset I listed as "already have"? How many rows, if stated?
  9. What you actually opened: the data file, the data landing page, or only the paper or abstract
- After the table: the three you would add first and why, then any data-quality traps you saw (duplicates, mixed units, mixed test standards, computed values mixed with measured ones).
```
