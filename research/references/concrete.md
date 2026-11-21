# References — Concrete (live app)

Sourced numbers behind the concrete mix designer. Canonical values live in
`public/model/factors.json`; this file mirrors them in citation form. All were retrieved through
literature-search tools, and a second, independent search was run for each. That second search did
not confirm three densities (slag, fly ash, superplasticizer), and the slag and fly ash CO2 factors
were seen only in a secondary table; each entry below says which. PDFs were not opened by hand.

> Factors are a **consistent way to compare mixes**, not exact footprints. Published values differ,
> some by a lot. The Flower & Sanjayan factors are for Australian concrete and include transport to
> the batching plant.

### Portland cement — CO2 0.82 kgCO2e/kg, density 3150 kg/m3
- **Used in:** `public/model/factors.json`, the app's CO2 ranking
- **CO2 source:** Flower & Sanjayan (2007), *Int. J. Life Cycle Assessment* — https://doi.org/10.1007/s11367-007-0327-3 (confirmed by a second search)
- **Density source:** Chandrakar & Singh (2017), specific gravity 3.15 — https://scispace.com/papers/an-experimental-study-on-high-performance-concrete-using-24q3e. **Caveat:** the paper is listed on SciSpace but we could not find it in Crossref or OpenAlex. The value 3.15 itself was confirmed in other abstracts by a second search.

### Blast furnace slag — CO2 0.143 kgCO2e/kg, density 2880 kg/m3
- **CO2 source:** Flower & Sanjayan (2007), as tabulated in Kubissa, Jaskulski et al. (2019), *Architecture Civil Engineering Environment*, Table 1 — the link is to the original paper: https://doi.org/10.1007/s11367-007-0327-3. **Caveat:** seen only in that secondary table, and we did not record the secondary paper's title or DOI.
- **Density source:** Sudharsan et al. (2022), SG 2.88 — https://doi.org/10.22214/ijraset.2022.41047. **Caveat:** not confirmed by a second search.

### Fly ash — CO2 0.027 kgCO2e/kg, density 2200 kg/m3
- **CO2 source:** Flower & Sanjayan (2007), via Kubissa et al. (2019). **Caveat:** secondary table only.
- **Density source:** Sudharsan et al. (2022), SG 2.2. **Caveat:** not confirmed.

### Superplasticizer — CO2 2.98 kgCO2e/kg, density 1080 kg/m3
- **CO2 source:** Schiefer & Plank (2023), *J. Cleaner Production* — https://doi.org/10.1016/j.jclepro.2023.138785 (confirmed). **Caveat:** CO2 only; highest of three polymer types (2.40–2.98); basis (dry vs liquid) unstated; 1998 mixes may differ.
- **Density source:** Sudharsan et al. (2022), SG 1.08 for one product. **Caveat:** not confirmed.

### Coarse aggregate — CO2 0.0459 kgCO2e/kg, density 2680 kg/m3
- **CO2 source:** Flower & Sanjayan (2007), granite/hornfels (basalt is 0.0357) (confirmed).
- **Density source:** Sudharsan et al. (2022), SG 2.68 (2.6–2.7 range confirmed).

### Fine aggregate (sand) — CO2 0.0139 kgCO2e/kg
- **CO2 source:** Flower & Sanjayan (2007). See `public/model/factors.json` for the full entry.

### Water — CO2 0 (not counted), density 1000 kg/m3
- **Assumption:** no sourced factor found; Flower & Sanjayan describe water's emissions as minimal. Density by definition of specific gravity.

### Dataset — UCI Concrete Compressive Strength (1,030 results)
- **Source:** Yeh (1998), via UCI Machine Learning Repository. Used in `train/train.py`.
