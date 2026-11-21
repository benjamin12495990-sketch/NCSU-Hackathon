# Formulation Designer: low-carbon concrete mixes

A one-page web app that proposes new concrete mixes for a strength you choose and ranks them by
embodied CO2. Built from scratch at TRIANGULATE, an AI hackathon at NC State.

- **Grand Challenge:** Engineer the Tools of Scientific Discovery.
- **User:** a civil or materials engineering researcher choosing which trial mixes to cast.
- **Problem:** every strength test means casting a mix and waiting up to 28 days for it to cure, so
  only a handful of the possible mixes ever get tried.

## The page

The page is titled **Formulation Designer**. A row of pills at the top lists seven families of
materials. **Concrete is live.** The other six (polymers, corrosive solutions, cleaning products,
insulators, general materials, general chemistry) are marked "Planned": pressing one shows what it
would predict, vary and rank by, and says that nothing is generated for it yet. The page follows
the visual rules in [`taste.md`](taste.md).

## What it does

Enter a target strength in MPa and a curing age. In the browser, with no server, the app:

1. generates about 12,000 mixes by varying each of the 306 lab-tested mixes in the dataset
   (binder amount, share of slag and fly ash, water/binder ratio, superplasticizer dose), rescaling
   the aggregates so each mix keeps its parent's volume;
2. removes any mix that already exists in the dataset: within 5 kg of a tested mix on every
   ingredient, the same rule the training script uses to decide what counts as one mix;
3. predicts each mix's strength with two models trained for this project: the expected strength,
   and a cautious lower estimate;
4. keeps the mixes whose cautious estimate reaches the target (or the expected strength, if you turn
   caution off);
5. ranks them by embodied CO2 per cubic metre and gives each a formula name built from its recipe,
   for example `Ternary blend C30-S35-F35-W40` (binder shares of cement, slag and fly ash in
   percent, then water ÷ binder × 100). The names are labels for this page, not standard
   designations;
6. checks each against its three nearest lab-tested mixes and marks it "neighbours passed" only when at
   least two of them are close to it and reached the target in the lab.

It also shows the lowest-CO2 mix that a lab has already tested for that target, so you can see
whether a new candidate is worth a trial at all.

## Environment sliders and the materials shortlist

Under "What does it have to survive?" each requirement has a switch that enables its slider:
humidity and water, freezing temperatures, highest service temperature, wear, and service life.
Choosing a kind of work (building, pavement, marine, aerospace and so on) presets the switches.

- **For the concrete design**, humidity and freezing map to ACI 318-19 exposure classes (W1, W2,
  C1, C2, F1 to F3). Each class caps the water/binder ratio and sets a minimum strength; the
  designer applies the strictest of the ones switched on. Wear from vehicles upward adds a 28 MPa
  minimum. Service temperature and service life do not change the generated mixes.
- **For comparison**, the page ranks 32 existing reference materials from six families (concrete and
  masonry, ferrous metals, light alloys, high-temperature alloys and ceramics, polymers, composites
  and timber). A material is left out if it fails a requirement that is switched on. The rest are
  ranked by CO2 per unit of strength, mass per unit of strength, or strength.
- **At the research frontier** lists eight research-stage materials, each with one headline number
  read in the abstract of the linked paper.

## The 3D view

The chosen mix is drawn as one cubic metre of particles (Three.js, `src/components/MixStage.jsx`).
The number of particles of each colour follows that ingredient's share of the mix's volume (mass
divided by density, from `factors.json`). Particle sizes and positions are **not** to scale, and
the page says so under the model. Cement pulses because it is most of the CO2. The three headline
mixes and every table row can be put on the stage.

The visual rules for the page are in [`taste.md`](taste.md).

## What is live

Everything on the page is computed in your browser from the JSON files in `public/model/`. Nothing
is mocked. There is no backend, no API and no API key.

Two kinds of numbers are not measured by us and are labelled on the page:

- **CO2 factors and densities** come from published papers (table on the page, with links). Water is
  not counted because we found no sourced factor.
- **The three preset strengths** are illustrative, not values from a building code.
- **Environment limits** are ACI 318-19 values as reported by secondary sources. The code itself is
  paywalled and we did not open it. Class C1 and the wear minimum are recalled values we could not
  confirm, and the page marks them.
- **The materials table** was compiled by AI research agents. 56 of its 232 numbers were seen in a
  search tool's summary of a cited paper (some after a unit conversion) and carry a dot on the page. The other 176 are typical handbook-level values recalled
  by the AI and not individually checked. The 1 to 5 humidity and wear ratings are AI judgments.
- **Kinds of work and the service-life rule** (50 years or more asks for one more step of humidity
  and wear resistance) are our own starting points, not values from a standard.
- **The 3D model** is an illustration; the page says so under it.
- **The chemical formula** under each mix name is an estimate. Concrete is a mixture, not one
  compound, so the page shows the bulk composition of the paste (cement, slag, fly ash and water)
  per silicon atom, for example `Ca0.93 Si Al0.53 Fe0.11 Mg0.11 S0.01 O4.02 · 4.09 H2O`. Aggregates
  and admixture are left out. The oxide analyses behind it are of mixed quality and none is a
  single verified analysis: cement and slag are midpoints of published ranges (the slag one from a
  patent abstract), and fly ash comes from one abstract with three oxides missing, counted as zero.
  They describe other materials than the dataset's own, whose chemistry is not recorded. The
  calcium figure depends most on the slag value, which is uncertain.

Not used by the live page: the neural network in `pipeline/` (an alternative model that turned out
less accurate) and the folders `domains/` and `docs/`, which hold plans for other domains. Only
concrete is built.

## How the models were trained

`train/train.py` reads the UCI Concrete Compressive Strength dataset: 1,030 lab results, of which
25 are exact repeats and are dropped.

- **Features (9):** kg per m3 of cement, blast furnace slag, fly ash, water, superplasticizer,
  coarse aggregate and fine aggregate; age in days; and water divided by total binder.
- **Expected-strength model:** scikit-learn `GradientBoostingRegressor`, 300 trees, depth 4,
  learning rate 0.05, subsample 0.8.
- **Cautious model:** the same settings with quantile loss at the 10th percentile, so about 9 lab
  results in 10 should beat it.
- **Split:** 80/20 **by mix**. The dataset holds the same mix at several ages, and the same mix
  twice with different rounding (185.7 kg in one row, 186 in another). Recipes that agree within
  5 kg on every ingredient count as one mix, and every result of a test mix is held out. The
  script checks that no test recipe has a training recipe that close.

| Test metric (62 held-out mixes, 219 lab results) | Value |
| --- | --- |
| Mean absolute error | 3.29 MPa |
| Root mean squared error | 4.37 MPa |
| R² | 0.912 |
| Share of test results at or above the cautious estimate | 87.7% (the aim was 90%) |
| Mean absolute error with a plain random split, for comparison only | 2.94 MPa |

The random-split number is the one most papers report on this dataset. It looks better because the
same mixes sit on both sides of the split. The live numbers are in `public/model/metrics.json`, and
the page reads them from there.

## Known failure modes

- **The search rewards model error.** Ranking by lowest CO2 pulls toward mixes where the model is
  too optimistic. Our first version proposed a mix predicted at 43 MPa whose nearest tested mix,
  with more cement at the same water/binder ratio, measured 29 MPa. The cautious model and the
  neighbour check were added because of that. They reduce the problem; they do not remove it.
- **With caution on, candidates often do not beat the best tested mix.** At 28 days the best
  cautious candidate is 5% lower in CO2 than the best tested mix at 25 MPa, but 20% higher at
  35 MPa, and none is found at 55 MPa or above. The page opens on 25 MPa. That is an honest result, not a bug:
  the tested mix has a real measurement, and a candidate must clear a stricter bar.
- **With caution off, candidates are experiments, not answers.** Their cautious estimates fall
  short of the target.
- **The cautious estimate is less cautious than intended** (87.7% coverage, not 90%). Where the
  two models cross, the app uses the lower of the two.
- **Strength only.** Nothing here models workability, setting time, durability, shrinkage or cost.
- **One dataset.** Published work reports large accuracy drops when a model trained on this dataset
  is tested on mixes from other sources.
- **Materials are not uniform.** The data records kilograms, not the chemistry or source of the
  slag, fly ash or aggregate.
- **CO2 totals are comparative, not exact.** Factors are from Australian data and include transport
  to the batching plant. Slag and fly ash factors were seen only in secondary tables. The
  superplasticizer factor is probably too high. Densities for slag, fly ash and superplasticizer
  come from a single paper and were not confirmed by a second search.
- **"New" means "not in this dataset".** A candidate may exist in the literature.
- **Exposure classes are only part of durability design.** ACI also requires air entrainment for
  freezing exposure, which the dataset does not record and the app does not model. The service
  temperature slider does not change the concrete design.
- **The materials ranking compares unlike strengths.** Yield, ultimate, compressive and flexural
  strengths sit in one column, and most values are single typical figures. It is a way to build a
  shortlist, not a substitute for a datasheet.
- **European limits were dropped.** We looked for the EN 206 exposure table too, but could confirm
  only 11 of its 60 values tonight, so the app uses ACI 318 only.

## How we used MCP servers

The CO2 factors, densities and accuracy figures were found and cross-checked tonight through MCP
servers connected to the coding agent. Six agents ran in parallel: two searched, two independently re-checked every
number with a different search, one gathered expert objections, and one looked for more servers.

| MCP server | What it was used for |
| --- | --- |
| SciSpace | Found the CO2 factors and densities, and the published accuracy figures. |
| Consensus | Confirmed the superplasticizer factor from a second source; expert objections. |
| FastTrack Literature Open | Confirmed the dataset paper and resolved the DOI of the CO2 source. |
| OpenAlex | Connected late; used to check the project's reference list (6 of 8 references matched). |
| Wiley Scholar Gateway, Firecrawl | Connected late; not used for anything in this project. |
| alphaXiv | Searched preprints; nothing from it is on the page. |
| Elicit | Tried; the account's plan has no API access, so nothing came back. |
| Claude Market and the connector registry | Searched for concrete, cement, life cycle and deployment servers. None exist for concrete. |
| Tenjin router | Installed. No paid call was made and no number on the page comes from it. |

What the checks changed: water has no CO2 factor because none could be sourced; slag and fly ash
are labelled as secondary; three densities are labelled as unconfirmed.

A second round of nine agents built the environment and materials data: two looked up exposure
limits (ACI 318 and EN 206), six compiled the materials table one family each, and one wrote the
frontier brief. They used SciSpace, Consensus and FastTrack Literature Open, plus web search for
pages that summarise ACI 318. Their raw output, with citations and quotes, is in `research/`.
Two reviewer agents also checked the code and the claims; their findings led to the stricter
train/test split and several wording changes.

## Files

| File | What it does |
| --- | --- |
| `train/train.py` | Trains both models, measures test error, and writes the JSON files below. |
| `train/data/concrete.csv` | The UCI dataset with short column names. |
| `public/model/features.json` | Ingredient and feature order. Python and JavaScript both follow it. |
| `public/model/model.json` | Expected-strength trees: feature, threshold, children and leaf value per node. |
| `public/model/model_low.json` | The cautious model, in the same format. |
| `public/model/known.json` | Every distinct tested mix with its measured strength at each age. |
| `public/model/space.json` | The ranges the labs covered. Generated mixes must stay inside them. |
| `public/model/metrics.json` | Test metrics, row counts, training date and the neighbour distance. |
| `public/model/parity.json` | 20 test rows with Python predictions from both models. |
| `public/model/test_points.json` | Measured and predicted strength for the test set (the chart). |
| `public/model/factors.json` | CO2 factors and densities with their sources. Written by hand from the literature search. |
| `src/lib/featurize.js` | Builds the same 9 features as Python. |
| `src/lib/predict.js` | Walks the trees and adds up the leaves. |
| `src/lib/co2.js` | Embodied CO2 and solid volume of a mix. |
| `src/lib/nearest.js` | Distances to tested mixes and their lab results. |
| `src/lib/design.js` | Generates, filters, ranks and checks the candidates. |
| `src/lib/name.js` | Builds each mix's formula name from its recipe. |
| `src/lib/formula.js` | Estimates the paste's chemical formula per silicon atom from oxide analyses. |
| `src/lib/domains.js` | The seven families of materials and which one is live. |
| `src/components/Formula.jsx` | Shows the formula with subscripts. |
| `src/components/Planned.jsx` | The "Planned, not built" card for the six other families. |
| `public/model/oxides.json` | Oxide analyses of cement, slag and fly ash, with an evidence label each. |
| `src/lib/loadModel.js` | Downloads the JSON files when the page opens. |
| `public/model/exposure.json` | ACI 318 exposure limits used by the sliders, with an evidence label per class. |
| `public/model/materials.json` | The 32 reference materials, with which values were sourced. |
| `public/model/frontier.json` | The eight frontier materials. |
| `research/*.json` | Raw output of the research agents: citations, quotes and evidence labels. |
| `src/lib/exposure.js` | Turns the environment sliders into concrete limits. |
| `src/lib/rank.js` | Filters and ranks the reference materials. |
| `src/lib/particles.js` | Turns a mix into particles for the 3D view. |
| `src/components/Requirements.jsx` | The switches and sliders. |
| `src/components/Shortlist.jsx` | The limits table, the materials shortlist and the frontier cards. |
| `src/components/MixStage.jsx` | The 3D view (Three.js). |
| `src/App.jsx` | The page. |
| `pipeline/train_nn.py` | An alternative neural-network model in NumPy. Not used by the live page. |
| `docs/SCOPE.md`, `domains/` | Plans for applying the same method to other domains. Nothing there is built. |
| `research/references/` | Citation notes per domain. Only `concrete.md` has entries. |
| `scripts/parity.mjs` | Checks that JavaScript predictions match Python. |

## Run it

```sh
npm install
npm run dev      # local page
npm run parity   # JavaScript vs Python on 20 rows and both models, must match within 0.01 MPa
npm run build    # static site in dist/
```

Retrain (optional, overwrites the model files in `public/model/` but not `factors.json`):

```sh
pip install -r train/requirements.txt
python train/train.py
npm run parity
```

## Project history

The first commits tonight built a superconductor screening app. The team then switched the demo to
concrete, and those files were replaced. All code in this repository was written during the
hackathon.

## Sources

Dataset: Yeh, I-C. (1998). Concrete Compressive Strength [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5PK67

Yeh, I-C. (1998). Modeling of strength of high-performance concrete using artificial neural
networks. *Cement and Concrete Research*, 28(12), 1797–1808.
https://doi.org/10.1016/S0008-8846(98)00165-3

CO2 factors: Flower, D. J. M. & Sanjayan, J. G. (2007). Green house gas emissions due to concrete
manufacture. *International Journal of Life Cycle Assessment*.
https://doi.org/10.1007/s11367-007-0327-3

Schiefer, C. & Plank, J. (2023). CO2 emission of polycarboxylate superplasticizers (PCEs) used in
concrete. *Journal of Cleaner Production*. https://doi.org/10.1016/j.jclepro.2023.138785

Kubissa, W., Jaskulski, R. et al. (2019). *Architecture Civil Engineering Environment*: the
secondary table for the slag and fly ash factors.

Environment limits: ACI 318-19, Building Code Requirements for Structural Concrete, Table 19.3.2.1,
as reported by the secondary sources listed in `research/aci318.json`.

Materials and frontier entries: each row links its paper on the page; full citations are in
`research/`.

Densities: Sudharsan, C. P., Gosh, S. & Sood, H. (2022). https://doi.org/10.22214/ijraset.2022.41047

Chandrakar, K. & Singh, V. V. (2017), for the cement density.
https://scispace.com/papers/an-experimental-study-on-high-performance-concrete-using-24q3e
