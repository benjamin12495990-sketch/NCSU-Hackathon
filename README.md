# Low-Carbon Concrete Mix Designer

A one-page web app that proposes new concrete mixes for a strength you choose and ranks them by
embodied CO2. Built from scratch at TRIANGULATE, an AI hackathon at NC State.

- **Grand Challenge:** Engineer the Tools of Scientific Discovery.
- **User:** a civil or materials engineering researcher choosing which trial mixes to cast.
- **Problem:** every strength test means casting a mix and waiting up to 28 days for it to cure, so
  only a handful of the possible mixes ever get tried.

## What it does

Enter a target strength in MPa and a curing age. In the browser, with no server, the app:

1. generates about 14,000 new mixes by varying each of the 355 lab-tested mixes in the dataset
   (binder amount, share of slag and fly ash, water/binder ratio, superplasticizer dose), rescaling
   the aggregates so each mix keeps its parent's volume;
2. removes any mix that is already in the dataset;
3. predicts each mix's strength with two models trained for this project: the expected strength,
   and a cautious lower estimate;
4. keeps the mixes whose cautious estimate reaches the target (or the expected strength, if you turn
   caution off);
5. ranks them by embodied CO2 per cubic metre;
6. checks each against its three nearest lab-tested mixes and marks it "lab-supported" only when at
   least two of them are close to it and reached the target in the lab.

It also shows the lowest-CO2 mix that a lab has already tested for that target, so you can see
whether a new candidate is worth a trial at all.

## What is live

Everything on the page is computed in your browser from the JSON files in `public/model/`. Nothing
is mocked. There is no backend, no API and no API key.

Two kinds of numbers are not measured by us and are labelled on the page:

- **CO2 factors and densities** come from published papers (table on the page, with links). Water is
  not counted because we found no sourced factor.
- **The three preset strengths** are illustrative, not values from a building code.

## How the models were trained

`train/train.py` reads the UCI Concrete Compressive Strength dataset: 1,030 lab results, of which
25 are exact repeats and are dropped.

- **Features (9):** kg per m3 of cement, blast furnace slag, fly ash, water, superplasticizer,
  coarse aggregate and fine aggregate; age in days; and water divided by total binder.
- **Expected-strength model:** scikit-learn `GradientBoostingRegressor`, 300 trees, depth 4,
  learning rate 0.05.
- **Cautious model:** the same settings with quantile loss at the 10th percentile, so about 9 lab
  results in 10 should beat it.
- **Split:** 80/20 **by mix**. The dataset holds the same mix at several ages, and the same mix
  twice with different rounding, so mixes are identified by amounts rounded to 1 kg and every
  result of a test mix is held out.

| Test metric (71 held-out mixes, 179 lab results) | Value |
| --- | --- |
| Mean absolute error | 4.46 MPa |
| Root mean squared error | 6.60 MPa |
| R² | 0.839 |
| Share of test results at or above the cautious estimate | 83.8% (the aim was 90%) |
| Mean absolute error with a plain random split, for comparison only | 2.94 MPa |

The random-split number is the one most papers report on this dataset. It looks better because the
same mixes sit on both sides of the split. The live numbers are in `public/model/metrics.json`, and
the page reads them from there.

## Known failure modes

- **The search rewards model error.** Ranking by lowest CO2 pulls toward mixes where the model is
  too optimistic. Our first version proposed a mix predicted at 43 MPa whose nearest tested mix,
  with more cement at the same water/binder ratio, measured 29 MPa. The cautious model and the
  lab-support check were added because of that. They reduce the problem; they do not remove it.
- **With caution on, candidates rarely beat the best tested mix.** That is an honest result, not a
  bug: the best tested mix has a real measurement, and a candidate must clear a stricter bar.
- **With caution off, candidates are experiments, not answers.** Their cautious estimates fall
  short of the target.
- **The cautious estimate is less cautious than intended** (83.8% coverage, not 90%).
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

## How we used MCP servers

All sourced numbers on the page were found and cross-checked tonight through MCP servers connected
to the coding agent. Six agents ran in parallel: two searched, two independently re-checked every
number with a different search, one gathered expert objections, and one looked for more servers.

| MCP server | What it was used for |
| --- | --- |
| SciSpace | Found the CO2 factors and densities, and the published accuracy figures. |
| Consensus | Confirmed the superplasticizer factor from a second source; expert objections. |
| FastTrack Literature Open | Confirmed the dataset paper and resolved the DOI of the CO2 source. |
| alphaXiv | Searched preprints; nothing from it is on the page. |
| Elicit | Tried; the account's plan has no API access, so nothing came back. |
| Claude Market and the connector registry | Searched for concrete, cement, life cycle and deployment servers. None exist for concrete. |
| Tenjin router | Installed. No paid call was made and no number on the page comes from it. |

What the checks changed: water has no CO2 factor because none could be sourced; slag and fly ash
are labelled as secondary; three densities are labelled as unconfirmed.

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
| `public/model/metrics.json` | Test metrics, row counts, training date and the lab-support distance. |
| `public/model/parity.json` | 20 test rows with Python predictions from both models. |
| `public/model/test_points.json` | Measured and predicted strength for the test set (the chart). |
| `public/model/factors.json` | CO2 factors and densities with their sources. Written by hand from the literature search. |
| `src/lib/featurize.js` | Builds the same 9 features as Python. |
| `src/lib/predict.js` | Walks the trees and adds up the leaves. |
| `src/lib/co2.js` | Embodied CO2 and solid volume of a mix. |
| `src/lib/nearest.js` | Distances to tested mixes and their lab results. |
| `src/lib/design.js` | Generates, filters, ranks and checks the candidates. |
| `src/lib/loadModel.js` | Downloads the JSON files when the page opens. |
| `src/App.jsx` | The page. |
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

Densities: Sudharsan, C. P., Gosh, S. & Sood, H. (2022). https://doi.org/10.22214/ijraset.2022.41047
