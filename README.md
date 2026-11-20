# Superconductor Candidate Generator

A one-page web app that predicts the superconducting critical temperature (Tc) of a chemical
formula and proposes new, untested compositions ranked by predicted Tc. Built from scratch at
TRIANGULATE, an AI hackathon at NC State.

- **Grand Challenge:** Engineer the Tools of Scientific Discovery.
- **User:** a materials science grad student deciding which superconductor compositions deserve
  weeks of synthesis or simulation time.
- **Problem:** screening candidates by hand or by simulation is slow, so most compositions never
  get looked at.

## What it does

Type a formula such as `YBa2Cu3O7`. In the browser, with no server, the app:

1. predicts its Tc in kelvin with a model trained for this project;
2. generates candidates by swapping one element at a time for a chemically similar one (same
   periodic group, and lanthanide for lanthanide), keeping every number unchanged;
3. removes every candidate whose composition is already in the dataset;
4. ranks the rest by predicted Tc and shows each one's nearest known material and a confidence flag.

## What is live

Everything on the page is real and computed in your browser. Nothing is mocked. There is no
backend, no API and no API key. The model is a set of JSON files downloaded once with the page.

## How the model was trained

`train/train.py` reads `unique_m.csv` from the UCI Superconductivty Data set (21,263 materials).

- **Features (87):** the fraction of each of 86 elements (H to Rn) in the formula, plus the number
  of distinct elements.
- **Model:** scikit-learn `RandomForestRegressor`, 100 trees, maximum depth 8.
- **Split:** random 80% train, 20% test, fixed seed.

| Test metric | Value |
| --- | --- |
| Mean absolute error | 8.49 K |
| R² | 0.857 |
| Training rows | 17,010 |
| Test rows | 4,253 |
| Distinct known compositions | 15,139 |

The live numbers are in `public/model/metrics.json`, and the page reads them from there.

## Known failure modes

- **The test error is optimistic.** The split is random, so near-duplicates of test materials
  (the same compound at a slightly different doping) were in training. Error on new chemistry is larger.
- **Candidates are predictions, not discoveries.** Nothing checks that a candidate can be
  synthesized, is stable, or superconducts at all.
- **The dataset holds only superconductors**, so the model cannot say "this is not a superconductor".
- **Composition only.** The model knows nothing about crystal structure, pressure or processing.
- **Ties.** A depth-8 forest gives many similar swaps the same predicted Tc, so the ranking
  between them is arbitrary.
- **"New" means "not in this one dataset".** A candidate may exist in the literature or in other databases.
- **Low-confidence flag.** A candidate is flagged when its distance to the nearest known material
  is above 0.0707, the 90th percentile of test-to-train nearest distances.
- **Swap targets** exclude noble gases and elements with no stable isotope (Tc, Pm, Po, At, Rn).

## Files

| File | What it does |
| --- | --- |
| `train/train.py` | Trains the model, measures test error, and writes the JSON files below. |
| `public/model/elements.json` | The element order. Python and JavaScript both read it, so the features cannot drift apart. |
| `public/model/model.json` | The 100 trees: feature index, threshold, left child, right child and leaf value per node. |
| `public/model/known.json` | Every distinct composition in the dataset with its formula and mean measured Tc. |
| `public/model/metrics.json` | Test MAE, R², row counts, training date and the low-confidence distance. |
| `public/model/parity.json` | 20 test formulas with their features and Python predictions. |
| `src/lib/parseFormula.js` | Parses a formula string (decimals and brackets allowed) and returns readable errors. |
| `src/lib/featurize.js` | Turns a parsed formula into the same 87 features Python used. |
| `src/lib/predict.js` | Walks each tree and averages the leaves. |
| `src/lib/generate.js` | Makes one-swap candidates, drops known ones, ranks by predicted Tc. |
| `src/lib/nearest.js` | Finds the nearest known material and the "already known" match. |
| `src/lib/analyze.js` | Runs the whole pipeline for one typed formula. |
| `src/lib/loadModel.js` | Downloads the JSON files when the page opens. |
| `src/App.jsx` | The page. |
| `scripts/parity.mjs` | Checks that JavaScript predictions match Python. |

## Run it

```sh
npm install
npm run dev      # local page
npm run parity   # JavaScript vs Python on 20 formulas, must match within 0.01 K
npm run build    # static site in dist/
```

Retrain (optional, overwrites `public/model/`):

```sh
pip install -r train/requirements.txt
python train/train.py
npm run parity
```

## Dataset

Hamidieh, K. (2018). Superconductivty Data [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C53P47

Hamidieh, K. (2018). A data-driven statistical model for predicting the critical temperature of a
superconductor. *Computational Materials Science*, 154, 346–354.
