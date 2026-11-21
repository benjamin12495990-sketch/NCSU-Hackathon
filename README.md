# Formulation Designer: low-carbon concrete mixes

A one-page web app that proposes new concrete mixes for a strength you choose and ranks them by
embodied CO2. Built from scratch at TRIANGULATE, an AI hackathon at NC State.

- **Grand Challenge:** Engineer the Tools of Scientific Discovery.
- **User:** a civil or materials engineering researcher choosing which trial mixes to cast.
- **Problem:** every strength test means casting a mix and waiting up to 28 days for it to cure, so
  only a handful of the possible mixes ever get tried.

## The page

The page is titled **Formulation Designer**. The three-bar button at the top left opens a menu of
seven simulations (families of materials). **Four are live: concrete, alloys (hardness), insulators
(dielectric constant) and polymers (glass transition).** Each trained family has its own page with a
"Train the network" button (see "The trained families" below). The other three (corrosive solutions,
cleaning products, general chemistry) are marked "Planned": choosing one shows what it would predict,
vary and rank by, and says that nothing is generated for it yet. General chemistry (water solubility)
has a model that was trained and checked offline; its scripts and results files are in
`domains/chemistry/`, but its data file is not in the repo or on the site. Its card says so and shows
its verified numbers. The switch at the
top right turns dark mode on and off; it starts from the system setting and remembers the choice.
The page follows the visual rules in [`taste.md`](taste.md).

## What it does

Enter a target strength in MPa and a curing age. In the browser (or on the server, if you press
"Ping server for a new batch"), the app:

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

"Design mixes" computes everything in your browser from the JSON files in `public/model/`. Nothing
is mocked, and no API key is needed for it.

"Ping server for a new batch" calls the same generate-and-rank pipeline server-side instead, as a
Vercel serverless function (`api/design.js`) that reuses `src/lib/design.js` directly. It takes a
fresh random seed on every call, so repeated pings return different candidates instead of the
page's fixed demo batch (the browser path keeps its fixed seed, so the demo still doesn't change
between clicks). It can also be called directly, e.g.:

```sh
curl -s https://ncsu-hackathon.vercel.app/api/design \
  -H 'Content-Type: application/json' \
  -d '{"targetMPa": 35, "ageDays": 28}'
```

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
less accurate) and `docs/`, which holds plans. In `domains/`, the folders `materials`, `insulators`,
`polymers` and `chemistry` hold the data-preparation script and trainer each family was built with;
the other folders are plans.

## The trained families

Three more families are live, each on one shared page (`src/components/FamilyPage.jsx`) that is
downloaded with its data only when it is opened. Every number on those pages is read from
`public/model/<family>/family.json`, which `scripts/build_family_files.mjs` writes from the family's
`result.json` and an independent re-check (`skeptic_check.json`), both copied into
`domains/<family>/` so the numbers can be checked from the repo; apart from rounding, nothing is typed
into the component. The browser trains one network (seed 11, the family's chosen settings) with
`src/lib/nnFamily.js`; the five-network number in the table was trained in Node with the same code.
`npm run family-parity` trains each family with `nnFamily.js` and with the agent's own trainer in
`domains/<family>/nn_generic.js` and requires identical test predictions (measured difference 0) and
the test error in `result.json`.

Single network trained by the page (seed 11), measured in Node and in headless Chromium: alloys
76.7 HV, insulators 0.121 log10, polymers 16.1 K. Wall time from pressing Train to done in headless
Chromium on a shared 4-CPU machine: alloys about 2.5 s and insulators about 6.3 s (both including
the live interval), polymers about 5.2 s.

For every family: settings were chosen by cross-validation on training rows; the test split was
scored once. Sources were not checked again through MCPs tonight, and the insulator element table
was typed from memory by its agent (it says so in `domains/insulators/prepare.py`).

### Alloys (hardness)

Data: Borg et al., Scientific Data 7, 430 (2020) (https://raw.githubusercontent.com/CitrineInformatics/MPEA_dataset/master/MPEA_dataset.csv), licence: licence not confirmed. 530 rows, 377 distinct alloys; 426 training and 104 held-out rows, split by alloy formula, never by row. Values are MEASURED in labs: hardness values collected from published papers.

| Model, on the 104 held-out rows | Average miss |
|---|---|
| Five networks averaged (Node, same code as the page) | 78.2 HV |
| Gradient-boosted trees, first settings | 73.3 HV |
| Gradient-boosted trees, tuned fairly by cross-validation | 78.2 HV |
| Always guess the training mean | 176.8 HV |

90% interval: ± 179.2 HV, held for 91 in 100 held-out rows. One width everywhere; it fails on alloy systems the model never saw (whole-system error about 102 HV instead of 78), where the true coverage is unknown and probably lower. The page also measures it live after training (3-fold grouped CV on training rows).

Safe sentence (from the independent re-check): "We tested our small from-scratch neural network on 104 hardness measurements of alloys it never saw during training: its typical miss is about 78 HV, against 177 HV for a guess of the average, and that is about the same as standard gradient-boosted trees (73 HV with default settings, 78 HV after tuning), so it is a rough screening tool, not a precise predictor."

### Generating untested alloys

The alloy page has a **Generate untested alloys** section (`src/components/AlloyGenerator.jsx`, the
rules in `src/lib/alloyGen.js`). It uses the network trained on the page; pressing Generate before
training runs the page's own train flow first, then generates with the live 90% interval (or the
results-file interval if the live one has not been measured, and the page says which). The rules:

- Only element sets that labs have tested, each element inside the range of fractions those tested
  alloys used (never widened), in steps of 0.001, capped by the user's maximum per element.
- By default only element sets that were tested by the chosen processing route (CAST unless changed).
  The route menu offers only ANNEAL, CAST, POWDER and WROUGHT (`routes_selectable` in `alloys.json`);
  UNKNOWN and OTHER are data labels, kept as network inputs only, and the generator refuses them.
- Distance is L1 on element fractions to the nearest tested alloy, any route. Under 0.10: not new,
  dropped. 0.10 to 0.20: "interpolation between tested alloys". 0.20 to 0.30: "untested composition
  near tested ones". 0.30 to 0.40: "far from tested alloys (least reliable)". Over 0.40 (the 90th percentile of the spacing between tested alloys): dropped.
- A tested alloy with an element left out and the rest rescaled counts as a copy and is dropped.
- A prediction at the clamp (107 or 1072 HV, the training range) is dropped, not ranked.
- Kept if the lower edge of the 90% range (prediction minus the half-width) reaches the target.
  Candidates the network rates more than one familiar-system error (78 HV, `family.json`) above
  their nearest tested alloy are ranked after all the others and labelled; inside each group the
  half-width is one constant, so the order follows the prediction.
- Each kept candidate shows the nearest tested alloy (formula, route, measured HV) and the gap
  "predicted minus its measured HV", plus its 3 nearest DISTINCT tested compositions (repeat
  measurements of one composition are one neighbour, shown as their mean with their routes) and how
  many of them measured at least the target. With a target set, by default a candidate is kept only
  if at least 2 of those 3 reached it. With the route rule on (the default), only measurements made
  by the chosen route count as support, and the gap is taken against the nearest composition's
  measurements by that route when it has any (the cell says which route and how many).
- When nothing is kept, the page says why (for a high target: the best lower edge any candidate
  reached) and shows no table.

Measured with `npm run alloy-check` (one network, seed 11, file half-width 179.2 HV, seed 1, all
elements, CAST) with the page's default settings (target 400 HV, route rule on, neighbour rule on,
gap limit 78 HV): 38 element sets used, 11400 compositions drawn, 8502 not new, 11 too far, 170
subset copies, 0 at the clamp, 1497 below the target, 147 removed by the neighbour rule (counting
only measurements made by casting), 224 kept, and 45 of those ranked last for a gap above 78 HV. The
top candidate, Al8.6 Co17.6 Cr17.2 Fe14.4 Ni23.0 Ti19.2, is predicted 752 HV; its nearest tested
alloy, Al0.667 Co0.833 Cr0.833 Fe0.833 Ni0.833 Ti1, was cast and measured 857 HV (gap −105 HV), and 2
of its 3 nearest tested compositions reached 400 HV by casting. Switching the route rule off: 49
sets, 621 kept before the neighbour rule, 262 of them from sets never made by CAST. At 600 and 800 HV
nothing is kept: the best lower edge any candidate reaches is 594 HV, while tested alloys in these
sets measured up to 905 HV. The generator takes about 180 ms and yields to the browser between
element sets. None of these compositions has been made or measured.

### Insulators (dielectric constant)

Data: Petousis et al., DOI 10.18126/racd-go9m (Hugging Face foundry-ml/dielectric_constant_v1-1), licence: CC BY 4.0. 1056 rows, 964 distinct compounds (reduced formulas); 843 training and 213 held-out rows, split by reduced formula, never by row. Values are CALCULATED by computer simulation (DFT), not measured in a lab.

| Model, on the 213 held-out rows | Average miss |
|---|---|
| Five networks averaged (Node, same code as the page) | 0.1211 log10 |
| Gradient-boosted trees, first settings | 0.1173 log10 |
| Gradient-boosted trees, tuned fairly by cross-validation | 0.1116 log10 |
| Always guess the training mean | 0.2131 log10 |

90% interval: x/÷ 1.76 (0.245 log10), held for 89 in 100 held-out rows. One width everywhere; it fails on compounds with a dielectric constant above 30 (coverage 39% on 18 held-out rows). The page also measures it live after training (3-fold grouped CV on training rows).

Safe sentence (from the independent re-check): "Our small neural network, written from scratch in JavaScript, estimates a compound's DFT-calculated dielectric constant from its formula, symmetry and cell volume with a typical error of about a factor of 1.3 (0.121 log10 on 213 held-out compounds, against 0.213 for always guessing the average). Standard gradient-boosted trees do as well or slightly better (0.112 to 0.117), and both badly underestimate the rare high-dielectric compounds."

### Polymers (glass transition)

Data: source file experimental_dataset.csv, citation and licence not confirmed. 666 rows, 386 groups (85 copolymer systems, the rest homopolymers); 514 training and 152 held-out rows, split by copolymer system, never by row. Values are MEASURED in labs: glass-transition temperatures collected from published papers.

| Model, on the 152 held-out rows | Average miss |
|---|---|
| Five networks averaged (Node, same code as the page) | 15.33 K |
| Gradient-boosted trees, first settings | 17.54 K |
| Gradient-boosted trees, tuned fairly by cross-validation | 16.37 K |
| Always guess the training mean | 66.44 K |

90% interval: ± 40.4 K, held for 89 in 100 held-out rows. One width everywhere; it fails on homopolymers (coverage 71%). No live measure on the page: The Fox inputs in this file were computed from this one split's training rows, so re-splitting in the browser would leak test values into training. The width shown is the one measured in Node with the Fox inputs recomputed for every fold.

Safe sentence (from the independent re-check): "We held out whole copolymer systems the model never saw (152 of 666 measured rows). On those rows our from-scratch JavaScript neural network predicts glass-transition temperature with an average error of 15.3 K. Guessing the average gives 66 K, and gradient-boosted trees give 16.4 to 17.5 K. The error is about 6 K on real copolymers and about 30 K on pure homopolymers."

## The neural network you can train on the page

The section "Train a neural network yourself" trains a network from random weights in the browser
when you press the button. It is written from scratch in `src/lib/nn.js`: 10 inputs (the tree
model's nine plus the logarithm of age), two hidden layers of 64 and 32 units with ReLU, 2,817
weights, the Adam optimiser and squared-error loss. Nothing pre-trained is downloaded.

- **What it learns from:** the same 786 training lab results as the trees (`public/model/nn_data.json`,
  written by `train/export_nn_data.py`).
- **How it is scored:** on the same 219 held-out lab results from 62 mixes as the trees, which it
  never trains on. The page plots that error after every pass.
- **Result:** it starts 15.9 MPa off and ends 3.9 MPa off (R² 0.86) after 300 passes, in about
  three seconds. The trees are at 3.3 MPa on the same mixes, so the designer keeps using the trees
  and the network is shown as a second opinion.
- **Limits:** the starting weights use a fixed seed, so every press gives the same run. Its
  settings were chosen after seven trial runs scored on the same held-out mixes, so 3.9 MPa is
  slightly flattering. Other seeds we tried ended between 4.0 and 4.6 MPa.

### How sure is it, and did it learn the physics?

After training, the page does two more things with the network it just trained.

- **A 90% interval.** The same design is trained three more times, each time leaving out a third
  of the training mixes and scoring on them. The interval is the size of miss that nine in ten of
  those 786 unseen results stayed inside: **± 11.1 MPa**. On the 219 held-out results it held for
  93 in 100. It is the same width everywhere, which is crude: misses are larger for young concrete.
- **A sweep test.** One mix is changed one thing at a time: more water for the same binder, then
  more curing time. Concrete should get weaker with the first and stronger with the second. The
  page plots the network and the trees and counts the steps that go the right way.
- **What the sweep shows** (`npm run sweep`, 26 tested mixes spread through the dataset): more
  water lowered the network's prediction on 23 of 26 mixes and more age raised it on 26 of 26. The
  trees got both right on all 26. On 6 mixes the network predicted a strength below zero at one
  day old, which is impossible; nothing in the network prevents it.
- **What that means.** The network has picked up both trends from the lab results alone. It does
  not understand them: it has no notion of water, cement or a reaction, and it breaks a physical
  limit that any engineer knows.

### Is the network as accurate as the trees?

About the same. We tried to make it better and report what we found, not what we hoped for.

| Model | Page's held-out split | Mean over eight other splits by mix |
| --- | --- | --- |
| Gradient-boosted trees (used by the designer) | 3.29 MPa | 4.25 MPa |
| One network (what the page trains) | 3.90 MPa | 4.29 MPa |
| Ensemble of five networks, richer inputs, Huber loss | 3.92 MPa | 4.17 MPa |

- The ensemble was chosen by cross-validation on training rows only (4.48 MPa against 4.75 MPa
  for the trees on the same folds), then scored once on the page's held-out mixes.
- Over the eight other splits it beats the trees on four, loses on three and ties one. Its average
  is 0.085 MPa (2%) lower, with a 95% confidence interval of −0.48 to +0.31 MPa for the difference.
  The interval includes zero, so this is a tie.
- The trees' 3.29 MPa comes from one split that happens to suit them: on the eight others they
  range from 3.7 to 5.3 MPa.
- Per-split numbers are in `research/nn_vs_trees.json`; the scripts are in `scripts/experiments/`
  (`make_splits.mjs`, `networks_on_splits.mjs`, `network_cross_validation.mjs`). The ensemble
  takes about 15 seconds to train in a browser, so the page trains the single network.

### Giving it the textbook laws (the switch under the button)

We ran a tournament of seven ideas for a better network, one per literature connector. Each agent
tuned its idea by cross-validation on training rows, scored it at most three times on the eight
splits above, and a second agent re-ran the result and read the code for leaks. One idea survived:
hand the network four textbook strength laws as extra inputs, computed from the same amounts
(water ÷ equivalent cement after Abrams, its inverse after Bolomey, Feret's paste ratio, and
binder ÷ water times the CEB-FIP age curve). We then checked it once on eight fresh splits that no
agent had seen.

| Mean error, MPa | Eight selection splits | Eight fresh splits |
| --- | --- | --- |
| Trees | 4.25 | 4.32 |
| One network, plain (switch off) | 4.29 | 4.38 |
| One network, with the laws (switch on) | 4.17 | 4.23 |
| Five networks, with the laws | 4.03 | 4.18 |

- Against the plain network the laws help on 11 of 16 splits (−0.13 MPa, 95% interval −0.28 to
  +0.01). Against the trees it is −0.08 MPa with an interval of −0.26 to +0.10: still a tie.
- On the page's own split the two designs are level (3.92 against 3.96 MPa over six seeds), but
  with the laws the network predicts a strength below zero about half as often.
- The page trains the plain network unless you switch the laws on, so the default run and the
  numbers above it are unchanged. With the switch on it shows 4.0 MPa and ± 10.3 MPa.
- The slag and fly-ash weights (0.9 and 0.4 kg of cement per kg) are our assumption, in the spirit
  of Babu and Kumar (2000) and Yeh (2013), not values from those papers.
- With the laws as inputs the sweep test is less of a surprise: the network was told what to look
  at, though not which way it points. Everything is in `research/physics_inputs.json`; rerun it
  with `scripts/experiments/make_splits.mjs` (seed base 2000 for the fresh splits),
  `score_on_splits.mjs` and `trees_on_splits.py`.

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

## The lab: evolving networks against a sealed future (`lab/`)

An experiment, separate from the site: small networks learn from the youngest lab results only
(up to 7 days, 253 rows), are rewarded for predicting 14 to 28 days (481 rows) with a penalty for
breaking physics (strength falling with age, or below zero), and have children in proportion to
how good they are; none is ever deleted. The 271 results older than 28 days are sealed in
`lab/vault/` with their SHA-256 and are scored exactly once, after the models have been named in
`lab/preregistration.json`. A guard makes every lab script except the one-time opener unable to
read the vault. `lab/net.mjs` reproduces `src/lib/nn.js` exactly (`npm run lab:parity`). Run
`npm run lab:smoke` for the smoke test; everything is in `lab/README.md`. Nothing on the page uses it.
Phase 2 adds physics and BOxCrete pre-training, the research-backed selection rules, a random-search
control and 13 pre-registered models (`lab/preregistration.json`); the vault is still sealed.

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
| `src/lib/nn.js` | The neural network: forward pass, backpropagation and Adam, written from scratch. `PANEL_DESIGN` is the textbook-laws design behind the switch. |
| `src/lib/sweep.js` | The sweep test: vary water or age alone and count which way the prediction moves. |
| `scripts/sweep.mjs` | `npm run sweep`: trains the page's default network and runs the sweep test on 26 tested mixes. Add `-- --physics` to test the textbook-laws design instead (water 20 of 26, age 26 of 26). |
| `src/components/NeuralNet.jsx` | The training panel: button, learning curve and the two prediction tables. |
| `public/model/nn_data.json` | The training and held-out rows the network uses, the same split as the trees. |
| `train/export_nn_data.py` | Writes `nn_data.json` from the same loading, features and split as `train/train.py`. |
| `src/lib/formula.js` | Estimates the paste's chemical formula per silicon atom from oxide analyses. |
| `src/lib/domains.js` | The seven families of materials and which ones are live. |
| `src/components/Formula.jsx` | Shows the formula with subscripts. |
| `src/components/Planned.jsx` | The "Planned, not built" card for the three families that are not live. For general chemistry it also shows the verified numbers of the model trained offline, read from `public/model/chemistry/family.json`. |
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
| `api/design.js` | Vercel serverless function: same pipeline as `design.js` plus the exposure limits and materials shortlist, over HTTP, with a fresh random seed per call. |
| `src/App.jsx` | The page. |
| `pipeline/train_nn.py` | An alternative neural-network model in NumPy. Not used by the live page. |
| `docs/SCOPE.md`, `domains/` | Plans for applying the same method to other domains. Four folders in `domains/` also hold what a trained family was built with (next rows). |
| `domains/<family>/prepare.py` | The script each family agent used to build its training file (materials, insulators, polymers, chemistry). Copied byte-identical from the agents' work folders so the method survives; it is not run by the site. |
| `domains/<family>/nn_generic.js` | Each family agent's own network trainer, copied byte-identical. `npm run family-parity` trains with it to prove that `src/lib/nnFamily.js` gives the same predictions. |
| `domains/<family>/result.json` | Each family agent's full results file (test errors, eight-split comparison, interval, sweep and caveats), copied unchanged. Every number in `public/model/<family>/family.json` comes from it or from `skeptic_check.json`, and the `source` field lists the paths used. |
| `domains/<family>/skeptic_check.json` | An independent re-run of each family: reproduced numbers, trees tuned fairly by cross-validation, and the safe sentence the page uses as its verdict. It is copied unchanged so the "tuned trees" row and the verdict can be checked from the repo. |
| `src/lib/nnFamily.js` | The trainer behind the three family pages: `seeded()` and `Mlp` copied unchanged from `nn.js`, plus every preprocessing option the families use. `nn.js` itself is not edited, so the concrete page cannot change. |
| `src/components/FamilyPage.jsx` | One page for every trained family: data facts, the Train button and learning curve, error table, interval, sweep or trend check and the caveat box. It is lazy-loaded and fetches its family's files only when opened. |
| `src/components/family.css` | The few layout rules the family pages add. Colours come only from the existing variables, so dark mode works. |
| `scripts/build_family_files.mjs` | `npm run family-files`: copies each family's `nn_data.json` into `public/model/<family>/` and writes `family.json` from `result.json` and `skeptic_check.json`, and copies those two files into `domains/<family>/`. It reads from the family agents' work folder (outside the repo; pass another folder as an argument), and each `family.json` has a `source` field naming the files and paths its numbers came from. |
| `scripts/family_parity.mjs` | `npm run family-parity`: trains each family twice (shared trainer and agent trainer) on the shipped file. It fails unless predictions match to 1e-9 and the test error equals `result.json`. |
| `public/model/<family>/nn_data.json` | The training and held-out rows for alloys, insulators and polymers, copied unchanged from the family agents. The browser trains on it; the polymer file's Fox columns belong to this one split, so the page never re-splits it. |
| `public/model/<family>/family.json` | Every number and caveat a family page shows, written by `build_family_files.mjs`. For chemistry it holds only the verified test numbers for the Planned card; no chemistry data is shipped. |
| `public/model/insulators/heldout.json` | The 213 held-out compounds (formula, Materials Project id, calculated band gap and calculated value) in the same order as the test rows. It feeds the ranking of existing compounds and the band-gap trend check. |
| `src/lib/alloyGen.js` | The alloy generator: draws untested compositions inside tested element sets, applies the novelty, distance, copy, clamp, target and neighbour-support rules, and counts every exclusion. Each candidate carries its nearest 3 tested alloys and the gap to the nearest one; `generateAsync()` gives the same answer while yielding to the browser. |
| `src/lib/alloyFeatures.js` | Turns an alloy composition and processing route into the 37 network inputs in `nn_data.json` order (element fractions, route and descriptors). `npm run alloy-check` shows it rebuilds all 530 training and test rows with a largest difference of 0. |
| `public/model/materials/alloys.json` | The 530 tested alloys (raw formula, element fractions, route, measured HV, split) in `nn_data.json` order, plus the element property table, the clamp range and the file's 90% half-width. The generator and the featurizer read it; it is about 20 kB gzipped. |
| `domains/materials/export_rows.py` | Writes `alloys.json` from the MPEA dataset, the family's `nn_data.json` and `result.json`, with the parse and descriptor code of `prepare.py`. Run it with `python3 -I`; it is not run by the site. |
| `scripts/alloy_check.mjs` | `npm run alloy-check`: checks feature parity on all 530 rows, then trains one network and asserts every generator rule on every kept candidate, including the 3 nearest tested alloys, the gap and the neighbour-support rule. It reads `nn_generic.js` and `result.json` from the materials work folder (`MATERIALS_SIM_DIR`). |
| `src/components/AlloyGenerator.jsx` | The "Generate untested alloys" section of the alloy page: switch-and-slider rows, the route and element choices, the counts sentence and the 10-row table. It reads the 78 and 102 HV errors from `family.json` and never shows an empty table. |
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
