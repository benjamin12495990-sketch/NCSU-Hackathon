# Pipeline report: 20261127T063145Z_nightly

Generated 2026-11-27 07:49:49. Folder: `gpu\runs\pipeline\20261127T063145Z_nightly`

## Nightly: the current model for every task

Retrained tonight: tab:alloys, tab:polymers, tab:electrolytes, tab:insulators, tab:concrete, tab:chemistry, tab:cleaning_pcmc, tab:cleaning_tension, tab:corrosion, crystal:bulk_modulus_kv, crystal:shear_modulus_gv, crystal:formation_energy_peratom, crystal:optb88vdw_bandgap. Unchanged: none.

| Task | Recommended model | Test MAE | Strongest trees MAE | 90% interval half-width / coverage | Trained |
| --- | --- | --- | --- | --- | --- |
| crystal:bulk_modulus_kv | blend | 0.08312 | 0.09154 | 0.1395 / 0.8759 | 2026-11-27 07:49:16 (20261127T063145Z_nightly) |
| crystal:formation_energy_peratom | network | 0.04027 | 0.1295 | 0.07798 / 0.8844 | 2026-11-27 07:49:16 (20261127T063145Z_nightly) |
| crystal:optb88vdw_bandgap | network | 0.1916 | 0.307 | 0.4508 / 0.8715 | 2026-11-27 07:49:16 (20261127T063145Z_nightly) |
| crystal:shear_modulus_gv | trees | 0.154 | 0.154 | 0.2982 / 0.8893 | 2026-11-27 07:49:16 (20261127T063145Z_nightly) |
| tab:alloys | trees | 70.9 | 70.9 | 161.6 / 0.885 | 2026-11-27 07:49:16 (20261127T063145Z_nightly) |
| tab:chemistry | blend | 0.7298 | 0.7736 | 1.667 / 0.9005 | 2026-11-27 07:49:16 (20261127T063145Z_nightly) |
| tab:cleaning_pcmc | trees | 0.5266 | 0.5266 | 1.242 / 0.8989 | 2026-11-27 07:49:16 (20261127T063145Z_nightly) |
| tab:cleaning_tension | trees | 4.844 | 4.844 | 10.23 / 0.9002 | 2026-11-27 07:49:16 (20261127T063145Z_nightly) |
| tab:concrete | trees | 12.06 | 12.06 | 20.42 / 0.8625 | 2026-11-27 07:49:16 (20261127T063145Z_nightly) |
| tab:corrosion | trees | 0.7723 | 0.7723 | 1.611 / 0.8981 | 2026-11-27 07:49:16 (20261127T063145Z_nightly) |
| tab:electrolytes | trees | 1.552 | 1.552 | 3.555 / 0.8975 | 2026-11-27 07:49:16 (20261127T063145Z_nightly) |
| tab:insulators | blend | 0.1164 | 0.1223 | 0.2462 / 0.9051 | 2026-11-27 07:49:16 (20261127T063145Z_nightly) |
| tab:polymers | blend | 14.05 | 16.07 | 37.6 / 0.9008 | 2026-11-27 07:49:16 (20261127T063145Z_nightly) |

Intervals are for the recommended model (for crystal runs made before trees saved validation predictions, for the network average). Recipes: `gpu/recipes.json`.

## At a glance

- Jobs: **32 finished**, 0 failed or timed out, 0 not run (stopped or out of time) of 32. GPU job time: 6.28 h.
- Table-based families (9 with results). Average of the network seeds against one tuned tree model: network clearly better on 3, trees clearly better on 2, no clear difference on 4. **One network at a time**: clearly better in every seed on 0, trees clearly better in every seed on 1, mixed or unclear on 8.
- Crystal-structure properties (0 with results), against the stronger trees. Average of runs: network clearly better on 0, trees clearly better on 0, no clear difference on 0. **Single runs**: clearly better in every run on 0, mixed or unclear on 0, trees clearly better in every run on 0.
- **90% prediction intervals**: coverage between 0.87 and 0.93 for 37 of 39 task/model rows (lowest 0.86); see "Prediction intervals".
- **Held-out chemical systems** (4 properties; networks by rule: fixed rule), against the strongest trees. Average of runs: network clearly better on 2, trees clearly better on 1, no clear difference on 1. Single runs: clearly better in every run on 2, mixed or unclear on 0, trees clearly better in every run on 2.
- GPU: peak temperature 64 C, peak power 73.5 W, peak memory 1556 MiB, average utilisation 35.2% (2234 samples).
- Sealed data: the sealed concrete folder (lab/vault/) is unchanged: same files, sizes and modification times; this pipeline never reads it.

**Averages versus single networks.** The headline "average" verdict compares several networks averaged together with ONE tree model, which favours the network just by averaging. The "one network at a time" verdict removes that. If the two disagree, the win is an ensemble effect; the fairness checks under each family also show a no-fit blend of networks and trees.

**How to read the verdicts.** "Network clearly better" means the whole 95% bootstrap interval of (network error minus trees error) is below zero; "trees clearly better" means it is entirely above zero; anything else is "no clear difference". The bootstrap resamples whole groups (mixes, alloy systems, papers, formulas), never single rows.

## Table-based families: network sweep against tuned trees

Each network number is the best of a large sweep, **chosen without looking at the rows it is scored on** (cross-fitted); the trees and the linear models get exactly the same treatment. MAE = typical size of the miss, in the unit shown (lower is better).

| Family | Predicts | Rows / groups | Guess the average | Best linear | Best trees | Network: average of seeds | Network: one seed | Average minus trees (95% interval) | Verdict: average | Verdict: one network at a time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Alloys (hardness) | Vickers hardness, HV | 426 / 301 | 173.8 | 115.8 | 70.9 | 75.94 | 77.94 | +5.05 [-1.39, +12] | **no clear difference** | clearly better in 0 of 5 runs, clearly worse in 1 |
| Polymers (glass transition) | glass-transition temperature Tg, K | 514 / 308 | 65.66 | 21.28 | 16.07 | 14.56 | 15 | -1.51 [-2.89, -0.0985] | **network clearly better** | clearly better in 0 of 5 runs |
| Solid electrolytes (conductivity) | log10 room-temperature ionic conductivity, log10(S/cm); OBELiX training partition only | 478 / 124 | 2.554 | 2.796 | 1.552 | 2.139 | 1.949 | +0.587 [+0.199, +1.08] | **trees clearly better** | trees clearly better in all 5 runs |
| Insulators (dielectric constant, calculated) | log10(poly_total), DFT-calculated dielectric constant | 843 / 771 | 0.207 | 0.1369 | 0.1223 | 0.1178 | 0.1178 | -0.00458 [-0.00994, +0.000734] | **no clear difference** | clearly better in 0 of 5 runs |
| Concrete (strength, ages 14-28 d from 1-7 d) | compressive strength, MPa (TRAIN = ages 1-7 d) | 253 / 161 | 15.8 | 11.14 | 12.06 | 6.251 | 12.91 | -5.8 [-7.46, -4.16] | **network clearly better** | clearly better in 2 of 5 runs, clearly worse in 2 |
| Chemistry (water solubility) | aqueous solubility LogS, log10 mol/L | 7982 / 7736 | 1.884 | 1.208 | 0.7736 | 0.7452 | 0.7772 | -0.0284 [-0.0398, -0.017] | **network clearly better** | clearly better in 0 of 5 runs |
| Surfactants (critical micelle concentration) | pCMC = -log10(critical micelle concentration, mol/L); SURFPRO | 1395 / 115 | 0.8991 | 0.8302 | 0.5266 | 0.5846 | 0.5879 | +0.058 [-0.00624, +0.111] | **no clear difference** | clearly better in 0 of 5 runs, clearly worse in 3 |
| Surfactants (surface tension at the CMC) | air-water surface tension at the CMC, mN/m; SURFPRO | 972 / 190 | 5.547 | 5.344 | 4.844 | 5.22 | 5.242 | +0.376 [+0.153, +0.595] | **trees clearly better** | clearly better in 0 of 5 runs, clearly worse in 2 |
| Corrosion (rate, NIST CORR-DATA, exploratory) | log10 corrosion rate, log10(mm/yr); NIST CORR-DATA point values | 5761 / 207 | 1.026 | 0.8344 | 0.7723 | 0.768 | 0.7734 | -0.00428 [-0.0221, +0.0154] | **no clear difference** | clearly better in 0 of 5 runs |

### Alloys (hardness)

- 288 network settings x 5 seeds, 9 tree and 5 linear baseline models, 3-fold grouped cross-validation (426 rows).
- Cross-fitted: network MAE 75.94 (R2 0.745), trees 70.9 (R2 0.782), linear 115.8, guess-the-average 173.8.
- Without the correction (best setting picked on the same folds it is scored on, **optimistic**): network 70.95, trees 70.45, linear 115.8.
- One seed instead of the average of 5: network MAE 78.76.
- Network versus linear models: -39.8 [-51, -29.8], network clearly better.
- Settings chosen on each fold: hidden [128, 64], 150 epochs, lr 0.002, decay 0.001, squared loss; hidden [64, 32], 150 epochs, lr 0.001, decay 0.0001, squared loss; hidden [64, 32], 150 epochs, lr 0.002, decay 0.0001, huber loss
  - **Fairness checks** (the headline compares the network average with ONE tree model):
    - One network at a time: MAE 77.94 on average; against the trees it is clearly better in 0 of 5 runs, clearly worse in 1.
    - Average of the 3 best tree models: MAE 71.72; the network average against it: +4.22 [-2.42, +11.4], no clear difference.
    - Half network average, half trees (no fitted weight): MAE 69.81; against the trees -1.08 [-4.07, +2.18] (no clear difference); against the network average -6.13 [-10.2, -2.46] (network clearly better).
    - Median error (the typical row, not pulled by a few bad ones): network average 51.97, one network 49.87, trees 49.02, top-3 trees averaged 52.6.

### Polymers (glass transition)

- 288 network settings x 5 seeds, 9 tree and 5 linear baseline models, 3-fold grouped cross-validation (514 rows).
- Cross-fitted: network MAE 14.56 (R2 0.923), trees 16.07 (R2 0.913), linear 21.28, guess-the-average 65.66.
- Without the correction (best setting picked on the same folds it is scored on, **optimistic**): network 14.3, trees 15.43, linear 21.04.
- One seed instead of the average of 5: network MAE 15.31.
- Network versus linear models: -6.72 [-8.44, -5.03], network clearly better.
- Settings chosen on each fold: hidden [32, 16], 400 epochs, lr 0.004, decay 0.001, huber loss; hidden [64, 32], 200 epochs, lr 0.004, decay 0.001, huber loss; hidden [64, 32], 400 epochs, lr 0.004, decay 0.001, squared loss
  - **Fairness checks** (the headline compares the network average with ONE tree model):
    - One network at a time: MAE 15 on average; against the trees it is clearly better in 0 of 5 runs.
    - Average of the 3 best tree models: MAE 15.03; the network average against it: -0.473 [-1.8, +0.886], no clear difference.
    - Half network average, half trees (no fitted weight): MAE 14.05; against the trees -2.02 [-2.87, -1.25] (network clearly better); against the network average -0.512 [-1.57, +0.384] (no clear difference).
    - Median error (the typical row, not pulled by a few bad ones): network average 7.104, one network 8.084, trees 8.76, top-3 trees averaged 6.991.
    - Reading: if the average wins but single networks do not, the gain comes from averaging several networks, not from one network being better than trees.

### Solid electrolytes (conductivity)

- 288 network settings x 5 seeds, 9 tree and 5 linear baseline models, 5-fold grouped cross-validation (478 rows).
- Cross-fitted: network MAE 2.139 (R2 0.198), trees 1.552 (R2 0.557), linear 2.796, guess-the-average 2.554.
- Without the correction (best setting picked on the same folds it is scored on, **optimistic**): network 1.813, trees 1.514, linear 2.724.
- One seed instead of the average of 5: network MAE 2.071.
- Network versus linear models: -0.657 [-1.3, +0.0664], no clear difference.
- Settings chosen on each fold: hidden [128, 64], 150 epochs, lr 0.004, decay 0.0001, squared loss; hidden [128, 64], 600 epochs, lr 0.004, decay 0.0001, squared loss; hidden [128, 64], 600 epochs, lr 0.004, decay 0.001, huber loss; ...
  - **Fairness checks** (the headline compares the network average with ONE tree model):
    - One network at a time: MAE 1.949 on average; against the trees it is trees clearly better in all 5 runs.
    - Average of the 3 best tree models: MAE 1.544; the network average against it: +0.595 [+0.178, +1.13], trees clearly better.
    - Half network average, half trees (no fitted weight): MAE 1.672; against the trees +0.121 [-0.144, +0.405] (no clear difference); against the network average -0.467 [-0.715, -0.269] (network clearly better).
    - Median error (the typical row, not pulled by a few bad ones): network average 1.386, one network 1.396, trees 1.008, top-3 trees averaged 1.029.

### Insulators (dielectric constant, calculated)

- 288 network settings x 5 seeds, 9 tree and 5 linear baseline models, 3-fold grouped cross-validation (843 rows).
- Cross-fitted: network MAE 0.1178 (R2 0.589), trees 0.1223 (R2 0.551), linear 0.1369, guess-the-average 0.207.
- Without the correction (best setting picked on the same folds it is scored on, **optimistic**): network 0.1161, trees 0.1212, linear 0.1344.
- One seed instead of the average of 5: network MAE 0.1172.
- Network versus linear models: -0.0191 [-0.0256, -0.0125], network clearly better.
- Settings chosen on each fold: hidden [64, 32], 300 epochs, lr 0.002, decay 0.03, huber loss; hidden [64, 32], 75 epochs, lr 0.004, decay 0.01, huber loss
  - **Fairness checks** (the headline compares the network average with ONE tree model):
    - One network at a time: MAE 0.1178 on average; against the trees it is clearly better in 0 of 5 runs.
    - Average of the 3 best tree models: MAE 0.1215; the network average against it: -0.00376 [-0.00901, +0.00138], no clear difference.
    - Half network average, half trees (no fitted weight): MAE 0.1164; against the trees -0.00594 [-0.00869, -0.00313] (network clearly better); against the network average -0.00136 [-0.0042, +0.0014] (no clear difference).
    - Median error (the typical row, not pulled by a few bad ones): network average 0.0802, one network 0.08038, trees 0.08212, top-3 trees averaged 0.07986.

### Concrete (strength, ages 14-28 d from 1-7 d)

- **Exploratory.** These concrete rows already informed project decisions. The network setting and each baseline were chosen on REWARD; the comparison is on MONITOR. The network setting was chosen on REWARD (ages 14-28 d); scores below are on MONITOR (ages 14-28 d, never used to choose anything).
  - mean: MAE 15.8, R2 -1.16
  - linear: MAE 11.14, R2 -0.00996
  - trees: MAE 12.06, R2 -0.0729
  - network: MAE 6.251, R2 0.581
  - ridge_ln_age_only: MAE 10.49, R2 0.0402
- Network versus trees: -5.8 [-7.46, -4.16], network clearly better.
- Network versus linear: -4.89 [-6.54, -3.29], network clearly better.
- Network versus ridge_ln_age_only: -4.23 [-6.08, -2.41], network clearly better.
  - **Fairness checks** (the headline compares the network average with ONE tree model):
    - One network at a time: MAE 12.91 on average; against the trees it is clearly better in 2 of 5 runs, clearly worse in 2.
    - Average of the 3 best tree models: MAE 12.49; the network average against it: -6.24 [-7.92, -4.53], network clearly better.
    - Half network average, half trees (no fitted weight): MAE 8.138; against the trees -3.92 [-4.68, -3.07] (network clearly better); against the network average +1.89 [+0.946, +2.91] (trees clearly better).
    - Median error (the typical row, not pulled by a few bad ones): network average 4.55, one network 6.85, trees 10.88, top-3 trees averaged 11.16.
    - Reading: if the average wins but single networks do not, the gain comes from averaging several networks, not from one network being better than trees.

### Chemistry (water solubility)

- 288 network settings x 5 seeds, 9 tree and 5 linear baseline models, 3-fold grouped cross-validation (7982 rows).
- Cross-fitted: network MAE 0.7452 (R2 0.787), trees 0.7736 (R2 0.771), linear 1.208, guess-the-average 1.884.
- Without the correction (best setting picked on the same folds it is scored on, **optimistic**): network 0.7452, trees 0.7736, linear 1.207.
- One seed instead of the average of 5: network MAE 0.7779.
- Network versus linear models: -0.462 [-0.484, -0.44], network clearly better.
- Settings chosen on each fold: hidden [256, 128], 60 epochs, lr 0.004, decay 0.0001, huber loss
  - **Fairness checks** (the headline compares the network average with ONE tree model):
    - One network at a time: MAE 0.7772 on average; against the trees it is clearly better in 0 of 5 runs.
    - Average of the 3 best tree models: MAE 0.7712; the network average against it: -0.026 [-0.0372, -0.0151], network clearly better.
    - Half network average, half trees (no fitted weight): MAE 0.7298; against the trees -0.0438 [-0.0499, -0.0377] (network clearly better); against the network average -0.0154 [-0.0216, -0.0093] (network clearly better).
    - Median error (the typical row, not pulled by a few bad ones): network average 0.5016, one network 0.528, trees 0.5228, top-3 trees averaged 0.518.
    - Reading: if the average wins but single networks do not, the gain comes from averaging several networks, not from one network being better than trees.

### Surfactants (critical micelle concentration)

- 288 network settings x 5 seeds, 9 tree and 5 linear baseline models, 5-fold grouped cross-validation (1395 rows).
- Cross-fitted: network MAE 0.5846 (R2 0.497), trees 0.5266 (R2 0.565), linear 0.8302, guess-the-average 0.8991.
- Without the correction (best setting picked on the same folds it is scored on, **optimistic**): network 0.5454, trees 0.5205, linear 0.8302.
- One seed instead of the average of 5: network MAE 0.5701.
- Network versus linear models: -0.246 [-0.34, -0.158], network clearly better.
- Settings chosen on each fold: hidden [256, 128], 100 epochs, lr 0.001, decay 0.0001, squared loss; hidden [256, 128], 400 epochs, lr 0.004, decay 0.001, squared loss; hidden [32, 16], 100 epochs, lr 0.002, decay 0.001, squared loss; ...
  - **Fairness checks** (the headline compares the network average with ONE tree model):
    - One network at a time: MAE 0.5879 on average; against the trees it is clearly better in 0 of 5 runs, clearly worse in 3.
    - Average of the 3 best tree models: MAE 0.5218; the network average against it: +0.0628 [+0.00367, +0.112], trees clearly better.
    - Half network average, half trees (no fitted weight): MAE 0.5306; against the trees +0.00402 [-0.0232, +0.0267] (no clear difference); against the network average -0.054 [-0.0853, -0.0165] (network clearly better).
    - Median error (the typical row, not pulled by a few bad ones): network average 0.4257, one network 0.4142, trees 0.3572, top-3 trees averaged 0.3517.

### Surfactants (surface tension at the CMC)

- 288 network settings x 5 seeds, 9 tree and 5 linear baseline models, 5-fold grouped cross-validation (972 rows).
- Cross-fitted: network MAE 5.22 (R2 -0.0191), trees 4.844 (R2 0.106), linear 5.344, guess-the-average 5.547.
- Without the correction (best setting picked on the same folds it is scored on, **optimistic**): network 4.85, trees 4.729, linear 5.344.
- One seed instead of the average of 5: network MAE 5.4.
- Network versus linear models: -0.124 [-0.833, +0.519], no clear difference.
- Settings chosen on each fold: hidden [128, 64], 100 epochs, lr 0.001, decay 0.001, squared loss; hidden [128, 64], 200 epochs, lr 0.002, decay 0.0001, squared loss; hidden [64, 32], 100 epochs, lr 0.004, decay 0.001, huber loss
  - **Fairness checks** (the headline compares the network average with ONE tree model):
    - One network at a time: MAE 5.242 on average; against the trees it is clearly better in 0 of 5 runs, clearly worse in 2.
    - Average of the 3 best tree models: MAE 4.772; the network average against it: +0.447 [+0.232, +0.67], trees clearly better.
    - Half network average, half trees (no fitted weight): MAE 4.936; against the trees +0.0916 [-0.0215, +0.201] (no clear difference); against the network average -0.284 [-0.402, -0.168] (network clearly better).
    - Median error (the typical row, not pulled by a few bad ones): network average 4.025, one network 4.057, trees 3.642, top-3 trees averaged 3.547.

### Corrosion (rate, NIST CORR-DATA, exploratory)

- 288 network settings x 5 seeds, 9 tree and 5 linear baseline models, 5-fold grouped cross-validation (5761 rows).
- Cross-fitted: network MAE 0.768 (R2 0.388), trees 0.7723 (R2 0.388), linear 0.8344, guess-the-average 1.026.
- Without the correction (best setting picked on the same folds it is scored on, **optimistic**): network 0.7642, trees 0.767, linear 0.8344.
- One seed instead of the average of 5: network MAE 0.7673.
- Network versus linear models: -0.0664 [-0.0885, -0.045], network clearly better.
- Settings chosen on each fold: hidden [256, 128], 30 epochs, lr 0.001, decay 0.03, huber loss; hidden [32, 16], 30 epochs, lr 0.004, decay 0.03, huber loss; hidden [64, 32], 30 epochs, lr 0.001, decay 0.03, huber loss
  - **Fairness checks** (the headline compares the network average with ONE tree model):
    - One network at a time: MAE 0.7734 on average; against the trees it is clearly better in 0 of 5 runs.
    - Average of the 3 best tree models: MAE 0.7662; the network average against it: +0.0018 [-0.0162, +0.0226], no clear difference.
    - Half network average, half trees (no fitted weight): MAE 0.7521; against the trees -0.0202 [-0.0297, -0.0085] (network clearly better); against the network average -0.0159 [-0.0255, -0.00669] (network clearly better).
    - Median error (the typical row, not pulled by a few bad ones): network average 0.6276, one network 0.624, trees 0.633, top-3 trees averaged 0.6284.

## Crystal-structure networks against trees (split by exact formula)

A crystal-graph network learns from the atom positions. The trees get element fractions, fractions plus cell volume, or (when that job ran) element-property statistics plus neighbour-geometry statistics; the opponent in each verdict is the tree set with the lowest *validation* error. Test error is on formulas the models never saw. All labels are **DFT calculations**, not measurements.

## Crystal-structure networks on HELD-OUT CHEMICAL SYSTEMS

The harder test. Every crystal whose element combination (Fe-O, Li-Fe-P-O, ...) lands in the test set is absent from training, so the test error says how the models do on element combinations they have never seen. Same models, same trees, same verdict rules as above. Expect larger errors than the formula split; what matters is the comparison. All labels are **DFT calculations**.

Neighbour rule: networks built with the fixed rule (every neighbour tied with the 12th is kept; invariant); structure-feature trees with the same.

| Property | Crystals (train / test) | Guess the average | Linear | Trees: fractions | Trees: + cell | Trees: + structure features | Network: mean of single runs | Network: average of runs | Average minus the strongest trees (95% interval) | Verdict: average | Verdict: single runs | Speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bulk modulus (log10 GPa) | 25597 / 3264 | 0.3258 | 0.1749 | 0.1447 | 0.1101 | 0.09154 | 0.1008 (3 runs) | 0.09012 | -0.00143 [-0.00627, +0.00329] vs rich | **no clear difference** | trees clearly better in all 3 runs | 5107 crystals/s |
| Shear modulus (log10 GPa) | 23890 / 3045 | 0.3639 | 0.2571 | 0.2196 | 0.1862 | 0.154 | 0.1808 (3 runs) | 0.1677 | +0.0137 [+0.00906, +0.0187] vs rich | **trees clearly better** | trees clearly better in all 3 runs | 5255 crystals/s |
| Formation energy (eV/atom) | 74920 / 9307 | 0.8116 | 0.4571 | 0.2313 | 0.1988 | 0.1295 | 0.04408 (2 runs) | 0.04027 | -0.0892 [-0.0966, -0.0827] vs rich | **network clearly better** | clearly better in all 2 runs | 3453 crystals/s |
| Band gap (eV) | 74920 / 9307 | 0.9454 | 0.7528 | 0.4144 | 0.3532 | 0.307 | 0.2048 (2 runs) | 0.1916 | -0.115 [-0.134, -0.095] vs rich | **network clearly better** | clearly better in all 2 runs | 5283 crystals/s |

### Bulk modulus (log10 GPa) (held-out systems)

```
network single-run test MAE over 3 runs: mean 0.1008, sd 0.0022
3264 test rows, 1588 formulas; difference = MAE(network) - MAE(trees); negative favours the network

pred_composition: trees MAE 0.1447
  network run 1 (20261127T095223Z_struct_bulk_modulus_kv_sys_s11): MAE 0.0988  diff -0.0459  95% [-0.0558, -0.0368]  network better on 64% of formulas  -> network better
  network run 2 (20261127T100112Z_struct_bulk_modulus_kv_sys_s12): MAE 0.1005  diff -0.0442  95% [-0.0546, -0.0348]  network better on 62% of formulas  -> network better
  network run 3 (20261127T101043Z_struct_bulk_modulus_kv_sys_s13): MAE 0.1031  diff -0.0416  95% [-0.0520, -0.0321]  network better on 62% of formulas  -> network better
  network average of 3 runs: MAE 0.0901  diff -0.0545  95% [-0.0645, -0.0456]  network better on 69% of formulas  -> network better
pred_comp_cell: trees MAE 0.1101
  network run 1 (20261127T095223Z_struct_bulk_modulus_kv_sys_s11): MAE 0.0988  diff -0.0113  95% [-0.0195, -0.0049]  network better on 54% of formulas  -> network better
  network run 2 (20261127T100112Z_struct_bulk_modulus_kv_sys_s12): MAE 0.1005  diff -0.0096  95% [-0.0166, -0.0029]  network better on 54% of formulas  -> network better
  network run 3 (20261127T101043Z_struct_bulk_modulus_kv_sys_s13): MAE 0.1031  diff -0.0070  95% [-0.0151, +0.0006]  network better on 54% of formulas  -> not distinguishable
  network average of 3 runs: MAE 0.0901  diff -0.0199  95% [-0.0274, -0.0135]  network better on 62% of formulas  -> network better
pred_rich: trees MAE 0.0915
  network run 1 (20261127T095223Z_struct_bulk_modulus_kv_sys_s11): MAE 0.0988  diff +0.0072  95% [+0.0015, +0.0130]  network better on 49% of formulas  -> trees better
  network run 2 (20261127T100112Z_struct_bulk_modulus_kv_sys_s12): MAE 0.1005  diff +0.0089  95% [+0.0034, +0.0143]  network better on 47% of formulas  -> trees better
  network run 3 (20261127T101043Z_struct_bulk_modulus_kv_sys_s13): MAE 0.1031  diff +0.0115  95% [+0.0062, +0.0171]  network better on 48% of formulas  -> trees better
  network average of 3 runs: MAE 0.0901  diff -0.0014  95% [-0.0063, +0.0033]  network better on 56% of formulas  -> not distinguishable
```

- Strongest trees by validation error: rich (composition: validation MAE 0.1288, comp+cell: validation MAE 0.1028, rich: validation MAE 0.08998).
  - **Fairness checks** (the headline compares the average of runs with ONE tree model):
    - One network at a time: MAE 0.1008 on average; against the trees it is trees clearly better in all 3 runs.
    - Half average of runs, half trees (no fitted weight): MAE 0.08312; against the trees -0.00843 [-0.0109, -0.00606] (network clearly better); against the average of runs -0.007 [-0.00953, -0.00447] (network clearly better).
    - Median error (the typical row, not pulled by a few bad ones): network average 0.04493, one network 0.05341, trees 0.05058.

### Shear modulus (log10 GPa) (held-out systems)

```
network single-run test MAE over 3 runs: mean 0.1808, sd 0.0017
3045 test rows, 1515 formulas; difference = MAE(network) - MAE(trees); negative favours the network

pred_composition: trees MAE 0.2196
  network run 1 (20261127T101918Z_struct_shear_modulus_gv_sys_s11): MAE 0.1789  diff -0.0407  95% [-0.0521, -0.0305]  network better on 56% of formulas  -> network better
  network run 2 (20261127T102730Z_struct_shear_modulus_gv_sys_s12): MAE 0.1811  diff -0.0385  95% [-0.0488, -0.0283]  network better on 58% of formulas  -> network better
  network run 3 (20261127T103539Z_struct_shear_modulus_gv_sys_s13): MAE 0.1823  diff -0.0373  95% [-0.0488, -0.0264]  network better on 54% of formulas  -> network better
  network average of 3 runs: MAE 0.1677  diff -0.0520  95% [-0.0629, -0.0419]  network better on 62% of formulas  -> network better
pred_comp_cell: trees MAE 0.1862
  network run 1 (20261127T101918Z_struct_shear_modulus_gv_sys_s11): MAE 0.1789  diff -0.0072  95% [-0.0143, -0.0007]  network better on 50% of formulas  -> network better
  network run 2 (20261127T102730Z_struct_shear_modulus_gv_sys_s12): MAE 0.1811  diff -0.0050  95% [-0.0120, +0.0023]  network better on 49% of formulas  -> not distinguishable
  network run 3 (20261127T103539Z_struct_shear_modulus_gv_sys_s13): MAE 0.1823  diff -0.0039  95% [-0.0111, +0.0035]  network better on 48% of formulas  -> not distinguishable
  network average of 3 runs: MAE 0.1677  diff -0.0185  95% [-0.0249, -0.0120]  network better on 54% of formulas  -> network better
pred_rich: trees MAE 0.1540
  network run 1 (20261127T101918Z_struct_shear_modulus_gv_sys_s11): MAE 0.1789  diff +0.0249  95% [+0.0190, +0.0311]  network better on 40% of formulas  -> trees better
  network run 2 (20261127T102730Z_struct_shear_modulus_gv_sys_s12): MAE 0.1811  diff +0.0272  95% [+0.0215, +0.0330]  network better on 39% of formulas  -> trees better
  network run 3 (20261127T103539Z_struct_shear_modulus_gv_sys_s13): MAE 0.1823  diff +0.0283  95% [+0.0226, +0.0346]  network better on 38% of formulas  -> trees better
  network average of 3 runs: MAE 0.1677  diff +0.0137  95% [+0.0091, +0.0187]  network better on 43% of formulas  -> trees better
```

- Strongest trees by validation error: rich (composition: validation MAE 0.1984, comp+cell: validation MAE 0.1736, rich: validation MAE 0.1486).
  - **Fairness checks** (the headline compares the average of runs with ONE tree model):
    - One network at a time: MAE 0.1808 on average; against the trees it is trees clearly better in all 3 runs.
    - Half average of runs, half trees (no fitted weight): MAE 0.1524; against the trees -0.00159 [-0.00432, +0.00113] (no clear difference); against the average of runs -0.0153 [-0.018, -0.0127] (network clearly better).
    - Median error (the typical row, not pulled by a few bad ones): network average 0.101, one network 0.1072, trees 0.09132.

### Formation energy (eV/atom) (held-out systems)

```
network single-run test MAE over 2 runs: mean 0.0441, sd 0.0008
9307 test rows, 3309 formulas; difference = MAE(network) - MAE(trees); negative favours the network

pred_composition: trees MAE 0.2313
  network run 1 (20261127T104348Z_struct_formation_energy_peratom_sys_s21): MAE 0.0435  diff -0.1878  95% [-0.1996, -0.1770]  network better on 91% of formulas  -> network better
  network run 2 (20261127T112309Z_struct_formation_energy_peratom_sys_s22): MAE 0.0446  diff -0.1867  95% [-0.1984, -0.1763]  network better on 91% of formulas  -> network better
  network average of 2 runs: MAE 0.0403  diff -0.1910  95% [-0.2030, -0.1804]  network better on 92% of formulas  -> network better
pred_comp_cell: trees MAE 0.1988
  network run 1 (20261127T104348Z_struct_formation_energy_peratom_sys_s21): MAE 0.0435  diff -0.1553  95% [-0.1659, -0.1458]  network better on 88% of formulas  -> network better
  network run 2 (20261127T112309Z_struct_formation_energy_peratom_sys_s22): MAE 0.0446  diff -0.1542  95% [-0.1644, -0.1447]  network better on 88% of formulas  -> network better
  network average of 2 runs: MAE 0.0403  diff -0.1585  95% [-0.1690, -0.1491]  network better on 90% of formulas  -> network better
pred_rich: trees MAE 0.1295
  network run 1 (20261127T104348Z_struct_formation_energy_peratom_sys_s21): MAE 0.0435  diff -0.0860  95% [-0.0931, -0.0795]  network better on 82% of formulas  -> network better
  network run 2 (20261127T112309Z_struct_formation_energy_peratom_sys_s22): MAE 0.0446  diff -0.0849  95% [-0.0922, -0.0782]  network better on 80% of formulas  -> network better
  network average of 2 runs: MAE 0.0403  diff -0.0892  95% [-0.0966, -0.0827]  network better on 84% of formulas  -> network better
```

- Strongest trees by validation error: rich (composition: validation MAE 0.2029, comp+cell: validation MAE 0.1807, rich: validation MAE 0.1165).
  - **Fairness checks** (the headline compares the average of runs with ONE tree model):
    - One network at a time: MAE 0.04408 on average; against the trees it is clearly better in all 2 runs.
    - Half average of runs, half trees (no fitted weight): MAE 0.0745; against the trees -0.055 [-0.0591, -0.0514] (network clearly better); against the average of runs +0.0342 [+0.0311, +0.0376] (trees clearly better).
    - Median error (the typical row, not pulled by a few bad ones): network average 0.0207, one network 0.02327, trees 0.07002.

### Band gap (eV) (held-out systems)

```
network single-run test MAE over 2 runs: mean 0.2048, sd 0.0065
9307 test rows, 3309 formulas; difference = MAE(network) - MAE(trees); negative favours the network

pred_composition: trees MAE 0.4144
  network run 1 (20261127T115935Z_struct_optb88vdw_bandgap_sys_s11): MAE 0.2094  diff -0.2050  95% [-0.2321, -0.1794]  network better on 39% of formulas  -> network better
  network run 2 (20261127T122419Z_struct_optb88vdw_bandgap_sys_s12): MAE 0.2002  diff -0.2142  95% [-0.2407, -0.1891]  network better on 40% of formulas  -> network better
  network average of 2 runs: MAE 0.1916  diff -0.2228  95% [-0.2505, -0.1974]  network better on 41% of formulas  -> network better
pred_comp_cell: trees MAE 0.3532
  network run 1 (20261127T115935Z_struct_optb88vdw_bandgap_sys_s11): MAE 0.2094  diff -0.1438  95% [-0.1667, -0.1219]  network better on 38% of formulas  -> network better
  network run 2 (20261127T122419Z_struct_optb88vdw_bandgap_sys_s12): MAE 0.2002  diff -0.1530  95% [-0.1764, -0.1305]  network better on 40% of formulas  -> network better
  network average of 2 runs: MAE 0.1916  diff -0.1616  95% [-0.1852, -0.1397]  network better on 41% of formulas  -> network better
pred_rich: trees MAE 0.3070
  network run 1 (20261127T115935Z_struct_optb88vdw_bandgap_sys_s11): MAE 0.2094  diff -0.0976  95% [-0.1167, -0.0763]  network better on 72% of formulas  -> network better
  network run 2 (20261127T122419Z_struct_optb88vdw_bandgap_sys_s12): MAE 0.2002  diff -0.1068  95% [-0.1243, -0.0867]  network better on 76% of formulas  -> network better
  network average of 2 runs: MAE 0.1916  diff -0.1154  95% [-0.1341, -0.0950]  network better on 79% of formulas  -> network better
```

- Strongest trees by validation error: rich (composition: validation MAE 0.3344, comp+cell: validation MAE 0.2935, rich: validation MAE 0.2541).
  - **Fairness checks** (the headline compares the average of runs with ONE tree model):
    - One network at a time: MAE 0.2048 on average; against the trees it is clearly better in all 2 runs.
    - Half average of runs, half trees (no fitted weight): MAE 0.2267; against the trees -0.0804 [-0.0929, -0.0678] (network clearly better); against the average of runs +0.0351 [+0.0252, +0.0434] (trees clearly better).
    - Median error (the typical row, not pulled by a few bad ones): network average 0.01731, one network 0.02131, trees 0.06898.
  - Band gap split: 70% of test crystals have gap exactly 0. On those, network MAE 0.06739 vs trees 0.1495; on the rest, network 0.4857 vs trees 0.6801.

## Prediction intervals (conformal)

For every prediction, an interval "value +/- w" built so that, on rows like the test rows, it contains the true value 90% of the time. w is calibrated on rows the model never trained on (validation crystals; for table families, the other cross-validation folds) and then **checked** on rows the calibration never saw. "Coverage" is how often the interval really contained the truth: 0.90 for a 90% interval means it can be trusted; well below means it is too narrow. For log10 targets a half-width w means "within a factor 10^w". Scaled intervals use the spread between network seeds, so confident rows get narrow intervals.

| Task | Split | Model | 90% half-width | 90% coverage | 80% coverage | 95% coverage | Scaled 90%: median half-width / coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Alloys (hardness) | - | trees | 161.6 | 0.885 | 0.808 | 0.955 | - |
| Alloys (hardness) | - | network average | 181.3 | 0.883 | 0.805 | 0.911 | - |
| Alloys (hardness) | - | blend | 154.8 | 0.883 | 0.803 | 0.948 | - |
| Polymers (glass transition) | - | trees | 41.79 | 0.905 | 0.800 | 0.951 | - |
| Polymers (glass transition) | - | network average | 40.41 | 0.901 | 0.804 | 0.953 | - |
| Polymers (glass transition) | - | blend | 37.6 | 0.901 | 0.809 | 0.951 | - |
| Solid electrolytes (conductivity) | - | trees | 3.555 | 0.897 | 0.810 | 0.939 | - |
| Solid electrolytes (conductivity) | - | network average | 5.42 | 0.900 | 0.797 | 0.954 | - |
| Solid electrolytes (conductivity) | - | blend | 4.184 | 0.900 | 0.793 | 0.948 | - |
| Insulators (dielectric constant, calculated) | - | trees | 0.259 | 0.903 | 0.803 | 0.953 | - |
| Insulators (dielectric constant, calculated) | - | network average | 0.2504 | 0.903 | 0.795 | 0.949 | - |
| Insulators (dielectric constant, calculated) | - | blend | 0.2462 | 0.905 | 0.801 | 0.951 | - |
| Concrete (strength, ages 14-28 d from 1-7 d) | - | trees | 20.42 | 0.863 | 0.762 | 0.938 | - |
| Concrete (strength, ages 14-28 d from 1-7 d) | - | network average | 14.01 | 0.919 | 0.825 | 0.938 | - |
| Concrete (strength, ages 14-28 d from 1-7 d) | - | blend | 15.72 | 0.894 | 0.756 | 0.944 | - |
| Chemistry (water solubility) | - | trees | 1.758 | 0.900 | 0.800 | 0.949 | - |
| Chemistry (water solubility) | - | network average | 1.694 | 0.900 | 0.800 | 0.950 | - |
| Chemistry (water solubility) | - | blend | 1.667 | 0.901 | 0.800 | 0.950 | - |
| Surfactants (critical micelle concentration) | - | trees | 1.242 | 0.899 | 0.806 | 0.953 | - |
| Surfactants (critical micelle concentration) | - | network average | 1.281 | 0.903 | 0.799 | 0.950 | - |
| Surfactants (critical micelle concentration) | - | blend | 1.219 | 0.899 | 0.799 | 0.951 | - |
| Surfactants (surface tension at the CMC) | - | trees | 10.23 | 0.900 | 0.798 | 0.934 | - |
| Surfactants (surface tension at the CMC) | - | network average | 11.3 | 0.893 | 0.807 | 0.939 | - |
| Surfactants (surface tension at the CMC) | - | blend | 10.33 | 0.895 | 0.803 | 0.936 | - |
| Corrosion (rate, NIST CORR-DATA, exploratory) | - | trees | 1.611 | 0.898 | 0.802 | 0.949 | - |
| Corrosion (rate, NIST CORR-DATA, exploratory) | - | network average | 1.606 | 0.900 | 0.801 | 0.950 | - |
| Corrosion (rate, NIST CORR-DATA, exploratory) | - | blend | 1.565 | 0.898 | 0.799 | 0.949 | - |
| Bulk modulus (log10 GPa) | system, fixed rule | network average (3 network runs) | 0.1603 | 0.873 | 0.750 | 0.947 | 0.1591 / 0.892 |
| Bulk modulus (log10 GPa) | system, fixed rule | trees | 0.1621 | 0.887 | 0.796 | 0.943 | - |
| Bulk modulus (log10 GPa) | system, fixed rule | blend (3 network runs) | 0.1395 | 0.876 | 0.771 | 0.940 | - |
| Shear modulus (log10 GPa) | system, fixed rule | network average (3 network runs) | 0.3377 | 0.893 | 0.785 | 0.949 | 0.3861 / 0.891 |
| Shear modulus (log10 GPa) | system, fixed rule | trees | 0.2982 | 0.889 | 0.785 | 0.948 | - |
| Shear modulus (log10 GPa) | system, fixed rule | blend (3 network runs) | 0.2995 | 0.889 | 0.788 | 0.946 | - |
| Formation energy (eV/atom) | system, fixed rule | network average (2 network runs) | 0.07798 | 0.884 | 0.778 | 0.938 | 0.1081 / 0.903 |
| Formation energy (eV/atom) | system, fixed rule | trees | 0.281 | 0.882 | 0.778 | 0.933 | - |
| Formation energy (eV/atom) | system, fixed rule | blend (2 network runs) | 0.1543 | 0.878 | 0.776 | 0.931 | - |
| Band gap (eV) | system, fixed rule | network average (2 network runs) | 0.4508 | 0.871 | 0.770 | 0.931 | 0.1041 / 0.902 |
| Band gap (eV) | system, fixed rule | trees | 0.7572 | 0.868 | 0.760 | 0.933 | - |
| Band gap (eV) | system, fixed rule | blend (2 network runs) | 0.5579 | 0.872 | 0.766 | 0.930 | - |


- These are **marginal** guarantees (true on average over rows like the test rows), not a promise for each single material. A material unlike anything in the data (a new element, a far-off composition) can be outside its interval more often; the held-out-elements rows show how much.

## What each job did

| Job | Status | Minutes | Attempts |
| --- | --- | --- | --- |
| tab_alloys | done | 7.6 | 1 |
| base_alloys | done | 0.2 | 1 |
| tab_polymers | done | 7.0 | 1 |
| base_polymers | done | 0.4 | 1 |
| tab_electrolytes | done | 14.7 | 1 |
| base_electrolytes | done | 0.5 | 1 |
| tab_insulators | done | 6.4 | 1 |
| base_insulators | done | 0.5 | 1 |
| tab_concrete | done | 1.8 | 1 |
| base_concrete | done | 0.2 | 1 |
| tab_chemistry | done | 15.7 | 1 |
| base_chemistry | done | 1.0 | 1 |
| tab_cleaning_pcmc | done | 59.7 | 1 |
| base_cleaning_pcmc | done | 6.3 | 1 |
| tab_cleaning_tension | done | 40.7 | 1 |
| base_cleaning_tension | done | 5.5 | 1 |
| tab_corrosion | done | 46.9 | 1 |
| base_corrosion | done | 6.0 | 1 |
| struct_bulk_modulus_kv_sys_s11 | done | 8.8 | 1 |
| struct_bulk_modulus_kv_sys_s12 | done | 9.5 | 1 |
| struct_bulk_modulus_kv_sys_s13 | done | 8.6 | 1 |
| trees_bulk_modulus_kv_sys | done | 4.0 | 1 |
| struct_shear_modulus_gv_sys_s11 | done | 8.2 | 1 |
| struct_shear_modulus_gv_sys_s12 | done | 8.2 | 1 |
| struct_shear_modulus_gv_sys_s13 | done | 8.1 | 1 |
| trees_shear_modulus_gv_sys | done | 3.8 | 1 |
| struct_formation_energy_peratom_sys_s21 | done | 39.2 | 1 |
| struct_formation_energy_peratom_sys_s22 | done | 36.5 | 1 |
| trees_formation_energy_peratom_sys | done | 23.3 | 1 |
| struct_optb88vdw_bandgap_sys_s11 | done | 24.7 | 1 |
| struct_optb88vdw_bandgap_sys_s12 | done | 24.6 | 1 |
| trees_optb88vdw_bandgap_sys | done | 32.7 | 1 |

## What this does and does not show

- Every number is from historical data or DFT calculations. None of it is a new measurement, so none of it is evidence of a discovery; a sealed set of new measurements is still the missing evidence.
- Tabular families: the old fixed test splits were already looked at by earlier work, so this report uses cross-validation with cross-fitted selection instead. Concrete is exploratory.
- Crystal properties on the formula split: crystals of the same structure type with different elements can sit on both sides, so expect the real error on genuinely new chemistries to be larger. The held-out-systems section (when present) is the harder test; even it keeps every element and most sub-systems in training, so it is not a "new element" test.
- Each comparison is one of many; with a 95% interval, a few "clear" results are expected by chance. Treat single clear wins as leads to confirm, not as findings.
- Rows of a family that share a source (paper, alloy system, mix) were always kept together in a fold.

## Suggested next steps

1. Held-out chemical systems (networks by rule: fixed rule): network average clearly better on Formation energy (eV/atom), Band gap (eV); trees clearly better on Shear modulus (log10 GPa). This is the comparison to quote.
2. The average of network seeds clearly beats the tuned trees on: Polymers (glass transition), Concrete (strength, ages 14-28 d from 1-7 d), Chemistry (water solubility). Single networks may not; see the "one network at a time" column.
