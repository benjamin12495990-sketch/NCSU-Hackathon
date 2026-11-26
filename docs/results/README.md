# Results from the laptop pipeline

Pipeline output (`gpu/runs/`) stays out of git (`.gitignore`): run folders hold weights, caches and logs. Reports worth keeping
are copied into this folder by hand. Each copy is the `REPORT.md` the pipeline wrote, unchanged except for the `Folder:` line
in its header, which gives the folder relative to the repo instead of the full local path. The original, `share.zip` and every
result file stay in the run folder on the laptop. A pipeline folder that is extended later (`-TieFix`, a new cycle) writes a
new `REPORT.md`, so each copy is named after the folder and the time its report was generated.

| Report | Pipeline folder | Generated (laptop clock) | Jobs | Covers |
| --- | --- | --- | --- | --- |
| [20261122T014943Z_overnight_REPORT_20261126T0916.md](20261122T014943Z_overnight_REPORT_20261126T0916.md) | `gpu/runs/pipeline/20261122T014943Z_overnight` | 2026-11-26 09:16:58 | 181 finished, 0 failed, 27.56 GPU hours | cycles 1-3: 9 table families; 4 crystal properties on formula, chemical-system and element splits; fine-tuning; evolutionary search (2 rounds for formation energy); conformal intervals |

To add the next one, on the laptop from the repo root (PowerShell). It refuses to replace a copy that is already here:

```powershell
$p = (Get-Content gpu\runs\pipeline\LAST_PIPELINE.txt -Raw).Trim()
$t = (Get-Item "$p\REPORT.md").LastWriteTime.ToString('yyyyMMddTHHmm')
$d = "docs\results\$(Split-Path $p -Leaf)_REPORT_$t.md"
if (Test-Path $d) { throw "$d already exists" } else { Copy-Item "$p\REPORT.md" $d }
```

Reports written by the code from 2026-11-26 on print the folder relative to the repo, and `share.zip` replaces this
computer's paths with `<repo>` and `~`. In an older report, replace the full local path on the `Folder:` line with
`gpu\runs\pipeline\<folder>` before committing; it contains the Windows user name. If you paste a report into a chat, read
it with `Get-Content -Raw -Encoding UTF8`: Windows PowerShell 5.1 otherwise reads the file as ANSI and turns characters
such as the em dash into `â€”`.

## What the 2026-11-26 report shows

Read the report for the numbers; this is the short version. Wording follows the table in `docs/AGENTS.md`. Every label is a
DFT calculation or a historical measurement, so none of this is new evidence about a material.

**Neighbour rule.** Every crystal network in this report, and the structure-feature ("rich") trees, which build their
neighbour-geometry features with the same 12-nearest-neighbour cut, used the original rule that breaks ties by listing
order (LES-017). `-TieFix` trains the held-out-systems networks (default settings) and trees again with the fixed rule, so
it measures the change on that split only. The formula-split, held-out-element, fine-tuning and search numbers stay
original-rule numbers until they are trained again.

**Already-scored test splits.** The formula and chemical-system test crystals were scored in cycles 1 and 2, and those
scores chose what cycle 3 did (search formation energy further, fine-tune the moduli). Cycle-3 results on those test
crystals (round 2 of the search, fine-tuning on the formula and system splits) are therefore marked exploratory. The
held-out-element split is new in cycle 3.

Intervals are 95% paired-bootstrap intervals of (error of A minus error of B), resampling whole formulas or groups.
"Strongest trees" means the tree model with the lowest validation error, which was the one with element-property and
neighbour-geometry features ("rich") in every crystal section.

### Crystals

1. **Held-out chemical systems (the comparison to quote).** Formation energy and band gap networks are clearly better than
   the strongest trees, for the average of 2 runs and for each run: formation energy -0.0834 [-0.0909, -0.076] eV/atom,
   band gap -0.115 [-0.135, -0.0934] eV. Shear modulus: trees clearly better, for the average of 3 runs (+0.00548
   [+0.000763, +0.0102] log10 GPa) and for each run. Bulk modulus: no clear difference for the average of 3 runs
   (+0.00177 [-0.00214, +0.00573]); trees clearly better than each single run.
2. **Held-out elements, default networks (one learned vector per element).** Trees clearly better on all four properties,
   for the average of 2 runs and for each run: bulk modulus +0.0545 [+0.0491, +0.0601] and shear modulus +0.0655
   [+0.0592, +0.0725] log10 GPa, formation energy +0.323 [+0.307, +0.342] eV/atom (network average 0.4796 against trees
   0.1563), band gap +0.107 [+0.0805, +0.133] eV. An element never seen in training keeps its random starting vector, so
   this network cannot be used on new elements.
3. **Held-out elements, element-property inputs (`--atom-input both`).** Formation energy: network clearly better than
   the strongest trees, for the average of 2 runs and for each run, -0.0223 [-0.0272, -0.0173] eV/atom. Band gap: no
   clear difference, -0.00716 [-0.0214, +0.00579] eV. Bulk and shear modulus: trees still clearly better, for the average
   and for each run (+0.00576 [+0.00164, +0.00993] and +0.00853 [+0.00335, +0.0137] log10 GPa), against +0.0545 and
   +0.0655 for the learned-vector networks on the same rows. The report has no interval for the difference between the two
   kinds of network.
4. **Band gap on held-out elements, by sub-split (no interval).** 76% of the held-out-element test crystals have a DFT gap
   of exactly 0. On those the networks' MAE is lower than the trees' (0.162 property inputs, 0.174 learned vectors, against
   0.202); on the crystals with a gap it is higher (0.9719 and 1.408 against 0.8752). On held-out chemical systems the
   network's MAE is lower on both parts (gapped crystals 0.4841 against 0.6856).
5. **Moduli: a half-and-half blend beats the strongest trees on 5 of 6 property/split pairs.** Half the network average
   plus half the strongest trees, no fitted weight, is clearly better than the trees for bulk modulus on the formula split
   (-0.00609 [-0.00813, -0.00393]), held-out systems (-0.00482 [-0.00713, -0.00264]) and held-out elements with property
   inputs (-0.00802 [-0.0105, -0.00558]), and for shear modulus on held-out systems (-0.00517 [-0.00781, -0.00249]) and
   held-out elements with property inputs (-0.00512 [-0.00778, -0.00241]). Shear on the formula split: no clear difference
   (-0.00209 [-0.00455, +0.000394]); there the fine-tuned network average has a lower MAE than the blend (0.1487 against
   0.1501, no interval). With learned element vectors on held-out elements the blend is clearly worse than the trees
   (bulk +0.02 [+0.0171, +0.0228], shear +0.024 [+0.0206, +0.0277]). The blend uses the average of 2 or 3 networks and the
   report has no single-network check of it; it was not compared with the fine-tuned or evolved networks.
6. **Search.** Formation energy: the average of 2 runs of the evolved settings is clearly better than the average of 2
   default runs on held-out systems, -0.00535 [-0.00699, -0.00366] eV/atom (0.04122 against 0.04657). Round 2
   (atom_features 160, conv 6) against the round-1 winner, exploratory (test systems already scored in cycle 2):
   -0.00122 [-0.00224, -0.000147] (0.04 against 0.04122, about 3%), with the upper end of the interval close to zero; its
   finalist runs took 48.5-48.6 minutes each against 36.1 for the round-1 finalists. The report has no per-run verdict for
   these comparisons. Evolved against default settings, no clear difference: bulk modulus -0.00384 [-0.00849, +0.000641]
   and shear modulus +0.00107 [-0.00283, +0.00509] log10 GPa, band gap -0.00601 [-0.013, +0.00134] eV.
7. **Fine-tuning from the formation-energy network** (average of runs; the report has no per-run verdict for fine-tuned
   against from scratch; formula and system splits exploratory, as above). Fine-tuned clearly better than from scratch only
   for shear modulus on the formula split, -0.0146 [-0.0188, -0.0107]. No clear difference for bulk modulus on the formula
   split (+0.00333 [-0.000392, +0.00699]), held-out systems (-0.00157 [-0.00547, +0.00344]) and held-out elements
   (+0.00129 [-0.00145, +0.00405]), or for shear on held-out systems (-0.00328 [-0.00812, +0.0016]). From scratch clearly
   better for shear on held-out elements, +0.012 [+0.00593, +0.0179]. No single fine-tuned run is clearly better than the
   strongest trees on any split. Every fine-tuned seed of a split starts from the same formation-energy network, and the
   learning rate differs (1e-3 against 2e-3), so this tests the recipe, not fine-tuning in general.

### Prediction intervals

The 90% intervals cover 0.87 to 0.93 of test rows in 44 of 59 rows of the table. They are too narrow on held-out elements
for band gap (0.832 to 0.846 for networks, trees and blends alike) and for formation energy with learned element vectors
(0.851), and for concrete trees (0.863, exploratory). They are wider than needed on held-out elements for the moduli trees
(0.936, 0.937) and for the property-input blends (bulk 0.940, shear 0.932, formation energy 0.931). The intervals are
calibrated on validation crystals whose unseen elements (Tb, Tm, Yb, Pr, Ag, Mo) differ from the test crystals' (Pu, Pa,
Hf, I, He, Ce, Cr), so "90%" should not be quoted for a material with an element outside training.

### Table families

- One network at a time is never clearly better than the tuned trees in any of 5 runs of any family, except concrete
  (exploratory: clearly better in 2 of 5, clearly worse in 2).
- The average of 5 networks is clearly better than one tree model on polymers (-1.51 [-2.89, -0.0985] K), water solubility
  (-0.0284 [-0.0398, -0.017]) and concrete (exploratory). Against the average of the 3 best tree models, only water
  solubility stays clearly better (-0.026 [-0.0372, -0.0151]), with concrete (exploratory, -6.24 [-7.92, -4.53] MPa). The
  3 best tree models are clearly better than the network average on solid electrolytes (+0.595 [+0.178, +1.13]), critical
  micelle concentration (+0.0628 [+0.00367, +0.112]) and surface tension (+0.447 [+0.232, +0.67]).
- Half network average, half trees is clearly better than the trees on polymers (-2.02 [-2.87, -1.25]), insulators
  (-0.00594 [-0.00869, -0.00313]) and water solubility (-0.0438 [-0.0499, -0.0377]), and on corrosion (-0.0202 [-0.0297,
  -0.0085]) and concrete (-3.92 [-4.68, -3.07]), both exploratory.
- Concrete (exploratory): the average of 5 networks has MAE 6.251 while one network at a time averages 12.91, and the
  median error of one network is 6.85. A few seeds miss badly when extrapolating from ages 1-7 d to 14-28 d, and averaging
  hides it. Five seeds are few to rely on that.

### Predictions made before cycle 3 ran

Stated in the working chat before the run, not saved in a file, so this is not a preregistration. Four predictions are too
few to say anything about calibration.

| Prediction | Stated chance | Outcome |
| --- | --- | --- |
| The default network loses to trees on held-out elements | about 75% | Happened: trees clearly better on all 4 properties, average and every run |
| Element-property inputs win back formation energy and band gap | about 50% | Half: formation energy clearly better than trees; band gap no clear difference |
| Fine-tuning closes the moduli gap | about 45% | Mostly not: clearly better than scratch only for shear on the formula split (exploratory); no fine-tuned run beats the trees |
| Round 2 of the search gains a few percent | about 55% | Happened, exploratory: about 3% for the average of 2 runs, upper end of the interval -0.000147 |

### What it does not show

- The crystal networks and the rich trees used the original neighbour rule (above).
- Two or three runs per setting. Of the more than 100 comparisons in the report, a few "clear" results are expected by
  chance at 95%.
- The held-out-element split is one fixed choice of 7 test and 6 validation elements. The intervals resample formulas
  within that choice, not the choice of elements; another set of elements could rank the models differently.
- Element-property inputs were tested on the held-out-element split only, so this report says nothing about them on the
  formula or chemical-system splits.

### Leads for the next cycle (not decisions)

Changing a default or `gpu/recipes.json` needs a reviewed commit and a row in `docs/DECISIONS.md`.

1. Run `-TieFix` first; it re-measures the held-out-systems comparison with the fixed rule. The other crystal numbers need
   their own retraining before they stop being original-rule numbers.
2. Run element-property-input networks on the chemical-system split. If they are not worse there, they could be the default,
   since they are the only crystal networks here that work on unseen elements.
3. For moduli, score the half-and-half blend as its own model in the nightly run, so it gets single-run and interval checks
   of its own.
4. Formation energy: the round-2 settings have the lowest test MAE seen so far (exploratory), at about 35% more training
   time per run. A confirmation needs crystals not scored before; another score on the same test systems is exploratory too.
