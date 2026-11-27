# How much physics and chemistry the models capture

Written 2026-11-27. This is an assessment, not a new result: nothing was trained or scored for it. The only checks run for
it were the three tree-model probes in section 4 (re-run, every number reproduced), the JARVIS-Leaderboard comparison in
section 5 (re-run) and a few file checks, each named where it is used.

Labels in the conclusions: **[Certain]** strong evidence, **[Likely]** evidence that is not proof, **[Speculative]** an
educated guess. DFT values are DFT-calculated, never measured (DEC-002). Wording of comparisons follows the table in
`docs/AGENTS.md`.

## 1. The short answer

**The graduate student / PhD / researcher scale does not fit these models.** [Certain] That scale describes people who
reason with laws, design experiments and explain results. Every model in this project is a property regressor: it maps a
fixed set of inputs (a concrete recipe and an age, an alloy composition, a crystal graph) to one number, fitted to
DFT-calculated labels or to historical laboratory measurements. It holds no law as such. It interpolates between the rows
it was fitted to, and it is only known to be right where it was tested.

"PhD level" comes from question-answering benchmarks. GPQA has 448 main and 546 extended expert-written questions; on
the extended set PhD-level experts score 65% (74% after discounting clear mistakes), 57.3% on its 227 physics questions
and 72.0% on its 214 chemistry questions ([arXiv 2311.12022](https://arxiv.org/abs/2311.12022)). A regressor cannot sit that
test: it takes a composition and returns a number.

The fairest published comparisons that do exist:

| Framework | Its levels | Where these models sit |
| --- | --- | --- |
| Levels of AGI, Morris et al., ICML 2024 ([arXiv 2311.02462](https://arxiv.org/abs/2311.02462)) | Performance 0 (no AI) to 5 (superhuman), each narrow or general; autonomy from "tool" to "agent". The top narrow cell holds chess engines such as Stockfish. | Narrow AI, used as a tool. The performance level is not measured: it needs a comparison with skilled people estimating the same property, and nobody has run one. |
| Agentic science survey, Wei et al. ([arXiv 2508.14111](https://arxiv.org/abs/2508.14111)) | Level 1 computational oracle ("sophisticated function approximators that require constant human guidance"), 2 automated research assistant, 3 autonomous partner, 4 generative architect | Level 1. The nightly pipeline automates training and reports, but a person makes every model decision (DEC-018). |
| Krenn et al., Nat Rev Phys 2022 ([doi:10.1038/s42254-022-00518-3](https://doi.org/10.1038/s42254-022-00518-3)) | Computational microscope, source of inspiration, agent of understanding | At most an instrument that people interpret. The authors write "we have not found any evidence of computers acting as true agents of understanding in science yet" and "naive applications of neural networks cannot be agents of understanding". |

[Likely] Narrow AI tool, Level 1 oracle. This placement is this document's reading of the frameworks, not a published
rating of the project.

**Accuracy within the data, against published models.** Published numbers: ALIGNN paper Table 3, JARVIS-DFT 2021.8.18,
55,722 materials, random 80/10/10 split ([arXiv 2106.01829](https://arxiv.org/abs/2106.01829)). Project numbers: the
2026-11-26 pipeline report, `docs/results/20261122T014943Z_overnight_REPORT_20261126T2001.md`, single-run mean (average
of 2 or 3 runs in brackets). The formula split uses the original neighbour rule, the held-out-systems split the fixed one;
both test splits were scored in cycles 1 and 2, so these numbers are exploratory (`docs/results/README.md`,
"Already-scored test splits"; LES-019).

| Property (DFT-calculated) | Project, split by formula | Project, held-out systems | CGCNN / ALIGNN (published) | Like for like? |
| --- | --- | --- | --- | --- |
| Formation energy | 0.0480 (0.0437) eV/atom | 0.0525 (0.0477) eV/atom | 0.063 / 0.033 eV/atom | Partly. Same database family and functional (OptB88vdW). The project file is newer (2025-09-24, 93,902 crystals, 1.7 times as many). Its splits are stricter: no formula, or no chemical system, sits on both sides, while a random split lets close relatives straddle and is optimistic ([MD-HIT](https://doi.org/10.1038/s41524-024-01426-z); [Omee 2024](https://doi.org/10.1038/s41524-024-01316-4)). |
| Band gap (OptB88vdW) | 0.1795 (0.1687) eV | 0.2038 (0.1912) eV | 0.20 / 0.14 eV | Partly, as above. |
| Bulk modulus Kv | 0.102 (0.0915) log10 GPa | 0.1012 (0.0902) log10 GPa | 14.47 / 10.40 GPa | No. The project error is in log10 GPa; converting needs the test predictions, which stay on the laptop. |
| Shear modulus Gv | 0.1746 (0.1633) log10 GPa | 0.1808 (0.1677) log10 GPa | 11.75 / 9.48 GPa | No (units). |

The nearest log-unit yardstick for shear is Matbench log_gvrh (Materials Project, Voigt-Reuss-Hill averaging, filtered to
crystals within 150 meV of the hull and with bounded moduli): best random-split model 0.0670 log10 GPa, and on a held-out
cluster split CGCNN 0.0585 and ALIGNN 0.0974 (Omee 2024, Tables 2 and 4). The project's shear error is about 2.5 times
larger, on a different database: JARVIS reports Voigt averages (median Gv/G_VRH 1.072 inside the cached file, about 0.03
log10), and the project keeps every crystal with Gv > 0 with no stability filter (`gpu/structure/train_structure.py`).
[Speculative] Unstable or soft crystals in the labels explain part of the gap; nothing has tested this.

Matbench Discovery scores a different task: picking stable crystals. CGCNN reaches F1 0.507
([arXiv 2308.14920](https://arxiv.org/abs/2308.14920) v3, Table 1); the November 2026 leader, DPA-4.1-Max-OAM-PFT,
reaches F1 0.935 (snapshot in [arXiv 2611.33538](https://arxiv.org/abs/2611.33538); the live site was not reachable). The
project's models were never run on it, and a low formation-energy error does not carry over: CGCNN's formation-energy
parity R2 is 0.93 while its hull-distance R2 is -0.60 (Matbench Discovery), and "accurate predictions of formation energy
do not imply accurate predictions of stability" ([Bartel 2020](https://doi.org/10.1038/s41524-020-00362-y)).

- [Likely] On formation energy and band gap the networks sit between the published CGCNN and ALIGNN numbers on the same
  database family, except single runs on held-out systems for band gap (0.2038 against CGCNN's 0.20 eV). This is not a
  ranking: data version and split both differ.
- [Certain] Against tuned trees that see structure statistics, on the same rows and held-out chemical systems
  (exploratory): the networks are clearly better on formation energy and band gap in every run, and not better on the
  moduli (section 4).
- [Certain] Nothing here places the models on a stability benchmark such as Matbench Discovery.

## 2. The hidden assumption: "all known laws of physics completely known"

**A regressor cannot have this property.** [Certain] A law holds for every input, including ones never measured. A
regressor is fitted to finite rows and is checked only where someone tests it. What a model can have is (a) a chosen set
of laws built into its structure, and (b) a test suite that measures how well it obeys the rest.

**Accurate prediction does not show that a law was learned.**
- Vafa et al., ICML 2025 ([arXiv 2507.06952](https://arxiv.org/abs/2507.06952)): a 109M-parameter transformer trained on
  10M simulated solar systems predicts orbits with R2 > 0.9999. Fine-tuned to output forces, it gives a force law that
  symbolic regression shows to be nonsensical, and a different one for different samples.
- Liu et al. ([arXiv 2602.06923](https://arxiv.org/abs/2602.06923)): with a context of 2 steps the model encodes the
  gravitational force (probe R2 about 0.999); with 100 steps it encodes ellipse parameters (R2 about 0.998), a curve
  fitter that predicts better over long horizons. "Prediction is necessary but not sufficient."
- Yuan and Liu ([arXiv 2610.06877](https://arxiv.org/abs/2610.06877), abstract only): distinct dynamics can generate
  identical records, so fitting data cannot by itself certify a law.
- In this project: the lab's plain concrete network broke the age and water rules on only 0.6% of query pairs on the
  sealed-test mixes, yet missed the sealed 56-to-365-day results by 113.6 MPa RMSE (`lab/results/vault_opening.json`).
  [Certain] Passing a physics probe does not make a model accurate.

**Three ways physics gets into a model** (Karniadakis et al. 2021,
[doi:10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5), Box 2): observational bias (it is in the
data), inductive bias (the architecture guarantees it, for example invariance) and learning bias (a loss penalty or hard
constraint, for example monotonicity). Unconstrained trees and MLPs get only the first.

**Built-in physics pays off mostly when extrapolating along the constrained variable.**

| Study | Built in | Inside the data | Extrapolating |
| --- | --- | --- | --- |
| Zirconia conductivity, 1,351 records ([Zhang 2026](https://doi.org/10.1088/2632-2153/aeb9a7), abstract) | Arrhenius form | On held-out material groups the unconstrained network had the lower RMSE | Trained at 800 °C or below, tested at 900 °C or above (47 points): RMSE 0.4187 to 0.3609 log10 S/cm, R2 0.8562 to 0.8932 |
| This project's concrete lab: models trained on ages 1 to 7 days (`lab/README.md`; sealed test in `lab/results/vault_opening.json`, DEC-008, LES-006) | ln(age) as an input; synthetic pre-training from the Powers and fib strength laws | Not reported separately | Ages 14 to 28 d (MONITOR): plain network 8.57 MPa RMSE with pre-training, 14.24 without (medians of 10 seeds; no interval). Ages 56 to 365 d (271 sealed results, scored once): ridge on ln(age) 9.07, pre-trained plain network 11.29, champion 11.50, fib equation alone 13.06, gradient-boosted trees 24.28, random forest 25.44, plain network without pre-training 113.6 MPa |
| UCI concrete, 1,030 mixes ([Sadeghi 2026](https://al-kindipublisher.com/index.php/jcsts/article/download/13366/12176); its DOI is not registered) | Four physics features (w/b, binder, aggregate/binder, log age) | No significant accuracy gain for XGBoost (paired t-test p = 0.74) | Not tested (random splits) |
| Coconut-fibre concrete, 586 records from 30 studies ([Thapa 2026](https://doi.org/10.1007/s44379-026-00104-9), abstract) | Abrams law, hydration kinetics, monotonicity penalty | Physics-guided network R2 0.9482 against 0.9673 for gradient boosting; full age-monotonicity compliance | Not reported |
| UCI-type concrete, 1,030 mixes ([Qu 2026](https://doi.org/10.1371/journal.pone.0350575), abstract) | Physics baseline plus a residual network | R2 0.9252, RMSE 4.39 MPa; similar accuracy from 40% of the training data | Not reported |
| Extrapolation benchmark ([Muckley 2023](https://doi.org/10.1039/d3dd00082f), one-line abstract) | (simple linear models) | - | "Simple linear models perform equivalently to black box machine learning models on extrapolation tasks" |

Two corrections from the source check:
- arXiv [2608.03805](https://arxiv.org/abs/2608.03805) (refractory alloys, 393 DFT-calculated alloys) does not build
  Born stability into its trees. It checks it after prediction, and credits its 100% compliance to Born-stable training
  data and feature design. Inside the data, the data carried the physics.
- Mazumder 2026 ([doi:10.1007/s41062-026-02981-8](https://doi.org/10.1007/s41062-026-02981-8)) is mainly a three-class
  strength classifier; its R2 of 0.834 comes from a derived output, and its authors say the constraints helped. It is not
  evidence that consistency costs accuracy, and it is not used here.

New elements are a separate kind of extrapolation. On the Materials Project, most leave-one-element-out tasks still
interpolate in representation space (85% of ALIGNN tasks and 68% of XGBoost tasks reach R2 > 0.95); the failures sit on
H, F and O, and for truly out-of-domain chemistry more data can hurt (leave-H-out error rose 5 times as training grew from
10^4 to 10^6 crystals) ([Li et al., arXiv 2406.06489](https://arxiv.org/abs/2406.06489)). Models rarely predict values
beyond their training range ([Known Unknowns, arXiv 2502.05970](https://arxiv.org/abs/2502.05970)); trees cannot, by
construction, which the sealed concrete test shows (24.28 and 25.44 MPa).

- [Likely] Built-in physics buys consistency and extrapolation along the constrained variable (age, temperature), and
  little accuracy inside the data or on new compositions.
- [Likely] For this project the largest gains from physics are on age (concrete) and temperature (corrosion,
  electrolytes once temperature data exist).

## 3. What is built in today, and what is not

| Model | Built in | Status |
| --- | --- | --- |
| Crystal network (`gpu/structure/model.py`, `graph.py`) | Atoms enter only by element; edges only by distance (40 Gaussians, 0 to 8 Å); messages summed, then averaged over atoms. So the prediction does not change under rotation, translation, mirroring (it cannot tell left- from right-handed crystals), relisting of atoms or a supercell. | Exact only under the fixed neighbour rule (DEC-020). Measured on 300 held-out-systems test crystals, one network per property (seed 11): largest change 2.38e-7 to 1.43e-6 of the unit, no rewrite above 1% of the spread. Under the original rule 0.3 to 1.8% of rewrites moved more than that, by up to 0.044 and 0.0492 log10 GPa, 0.098 eV/atom and 0.025 eV (report, "Neighbour-rule fix"). The formula-split, held-out-element, fine-tuning and search runs are still original-rule runs. |
| Moduli, dielectric constant, conductivity, corrosion rate, pCMC | Trained on log10 values, so the prediction is always positive | By construction. It does not stop a dielectric constant below 1. |
| GPU-kit networks (`gpu/train_tabular.py`) | Inputs clipped at 3 standard deviations (alloys, electrolytes) or 5 (chemistry, both surfactant tasks, corrosion); alloy outputs also clamped to the training range | By construction. A range limit, not physics: predictions go flat when extrapolating. |
| Polymers (`domains/polymers/work/prepare.py`) | A Fox-equation Tg from training homopolymers (4 columns) and 4x(1-x) as inputs. Monomers are put in sorted order (line 65) and SMILES are canonicalised, so swapping the two monomers cannot change the prediction for any row that goes through `prepare.py`. | Swap symmetry holds by construction for those rows; hand-built inputs that bypass `prepare.py` are not covered. |
| Alloys (`domains/materials/work/prepare.py`) | Hume-Rothery-style descriptors as inputs: mixing entropy, radius mismatch, electronegativity and valence-electron mean and spread, mean melting point | Inputs only; no constraint. |
| Lab concrete networks (`lab/`) | Synthetic pre-training from Powers' gel-space law and the fib curve, each calibrated on TRAIN; Abrams, Bolomey and Feret inputs as an option (`lab/genome2.mjs`; on the page, the "textbook strength laws" design, `PANEL_DESIGN` in `src/lib/nn.js`); selection penalised for age violations (f(t) against f(2t)) and water violations (`lab/reward2.mjs`) | The only concrete models with textbook laws built in (the page's network gets the same law inputs only with its "textbook strength laws" switch on, `src/components/NeuralNet.jsx`). The site's concrete trees (`train/train.py`, GradientBoostingRegressor) and the GPU-kit concrete networks (the nine raw inputs, `gpu/data.py`) get none of this. |

Existing checks: `gpu/structure/invariance.py` (networks only; scheduled only in the neighbour-fix plan,
`gpu/pipeline/plan.py`, not every night), `lab/reward2.mjs` (the lab networks), and `scripts/sweep.mjs` (`npm run sweep`:
the page's concrete network, which passes if it gets the overall water and age direction right on more than half of the
mixes; not run for this document, because it reads two of the blocked files).

Nothing in the project enforces:
- conservation of mass or volume, or units (no model knows units);
- a band gap of at least 0 (the network's head ends in a plain linear layer) or a dielectric constant of at least 1;
- formation energies tied to elemental reference states or to the convex hull;
- energy-against-volume physics: that a relaxed crystal is an energy minimum, that a solid stiffens under compression, or
  any link between the bulk and shear networks;
- Arrhenius temperature behaviour (corrosion has a temperature input; the electrolyte loader has none, because OBELiX
  holds room-temperature values only);
- monotonicity in concrete age or water for the site trees and the GPU-kit networks, or that the cautious model sits
  below the expected one;
- a smooth neighbour cut (a neighbour just outside the 0.001 Å tolerance is dropped whole), bond angles, spin,
  magnetism, oxidation states or temperature (the labels are 0 K DFT);
- cell-choice invariance of the tree features. Measured on NaCl written as its 2-atom and its 8-atom cell: 6 of 63
  structure features differ (lattice lengths 3.9881 against 5.64 Å, angles 60 against 90°). The composition-plus-cell
  trees use atoms per cell (`gpu/structure/baseline_trees.py` line 49) and the insulator features use the site count
  (`domains/insulators/work/prepare.py` line 142). This was measured in a research session on 2026-11-27 and reproduced by
  a second one; the script is not in the repository.
- a sound description of unseen elements. `gpu/structure/element_table.json` (mendeleev 1.3.0) has 7 properties.
  Re-checked for this document: 18 elements have their electronegativity filled with the table mean (1.7486), among them
  the test element He and the validation elements Tb and Yb; all 30 lanthanides and actinides are coded as group 3; the
  valence counts are inconsistent (Ce 4, Pu 8, Tb 11, Lu 17). There is no f-electron count, ionisation energy, oxidation
  state or magnetic moment.

## 4. The evidence

Crystal results from `docs/results/20261122T014943Z_overnight_REPORT_20261126T2001.md`; intervals are 95% paired
bootstrap over whole groups, network minus strongest trees (negative favours the network). All crystal labels are
DFT-calculated. The held-out-element split is new in cycle 3 (`docs/results/README.md`, "Already-scored test splits"); the
earlier 09:16 report of the same pipeline shows the same runs with the same numbers, so the held-out-element results are
not exploratory.

| Test | Result | Reading |
| --- | --- | --- |
| Formation energy: held-out chemical systems against held-out elements (Pu, Pa, Hf, I, He, Ce, Cr; 10% of crystals), both with the original neighbour rule | Held-out systems: network 0.0515 (0.0466) eV/atom. Held-out elements, learned element vectors: network 0.5511 (0.4796), strongest trees 0.1563, trees on element fractions only 0.4301; +0.323 [+0.307, +0.342] | Trees clearly better on held-out elements, for the average and both runs. The network's error grows about 10 times from new systems to new elements, and it loses even to trees that cannot use an unseen element. |
| The same, other properties (learned element vectors) | Bulk +0.0545 [+0.0491, +0.0601], shear +0.0655 [+0.0592, +0.0725] log10 GPa; band gap +0.107 [+0.0805, +0.133] eV | Trees clearly better on all four, for the average and every run (2 runs). |
| Held-out elements, atoms described by element properties | Formation energy 0.134 against 0.1563, -0.0223 [-0.0272, -0.0173]; band gap -0.00716 [-0.0214, +0.00579]; bulk +0.00576 [+0.00164, +0.00993]; shear +0.00853 [+0.00335, +0.0137] | Network clearly better on formation energy (both runs); no clear difference on band gap; trees clearly better on both moduli (both runs). |
| Band gap on held-out elements, crystals with a gap (76% of test crystals have a gap of exactly 0) | Network 1.408 eV (learned vectors) or 0.9719 eV (element properties) against 0.8752 eV for the trees; no interval. Over all held-out-element test crystals, not only those with a gap, the 90% band-gap intervals covered 0.832 to 0.846 (networks, trees and blends) | On one split, 2 runs; no interval for the gapped subset. The intervals are too narrow for new elements. |
| Held-out chemical systems, fixed neighbour rule (exploratory: the split was scored in cycle 2) | Formation energy -0.0818 [-0.089, -0.0752]; band gap -0.116 [-0.134, -0.0951]; bulk -0.00135 [-0.00619, +0.00335]; shear +0.0137 [+0.00906, +0.0187] | Network clearly better on formation energy and band gap (every run); no clear difference on bulk for the average (trees clearly better in all 3 single runs); trees clearly better on shear (every run). |
| Pre-registered sealed concrete test, 271 results at 56 to 365 days, scored once (DEC-008, LES-006) | Ridge on the nine inputs with ln(age): 9.07 MPa RMSE. Champion of the evolutionary search: 11.50. Champion minus ridge +2.43 [1.08, 3.74] | The test failed. On MONITOR (14 to 28 days) the champion had been ahead (8.30, median of 10 seeds, against 8.63; no interval); the far-future test reversed the order. |
| Table families: one network at a time against tuned trees, 5 runs each | Clearly better in every run on 0 of 9 families, trees clearly better in every run on 1 (electrolytes: trees 1.552, network average 2.139, guessing the average 2.554 log10 S/cm). Surface tension at the CMC: trees R2 0.106, network average R2 -0.0191 (worse than the mean in squared error). Concrete (exploratory): clearly better in 2 of 5 runs, clearly worse in 2 | On no table family does a single network beat tuned trees in every run. |

**Physics probes, measured** (exported tree models; re-run for this document on 2026-11-27 with the research session's
probe scripts, which read only `public/pipeline/corrosion_model.json`, `public/pipeline/electrolyte_model.json`,
`public/model/model.json`, `public/model/model_low.json` and the mix amounts in `lab/data/train.json`; the evaluator
reproduces the exported check predictions to 1.1e-7 and 1.5e-7). Each is one exported model, not repeated on retrained
ones. Concrete numbers are exploratory: the site model was trained on all ages before the sealed rows were split off.

| Check | Result | Verdict and correction |
| --- | --- | --- |
| C1 concrete age: 181 TRAIN mixes, ages 1 to 365 d, site trees | Strength falls with age somewhere in 8 of 181 mixes (4%), by up to 1.13 MPa; at 14 to 28 d (2 mixes), 56 to 90 d (7), 90 to 180 d (2) | Fail. The lab's own check compares f(t) with f(2t), not neighbouring sweep ages. |
| C2 concrete water (Abrams): w/b +0.05 at fixed binder, ages 3, 7, 28, 90 d | Strength rises in 12.3% of 724 pairs, by up to 6.73 MPa (median change -2.64 MPa) | Fail. |
| C3 concrete bounds | 0 negative predictions over the sweep; 0 for every model in the sealed test | Pass. |
| C4 cautious model below expected model | The 10th-percentile model sits above the expected model at 75 of 1,629 sweep points (4.6%), by up to 5.65 MPa | Fail. |
| C5 concrete input consistency | Re-checked here: w/b equals water/(cement + slag + fly ash) within 1e-4 on all 253 TRAIN rows; total mass 2,227 to 2,463.5 kg/m³ (median 2,357.6) | The proposed 1e-6 tolerance fails on 229 of 253 rows only because w/b is stored to 4 decimals (largest gap 4.97e-5); use 1e-4. |
| Same rules on the sealed-test mixes (lab models; age-doubling and w/b pairs) | Share of pairs breaking a rule by more than 0.01 MPa, for the champion and the ten models it was compared with: per-mix age curve 13.2%, gradient-boosted trees 10.1%, random forest 8.2%, PANEL_DESIGN network 1.5%, plain network 0.60%, champion 0.45%, pre-trained plain network 0.41%, random-search champion 0.26%, ridge and both equations 0 (`lab/results/vault_opening.json`) | The networks listed break the rules rarely (0.26 to 1.5% of pairs) whether or not selection penalised violations: the plain, pre-trained plain and PANEL_DESIGN networks are baselines that were never selected (`lab/README.md`, "The baselines exactly"). Ridge and both equations never break them; the two tree models (8.2 and 10.1%) and the per-mix curve (13.2%) often do. |
| E1 electrolytes: remove lithium, renormalise (478 compositions, all with Li) | Prediction lower in 76%, unchanged in 1%, higher in 23%; median Li-free prediction -6.66 log10 S/cm | Fail: at least 95% should fall. The second half of the pass bar (at or below the 5th percentile of the labels, -15.00) cannot be met by an averaging model: 29 of the 478 labels sit exactly at a 1e-15 S/cm floor (re-checked here), so that bar needs a new definition, chosen by a person and labelled post-hoc. The 23% failure stands. |
| E2 electrolytes: Nernst-Einstein ceiling, log10 σ at most 0 (about 0.6 S/cm from n about 1e22 cm⁻³ and D about 1e-5 cm²/s, typical values) | Largest prediction -2.11 on the records and -2.48 on random re-weightings of each record's elements; 0% above 0 | Pass. |
| R1 corrosion temperature: 311 material/environment pairs with at least 5 records, 20 to 100 °C | Non-decreasing in only 28 pairs (9%); 283 dip somewhere, median largest dip 0.012 log10; 17 pairs (5%) dip by more than 0.1 log10, by at most 0.31. Implied activation energy from 25 to 75 °C: median 5.8 kJ/mol (10th to 90th percentile 1.2 to 12.5), negative for 3% | Weak temperature response; partial fail. Aerated open systems can legitimately peak with temperature, so a dip is a flag to check by hand, not proof of an error. |
| R2 corrosion exposure time, 1 h to 10^4 h | Average rate higher after 10^4 h than after 1 h in 0 of 311 pairs | Pass. |
| X1 crystal rewrite invariance | Network: pass under the fixed rule (section 3). Trees: 6 of 63 inputs change with cell choice on one crystal (NaCl, section 3); predictions not run | Network pass. Trees not run; their inputs are not cell-invariant. |

Corrections from the verification of the research, applied above and in sections 3 and 6: the E1 bar on the size of
the fall is degenerate, but the 23% failure stands; the C5 tolerance is 1e-4, not 1e-6; the polymer monomer-swap check
(P1), which the research expected to fail, holds by construction for every row that goes through `prepare.py`; input
clipping covers six tasks, not only alloys; and the element-table problems are worse than first stated (Lu with
valence 17; 18 elements, not one, with a mean-filled electronegativity).

- [Certain] With learned element vectors the crystal networks carry no usable chemistry to elements they have not seen:
  an unseen element keeps its random starting vector (DEC-016), so what they learned is tied to element identities.
- [Likely] Element-property inputs carry some chemistry across (formation energy), but the element table is too crude to
  say how much (section 3).
- [Certain] The exported tree models break simple, well-established trends at measurable rates: concrete age (4% of
  mixes), concrete water (12.3% of pairs), lithium as the carrier (23% of compositions).
- [Certain] In the one pre-registered extrapolation test, the champion network (scored, as registered, by the mean
  prediction of its 10 seeds) was clearly worse than a ridge regression with age entered as ln(age): +2.43 [1.08, 3.74]
  MPa RMSE on 271 sealed results. Each of its seeds alone also scored above the ridge (10.06 to 13.13 MPa; no interval
  for single seeds). The ridge also had the lowest RMSE of all 15 models scored (9.07 MPa; next, the archived
  random-search champion, 9.95), but only the champion was compared with it by an interval.

## 5. The DFT label against reality

The crystal and insulator models are scored against DFT-calculated labels. Against experiment, the label has its own
error. JARVIS-Leaderboard rows: recomputed for this document from the cached `gpu/data_cache/jarvis/jdft_3d-9-24-2025.json.zip`
and the leaderboard's experimental files
([usnistgov/jarvis_leaderboard](https://github.com/usnistgov/jarvis_leaderboard), `benchmarks/ES/SinglePropertyPrediction/`;
NIST public-domain terms; paper [doi:10.1038/s41524-024-01259-w](https://doi.org/10.1038/s41524-024-01259-w)). Model
column: held-out chemical systems, fixed rule, average of runs (report).

| Property | Label against experiment | Materials compared | Model against label | Larger part of the error |
| --- | --- | --- | --- | --- |
| Band gap, OptB88vdW | MAE 1.355 eV, every one an underestimate (the leaderboard lists 1.3552). TBmBJ, also in the file: 0.476 eV (53; the leaderboard lists 0.4981 on 54). HSE: 0.268 eV (6) | 54 | 0.191 eV | The label, about 7 times. [Certain] |
| Band gap, OptB88vdW, Zhuo et al. 2018 measured gaps ([doi:10.1021/acs.jpclett.8b00124](https://doi.org/10.1021/acs.jpclett.8b00124)) | MAE 1.06 eV, mean -0.98 eV; TBmBJ 0.74 eV (506). Matched by composition to the lowest-energy ICSD-linked JARVIS polymorph, which can pick the wrong phase. Recomputed by the research session and its verifier, not re-run for this document | 744 compositions | 0.191 eV | The label. [Certain] |
| Formation energy | JARVIS against experiment: MAE 0.128 eV/atom on 1,317 materials, as Perplexity reports the JARVIS-DFT paper's Table 7 (`research/perplexity/round1/F9.md`) [UNVERIFIED - not opened]. For scale: OQMD, whose elemental energies are fitted to experiment, 0.096 eV/atom on 1,670 compounds, and experiments differ from each other by 0.082 eV/atom ([Kirklin 2015](https://doi.org/10.1038/npjcompumats.2015.10), abstract); databases differ from each other by up to 0.105 eV/atom ([Hegde, arXiv 2007.01988](https://arxiv.org/abs/2007.01988)) | 1,317 (unverified) | 0.0477 eV/atom | The label, about 2 to 3 times. [Likely] |
| Bulk modulus Kv | MAE 5.73 GPa, mean -4.90 GPa, mean relative error 6.4% (for example Si 87.3 against 99.2 GPa). Mostly elements and simple binaries | 21 | 0.0902 log10 GPa, about a factor 1.23 (over the whole test set, not these 21) | The model. [Likely] |
| Shear modulus Gv | No open measured set keyed to JARVIS was found. Averaging alone shifts the label: median Gv/G_VRH 1.072 | 0 | 0.168 log10 GPa | Unknown. |
| Dielectric constant (insulators family, Materials Project DFPT total) | The 16 leaderboard values agree with high-frequency values (12.6% error) but not with static DFPT totals (30.5%, mean +3.2), so they cannot check this target. PBE against experiment: 16.2% mean absolute relative deviation on 88 compounds (Petousis 2016, PRB 93, 115151, abstract; which quantity was not checked). The same crystals in JARVIS and the Materials Project: median ratio 1.06, 27.6% differ by more than 25% (645 materials) | 88 (published) | 0.1178 log10, about a factor 1.31 (grouped CV) | [Speculative] probably the model: 0.118 log10 against about 0.065 to 0.077 log10 for a 16.2% deviation, on different materials and possibly a different quantity. |

- [Certain] For band gap, a perfect copy of the label would still miss measured gaps by about 1.4 eV on these 54
  materials. More rows of the same label buy little against experiment.
- [Likely] For formation energy the label error is 2 to 3 times the model's error, and the experiments themselves
  disagree by more than the model misses the label.
- [Likely] For bulk modulus the model, not the label, is the larger error.

Measured families have a floor of their own. Aqueous solubility: laboratories differ by about 0.5 to 0.6 log units, and
single compounds by up to 1.0 ([Llompart 2024](https://doi.org/10.1038/s41597-024-03105-6)); the project's network
average misses by 0.745 log units MAE on grouped folds (report), so [Likely] above that floor but not far above it.
Concrete cylinder strength: about 5% coefficient of variation between laboratories, from unofficial copies of ASTM C39
(`research/perplexity/ROUND1_DIGEST.md`) [UNVERIFIED - not opened]; the sealed-test errors (9.07 to 11.50 MPa RMSE) are
far above that.

## 6. A proposed physics-consistency suite

A proposal; no code is written. Pass bands that are not hard laws are calibrated on the training data's own measured
series and fixed in code before any run (DEC-019). Typical values marked "cite first" have no source yet. Status:
measured pass, measured fail, not run, or holds by construction.

**Concrete** (site trees, GPU-kit networks, lab networks; query mixes are the distinct TRAIN mixes, amounts only)

| Check | Sweep | Pass condition | Status |
| --- | --- | --- | --- |
| C1 Age | Ages 1, 3, 7, 14, 28, 56, 90, 180, 365 d | No step falls by more than 0.01 MPa | Site trees: measured fail (4% of mixes). Lab pre-trained networks: no age violation in the lab's f(t)-against-f(2t) check on the sealed-test mixes (a different sweep; C1 itself not run on them) |
| C2 Abrams | w/b +0.05 by adding water at fixed binder, ages 3, 7, 28, 90 d | No rise above 0.01 MPa | Site trees: measured fail (12.3%). Others: not run |
| C3 Bounds | C1 sweep | No prediction below 0 | Measured pass |
| C4 Non-crossing | C1 sweep, cautious and expected models | Cautious at or below expected everywhere | Measured fail (4.6%) |
| C5 Inputs | Every scored or generated row | w/b consistent within 1e-4; total mass inside 2,227 to 2,463.5 kg/m³; absolute volume 1 ± 0.07 m³ with cited specific gravities | w/b and mass: pass on TRAIN; generated rows and volume: not run (cite first) |
| C6 Age-curve shape (networks) | Predicted 7 d / 28 d ratio | Inside the band of measured ratios in REWARD (the selection tier; MONITOR also has these ages but is kept for scoring). The lab's fib curve gives 0.78 | Not run |

**Alloys (hardness)**

| Check | Sweep | Pass condition | Status |
| --- | --- | --- | --- |
| A1 Bounds | All sweeps | No HV at or below 0; report the share of predictions on the output clamp and with an input clipped | Not run |
| A2 Descriptors | Every scored or generated row | Fractions sum to 1; recomputed entropy, radius mismatch, VEC, electronegativity and melting-point descriptors agree within 1e-3 | Not run |
| A3 Dilute limit | Add a solute at 0, 0.002, 0.005 | Change at 0.002 below 0.1 × CV MAE (about 7 HV; MAE 75.94 network, 70.9 trees) | Not run |
| A4 Route support | Candidate generation | Same-route neighbours back the prediction (`scripts/alloy_check.mjs`, LES-004) | Exists in the site generator; not a pipeline gate |

**Polymers (Tg)**

| Check | Sweep | Pass condition | Status |
| --- | --- | --- | --- |
| P1 Monomer swap | (a, b, x) against (b, a, 1-x) | Change below 1.5 K | Holds by construction through `prepare.py`; test only inputs that bypass it |
| P2 Homopolymer end | x = 1, a = b | Within the CV MAE (14.56 K network, 16.07 K trees) of the measured homopolymer | Not run |
| P3 Composition bounds | x from 0 to 1 in steps of 0.1, pairs with both homopolymer Tgs known | Inside [min - MAE, max + MAE], monotone within tolerance (Fox equation, no source cited here; Gordon and Taylor 1952, [doi:10.1002/jctb.5010020901](https://doi.org/10.1002/jctb.5010020901)); a warning where the measured data are not monotone | Not run |
| P4 SMILES writing | Random SMILES of the same monomer | Identical prediction | Holds by construction (SMILES canonicalised) |

**Insulators (dielectric constant, DFT-calculated)**

| Check | Sweep | Pass condition | Status |
| --- | --- | --- | --- |
| I1 Floor | Test rows and all sweeps | Predicted log10 ε at least 0 | Not run |
| I2 Cell convention | Double the site count, all else fixed | Change below 0.01 log10 | Not run; expected to fail while the site count is an input |
| I3 Density (warning) | Volume per atom × 0.95 to 1.05, density and packing scaled with it | ε does not rise with volume in at least 90% of crystals (Clausius-Mossotti direction; soft-mode crystals may fail) | Not run |
| I4 Formula scaling | AB2 against A2B4 | Identical | Holds by construction; keep as a regression test |

**Water solubility (AqSolDB LogS)**

| Check | Sweep | Pass condition | Status |
| --- | --- | --- | --- |
| S1 Pure-substance ceiling | Test and generated rows | LogS at most log10(3000 / molar mass), i.e. density at most 3 g/cm³ | Not run |
| S2 Homologous series | n-alkanes, n-alkanols, n-alkanoic acids, alkylbenzenes, C2 to C12 | LogS falls with chain length from C4; slope per CH2 within ±50% of the measured members' slope | Not run; needs an RDKit featuriser that reproduces AqSolDB's precomputed descriptors (`domains/chemistry/work/prepare.py` uses them as given) |
| S3 LogP sign (report only) | MolLogP ±0.5 on real rows | Prediction falls in at least 95% (general solubility equation, [doi:10.1002/1520-6017(200102)90:2<234::aid-jps14>3.0.co;2-v](https://doi.org/10.1002/1520-6017(200102)90:2<234::aid-jps14>3.0.co;2-v)); off the data, so not a gate | Not run |
| S4 SMILES writing | Random SMILES | Identical prediction | Not testable through the current pipeline (same featuriser gap as S2) |

**Solid electrolytes (OBELiX conductivity)**

| Check | Sweep | Pass condition | Status |
| --- | --- | --- | --- |
| E1 Carrier present | Remove Li, renormalise | Prediction falls in at least 95% of compositions; a bar on the size of the fall is for a person to set (section 4) | Measured fail (76%) |
| E2 Ceiling | Records and random re-weightings | log10 σ at most 0 | Measured pass |
| E3 Charge neutrality | Generated compositions | An oxidation-state assignment summing to zero exists | Not run |
| E4 Formula scaling | Li7La3Zr2O12 against Li14La6Zr4O24 | Identical | Holds by construction |
| E5 Arrhenius | Temperature | ln(σT) linear in 1/T, activation energy inside the band of measured series | Not testable: no temperature data |

**Surfactants (SURFPRO)**

| Check | Sweep | Pass condition | Status |
| --- | --- | --- | --- |
| K1 Chain length (Klevens, [doi:10.1007/bf02635002](https://doi.org/10.1007/bf02635002)) | Alkyl sulfates C8 to C16, alkyltrimethylammonium bromides C10 to C16, CnE8 ethoxylates | pCMC rises with chain length; slope per CH2 inside the band of measured SURFPRO series | Not run |
| K2 Surface-tension bounds | All rows | At most the surface tension of water at that temperature (about 72.8 mN/m at 20 °C, cite first); at least the training minimum | Not run |
| K3 Temperature | 10 to 60 °C | Change per 10 °C at most the 99th percentile of measured same-surfactant changes (no universal sign) | Not run |
| K4 SMILES writing | Random SMILES | Identical descriptors and prediction | Not run |

**Corrosion (NIST CORR-DATA)**

| Check | Sweep | Pass condition | Status |
| --- | --- | --- | --- |
| R1 Temperature | 20 to 100 °C, pair's median concentration and hours | No fall above 0.1 log10 between 10 °C steps unless the pair's own measured series falls; implied activation energy inside the band from that pair's data | Measured partial fail (17 of 311 pairs fall by more than 0.1) |
| R2 Exposure time | 1 h to 10^4 h | Average rate does not rise | Measured pass |
| R3 Units | What-if queries | Concentration used only in vol% (the only unit the model reads); others refused or converted | Not run as a gate |
| R4 Range | All queries | Inside the training range of log10 mm/yr, or flagged | Not run |

**Crystals (JARVIS, networks and structure-feature trees)**

| Check | Sweep | Pass condition | Status |
| --- | --- | --- | --- |
| X1 Rewrite invariance | Relist atoms, shift origin, rotate, 2×2×2 supercell (`gpu/structure/invariance.py`); for trees also primitive against conventional cell | Largest change at most 1e-4 of the unit | Networks: measured pass (fixed rule, one seed per property). Trees: 6 of 63 inputs change with cell choice on one crystal (NaCl, section 3); predictions not run |
| X2 Band gap at least 0 | Test predictions | None below -0.05 eV | Not run |
| X3 Elemental reference | Single-element test crystals | Predicted formation energy at least -MAE; check the labels' own values first | Not run |
| X4 Stiffening under compression (Murnaghan, [doi:10.1073/pnas.30.9.244](https://doi.org/10.1073/pnas.30.9.244)) | Lattice × 0.97 to 1.03 | log10 K falls as the lattice grows in at least 90% of crystals; implied K' between 2 and 8 (K' about 4 is typical, cite first) | Not run |
| X5 Energy minimum | Same strains, formation-energy network | E(0.97) and E(1.03) above E(1) in at least 90% of crystals; curvature within a factor 2 of the bulk-modulus network | Not run; expected to fail (no strained crystals in training) |
| X6 Neighbour-cut smoothness | Random 0.002 Å displacements | Largest change at most 0.1 × test MAE | Not run |

**How to wire it into the nightly pipeline.**
1. A `gpu/physics/` module with one function per check. It takes a run's best weights or an exported tree model and
   reads labels only to report scale, as `gpu/structure/invariance.py` does. The research session's probe scripts (not
   in the repository) are the starting point for C1 to C4, E1, E2, R1 and R2.
2. One `physics.json` per run, written new and never overwritten: per check, the sweep size, pass rate, worst case,
   threshold, code version and neighbour rule.
3. A "Physics consistency" section in the report (`gpu/pipeline/report.py`): one row per task and check, with pass rate
   and worst case, next to the error tables.
4. A gate. Hard checks (C3, C4, E2, I1, S1, K2, X1, X2) mark a task's physics status as failed in the registry and the
   report. They never change `gpu/recipes.json` by themselves: that still takes a reviewed commit and a `DECISIONS.md`
   row (DEC-018). Violations are counted, never silently clipped. Soft checks are reported, not gated.
5. A smoke test in `gpu/tests/smoke.py` with untrained networks and synthetic mixes, including one model built to break a
   rule, which must fail it.
6. For the site: extend `scripts/sweep.mjs` to every live slider (no NaN, inside physical bounds, no dead input), as plain
   node scripts; `package.json` has no test runner.

Get Physics Done can supply protocols here but not the checks themselves. Its own check tool says "This tool is
non-authoritative: it does not perform physics verification" (`src/gpd/mcp/servers/verification_server.py` line 2151),
and its ML protocol asks for symmetry tests under random rotations with the spread reported as a quality metric
(`src/gpd/specs/references/protocols/machine-learning-physics.md` lines 85 and 405). Its "~60% of all physics errors"
coverage figure (`src/gpd/specs/references/verification/core/verification-core.md` line 16) has no source. (get-physics-done v1.2.2,
https://github.com/psi-oss/get-physics-done, commit 0f417695, Copyright 2026 Physical Superintelligence PBC, Apache-2.0.)

**Cost.** No GPU time and no money. The three tree-model probes ran in 11 s on a CPU here. Network invariance is
estimated at 3 to 6 CPU minutes per crystal task (`est_minutes` in `gpu/pipeline/plan.py`). [Speculative] The strain
sweeps add a few CPU minutes per network, and writing the module and report section takes 1 to 2 days, half a day for
the checks already prototyped.

## 7. What would move the models up, ranked

None of these steps turns a regressor into an agent of understanding. [Certain] They make the models obey more laws,
come closer to experiment, or be measured more honestly. Ranked by what each step buys for its cost, cheapest sure gain
first.

| Rank | Step | Expected effect | Label |
| --- | --- | --- | --- |
| 1 | Build the suite in section 6 and run it every night | Turns "does it obey law X" from an assertion into a number per task; finds the network failures that no one has measured (X2 to X6, P2, P3, S1 to S3, K1 to K3) | [Certain] for the measurement; [Likely] it finds new failures in X5 and X2 |
| 2 | Fix the cheap hard-constraint defects: monotonic constraints on concrete trees (age +1, water and w/b -1; `monotonic_cst` exists in scikit-learn 1.9.1 for HistGradientBoosting, ExtraTrees and RandomForest, not for the GradientBoosting model the site uses); cautious model capped at the expected one; a band-gap head that cannot go below 0 (softplus, or metal/non-metal classifier then regression); a canonical cell for tree features or drop lattice and site-count inputs | C1, C2, C4, X2, X1 (trees) and I2 failures go to zero by construction. Accuracy inside the data: [Likely] no clear difference (Sadeghi 2026 found none for physics features), to be judged against the unconstrained models under DEC-004 | [Certain] for the violations; [Likely] for accuracy |
| 3 | Train band gap on the 21,559 TBmBJ gaps already in the cached file, keeping the 54 leaderboard materials out of training and validation | Error against measured gaps falls below the OptB88vdW label's 1.355 eV. On the same materials, the error against experiment is at most the model's error against TBmBJ plus 0.476 eV, so it stays below 1.0 eV unless the model misses TBmBJ there by more than 0.52 eV. Today it misses OptB88vdW by 0.19 to 0.20 eV on average, trained on 74,920 crystals against 21,559 TBmBJ gaps in the whole file | [Likely]; probability about 0.8 of going below 1.0 eV on the leaderboard's measured gaps |
| 4 | Physics-shaped extrapolation: a residual on a strength-age law for concrete (Qu 2026 design), an Arrhenius form for corrosion temperature, and for electrolytes once temperature data exist (Bradford 2023, [doi:10.1021/acscentsci.2c01123](https://doi.org/10.1021/acscentsci.2c01123)) | Better extrapolation along age and temperature (zirconia 0.4187 to 0.3609; the sealed concrete ranking); little change on new compositions | [Likely] |
| 5 | Repair the element table (no mean-filled electronegativity, real f-block groups, consistent valence) and add f-electron count, ionisation energy and oxidation states | Narrows the held-out-element gap on the moduli | [Speculative] |
| 6 | Measure the label against experiment directly for formation energy (Kim 2017, 1,276 values with MP and OQMD ids, [doi:10.1038/sdata.2017.162](https://doi.org/10.1038/sdata.2017.162)) | Replaces the unverified 0.128 eV/atom with a measured label error | [Certain] for the measurement |
| 7 | Pretrained universal-potential embeddings as crystal inputs (Witman 2026: element-wise CV MAE 0.38 eV against 0.72 eV for a graph network from scratch on defect energies, [doi:10.26434/chemrxiv.15003938/v2](https://doi.org/10.26434/chemrxiv.15003938/v2); the DOI may not resolve, OpenAlex W7162515451 opens) | Better transfer to new elements | [Speculative] on this project's splits; needs pretrained potentials the project does not use (DEC-021) |
| 8 | Electrolytes: refuse Li-free candidates in generation | Removes the E1 failure from the shortlist, not from the model | [Certain] |
| 9 | More rows of the same kind | Small gains against experiment for band gap and formation energy, where the label error dominates; for solubility, cleaner labels rather than more rows | [Likely] |
| 10 | A fresh sealed set of new measurements for one family | The only evidence of discovery; the concrete vault is spent (LES-006) | [Certain] that it is needed; costs laboratory time |

Probabilities are this document's judgement, not measurements.

## 8. The strongest counterargument

"This is too harsh. On held-out chemical systems the crystal networks beat structure-aware trees on formation energy and
band gap in every run, and their formation-energy error (0.048 eV/atom) is smaller than the label's own distance from
experiment. That is exactly the job of a surrogate. The element-property network did carry formation energy to unseen
elements (0.134 against 0.1563 eV/atom). The concrete age drops (at most 1.13 MPa) are inside the scatter of a cylinder
test. And Vafa's result is about transformers on sequences, not regressors."

Parts of it are right. The assessment does not say the models are poor: inside their data they are accurate, and for
formation energy the label is the larger error. It says they are curve fitters, which is what the user's question was
about. The age drops are small, but the water violations (up to 6.73 MPa), the cautious-model crossings (up to 5.65 MPa)
and the lithium result (23%) are not. The element-property result is the best sign of transferable chemistry in the
project, and it holds for one property of four. Vafa's setting is different, but its point carries over: the lab's plain
concrete network obeyed the age and water rules on 99.4% of query pairs and still missed the sealed test by 113.6 MPa.

Evidence that would change this assessment:
- X4 and X5 pass (stiffening under compression, relaxed crystals at an energy minimum) for networks never trained on
  strained crystals: that would show energy-volume physics learned from equilibrium data alone.
- Element-property networks beat the strongest trees on all four properties on held-out elements, with every run.
- An inductive-bias probe in the style of Vafa et al.: fine-tune on synthetic Abrams-law or Arrhenius data, and the
  extrapolations follow the law.
- The physics suite passes at 99% or more on networks and trees, including sweeps outside the training range.
- A model ranks new, sealed measurements correctly.

Evidence that would make it worse: a share of negative band-gap predictions, X5 failing broadly, or element-property
gains vanishing on another set of held-out elements.

## What this document did not check

- No network check was run: trained weights and test predictions stay on the laptop (no `gpu/runs/` in this checkout).
  Every network status in section 6 is "not run" or comes from the report.
- `npm run sweep` was not run (it reads two blocked files).
- The Zhuo comparison, the NaCl cell test and the JARVIS label counts were computed by research sessions and their
  verifiers, not re-run here. The JARVIS-Leaderboard numbers, the probes, the element table, the electrolyte label floor
  and the concrete input checks were re-run here. The sealed-test RMSEs and violation rates were read from
  `lab/results/vault_opening.json`, not recomputed.
- Read from abstracts only: the zirconia, Thapa, Qu, Bradford, Kirklin, Petousis, Witman, MD-HIT and Bartel papers, and
  arXiv 2610.06877; Muckley 2023 from one line. The Matbench Discovery leaderboard figures come from a preprint snapshot.
- The JARVIS formation-energy comparison with experiment (0.128 eV/atom, 1,317 materials) and the concrete C39 scatter
  are unverified leads.
