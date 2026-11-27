# src/components/sim/: the pipeline pages

Pages for the families trained in the GPU pipeline, and the pipeline sections added to the older family pages. Every simulation
should be **interactive the way the concrete designer is**: the visitor says what they need, the page proposes or finds candidates,
predicts them with a cautious estimate, ranks them, and shows the measured neighbours. Every number comes from `public/pipeline/`
(written by `python -m gpu.export_site`; see `gpu/export_site.py` for how each value is computed). Nothing is typed by hand.

Read first: `taste.md` (visual rules: flat, light, one accent `var(--accent)`, 1px borders, 4px corners, no shadows/gradients/pills,
colours only through CSS variables so dark mode works), `docs/AGENTS.md` (ground rules and the wording table), and `src/App.jsx`
(the concrete designer: the interaction to match).

## Wording rules (docs/AGENTS.md)

- A prediction is labelled a prediction; a measured value is labelled measured; DFT values are "DFT-calculated", never "measured".
- "X clearly better" only when the export says so (`better` set). Otherwise "no clear difference" **with its interval**.
- An average of several networks is called an average; say "one network at a time" results where the export has them.
- Anything the export marks `exploratory` says so next to the number.
- Every page keeps a "How this can be wrong" box (SimPage provides it).

## Files and fields

`loadPipeline(name)` in `src/lib/pipeline.js` fetches `public/pipeline/<name>` (null when absent: show a plain note, never crash).
`predictTrees(model, x)` walks the exported gradient-boosted trees. `fmt`, `signed`, `interval`, `comparisonText`, `factor`,
`MODELS`, `matchRows` are shared formatting helpers.

- `index.json`: `{generated, rehearsal, tasks: [...]}`. A task: `id` ("tab:corrosion", "crystal:bulk_modulus_kv"), `kind`
  ("table" | "crystal"), `name`, `title`, `site` (menu id: materials, polymers, insulators, chemistry, electrolytes, cleaning,
  corrosives, concrete, crystals), `trained` (date), `recommended`, `exploratory` (text or null), `status` ("ok" | "missing" |
  "error", with `reason`), `unit`, `rows`, `groups`, `mode`, `mae: {network, trees, blend, mean}`, `network_vs_trees`,
  `blend_vs_trees`, `blend_vs_network` (each `{diff, ci95:[lo,hi], first, second, better, verdict}`), `single_network:
  {n_runs, clearly_better, clearly_worse, mean_mae}`, `interval90: {network|trees|blend: {half_width, coverage}}`, `rows_file`;
  crystals also `n_runs`, `neighbor_rule`, `strongest_trees`. If `rehearsal` is true, show a visible "rehearsal: numbers mean
  nothing" flag.
- `tab_<family>.json`: `{task, unit, kind:"table", split, rows_total, rows_shown, sample_rule, interval90, rows:[{label:{...},
  group, y, network, trees, blend}]}`. The rows are out of fold (each predicted by models that never trained on it). Label fields
  per family: corrosion `{material, material_class, material_family, environment, conc_vol_pct, temp_C, hours, reference, ...}`;
  cleaning `{smiles, type, temp_C}`; electrolytes `{composition, id}`; chemistry `{name, id}`; alloys/polymers/insulators
  `{group}`.
- `crystal_<target>.json`: same, rows `{jid, formula, system, y, network, trees, blend}` (held-out-systems test crystals).
- `structures.json`: `{cells: {jid: {formula, L: 3x3 lattice in angstrom (rows are vectors), X: fractional coords, E: elements}}}`.
- `corrosion_model.json`: `{features, numeric_medians, init, trees, cv: {mae, interval90_half_width, mean_baseline_mae, folds},
  materials: [{name, class, family, rows}], environments: [{name, rows}], checks: [{x_nonzero, pred}], records: [{material,
  environment, conc_vol_pct, temp_C, hours, reference, y}], unit, source}`. Target is log10(mm/yr).
- `electrolyte_model.json`: `{elements, init, trees, cv: {...}, checks: [{fractions, pred}], records: [{composition, id, y, oof,
  fractions}], unit, source}`. Target is log10(S/cm).
- `surfactants.json`: `{properties: {pcmc|tension: {unit, rows_measured, cv_mae, interval90_half_width, mean_baseline_mae, folds}},
  surfactants: [{smiles, type, temp_C, pcmc: {measured, predicted, out_of_fold}, tension: {...}}], source}`.

## Rebuilding inputs in the browser

- Corrosion (`corrosion_model.json`): vector of `features.length` zeros. For each category in material_class, material_family,
  material, environment, conc_qualifier, temp_qualifier: value = the chosen text (empty -> "none"); set 1 at
  `"<category>=<value>"` if that is a feature, else at `"<category>=(infrequent)"` if that is one. A material's class and family
  come from `materials`. A concentration/temperature/hours the visitor gives sets `conc_qualifier` / `temp_qualifier` to "exact";
  numbers: `conc_vol_pct` = value, `temp_C_mid` = value, `log10_hours` = log10(max(hours, 0.1)); a blank number = its
  `numeric_medians` value and 1 at `"<name>__missing"` if that is a feature. Verified in Python on all 5761 rows. A unit test must
  check `predictTrees` against every `checks` entry (|difference| < 1e-6).
- Electrolytes (`electrolyte_model.json`): vector over `elements` of atomic fractions summing to 1. Test against `checks` too.

## Components (one owner each)

- `SimDashboard.jsx`: page "All simulations: nightly results". One row per task in `index.json`: title, family (link via the
  `onOpen(site)` prop to that page), recommended model, MAE of network / trees / blend with the unit, the 90% interval of the
  recommended model (half-width and measured coverage), the network-vs-trees verdict in the project's wording, trained date,
  exploratory flag. Missing/error tasks listed with their reason. A short "how to read this" note.
- `PipelineResults.jsx` (`{task}`): for one task: the three models' MAE and their 95% comparisons (`comparisonText`), the 90%
  intervals and coverage, "one network at a time" counts; a parity plot (SVG, truth vs prediction, model toggle, the diagonal and
  the ±90% band, points clickable) of the rows file; a search box over labels (`matchRows`) and a small table of the matching rows
  with truth and all three predictions. Show `rows_shown` of `rows_total` and the sample rule.
- `CorrosionDesigner.jsx`: the corrosion designer. Inputs: environment (searchable select from `environments`, most rows first),
  concentration vol% (optional), temperature °C (optional), exposure hours (optional), maximum acceptable rate (mm/yr, presets
  like 0.1 / 0.5 / 1.25), optional material class filter. Output: every material in `materials` predicted in that environment,
  ranked by the cautious upper estimate (10^(pred + half-width), mm/yr), with prediction, upper estimate, whether it meets the
  limit, and the measured neighbours: the `records` for that material in that environment (count, measured rates, temperatures),
  flagged "measured neighbours agree" only when the measured rates also meet the limit. Say plainly when a material has no
  record in that environment (pure extrapolation). Show the model's CV error and that it is a compact model separate from the
  pipeline's tuned trees.
- `ElectrolyteDesigner.jsx`: target conductivity (S/cm, presets 1e-4, 1e-3, 1e-2). Candidates: the top measured compositions
  (by measured y) and new ones made by nudging them (move 2-5% of the fraction between two elements of the same kind, an anion-forming non-metal with another or any other element with any other (H is never moved), and say that a nudge does not keep the formula charge-balanced, or swap a minor
  element for one seen in the data), renormalised; predict with `predictTrees`; rank by the cautious lower estimate (pred -
  half-width); show the nearest measured compositions (L1 distance over fractions) and their measured values; label new
  compositions "untested prediction". The interval is about ±4 log units: say in plain words that the model can show a direction
  but cannot rank close candidates.
- `SurfactantFinder.jsx`: requirements: pCMC at least x (higher = lower CMC) and/or surface tension at most y mN/m, optional type
  filter. List surfactants meeting both, using the measured value where it exists and the prediction otherwise (marked
  "predicted"), with cautious bounds (prediction ± half-width); show SMILES (monospace, wrapped), type, measured vs predicted per
  property; sort by how far inside the requirements. Also show, for measured rows, how close the out-of-fold prediction was
  (the model's honesty).
- `CrystalFinder.jsx` (+ a `CrystalViewer.jsx` you create): requirements on the four DFT properties (ranges; bulk/shear moduli
  are log10 GPa in the data: let the visitor type GPa), choose the model (network / trees / blend) and whether to require the
  cautious edge; search a formula; list held-out crystals meeting the requirements from the four `crystal_*.json` files (a
  crystal may lack a property); selecting one shows its unit cell in 3D (three.js is installed; lazy-load it; atoms as spheres
  coloured by element with a legend, the cell edges, drag to rotate; respects prefers-reduced-motion) and a table of its DFT
  values against the three predictions with intervals. DFT values are labelled DFT-calculated.
- `PropertyFinder.jsx` (`{family, index, tasks}`): for alloys (materials), insulators, polymers and chemistry: a target (at least /
  at most a value in the task's unit), the model (default: the task's `recommended` if it is one of network/trees/blend, else
  blend), and "cautious" (use the interval edge). Lists rows of the task's rows file that are predicted to meet it, beside their
  measured value (they are held-out rows, predicted by models that never trained on them), and how many of the predicted hits
  really meet it (precision) — the honest test of the finder.
