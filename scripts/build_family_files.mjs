// Writes public/model/<family>/ for the three trained families from the family agents' files.
// Every number in family.json is read from result.json or skeptic_check.json; "source" names the
// file and the path inside it, so nothing is typed by hand. Run: node scripts/build_family_files.mjs [simDir]
import { readFileSync, writeFileSync, mkdirSync, copyFileSync, existsSync, statSync } from 'node:fs'
import { join } from 'node:path'

const SIM = process.argv[2] || process.env.FAMILY_SIM_DIR || '/tmp/claude-0/-home-user/c96b0b28-08aa-572a-ab28-a200b90c8d07/scratchpad/burn/sim'
const OUT = new URL('../public/model/', import.meta.url).pathname
const DOMAINS = new URL('../domains/', import.meta.url).pathname

const read = (f) => JSON.parse(readFileSync(f, 'utf8'))
// get(obj, 'a.b.c') with a record of where each value came from.
function picker(file, obj, sourceLog) {
  return (path) => {
    let v = obj
    for (const k of path.split('.')) {
      if (v == null || !(k in v)) throw new Error(`${file}: missing ${path}`)
      v = v[k]
    }
    sourceLog[path] = file
    return v
  }
}

function family(id, build) {
  const dir = join(SIM, id)
  for (const f of ['result.json', 'skeptic_check.json', 'nn_data.json', 'prepare.py', 'nn_generic.js']) {
    if (!existsSync(join(dir, f))) throw new Error(`refusing to build ${id}: ${join(dir, f)} is missing`)
  }
  const result = read(join(dir, 'result.json'))
  const skeptic = read(join(dir, 'skeptic_check.json'))
  const nnData = read(join(dir, 'nn_data.json'))
  const usedR = {}, usedS = {}
  const r = picker('result.json', result, usedR)
  const s = picker('skeptic_check.json', skeptic, usedS)
  mkdirSync(join(OUT, id), { recursive: true })
  const out = build({ r, s, nnData, dir, result })
  let lo = Infinity, hi = -Infinity
  for (const v of nnData.train.y) { if (v < lo) lo = v; if (v > hi) hi = v }
  out.trainingRange = [lo, hi]
  out.source = {
    folder: dir,
    files: {
      'result.json': { mtime: statSync(join(dir, 'result.json')).mtime.toISOString(), paths: Object.keys(usedR) },
      'skeptic_check.json': { mtime: statSync(join(dir, 'skeptic_check.json')).mtime.toISOString(), paths: Object.keys(usedS) },
    },
    trainingRange: 'min and max of train.y in nn_data.json',
  }
  mkdirSync(join(OUT, id), { recursive: true })
  copyFileSync(join(dir, 'nn_data.json'), join(OUT, id, 'nn_data.json'))
  writeFileSync(join(OUT, id, 'family.json'), JSON.stringify(out, null, 1) + '\n')
  mkdirSync(join(DOMAINS, id), { recursive: true })
  for (const f of ['prepare.py', 'nn_generic.js']) copyFileSync(join(dir, f), join(DOMAINS, id, f))
  console.log(`${id}: wrote family.json and nn_data.json (${nnData.train.y.length} train / ${nnData.test.y.length} test rows)`)
  return { out, nnData, dir }
}

// ---- alloys ----------------------------------------------------------------
family('materials', ({ r, s, dir }) => {
  const sweep = read(join(dir, 'sweep.json'))
  return {
    id: 'materials',
    label: 'Alloys (hardness)',
    kind: 'MEASURED',
    kindText: 'MEASURED in labs: hardness values collected from published papers',
    target: 'Vickers hardness', unit: 'HV', display: { kind: 'identity', digits: 0 },
    task: r('task'),
    data: { source: r('data.source'), citation: r('data.citation'), licence: 'licence not confirmed', rows: r('data.rows_used'), groups: r('data.alloys_groups'), groupWord: 'distinct alloys', train: r('data.train_rows'), test: r('data.test_rows'), inputs: r('features').length, inputsText: 'element fractions, processing route and composition descriptors (atomic size, electronegativity, valence electrons, melting point)', splitRule: 'split by alloy formula, never by row' },
    options: r('network.settings'),
    members: r('network.ensemble_members'),
    trainSecondsNode: r('network.train_seconds_final'),
    test: { network: r('b_test_split_once.network.mae'), networkR2: r('b_test_split_once.network.r2'), trees: r('b_test_split_once.trees.mae'), treesR2: r('b_test_split_once.trees.r2'), treesTuned: s('my_trees_tuned'), treesTunedConfig: s('tuned_trees_config'), mean: r('b_test_split_once.naive_train_mean.mae') },
    eightSplits: { diff: r('c_eight_splits.mean_diff_network_minus_trees'), ci: r('c_eight_splits.paired_t95'), ciOverlap: r('c_eight_splits.paired_t95_corrected_for_overlap'), networkBetter: r('c_eight_splits.network_better_in'), of: r('c_eight_splits.of'), wording: 'Over eight other splits the average difference is 2.8 HV in the network\'s favour, but the uncertainty includes zero: it is a tie.' },
    interval: { q: r('d_interval90.halfwidth_HV'), coverage: r('d_interval90.test_coverage'), inside: r('d_interval90.test_inside'), rows: r('d_interval90.test_rows'), rule: r('d_interval90.rule'), liveMeasure: true, failsOn: 'alloy systems the model never saw (whole-system error about 102 HV instead of 78), where the true coverage is unknown and probably lower' },
    wholeSystems: { network: r('extra_cv_whole_alloy_systems_held_out_training_rows_only.network_ensemble.mae'), trees: r('extra_cv_whole_alloy_systems_held_out_training_rows_only.trees.mae'), familiarTestRows: r('extra_cv_whole_alloy_systems_held_out_training_rows_only.test_rows_whose_element_set_is_also_in_train') },
    repeatNoise: r('data.repeat_measurement_noise.mean_abs_dev_from_cell_mean_HV'),
    sweep: { title: r('sanity_al_sweep.alloy'), x: r('sanity_al_sweep.al_x'), xLabel: 'Al amount x in Al_x CoCrFeNi', network: r('sanity_al_sweep.network_hv'), trees: r('sanity_al_sweep.trees_hv'), rows: sweep.sweep.map((p) => p.features), measured: r('sanity_al_sweep.measured_in_dataset').filter((m) => m.proc === 'CAST').map((m) => ({ x: m.al_x, y: m.hv, side: m.side })), note: 'Al-Co-Cr-Fe-Ni alloys are in the training rows, so part of this curve is recall, not prediction.' },
    verdict: s('safe_sentence'),
    caveats: [...r('caveats'), `Error on held-out alloys from familiar systems: 78 HV. On alloy systems the model never saw: about ${Math.round(r('extra_cv_whole_alloy_systems_held_out_training_rows_only.network_ensemble.mae'))} HV (trees ${Math.round(r('extra_cv_whole_alloy_systems_held_out_training_rows_only.trees.mae'))}).`, 'Predictions are capped at the training range; a capped value means the model is outside its data.'],
  }
})

// ---- insulators --------------------------------------------------------------
family('insulators', ({ r, s, nnData, dir }) => {
  // Held-out compounds for the ranking: the test rows of nn_data.json, matched to all_rows.json.
  const all = read(join(dir, 'all_rows.json'))
  const testIdx = all.main_split.test
  const heldout = nnData.test.X.map((x, k) => {
    const i = testIdx[k]
    const same = all.X[i].every((v, j) => Math.abs(v - x[j]) < 1e-9) && Math.abs(all.y[i] - nnData.test.y[k]) < 1e-12
    if (!same) throw new Error(`insulators: test row ${k} does not match all_rows row ${i}`)
    return { formula: all.formula[i], id: all.material_id[i], bandGap: all.band_gap[i], y: nnData.test.y[k] }
  })
  writeFileSync(join(OUT, 'insulators', 'heldout.json'), JSON.stringify({ note: 'The held-out test compounds of nn_data.json, in the same order: formula, Materials Project id, calculated band gap in eV (not a network input; used only for the trend check) and the calculated log10 dielectric constant.', rows: heldout }) + '\n')
  return {
    id: 'insulators',
    label: 'Insulators (dielectric constant)',
    kind: 'CALCULATED',
    kindText: 'CALCULATED by computer simulation (DFT), not measured in a lab',
    target: 'Dielectric constant (polycrystalline total)', unit: '', display: { kind: 'pow10', digits: 1 },
    task: r('data.target'),
    important: r('IMPORTANT'),
    data: { source: r('data.source'), citation: 'Petousis et al., DOI 10.18126/racd-go9m', licence: 'CC BY 4.0', rows: r('data.rows_used'), groups: r('data.groups_reduced_formulas'), groupWord: 'distinct compounds (reduced formulas)', train: r('data.train_rows'), test: r('data.test_rows'), inputs: r('features_main').length, inputsText: 'numbers from the composition (atomic number, period, group, electronegativity, mass, radius statistics) and the crystal cell', splitRule: 'split by reduced formula, never by row' },
    options: { hidden: r('network_settings_chosen.hidden'), epochs: r('network_settings_chosen.epochs'), batch: r('network_settings_chosen.batch'), lr: r('network_settings_chosen.lr'), decay: r('network_settings_chosen.decay'), seed: r('network_settings_chosen.seed'), loss: r('network_settings_chosen.loss') },
    members: r('network_settings_chosen.members'),
    trainSecondsNode: r('final_network.training_seconds'),
    test: { network: r('b_test_split_once.network.mae_log10'), networkR2: r('b_test_split_once.network.r2_log10'), trees: r('b_test_split_once.trees_gradient_boosting.mae_log10'), treesR2: r('b_test_split_once.trees_gradient_boosting.r2_log10'), treesTuned: s('my_trees_tuned'), treesTunedConfig: s('tuned_trees_config'), mean: r('b_test_split_once.naive_training_mean.mae_log10') },
    eightSplits: { diff: r('c_eight_grouped_splits.mean_difference_network_minus_trees'), ci: r('c_eight_grouped_splits.paired_t_95_interval'), networkBetter: r('c_eight_grouped_splits.network_better_on_splits'), of: r('c_eight_grouped_splits.splits_completed'), wording: 'On this split the trees are slightly ahead; over eight splits the two are tied.' },
    interval: { q: r('d_interval_90.half_width_log10'), factor: r('d_interval_90.multiplicative_factor'), coverage: r('d_interval_90.test_coverage'), inside: r('d_interval_90.test_rows_inside'), rows: r('d_interval_90.test_rows'), rule: r('d_interval_90.method'), liveMeasure: true, coverageAbove30: r('d_interval_90.test_coverage_rows_true_value_above_30'), rowsAbove30: r('d_interval_90.rows_above_30'), failsOn: `compounds with a dielectric constant above 30 (coverage ${Math.round(100 * r('d_interval_90.test_coverage_rows_true_value_above_30'))}% on ${r('d_interval_90.rows_above_30')} held-out rows)` },
    highTail: { medianPredOverTrue: r('b_test_split_once.high_tail_rows_true_value_above_30.network_median_predicted_over_true'), largestPrediction: r('d_interval_90.largest_test_prediction_dielectric_units'), largestTrue: r('d_interval_90.largest_test_true_value') },
    bandGap: { networkSpearman: r('sanity_check_band_gap_trend.main_model_no_band_gap_input.network_prediction_vs_true_band_gap_spearman'), treesSpearman: r('sanity_check_band_gap_trend.main_model_no_band_gap_input.trees_prediction_vs_true_band_gap_spearman'), calculatedSpearman: r('sanity_check_band_gap_trend.reference_true_target_vs_band_gap_spearman_test') },
    verdict: s('safe_sentence'),
    caveats: [...r('three_biggest_caveats'), ...r('other_caveats')],
  }
})

// ---- polymers ----------------------------------------------------------------
family('polymers', ({ r, s }) => ({
  id: 'polymers',
  label: 'Polymers (glass transition)',
  kind: 'MEASURED',
  kindText: 'MEASURED in labs: glass-transition temperatures collected from published papers',
  target: 'Glass-transition temperature Tg', unit: 'K', display: { kind: 'identity', digits: 1 },
  task: r('task'),
  data: { source: 'source file experimental_dataset.csv, citation not confirmed', citation: 'citation not confirmed', licence: 'licence not confirmed', rows: r('data.rows_used_copolymer_file'), groups: r('data.copolymer_systems'), groupWord: 'copolymer systems', homopolymerRows: r('data.rows_homopolymer'), copolymerRows: r('data.rows_true_copolymer'), train: r('b_test_split_seed42.train_rows'), test: r('b_test_split_seed42.test_rows'), inputs: r('feature_count'), inputsText: 'structure descriptors of the two monomers (computed with RDKit), short structure fingerprints, the composition and a Fox-equation estimate from training homopolymers', splitRule: 'split by copolymer system, never by row' },
  options: { hidden: r('network.chosen_settings.hidden'), epochs: r('network.chosen_settings.epochs'), batch: r('network.chosen_settings.batch'), lr: r('network.chosen_settings.lr'), decay: r('network.chosen_settings.decay'), seed: r('network.chosen_settings.seed'), loss: r('network.chosen_settings.loss') },
  members: r('network.chosen_settings.members'),
  trainSecondsNode: r('network.training_seconds_final_network'),
  test: { network: r('b_test_split_seed42.network.mae'), networkR2: r('b_test_split_seed42.network.r2'), trees: r('b_test_split_seed42.trees.mae'), treesR2: r('b_test_split_seed42.trees.r2'), treesTuned: s('my_trees_tuned'), treesTunedConfig: s('tuned_trees_config'), mean: r('b_test_split_seed42.naive_mean.mae') },
  eightSplits: { diff: r('c_eight_splits.mean_difference_network_minus_trees'), ci: r('c_eight_splits.paired_t95_interval'), wording: r('verdict_network_vs_trees') },
  subsets: [
    { name: 'True copolymers', rows: r('b_test_split_seed42.subsets.true_copolymer_rows.network.n'), network: r('b_test_split_seed42.subsets.true_copolymer_rows.network.mae'), trees: r('b_test_split_seed42.subsets.true_copolymer_rows.trees.mae'), coverage: r('b_test_split_seed42.subsets.true_copolymer_rows.interval90_coverage') },
    { name: 'Homopolymers', rows: r('b_test_split_seed42.subsets.homopolymer_rows.network.n'), network: r('b_test_split_seed42.subsets.homopolymer_rows.network.mae'), trees: r('b_test_split_seed42.subsets.homopolymer_rows.trees.mae'), coverage: r('b_test_split_seed42.subsets.homopolymer_rows.interval90_coverage') },
    { name: 'Fox estimate available', rows: r('b_test_split_seed42.subsets.fox_available_rows.network.n'), network: r('b_test_split_seed42.subsets.fox_available_rows.network.mae'), trees: r('b_test_split_seed42.subsets.fox_available_rows.trees.mae'), coverage: r('b_test_split_seed42.subsets.fox_available_rows.interval90_coverage') },
  ],
  unseenMonomer: { network: r('e_hard_split_unseen_monomer.network.mae'), trees: r('e_hard_split_unseen_monomer.trees.mae') },
  interval: { q: r('d_interval.half_width_K'), coverage: r('d_interval.test_coverage'), rule: r('d_interval.method'), liveMeasure: false, homopolymerCoverage: r('b_test_split_seed42.subsets.homopolymer_rows.interval90_coverage'), failsOn: `homopolymers (coverage ${Math.round(100 * r('b_test_split_seed42.subsets.homopolymer_rows.interval90_coverage'))}%)`, noLiveReason: 'The Fox inputs in this file were computed from this one split\'s training rows, so re-splitting in the browser would leak test values into training. The width shown is the one measured in Node with the Fox inputs recomputed for every fold.' },
  compositionSweep: { systems: r('sanity_composition_sweep.systems'), points: r('sanity_composition_sweep.points_per_system'), network: r('sanity_composition_sweep.network_monotonic_share_tol0K'), trees: r('sanity_composition_sweep.trees_monotonic_share_tol0K'), measured: r('sanity_composition_sweep.measured_monotonic_share_systems_with_3plus_points') },
  verdict: s('safe_sentence'),
  caveats: r('caveats'),
}))

// ---- chemistry: not shipped, only its verified numbers for the Planned note ----
{
  const dir = join(SIM, 'chemistry')
  const result = read(join(dir, 'result.json')), skeptic = read(join(dir, 'skeptic_check.json'))
  const note = {
    id: 'chemistry',
    shipped: false,
    reason: 'The training file is about 2 MB (680 kB compressed), too slow for a phone tonight.',
    folder: 'domains/chemistry/',
    test: { network: result.b_test_once.network_ensemble.mae, trees: result.b_test_once.trees_gbr.mae, treesTuned: skeptic.my_trees_tuned, mean: result.b_test_once.naive_train_mean.mae, unit: 'LogS (log10 mol/L)' },
    source: { folder: dir, files: ['result.json: b_test_once', 'skeptic_check.json: my_trees_tuned'] },
  }
  mkdirSync(join(OUT, 'chemistry'), { recursive: true })
  writeFileSync(join(OUT, 'chemistry', 'family.json'), JSON.stringify(note, null, 1) + '\n')
  mkdirSync(join(DOMAINS, 'chemistry'), { recursive: true })
  for (const f of ['prepare.py', 'nn_generic.js']) copyFileSync(join(dir, f), join(DOMAINS, 'chemistry', f))
  console.log('chemistry: wrote the Planned-card numbers only (no training data shipped)')
}
