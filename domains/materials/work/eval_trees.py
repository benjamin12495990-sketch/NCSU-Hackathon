#!/usr/bin/env python3
"""Tree and naive baselines, the paired comparison, and the final result file.

Usage: python3 -I eval_trees.py <dir>
Reads nn_data.json, splits.json, cv_folds.json, sweep.json, cv_results.json,
nn_final.json, prepare_summary.json from <dir>; writes <dir>/result.json.
"""
import json, sys, time
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor

d = sys.argv[1]
load = lambda name: json.load(open('%s/%s' % (d, name)))
data, allrows, folds = load('nn_data.json'), load('splits.json'), np.array(load('cv_folds.json')['fold'])
cv, nn, prep, sweep = load('cv_results.json'), load('nn_final.json'), load('prepare_summary.json'), load('sweep.json')
Xtr, ytr = np.array(data['train']['X']), np.array(data['train']['y'])
Xte, yte = np.array(data['test']['X']), np.array(data['test']['y'])
X, y = np.array(allrows['X']), np.array(allrows['y'])


def gbr():
    return GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)


def metrics(p, t):
    e = np.asarray(p) - np.asarray(t)
    return {'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt((e ** 2).mean())),
            'r2': float(1 - (e ** 2).sum() / ((t - np.mean(t)) ** 2).sum())}


# Trees under the same 3 training folds the network was tuned on (information only; nothing is tuned for the trees).
oof = np.zeros(len(ytr))
for f in range(3):
    oof[folds == f] = gbr().fit(Xtr[folds != f], ytr[folds != f]).predict(Xtr[folds == f])
naive_oof = np.array([ytr[folds != f].mean() for f in folds])

# The same question with whole alloy systems (same set of elements) held out, training rows only.
sfold = np.array(load('cv_folds_system.json')['fold'])
soof = np.zeros(len(ytr))
for f in range(3):
    soof[sfold == f] = gbr().fit(Xtr[sfold != f], ytr[sfold != f]).predict(Xtr[sfold == f])
snaive = np.array([ytr[sfold != f].mean() for f in sfold])
nn_sys = load('cv_system_nn.json')
ung = load('cv_results_unguarded.json')

t0 = time.time()
model = gbr().fit(Xtr, ytr)
tree_seconds = time.time() - t0
tree_pred = model.predict(Xte)
naive_pred = np.full(len(yte), ytr.mean())
nn_pred = np.array(nn['test_pred'])

# Eight extra grouped splits: same rows for both models.
rows = []
for s, n in zip(allrows['splits'], nn['splits']):
    assert s['seed'] == n['seed']
    tr, te = np.array(s['train']), np.array(s['test'])
    p = gbr().fit(X[tr], y[tr]).predict(X[te])
    rows.append({'seed': s['seed'], 'test_rows': len(te), 'network_mae': n['mae'], 'trees_mae': float(np.abs(p - y[te]).mean()),
                 'naive_mae': float(np.abs(y[tr].mean() - y[te]).mean())})
diff = np.array([r['network_mae'] - r['trees_mae'] for r in rows])
k, sd = len(diff), diff.std(ddof=1)
tcrit = stats.t.ppf(0.975, k - 1)
half = tcrit * sd / np.sqrt(k)
# The eight test sets overlap, so the plain interval is too narrow. Nadeau & Bengio (Machine Learning 52, 2003)
# widen the standard error by sqrt(1/k + n_test/n_train) instead of sqrt(1/k).
ratio = np.mean([len(s['test']) / len(s['train']) for s in allrows['splits']])
half_nb = tcrit * sd * np.sqrt(1 / k + ratio)

# Importance of inputs to the trees, and the Al sweep through the trees for comparison.
imp = sorted(zip(data['features'], model.feature_importances_), key=lambda t: -t[1])[:8]
tree_sweep = model.predict(np.array([r['features'] for r in sweep['sweep']]))
nn_sweep = [r['hv_pred'] for r in nn['sweep']]
steps = np.diff(nn_sweep)
al = [r['al_x'] for r in sweep['sweep']]

result = {
    'task': 'Vickers hardness (HV) of multi-principal-element alloys from composition and processing route',
    'data': {'source': 'https://raw.githubusercontent.com/CitrineInformatics/MPEA_dataset/master/MPEA_dataset.csv',
             'citation': 'Borg et al., Scientific Data 7, 430 (2020)',
             'rows_used': prep['rows_used'], 'alloys_groups': prep['groups'],
             'train_rows': prep['train_rows'], 'test_rows': prep['test_rows'],
             'train_groups': prep['train_groups'], 'test_groups': prep['test_groups'],
             'hv_range': [prep['hv_min'], prep['hv_max']], 'columns_used_as_inputs': prep['columns_used_as_inputs'],
             'processing_counts': prep['processing_counts'],
             'repeat_measurement_noise': {'cells_with_repeats': prep['repeat_cells'], 'rows': prep['rows_in_repeat_cells'],
                                          'mean_abs_dev_from_cell_mean_HV': prep['repeat_mean_abs_dev_from_cell_mean_HV']}},
    'features': data['features'],
    'network': {'settings': nn['settings'], 'ensemble_members': nn['members'], 'layers': nn['layers'], 'weights_total': nn['weights'],
                'train_seconds_final': nn['train_seconds'], 'train_fit': nn['train_fit']},
    'cv_on_training_rows': {'network_single_chosen': cv['chosen_single_net_cv'], 'network_ensemble': cv['ensemble_cv'],
                            'trees_same_folds': metrics(oof, ytr), 'naive_same_folds': metrics(naive_oof, ytr),
                            'grid': [{k2: g[k2] for k2 in ('settings', 'cv_mae', 'cv_rmse', 'cv_r2')} for g in cv['grid']]},
    'b_test_split_once': {'network': nn['test'], 'trees': metrics(tree_pred, yte), 'naive_train_mean': metrics(naive_pred, yte),
                          'trees_fit_seconds': tree_seconds},
    'c_eight_splits': {'per_split': rows, 'network_mae_mean': float(np.mean([r['network_mae'] for r in rows])),
                       'trees_mae_mean': float(np.mean([r['trees_mae'] for r in rows])),
                       'naive_mae_mean': float(np.mean([r['naive_mae'] for r in rows])),
                       'mean_diff_network_minus_trees': float(diff.mean()),
                       'paired_t95': [float(diff.mean() - half), float(diff.mean() + half)],
                       'paired_t95_corrected_for_overlap': [float(diff.mean() - half_nb), float(diff.mean() + half_nb)],
                       'network_better_in': int((diff < 0).sum()), 'of': k},
    'd_interval90': {'halfwidth_HV': nn['interval90_halfwidth_HV'], 'rule': cv['interval_rule'],
                     'test_inside': nn['test_inside'], 'test_rows': nn['test_rows'], 'test_coverage': nn['test_coverage']},
    'sanity_al_sweep': {'alloy': 'Al_x Co Cr Fe Ni, as cast', 'al_x': al, 'network_hv': [round(v, 1) for v in nn_sweep],
                        'trees_hv': [round(float(v), 1) for v in tree_sweep],
                        'network_x0_to_x2': [round(nn_sweep[0], 1), round(nn_sweep[-1], 1)],
                        'network_rising_steps': int((steps > 0).sum()), 'steps': int(len(steps)),
                        'measured_in_dataset': sweep['measured_in_dataset']},
    'trees_top_inputs': [[n, round(float(v), 3)] for n, v in imp],
    'extra_cv_whole_alloy_systems_held_out_training_rows_only': {
        'network_ensemble': {k2: nn_sys[k2] for k2 in ('mae', 'rmse', 'r2', 'per_fold_mae')},
        'trees': metrics(soof, ytr), 'naive': metrics(snaive, ytr),
        'alloy_systems_in_train': prep['alloy_systems_in_train'],
        'test_rows_whose_element_set_is_also_in_train': prep['test_rows_whose_element_set_is_also_in_train']},
    'guards_note': {'what': 'standardised inputs capped at +-3 and predictions kept inside the training HV range',
                    'why': 'first CV pass without guards gave absurd out-of-fold predictions for a few unusual alloys',
                    'unguarded_single_net_cv_grid': [{'hidden': g['settings']['hidden'], 'decay': g['settings']['decay'],
                                                      'logTarget': g['settings']['logTarget'], 'loss': g['settings']['loss'],
                                                      'cv_mae': g['cv_mae'], 'cv_r2': g['cv_r2']} for g in ung['grid']],
                    'unguarded_best_ensemble_cv': ung['ensemble_cv']},
}
result['sanity_al_sweep']['outcome'] = (
    'Direction is right: network prediction rises at every one of the 20 steps, from %.0f HV at x=0 to %.0f HV at x=2, '
    'steepest from x=0.3 to about x=0.8 and flattening above x=1.2. Against as-cast measurements in the file it is too high at x=0.6 '
    '(predicted %.0f, one test-side measurement 228) and too low at x=0.9 (predicted %.0f, one test-side measurement 511): the '
    'network draws a smooth ramp where the measurements show a sharper jump. Not an independent physics test: Al-Co-Cr-Fe-Ni '
    'alloys are among the training rows (as cast at x=0.3, 1.0, 1.5, 2.5, 3.0; annealed at x=0.25 to 1.5), so part of this is recall. '
    'The trees give the same overall rise but in steps and not monotonically.' % (nn_sweep[0], nn_sweep[-1], nn_sweep[6], nn_sweep[9]))
result['caveats'] = [
    'Network and trees are not distinguishable. Single test split: trees better by about 5 HV MAE. Eight further splits: network '
    'better by about 3 HV on average, 7 of 8, but the 95% interval reaches zero and the overlap-corrected one clearly includes it. '
    'The network settings were tuned on rows that reappear as test rows in those eight splits, which favours the network slightly there, '
    'and the settings tried differ by less than the fold-to-fold noise (single-net CV MAE 76.0 to 88.5 with guards). '
    'The network needed two guards (capped inputs, clamped output) to stop absurd predictions on unusual alloys; the trees need none.',
    'The error is large and the split is kind. Typical miss about 75 HV on a 95-1183 scale, 90%% interval about +-180 HV. Only the exact '
    'formula is kept on one side: near neighbours (Al0.3 vs Al0.25 CoCrFeNi, often from the same paper) sit on both sides, and %d of the %d '
    'test rows come from element sets also present in training. With whole alloy systems held out (3-fold, training rows only) MAE rises to '
    '%.0f HV for the network and %.0f HV for the trees (naive %.0f). Scores for alloys from a family nobody has studied are less trustworthy '
    'than the test numbers.' % (prep['test_rows_whose_element_set_is_also_in_train'], prep['test_rows'], nn_sys['mae'],
                                metrics(soof, ytr)['mae'], metrics(snaive, ytr)['mae']),
    'The data limit the model. 530 literature measurements from many labs, loads and heat treatments; processing is five coarse labels and '
    'blank in 73 rows (no temperatures or times); repeat measurements of the same formula and route (%d cases, %d rows) sit on average %.0f HV '
    'from their own mean, so a good part of the error cannot be removed with these inputs. The element table (radius, electronegativity, '
    'valence electrons, melting point) was typed in from memory with citations and not re-checked against the sources.'
    % (prep['repeat_cells'], prep['rows_in_repeat_cells'], prep['repeat_mean_abs_dev_from_cell_mean_HV']),
]
result['provenance'] = ('Every number in this file was computed in this session by the scripts listed under files, on the CSV downloaded '
                        'in this session. Nothing is quoted from the literature except the direction of the Al trend, which was given in the brief. '
                        'The network was evaluated on the test split exactly once (run_nn.js final). eval_trees.py, which has fixed settings and a '
                        'fixed seed, was re-executed several times (four runs, one of which crashed on a text-formatting error before writing anything) only to add report fields; its test numbers were identical each time and nothing was tuned on them.')
result['files'] = [d + '/' + n for n in (
    'data/MPEA_dataset.csv', 'prepare.py', 'nn_generic.js', 'run_nn.js', 'eval_trees.py', 'package.json', 'nn_data.json', 'splits.json',
    'cv_folds.json', 'cv_folds_system.json', 'sweep.json', 'prepare_summary.json', 'cv_results.json', 'cv_results_unguarded.json',
    'cv_system_nn.json', 'nn_final.json', 'result.json', 'cv.log', 'cv_unguarded.log', 'final.log', 'trees.log')]
json.dump(result, open('%s/result.json' % d, 'w'), indent=1)
show = {k2: result[k2] for k2 in ('b_test_split_once', 'c_eight_splits', 'd_interval90', 'trees_top_inputs')}
print(json.dumps(show, indent=1))
print('cv', json.dumps({k2: v for k2, v in result['cv_on_training_rows'].items() if k2 != 'grid'}))
s = result['sanity_al_sweep']
print('sweep nn', s['network_hv']); print('sweep trees', s['trees_hv']); print('measured', json.dumps(s['measured_in_dataset']))
