# Computes every reported metric from saved predictions and writes result.json.
# Usage: python3 -I evaluate.py <workdir> <csv> [notes.json]
import csv, json, math, sys, os
import numpy as np
W, csv_path = sys.argv[1], sys.argv[2]
notes = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
L = lambda name: json.load(open(os.path.join(W, name)))
a, info, cvr, nn, trees, sp = L('all_rows.json'), L('prepare_info.json'), L('cv_result.json'), L('nn_final.json'), L('trees.json'), L('splits.json')['splits']
nn_sp = L('nn_splits.json') if os.path.exists(os.path.join(W, 'nn_splits.json')) else []
y = np.array(a['y']); g = np.array(a['g']); lab = np.array(a['label']); tr = np.array(a['main_train']); te = np.array(a['main_test'])
def met(p, t):
    p = np.asarray(p); t = np.asarray(t); e = p - t
    return {'n': int(len(t)), 'mae': round(float(np.abs(e).mean()), 4), 'rmse': round(float(np.sqrt((e ** 2).mean())), 4),
            'r2': round(float(1 - (e ** 2).sum() / ((t - t.mean()) ** 2).sum()), 4)}
yt = y[te]; p_nn = np.array(nn['pred']); p_tr = np.array(trees['pred']); p_nv = np.full(len(te), y[tr].mean())
res = {}
res['data'] = {'source': 'AqSolDB data_curated.csv (Sorkun, Khetan, Er 2019)', 'csv_rows': info['csv_rows'], 'rows_used': info['rows_used'],
               'rows_dropped': len(info['dropped']), 'missing_or_non_finite_descriptor_rows': 0 if not info['dropped'] else len(info['dropped']),
               'groups_inchikey_first_block': info['groups'], 'groups_with_more_than_one_row': info['groups_with_more_than_one_row'],
               'train_rows': info['train_rows'], 'test_rows': info['test_rows'], 'train_groups': info['train_groups'], 'test_groups': info['test_groups'],
               'target': 'Solubility (LogS, log10 mol/L)', 'target_mean_sd_all_rows': [round(info['y_mean'], 3), round(info['y_sd'], 3)],
               'balabanj_zero_rows_kept_as_is': info['balabanj_zero_rows']}
res['features'] = a['features']
res['transforms'] = {'log1p_columns_decided_on_training_rows': nn['log1p_columns'],
                     'rule': 'log1p if training min >= 0 and training skewness > 1; then z-score with training mean/sd; then clip to +-5 sd; target z-scored with training mean/sd',
                     'not_used_as_inputs': ['SD', 'Occurrences', 'Group', 'SMILES', 'InChI', 'Name']}
res['network'] = {'chosen_settings': cvr['chosen'], 'layers': nn['layers'], 'weights_all_members': nn['weights'], 'train_seconds_final_ensemble_node': nn['train_seconds'],
                  'optimizer': 'Adam, lr decays linearly to 10% over the epochs, L2 weight decay inside the gradient (Mlp.step as copied)'}
oof_t = np.array(trees['oof']); oof_n = np.array(cvr['oof'])
res['a_cv_on_training_rows'] = {'folds': 3, 'grouped': True,
    'grid_single_seed': [{'hidden': r['cfg']['hidden'], 'decay': r['cfg']['decay'], 'loss': r['cfg']['loss'], 'log1p': r['cfg']['log1p'],
                          'cv_mae': round(r['mae'], 4), 'cv_rmse': round(r['rmse'], 4), 'sec_3_folds': round(r['sec'], 1)} for r in cvr['table']],
    'chosen_by': 'lowest CV MAE with one seed, then run as a 5-seed ensemble',
    'network_ensemble_cv': met(oof_n, y[tr]), 'trees_cv_same_folds': met(oof_t, y[tr])}
single = [met(m, yt)['mae'] for m in nn['member_pred']]
res['b_test_once'] = {'network_ensemble': met(p_nn, yt), 'trees_gbr': met(p_tr, yt), 'naive_train_mean': met(p_nv, yt),
                      'network_single_members_mae': single}
# grouped bootstrap of the MAE difference on the test split
rng = np.random.default_rng(0); gt = g[te]; ug, inv = np.unique(gt, return_inverse=True)
d_abs = np.abs(p_nn - yt) - np.abs(p_tr - yt); gs = np.bincount(inv, weights=d_abs); gc = np.bincount(inv)
bs = []
for _ in range(4000):
    k = rng.integers(0, len(ug), len(ug)); bs.append(gs[k].sum() / gc[k].sum())
res['b_test_once']['mae_network_minus_trees'] = {'value': round(float(d_abs.mean()), 4), 'grouped_bootstrap_95': [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
# (c) eight extra splits
rows = []
tp = {s['seed']: s['pred'] for s in trees['splits']}; npd = {s['seed']: s for s in nn_sp}
for s in sp:
    if s['seed'] not in npd: continue
    t = y[s['test']]
    rows.append({'seed': s['seed'], 'mae_network': met(npd[s['seed']]['pred'], t)['mae'], 'mae_trees': met(tp[s['seed']], t)['mae'],
                 'mae_naive': met(np.full(len(t), y[s['train']].mean()), t)['mae'], 'network_train_seconds': round(npd[s['seed']]['sec'], 1)})
c = {'splits_done': len(rows), 'per_split': rows}
if len(rows) >= 2:
    d = np.array([r['mae_network'] - r['mae_trees'] for r in rows]); n = len(d)
    from scipy import stats
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)
    c.update({'mean_mae_network': round(float(np.mean([r['mae_network'] for r in rows])), 4), 'mean_mae_trees': round(float(np.mean([r['mae_trees'] for r in rows])), 4),
              'mean_difference_network_minus_trees': round(float(d.mean()), 4), 'paired_t_95_interval': [round(float(d.mean() - half), 4), round(float(d.mean() + half), 4)],
              'network_better_in_n_splits': int((d < 0).sum()),
              'note': 'splits overlap heavily (resampled from the same 9,982 rows), so the t interval is narrower than a truly independent one would be'})
res['c_eight_splits'] = c
# (d) conformal-style 90% interval
q = cvr['q90_abs_error']; ae = np.abs(p_nn - yt)
qt = float(np.sort(np.abs(oof_t - y[tr]))[min(len(tr), math.ceil((len(tr) + 1) * 0.9)) - 1])
res['d_interval_90'] = {'network_half_width_logS': round(q, 3), 'from': 'absolute out-of-fold errors of the 5-seed ensemble in 3-fold grouped CV on training rows (n=%d)' % cvr['n_cv_rows'],
                        'test_coverage': round(float((ae <= q).mean()), 4), 'test_rows': int(len(te)),
                        'trees_half_width_logS': round(qt, 3), 'trees_test_coverage': round(float((np.abs(p_tr - yt) <= qt).mean()), 4)}
# (e) reliability labels, checked against the data itself
recs = {}
with open(csv_path, newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        recs.setdefault(r['Group'], []).append((float(r['SD']), int(float(r['Occurrences']))))
lab_def = {k: {'rows': len(v), 'occurrences_min_max': [min(o for _, o in v), max(o for _, o in v)], 'sd_min_max': [round(min(s for s, _ in v), 3), round(max(s for s, _ in v), 3)],
               'sd_median': round(float(np.median([s for s, _ in v])), 3)} for k, v in sorted(recs.items())}
lt = lab[te]; rel = np.isin(lt, ['G3', 'G5'])
per = {k: {'n': int((lt == k).sum()), 'mae_network': met(p_nn[lt == k], yt[lt == k])['mae'], 'mae_trees': met(p_tr[lt == k], yt[lt == k])['mae']} for k in sorted(set(lt))}
res['e_reliable_rows_G3_G5'] = {'label_meaning_checked_in_csv': lab_def, 'network_ensemble': met(p_nn[rel], yt[rel]), 'trees_gbr': met(p_tr[rel], yt[rel]),
                               'naive_train_mean': met(p_nv[rel], yt[rel]), 'other_rows_network': met(p_nn[~rel], yt[~rel]), 'other_rows_trees': met(p_tr[~rel], yt[~rel]),
                               'per_label_test': per, 'interval_coverage_reliable_rows': round(float((ae[rel] <= q).mean()), 4)}
sw = nn['sweeps']
res['sanity_sweeps'] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'mean_curve'} for k, v in sw.items()}
for k in sw: res['sanity_sweeps'][k]['mean_curve_first_mid_last'] = [round(sw[k]['mean_curve'][i], 3) for i in (0, 10, 20)]
res['trees_feature_importance_top5'] = sorted(((k, round(v, 3)) for k, v in trees['importances'].items()), key=lambda t: -t[1])[:5]
res.update(notes)
json.dump(res, open(os.path.join(W, 'result.json'), 'w'), indent=1)
print(json.dumps(res))
