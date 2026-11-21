# Chooses the network settings by grouped CV on training rows, then evaluates once.
# Usage: python3 -I evaluate.py <work dir>
import sys, json, math, subprocess, time, numpy as np
from sklearn.ensemble import GradientBoostingRegressor
W = sys.argv[1]
T0 = time.time()
D = json.load(open(f'{W}/tasks.json')); tasks = D['tasks']; feats = D['features']
prep = json.load(open(f'{W}/prepare_report.json'))
rows = json.load(open(f'{W}/rows.json'))
iX, iAv, iFox = feats.index('x_a'), feats.index('fox_available'), feats.index('fox_tg_from_train')

def node(names, opts, tag):
    out = f'{W}/runs/{tag}.json'
    subprocess.run(['nice', '-n', '15', 'node', f'{W}/run_nn.js', f'{W}/tasks.json', out, ','.join(names), json.dumps(opts)], check=True, cwd=W)
    return json.load(open(out))

def metrics(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float); e = p - y
    return dict(mae=round(float(np.mean(np.abs(e))), 2), rmse=round(float(np.sqrt(np.mean(e ** 2))), 2),
                r2=round(float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), 3), n=int(len(y)))

def gbr(t):
    m = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
    m.fit(np.array(t['train']['X']), t['train']['y'])
    return m

CV = ['cv0', 'cv1', 'cv2']
ycv = np.concatenate([tasks[k]['test']['y'] for k in CV])
base = dict(hidden=[64, 32], epochs=300, batch=32, lr=0.004, decay=1e-3, loss='squared', seed=11)
grid = [dict(base), dict(base, loss='huber'), dict(base, loss='huber', decay=1e-2), dict(base, loss='huber', decay=3e-2),
        dict(base, loss='huber', decay=1e-2, hidden=[32, 16]), dict(base, loss='huber', decay=1e-2, hidden=[128, 64], epochs=200),
        dict(base, loss='huber', decay=1e-2, epochs=150), dict(base, loss='huber', decay=1e-2, hidden=[32]),
        dict(base, loss='huber', decay=1e-1), dict(base, loss='huber', decay=3e-2, hidden=[32, 16], lr=0.002)]
cv_table = []
for k, cfg in enumerate(grid):
    o = node(CV, dict(cfg, members=1), f'cv_cfg{k}')
    p = np.concatenate([o[c]['pred'] for c in CV])
    cv_table.append(dict(config=cfg, cv_mae_single_net=round(float(np.mean(np.abs(p - ycv))), 2)))
    print(k, cv_table[-1], flush=True)
best = min(range(len(grid)), key=lambda k: cv_table[k]['cv_mae_single_net'])
cfg = grid[best]
o = node(CV, dict(cfg, members=5), 'cv_best_ens5')
p_ens = np.concatenate([o[c]['pred'] for c in CV])
cv_ens = float(np.mean(np.abs(p_ens - ycv)))
members = 5 if cv_ens <= cv_table[best]['cv_mae_single_net'] else 1
if members == 5:
    oof_abs = np.sort(np.abs(p_ens - ycv))
else:
    o1 = json.load(open(f'{W}/runs/cv_cfg{best}.json')); oof_abs = np.sort(np.abs(np.concatenate([o1[c]['pred'] for c in CV]) - ycv))
ncal = len(oof_abs); q90 = float(oof_abs[min(ncal, math.ceil((ncal + 1) * 0.9)) - 1])
gb_cv = np.concatenate([gbr(tasks[c]).predict(np.array(tasks[c]['test']['X'])) for c in CV])
final_opts = dict(cfg, members=members)
print('chosen', final_opts, 'cv ens', cv_ens, 'q90', q90, flush=True)

SPL = [f's{s}' for s in range(1001, 1009)]
fin = node(['main'] + SPL + ['hard'], final_opts, 'final')

def evaluate(name):
    t = tasks[name]; y = np.array(t['test']['y']); Xte = np.array(t['test']['X'])
    m = gbr(t); pg = m.predict(Xte); pn = np.array(fin[name]['pred'])
    naive = np.full(len(y), np.mean(t['train']['y']))
    cop = (Xte[:, iX] > 0) & (Xte[:, iX] < 1); av = Xte[:, iAv] == 1
    r = dict(train_rows=len(t['train']['y']), test_rows=len(y), network=metrics(pn, y), trees=metrics(pg, y), naive_mean=metrics(naive, y),
             network_train_mae=round(float(np.mean(np.abs(np.array(fin[name]['train_pred']) - np.array(t['train']['y'])))), 2),
             interval90_half_width_K=round(q90, 1), interval90_test_coverage=round(float(np.mean(np.abs(pn - y) <= q90)), 3),
             subsets={})
    for lab, msk in [('true_copolymer_rows', cop), ('homopolymer_rows', ~cop), ('fox_available_rows', av), ('fox_not_available_rows', ~av)]:
        if msk.sum() >= 3:
            r['subsets'][lab] = dict(network=metrics(pn[msk], y[msk]), trees=metrics(pg[msk], y[msk]), naive_mean=metrics(naive[msk], y[msk]),
                                     interval90_coverage=round(float(np.mean(np.abs(pn[msk] - y[msk]) <= q90)), 3))
    if av.sum() >= 3:
        r['subsets']['fox_available_rows']['fox_equation_alone'] = metrics(Xte[av, iFox], y[av])
    return r, m, pn, pg

res_main, gb_main, pn_main, pg_main = evaluate('main')
res_hard = evaluate('hard')[0]
per = []
for s in SPL:
    r = evaluate(s)[0]
    per.append(dict(seed=int(s[1:]), network_mae=r['network']['mae'], trees_mae=r['trees']['mae'], naive_mae=r['naive_mean']['mae'],
                    network_r2=r['network']['r2'], trees_r2=r['trees']['r2'], coverage90=r['interval90_test_coverage']))
d = np.array([p['network_mae'] - p['trees_mae'] for p in per]); half = 2.364624 * d.std(ddof=1) / math.sqrt(len(d))

# composition sweep on test copolymer systems
ex = tasks['main']['extra']; eg = np.array(ex['g']); pn_sw = np.array(fin['main']['extra_pred']); pg_sw = gb_main.predict(np.array(ex['X']))
def mono(v, tol):
    dv = np.diff(v); return bool(np.all(dv >= -tol) or np.all(dv <= tol))
te_idx = tasks['main']['test']['idx']
sw = dict(systems=int(len(set(eg))), points_per_system=21)
for lab, pv in [('network', pn_sw), ('trees', pg_sw)]:
    for tol in (0.0, 1.0):
        sw[f'{lab}_monotonic_share_tol{tol:g}K'] = round(float(np.mean([mono(pv[eg == gid], tol) for gid in sorted(set(eg))])), 3)
meas = []
for gid in sorted(set(eg)):
    pts = sorted((rows[i]['xa'], rows[i]['tg']) for i in te_idx if rows[i]['g'] == gid)
    if len(pts) >= 3: meas.append(mono(np.array([p[1] for p in pts]), 0.0))
sw['measured_monotonic_share_systems_with_3plus_points'] = round(float(np.mean(meas)), 3); sw['measured_systems_counted'] = len(meas)

result = dict(
    task='Predict measured glass-transition temperature Tg (K) of homopolymers and binary copolymers',
    numbers_run_by='every number in this file was computed in this session by prepare.py / evaluate.py / run_nn.js; nothing is quoted from a paper',
    data=dict(rows_used_copolymer_file=prep['rows_used_copolymer_file'], rows_used_homopolymer_file=0,
              training_set_choice='copolymer file alone: it already contains 315 homopolymer rows (301 with empty SMILES_1), and 304 of the 313 distinct structures in the Bicerano file are among them (293 with the same Tg within 0.5 K); merging would mostly add duplicates, and the 9 new rows were left out',
              groups=prep['groups'], distinct_monomers=prep['distinct_monomers'], copolymer_systems=prep['copolymer_systems'],
              rows_true_copolymer=prep['rows_true_copolymer'], rows_homopolymer=prep['rows_homopolymer'],
              duplicates=prep['duplicate_structure_composition_rows'], missing_tg=prep['missing_Tg'],
              composition_basis='not stated in the file; column x1 = comp_0/100 looks like a mole fraction, so comp was treated as mole percent and converted to weight fraction (repeat-unit molar mass) for the Fox estimate',
              homopolymer_file_check=prep['homopolymer_file']),
    rdkit_installed=True, rdkit_version='2026.09.2',
    features=feats, feature_count=len(feats), feature_set_choice=prep['feature_set_choice'],
    fox_feature_note='homopolymer Tg values come only from training rows of the split in use, and never from the row\'s own copolymer system (leave-own-group-out), recomputed for every CV fold and every split',
    network=dict(chosen_settings=final_opts, layers=fin['main']['layers'], weights=fin['main']['weights'],
                 cv_grid=cv_table, cv_mae_chosen_single_net=cv_table[best]['cv_mae_single_net'], cv_mae_chosen_ensemble5=round(cv_ens, 2),
                 cv_mae_trees_same_folds=round(float(np.mean(np.abs(gb_cv - ycv))), 2),
                 training_seconds_final_network=round(fin['main']['seconds'], 2)),
    b_test_split_seed42=res_main,
    c_eight_splits=dict(per_split=per, mean_network_mae=round(float(np.mean([p['network_mae'] for p in per])), 2),
                        mean_trees_mae=round(float(np.mean([p['trees_mae'] for p in per])), 2),
                        mean_difference_network_minus_trees=round(float(d.mean()), 2),
                        paired_t95_interval=[round(float(d.mean() - half), 2), round(float(d.mean() + half), 2)],
                        mean_coverage90=round(float(np.mean([p['coverage90'] for p in per])), 3)),
    d_interval=dict(method='absolute-error quantile of out-of-fold predictions from 3-fold grouped CV on the 514 training rows (split-conformal style), ceil((n+1)*0.9)-th smallest',
                    calibration_rows=ncal, half_width_K=round(q90, 1), test_coverage=res_main['interval90_test_coverage'],
                    test_coverage_true_copolymer_rows=res_main['subsets'].get('true_copolymer_rows', {}).get('interval90_coverage'),
                    hard_split_coverage=res_hard['interval90_test_coverage']),
    e_hard_split_unseen_monomer=dict(prep['hard_split'], **res_hard),
    sanity_composition_sweep=sw,
    evaluate_seconds=round(time.time() - T0, 1))
json.dump(result, open(f'{W}/result_core.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in result.items() if k not in ('features',)}, indent=1))
