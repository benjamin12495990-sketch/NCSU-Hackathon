"""Turns a pipeline folder's raw results into honest comparisons.

Tabular families (network sweep vs CPU baselines on the same folds)
  A sweep tries hundreds of network settings; the best one on the cross-validation folds looks better than it is, because
  it was picked for exactly that score. Here, for each fold, the setting is chosen using ONLY the other folds' errors and
  then scored on the fold it did not see ("cross-fitted selection"). The same is done for the tree models and the linear
  models, so every method gets the same treatment. The difference between methods is then a paired bootstrap over whole
  groups (mixes, alloy systems, papers, ...), 2,000 resamples, fixed seed. The numbers without this correction are also
  reported, labelled as optimistic. Concrete has no grouped folds (train on ages 1-7 d, score 14-28 d): the setting is chosen
  on REWARD and the comparison is made on MONITOR, which played no part in the choice; all of it is exploratory.

Structure families (crystal-graph network vs tree baselines): gpu/structure/compare_structure.py.

A "verdict" is only ever one of: network clearly better (the whole 95% interval of MAE(network) - MAE(trees) is below zero),
trees clearly better (the whole interval is above zero), or no clear difference.
"""
import glob
import json
from pathlib import Path

import numpy as np

from gpu.structure.compare_structure import _paired, compare


def _first(pipe, jid, pattern):
    hits = sorted(glob.glob(str(Path(pipe) / 'jobs' / jid / pattern), recursive=True))
    return Path(hits[-1]) if hits else None


def verdict(lo, hi):
    return 'network clearly better' if hi < 0 else 'trees clearly better' if lo > 0 else 'no clear difference'


def _m(p, y):
    e = np.asarray(p) - np.asarray(y)
    sst = float(((y - np.mean(y)) ** 2).sum())
    return {'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt((e ** 2).mean())), 'r2': float(1 - (e ** 2).sum() / sst) if sst > 0 else None}


def crossfit(P, y, folds):
    """P: [candidates, rows] out-of-fold predictions. For each fold, take the candidate with the lowest MAE on the OTHER folds
    and keep its predictions on this fold. Returns (predictions [rows], chosen candidate per fold)."""
    err = np.abs(P - y)
    out, chosen = np.empty(len(y)), []
    for f in range(int(folds.max()) + 1):
        c = int(np.argmin(err[:, folds != f].mean(axis=1)))
        out[folds == f] = P[c, folds == f]
        chosen.append(c)
    return out, chosen


def _cfg_text(c):
    return f'hidden {c["hidden"]}, {c["epochs"]} epochs, lr {c["lr"]:g}, decay {c["decay"]:g}, {c["loss"]} loss'


def analyze_tabular(pipe, family):
    from gpu.data import load
    sweep = _first(pipe, f'tab_{family}', '*/*/result.json')
    base = _first(pipe, f'base_{family}', 'baselines.json')
    if sweep is None or base is None:
        return {'family': family, 'status': 'missing', 'reason': ('the network sweep' if sweep is None else 'the baselines') + ' did not finish'}
    ds = load(family)
    res = json.loads(sweep.read_text())
    bj = json.loads(base.read_text())
    if res['data']['data_sha256'] != ds.data_sha256 or bj['data_sha256'] != ds.data_sha256:
        return {'family': family, 'status': 'error', 'reason': 'the sweep, the baselines and the current data files disagree (data hashes differ)'}
    P = np.load(sweep.parent / 'predictions.npz')
    B = np.load(base.with_name('baselines_predictions.npz'))
    names, groups_of = bj['model_names'], bj['groups']
    cfgs = res['configs']
    out = {'family': family, 'status': 'ok', 'target': ds.target, 'rows': int(len(ds.y)), 'groups': int(len(set(map(str, ds.groups)))),
           'n_network_settings': len(cfgs), 'n_seeds': int(P[[k for k in P.files if k.startswith(('cv_', 'fixed_'))][0]].shape[1]),
           'sweep_seconds': res.get('wall_seconds'), 'baseline_seconds': bj.get('wall_seconds'), 'device': res.get('device')}
    gi = lambda g: [i for i, n in enumerate(names) if groups_of[n] == g]  # noqa: E731
    if bj['mode'] == 'cv':
        y, folds, grp = ds.y, ds.folds, np.array([str(g) for g in ds.groups], dtype=object)
        ens = P['cv_val'].mean(axis=1)                         # [settings, rows]: the average of the seeds
        single = P['cv_val'][:, 0, :]                          # one seed only
        oof = B['oof']
        net_cf, net_pick = crossfit(ens, y, folds)
        one_cf, _ = crossfit(single, y, folds)
        out['mode'] = 'cv'
        out['folds'] = int(folds.max() + 1)
        out['n_trees_models'], out['n_linear_models'] = len(gi('trees')), len(gi('linear'))
        methods = {'mean': oof[gi('mean')[0]]}
        picks = {}
        for g in ('linear', 'trees'):
            idx = gi(g)
            methods[g], ch = crossfit(oof[idx], y, folds)
            picks[g] = [names[idx[c]] for c in ch]
        methods['network'] = net_cf
        out['methods'] = {k: _m(v, y) for k, v in methods.items()}
        out['network_single_seed'] = _m(one_cf, y)
        out['optimistic_best_of_all'] = {'network': min(_m(ens[i], y)['mae'] for i in range(len(ens))),
                                         **{g: min(_m(oof[i], y)['mae'] for i in gi(g)) for g in ('linear', 'trees')}}
        out['chosen'] = {'network': [_cfg_text(cfgs[c]) for c in net_pick], **picks}
        for g in ('trees', 'linear'):
            d, lo, hi, share = _paired(np.abs(net_cf - y), np.abs(methods[g] - y), grp)
            out[f'network_vs_{g}'] = {'diff': d, 'ci95': [lo, hi], 'share_groups_network_better': share, 'verdict': verdict(lo, hi)}
        out['note'] = 'Settings were chosen on the other folds only (cross-fitted), so the best-of-many advantage is removed.'
    else:
        sel, evl = 'test', next(iter(ds.extra))
        y_sel, y_ev, grp = ds.y_test, ds.extra[evl][1], np.array([str(g) for g in ds.extra[evl][2]], dtype=object)
        ens_sel, ens_ev = P[f'fixed_{sel}'].mean(axis=1), P[f'fixed_{evl}'].mean(axis=1)
        ci = int(np.argmin([np.abs(e - y_sel).mean() for e in ens_sel]))
        out['mode'] = 'fixed'
        out['selection_set'], out['evaluation_set'] = 'REWARD (ages 14-28 d)', 'MONITOR (ages 14-28 d, never used to choose anything)'
        methods = {'mean': B[f'pred_{evl}'][gi('mean')[0]]}
        chosen = {'network': _cfg_text(cfgs[ci])}
        for g in ('linear', 'trees'):
            idx = gi(g)
            c = idx[int(np.argmin([np.abs(B[f'pred_{sel}'][i] - y_sel).mean() for i in idx]))]
            methods[g] = B[f'pred_{evl}'][c]
            chosen[g] = names[c]
        methods['network'] = ens_ev[ci]
        if 'ridge_ln_age_only' in names:
            methods['ridge_ln_age_only'] = B[f'pred_{evl}'][names.index('ridge_ln_age_only')]
        out['methods'] = {k: _m(v, y_ev) for k, v in methods.items()}
        out['methods_on_selection_set'] = {'network': _m(ens_sel[ci], y_sel)}
        out['chosen'] = chosen
        out['n_trees_models'], out['n_linear_models'] = len(gi('trees')), len(gi('linear'))
        for g in ('trees', 'linear', 'ridge_ln_age_only'):
            if g in methods:
                d, lo, hi, share = _paired(np.abs(methods['network'] - y_ev), np.abs(methods[g] - y_ev), grp)
                out[f'network_vs_{g}'] = {'diff': d, 'ci95': [lo, hi], 'share_groups_network_better': share, 'verdict': verdict(lo, hi)}
        out['note'] = ('These concrete rows already informed project decisions. The network setting and each baseline were chosen on REWARD; the comparison is on MONITOR.')
    return out


def analyze_structure(pipe, target):
    trees = _first(pipe, f'trees_{target}', 'trees_predictions.npz')
    run_dirs = []
    for job in sorted(glob.glob(str(Path(pipe) / 'jobs' / f'struct_{target}_s*'))):
        done = [r.parent for r in map(Path, sorted(glob.glob(str(Path(job) / 'structure' / '*' / 'result.json')))) if (r.parent / 'predictions.npz').exists()]
        if done:
            run_dirs.append(done[-1])  # one run per seed job: the latest complete one, never two copies of the same seed
    if trees is None or not run_dirs:
        return {'target': target, 'status': 'missing', 'reason': ('the tree baseline' if trees is None else 'the network runs') + ' did not finish'}
    try:
        cmp_out, lines = compare(run_dirs, trees)
    except ValueError as e:
        return {'target': target, 'status': 'error', 'reason': str(e)}
    res = [json.loads((r / 'result.json').read_text()) for r in run_dirs]
    tj = json.loads(trees.with_name('trees.json').read_text())
    out = {'target': target, 'status': 'ok', 'label': res[0]['target'], 'n_runs': len(run_dirs), 'device': res[0]['device'],
           'counts': res[0]['counts'], 'split_sizes': res[0]['split']['sizes'],
           'mean_baseline_test_mae': res[0]['baselines']['mean']['test']['mae'],
           'ridge_test_mae': res[0]['baselines']['ridge_element_fractions']['test']['mae'],
           'network_runs': [{'seed': r['config']['seed'], 'best_epoch': r['cgcnn']['best_epoch_by_val'], 'test': r['cgcnn']['test'],
                             'train_seconds': r['timing']['train_seconds'], 'structures_per_second': r['timing']['train_structures_per_second']} for r in res],
           'trees': {k: {'selected': v['selected'], 'test': v['test']} for k, v in tj['feature_sets'].items()},
           'compare': cmp_out, 'compare_lines': lines}
    for key, comps in cmp_out['comparisons'].items():
        avg = [v for k, v in comps.items() if k.startswith('network average')] or list(comps.values())[:1]
        out.setdefault('verdicts', {})[key] = {'diff': avg[0]['diff'], 'ci95': avg[0]['ci95'], 'verdict': verdict(*avg[0]['ci95'])}
    return out


def analyze_all(pipe, plan):
    fams = [j['family'] for j in plan['jobs'] if j['id'].startswith('tab_')]
    targets = sorted({j['target'] for j in plan['jobs'] if j['id'].startswith('struct_')}, key=lambda t: [j['target'] for j in plan['jobs'] if j.get('target')].index(t))
    res = {'tabular': [], 'structure': []}
    for f in fams:
        try:
            res['tabular'].append(analyze_tabular(pipe, f))
        except Exception as e:  # one broken family must not stop the report
            res['tabular'].append({'family': f, 'status': 'error', 'reason': repr(e)})
    for t in targets:
        try:
            res['structure'].append(analyze_structure(pipe, t))
        except Exception as e:
            res['structure'].append({'target': t, 'status': 'error', 'reason': repr(e)})
    return res
