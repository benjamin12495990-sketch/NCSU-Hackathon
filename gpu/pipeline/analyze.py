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
import fnmatch
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


def crossfit_topk(P, y, folds, k=3):
    """Like crossfit, but averages the k best candidates (chosen on the other folds) instead of keeping one."""
    err = np.abs(P - y)
    out = np.empty(len(y))
    for f in range(int(folds.max()) + 1):
        best = np.argsort(err[:, folds != f].mean(axis=1))[:k]
        out[folds == f] = P[best][:, folds == f].mean(axis=0)
    return out


def _cmp(a, b, y, grp):
    """MAE(a) - MAE(b), paired bootstrap over groups; negative favours a."""
    d, lo, hi, _ = _paired(np.abs(a - y), np.abs(b - y), grp)
    return {'diff': d, 'ci95': [lo, hi], 'verdict': verdict(lo, hi)}


def fairness(y, grp, net_ens, singles, trees, trees_top=None):
    """Checks that the headline comparison (average of several networks against ONE tree model) is not an ensembling effect.
      singles    one prediction vector per network seed/run (a single network each)
      trees_top  predictions of an average of several tree models, if available
    A blend is half the network average and half the trees, with no fitted weight, so it has no selection to be optimistic about."""
    maes = [float(np.abs(s - y).mean()) for s in singles]
    vs = [_cmp(s, trees, y, grp) for s in singles]
    blend = 0.5 * net_ens + 0.5 * trees
    out = {'single': {'maes': maes, 'mean_mae': float(np.mean(maes)), 'vs_trees': vs,
                      'n_clearly_better': sum(v['verdict'] == 'network clearly better' for v in vs),
                      'n_clearly_worse': sum(v['verdict'] == 'trees clearly better' for v in vs), 'n': len(vs)},
           'blend': {'mae': float(np.abs(blend - y).mean()), 'vs_trees': _cmp(blend, trees, y, grp), 'vs_network_average': _cmp(blend, net_ens, y, grp)},
           'median_ae': {'network average': float(np.median(np.abs(net_ens - y))), 'one network': float(np.median(np.abs(singles[0] - y))),
                         'trees': float(np.median(np.abs(trees - y)))}}
    if trees_top is not None:
        out['trees_top3'] = {'mae': float(np.abs(trees_top - y).mean()), 'network_average_vs_it': _cmp(net_ens, trees_top, y, grp)}
        out['median_ae']['top-3 trees averaged'] = float(np.median(np.abs(trees_top - y)))
    return out


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
        singles = [crossfit(P['cv_val'][:, s, :], y, folds)[0] for s in range(P['cv_val'].shape[1])]
        out['fairness'] = fairness(y, grp, net_cf, singles, methods['trees'], crossfit_topk(oof[gi('trees')], y, folds))
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
        top = np.argsort([np.abs(B[f'pred_{sel}'][i] - y_sel).mean() for i in gi('trees')])[:3]
        out['fairness'] = fairness(y_ev, grp, ens_ev[ci], [P[f'fixed_{evl}'][ci, s, :] for s in range(P[f'fixed_{evl}'].shape[1])], methods['trees'],
                                   np.mean([B[f'pred_{evl}'][gi('trees')[i]] for i in top], axis=0))
        out['note'] = ('These concrete rows already informed project decisions. The network setting and each baseline were chosen on REWARD; the comparison is on MONITOR.')
    return out


TREE_KEYS = {'composition': 'pred_composition', 'comp+cell': 'pred_comp_cell', 'rich': 'pred_rich'}
TREE_TITLES = {'pred_composition': 'trees on element fractions', 'pred_comp_cell': 'trees on fractions + cell volume',
               'pred_rich': 'trees on element statistics + neighbour geometry'}


SPLIT_SPECS = {  # (network job glob, tree jobs) per split; elementP = leave-out-elements with element-property atom inputs
    'formula': ('struct_{t}_s[0-9]*', ['trees_{t}', 'trees_{t}_rich']),
    'system': ('struct_{t}_sys_s[0-9]*', ['trees_{t}_sys']),
    'element': ('struct_{t}_elem_s[0-9]*', ['trees_{t}_elem']),
    'elementP': ('struct_{t}_elemP_s[0-9]*', ['trees_{t}_elem']),
    'systemT': ('struct_{t}_sysT_s[0-9]*', ['trees_{t}_sysT']),   # held-out systems again, under the invariant neighbour rule
}


def rule_of_run(run_dir):
    """The neighbour-cut rule a crystal-network run was trained with ('first' for runs made before the rule was recorded)."""
    try:
        return json.loads((Path(run_dir) / 'run_config.json').read_text())['config'].get('neighbor_ties', 'first')
    except (OSError, ValueError, KeyError):
        return 'unknown'


def rule_of_trees(tree_file):
    """The rule behind a tree run's structure features: 'first' for files made before it was recorded, None if it built no such features."""
    try:
        tj = json.loads(Path(tree_file).with_name('trees.json').read_text())
    except (OSError, ValueError):
        return 'unknown'
    if 'rich' not in tj.get('feature_sets', {}):
        return None
    return tj.get('neighbor_ties') or 'first'


def analyze_structure(pipe, target, split='formula'):
    """Crystal networks against tree baselines on one split: 'formula', 'system', 'element' or 'elementP' (see SPLIT_SPECS).
    The comparator is the tree feature set with the lowest VALIDATION error, so the opponent is picked without the test rows."""
    pipe = Path(pipe)
    job_globs, tree_jobs = SPLIT_SPECS[split][0].format(t=target), [j.format(t=target) for j in SPLIT_SPECS[split][1]]
    tree_files = [p for p in (_first(pipe, j, 'trees_predictions.npz') for j in tree_jobs) if p is not None]
    run_dirs = []
    for job in sorted(glob.glob(str(pipe / 'jobs' / job_globs))):
        if not Path(job).name.rsplit('_s', 1)[-1].isdigit():
            continue   # only <job>_s<seed>: never a stray folder whose name merely starts like one
        done = [r.parent for r in map(Path, sorted(glob.glob(str(Path(job) / 'structure' / '*' / 'result.json')))) if (r.parent / 'predictions.npz').exists()]
        if done:
            run_dirs.append(done[-1])  # one run per seed job: the latest complete one, never two copies of the same seed
    if not tree_files or not run_dirs:
        return {'target': target, 'split': split, 'status': 'missing', 'reason': ('the tree baseline' if not tree_files else 'the network runs') + ' did not finish'}
    try:
        cmp_out, lines = compare(run_dirs, tree_files)
    except ValueError as e:
        return {'target': target, 'split': split, 'status': 'error', 'reason': str(e)}
    res = [json.loads((r / 'result.json').read_text()) for r in run_dirs]
    trees = {}
    for f in tree_files:
        for k, v in json.loads(f.with_name('trees.json').read_text())['feature_sets'].items():
            trees[k] = {'selected': v['selected'], 'val': v['val'], 'test': v['test'], 'n_features': v.get('n_features')}
    avail = [k for k in trees if TREE_KEYS.get(k) in cmp_out['comparisons']]
    strongest = min(avail, key=lambda k: trees[k]['val']['mae'])
    key = TREE_KEYS[strongest]
    out = {'target': target, 'split': split, 'status': 'ok', 'label': res[0]['target'], 'n_runs': len(run_dirs), 'device': res[0]['device'],
           'counts': res[0]['counts'], 'split_sizes': res[0]['split']['sizes'], 'split_groups': res[0]['split'].get('groups') or res[0]['split'].get('formulas'),
           'mean_baseline_test_mae': res[0]['baselines']['mean']['test']['mae'],
           'ridge_test_mae': res[0]['baselines']['ridge_element_fractions']['test']['mae'],
           'network_runs': [{'seed': r['config']['seed'], 'best_epoch': r['cgcnn']['best_epoch_by_val'], 'test': r['cgcnn']['test'], 'val': r['cgcnn']['val'],
                             'train_seconds': r['timing']['train_seconds'], 'structures_per_second': r['timing']['train_structures_per_second']} for r in res],
           'trees': trees, 'strongest_trees': strongest, 'strongest_trees_key': key,
           'neighbor_rules': {'networks': sorted({rule_of_run(r) for r in run_dirs}),
                              'trees': sorted({x for x in (rule_of_trees(f) for f in tree_files) if x})},
           'compare': cmp_out, 'compare_lines': lines}
    try:
        from gpu.structure.compare_structure import align
        tz, y, forms, preds = align(run_dirs, tree_files)
        out['fairness'] = fairness(y, forms, np.mean(preds, axis=0), preds, tz[key])
        out['fairness']['against'] = key
        if target in ('optb88vdw_bandgap', 'mbj_bandgap'):
            net, tr, zero = np.mean(preds, axis=0), tz[key], y == 0
            out['fairness']['by_zero_gap'] = {'share_exact_zero': float(zero.mean()),
                                              'network_mae_zero': float(np.abs(net - y)[zero].mean()) if zero.any() else None,
                                              'trees_mae_zero': float(np.abs(tr - y)[zero].mean()) if zero.any() else None,
                                              'network_mae_nonzero': float(np.abs(net - y)[~zero].mean()), 'trees_mae_nonzero': float(np.abs(tr - y)[~zero].mean())}
    except Exception as e:  # the headline comparison above still stands
        out['fairness_error'] = repr(e)
    for k, comps in cmp_out['comparisons'].items():
        avg = [v for kk, v in comps.items() if kk.startswith('network average')] or list(comps.values())[:1]
        out.setdefault('verdicts', {})[k] = {'diff': avg[0]['diff'], 'ci95': avg[0]['ci95'], 'verdict': verdict(*avg[0]['ci95'])}
    return out


def _job_rule(pipe, jid):
    """The rule of a network job's finished run, or None if it has none yet."""
    r = _runs(pipe, jid)
    return rule_of_run(r[0]) if r else None


def _runs(pipe, pattern):
    out = []
    for job in sorted(glob.glob(str(Path(pipe) / 'jobs' / pattern))):
        if not Path(job).name.rsplit('_s', 1)[-1].isdigit():
            continue
        done = [r.parent for r in map(Path, sorted(glob.glob(str(Path(job) / 'structure' / '*' / 'result.json')))) if (r.parent / 'predictions.npz').exists()]
        if done:
            out.append(done[-1])
    return out


def _vs(a_pred, b_pred, y, grp, a_name, b_name):
    c = _cmp(a_pred, b_pred, y, grp)
    return {**c, 'verdict': f'{a_name} clearly better' if c['ci95'][1] < 0 else f'{b_name} clearly better' if c['ci95'][0] > 0 else 'no clear difference'}


def _strongest_trees(tree_file, tz):
    tj = json.loads(tree_file.with_name('trees.json').read_text())['feature_sets']
    k = min((k for k in tj if TREE_KEYS.get(k) in tz.files), key=lambda k: tj[k]['val']['mae'])
    return k, tz[TREE_KEYS[k]]


def compare_run_sets(new_runs, old_runs, tree_file, new_name, old_name):
    """Two sets of network runs on the same test rows (each averaged over its runs) against each other and the strongest trees."""
    from gpu.structure.compare_structure import align
    tz, y, grp, new = align(new_runs, [tree_file])
    _, _, _, old = align(old_runs, [tree_file])
    strongest, tr = _strongest_trees(tree_file, tz)
    na, oa = np.mean(new, axis=0), np.mean(old, axis=0)
    rn, ro = sorted({rule_of_run(r) for r in new_runs}), sorted({rule_of_run(r) for r in old_runs})
    rt = rule_of_trees(tree_file)   # None: the trees used no structure features, so no rule is involved on their side
    return {'n_new_runs': len(new), 'n_old_runs': len(old), 'strongest_trees': strongest,
            'neighbor_rules': {'new': rn, 'old': ro, 'trees': rt}, 'mixed_neighbor_rules': rn != ro,
            'mixed_new_vs_trees': rt is not None and rn != [rt], 'mixed_old_vs_trees': rt is not None and ro != [rt],
            'new_single_mae_mean': float(np.mean([np.abs(p - y).mean() for p in new])), 'new_average_mae': float(np.abs(na - y).mean()),
            'old_single_mae_mean': float(np.mean([np.abs(p - y).mean() for p in old])), 'old_average_mae': float(np.abs(oa - y).mean()),
            'trees_mae': float(np.abs(tr - y).mean()),
            'new_vs_old': _vs(na, oa, y, grp, new_name, old_name), 'new_vs_trees': _vs(na, tr, y, grp, new_name, 'trees'),
            'old_vs_trees': _vs(oa, tr, y, grp, old_name, 'trees'),
            'single_new_vs_trees': [_vs(p, tr, y, grp, new_name, 'trees')['verdict'] for p in new]}


def analyze_finetune(pipe, plan):
    """Moduli networks that start from the formation-energy network, against the same networks trained from scratch on the same split."""
    out = []
    specs = (('formula', 'ft_{t}_s[0-9]*', 'struct_{t}_s[0-9]*', 'trees_{t}_rich', 'trees_{t}'), ('system', 'ft_sys_{t}_s[0-9]*', 'struct_{t}_sys_s[0-9]*', 'trees_{t}_sys', None),
             ('element', 'ft_elemP_{t}_s[0-9]*', 'struct_{t}_elemP_s[0-9]*', 'trees_{t}_elem', None))
    targets = sorted({j['target'] for j in plan['jobs'] if j.get('group') == 'finetune'})
    for t in targets:
        for split, ft_pat, scratch_pat, tree_job, tree_job2 in specs:
            ft, sc = _runs(pipe, ft_pat.format(t=t)), _runs(pipe, scratch_pat.format(t=t))
            tree_file = _first(pipe, tree_job.format(t=t), 'trees_predictions.npz') or (tree_job2 and _first(pipe, tree_job2.format(t=t), 'trees_predictions.npz'))
            if not any(j.get('group') == 'finetune' and j.get('target') == t and j.get('split') == split for j in plan['jobs']):
                continue
            e = {'target': t, 'split': split, 'status': 'missing', 'n_finetuned': len(ft), 'n_scratch': len(sc)}
            if ft and sc and tree_file is not None:
                try:
                    e.update(compare_run_sets(ft, sc, tree_file, 'fine-tuned', 'from scratch'))
                    e['status'] = 'ok'
                except Exception as ex:  # noqa: BLE001
                    e['status'], e['reason'] = 'error', repr(ex)
            out.append(e)
    return out


def analyze_evolution(pipe, plan):
    """Every search round (gpu/pipeline/evolve.py): the search curve, and the one fair test of whether the round helped: its winner
    retrained on fresh seeds against what it started from (the defaults, or the previous round's winner), on held-out systems."""
    from gpu.pipeline import evolve as E
    rounds = list(plan.get('cycle_history', [])) + ([plan['cycle']] if plan.get('cycle') else [])
    if not rounds:
        return None
    res = []
    for cyc in rounds:
        rnd = cyc.get('round')
        for t in cyc['targets']:
            base = (cyc.get('bases') or {}).get(t, E.DEFAULT)
            arch = E.archive(pipe, plan['jobs'], t, rnd)
            done = [x for x in arch if x['val_mae'] is not None]
            start = next((x for x in done if E.gkey(x['genes']) == E.gkey(base)), None)
            best = E.best_of(arch)
            gens = max([x['gen'] for x in arch] or [-1]) + 1
            e = {'target': t, 'round': rnd or 'r1', 'n_individuals': len(arch), 'n_finished': len(done), 'generations': gens,
                 'start_label': 'the defaults' if E.gkey(base) == E.gkey(E.DEFAULT) else 'the previous winner', 'start_changes': E.describe_genes(base),
                 'individuals': [{**x, 'changes': E.describe_genes(x['genes']), 'neighbor_rule': _job_rule(pipe, x['id'])} for x in sorted(arch, key=lambda x: (x['gen'], x['id']))],
                 'best_by_generation': [min([x['val_mae'] for x in done if x['gen'] <= g] or [None]) for g in range(gens)],
                 'default_val_mae': start['val_mae'] if start else None, 'best_val_mae': best['val_mae'] if best else None,
                 'best_id': best['id'] if best else None, 'best_changes': E.describe_genes(best['genes']) if best else None,
                 'default_won': bool(best and E.gkey(best['genes']) == E.gkey(base)), 'status': 'ok' if done else 'missing',
                 'final_epochs': cyc.get('final_epochs'), 'compare_label': cyc.get('compare_label', 'the default settings'),
                 'round_rule': cyc.get('neighbor_ties', 'first')}
            e['search_rules'] = sorted({x['neighbor_rule'] for x in e['individuals'] if x['neighbor_rule'] is not None})
            fjobs = [j for j in plan['jobs'] if j.get('group') == 'evolve_final' and j.get('round') == rnd and j.get('target') == t]
            if fjobs:  # what the finalists really are: their own stored settings, not whatever the archive's best is now
                e['finalist_changes'] = E.describe_genes(fjobs[0]['genes'])
                e['finalist_winner_id'] = fjobs[0].get('winner_id')
                e['finalist_differs_from_best'] = bool(best and E.gkey(fjobs[0]['genes']) != E.gkey(best['genes']))
            finals = _runs(pipe, (f'evof_{t}_s[0-9]*' if rnd is None else f'evof_{rnd}_{t}_s[0-9]*'))
            olds = _runs(pipe, cyc.get('compare_to', 'struct_{t}_sys_s[0-9]*').format(t=t))
            tree_file = _first(pipe, f'trees_{t}_sys', 'trees_predictions.npz')
            if finals and not (olds and tree_file is not None):
                e['finalists_error'] = ('no runs of ' + e['compare_label'] + ' in this folder to compare with' if not olds else 'no tree baseline on the held-out-systems split in this folder')
            if finals and olds and tree_file is not None:
                try:
                    c = compare_run_sets(finals, olds, tree_file, 'evolved', 'default' if rnd is None else 'previous winner')
                    e['finalists'] = {'n_evolved_runs': c['n_new_runs'], 'n_default_runs': c['n_old_runs'], 'strongest_trees': c['strongest_trees'],
                                      'evolved_single_mae_mean': c['new_single_mae_mean'], 'evolved_average_mae': c['new_average_mae'],
                                      'default_single_mae_mean': c['old_single_mae_mean'], 'default_average_mae': c['old_average_mae'], 'trees_mae': c['trees_mae'],
                                      'evolved_vs_default': c['new_vs_old'], 'evolved_vs_trees': c['new_vs_trees'], 'default_vs_trees': c['old_vs_trees'],
                                      'neighbor_rules': c['neighbor_rules'], 'mixed_neighbor_rules': c['mixed_neighbor_rules']}
                except Exception as ex:  # noqa: BLE001
                    e['finalists_error'] = repr(ex)
            res.append(e)
    return res


def analyze_tiefix(pipe, plan):
    """The neighbour-rule fix measured: the held-out-systems networks and trees trained again under the invariant rule, against the
    originals on the same test crystals (same seeds, settings and rows), and how much one network of each kind changes its predictions
    when a crystal is only rewritten (gpu/structure/invariance.py).
    Each side is chosen by the rule its runs RECORDED, not by job name: the original side keeps only runs made with 'first', the fixed side
    only runs made with 'all'; anything else is listed as excluded. If one side is left empty, the status says so and nothing is compared.
    The interval of "fixed minus original" resamples test crystals with the trained networks held fixed. It does not include which seeds
    happened to be trained, so the per-seed differences and the seed-to-seed spread are reported beside it."""
    pipe = Path(pipe)
    targets = []
    for j in plan['jobs']:
        if j.get('target') and fnmatch.fnmatch(j['id'], f'struct_{j["target"]}_sysT_s[0-9]*') and j['target'] not in targets:
            targets.append(j['target'])
    out = []
    seed = lambda r: json.loads((r / 'run_config.json').read_text())['config']['seed']  # noqa: E731
    for t in targets:
        new_all, old_all = _runs(pipe, f'struct_{t}_sysT_s[0-9]*'), _runs(pipe, f'struct_{t}_sys_s[0-9]*')
        new_ok = [r for r in new_all if rule_of_run(r) == 'all']
        old_ok = [r for r in old_all if rule_of_run(r) == 'first']
        excluded = [f'{r.parent.parent.name} ({rule_of_run(r)})' for r in new_all + old_all if r not in new_ok + old_ok]
        both = sorted({seed(r) for r in new_ok} & {seed(r) for r in old_ok})
        new = sorted((r for r in new_ok if seed(r) in both), key=seed)
        old = sorted((r for r in old_ok if seed(r) in both), key=seed)   # the same seeds on both sides, in the same order
        tf_new, tf_old = _first(pipe, f'trees_{t}_sysT', 'trees_predictions.npz'), _first(pipe, f'trees_{t}_sys', 'trees_predictions.npz')
        e = {'target': t, 'status': 'missing', 'seeds': both, 'n_new_runs_finished': len(new_all), 'n_old_runs_finished': len(old_all), 'excluded_runs': excluded}
        for rule in ('first', 'all'):
            f = pipe / 'jobs' / f'inv_{t}_{rule}' / 'invariance.json'
            try:
                e.setdefault('invariance', {})[rule] = json.loads(f.read_text())
            except (OSError, ValueError):
                pass
        if old_all and not old_ok:
            e['status'], e['reason'] = 'error', ('the held-out-systems runs of this folder were not made with the original rule (' + ', '.join(excluded)
                                                 + '), so there is no original set to compare with')
        elif new_all and not new_ok:
            e['status'], e['reason'] = 'error', 'the retrained runs did not record the fixed rule (' + ', '.join(excluded) + ')'
        elif new and old and (tf_new or tf_old) is not None:   # the network comparison needs a tree file only to align rows
            try:
                e.update(compare_run_sets(new, old, tf_new or tf_old, 'fixed rule', 'original rule'))
                e['status'] = 'ok'
                from gpu.structure.compare_structure import align
                _, y, _, pn_runs = align(new, [tf_new or tf_old])
                _, _, _, po_runs = align(old, [tf_new or tf_old])
                ps = [{'seed': seed(rn_), 'fixed_mae': float(np.abs(a_ - y).mean()), 'original_mae': float(np.abs(b_ - y).mean())} for rn_, a_, b_ in zip(new, pn_runs, po_runs)]
                for x in ps:
                    x['diff'] = x['fixed_mae'] - x['original_mae']
                e['per_seed'] = ps
                e['seeds_lower_fixed'] = sum(x['diff'] < 0 for x in ps)
                e['seed_sd'] = {'fixed': float(np.std([x['fixed_mae'] for x in ps], ddof=1)) if len(ps) > 1 else None,
                                'original': float(np.std([x['original_mae'] for x in ps], ddof=1)) if len(ps) > 1 else None}
                rn, ro = [json.loads((r / 'result.json').read_text()) for r in new], [json.loads((r / 'result.json').read_text()) for r in old]
                e['label'] = rn[0]['target']
                e['edges_ratio'] = float(np.mean([r['graphs']['edges'] for r in rn]) / np.mean([r['graphs']['edges'] for r in ro]))
                e['edges_per_atom_fixed'] = rn[0]['graphs'].get('edges_per_atom')
                e['train_seconds_ratio'] = float(np.mean([r['timing']['train_seconds'] for r in rn]) / np.mean([r['timing']['train_seconds'] for r in ro]))
                e['devices'] = sorted({r['device'] for r in rn + ro})
                rules_t = {'fixed': rule_of_trees(tf_new) if tf_new is not None else None, 'original': rule_of_trees(tf_old) if tf_old is not None else None}
                if tf_new is None or tf_old is None:
                    e['trees_missing'] = ' and '.join(x for x, f in ((f'the fixed-rule trees (trees_{t}_sysT)', tf_new), (f'the original trees (trees_{t}_sys)', tf_old)) if f is None) + ' did not finish'
                elif rules_t['fixed'] != 'all' or rules_t['original'] != 'first':
                    e['trees_missing'] = f'the tree files did not record the expected rules (fixed side {rules_t["fixed"]}, original side {rules_t["original"]})'
                else:
                    tzn, y, grp, _ = align(new[:1], [tf_new])
                    tzo, yo, _, _ = align(new[:1], [tf_old])       # raises if the old trees are on other rows
                    ix = np.array([{j: i for i, j in enumerate(tzo['jid'])}[j] for j in tzn['jid']])   # old rows in the new file's order
                    if not np.allclose(y, yo[ix]):
                        raise ValueError('the two tree runs disagree about the test labels')
                    kn, pn = _strongest_trees(tf_new, tzn)
                    ko, po = _strongest_trees(tf_old, tzo)
                    po = po[ix]
                    tjn = json.loads(tf_new.with_name('trees.json').read_text())['feature_sets']
                    tjo = json.loads(tf_old.with_name('trees.json').read_text())['feature_sets']
                    # The rule effect on the trees is the SAME feature set under both rules: only the structure features ('rich') depend on it.
                    # Each side's strongest set is shown too, but comparing two different sets would report a feature-set difference as a rule effect.
                    same = 'rich' if ('rich' in tjn and 'rich' in tjo and TREE_KEYS['rich'] in tzn.files and TREE_KEYS['rich'] in tzo.files) else None
                    e['trees'] = {'fixed_strongest': kn, 'original_strongest': ko, 'fixed_mae': float(np.abs(pn - y).mean()), 'original_mae': float(np.abs(po - y).mean()),
                                  'rules': rules_t, 'compared_set': same}
                    if same:
                        rn_p, ro_p = tzn[TREE_KEYS[same]], tzo[TREE_KEYS[same]][ix]
                        e['trees']['fixed_vs_original'] = _vs(rn_p, ro_p, y, grp, 'fixed rule', 'original rule')
                        e['trees']['same_set_mae'] = {'fixed': float(np.abs(rn_p - y).mean()), 'original': float(np.abs(ro_p - y).mean())}
            except Exception as ex:  # noqa: BLE001
                e['status'], e['reason'] = 'error', repr(ex)
        out.append(e)
    return out


# ---- prediction intervals ------------------------------------------------------------------------------------------------------
LEVELS = (0.8, 0.9, 0.95)


def _q(scores, level):
    """Split-conformal quantile: the ceil((n+1) * level)-th smallest score."""
    s = np.sort(np.asarray(scores, float))
    if not len(s):
        return float('nan')
    k = int(np.ceil((len(s) + 1) * level)) - 1
    return float(s[min(max(k, 0), len(s) - 1)])


def conformal_split(val_res, test_res, val_sigma=None, test_sigma=None):
    """Intervals pred +/- q, q from the validation residuals; coverage measured on the test rows. With sigma (the spread between
    seeds), also scaled intervals pred +/- q * sigma: narrower where the seeds agree, wider where they do not."""
    out = {}
    for lv in LEVELS:
        q = _q(np.abs(val_res), lv)
        r = {'half_width': q, 'coverage': float(np.mean(np.abs(test_res) <= q))}
        if val_sigma is not None and test_sigma is not None:
            eps = max(float(np.median(val_sigma)) * 0.1, 1e-9)
            qn = _q(np.abs(val_res) / (val_sigma + eps), lv)
            w = qn * (test_sigma + eps)
            r['scaled'] = {'mean_half_width': float(w.mean()), 'median_half_width': float(np.median(w)), 'coverage': float(np.mean(np.abs(test_res) <= w))}
        out[f'{lv:.2f}'] = r
    return out


def conformal_cv(res, folds):
    """Cross-conformal: each fold's interval comes from the OTHER folds' out-of-fold residuals; coverage is pooled over folds."""
    out = {}
    for lv in LEVELS:
        cov, width = [], []
        for f in range(int(folds.max()) + 1):
            q = _q(np.abs(res[folds != f]), lv)
            cov.append(np.abs(res[folds == f]) <= q)
            width.append(np.full((folds == f).sum(), q))
        out[f'{lv:.2f}'] = {'half_width': float(np.concatenate(width).mean()), 'coverage': float(np.concatenate(cov).mean())}
    return out


def _run_preds(run_dirs):
    """(val y, val ensemble pred, val sigma, test y, test ensemble pred, test sigma) for runs on the same split, aligned by record id."""
    zs = [np.load(Path(r) / 'predictions.npz') for r in run_dirs]
    vj, tj = zs[0]['val_jid'], zs[0]['test_jid']
    for z in zs[1:]:
        if not (np.array_equal(z['val_jid'], vj) and np.array_equal(z['test_jid'], tj)):
            raise ValueError('runs are on different rows')
    vp, tp = np.stack([z['val_pred'] for z in zs]), np.stack([z['test_pred'] for z in zs])
    sig = (vp.std(0), tp.std(0)) if len(zs) > 1 else (None, None)
    return zs[0]['val_y'], vp.mean(0), sig[0], zs[0]['test_y'], tp.mean(0), sig[1]


def analyze_intervals(pipe, plan, res):
    """90% (and 80/95%) prediction intervals for every task that has the data for them, with the coverage actually reached on rows
    the calibration never saw. Coverage near the nominal level means the intervals can be trusted on data like the test rows."""
    out = {'tabular': [], 'crystal': []}
    from gpu.data import load
    for t in res.get('tabular', []):
        if t.get('status') != 'ok':
            continue
        fam = t['family']
        try:
            sweep = _first(pipe, f'tab_{fam}', '*/*/result.json')
            base = _first(pipe, f'base_{fam}', 'baselines.json')
            ds = load(fam)
            P = np.load(sweep.parent / 'predictions.npz')
            B = np.load(base.with_name('baselines_predictions.npz'))
            bj = json.loads(base.read_text())
            names, groups_of = bj['model_names'], bj['groups']
            tree_idx = [i for i, n in enumerate(names) if groups_of[n] == 'trees']
            if bj['mode'] == 'cv':
                y, folds = ds.y, ds.folds
                net, _ = crossfit(P['cv_val'].mean(axis=1), y, folds)
                trees, _ = crossfit(B['oof'][tree_idx], y, folds)
                models = {'trees': trees, 'network average': net, 'blend': 0.5 * (net + trees)}
                out['tabular'].append({'task': fam, 'unit': t['target'], 'method': 'cross-conformal over the grouped folds',
                                       'models': {k: conformal_cv(v - y, folds) for k, v in models.items()}})
            else:
                sel, evl = 'test', next(iter(ds.extra))
                y_sel, y_ev = ds.y_test, ds.extra[evl][1]
                ens_sel, ens_ev = P[f'fixed_{sel}'].mean(axis=1), P[f'fixed_{evl}'].mean(axis=1)
                ci = int(np.argmin([np.abs(e - y_sel).mean() for e in ens_sel]))
                c = tree_idx[int(np.argmin([np.abs(B[f'pred_{sel}'][i] - y_sel).mean() for i in tree_idx]))]
                models = {'trees': (B[f'pred_{sel}'][c], B[f'pred_{evl}'][c]), 'network average': (ens_sel[ci], ens_ev[ci])}
                models['blend'] = (0.5 * (models['trees'][0] + models['network average'][0]), 0.5 * (models['trees'][1] + models['network average'][1]))
                out['tabular'].append({'task': fam, 'unit': t['target'], 'method': 'split conformal: calibrated on REWARD, checked on MONITOR (exploratory)',
                                       'models': {k: conformal_split(v[0] - y_sel, v[1] - y_ev) for k, v in models.items()}})
        except Exception as e:  # noqa: BLE001
            out['tabular'].append({'task': fam, 'error': repr(e)})
    for key, split in (('structure', 'formula'), ('structure_system', 'system'), ('structure_element', 'element'), ('structure_elementP', 'elementP'),
                       ('structure_systemT', 'systemT')):
        for s in res.get(key, []):
            if s.get('status') != 'ok':
                continue
            try:
                runs = _runs(pipe, SPLIT_SPECS[split][0].format(t=s['target']))
                vy, vp, vs, ty, tp, ts = _run_preds(runs)
                models = {'network average': conformal_split(vp - vy, tp - ty, vs, ts)}
                z0 = np.load(Path(runs[0]) / 'predictions.npz')
                for tj in SPLIT_SPECS[split][1]:  # trees and blend too, when the tree run saved validation predictions
                    tf = _first(pipe, tj.format(t=s['target']), 'trees_predictions.npz')
                    if tf is None:
                        continue
                    tz = np.load(tf)
                    key = 'val_' + s['strongest_trees_key']
                    if key not in tz.files or s['strongest_trees_key'] not in tz.files:
                        continue
                    vix = {j: i for i, j in enumerate(tz['val_jid'])}
                    tix = {j: i for i, j in enumerate(tz['jid'])}
                    tv = tz[key][[vix[j] for j in z0['val_jid']]]
                    tt = tz[s['strongest_trees_key']][[tix[j] for j in z0['test_jid']]]
                    models['trees'] = conformal_split(tv - vy, tt - ty)
                    models['blend'] = conformal_split(0.5 * (vp + tv) - vy, 0.5 * (tp + tt) - ty)
                    break
                out['crystal'].append({'task': s['target'], 'split': split, 'n_runs': len(runs), 'unit': s['label'], 'neighbor_rules': sorted({rule_of_run(r) for r in runs}),
                                       'method': 'split conformal on the validation crystals (same kind of hold-out as test)', 'models': models})
            except Exception as e:  # noqa: BLE001
                out['crystal'].append({'task': s['target'], 'split': split, 'error': repr(e)})
    return out


def analyze_all(pipe, plan):
    order = [j['target'] for j in plan['jobs'] if j.get('target')]
    fams = [j['family'] for j in plan['jobs'] if j['id'].startswith('tab_')]
    res = {'tabular': [], 'structure': []}
    for f in fams:
        try:
            res['tabular'].append(analyze_tabular(pipe, f))
        except Exception as e:  # one broken family must not stop the report
            res['tabular'].append({'family': f, 'status': 'error', 'reason': repr(e)})
    keys = {'formula': 'structure', 'system': 'structure_system', 'element': 'structure_element', 'elementP': 'structure_elementP', 'systemT': 'structure_systemT'}
    for split, key in keys.items():
        pat = SPLIT_SPECS[split][0]
        targets = sorted({j['target'] for j in plan['jobs'] if j.get('target') and fnmatch.fnmatch(j['id'], pat.format(t=j['target']))}, key=order.index)
        res[key] = []
        for t in targets:
            try:
                res[key].append(analyze_structure(pipe, t, split))
            except Exception as e:
                res[key].append({'target': t, 'split': split, 'status': 'error', 'reason': repr(e)})
    try:
        res['finetune'] = analyze_finetune(pipe, plan)
    except Exception as e:  # noqa: BLE001
        res['finetune'] = [{'status': 'error', 'target': '-', 'reason': repr(e)}]
    try:
        res['evolution'] = analyze_evolution(pipe, plan)
    except Exception as e:  # noqa: BLE001
        res['evolution'] = [{'status': 'error', 'target': '-', 'reason': repr(e)}]
    try:
        res['tiefix'] = analyze_tiefix(pipe, plan)
    except Exception as e:  # noqa: BLE001
        res['tiefix'] = [{'status': 'error', 'target': '-', 'reason': repr(e)}]
    try:
        res['intervals'] = analyze_intervals(pipe, plan, res)
    except Exception as e:  # noqa: BLE001
        res['intervals'] = {'error': repr(e)}
    return res
