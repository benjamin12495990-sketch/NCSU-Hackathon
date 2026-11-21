"""Paired comparison of crystal-graph network runs against tree baselines on the same test rows.

  python -m gpu.structure.compare_structure --runs gpu/runs/structure/<run> [<run> ...] --trees gpu/runs/structure/trees_x_predictions.npz

Each run folder needs predictions.npz (written by train_structure.py, or by predict_run.py for older runs); --trees is the
file written next to baseline_trees.py's JSON. Rows are matched by record id. The test split is grouped by reduced formula, so
the bootstrap resamples whole formulas (2,000 draws, fixed seed): the difference is MAE(network) - MAE(trees), so a negative
number favours the network. With several runs it also scores the average of their predictions (a seed ensemble).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from gpu.common import write_json_new

B = 2000


def _load_run(path):
    z = np.load(Path(path) / 'predictions.npz')
    return {'jid': z['test_jid'], 'formula': z['test_formula'], 'y': z['test_y'], 'pred': z['test_pred']}


def _paired(err_a, err_b, formulas):
    """Bootstrap over formulas of MAE(a) - MAE(b); returns (diff, lo, hi, share of formulas where a is better)."""
    names, inv = np.unique(formulas, return_inverse=True)
    n = len(names)
    cnt = np.bincount(inv, minlength=n).astype(float)
    sa, sb = np.bincount(inv, weights=err_a, minlength=n), np.bincount(inv, weights=err_b, minlength=n)
    rng = np.random.default_rng(909)
    draws = rng.integers(0, n, size=(B, n))
    diffs = (sa[draws].sum(1) - sb[draws].sum(1)) / cnt[draws].sum(1)
    d = (sa.sum() - sb.sum()) / cnt.sum()
    better = float(np.mean(sa / cnt < sb / cnt))
    return float(d), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), better


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--trees', required=True)
    ap.add_argument('--output', help='optional new JSON path')
    a = ap.parse_args(argv)
    t = np.load(a.trees)
    runs = [_load_run(r) for r in a.runs]
    order = {j: i for i, j in enumerate(t['jid'])}
    for r, path in zip(runs, a.runs):
        miss = [j for j in r['jid'] if j not in order]
        if miss or len(r['jid']) != len(t['jid']):
            sys.exit(f'{path}: test rows do not match the tree baseline (different data, split or --limit)')
    base = runs[0]
    idx = [np.array([order[j] for j in r['jid']]) for r in runs]
    y = t['y']
    preds = [np.empty(len(y)) for _ in runs]
    for p, r, ix in zip(preds, runs, idx):
        p[ix] = r['pred']
    formulas = np.empty(len(y), dtype=object)
    formulas[idx[0]] = base['formula']
    models = {f'network run {i + 1} ({Path(path).name})': p for i, (p, path) in enumerate(zip(preds, a.runs))}
    if len(runs) > 1:
        models[f'network average of {len(runs)} runs'] = np.mean(preds, axis=0)
    out = {'test_rows': int(len(y)), 'formulas': int(len(set(formulas))), 'bootstrap': f'{B} resamples of whole formulas, seed 909', 'comparisons': {}}
    maes = [float(np.abs(p - y).mean()) for p in preds]
    if len(runs) > 1:
        out['network_single_run_mae'] = {'mean': float(np.mean(maes)), 'sd': float(np.std(maes, ddof=1)), 'runs': maes}
        print(f'network single-run test MAE over {len(runs)} runs: mean {np.mean(maes):.4f}, sd {np.std(maes, ddof=1):.4f}')
    print(f'{len(y)} test rows, {len(set(formulas))} formulas; difference = MAE(network) - MAE(trees); negative favours the network\n')
    for key in [k for k in t.files if k.startswith('pred_')]:
        tp = t[key]
        tmae = float(np.abs(tp - y).mean())
        print(f'{key}: trees MAE {tmae:.4f}')
        for name, p in models.items():
            d, lo, hi, share = _paired(np.abs(p - y), np.abs(tp - y), formulas)
            verdict = 'network better' if hi < 0 else 'trees better' if lo > 0 else 'not distinguishable'
            print(f'  {name}: MAE {np.abs(p - y).mean():.4f}  diff {d:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  network better on {share:.0%} of formulas  -> {verdict}')
            out['comparisons'].setdefault(key, {})[name] = {'network_mae': float(np.abs(p - y).mean()), 'trees_mae': tmae, 'diff': d, 'ci95': [lo, hi],
                                                             'share_formulas_network_better': share, 'verdict': verdict}
    if a.output:
        write_json_new(a.output, out)
        print(f'\nwrote {a.output}')


if __name__ == '__main__':
    main()
