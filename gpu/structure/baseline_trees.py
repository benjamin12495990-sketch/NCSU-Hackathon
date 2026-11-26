"""Tree baselines for a JARVIS-DFT target (default bulk modulus), on the same records, target and split as train_structure.py.

  python -m gpu.structure.baseline_trees --output gpu/runs/structure/trees_<tag>.json

The crystal-graph network is only worth its cost if it beats strong simple models, and the ridge baseline in
train_structure.py is weak (every other family in this project showed tuned trees well ahead of linear models).
Three feature sets (--feature-sets, default the first two):
  composition  = element fractions (the ridge baseline's features)
  comp+cell    = composition plus volume per atom and atoms per cell (a little structure, no graph)
  rich         = comp+cell plus element-property statistics and neighbour-geometry statistics (gpu/structure/features.py):
                 the strongest tree baseline here, and the fair opponent for a crystal-graph network
--split formula|system picks the split exactly as train_structure.py does.
Settings are chosen on the validation rows only; the test rows are scored once, for the selected setting.
Refuses to overwrite --output. CPU only.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpu.common import hardware, now_iso, write_json_new
from gpu.structure.train_structure import DEFAULT_ZIP, SPLIT_MODES, SPLIT_RULES, TARGETS, element_fractions, group_of, load_records, metrics, parts_of


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(DEFAULT_ZIP))
    ap.add_argument('--target', default='bulk_modulus_kv', choices=sorted(TARGETS))
    ap.add_argument('--split', default='formula', choices=SPLIT_MODES)
    ap.add_argument('--feature-sets', default='composition,comp+cell', help='comma list from: composition, comp+cell, rich')
    ap.add_argument('--neighbor-ties', choices=['all', 'first'], default='all',
                    help="neighbour-cut rule for the rich feature set: 'all' (invariant, default) or 'first' (the original rule)")
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--limit', type=int, default=0, help='first N kept records (same rule as train_structure.py), for small tests')
    ap.add_argument('--output', required=True, help='new JSON path (refused if it exists)')
    a = ap.parse_args(argv)
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

    recs, source, counts = load_records(SimpleNamespace(synthetic=0, data=a.data, limit=a.limit, seed=a.seed, target=a.target))
    parts = parts_of(recs, a.split)
    idx = {p: np.where(parts == p)[0] for p in ('train', 'val', 'test')}
    y = np.array([r['y'] for r in recs])
    F = np.stack([element_fractions(r['atoms']) for r in recs])
    cell = np.array([[abs(np.linalg.det(np.array(r['atoms']['lattice_mat']))) / len(r['atoms']['elements']), len(r['atoms']['elements'])] for r in recs])
    wanted = [x.strip() for x in a.feature_sets.split(',') if x.strip()]
    bad = [x for x in wanted if x not in ('composition', 'comp+cell', 'rich')]
    if bad or not wanted:
        sys.exit(f'--feature-sets: unknown {bad or "(empty)"}; choose from composition, comp+cell, rich')
    sets = {'composition': F, 'comp+cell': np.hstack([F, cell])}
    if 'rich' in wanted:
        from gpu.structure.features import build as build_rich
        sets['rich'] = np.hstack([F, cell, build_rich(recs, threads=a.threads, ties=a.neighbor_ties)])
    sets = {k: v for k, v in sets.items() if k in wanted}
    models = {
        'extratrees_300_leaf1': lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=1, max_features=0.5, n_jobs=a.threads, random_state=a.seed),
        'extratrees_300_leaf2': lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.5, n_jobs=a.threads, random_state=a.seed),
        'histgb_lr0.1': lambda: HistGradientBoostingRegressor(learning_rate=0.1, max_iter=400, early_stopping=False, random_state=a.seed),
        'histgb_lr0.05': lambda: HistGradientBoostingRegressor(learning_rate=0.05, max_iter=800, early_stopping=False, random_state=a.seed),
    }
    pred_path = Path(a.output).with_name(Path(a.output).stem + '_predictions.npz')
    if pred_path.exists():
        sys.exit(f'{pred_path} exists; refusing to overwrite')
    test_preds = {}
    ytr = y[idx['train']]
    out = {'task': TARGETS[a.target]['label'] + ' (DFT-calculated)', 'target': a.target, 'counts': counts, 'source': {k: v for k, v in source.items() if k != 'zip'},
           'split_sizes': {p: int(len(v)) for p, v in idx.items()}, 'split_mode': a.split, 'split_rule': SPLIT_RULES[a.split],
           'neighbor_ties': a.neighbor_ties if 'rich' in wanted else None,
           'mean': {p: metrics(np.full(len(idx[p]), ytr.mean()), y[idx[p]]) for p in ('val', 'test')}, 'feature_sets': {}, 'created': now_iso(),
           'hardware': hardware(), 'selection': 'by validation MAE within each feature set; test scored once for the selected model'}
    for sname, X in sets.items():
        rows = {}
        for mname, make in models.items():
            t0 = time.perf_counter()
            m = make()
            if sname == 'rich' and isinstance(m, ExtraTreesRegressor):
                m.set_params(max_features=0.33)  # ~190 columns: fewer per split keeps the forest fast
            m = m.fit(X[idx['train']], ytr)
            v = metrics(m.predict(X[idx['val']]), y[idx['val']])
            rows[mname] = {'val': v, 'seconds': round(time.perf_counter() - t0, 1), '_model': m}
            print(f'{sname:12s} {mname:22s} val MAE {v["mae"]:.4f}  {rows[mname]["seconds"]:.0f} s', flush=True)
        best = min(rows, key=lambda k: rows[k]['val']['mae'])
        test = metrics(rows[best]['_model'].predict(X[idx['test']]), y[idx['test']])
        test_preds['pred_' + sname.replace('+', '_')] = rows[best]['_model'].predict(X[idx['test']])
        test_preds['val_pred_' + sname.replace('+', '_')] = rows[best]['_model'].predict(X[idx['val']])  # for calibrating intervals
        out['feature_sets'][sname] = {'n_features': int(X.shape[1]), 'selected': best, 'val': rows[best]['val'], 'test': test,
                                      'all_val_mae': {k: v['val']['mae'] for k, v in rows.items()}}
        print(f'  -> {sname}: selected {best}; test MAE {test["mae"]:.4f}, R2 {test["r2"]:.3f}', flush=True)
    write_json_new(a.output, out)
    np.savez_compressed(pred_path, jid=np.array([recs[i]['jid'] for i in idx['test']]), formula=np.array([recs[i]['formula'] for i in idx['test']]),
                        y=y[idx['test']], group=np.array([group_of(recs[i], a.split) for i in idx['test']]),
                        val_jid=np.array([recs[i]['jid'] for i in idx['val']]), val_y=y[idx['val']], **test_preds)
    print(f'mean baseline test MAE {out["mean"]["test"]["mae"]:.4f}; wrote {a.output}')


if __name__ == '__main__':
    main()
