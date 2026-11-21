"""CPU baselines for a tabular family, on exactly the same rows and folds as gpu/train_tabular.py.

  python -m gpu.baselines_tabular --family alloys --threads 2 --output gpu/runs/baselines/alloys.json

A family with grouped CV folds (everything except concrete) is scored by CV: every row gets one out-of-fold prediction
from each model, trained on the other folds. Concrete (train on ages 1-7 d, score 14-28 d) uses its fixed split: fit on
TRAIN, predict REWARD ('test') and MONITOR. Models: the training mean; ridge (5 strengths); extra trees (4 settings); the
project's reference gradient boosting; histogram gradient boosting (4 settings); and, for concrete only, ridge on ln(age).

Writes <output> (metrics per model) and <output stem>_predictions.npz (the predictions, for paired comparisons in
gpu/pipeline/analyze.py). Refuses to overwrite either file. Never reads sealed data: everything goes through gpu.data.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from gpu.common import git_info, hardware, now_iso, write_json_new
from gpu.data import load


def metrics(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    e = p - y
    sst = float(((y - y.mean()) ** 2).sum())
    return {'n': int(len(y)), 'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt((e ** 2).mean())),
            'r2': float(1 - (e ** 2).sum() / sst) if sst > 0 else None}


class MeanModel:
    def fit(self, X, y):
        self.m = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


def model_zoo(ds, threads, seed):
    """name -> (group, factory). Groups: mean, linear, trees."""
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    zoo = {'mean': ('mean', MeanModel)}
    for a in (0.01, 0.1, 1.0, 10.0, 100.0):
        zoo[f'ridge_alpha{a:g}'] = ('linear', lambda a=a: make_pipeline(StandardScaler(), Ridge(alpha=a)))
    if 'age' in ds.features:  # concrete: how much does knowing only the age (and nothing about the mix) explain?
        ia = ds.features.index('age')
        only = FunctionTransformer(lambda X, ia=ia: np.log(np.clip(np.asarray(X)[:, [ia]], 1e-9, None)))
        plus = FunctionTransformer(lambda X, ia=ia: np.hstack([np.asarray(X), np.log(np.clip(np.asarray(X)[:, [ia]], 1e-9, None))]))
        zoo['ridge_ln_age_only'] = ('linear', lambda: make_pipeline(only, Ridge(alpha=1e-6)))
        zoo['ridge_all_plus_ln_age'] = ('linear', lambda: make_pipeline(plus, StandardScaler(), Ridge(alpha=1.0)))
    for leaf in (1, 3):
        for mf in (0.5, 1.0):
            zoo[f'extratrees_leaf{leaf}_mf{mf:g}'] = ('trees', lambda leaf=leaf, mf=mf: ExtraTreesRegressor(
                n_estimators=300, min_samples_leaf=leaf, max_features=mf, n_jobs=threads, random_state=seed))
    zoo['gbr_reference'] = ('trees', lambda: GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42))
    for lr in (0.05, 0.1):
        for leaves in (15, 31):
            zoo[f'histgb_lr{lr:g}_leaves{leaves}'] = ('trees', lambda lr=lr, leaves=leaves: HistGradientBoostingRegressor(
                learning_rate=lr, max_leaf_nodes=leaves, max_iter=300, early_stopping=False, random_state=seed))
    return zoo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--family', required=True)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--output', required=True, help='new JSON path (refused if it exists)')
    a = ap.parse_args(argv)
    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out_path.with_name(out_path.stem + '_predictions.npz')
    for p in (out_path, npz_path):
        if p.exists():
            sys.exit(f'refusing to overwrite {p}')
    ds = load(a.family)
    zoo = model_zoo(ds, a.threads, a.seed)
    names = list(zoo)
    fixed = a.family == 'concrete' or ds.folds is None
    t_all = time.perf_counter()
    result = {'family': a.family, 'mode': 'fixed' if fixed else 'cv', 'data_sha256': ds.data_sha256, 'data': ds.summary(),
              'model_names': names, 'groups': {n: zoo[n][0] for n in names}, 'created': now_iso(), 'git': git_info(),
              'hardware': hardware(), 'models': {}}
    arrays = {'model_names': np.array(names)}
    secs = {n: 0.0 for n in names}
    if not fixed:
        K = int(ds.folds.max()) + 1
        oof = np.full((len(names), len(ds.y)), np.nan)
        for f in range(K):
            tr, va = ds.folds != f, ds.folds == f
            for i, n in enumerate(names):
                t0 = time.perf_counter()
                oof[i, va] = zoo[n][1]().fit(ds.X[tr], ds.y[tr]).predict(ds.X[va])
                secs[n] += time.perf_counter() - t0
            print(f'  fold {f + 1}/{K} done ({time.perf_counter() - t_all:.0f} s)', flush=True)
        arrays['oof'] = oof
        for i, n in enumerate(names):
            result['models'][n] = {'group': zoo[n][0], 'cv': metrics(oof[i], ds.y), 'fit_seconds': secs[n]}
        order = sorted(names, key=lambda n: result['models'][n]['cv']['mae'])
        for n in order[:5]:
            m = result['models'][n]['cv']
            print(f'  {n:28s} CV MAE {m["mae"]:.4g}  R2 {m["r2"]:.3f}')
    else:
        sets = {'test': (ds.X_test, ds.y_test), **{k: (v[0], v[1]) for k, v in ds.extra.items()}}
        preds = {k: np.full((len(names), len(v[1])), np.nan) for k, v in sets.items()}
        for i, n in enumerate(names):
            t0 = time.perf_counter()
            m = zoo[n][1]().fit(ds.X, ds.y)
            for k, (X, _) in sets.items():
                preds[k][i] = m.predict(X)
            secs[n] = time.perf_counter() - t0
            result['models'][n] = {'group': zoo[n][0], 'fit_seconds': secs[n], **{k: metrics(preds[k][i], sets[k][1]) for k in sets}}
        for k in sets:
            arrays[f'pred_{k}'] = preds[k]
        result['note'] = 'fixed split: fitted on the training rows; "test" is the family\'s selection set (concrete: REWARD), others (MONITOR) are reported only'
        order = sorted(names, key=lambda n: result['models'][n]['test']['mae'])
        for n in order[:5]:
            print(f'  {n:28s} ' + '  '.join(f'{k} MAE {result["models"][n][k]["mae"]:.4g}' for k in sets))
    result['wall_seconds'] = time.perf_counter() - t_all
    np.savez_compressed(npz_path, **arrays)
    write_json_new(out_path, result)
    print(f'{a.family}: {len(names)} baseline models in {result["wall_seconds"]:.0f} s; wrote {out_path}')


if __name__ == '__main__':
    main()
