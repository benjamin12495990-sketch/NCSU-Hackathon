# Re-runs the independent check's tuned-tree comparison (skeptic_check.json: my_trees_tuned).
# Usage: python3 -I trees_tuned.py <all_rows.json> <out.json>
# Settings are chosen by 3-fold grouped CV on the main training rows only (the same cv_fold as
# nn_run.js); the chosen setting is then fitted once on all training rows and predicts the test rows.
import json, sys, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
a = json.load(open(sys.argv[1]))
X = np.array(a['X']); y = np.array(a['y']); tr = np.array(a['main_train']); te = np.array(a['main_test']); fold = np.array(a['cv_fold'])
GRID = [(300, 3, .1), (300, 4, .05), (300, 6, .05), (300, 6, .1), (600, 4, .05), (600, 4, .1), (600, 6, .03), (600, 6, .05)]
mk = lambda n, d, lr: GradientBoostingRegressor(n_estimators=n, max_depth=d, learning_rate=lr, subsample=0.8, random_state=42)
Xt, yt = X[tr], y[tr]
def fold_pred(cfg, f):
    m = mk(*cfg).fit(Xt[fold != f], yt[fold != f]); return cfg, f, m.predict(Xt[fold == f])
t0 = time.time()
jobs = Parallel(n_jobs=2)(delayed(fold_pred)(c, f) for c in GRID for f in range(3))
table = []
for c in GRID:
    oof = np.zeros(len(tr))
    for cc, f, p in jobs:
        if cc == c: oof[fold == f] = p
    table.append({'n_estimators': c[0], 'max_depth': c[1], 'learning_rate': c[2], 'cv_mae': round(float(np.abs(oof - yt).mean()), 4)})
best = min(table, key=lambda r: r['cv_mae'])
m = mk(best['n_estimators'], best['max_depth'], best['learning_rate']).fit(Xt, yt)
pred = m.predict(X[te])
out = {'grid_cv_3fold_grouped': table, 'chosen': best, 'test_pred': pred.tolist(),
       'test_mae': round(float(np.abs(pred - y[te]).mean()), 4), 'seconds': round(time.time() - t0, 1)}
json.dump(out, open(sys.argv[2], 'w')); print('tuned trees', best, out['test_mae'], out['seconds'], 's')
