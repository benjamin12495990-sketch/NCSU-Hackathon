#!/usr/bin/env python3
"""Tree baseline for the insulators model. Targets are DFT-CALCULATED, not measured.
Usage: python3 -I trees.py <work_dir>   (reads nn_data*.json, splits.json, all_rows.json, cv_folds.json)
"""
import json, sys, time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

W = sys.argv[1]
SWEEP = [0.5 + 0.75 * i for i in range(9)]  # band-gap sweep grid in eV, 0.5 .. 6.5 (same grid in run_nn.js)


def gbr():
    return GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)


def load(name):
    d = json.load(open(f"{W}/{name}"))
    return (np.array(d['train']['X']), np.array(d['train']['y']), np.array(d['test']['X']), np.array(d['test']['y']), d['features'])


out = {"note": "Targets are DFT-calculated dielectric constants, not lab measurements. Predictions are log10(poly_total)."}
Xtr, ytr, Xte, yte, feats = load('nn_data.json')
t = time.time(); m = gbr().fit(Xtr, ytr); out['fit_seconds_main'] = round(time.time() - t, 2)
out['test_pred_main'] = m.predict(Xte).tolist()
out['naive_train_mean'] = float(ytr.mean())
imp = sorted(zip(feats, m.feature_importances_), key=lambda kv: -kv[1])[:8]
out['top_importances_main'] = [[k, round(float(v), 4)] for k, v in imp]

# 3-fold grouped CV on training rows (reference only; the tree settings are fixed, not tuned)
folds = json.load(open(f"{W}/cv_folds.json"))['folds']
oof = np.zeros(len(ytr))
for f in folds:
    f = np.array(f); mask = np.ones(len(ytr), bool); mask[f] = False
    oof[f] = gbr().fit(Xtr[mask], ytr[mask]).predict(Xtr[f])
out['cv_mae_log10_main'] = float(np.abs(oof - ytr).mean())
naive_oof = np.zeros(len(ytr))
for f in folds:
    f = np.array(f); mask = np.ones(len(ytr), bool); mask[f] = False
    naive_oof[f] = ytr[mask].mean()
out['cv_mae_log10_naive'] = float(np.abs(naive_oof - ytr).mean())

# second model: same inputs plus calculated band gap
Xtrb, ytrb, Xteb, yteb, featsb = load('nn_data_bandgap.json')
mb = gbr().fit(Xtrb, ytrb)
out['test_pred_bandgap'] = mb.predict(Xteb).tolist()
out['band_gap_importance'] = float(mb.feature_importances_[-1])
out['train_band_gap_quantiles_5_50_95'] = [float(v) for v in np.quantile(Xtrb[:, -1], [0.05, 0.5, 0.95])]
sweep = []
for v in SWEEP:
    Xs = Xteb.copy(); Xs[:, -1] = v
    sweep.append(mb.predict(Xs).tolist())
out['sweep_grid_eV'] = SWEEP
out['sweep_pred_bandgap'] = sweep  # [grid point][test row]

# eight repeated grouped splits
allr = json.load(open(f"{W}/all_rows.json")); X = np.array(allr['X']); y = np.array(allr['y'])
res = []
for s in json.load(open(f"{W}/splits.json"))['splits']:
    tr, te = np.array(s['train']), np.array(s['test'])
    p = gbr().fit(X[tr], y[tr]).predict(X[te])
    res.append({"seed": s['seed'], "mae_log10_trees": float(np.abs(p - y[te]).mean()),
                "mae_log10_naive": float(np.abs(y[tr].mean() - y[te]).mean())})
out['splits'] = res
json.dump(out, open(f"{W}/trees.json", 'w'))
print(json.dumps({k: out[k] for k in ['fit_seconds_main', 'cv_mae_log10_main', 'cv_mae_log10_naive', 'band_gap_importance', 'top_importances_main', 'train_band_gap_quantiles_5_50_95']}))
print([round(r['mae_log10_trees'], 4) for r in res])
