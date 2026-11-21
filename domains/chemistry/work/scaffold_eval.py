# Usage: python3 -I scaffold_eval.py <all_rows.json> <scaffold_split.json> <nn_splits.json> <out.json>
import json, sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
a = json.load(open(sys.argv[1])); sf = json.load(open(sys.argv[2])); s = sf['splits'][0]; nn = json.load(open(sys.argv[3]))[0]
X = np.array(a['X']); y = np.array(a['y']); tr, te = np.array(s['train']), np.array(s['test'])
def met(p, t):
    e = np.asarray(p) - t
    return {'n': int(len(t)), 'mae': round(float(np.abs(e).mean()), 4), 'rmse': round(float(np.sqrt((e ** 2).mean())), 4), 'r2': round(float(1 - (e ** 2).sum() / ((t - t.mean()) ** 2).sum()), 4)}
gb = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42).fit(X[tr], y[tr])
out = {'split': 'Bemis-Murcko scaffold of the largest fragment (RDKit), largest scaffold sets in train until 80%', 'train_rows': int(len(tr)), 'test_rows': int(len(te)),
       'scaffolds': sf['scaffolds'], 'acyclic_rows_all_in_train': sf['acyclic_rows_all_in_train'], 'smiles_failed': sf['smiles_failed'],
       'inchikey_groups_on_both_sides': sf['inchikey_groups_on_both_sides'],
       'network_ensemble': met(nn['pred'], y[te]), 'trees_gbr': met(gb.predict(X[te]), y[te]), 'naive_train_mean': met(np.full(len(te), y[tr].mean()), y[te]),
       'network_train_seconds': nn['sec'], 'note': 'network settings are the ones chosen by CV on the main training rows; one run, no tuning on this split'}
json.dump(out, open(sys.argv[4], 'w'), indent=1); print(json.dumps(out))
