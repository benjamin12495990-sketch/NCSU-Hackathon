# Tree baseline with fixed settings. Usage: python3 -I trees.py <all_rows.json> <splits.json> <out.json>
# Writes predictions only; metrics are computed later in evaluate.py.
import json, sys, time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
a = json.load(open(sys.argv[1])); sp = json.load(open(sys.argv[2]))['splits']
X = np.array(a['X']); y = np.array(a['y']); tr = np.array(a['main_train']); te = np.array(a['main_test']); fold = np.array(a['cv_fold'])
mk = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
t0 = time.time(); m = mk().fit(X[tr], y[tr]); sec = time.time() - t0
out = {'train_seconds': sec, 'pred': m.predict(X[te]).tolist(), 'importances': dict(zip(a['features'], m.feature_importances_.tolist()))}
oof = np.zeros(len(tr))
for f in range(3):
    oof[fold == f] = mk().fit(X[tr][fold != f], y[tr][fold != f]).predict(X[tr][fold == f])
out['oof'] = oof.tolist()
out['splits'] = [{'seed': s['seed'], 'pred': mk().fit(X[s['train']], y[s['train']]).predict(X[s['test']]).tolist()} for s in sp]
json.dump(out, open(sys.argv[3], 'w'))
print('trees done', round(time.time() - t0, 1), 's')
