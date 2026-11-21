"""Tree baseline on a splits file, with the same settings as train/train.py.
Usage: python3 scripts/experiments/trees_on_splits.py splits.json"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
d = json.load(open(str(Path(__file__).resolve().parents[2] / 'public/model/nn_data.json')))
X = np.array(d['train']['X'] + d['test']['X'], dtype=float); y = np.array(d['train']['y'] + d['test']['y'], dtype=float)
P = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
out = []
for mask in json.load(open(sys.argv[1])):
    m = np.array(mask, dtype=bool)
    out.append(float(np.mean(np.abs(GradientBoostingRegressor(**P).fit(X[~m], y[~m]).predict(X[m]) - y[m]))))
print(json.dumps({"per_split": [round(v, 3) for v in out], "mean": round(float(np.mean(out)), 3)}))
