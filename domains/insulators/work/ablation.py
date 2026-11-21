#!/usr/bin/env python3
"""Ablation by 3-fold grouped CV on TRAINING rows only (no test rows): how much of the tree
model's skill depends on the relaxed-cell volume, which is itself an output of a DFT relaxation.
Targets are DFT-calculated dielectric constants, not measurements.
Usage: python3 -I ablation.py <work_dir>
"""
import json, sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
W = sys.argv[1]
d = json.load(open(f"{W}/nn_data.json")); folds = json.load(open(f"{W}/cv_folds.json"))['folds']
X = np.array(d['train']['X']); y = np.array(d['train']['y']); F = d['features']
vol = {'volume_per_atom', 'density_g_cm3', 'packing_fraction_covalent'}
cell = vol | {'nsites'} | {f for f in F if f.startswith('system_')}
sets = {'all_42_features': F, 'without_volume_derived_3': [f for f in F if f not in vol],
        'composition_only_no_cell_info': [f for f in F if f not in cell],
        'volume_and_cell_only_no_composition': [f for f in F if f in cell]}
out = {"note": "3-fold grouped CV on training rows only; gradient boosting with the fixed baseline settings; MAE in log10 units. Targets are DFT-calculated, not measured."}
for name, keep in sets.items():
    cols = [F.index(f) for f in keep]; oof = np.zeros(len(y))
    for f in folds:
        f = np.array(f); m = np.ones(len(y), bool); m[f] = False
        g = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
        oof[f] = g.fit(X[m][:, cols], y[m]).predict(X[f][:, cols])
    out[name] = round(float(np.abs(oof - y).mean()), 4)
json.dump(out, open(f"{W}/ablation.json", 'w')); print(json.dumps(out))
