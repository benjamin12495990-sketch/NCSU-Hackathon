# Reads the AqSolDB curated CSV (untrusted data, read as plain text only) and
# writes grouped train/test splits. Usage: python3 -I prepare.py <csv> <outdir>
import csv, json, math, sys
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

csv_path, out = sys.argv[1], sys.argv[2]
FEATURES = ['MolWt', 'MolLogP', 'MolMR', 'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
            'NumRotatableBonds', 'NumValenceElectrons', 'NumAromaticRings', 'NumSaturatedRings',
            'NumAliphaticRings', 'RingCount', 'TPSA', 'LabuteASA', 'BalabanJ', 'BertzCT']
EXPECT = ['ID', 'Name', 'InChI', 'InChIKey', 'SMILES', 'Solubility', 'SD', 'Occurrences', 'Group'] + FEATURES
with open(csv_path, newline='', encoding='utf-8') as f:
    rd = csv.reader(f)
    header = next(rd)
    assert header == EXPECT, header
    rows = list(rd)
ix = {h: i for i, h in enumerate(header)}
X, y, g, lab, ids, dropped = [], [], [], [], [], []
for r in rows:
    try:
        xs = [float(r[ix[c]]) for c in FEATURES]; t = float(r[ix['Solubility']])
    except ValueError:
        dropped.append([r[0], 'unparseable number']); continue
    key = r[ix['InChIKey']]
    if not all(math.isfinite(v) for v in xs) or not math.isfinite(t):
        dropped.append([r[0], 'non-finite value']); continue
    if len(key) < 14:
        dropped.append([r[0], 'short InChIKey']); continue
    X.append(xs); y.append(t); g.append(key[:14]); lab.append(r[ix['Group']]); ids.append(r[0])
X = np.array(X); y = np.array(y); g = np.array(g)
n = len(y)
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, g))
assert not set(g[tr]) & set(g[te])
folds = np.zeros(len(tr), dtype=int)
for k, (_, va) in enumerate(GroupKFold(n_splits=3).split(X[tr], y[tr], g[tr])):
    folds[va] = k
part = lambda idx: {'X': X[idx].tolist(), 'y': y[idx].tolist(), 'g': g[idx].tolist()}
json.dump({'features': FEATURES, 'train': part(tr), 'test': part(te)}, open(out + '/nn_data.json', 'w'))
splits = []
for seed in range(1001, 1009):
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, g))
    assert not set(g[a]) & set(g[b])
    splits.append({'seed': seed, 'train': a.tolist(), 'test': b.tolist()})
json.dump({'note': 'row indices into all_rows.json (cleaned CSV order); grouped 80/20 by InChIKey first block', 'splits': splits},
          open(out + '/splits.json', 'w'))
json.dump({'features': FEATURES, 'X': X.tolist(), 'y': y.tolist(), 'g': g.tolist(), 'label': lab, 'id': ids,
           'main_train': tr.tolist(), 'main_test': te.tolist(), 'cv_fold': folds.tolist()}, open(out + '/all_rows.json', 'w'))
from collections import Counter
info = {'csv_rows': len(rows), 'rows_used': n, 'dropped': dropped, 'groups': int(len(set(g))),
        'train_rows': int(len(tr)), 'test_rows': int(len(te)), 'train_groups': int(len(set(g[tr]))), 'test_groups': int(len(set(g[te]))),
        'groups_with_more_than_one_row': int(sum(1 for c in Counter(g.tolist()).values() if c > 1)),
        'label_counts_all': dict(Counter(lab)), 'label_counts_test': dict(Counter(np.array(lab)[te].tolist())),
        'cv_fold_sizes': np.bincount(folds).tolist(),
        'feature_min': dict(zip(FEATURES, X.min(0).tolist())), 'feature_max': dict(zip(FEATURES, X.max(0).tolist())),
        'balabanj_zero_rows': int((X[:, FEATURES.index('BalabanJ')] == 0).sum()),
        'y_min': float(y.min()), 'y_max': float(y.max()), 'y_mean': float(y.mean()), 'y_sd': float(y.std())}
json.dump(info, open(out + '/prepare_info.json', 'w'), indent=1)
print(json.dumps(info))
