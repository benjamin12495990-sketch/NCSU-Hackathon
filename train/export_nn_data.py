"""Export the train and test rows for the in-browser neural network.

Run:  python train/export_nn_data.py
It reuses train.py's loading, features and split, so the network in the page
learns from exactly the rows the trees learned from and is scored on exactly
the held-out mixes the trees are scored on. Writes public/model/nn_data.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

df = T.load()
X, y = T.featurize(df), df["strength"].to_numpy(dtype=np.float64)
groups = T.mix_groups(df)
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=T.SEED).split(X, y, groups))
metrics = json.loads((T.OUT / "metrics.json").read_text())
assert (len(tr), len(te)) == (metrics["train_rows"], metrics["test_rows"]), "split differs from the trained model's"

def rows(idx):
    return {"X": [[round(float(v), 4) for v in X[i]] for i in idx], "y": [round(float(y[i]), 2) for i in idx]}

train = rows(tr)
# Mix id per training row, so settings can be chosen by cross-validation that
# keeps whole mixes together, using training rows only.
train["g"] = [int(groups[i]) for i in tr]
test = rows(te)
test["g"] = [int(groups[i]) for i in te]
out = {"features": T.FEATURES, "train": train, "test": test,
       "note": "Same rows and same held-out mixes as train/train.py (split by mix, seed 42)."}
(T.OUT / "nn_data.json").write_text(json.dumps(out, separators=(",", ":")))
print(f"train {len(tr)} rows, test {len(te)} rows, {(T.OUT / 'nn_data.json').stat().st_size / 1e3:.0f} kB")
