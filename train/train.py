"""Train the Tc model and export everything the browser needs as JSON.

Run once:  python train/train.py
Data: UCI "Superconductivty Data" (Hamidieh, 2018), file unique_m.csv.
Outputs go to public/model/: elements.json, model.json, metrics.json,
known.json, parity.json.
"""
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "train" / "data" / "unique_m.csv"
OUT = ROOT / "public" / "model"
URL = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
N_TREES, MAX_DEPTH, SEED = 100, 8, 42


def load():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        DATA.write_bytes(z.read("unique_m.csv"))
    df = pd.read_csv(DATA)
    assert "critical_temp" in df.columns and "material" in df.columns, df.columns
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()

    # The element order is defined once, here, and both languages read the file.
    elements = [c for c in df.columns if c not in ("critical_temp", "material")]
    (OUT / "elements.json").write_text(json.dumps(elements))
    elements = json.loads((OUT / "elements.json").read_text())

    counts = df[elements].to_numpy(dtype=np.float64)
    keep = counts.sum(axis=1) > 0
    df, counts = df[keep].reset_index(drop=True), counts[keep]
    frac = counts / counts.sum(axis=1, keepdims=True)
    n_distinct = (counts > 0).sum(axis=1, keepdims=True).astype(np.float64)
    X = np.hstack([frac, n_distinct])  # features: fractions + distinct count
    y = df["critical_temp"].to_numpy(dtype=np.float64)

    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)
    model = RandomForestRegressor(
        n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=SEED, n_jobs=-1
    )
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    mae, r2 = mean_absolute_error(y[te], pred), r2_score(y[te], pred)

    # Low-confidence threshold: 90th percentile of the distance from each
    # test material to its nearest training material (fraction vectors only).
    nn = NearestNeighbors(n_neighbors=1).fit(frac[tr])
    dist, _ = nn.kneighbors(frac[te])
    threshold = float(np.percentile(dist[:, 0], 90))

    trees = []
    for est in model.estimators_:
        t = est.tree_
        trees.append({
            "f": t.feature.tolist(),              # feature index, -2 = leaf
            "t": [float(v) for v in t.threshold],  # go left if x <= t
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "v": [round(float(v), 4) for v in t.value[:, 0, 0]],
        })
    (OUT / "model.json").write_text(json.dumps({"trees": trees}, separators=(",", ":")))

    # Round-trip check: predictions rebuilt from the exported (rounded) leaves.
    def export_predict(row):
        row32 = row.astype(np.float32)  # sklearn compares in float32
        total = 0.0
        for tr_ in trees:
            n = 0
            while tr_["f"][n] != -2:
                n = tr_["l"][n] if row32[tr_["f"][n]] <= tr_["t"][n] else tr_["r"][n]
            total += tr_["v"][n]
        return total / len(trees)

    # Known materials: one entry per distinct composition, mean measured Tc.
    known = {}
    for i in range(len(df)):
        nz = np.nonzero(frac[i])[0]
        key = "|".join(f"{j}:{frac[i, j]:.4f}" for j in nz)
        e = known.setdefault(key, {"m": str(df["material"][i]), "tc": [],
                                   "c": [x for j in nz for x in (int(j), round(float(frac[i, j]), 4))]})
        e["tc"].append(y[i])
    known_list = [[e["m"], round(float(np.mean(e["tc"])), 2), e["c"]] for e in known.values()]
    (OUT / "known.json").write_text(json.dumps(known_list, separators=(",", ":")))

    rng = np.random.default_rng(SEED)
    parity = []
    for i in rng.choice(te, size=20, replace=False):
        parity.append({"formula": str(df["material"][i]),
                       "features": [float(v) for v in X[i]],
                       "prediction": export_predict(X[i])})
    (OUT / "parity.json").write_text(json.dumps(parity))
    drift = max(abs(p["prediction"] - model.predict(np.array([p["features"]]))[0]) for p in parity)

    metrics = {
        "test_mae_K": round(float(mae), 3),
        "test_r2": round(float(r2), 4),
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "distinct_known_compositions": len(known_list),
        "low_confidence_distance": threshold,
        "model": f"RandomForestRegressor({N_TREES} trees, max_depth {MAX_DEPTH})",
        "split": "random 80/20; near-duplicate compositions can sit on both sides, so the MAE is optimistic",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "UCI Superconductivty Data (Hamidieh, 2018), unique_m.csv",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))

    print(json.dumps(metrics, indent=1))
    print(f"export vs sklearn max drift on parity rows: {drift:.6f} K")
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:16s} {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
