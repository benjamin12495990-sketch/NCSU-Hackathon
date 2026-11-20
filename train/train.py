"""Train the concrete strength model and export everything the browser needs.

Run once:  python train/train.py
Data: UCI "Concrete Compressive Strength" (Yeh, 1998). 1,030 lab results:
seven ingredient amounts in kg per cubic metre, the age in days, and the
measured compressive strength in MPa.
Outputs go to public/model/: features.json, model.json (expected strength),
model_low.json (cautious lower estimate), metrics.json, known.json,
space.json, parity.json, test_points.json.
"""
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "train" / "data" / "concrete.csv"
OUT = ROOT / "public" / "model"
URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
INGREDIENTS = ["cement", "slag", "flyash", "water", "superplasticizer", "coarse", "fine"]
FEATURES = INGREDIENTS + ["age", "water_binder_ratio"]
SEED = 42
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=SEED)


def load():
    if not CSV.exists():  # rebuild the CSV from the original UCI download
        with urlopen(URL) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        raw = pd.read_excel(io.BytesIO(z.read("Concrete_Data.xls")))
        assert raw.shape[1] == 9, raw.columns
        raw.columns = INGREDIENTS + ["age", "strength"]
        CSV.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(CSV, index=False)
    df = pd.read_csv(CSV)
    assert list(df.columns) == INGREDIENTS + ["age", "strength"], df.columns
    return df.drop_duplicates().reset_index(drop=True)  # 25 rows are exact repeats


def featurize(df):
    """Seven ingredients, age, and water divided by total binder."""
    binder = df["cement"] + df["slag"] + df["flyash"]
    wb = df["water"] / binder
    return np.column_stack([df[INGREDIENTS + ["age"]].to_numpy(dtype=np.float64), wb.to_numpy()])


def export_trees(model):
    trees = []
    for est in model.estimators_[:, 0]:
        t = est.tree_
        trees.append({
            "f": t.feature.tolist(),               # feature index, -2 marks a leaf
            "t": [float(v) for v in t.threshold],  # go left when x <= t
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "v": [float(v) for v in t.value[:, 0, 0]],
        })
    return {"init": float(model.init_.constant_.ravel()[0]),
            "learning_rate": float(model.learning_rate), "trees": trees}


def export_predict(exported, row):
    """Python copy of the browser's tree walk, used for the parity file."""
    row32 = row.astype(np.float32)  # scikit-learn compares features as float32
    total = exported["init"]
    for t in exported["trees"]:
        n = 0
        while t["f"][n] != -2:
            n = t["l"][n] if row32[t["f"][n]] <= t["t"][n] else t["r"][n]
        total += exported["learning_rate"] * t["v"][n]
    return total


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    (OUT / "features.json").write_text(json.dumps({"ingredients": INGREDIENTS, "features": FEATURES}))
    X, y = featurize(df), df["strength"].to_numpy(dtype=np.float64)

    # The same mix appears at several ages. Splitting by MIX keeps every age of
    # a test mix out of training, so the test error is for mixes never seen.
    # The file also holds the same mix twice with amounts rounded differently,
    # so mixes are identified by their amounts rounded to the nearest kg.
    rounded = df[INGREDIENTS].round(0)
    mix_id = rounded.groupby(INGREDIENTS, sort=False).ngroup().to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, mix_id))
    model = GradientBoostingRegressor(**PARAMS).fit(X[tr], y[tr])
    pred = model.predict(X[te])
    mae, r2 = mean_absolute_error(y[te], pred), r2_score(y[te], pred)
    rmse = float(np.sqrt(np.mean((y[te] - pred) ** 2)))

    # A second model for caution. Instead of the expected strength it learns the
    # 10th percentile: a value that about 9 in 10 lab results should beat. The
    # designer only keeps a mix when this lower estimate reaches the target.
    low_model = GradientBoostingRegressor(loss="quantile", alpha=0.1, **PARAMS).fit(X[tr], y[tr])
    low_pred = low_model.predict(X[te])
    coverage = float(np.mean(y[te] >= low_pred))  # how often that held on test data

    # For comparison only: the easier random split that leaks mixes across the split.
    rtr, rte = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)
    leaky_mae = mean_absolute_error(y[rte], GradientBoostingRegressor(**PARAMS).fit(X[rtr], y[rtr]).predict(X[rte]))

    exported, exported_low = export_trees(model), export_trees(low_model)
    (OUT / "model.json").write_text(json.dumps(exported, separators=(",", ":")))
    (OUT / "model_low.json").write_text(json.dumps(exported_low, separators=(",", ":")))

    # Search space for the mix generator: stay inside what the labs actually made.
    mixes = df.groupby(mix_id)[INGREDIENTS].mean().to_numpy(dtype=np.float64)
    binder = mixes[:, 0] + mixes[:, 1] + mixes[:, 2]
    lo, hi = mixes.min(axis=0), mixes.max(axis=0)
    space = {
        "ranges": {n: [float(lo[i]), float(hi[i])] for i, n in enumerate(INGREDIENTS)},
        "limits": {
            "binder": [float(binder.min()), float(binder.max())],
            "water_binder_ratio": [float(np.percentile(mixes[:, 3] / binder, 1)), float(np.percentile(mixes[:, 3] / binder, 99))],
            "slag_fraction_max": float(np.percentile(mixes[:, 1] / binder, 99)),
            "flyash_fraction_max": float(np.percentile(mixes[:, 2] / binder, 99)),
            "scm_fraction_max": float(np.percentile((mixes[:, 1] + mixes[:, 2]) / binder, 99)),
            "superplasticizer_per_binder_max": float(np.percentile(mixes[:, 4] / binder, 99)),
        },
    }
    (OUT / "space.json").write_text(json.dumps(space, indent=1))

    # Low-confidence threshold: distance from each test mix to its nearest
    # training mix, each ingredient scaled by its range; take the 90th percentile.
    scale = hi - lo
    def distinct(idx):
        return df.iloc[idx].groupby(mix_id[idx])[INGREDIENTS].mean().to_numpy(dtype=np.float64) / scale
    a, b = distinct(te), distinct(tr)
    nearest = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    threshold = float(np.percentile(nearest, 90))

    # Every distinct mix in the dataset with its measured strength at each age.
    known = []
    for _, g in df.groupby(mix_id, sort=False):
        ages = g.groupby("age")["strength"].mean()
        known.append({"m": [round(float(v), 1) for v in g[INGREDIENTS].mean()],
                      "s": [[int(a_), round(float(s_), 2)] for a_, s_ in ages.items()]})
    (OUT / "known.json").write_text(json.dumps(known, separators=(",", ":")))

    rng = np.random.default_rng(SEED)
    parity = [{"mix": [float(v) for v in df.iloc[i][INGREDIENTS]], "age": int(df["age"][i]),
               "features": [float(v) for v in X[i]], "prediction": export_predict(exported, X[i]),
               "prediction_low": export_predict(exported_low, X[i])}
              for i in rng.choice(te, size=20, replace=False)]
    (OUT / "parity.json").write_text(json.dumps(parity))
    drift = max(max(abs(p["prediction"] - model.predict(np.array([p["features"]]))[0]),
                    abs(p["prediction_low"] - low_model.predict(np.array([p["features"]]))[0])) for p in parity)

    (OUT / "test_points.json").write_text(json.dumps(
        [[round(float(m), 2), round(float(p), 2)] for m, p in zip(y[te], pred)], separators=(",", ":")))

    metrics = {
        "test_mae_MPa": round(float(mae), 2),
        "test_rmse_MPa": round(rmse, 2),
        "test_r2": round(float(r2), 3),
        "lower_estimate_coverage": round(coverage, 3),
        "lower_estimate_mean_gap_MPa": round(float(np.mean(pred - low_pred)), 2),
        "random_split_mae_MPa": round(float(leaky_mae), 2),
        "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "train_mixes": int(len(np.unique(mix_id[tr]))), "test_mixes": int(len(np.unique(mix_id[te]))),
        "known_mixes": len(known),
        "low_confidence_distance": threshold,
        "model": "GradientBoostingRegressor(300 trees, max_depth 4, learning_rate 0.05)",
        "lower_model": "the same settings with quantile loss, alpha 0.1 (10th percentile)",
        "split": "80/20 split by mix: every age of a test mix is held out of training",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "UCI Concrete Compressive Strength (Yeh, 1998)",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))
    print(json.dumps(space["limits"], indent=1))
    print(f"export vs scikit-learn, largest difference on parity rows: {drift:.2e} MPa")
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:18s} {f.stat().st_size / 1e3:8.1f} kB")


if __name__ == "__main__":
    main()
