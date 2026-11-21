"""Gradient-boosted trees baseline for the lab (run with python3 -I).

fit:     python3 -I lab/trees.py fit      -> trains on lab/data/train.json with the site's settings
         (n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42) on the nine
         features, saves lab/results/models/trees.pkl once (never overwritten) and prints its SHA-256.
predict: python3 -I lab/trees.py predict <sha256>  -> reads {"X": [...]} on stdin, checks the pickle's
         hash, writes predictions as JSON to stdout. It never opens any file but the pickle and the data tiers.
"""
import hashlib, json, os, pickle, sys

LAB = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(LAB, "results", "models", "trees.pkl")
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fit():
    from sklearn.ensemble import GradientBoostingRegressor
    import numpy as np
    with open(os.path.join(LAB, "data", "train.json")) as f:
        train = json.load(f)
    model = GradientBoostingRegressor(**PARAMS).fit(np.array(train["X"]), np.array(train["y"]))
    blob = pickle.dumps(model, protocol=4)
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    if os.path.exists(MODEL):
        with open(MODEL, "rb") as f:
            old = pickle.load(f)
        with open(os.path.join(LAB, "data", "select.json")) as f:
            X = np.array(json.load(f)["X"])
        if not np.array_equal(old.predict(X), model.predict(X)):
            sys.exit("trees.pkl exists and predicts differently from a fresh fit; refusing to overwrite")
    else:
        with open(MODEL, "xb") as f:  # 'x': fails if the file exists
            f.write(blob)
    import sklearn
    print(json.dumps({"sha256": sha(MODEL), "params": PARAMS, "sklearn": sklearn.__version__}))


def predict(expect):
    import numpy as np
    if sha(MODEL) != expect:
        sys.exit("trees.pkl SHA-256 does not match")
    with open(MODEL, "rb") as f:
        model = pickle.load(f)
    rows = json.load(sys.stdin)["X"]
    print(json.dumps([float(v) for v in model.predict(np.array(rows))]))


if __name__ == "__main__":
    fit() if sys.argv[1] == "fit" else predict(sys.argv[2])
