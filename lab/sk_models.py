"""Phase-2 scikit-learn baselines, trained on lab/data/train.json only (run with python3 -I).

fit <name>             trains one model and saves lab/results/models2/<name>.pkl once (never overwritten:
                       an existing file is accepted only if it predicts REWARD identically); prints its SHA-256.
predict <name> <sha>   reads {"X": [...]} on stdin, checks the pickle's hash, prints predictions as JSON.
Models (no tuning on any selection rows):
  gbt    GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
  rf     RandomForestRegressor(random_state=42), scikit-learn defaults otherwise
  ridge  StandardScaler + Ridge(alpha=1.0) on the nine features with age replaced by ln(age)
"""
import hashlib, json, os, pickle, sys

LAB = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(LAB, "results", "models2")


def build(name):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "gbt":
        return GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
    if name == "rf":
        return RandomForestRegressor(random_state=42)
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    raise SystemExit("unknown model " + name)


def features(name, X):
    import numpy as np
    X = np.array(X, dtype=float)
    if name == "ridge":
        X = X.copy()
        X[:, 7] = np.log(X[:, 7])
    return X


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fit(name):
    import numpy as np, sklearn
    with open(os.path.join(LAB, "data", "train.json")) as f:
        train = json.load(f)
    model = build(name).fit(features(name, train["X"]), np.array(train["y"]))
    path = os.path.join(DIR, name + ".pkl")
    os.makedirs(DIR, exist_ok=True)
    if os.path.exists(path):
        with open(path, "rb") as f:
            old = pickle.load(f)
        with open(os.path.join(LAB, "data", "reward.json")) as f:
            X = features(name, json.load(f)["X"])
        if not np.array_equal(old.predict(X), model.predict(X)):
            sys.exit(name + ".pkl exists and predicts differently from a fresh fit; refusing to overwrite")
    else:
        with open(path, "xb") as f:
            f.write(pickle.dumps(model, protocol=4))
    print(json.dumps({"sha256": sha(path), "file": "results/models2/" + name + ".pkl", "model": repr(model), "sklearn": sklearn.__version__}))


def predict(name, expect):
    path = os.path.join(DIR, name + ".pkl")
    if sha(path) != expect:
        sys.exit(name + ".pkl SHA-256 does not match")
    with open(path, "rb") as f:
        model = pickle.load(f)
    rows = json.load(sys.stdin)["X"]
    print(json.dumps([float(v) for v in model.predict(features(name, rows))]))


if __name__ == "__main__":
    fit(sys.argv[2]) if sys.argv[1] == "fit" else predict(sys.argv[2], sys.argv[3])
