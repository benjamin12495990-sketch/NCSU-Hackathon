"""Alternative training pipeline: a neural network instead of gradient-boosted trees.

The live app (train/train.py) uses a GradientBoostingRegressor. This is a drop-in
*alternative* that learns the same two things with a small multilayer perceptron:

  - an EXPECTED-strength model: a ReLU MLP trained with mean-squared error;
  - a CAUTIOUS lower estimate: the expected model minus a conformal offset, chosen
    on a held-out calibration set so a target share of lab results clear it
    (split-conformal prediction -- an alternative to GBR's quantile loss; on the
    current split it reaches about 81% coverage, short of its 90% target).

It reuses the project's data rule: the same 9 features, the same 25 repeated rows
dropped, and the same grouping BY MIX (recipes within 5 kg on every ingredient are
one mix; every age of a test mix is held out). The two scripts draw their test
mixes separately and this one also sets aside a calibration set, so the two
models are NOT scored on the same test set: compare the numbers loosely.

Pure NumPy -- no scikit-learn, no PyTorch. The network is just matmuls, so the
exported weights (pipeline/out/<task>/nn_model.json) can run in the browser the
same way the trees do, as an alternative inference path.

Run:  python3 pipeline/train_nn.py
Outputs go to pipeline/out/<task>/ and NEVER to public/model/, so this pipeline
does not disturb the app's own exported models.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ---- Config (domain-agnostic: point these at any tabular dataset) -----------
CONFIG = {
    "task": "concrete",
    "csv": ROOT / "train" / "data" / "concrete.csv",
    "ingredients": ["cement", "slag", "flyash", "water", "superplasticizer", "coarse", "fine"],
    "extra_features": ["age"],
    # derived feature: numerator / sum(denominator)
    "ratios": [{"name": "water_binder_ratio", "num": "water", "den": ["cement", "slag", "flyash"]}],
    "target": "strength",
    # rows whose group_by columns agree within group_tolerance on every column
    # count as one entity (one mix); the split keeps every row of a test entity
    # out of training. Same rule as mix_groups() in train/train.py.
    "group_by": ["cement", "slag", "flyash", "water", "superplasticizer", "coarse", "fine"],
    "group_tolerance": 5.0,
    "hidden": [64, 32],
    "target_coverage": 0.90,   # share of lab results the cautious estimate should clear
    "epochs": 1500,
    "batch_size": 64,
    "lr": 3e-3,
    "seed": 42,
}
OUT = ROOT / "pipeline" / "out" / CONFIG["task"]


# ---- Data -------------------------------------------------------------------
def same_entity_groups(recipes, tol):
    """One id per entity. The data repeats some mixes with amounts rounded
    differently, so recipes within `tol` on every column are merged."""
    uniq, inverse = np.unique(recipes, axis=0, return_inverse=True)
    parent = list(range(len(uniq)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    close = np.abs(uniq[:, None, :] - uniq[None, :, :]).max(axis=2) <= tol
    for i, j in zip(*np.nonzero(np.triu(close, 1))):
        parent[find(int(i))] = find(int(j))
    roots = np.array([find(i) for i in range(len(uniq))])
    return roots[np.ravel(inverse)]


def load(cfg):
    with open(cfg["csv"]) as f:
        rows = list(csv.DictReader(f))
    # drop exact repeated rows, as train/train.py does (25 in the concrete data)
    seen, unique = set(), []
    for r in rows:
        key = tuple(float(v) for v in r.values())
        if key not in seen:
            seen.add(key)
            unique.append(r)
    rows = unique
    cols = {c: np.array([float(r[c]) for r in rows]) for c in rows[0].keys()}
    feat_names = list(cfg["ingredients"]) + list(cfg["extra_features"])
    base = [cols[c] for c in feat_names]
    for ratio in cfg["ratios"]:
        den = sum(cols[d] for d in ratio["den"])
        base.append(cols[ratio["num"]] / den)
        feat_names.append(ratio["name"])
    X = np.column_stack(base).astype(np.float64)
    y = cols[cfg["target"]].astype(np.float64)
    group = same_entity_groups(np.column_stack([cols[c] for c in cfg["group_by"]]), cfg["group_tolerance"])
    return X, y, group.astype(int), feat_names


def split_by_group(group, test_frac, rng):
    """Hold out whole entities (mixes), so no mix appears in both splits."""
    groups = np.unique(group)
    rng.shuffle(groups)
    n_test = int(round(len(groups) * test_frac))
    test_groups = set(groups[:n_test].tolist())
    te = np.array([i for i, g in enumerate(group) if g in test_groups])
    tr = np.array([i for i, g in enumerate(group) if g not in test_groups])
    return tr, te


# ---- A tiny MLP in NumPy (He init, ReLU, Adam) ------------------------------
class MLP:
    def __init__(self, sizes, seed):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for nin, nout in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.standard_normal((nin, nout)) * np.sqrt(2.0 / nin))
            self.b.append(np.zeros(nout))
        self.params = self.W + self.b
        self._m = [np.zeros_like(p) for p in self.params]
        self._v = [np.zeros_like(p) for p in self.params]
        self._t = 0

    def forward(self, X):
        self.acts, a = [X], X
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            a = np.maximum(z, 0.0) if i < len(self.W) - 1 else z  # linear output
            self.acts.append(a)
        return a.ravel()

    def backward(self, grad_out):
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.b)
        delta = grad_out[:, None]
        for i in reversed(range(len(self.W))):
            grads_W[i] = self.acts[i].T @ delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (self.acts[i] > 0)  # ReLU grad
        return grads_W + grads_b

    def step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for i, g in enumerate(grads):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * (g * g)
            mhat = self._m[i] / (1 - b1 ** self._t)
            vhat = self._v[i] / (1 - b2 ** self._t)
            self.params[i] -= lr * mhat / (np.sqrt(vhat) + eps)


def train(net, X, y, epochs, batch_size, lr, seed):
    rng = np.random.default_rng(seed)
    n = len(X)
    for e in range(epochs):
        lr_e = lr * (0.1 + 0.9 * (1 - e / epochs))  # decay so the fit settles
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            pred = net.forward(X[idx])
            grad = (2.0 * (pred - y[idx])) / len(idx)   # mean-squared-error gradient
            net.step(net.backward(grad), lr_e)


def fit_mlp(X, y, sizes, cfg, seed):
    """Standardize on the given data, train, return (net, standardization)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    ym, ys = y.mean(), y.std() + 1e-9
    net = MLP(sizes, seed)
    train(net, (X - mu) / sd, (y - ym) / ys, cfg["epochs"], cfg["batch_size"], cfg["lr"], seed)
    return net, (mu, sd, ym, ys)


def predict(net, std, X):
    mu, sd, ym, ys = std
    return net.forward((X - mu) / sd) * ys + ym


# ---- Main -------------------------------------------------------------------
def main():
    cfg = CONFIG
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    X, y, group, feat_names = load(cfg)
    sizes = [X.shape[1]] + cfg["hidden"] + [1]

    # 1) hold out whole test mixes
    tr, te = split_by_group(group, 0.2, rng)
    # 2) inside training, hold out a calibration set of whole mixes for conformal
    sub_fit, sub_cal = split_by_group(group[tr], 0.2, rng)
    fit, cal = tr[sub_fit], tr[sub_cal]

    # expected-strength network, trained on the fit set only
    net, std = fit_mlp(X[fit], y[fit], sizes, cfg, cfg["seed"])
    pred = predict(net, std, X[te])

    # conformal offset: how much predictions tend to exceed truth on calibration.
    # cautious = expected - offset gives ~target_coverage of y >= cautious.
    over = predict(net, std, X[cal]) - y[cal]
    offset = float(np.percentile(over, cfg["target_coverage"] * 100))
    pred_low = pred - offset

    mae = float(np.mean(np.abs(y[te] - pred)))
    rmse = float(np.sqrt(np.mean((y[te] - pred) ** 2)))
    r2 = 1 - float(np.sum((y[te] - pred) ** 2)) / float(np.sum((y[te] - y[te].mean()) ** 2))
    coverage = float(np.mean(y[te] >= pred_low))

    # context: the easier random split that leaks mixes across train/test
    rtr, rte = split_by_group(np.arange(len(X)), 0.2, rng)
    netR, stdR = fit_mlp(X[rtr], y[rtr], sizes, cfg, cfg["seed"])
    leaky_mae = float(np.mean(np.abs(y[rte] - predict(netR, stdR, X[rte]))))

    metrics = {
        "test_mae_MPa": round(mae, 2),
        "test_rmse_MPa": round(rmse, 2),
        "test_r2": round(r2, 3),
        "lower_estimate_coverage": round(coverage, 3),
        "lower_estimate_mean_gap_MPa": round(offset, 2),
        "conformal_offset_MPa": round(offset, 2),
        "target_coverage": cfg["target_coverage"],
        "random_split_mae_MPa": round(leaky_mae, 2),
        "fit_rows": int(len(fit)), "calibration_rows": int(len(cal)), "test_rows": int(len(te)),
        "fit_mixes": int(len(np.unique(group[fit]))),
        "calibration_mixes": int(len(np.unique(group[cal]))),
        "test_mixes": int(len(np.unique(group[te]))),
        "model": f"NeuralNet MLP {sizes} ReLU, Adam, {cfg['epochs']} epochs (pure NumPy)",
        "lower_model": f"expected minus split-conformal offset for {int(cfg['target_coverage']*100)}% coverage",
        "split": "80/20 by mix (test); training further split 80/20 by mix for conformal calibration",
        "features": feat_names,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": cfg["csv"].name,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))

    # browser-runnable export: standardization + weights (plain matrices) + offset
    mu, sd, ym, ys = std
    (OUT / "nn_model.json").write_text(json.dumps({
        "features": feat_names,
        "x_mean": mu.tolist(), "x_std": sd.tolist(), "y_mean": ym, "y_std": ys,
        "weights": {"W": [w.tolist() for w in net.W], "b": [bb.tolist() for bb in net.b]},
        "hidden_activation": "relu",
        "cautious_offset": offset,  # cautious strength = expected - cautious_offset
    }, separators=(",", ":")))
    (OUT / "test_points.json").write_text(json.dumps(
        [[round(float(a), 2), round(float(b), 2)] for a, b in zip(y[te], pred)], separators=(",", ":")))

    print(json.dumps(metrics, indent=1))
    # the live app's model, read from its own metrics file so this line cannot go stale
    base = json.loads((ROOT / "public" / "model" / "metrics.json").read_text())
    print(f"\nBaseline (GBR, public/model/metrics.json): MAE {base['test_mae_MPa']}  RMSE {base['test_rmse_MPa']}"
          f"  R2 {base['test_r2']}  coverage {base['lower_estimate_coverage']}")
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:16s} {f.stat().st_size / 1e3:8.1f} kB")


if __name__ == "__main__":
    main()
