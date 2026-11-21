#!/usr/bin/env python3
"""OBELiX formula-only baseline with grouped cross-validation (official train partition only).

Target: log10 of experimental room-temperature ionic conductivity (S/cm).
Features: element fractions parsed from the Composition string.
Rows: the 478 IDs in work/data/train_idx.csv. Rows in test_idx.csv are never read for scoring.
Groups: connected components of rows that share a composition string or any DOI token.
Selection: pooled out-of-fold MAE; any model within 0.01 log10 units of the best is resolved
in favour of the simplest one (mean < Ridge (largest alpha first) < ExtraTrees).

Run from the repository root:  python3 domains/electrolytes/work/baseline.py
Output: a new JSON file in domains/electrolytes/work/reproductions/ (never overwrites).
"""
import hashlib, json, os, platform, re, sys, time, zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "reproductions")
SEED = 0
N_FOLDS = 5
TIE_TOL = 0.01
TARGET_COL = "Ionic conductivity (S cm-1)"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_formula(s):
    out = defaultdict(float)
    for el, num in re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", s):
        out[el] += float(num) if num else 1.0
    return dict(out)


def components(keys_per_row):
    """Union-find over rows that share any key."""
    parent = list(range(len(keys_per_row)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, keys in enumerate(keys_per_row):
        for k in keys:
            if k in first:
                a, b = find(i), find(first[k])
                if a != b:
                    parent[a] = b
            else:
                first[k] = i
    roots = [find(i) for i in range(len(keys_per_row))]
    relabel = {r: j for j, r in enumerate(dict.fromkeys(roots))}
    return np.array([relabel[r] for r in roots])


HALOGENS = {"F", "Cl", "Br", "I"}


def family(comp):
    """Coarse structural family from composition pattern only (heuristic, documented in README)."""
    els = set(comp)
    o = comp.get("O", 0.0)
    if {"P", "S"} <= els and els & {"Cl", "Br", "I"} and "O" not in els:
        return "argyrodite-like (Li-P-S-halide)"
    if "S" in els and "O" not in els:
        return "other sulfide / thiophosphate"
    if "La" in els and "O" in els and els & {"Zr", "Ta", "Nb", "Te", "W", "Sb", "Hf", "Sn", "Bi"} \
            and "Ti" not in els and 3.0 <= o / comp["La"] <= 5.0:
        return "garnet-like (Li-La-M-O, O/La~4)"
    if "La" in els and "Ti" in els and "O" in els:
        return "perovskite-like (Li-La-Ti-O)"
    if "P" in els and "O" in els and els & {"Ti", "Ge", "Zr", "Hf", "Sn"} and 3.5 <= o / comp["P"] <= 4.5:
        return "NASICON-like (Li-M(IV)-P-O)"
    if "O" not in els and "S" not in els and els & HALOGENS:
        return "halide"
    if "O" in els:
        return "other oxide"
    return "other"


def cif_audit(path):
    """Count CIFs that parse with a minimal reader: cell lengths/angles and >=1 fractional site."""
    res = {"files": 0, "parsed": 0, "failed": [], "ids_in_table": 0, "cell_a_matches_table": 0}
    table = pd.read_csv(os.path.join(DATA, "processed.csv")).set_index("ID")
    num = r"([-+]?[0-9]*\.?[0-9]+)(?:\([0-9]+\))?"
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".cif"):
                continue
            res["files"] += 1
            cid = os.path.basename(name)[:-4]
            txt = z.read(name).decode("utf-8", "replace")
            cell = {}
            for key in ("a", "b", "c", "alpha", "beta", "gamma"):
                m = re.search(r"_cell_(?:length|angle)_" + key + r"\s+" + num, txt)
                if m:
                    cell[key] = float(m.group(1))
            sites = 0
            if "_atom_site_fract_x" in txt:
                block = txt.split("_atom_site_fract_x", 1)[1]
                for line in block.splitlines()[1:]:
                    toks = line.split()
                    if not toks or toks[0].startswith("_"):
                        continue
                    if toks[0] in ("loop_",) or toks[0].startswith("#"):
                        break
                    if len(toks) >= 4:
                        sites += 1
            if len(cell) == 6 and sites > 0:
                res["parsed"] += 1
            else:
                res["failed"].append(cid)
            if cid in table.index:
                res["ids_in_table"] += 1
                if "a" in cell and abs(cell["a"] - float(table.loc[cid, "a"])) < 1e-3:
                    res["cell_a_matches_table"] += 1
    res["failed"] = res["failed"][:20]
    return res


def main():
    t0 = time.time()
    paths = {k: os.path.join(DATA, k) for k in ("processed.csv", "train_idx.csv", "test_idx.csv", "all_cifs.zip")}
    table = pd.read_csv(paths["processed.csv"])
    train_ids = pd.read_csv(paths["train_idx.csv"])["ID"].astype(str)
    df = table[table["ID"].isin(set(train_ids))].reset_index(drop=True)
    assert len(df) == len(train_ids) == 478, len(df)

    comps = [parse_formula(c) for c in df["Composition"]]
    elements = sorted({e for c in comps for e in c})  # vocabulary from the training partition only
    X = np.zeros((len(df), len(elements)))
    for i, c in enumerate(comps):
        tot = sum(c.values())
        for e, v in c.items():
            X[i, elements.index(e)] = v / tot
    y = np.log10(df[TARGET_COL].to_numpy(float))

    keys = []
    for comp, doi in zip(df["Composition"], df["DOI"].fillna("")):
        k = {"comp:" + comp.strip()}
        k |= {"doi:" + t.strip().lower() for t in str(doi).split("|") if t.strip()}
        keys.append(k)
    groups = components(keys)
    n_groups = int(groups.max() + 1)
    sizes = Counter(groups)

    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    fold_of = np.empty(len(y), int)
    for f, (_, te) in enumerate(folds):
        fold_of[te] = f

    # (name, complexity rank, factory); lower rank = simpler
    alphas = [100.0, 10.0, 1.0, 0.1, 0.01]
    specs = [("mean", 0, lambda: DummyRegressor(strategy="mean"))]
    for r, a in enumerate(alphas):
        specs.append((f"ridge_alpha_{a:g}", 1 + r, lambda a=a: make_pipeline(StandardScaler(), Ridge(alpha=a))))
    specs.append(("extratrees_500_leaf1", 10, lambda: ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=1, max_features=1.0, random_state=SEED, n_jobs=2)))
    specs.append(("extratrees_500_leaf3_sqrt", 11, lambda: ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=3, max_features="sqrt", random_state=SEED, n_jobs=2)))

    results, oof_by = {}, {}
    for name, rank, make in specs:
        oof = np.empty_like(y)
        per_fold = []
        for f, (tr, te) in enumerate(folds):
            m = make().fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
            per_fold.append(float(np.mean(np.abs(oof[te] - y[te]))))
        err = oof - y
        results[name] = {
            "complexity_rank": rank,
            "pooled_oof_mae": round(float(np.mean(np.abs(err))), 4),
            "pooled_oof_rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "pooled_oof_r2": round(float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)), 4),
            "fold_mae": [round(v, 4) for v in per_fold],
        }
        oof_by[name] = oof
    best = min(r["pooled_oof_mae"] for r in results.values())
    eligible = [n for n, r in results.items() if r["pooled_oof_mae"] <= best + TIE_TOL]
    selected = min(eligible, key=lambda n: results[n]["complexity_rank"])

    # Error by coarse family (selected model and mean baseline), out-of-fold
    fam = np.array([family(c) for c in comps])
    by_family = {}
    for fname in sorted(set(fam)):
        m = fam == fname
        by_family[fname] = {
            "n": int(m.sum()),
            "n_groups": int(len(set(groups[m]))),
            "selected_mae": round(float(np.mean(np.abs(oof_by[selected][m] - y[m]))), 3),
            "mean_baseline_mae": round(float(np.mean(np.abs(oof_by["mean"][m] - y[m]))), 3),
            "target_mean_log10": round(float(y[m].mean()), 3),
        }

    # Fold-assignment sensitivity (reported only; NOT used for selection): reshuffled group folds
    sensitivity = {}
    for s in range(5):
        fs = list(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=s).split(X, y, groups))
        row = {}
        for name, rank, make in specs:
            if name not in ("mean", selected, "extratrees_500_leaf3_sqrt", "ridge_alpha_100"):
                continue
            oof = np.empty_like(y)
            for tr, te in fs:
                oof[te] = make().fit(X[tr], y[tr]).predict(X[te])
            row[name] = round(float(np.mean(np.abs(oof - y))), 4)
        sensitivity[f"shuffle_seed_{s}"] = row

    report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "task": "log10 ionic conductivity (S/cm), OBELiX official train partition, formula-only",
        "command": "python3 " + " ".join(sys.argv),
        "data_sha256": {k: sha256(p) for k, p in paths.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "rows": len(df), "features": len(elements), "elements": elements,
        "groups": {"definition": "connected components of shared Composition string or shared DOI token (split on '|')",
                   "n_groups": n_groups, "largest_group": int(max(sizes.values())),
                   "singletons": int(sum(1 for v in sizes.values() if v == 1))},
        "folds": {"method": "sklearn GroupKFold(n_splits=5), no shuffle", "n_folds": N_FOLDS,
                  "rows_per_fold": [int(len(te)) for _, te in folds],
                  "groups_per_fold": [int(len(set(groups[te]))) for _, te in folds]},
        "selection_rule": f"min pooled OOF MAE; within {TIE_TOL} log10 of min prefer lowest complexity_rank",
        "models": results,
        "selected_model": selected,
        "selected_pooled_oof_mae": results[selected]["pooled_oof_mae"],
        "mean_baseline_pooled_oof_mae": results["mean"]["pooled_oof_mae"],
        "multiplicative_error_scale_of_selected": round(10 ** results[selected]["pooled_oof_mae"], 1),
        "by_family_oof": by_family,
        "fold_sensitivity_not_used_for_selection": sensitivity,
        "family_rule": "composition-pattern heuristic in baseline.py:family(); not a crystallographic assignment",
        "snapshot_comparison": {"snapshot_groups": 124, "snapshot_selected": "ExtraTrees",
                                "snapshot_selected_mae": 1.77, "snapshot_mean_mae": 2.55},
        "holdout": "test_idx.csv rows (121) were NOT scored; their labels are public, so they are not blind.",
        "cif_audit": cif_audit(paths["all_cifs.zip"]),
        "elapsed_seconds": None,
    }
    report["elapsed_seconds"] = round(time.time() - t0, 2)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "obelix_group_cv.json")
    if os.path.exists(out):
        out = os.path.join(OUT_DIR, f"obelix_group_cv_{report['run_id']}.json")
    with open(out, "x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("groups", "selected_model", "selected_pooled_oof_mae",
                                             "mean_baseline_pooled_oof_mae")}, indent=1))
    for n, r in results.items():
        print(f"{n:28s} MAE {r['pooled_oof_mae']:.3f}  R2 {r['pooled_oof_r2']:.3f}")
    print(json.dumps(by_family, indent=1))
    print(json.dumps(sensitivity, indent=1))
    print("cif:", {k: v for k, v in report["cif_audit"].items() if k != "failed"})
    print("wrote", out)


if __name__ == "__main__":
    main()
