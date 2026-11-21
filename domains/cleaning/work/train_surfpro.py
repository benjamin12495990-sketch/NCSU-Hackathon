#!/usr/bin/env python3
"""Paper-held-out baselines for SURFPRO pCMC and AW_ST_CMC (surface tension at the CMC).

Usage (repository root):  python3 -I domains/cleaning/work/train_surfpro.py
Reads work/cleaned/surfpro_clean.csv (from prepare_surfpro.py). Writes a new JSON report in
work/reproductions/ (surfpro_group_cv.json, or a time-stamped name if it exists).

Features (computed per molecule from SMILES with RDKit, counterions included):
  all RDKit 2D descriptors (Descriptors.CalcMolDescriptors), a 512-bit Morgan count fingerprint
  (radius 2), the surfactant type one-hot, and the CMC measurement temperature (C).
Evaluation: 5-fold GroupKFold over measurement-reference groups (group_meas_<task>), so no paper's
rows are split across folds. Model settings are fixed in advance; the selected model is the lowest
pooled out-of-fold MAE, preferring the simpler model within TIE_TOL. Imputation and scaling are fitted
inside each training fold. No external holdout exists, so no test score is reported.
"""
import hashlib, json, os, platform, sys, time, warnings
from datetime import datetime, timezone
import numpy as np, pandas as pd, sklearn, rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, 'cleaned', 'surfpro_clean.csv')
SRC = os.path.join(HERE, 'data', 'S129_SURFPRO.csv')
TIE_TOL = {'pCMC': 0.01, 'AW_ST_CMC': 0.1}  # log units / mN/m
SEED = 0


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def featurize(df):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    desc_rows, fp_rows, failed = [], [], []
    for i, smi in enumerate(df.SMILES):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            failed.append(int(df.Index.iloc[i])); desc_rows.append({}); fp_rows.append(np.full(512, np.nan)); continue
        desc_rows.append(Descriptors.CalcMolDescriptors(m))
        fp_rows.append(gen.GetCountFingerprintAsNumPy(m).astype(float))
    D = pd.DataFrame(desc_rows).astype(float).replace([np.inf, -np.inf], np.nan)
    D = D.clip(-1e6, 1e6)
    F = pd.DataFrame(np.vstack(fp_rows), columns=[f'mfp_{j}' for j in range(512)])
    T = pd.get_dummies(df.Surfactant_Type, prefix='type').astype(float).reset_index(drop=True)
    X = pd.concat([D.reset_index(drop=True), F, T, df[['temp_C_cmc']].reset_index(drop=True)], axis=1)
    return X, failed, D.shape[1]


def models():
    imp = lambda: SimpleImputer(strategy='median')
    return [
        ('mean', 0, lambda: DummyRegressor()),
        ('ridge_alpha_100', 1, lambda: make_pipeline(imp(), StandardScaler(), Ridge(alpha=100.0))),
        ('ridge_alpha_10', 2, lambda: make_pipeline(imp(), StandardScaler(), Ridge(alpha=10.0))),
        ('ridge_alpha_1', 3, lambda: make_pipeline(imp(), StandardScaler(), Ridge(alpha=1.0))),
        ('extratrees_500_leaf1_sqrt', 10, lambda: make_pipeline(imp(), ExtraTreesRegressor(
            n_estimators=500, max_features='sqrt', min_samples_leaf=1, random_state=SEED, n_jobs=2))),
        ('extratrees_500_leaf3_0.3', 11, lambda: make_pipeline(imp(), ExtraTreesRegressor(
            n_estimators=500, max_features=0.3, min_samples_leaf=3, random_state=SEED, n_jobs=2))),
        ('histgb_lr0.05_500', 12, lambda: HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=500, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED)),
    ]


def run_task(task, df, X):
    m = df[task].notna().to_numpy()
    Xt, y = X[m].to_numpy(float), df.loc[m, task].to_numpy(float)
    groups = df.loc[m, f'group_meas_{task}'].astype(int).to_numpy()
    types = df.loc[m, 'Surfactant_Type'].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(Xt, y, groups))
    res, oofs = {}, {}
    for name, rank, make in models():
        t0 = time.time(); oof = np.empty_like(y); fm = []
        for tr, te in folds:
            oof[te] = make().fit(Xt[tr], y[tr]).predict(Xt[te]); fm.append(float(np.mean(np.abs(oof[te] - y[te]))))
        e = oof - y
        res[name] = {'complexity_rank': rank, 'pooled_oof_mae': round(float(np.mean(np.abs(e))), 4),
                     'pooled_oof_rmse': round(float(np.sqrt(np.mean(e ** 2))), 4),
                     'pooled_oof_r2': round(float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), 4),
                     'fold_mae': [round(v, 4) for v in fm], 'seconds': round(time.time() - t0, 1)}
        oofs[name] = oof
        print(task, name, res[name]['pooled_oof_mae'], res[name]['pooled_oof_r2'], flush=True)
    best = min(r['pooled_oof_mae'] for r in res.values())
    sel = min((n for n, r in res.items() if r['pooled_oof_mae'] <= best + TIE_TOL[task]), key=lambda n: res[n]['complexity_rank'])
    by_type = {t: {'n': int((types == t).sum()), 'n_groups': int(len(set(groups[types == t]))),
                   'selected_mae': round(float(np.mean(np.abs(oofs[sel][types == t] - y[types == t]))), 3),
                   'mean_baseline_mae': round(float(np.mean(np.abs(oofs['mean'][types == t] - y[types == t]))), 3)}
               for t in sorted(set(types))}
    # grouped bootstrap of (selected - mean) MAE difference, resampling reference groups
    rng = np.random.default_rng(0); ug, inv = np.unique(groups, return_inverse=True)
    d = np.abs(oofs[sel] - y) - np.abs(oofs['mean'] - y); gs = np.bincount(inv, weights=d); gc = np.bincount(inv)
    bs = [gs[k].sum() / gc[k].sum() for k in (rng.integers(0, len(ug), len(ug)) for _ in range(2000))]
    return {'rows': int(m.sum()), 'unit': 'log10(1/(mol/L))' if task == 'pCMC' else 'mN/m',
            'groups': {'definition': 'connected components of measurement-reference tokens (SubRef if present, else Ref)',
                       'n_groups': int(len(ug)), 'largest_group_rows': int(gc.max())},
            'folds': {'method': 'GroupKFold(n_splits=5), no shuffle', 'rows_per_fold': [int(len(te)) for _, te in folds],
                      'groups_per_fold': [int(len(set(groups[te]))) for _, te in folds]},
            'target_sd': round(float(y.std()), 4), 'models': res, 'selected_model': sel,
            'selected_minus_mean_mae': {'value': round(float(d.mean()), 4),
                                        'group_bootstrap_95': [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]},
            'selection_note': f'selected on the same folds it is scored on (7 fixed candidates); tie tolerance {TIE_TOL[task]}',
            'by_surfactant_type_oof': by_type}


def main():
    t0 = time.time()
    df = pd.read_csv(CLEAN)
    X, failed, n_desc = featurize(df)
    rep = {'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
           'command': 'python3 -I ' + ' '.join(sys.argv),
           'data_sha256': {'S129_SURFPRO.csv': sha(SRC), 'surfpro_clean.csv': sha(CLEAN)},
           'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                        'sklearn': sklearn.__version__, 'rdkit': rdkit.__version__},
           'features': {'rdkit_descriptors': n_desc, 'morgan_count_bits': 512, 'type_onehot': int(sum(c.startswith('type_') for c in X.columns)),
                        'temperature': 1, 'total': int(X.shape[1]), 'smiles_failed': failed},
           'what_this_is': 'Surfactant physical-property prediction (pCMC, surface tension at CMC). NOT cleaning efficacy.',
           'tasks': {}}
    for task in ('pCMC', 'AW_ST_CMC'):
        rep['tasks'][task] = run_task(task, df, X)
    rep['elapsed_seconds'] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(HERE, 'reproductions'), exist_ok=True)
    out = os.path.join(HERE, 'reproductions', 'surfpro_group_cv.json')
    if os.path.exists(out):
        out = out[:-5] + '_' + rep['run_id'] + '.json'
    with open(out, 'x') as f:
        json.dump(rep, f, indent=1)
    for t, r in rep['tasks'].items():
        print(t, r['groups'], r['selected_model'], r['models'][r['selected_model']]['pooled_oof_mae'], 'mean', r['models']['mean']['pooled_oof_mae'], r['selected_minus_mean_mae'])
        print(json.dumps(r['by_surfactant_type_oof']))
    print('wrote', out, rep['elapsed_seconds'], 's')


if __name__ == '__main__':
    main()
