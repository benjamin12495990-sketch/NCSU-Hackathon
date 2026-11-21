#!/usr/bin/env python3
"""EXPLORATORY reference-grouped baseline: log10 corrosion rate (mm/yr) from simple encodings.

Usage (repository root):  python3 -I domains/corrosives/work/baseline_corrdata.py
Reads work/cleaned/corrdata_normalized.csv (from parse_corrdata.py). Writes a new JSON report
work/reproductions/corrdata_baseline_group_cv.json (time-stamped name if it exists).

Rows: numeric rates that are plain point values (rate_qualifier == 'exact'), > 0 mm/yr, and whose two
rate columns do not disagree by more than 2x. Bounds ('max'/'min'), ranges, zero rates and letter
ratings are excluded from this regression (counts are reported).
Features: material class, material family, material, environment (one-hot, categories seen fewer
than 10 times in the training fold pooled as 'infrequent'); concentration value when its unit is the
column default vol% (else missing) and the concentration qualifier; temperature in C (point value or
range midpoint) and its qualifier; log10 exposure hours (midpoint) when given.
All encoders and imputers are fitted inside each training fold.
Evaluation: 5-fold GroupKFold by Reference # (primary), and separately by material (secondary).
"""
import hashlib, json, os, platform, sys, time, warnings
from datetime import datetime, timezone
import numpy as np, pandas as pd, sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, 'cleaned', 'corrdata_normalized.csv')
CAT = ['material_class', 'material_family', 'material', 'environment', 'conc_qualifier', 'temp_qualifier']
NUMC = ['conc_vol_pct', 'temp_C_mid', 'log10_hours']
TIE_TOL = 0.01


def load():
    d = pd.read_csv(CLEAN, keep_default_na=False, na_values=[''], low_memory=False,
                    dtype={'reference_number': str, 'material': str, 'environment': str})
    n0 = len(d)
    num = d[d.rate_status == 'numeric']
    counts = {'all_rows': n0, 'numeric_rate_rows': len(num),
              'excluded_bounds_or_ranges': int((num.rate_qualifier != 'exact').sum())}
    ex = num[num.rate_qualifier == 'exact']
    counts['excluded_zero_rate'] = int((ex.rate_mm_y <= 0).sum())
    counts['excluded_large_column_disagreement'] = int(((ex.rate_mm_y > 0) & (ex.rate_columns_consistent == 'large_disagreement')).sum())
    ex = ex[(ex.rate_mm_y > 0) & (ex.rate_columns_consistent != 'large_disagreement')]
    counts['excluded_missing_reference_number'] = int(ex.reference_number.isna().sum())
    ex = ex[ex.reference_number.notna()].copy()
    counts['rows_used'] = len(ex)
    ex['conc_vol_pct'] = np.where(ex.conc_unit == 'vol% (column default)', ex.conc_value, np.nan)
    mid = ex.temp_C_value.where(ex.temp_C_value.notna(), (ex.temp_C_low + ex.temp_C_high) / 2)
    ex['temp_C_mid'] = mid
    ex['log10_hours'] = np.log10(((ex.duration_h_low + ex.duration_h_high) / 2).clip(lower=0.1))
    for c in CAT:
        ex[c] = ex[c].fillna('none').astype(str)
    return ex.reset_index(drop=True), counts


def pre(scale):
    num = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), *( [StandardScaler()] if scale else []))
    return ColumnTransformer([('cat', OneHotEncoder(min_frequency=10, handle_unknown='infrequent_if_exist', sparse_output=False), CAT),
                              ('num', num, NUMC)])


SPECS = [('mean', 0, lambda: DummyRegressor()),
         ('ridge_alpha_10', 1, lambda: make_pipeline(pre(True), Ridge(alpha=10.0))),
         ('ridge_alpha_1', 2, lambda: make_pipeline(pre(True), Ridge(alpha=1.0))),
         ('extratrees_300_leaf3', 10, lambda: make_pipeline(pre(False), ExtraTreesRegressor(
             n_estimators=300, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=2)))]


def evaluate(df, y, groups, label):
    folds = list(GroupKFold(n_splits=5).split(df, y, groups))
    res, oofs = {}, {}
    for name, rank, make in SPECS:
        t0 = time.time(); oof = np.empty_like(y); fm = []
        for tr, te in folds:
            oof[te] = make().fit(df.iloc[tr], y[tr]).predict(df.iloc[te]); fm.append(float(np.mean(np.abs(oof[te] - y[te]))))
        e = oof - y
        res[name] = {'complexity_rank': rank, 'pooled_oof_mae': round(float(np.mean(np.abs(e))), 4),
                     'pooled_oof_r2': round(float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), 4),
                     'fold_mae': [round(v, 4) for v in fm], 'seconds': round(time.time() - t0, 1)}
        oofs[name] = oof
        print(label, name, res[name]['pooled_oof_mae'], res[name]['pooled_oof_r2'], flush=True)
    best = min(r['pooled_oof_mae'] for r in res.values())
    sel = min((n for n, r in res.items() if r['pooled_oof_mae'] <= best + TIE_TOL), key=lambda n: res[n]['complexity_rank'])
    ug, inv = np.unique(groups, return_inverse=True)
    d = np.abs(oofs[sel] - y) - np.abs(oofs['mean'] - y); gs = np.bincount(inv, weights=d); gc = np.bincount(inv)
    rng = np.random.default_rng(0)
    bs = [gs[k].sum() / gc[k].sum() for k in (rng.integers(0, len(ug), len(ug)) for _ in range(2000))]
    by_class = {c: {'n': int((df.material_class == c).sum()),
                    'selected_mae': round(float(np.mean(np.abs(oofs[sel] - y)[df.material_class == c])), 3),
                    'mean_mae': round(float(np.mean(np.abs(oofs['mean'] - y)[df.material_class == c])), 3)}
                for c in df.material_class.value_counts().index}
    return {'n_groups': int(len(ug)), 'largest_group_rows': int(gc.max()),
            'folds': {'method': 'GroupKFold(n_splits=5), no shuffle', 'rows_per_fold': [int(len(te)) for _, te in folds],
                      'groups_per_fold': [int(len(set(groups[te]))) for _, te in folds]},
            'models': res, 'selected_model': sel,
            'selected_minus_mean_mae': {'value': round(float(d.mean()), 4),
                                        'group_bootstrap_95': [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]},
            'by_material_class_oof': by_class}


def main():
    t0 = time.time()
    df, counts = load()
    y = np.log10(df.rate_mm_y.to_numpy(float))
    rep = {'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'), 'label': 'EXPLORATORY baseline',
           'command': 'python3 -I ' + ' '.join(sys.argv),
           'data_sha256': {'corrdata_normalized.csv': hashlib.sha256(open(CLEAN, 'rb').read()).hexdigest()},
           'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
           'target': 'log10 corrosion rate, mm/yr (point values only)', 'row_counts': counts,
           'target_summary': {'mean': round(float(y.mean()), 3), 'sd': round(float(y.std()), 3),
                              'min_mm_y': float(df.rate_mm_y.min()), 'max_mm_y': float(df.rate_mm_y.max())},
           'features': {'categorical': CAT, 'numeric': NUMC},
           'selection_rule': f'min pooled OOF MAE within each grouping, simpler model within {TIE_TOL}; selected on the folds it is scored on',
           'grouped_by_reference': evaluate(df, y, df.reference_number.to_numpy(dtype=object), 'ref'),
           'grouped_by_material': evaluate(df, y, df.material.to_numpy(dtype=object), 'material'),
           'not_safety_advice': 'Historical single observations; not a guarantee of service performance.'}
    rep['elapsed_seconds'] = round(time.time() - t0, 1)
    out = os.path.join(HERE, 'reproductions', 'corrdata_baseline_group_cv.json')
    if os.path.exists(out):
        out = out[:-5] + '_' + rep['run_id'] + '.json'
    with open(out, 'x') as f:
        json.dump(rep, f, indent=1)
    for k in ('grouped_by_reference', 'grouped_by_material'):
        r = rep[k]; print(k, r['n_groups'], r['selected_model'], r['models'][r['selected_model']]['pooled_oof_mae'],
                          'mean', r['models']['mean']['pooled_oof_mae'], r['selected_minus_mean_mae'])
    print(counts, 'wrote', out, rep['elapsed_seconds'])


if __name__ == '__main__':
    main()
