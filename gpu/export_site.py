"""Writes the website's pipeline files (public/pipeline/) from the nightly registry, so the site shows what the pipeline really trained:
for each task, how the network average, the strongest tree model and their no-fit blend did on rows they never trained on, with 90%
intervals; the held-out rows themselves (a capped, fixed sample); for crystals, the unit cells of those rows for the 3D view; and for
corrosion a small model that runs in the browser for "what if" questions.

  python -m gpu.export_site                       # after a nightly: reads gpu/runs/pipeline/registry.json, writes public/pipeline/
  python -m gpu.export_site --registry gpu/runs/pipeline/registry_smoke.json --out <folder>   # from a rehearsal (marked as one)

Every number is computed by gpu/pipeline/analyze.py from the run folders the registry points at; nothing is typed here. The files
carry no local paths (folders appear by name only). Concrete is listed with its numbers but without rows: its page is the live
designer, and its rows stay off the site. The sealed files are never read (gpu/data.py refuses them). Files in --out are replaced;
git keeps the previous export.
"""
import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from gpu.common import REPO, RUNS

OUT = REPO / 'public' / 'pipeline'
SITE_ID = {'alloys': 'materials', 'polymers': 'polymers', 'insulators': 'insulators', 'chemistry': 'chemistry', 'electrolytes': 'electrolytes',
           'cleaning_pcmc': 'cleaning', 'cleaning_tension': 'cleaning', 'corrosion': 'corrosives', 'concrete': 'concrete'}
TASK_TITLE = {'alloys': 'Alloy hardness', 'polymers': 'Polymer glass transition', 'insulators': 'Dielectric constant (DFT)',
              'chemistry': 'Water solubility', 'electrolytes': 'Solid-electrolyte ionic conductivity', 'cleaning_pcmc': 'Critical micelle concentration',
              'cleaning_tension': 'Surface tension at the CMC', 'corrosion': 'Corrosion rate', 'concrete': 'Concrete strength (14-28 d from 1-7 d)',
              'bulk_modulus_kv': 'Bulk modulus (DFT)', 'shear_modulus_gv': 'Shear modulus (DFT)', 'formation_energy_peratom': 'Formation energy (DFT)',
              'optb88vdw_bandgap': 'Band gap, OptB88vdW (DFT)', 'mbj_bandgap': 'Band gap, TBmBJ (DFT)'}
EXPLORATORY = {'concrete': 'these concrete rows already informed project decisions',
               'corrosion': 'historical point values of mixed quality; not safety advice'}
CRYSTAL_EXPLORATORY = 'the held-out-systems test crystals were scored in earlier cycles'
# A target added later has its own note (DEC-022): its first night is a clean test; later nights score the same crystals again.
CRYSTAL_EXPLORATORY_BY = {'mbj_bandgap': 'the TBmBJ band gap was a clean test only on the first night it was trained; each later night scores '
                                         'the same test crystals again with more seeds'}
LEVEL = '0.90'


def _no_paths(text):
    """An error reason for the published index.json, with this computer's repo and home folders replaced (gpu/pipeline/report.py), and
    any other absolute path cut down to its last component."""
    import re
    from gpu.pipeline.report import _local_path_patterns
    s = str(text or '')
    for rx, label in _local_path_patterns():
        s = rx.sub(label, s)
    return re.sub(r'(?:[A-Za-z]:)?[\\/]+(?:[^\s\'"\\/:]+[\\/]+)+([^\s\'"\\/:]+)', r'\1', s)


def _r(x, sig=4):
    """Round for the site (4 significant figures); None stays None."""
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x) or x == 0:
        return 0.0 if x == 0 else None
    return float(f'{x:.{sig}g}')


def _ci(c, first, second):
    """A paired comparison (MAE of first minus MAE of second, 95% interval over groups), worded with the two names. The analysis
    always says 'network'/'trees' for the two sides, whatever they are; here the side that is really better is named."""
    if not c:
        return None
    lo, hi = float(c['ci95'][0]), float(c['ci95'][1])
    better = first if hi < 0 else second if lo > 0 else None
    return {'diff': _r(c['diff']), 'ci95': [_r(lo), _r(hi)], 'first': first, 'second': second, 'better': better,
            'verdict': f'{better} clearly better' if better else 'no clear difference'}


def _single(f):
    """One network at a time against the trees: how many single runs are clearly better or clearly worse."""
    s = (f or {}).get('single')
    return None if not s else {'n_runs': s['n'], 'clearly_better': s['n_clearly_better'], 'clearly_worse': s['n_clearly_worse'], 'mean_mae': _r(s['mean_mae'])}


def _sample(keys, cap):
    """Indices of a fixed sample of at most cap rows, ordered by a hash of the row key (stable across nights, not a block of ids)."""
    order = sorted(range(len(keys)), key=lambda i: hashlib.sha256(str(keys[i]).encode()).hexdigest())
    return sorted(order[:cap])


def _interval(resid_cal, resid_eval, folds=None):
    from gpu.pipeline.analyze import conformal_cv, conformal_split
    iv = conformal_cv(resid_cal, folds) if folds is not None else conformal_split(resid_cal, resid_eval)
    v = iv[LEVEL]
    return {'half_width': _r(v['half_width']), 'coverage': _r(v['coverage'], 3)}


def tabular_task(pipe, family, cap):
    """Out-of-fold predictions of the network average (settings cross-fitted), the strongest trees (cross-fitted) and their 50/50 blend,
    the same quantities the report's intervals section uses. Returns (summary entry, rows file or None)."""
    from gpu.data import load
    from gpu.pipeline.analyze import _first, _m, _paired, analyze_tabular, crossfit, verdict
    t = analyze_tabular(pipe, family)
    if t.get('status') != 'ok':
        return {'status': t.get('status'), 'reason': _no_paths(t.get('reason'))}, None
    entry = {'status': 'ok', 'unit': t['target'], 'rows': t['rows'], 'groups': t['groups'], 'mode': t['mode'],
             'mae': {'network': _r(t['methods']['network']['mae']), 'trees': _r(t['methods']['trees']['mae']), 'mean': _r(t['methods']['mean']['mae'])},
             'network_vs_trees': _ci(t.get('network_vs_trees'), 'network', 'trees'),
             'single_network': _single(t.get('fairness'))}
    f = t.get('fairness') or {}
    if 'blend' in f:
        entry['mae']['blend'] = _r(f['blend']['mae'])
        entry['blend_vs_trees'] = _ci(f['blend'].get('vs_trees'), 'blend', 'trees')
        entry['blend_vs_network'] = _ci(f['blend'].get('vs_network_average'), 'blend', 'network')
    ds = load(family)
    sweep, base = _first(pipe, f'tab_{family}', '*/*/result.json'), _first(pipe, f'base_{family}', 'baselines.json')
    P, B = np.load(sweep.parent / 'predictions.npz'), np.load(base.with_name('baselines_predictions.npz'))
    bj = json.loads(base.read_text())
    tree_idx = [i for i, n in enumerate(bj['model_names']) if bj['groups'][n] == 'trees']
    if t['mode'] != 'cv':
        # Concrete: numbers only, no rows on the site. Its intervals are computed exactly as the report's intervals section does
        # (gpu/pipeline/analyze.py analyze_intervals): split conformal, calibrated on the fixed test rows, coverage checked on the
        # extra split (MONITOR).
        evl = next(iter(ds.extra))
        y_sel, y_ev = ds.y_test, ds.extra[evl][1]
        ens_sel, ens_ev = P['fixed_test'].mean(axis=1), P[f'fixed_{evl}'].mean(axis=1)
        ci = int(np.argmin([np.abs(e - y_sel).mean() for e in ens_sel]))
        c = tree_idx[int(np.argmin([np.abs(B['pred_test'][i] - y_sel).mean() for i in tree_idx]))]
        pairs = {'trees': (B['pred_test'][c], B[f'pred_{evl}'][c]), 'network': (ens_sel[ci], ens_ev[ci])}
        pairs['blend'] = (0.5 * (pairs['trees'][0] + pairs['network'][0]), 0.5 * (pairs['trees'][1] + pairs['network'][1]))
        entry['interval90'] = {k: _interval(v[0] - y_sel, v[1] - y_ev) for k, v in pairs.items()}
        entry['interval_method'] = f'split conformal: calibrated on the fixed test rows, coverage checked on {evl} (exploratory)'
        return entry, None
    y, folds = ds.y, ds.folds
    preds = {'network': crossfit(P['cv_val'].mean(axis=1), y, folds)[0], 'trees': crossfit(B['oof'][tree_idx], y, folds)[0]}
    preds['blend'] = 0.5 * (preds['network'] + preds['trees'])
    entry['interval90'] = {k: _interval(v - y, None, folds) for k, v in preds.items()}
    labels = ds.labels or [{} for _ in y]
    keys = [json.dumps(labels[i], sort_keys=True) + f'#{i}' for i in range(len(y))]
    pick = _sample(keys, cap)
    rows = [{'label': labels[i], 'group': str(ds.groups[i]), 'y': _r(y[i]), **{k: _r(v[i]) for k, v in preds.items()}} for i in pick]
    return entry, {'task': f'tab:{family}', 'unit': t['target'], 'kind': 'table', 'split': ds.folds_source,
                   'rows_total': int(len(y)), 'rows_shown': len(rows), 'sample_rule': 'a fixed sample ordered by a hash of the row',
                   'interval90': entry['interval90'], 'rows': rows}


def crystal_task(pipe, target, cap):
    """Held-out-systems test crystals: the network average over the seeds, the strongest trees (chosen on validation crystals) and the
    50/50 blend, with split-conformal 90% intervals calibrated on the validation crystals."""
    from gpu.pipeline.analyze import SPLIT_SPECS, _first, _run_preds, _runs, analyze_structure, rule_of_run
    s = analyze_structure(pipe, target, 'system')
    if s.get('status') != 'ok':
        return {'status': s.get('status'), 'reason': _no_paths(s.get('reason'))}, None, []
    runs = _runs(pipe, SPLIT_SPECS['system'][0].format(t=target))
    vy, vp, vs, ty, tp, ts = _run_preds(runs)
    z0 = np.load(Path(runs[0]) / 'predictions.npz')
    tf = _first(pipe, f'trees_{target}_sys', 'trees_predictions.npz')
    tz = np.load(tf)
    key = s['strongest_trees_key']
    vix, tix = {j: i for i, j in enumerate(tz['val_jid'])}, {j: i for i, j in enumerate(tz['jid'])}
    tv, tt = tz['val_' + key][[vix[j] for j in z0['val_jid']]], tz[key][[tix[j] for j in z0['test_jid']]]
    preds = {'network': (vp, tp), 'trees': (tv, tt), 'blend': (0.5 * (vp + tv), 0.5 * (tp + tt))}
    avg = [c for k, c in s['compare']['comparisons'][key].items() if k.startswith('network average')]
    f = s.get('fairness') or {}
    entry = {'status': 'ok', 'unit': s['label'], 'rows': int(len(ty)), 'groups': int(len(set(z0['test_group']))), 'mode': 'held-out chemical systems',
             'n_runs': s['n_runs'], 'neighbor_rule': sorted({rule_of_run(r) for r in runs}),
             'mae': {'network': _r(avg[0]['network_mae'] if avg else np.abs(tp - ty).mean()), 'trees': _r(s['trees'][s['strongest_trees']]['test']['mae']),
                     'blend': _r(np.abs(preds['blend'][1] - ty).mean()), 'mean': _r(s['mean_baseline_test_mae'])},
             'strongest_trees': s['strongest_trees'], 'network_vs_trees': _ci(s['verdicts'].get(key), 'network', 'trees'),
             'blend_vs_trees': _ci((f.get('blend') or {}).get('vs_trees'), 'blend', 'trees'),
             'blend_vs_network': _ci((f.get('blend') or {}).get('vs_network_average'), 'blend', 'network'),
             'single_network': _single(f),
             'interval90': {k: _interval(v[0] - vy, v[1] - ty) for k, v in preds.items()}}
    jids = [str(j) for j in z0['test_jid']]
    pick = _sample(jids, cap)
    rows = [{'jid': jids[i], 'formula': str(z0['test_formula'][i]), 'system': str(z0['test_group'][i]), 'y': _r(ty[i]),
             **{k: _r(v[1][i]) for k, v in preds.items()}} for i in pick]
    return entry, {'task': f'crystal:{target}', 'unit': s['label'], 'kind': 'crystal', 'split': 'held-out chemical systems (test crystals)',
                   'rows_total': int(len(ty)), 'rows_shown': len(rows), 'sample_rule': 'a fixed sample ordered by a hash of the record id',
                   'interval90': entry['interval90'], 'neighbor_rule': entry['neighbor_rule'], 'rows': rows}, [r['jid'] for r in rows]


def structures(jids, data_zip, max_atoms):
    """Unit cells (lattice in angstrom, fractional coordinates, elements) of the given JARVIS records, for the 3D view."""
    want = set(jids)
    with zipfile.ZipFile(data_zip) as z:
        name = next(n for n in z.namelist() if n.endswith('.json'))
        recs = json.loads(z.read(name))
    out, skipped = {}, 0
    for r in recs:
        if r.get('jid') not in want:
            continue
        a = r['atoms']
        L, X, els = np.array(a['lattice_mat'], float), np.array(a['coords'], float).reshape(-1, 3), list(a['elements'])
        if a.get('cartesian', False):
            X = X @ np.linalg.inv(L)
        if len(els) > max_atoms:
            skipped += 1
            continue
        out[r['jid']] = {'formula': r.get('formula'), 'L': [[_r(v, 6) for v in row] for row in L], 'X': [[_r(v, 6) for v in row] for row in X % 1.0], 'E': els}
    return out, skipped


def corrosion_model(seed=0):
    """A compact gradient-boosted tree model for the corrosion page's "what if" form, trained on all rows the pipeline uses, scored with
    the pipeline's own grouped folds (out-of-fold MAE and a 90% interval from out-of-fold residuals). It is a separate, smaller model
    than the pipeline's tuned trees, and the page says so. The feature recipe is exported so the browser rebuilds inputs exactly, and
    check rows let a test confirm the browser gives the same predictions."""
    from sklearn.ensemble import GradientBoostingRegressor
    from gpu.data import load
    ds = load('corrosion')
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=seed)
    oof = np.empty(len(ds.y))
    for f in range(int(ds.folds.max()) + 1):
        m = GradientBoostingRegressor(**params).fit(ds.X[ds.folds != f], ds.y[ds.folds != f])
        oof[ds.folds == f] = m.predict(ds.X[ds.folds == f])
    res = np.sort(np.abs(oof - ds.y))
    q = float(res[min(len(res) - 1, int(np.ceil((len(res) + 1) * 0.9)) - 1)])
    model = GradientBoostingRegressor(**params).fit(ds.X, ds.y)
    trees = []
    for est in model.estimators_[:, 0]:
        tr = est.tree_
        trees.append({'f': tr.feature.tolist(), 't': [_r(v, 7) for v in tr.threshold], 'l': tr.children_left.tolist(), 'r': tr.children_right.tolist(),
                      'v': [_r(v[0][0] * model.learning_rate, 7) for v in tr.value]})
    init = float(np.mean(ds.y)) if not hasattr(model.init_, 'constant_') else float(np.ravel(model.init_.constant_)[0])
    names = list(ds.features)
    num = {n: float(np.median(ds.X[:, names.index(n)])) for n in ('conc_vol_pct', 'temp_C_mid', 'log10_hours') if n in names}
    labels = ds.labels
    mats = {}
    for lab in labels:
        mats.setdefault(lab['material'], {}).setdefault((lab['material_class'], lab['material_family']), 0)
        mats[lab['material']][(lab['material_class'], lab['material_family'])] += 1
    materials = [{'name': m, 'class': max(v, key=v.get)[0], 'family': max(v, key=v.get)[1], 'rows': sum(v.values())} for m, v in sorted(mats.items())]
    envs = {}
    for lab in labels:
        envs[lab['environment']] = envs.get(lab['environment'], 0) + 1
    rng = np.random.default_rng(seed)
    checks = [{'x_nonzero': {names[j]: _r(ds.X[i, j], 9) for j in np.nonzero(ds.X[i])[0]}, 'pred': float(model.predict(ds.X[i:i + 1])[0])}
              for i in rng.choice(len(ds.y), size=min(12, len(ds.y)), replace=False)]
    return {'kind': 'gradient-boosted trees (scikit-learn GradientBoostingRegressor)', 'params': params, 'unit': ds.target,
            'features': names, 'numeric_medians': num, 'init': _r(init, 9), 'trees': trees,
            'cv': {'folds': ds.folds_source, 'mae': _r(np.abs(oof - ds.y).mean()), 'interval90_half_width': _r(q),
                   'mean_baseline_mae': _r(np.abs(ds.y - ds.y.mean()).mean())},
            'materials': materials, 'environments': [{'name': e, 'rows': n} for e, n in sorted(envs.items())], 'checks': checks,
            'records': [{k: lab[k] for k in ('material', 'environment', 'conc_vol_pct', 'temp_C', 'hours', 'reference')} | {'y': _r(ds.y[i])}
                        for i, lab in enumerate(labels)]}


def _gbm_oof(X, y, folds, params):
    """Out-of-fold predictions with the pipeline's grouped folds, the final model on all rows, and the 90% half-width of the
    out-of-fold absolute residuals (split-conformal style quantile ceil((n+1)*0.9))."""
    from sklearn.ensemble import GradientBoostingRegressor
    oof = np.empty(len(y))
    for f in range(int(folds.max()) + 1):
        m = GradientBoostingRegressor(**params).fit(X[folds != f], y[folds != f])
        oof[folds == f] = m.predict(X[folds == f])
    res = np.sort(np.abs(oof - y))
    q = float(res[min(len(res) - 1, int(np.ceil((len(res) + 1) * 0.9)) - 1)])
    return oof, GradientBoostingRegressor(**params).fit(X, y), q


def _trees_json(model):
    out = []
    for est in model.estimators_[:, 0]:
        tr = est.tree_
        out.append({'f': tr.feature.tolist(), 't': [_r(v, 7) for v in tr.threshold], 'l': tr.children_left.tolist(), 'r': tr.children_right.tolist(),
                    'v': [_r(v[0][0] * model.learning_rate, 7) for v in tr.value]})
    return out


def _init(model, y):
    return float(np.ravel(model.init_.constant_)[0]) if hasattr(model.init_, 'constant_') else float(np.mean(y))


def electrolyte_model(seed=0):
    """Solid-electrolyte "designer" model: gradient-boosted trees on element fractions (the pipeline's own inputs for this family), scored
    with its grouped folds. Runs in the browser on compositions the page generates by nudging known ones."""
    from gpu.data import load
    ds = load('electrolytes')
    params = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed)
    oof, model, q = _gbm_oof(ds.X, ds.y, ds.folds, params)
    elements = [n[2:] for n in ds.features]
    rng = np.random.default_rng(seed)
    checks = [{'fractions': {elements[j]: _r(ds.X[i, j], 9) for j in np.nonzero(ds.X[i])[0]}, 'pred': float(model.predict(ds.X[i:i + 1])[0])}
              for i in rng.choice(len(ds.y), size=min(12, len(ds.y)), replace=False)]
    return {'kind': 'gradient-boosted trees (scikit-learn GradientBoostingRegressor)', 'params': params, 'unit': ds.target, 'elements': elements,
            'init': _r(_init(model, ds.y), 9), 'trees': _trees_json(model),
            'cv': {'folds': ds.folds_source, 'mae': _r(np.abs(oof - ds.y).mean()), 'interval90_half_width': _r(q),
                   'mean_baseline_mae': _r(np.abs(ds.y - ds.y.mean()).mean())},
            'checks': checks,
            'records': [{'composition': lab['composition'], 'id': lab['id'], 'y': _r(ds.y[i]), 'oof': _r(oof[i]),
                         'fractions': {elements[j]: _r(ds.X[i, j], 6) for j in np.nonzero(ds.X[i])[0]}} for i, lab in enumerate(ds.labels)]}


def surfactant_table(seed=0):
    """Every SURFPRO surfactant with its measured pCMC and surface tension where they exist, and a prediction of each property from a
    gradient-boosted tree model trained on the rows that have it (out-of-fold for measured rows, so it can be checked; the full model for
    the rest). The inputs (RDKit descriptors, Morgan counts) are built exactly as gpu/data.py builds them for training; a surfactant's
    missing inputs are filled with the medians of the training rows."""
    import csv
    import io
    from gpu.data import REPO as DREPO, _load_cleaning, guard_path, read_bytes
    base = DREPO / 'domains/cleaning/work/cleaned'
    rows = list(csv.DictReader(io.StringIO(read_bytes(base / 'surfpro_clean.csv').decode('utf-8'))))
    guard_path(base / 'surfpro_features.npz')
    z = np.load(io.BytesIO(read_bytes(base / 'surfpro_features.npz')), allow_pickle=False)
    Xall, fnames = z['X'].astype(float), [str(f) for f in z['features']]
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=seed)
    out = [{'smiles': r['SMILES'], 'type': r['Surfactant_Type'], 'temp_C': r['temp_C_cmc']} for r in rows]
    summary = {}
    for task, fam, key in (('pCMC', 'cleaning_pcmc', 'pcmc'), ('AW_ST_CMC', 'cleaning_tension', 'tension')):
        ds = _load_cleaning(task, '', fam)
        keep = [i for i, r in enumerate(rows) if r[task] not in ('', None)]
        Xk = Xall[keep]
        miss = np.isnan(Xk)
        cols = np.where(miss.any(axis=0))[0]
        med = np.nanmedian(Xk, axis=0)
        med = np.where(np.isnan(med), 0.0, med)

        def transform(Xr):
            m = np.isnan(Xr)
            Xf = np.where(m, med, Xr)
            if len(cols):
                Xf = np.hstack([Xf, m[:, cols].astype(float)])
            return Xf
        names = fnames + [f'{fnames[c]}__missing' for c in cols]
        varying = [j for j in range(len(names)) if np.ptp(transform(Xk)[:, j]) > 0]
        Xt = transform(Xk)[:, varying]
        assert Xt.shape == ds.X.shape and np.allclose(Xt, ds.X), f'{task}: the rebuilt inputs differ from the training inputs'
        oof, model, q = _gbm_oof(ds.X, ds.y, ds.folds, params)
        pred_all = model.predict(transform(Xall)[:, varying])
        for i in range(len(rows)):
            out[i][key] = {'measured': None, 'predicted': _r(pred_all[i]), 'out_of_fold': False}
        for j, i in enumerate(keep):
            out[i][key] = {'measured': _r(ds.y[j]), 'predicted': _r(oof[j]), 'out_of_fold': True}
        summary[key] = {'unit': 'pCMC = -log10(CMC, mol/L)' if key == 'pcmc' else 'mN/m', 'rows_measured': len(keep),
                        'cv_mae': _r(np.abs(oof - ds.y).mean()), 'interval90_half_width': _r(q), 'mean_baseline_mae': _r(np.abs(ds.y - ds.y.mean()).mean()),
                        'folds': ds.folds_source}
    return {'kind': 'gradient-boosted trees per property (scikit-learn GradientBoostingRegressor)', 'params': params, 'properties': summary,
            'surfactants': out}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, separators=(',', ':'), ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def write_browser_models(out, generated, index):
    """The small models and tables the designer pages run in the browser. They depend only on data in the repository, not on a nightly."""
    for fname, fn, src in (('corrosion_model.json', corrosion_model, 'NIST CORR-DATA (NIST open data licence)'),
                           ('electrolyte_model.json', electrolyte_model, 'OBELiX training partition (CC BY 4.0)'),
                           ('surfactants.json', surfactant_table, 'SURFPRO, Zenodo 15861809 (CC BY 4.0)')):
        try:
            write(out / fname, {**fn(), 'generated': generated, 'source': src})
            index.setdefault('browser_files', []).append(fname)
            print(f'{fname} written', flush=True)
        except Exception as ex:  # noqa: BLE001
            index.setdefault('browser_errors', {})[fname] = repr(ex)[:300]
            print(f'{fname} FAILED: {ex!r}', flush=True)
    return index


def main(argv=None):
    from gpu.structure.train_structure import DEFAULT_ZIP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--registry', default=str(RUNS / 'pipeline' / 'registry.json'))
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--max-rows', type=int, default=2000, help='held-out rows per task shown on the site (a fixed hash-ordered sample)')
    ap.add_argument('--max-atoms', type=int, default=48, help='crystals with more atoms in the cell are listed without a 3D view')
    ap.add_argument('--data', default=str(DEFAULT_ZIP))
    ap.add_argument('--browser-models-only', action='store_true', help='write only the corrosion, electrolyte and surfactant files (they need no nightly)')
    ap.add_argument('--no-browser-models', action='store_true', help='skip the small models trained here for the corrosion, electrolyte and surfactant pages')
    a = ap.parse_args(argv)
    reg_path, out = Path(a.registry), Path(a.out)
    if a.browser_models_only:
        return write_browser_models(out, time.strftime('%Y-%m-%d %H:%M:%S'), {})
    try:
        reg = json.loads(reg_path.read_text())['tasks']
    except (OSError, ValueError, KeyError):
        sys.exit(f'no registry at {reg_path}: run a nightly first (.\\gpu\\run_overnight.ps1 -Nightly)')
    rehearsal = 'smoke' in reg_path.name
    index = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'rehearsal': rehearsal, 'registry': reg_path.name, 'tasks': []}
    crystal_jids = []
    for k in sorted(reg):
        e = reg[k]
        kind, name = k.split(':', 1)
        pipe = Path(e['folder'])
        base = {'id': k, 'kind': 'table' if kind == 'tab' else 'crystal', 'name': name, 'title': TASK_TITLE.get(name, name),
                'site': SITE_ID.get(name, 'crystals') if kind == 'tab' else 'crystals', 'trained': e.get('date'), 'folder': pipe.name,
                'recommended': e.get('recommended'), 'exploratory': EXPLORATORY.get(name) if kind == 'tab' else CRYSTAL_EXPLORATORY_BY.get(name, CRYSTAL_EXPLORATORY)}
        if not pipe.exists():
            index['tasks'].append({**base, 'status': 'missing', 'reason': 'its run folder is not on this computer'})
            continue
        try:
            if kind == 'tab':
                entry, rows = tabular_task(pipe, name, a.max_rows)
            else:
                entry, rows, jids = crystal_task(pipe, name, a.max_rows)
                crystal_jids += jids
        except Exception as ex:  # noqa: BLE001 - one broken task must not stop the export
            index['tasks'].append({**base, 'status': 'error', 'reason': _no_paths(repr(ex))[:300]})
            continue
        item = {**base, **entry}
        if rows is not None:
            fname = k.replace(':', '_') + '.json'
            write(out / fname, {**rows, 'generated': index['generated'], 'rehearsal': rehearsal, 'trained': base['trained'], 'title': base['title']})
            item['rows_file'] = fname
        index['tasks'].append(item)
        print(f'{k}: {item.get("status")}' + (f', {rows["rows_shown"]} rows' if rows else ''), flush=True)
    if crystal_jids:
        st, skipped = structures(sorted(set(crystal_jids)), a.data, a.max_atoms)
        write(out / 'structures.json', {'generated': index['generated'], 'source': 'JARVIS-DFT dft_3d (CC BY 4.0, figshare 10.6084/m9.figshare.6815699)',
                                        'max_atoms': a.max_atoms, 'skipped_larger_cells': skipped, 'cells': st})
        index['structures_file'] = 'structures.json'
        print(f'structures: {len(st)} unit cells ({skipped} larger cells left out)', flush=True)
    if not a.no_browser_models:
        write_browser_models(out, index['generated'], index)
    write(out / 'index.json', index)
    print(f'wrote {out / "index.json"} ({len(index["tasks"])} tasks{"; REHEARSAL: the numbers mean nothing" if rehearsal else ""})')
    return index


if __name__ == '__main__':
    main()
