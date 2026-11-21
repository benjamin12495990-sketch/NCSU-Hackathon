#!/usr/bin/env python3
"""Builds model inputs for Vickers hardness from the MPEA dataset.

Usage: python3 -I prepare.py <MPEA_dataset.csv> <out_dir>

Inputs used: FORMULA and "PROPERTY: Processing method" only. Everything else
in the file (phases, microstructure, grain size, densities, moduli, strengths,
test type, references) is known only after the alloy is made, or is another
measurement, and is deliberately ignored.
"""
import csv, json, math, re, sys
from collections import Counter, defaultdict
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Small element table, typed in by hand (from memory, not re-checked against
# the sources in this session):
#   r   metallic radius in pm, values as used in the high-entropy-alloy
#       literature (Senkov & Miracle, Mater. Res. Bull. 36 (2001) 2183;
#       reused by Guo & Liu, Prog. Nat. Sci.: Mater. Int. 21 (2011) 433)
#   chi Pauling electronegativity (CRC Handbook of Chemistry and Physics)
#   vec valence electron count including d electrons (Guo, Ng, Lu, Liu,
#       J. Appl. Phys. 109 (2011) 103505)
#   tm  melting point in K (CRC Handbook)
ELEMENTS = {
    'Al': (143.2, 1.61, 3, 933), 'Co': (125.1, 1.88, 9, 1768), 'Cr': (124.9, 1.66, 6, 2180),
    'Cu': (127.8, 1.90, 11, 1358), 'Fe': (124.1, 1.83, 8, 1811), 'Ni': (124.6, 1.91, 10, 1728),
    'Mn': (135.0, 1.55, 7, 1519), 'Ti': (146.2, 1.54, 4, 1941), 'V': (131.6, 1.63, 5, 2183),
    'Nb': (142.9, 1.60, 5, 2750), 'Mo': (136.3, 2.16, 6, 2896), 'Ta': (143.0, 1.50, 5, 3290),
    'W': (136.7, 2.36, 6, 3695), 'Zr': (160.3, 1.33, 4, 2128), 'Hf': (157.8, 1.30, 4, 2506),
    'Si': (115.3, 1.90, 4, 1687), 'C': (77.3, 2.55, 4, 3823), 'B': (82.0, 2.04, 3, 2349),
    'Sn': (162.0, 1.96, 4, 505), 'Zn': (139.5, 1.65, 12, 693), 'Mg': (160.1, 1.31, 2, 923),
    'Li': (151.9, 0.98, 1, 454), 'Sc': (164.1, 1.36, 3, 1814), 'Y': (180.2, 1.22, 3, 1799),
    'Re': (137.5, 1.90, 7, 3459), 'Pd': (137.5, 2.20, 10, 1828), 'Ag': (144.5, 1.93, 11, 1235),
    'Au': (144.2, 2.54, 11, 1337), 'Nd': (182.1, 1.14, 3, 1297), 'Ge': (124.0, 2.01, 4, 1211),
    'Ga': (139.2, 1.81, 3, 303), 'Ca': (197.6, 1.00, 2, 1115), 'Be': (112.8, 1.57, 2, 1560),
    'N': (75.0, 3.04, 5, 63), 'Ir': (135.7, 2.20, 9, 2719), 'Ru': (133.8, 2.20, 8, 2607),
    'Rh': (134.5, 2.28, 9, 2237), 'Pt': (138.7, 2.28, 10, 2041), 'Os': (135.2, 2.20, 8, 3306),
    'Gd': (180.1, 1.20, 3, 1586), 'Tb': (178.1, 1.20, 3, 1629), 'Dy': (177.4, 1.22, 3, 1680),
    'Ho': (176.6, 1.23, 3, 1747), 'Er': (175.7, 1.24, 3, 1802), 'Tm': (174.6, 1.25, 3, 1818),
    'Lu': (173.5, 1.27, 3, 1925), 'La': (187.9, 1.10, 3, 1193), 'Ce': (182.5, 1.12, 3, 1071),
    'Pr': (182.8, 1.13, 3, 1204), 'Sm': (180.4, 1.17, 3, 1345), 'P': (106.0, 2.19, 5, 317),
    'S': (103.0, 2.58, 6, 388), 'O': (64.0, 3.44, 6, 54), 'Sb': (161.0, 2.05, 5, 904),
    'Bi': (162.0, 2.02, 5, 545), 'Pb': (175.0, 2.33, 4, 601), 'In': (165.9, 1.78, 3, 430),
}
HV, FORMULA, PROC = 'PROPERTY: HV', 'FORMULA', 'PROPERTY: Processing method'
TOKEN = re.compile(r'([A-Z][a-z]?)(\d*\.?\d*)')


def parse(formula):
    """'Al0.25 Co1 Fe1 Ni1' -> {'Al': 0.0769, ...} with fractions summing to 1."""
    s = formula.replace(' ', '')
    out, pos = defaultdict(float), 0
    for m in TOKEN.finditer(s):
        if m.start() != pos:
            raise ValueError('cannot parse formula: %r' % formula)
        pos = m.end()
        out[m.group(1)] += float(m.group(2)) if m.group(2) else 1.0
    if pos != len(s) or not out:
        raise ValueError('cannot parse formula: %r' % formula)
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items() if v > 0}


def norm_formula(frac):
    return ' '.join('%s%.3f' % (el, frac[el]) for el in sorted(frac))


def descriptors(frac):
    xs = np.array(list(frac.values()))
    tab = np.array([ELEMENTS[el] for el in frac], dtype=float)
    r, chi, vec, tm = tab.T
    rbar, cbar, vbar = xs @ r, xs @ chi, xs @ vec
    return {
        'n_elements': float(len(frac)),
        'mix_entropy_R': float(-(xs * np.log(xs)).sum()),  # -sum x ln x, in units of R
        'radius_mean_pm': float(rbar),
        'radius_mismatch_pct': float(100 * math.sqrt((xs * (1 - r / rbar) ** 2).sum())),
        'eneg_mean': float(cbar),
        'eneg_spread': float(math.sqrt((xs * (chi - cbar) ** 2).sum())),
        'vec_mean': float(vbar),
        'vec_spread': float(math.sqrt((xs * (vec - vbar) ** 2).sum())),
        'tmelt_mean_K': float(xs @ tm),
    }


def main(csv_path, out_dir):
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    header = list(rows[0].keys())
    for col in (HV, FORMULA, PROC):
        assert col in header, 'missing column %r' % col
    kept = []
    for r in rows:
        v = (r[HV] or '').strip()
        if not v:
            continue
        hv = float(v)
        if not math.isfinite(hv) or hv <= 0:
            continue
        frac = parse(r[FORMULA])
        missing = [el for el in frac if el not in ELEMENTS]
        if missing:
            raise SystemExit('element(s) not in table: %s (formula %r)' % (missing, r[FORMULA]))
        proc = (r[PROC] or '').strip().upper() or 'UNKNOWN'
        kept.append((frac, proc, hv, r[FORMULA]))

    el_rows = Counter(el for frac, *_ in kept for el in frac)
    elements = sorted(el for el, c in el_rows.items() if c >= 5)
    rare = sorted(el for el, c in el_rows.items() if c < 5)
    procs = sorted({p for _, p, _, _ in kept})
    desc_names = list(descriptors(kept[0][0]).keys())
    features = ['x_' + el for el in elements] + ['x_other'] + ['proc_' + p for p in procs] + desc_names

    def row_features(frac, proc):
        d = descriptors(frac)
        return ([round(frac.get(el, 0.0), 6) for el in elements]
                + [round(sum(v for el, v in frac.items() if el not in elements), 6)]
                + [1.0 if proc == p else 0.0 for p in procs]
                + [round(d[k], 6) for k in desc_names])

    X = [row_features(frac, proc) for frac, proc, _, _ in kept]
    y = [hv for _, _, hv, _ in kept]
    g = [norm_formula(frac) for frac, _, _, _ in kept]
    Xa, ya, ga = np.array(X), np.array(y), np.array(g)

    def split(seed):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(Xa, ya, ga))
        assert not set(ga[tr]) & set(ga[te])
        return tr.tolist(), te.tolist()

    tr, te = split(42)
    pick = lambda idx: {'X': [X[i] for i in idx], 'y': [y[i] for i in idx], 'g': [g[i] for i in idx]}
    with open(out_dir + '/nn_data.json', 'w') as f:
        json.dump({'features': features, 'train': pick(tr), 'test': pick(te)}, f)

    # Eight more grouped splits over the same rows, for the paired comparison.
    extra = []
    for seed in range(1001, 1009):
        a, b = split(seed)
        extra.append({'seed': seed, 'train': a, 'test': b})
    with open(out_dir + '/splits.json', 'w') as f:
        json.dump({'features': features, 'X': X, 'y': y, 'g': g, 'main_split': {'seed': 42, 'train': tr, 'test': te}, 'splits': extra}, f)

    # Three grouped folds over the TRAINING rows only, shared by the network and the trees.
    tr_groups = sorted(set(ga[tr]))
    np.random.RandomState(7).shuffle(tr_groups)
    fold_of = {grp: i % 3 for i, grp in enumerate(tr_groups)}
    with open(out_dir + '/cv_folds.json', 'w') as f:
        json.dump({'fold': [fold_of[g[i]] for i in tr]}, f)

    # A harder question, still on training rows only: hold out whole alloy
    # systems (every alloy made of the same set of elements) instead of single formulas.
    system = [' '.join(sorted(frac)) for frac, _, _, _ in kept]
    tr_sys = sorted({system[i] for i in tr})
    np.random.RandomState(7).shuffle(tr_sys)
    sys_fold = {s: i % 3 for i, s in enumerate(tr_sys)}
    with open(out_dir + '/cv_folds_system.json', 'w') as f:
        json.dump({'fold': [sys_fold[system[i]] for i in tr]}, f)
    system_stats = {'alloy_systems_all_rows': len(set(system)), 'alloy_systems_in_train': len(tr_sys),
                    'test_rows_whose_element_set_is_also_in_train': sum(1 for i in te if system[i] in sys_fold)}

    # Sanity-check sweep: Al_x Co Cr Fe Ni, as cast. Model inputs only.
    sweep, measured = [], []
    for k in range(0, 21):
        x = k / 10
        frac = parse(('Al%g ' % x if x > 0 else '') + 'Co1 Cr1 Fe1 Ni1')
        sweep.append({'al_x': x, 'al_fraction': round(frac.get('Al', 0.0), 4), 'features': row_features(frac, 'CAST')})
    test_set = set(te)
    for i, (frac, proc, hv, raw) in enumerate(kept):
        if set(frac) <= {'Al', 'Co', 'Cr', 'Fe', 'Ni'} and {'Co', 'Cr', 'Fe', 'Ni'} <= set(frac):
            others = [frac[e] for e in ('Co', 'Cr', 'Fe', 'Ni')]
            if max(others) - min(others) < 1e-6:
                measured.append({'formula': raw, 'al_x': round(frac.get('Al', 0.0) / others[0], 3), 'proc': proc, 'hv': hv,
                                 'side': 'test' if i in test_set else 'train'})
    with open(out_dir + '/sweep.json', 'w') as f:
        json.dump({'features': features, 'sweep': sweep, 'measured_in_dataset': measured}, f)

    # How much do repeat measurements of the same inputs disagree? (noise floor)
    cells = defaultdict(list)
    for i in range(len(y)):
        cells[(g[i], kept[i][1])].append(y[i])
    dev = [abs(v - np.mean(vals)) for vals in cells.values() if len(vals) > 1 for v in vals]
    summary = {
        'csv_rows': len(rows), 'csv_columns': len(header), 'header': header,
        'columns_used_as_inputs': [FORMULA, PROC], 'target': HV,
        'rows_used': len(y), 'groups': len(set(g)), 'hv_min': min(y), 'hv_max': max(y), 'hv_mean': float(np.mean(y)),
        'n_features': len(features), 'features': features,
        'elements_with_own_column': elements, 'rare_elements_pooled_in_x_other': rare,
        'processing_counts': dict(Counter(p for _, p, _, _ in kept)),
        'train_rows': len(tr), 'test_rows': len(te),
        'train_groups': len(set(ga[tr])), 'test_groups': len(set(ga[te])), **system_stats,
        'repeat_cells': sum(1 for v in cells.values() if len(v) > 1), 'rows_in_repeat_cells': len(dev),
        'repeat_mean_abs_dev_from_cell_mean_HV': float(np.mean(dev)) if dev else None,
    }
    with open(out_dir + '/prepare_summary.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'header'}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
