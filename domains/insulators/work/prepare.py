#!/usr/bin/env python3
"""Prepare the insulators (dielectric constant) data for the browser network.

IMPORTANT: the target (poly_total) is CALCULATED with density functional theory
(DFPT, Petousis et al. / Materials Project), not measured in a lab. Any model
trained on it is a fast stand-in for that calculation.

Usage: python3 -I prepare.py <dielectric_table.csv> <out_dir>
"""
import csv, json, math, re, sys
from functools import reduce
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

NOTE = ("Targets are DFT-CALCULATED dielectric constants (poly_total, Petousis et al. 2017, "
        "Materials Project DFPT; Foundry dataset dielectric_constant_v1-1, CC BY 4.0, "
        "DOI 10.18126/racd-go9m), NOT laboratory measurements. y = log10(poly_total).")

# Element table, typed in by hand (not downloaded). Columns:
#   Z, period, group, Pauling electronegativity, standard atomic weight (u), covalent radius (pm)
# Sources: Z/period/group = periodic table; electronegativity = Pauling scale as tabulated in the
# CRC Handbook / Wikipedia "Electronegativities of the elements (data page)"; atomic weight = IUPAC
# abridged standard atomic weights; covalent radius = Cordero et al., Dalton Trans. 2008, 2832
# (C sp3; Mn, Fe, Co low-spin values). Typed from memory, spot-checked by the asserts below.
# Valence electrons are DERIVED from the group: group for groups 1-12, group-10 for groups 13-18.
EL = {
    'H': (1, 1, 1, 2.20, 1.008, 31), 'Li': (3, 2, 1, 0.98, 6.94, 128), 'Be': (4, 2, 2, 1.57, 9.0122, 96),
    'B': (5, 2, 13, 2.04, 10.81, 84), 'C': (6, 2, 14, 2.55, 12.011, 76), 'N': (7, 2, 15, 3.04, 14.007, 71),
    'O': (8, 2, 16, 3.44, 15.999, 66), 'F': (9, 2, 17, 3.98, 18.998, 57), 'Na': (11, 3, 1, 0.93, 22.990, 166),
    'Mg': (12, 3, 2, 1.31, 24.305, 141), 'Al': (13, 3, 13, 1.61, 26.982, 121), 'Si': (14, 3, 14, 1.90, 28.085, 111),
    'P': (15, 3, 15, 2.19, 30.974, 107), 'S': (16, 3, 16, 2.58, 32.06, 105), 'Cl': (17, 3, 17, 3.16, 35.45, 102),
    'K': (19, 4, 1, 0.82, 39.098, 203), 'Ca': (20, 4, 2, 1.00, 40.078, 176), 'Sc': (21, 4, 3, 1.36, 44.956, 170),
    'Ti': (22, 4, 4, 1.54, 47.867, 160), 'V': (23, 4, 5, 1.63, 50.942, 153), 'Cr': (24, 4, 6, 1.66, 51.996, 139),
    'Mn': (25, 4, 7, 1.55, 54.938, 139), 'Fe': (26, 4, 8, 1.83, 55.845, 132), 'Co': (27, 4, 9, 1.88, 58.933, 126),
    'Ni': (28, 4, 10, 1.91, 58.693, 124), 'Cu': (29, 4, 11, 1.90, 63.546, 132), 'Zn': (30, 4, 12, 1.65, 65.38, 122),
    'Ga': (31, 4, 13, 1.81, 69.723, 122), 'Ge': (32, 4, 14, 2.01, 72.630, 120), 'As': (33, 4, 15, 2.18, 74.922, 119),
    'Se': (34, 4, 16, 2.55, 78.971, 120), 'Br': (35, 4, 17, 2.96, 79.904, 120), 'Rb': (37, 5, 1, 0.82, 85.468, 220),
    'Sr': (38, 5, 2, 0.95, 87.62, 195), 'Y': (39, 5, 3, 1.22, 88.906, 190), 'Zr': (40, 5, 4, 1.33, 91.224, 175),
    'Nb': (41, 5, 5, 1.60, 92.906, 164), 'Mo': (42, 5, 6, 2.16, 95.95, 154), 'Ru': (44, 5, 8, 2.20, 101.07, 146),
    'Rh': (45, 5, 9, 2.28, 102.906, 142), 'Pd': (46, 5, 10, 2.20, 106.42, 139), 'Ag': (47, 5, 11, 1.93, 107.868, 145),
    'Cd': (48, 5, 12, 1.69, 112.414, 144), 'In': (49, 5, 13, 1.78, 114.818, 142), 'Sn': (50, 5, 14, 1.96, 118.710, 139),
    'Sb': (51, 5, 15, 2.05, 121.760, 139), 'Te': (52, 5, 16, 2.10, 127.60, 138), 'I': (53, 5, 17, 2.66, 126.904, 139),
    'Cs': (55, 6, 1, 0.79, 132.905, 244), 'Ba': (56, 6, 2, 0.89, 137.327, 215), 'La': (57, 6, 3, 1.10, 138.905, 207),
    'Hf': (72, 6, 4, 1.30, 178.49, 175), 'Ta': (73, 6, 5, 1.50, 180.948, 170), 'W': (74, 6, 6, 2.36, 183.84, 162),
    'Re': (75, 6, 7, 1.90, 186.207, 151), 'Os': (76, 6, 8, 2.20, 190.23, 144), 'Ir': (77, 6, 9, 2.20, 192.217, 141),
    'Pt': (78, 6, 10, 2.28, 195.084, 136), 'Au': (79, 6, 11, 2.54, 196.967, 136), 'Hg': (80, 6, 12, 2.00, 200.592, 132),
    'Tl': (81, 6, 13, 1.62, 204.38, 145), 'Pb': (82, 6, 14, 2.33, 207.2, 146), 'Bi': (83, 6, 15, 2.02, 208.980, 148),
}
PROPS = ['Z', 'period', 'group', 'electronegativity', 'mass', 'covalent_radius', 'valence_electrons']
ANIONS = ['O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I', 'N']
SYSTEMS = [('triclinic', 1, 2), ('monoclinic', 3, 15), ('orthorhombic', 16, 74), ('tetragonal', 75, 142),
           ('trigonal', 143, 167), ('hexagonal', 168, 194), ('cubic', 195, 230)]


def nval(group):
    return group if group <= 12 else group - 10


def props(el):
    z, per, grp, en, mass, rad = EL[el]
    return [z, per, grp, en, mass, rad, nval(grp)]


def parse(formula):
    """Formula string -> {element: count}. Handles nested parentheses."""
    tokens = re.findall(r'([A-Z][a-z]?|\(|\)|\d+\.?\d*)', formula)
    assert ''.join(tokens) == formula, formula
    stack = [{}]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        mult = 1.0
        if t == '(':
            stack.append({}); i += 1; continue
        if i + 1 < len(tokens) and re.fullmatch(r'\d+\.?\d*', tokens[i + 1]):
            mult = float(tokens[i + 1]); step = 2
        else:
            step = 1
        if t == ')':
            grp = stack.pop()
            for k, v in grp.items():
                stack[-1][k] = stack[-1].get(k, 0) + v * mult
        else:
            stack[-1][t] = stack[-1].get(t, 0) + mult
        i += step
    assert len(stack) == 1, formula
    return stack[0]


def reduced_key(counts):
    ints = [int(round(v)) for v in counts.values()]
    assert all(abs(v - round(v)) < 1e-9 for v in counts.values())
    g = reduce(math.gcd, ints)
    return ''.join(f"{el}{int(round(counts[el])) // g}" for el in sorted(counts))


def self_checks():
    # periodic-table consistency of the typed table
    for el, (z, per, grp, en, mass, rad) in EL.items():
        assert 1 <= per <= 6 and 1 <= grp <= 17 and 0.7 <= en <= 4.0 and 25 <= rad <= 250, el
        assert 1.0 * z <= mass <= 2.7 * z + 2, (el, z, mass)  # mass/Z between ~1 and ~2.6
    zs = sorted(EL.values())
    for a, b in zip(zs, zs[1:]):
        assert (b[1], b[2] if b[1] == a[1] else 99) > (a[1], a[2] if b[1] == a[1] else 0), (a, b)
    # molar masses I know: NaCl 58.44, SiO2 60.08, BaTiO3 233.19, Al2O3 101.96, Mg(OH)2 58.32
    mm = lambda f: sum(EL[e][4] * n for e, n in parse(f).items())
    for f, m in [('NaCl', 58.44), ('SiO2', 60.08), ('BaTiO3', 233.19), ('Al2O3', 101.96), ('Mg(HO)2', 58.32)]:
        assert abs(mm(f) - m) < 0.02, (f, mm(f))
    # known electronegativity orderings: F highest, Cs lowest, O > N > C > B, Cl > Br > I
    en = {e: v[3] for e, v in EL.items()}
    assert max(en, key=en.get) == 'F' and min(en, key=en.get) == 'Cs'
    assert en['O'] > en['N'] > en['C'] > en['B'] and en['Cl'] > en['Br'] > en['I']
    # radii: alkali metals grow down the group, halogens too
    r = {e: v[5] for e, v in EL.items()}
    assert r['Li'] < r['Na'] < r['K'] < r['Rb'] < r['Cs'] and r['F'] < r['Cl'] < r['Br'] < r['I']
    assert parse('Ba(SbO3)2') == {'Ba': 1, 'Sb': 2, 'O': 6} and reduced_key(parse('Mg(HO)2')) == 'H2Mg1O2'


def featurise(row):
    counts = parse(row['formula'])
    tot = sum(counts.values())
    els = list(counts)
    frac = np.array([counts[e] / tot for e in els])
    P = np.array([props(e) for e in els], dtype=float)  # elements x props
    mean = frac @ P
    spread = P.max(0) - P.min(0)
    std = np.sqrt(frac @ (P - mean) ** 2)
    nsites = float(row['nsites']); vol = float(row['volume'])
    vpa = vol / nsites
    mean_mass = mean[PROPS.index('mass')]
    density = mean_mass * 1.66054 / vpa  # g/cm3
    sphere = sum(f * 4 / 3 * math.pi * (EL[e][5] / 100) ** 3 for e, f in zip(els, frac))  # A^3 per atom
    packing = sphere / vpa
    sg = int(row['space_group'])
    onehot = [1.0 if lo <= sg <= hi else 0.0 for _, lo, hi in SYSTEMS]
    assert sum(onehot) == 1
    x = []
    for j in range(len(PROPS)):
        x += [mean[j], spread[j], std[j]]
    x += [float(len(els))]
    x += [counts.get(a, 0) / tot for a in ANIONS]
    x += [vpa, nsites, density, packing]
    x += onehot
    return x, reduced_key(counts)


FEATURES = ([f"{s}_{p}" for p in PROPS for s in ('mean', 'range', 'std')] + ['n_elements']
            + [f"frac_{a}" for a in ANIONS]
            + ['volume_per_atom', 'nsites', 'density_g_cm3', 'packing_fraction_covalent']
            + [f"system_{n}" for n, _, _ in SYSTEMS])


def main():
    table, out = sys.argv[1], sys.argv[2]
    self_checks()
    rows = list(csv.DictReader(open(table, newline='')))
    X, y, keys, bg = [], [], [], []
    for r in rows:
        x, key = featurise(r)
        t = float(r['poly_total'])
        assert t > 0 and all(math.isfinite(v) for v in x)
        X.append(x); y.append(math.log10(t)); keys.append(key); bg.append(float(r['band_gap']))
    assert len(FEATURES) == len(X[0])
    uniq = sorted(set(keys)); gid = {k: i for i, k in enumerate(uniq)}
    g = [gid[k] for k in keys]
    X = np.array(X); y = np.array(y); g = np.array(g); bg = np.array(bg)
    idx = np.arange(len(rows))

    def split(seed):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(idx, groups=g))
        assert not set(g[tr]) & set(g[te])
        return tr, te

    tr, te = split(42)
    rnd = lambda a: [[round(float(v), 6) for v in r] for r in a]

    def pack(Xm, feats):
        return {"features": feats,
                "train": {"X": rnd(Xm[tr]), "y": [round(float(v), 6) for v in y[tr]], "g": [int(v) for v in g[tr]]},
                "test": {"X": rnd(Xm[te]), "y": [round(float(v), 6) for v in y[te]], "g": [int(v) for v in g[te]]},
                "note": NOTE}

    json.dump(pack(X, FEATURES), open(f"{out}/nn_data.json", 'w'))
    Xbg = np.column_stack([X, bg])
    json.dump(pack(Xbg, FEATURES + ['band_gap_eV_calculated']), open(f"{out}/nn_data_bandgap.json", 'w'))

    splits = {"note": NOTE + " Row indices refer to all_rows.json / the CSV row order (0-based).", "splits": []}
    for seed in range(1001, 1009):
        a, b = split(seed)
        splits["splits"].append({"seed": seed, "train": a.tolist(), "test": b.tolist()})
    json.dump(splits, open(f"{out}/splits.json", 'w'))

    # 3-fold grouped CV on the TRAINING rows of the seed-42 split (indices into train arrays)
    folds = [te_.tolist() for _, te_ in GroupKFold(n_splits=3).split(tr, groups=g[tr])]
    json.dump({"note": NOTE + " Fold = positions within nn_data.json train arrays held out in that fold.",
               "folds": folds}, open(f"{out}/cv_folds.json", 'w'))

    json.dump({"note": NOTE, "features": FEATURES, "X": rnd(X), "y": [round(float(v), 6) for v in y],
               "g": g.tolist(), "group_names": uniq, "band_gap": bg.tolist(),
               "formula": [r['formula'] for r in rows], "material_id": [r['material_id'] for r in rows],
               "main_split": {"seed": 42, "train": tr.tolist(), "test": te.tolist()}},
              open(f"{out}/all_rows.json", 'w'))
    print(json.dumps({"rows": len(rows), "groups": len(uniq), "features": len(FEATURES),
                      "train_rows": len(tr), "test_rows": len(te),
                      "train_groups": len(set(g[tr])), "test_groups": len(set(g[te])),
                      "fold_sizes": [len(f) for f in folds]}))


if __name__ == '__main__':
    main()
