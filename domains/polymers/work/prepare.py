# Builds model inputs for the polymer Tg task. Usage:
#   python3 -I prepare.py <copolymer csv> <homopolymer csv> <out dir>
import sys, json, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdFingerprintGenerator
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
RDLogger.DisableLog('rdApp.*')
copo_path, homo_path, W = sys.argv[1:4]
NFP = 16

def canon(s):
    for p in ('[Th]', '[Lr]', '[Ce]'):
        s = s.replace(p, '[*]')
    m = Chem.MolFromSmiles(s)
    return None if m is None else Chem.MolToSmiles(m)

DESC = ['MolWt', 'LogP', 'TPSA', 'RotBonds', 'Rings', 'AromRings', 'FracCSP3', 'HBD', 'HBA', 'HeavyAtoms',
        'BackboneLen', 'SideHeavy', 'RotPerHeavy', 'AromAtomFrac', 'BackboneRingFrac', 'Halogens', 'Si']
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NFP)
_cache = {}
def describe(smi):
    if smi in _cache: return _cache[smi]
    m = Chem.MolFromSmiles(smi)
    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    heavy = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)
    if len(stars) == 2:
        path = list(Chem.GetShortestPath(m, stars[0], stars[1]))[1:-1]
    else:
        path = []
    bl = len(path)
    ring = sum(1 for i in path if m.GetAtomWithIdx(i).IsInRing())
    rot = rdMolDescriptors.CalcNumRotatableBonds(m)
    arom = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
    d = [Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m), rot,
         rdMolDescriptors.CalcNumRings(m), rdMolDescriptors.CalcNumAromaticRings(m),
         rdMolDescriptors.CalcFractionCSP3(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
         heavy, bl, heavy - bl, rot / max(heavy, 1), arom / max(heavy, 1), ring / max(bl, 1),
         sum(1 for a in m.GetAtoms() if a.GetSymbol() in ('F', 'Cl', 'Br', 'I')),
         sum(1 for a in m.GetAtoms() if a.GetSymbol() == 'Si')]
    fp = np.log1p(fpgen.GetCountFingerprintAsNumPy(m).astype(float))
    _cache[smi] = (np.array(d, float), fp, len(stars))
    return _cache[smi]

# ---- load and clean the copolymer file
raw = pd.read_csv(copo_path)
raw = raw.loc[:, ~raw.columns.str.startswith('Unnamed')]
rep = {'copolymer_file_rows_raw': int(len(raw)), 'columns_kept': list(raw.columns)}
rep['missing_SMILES_1_rows'] = int(raw.SMILES_1.isna().sum())
rep['missing_SMILES_1_all_comp0_100'] = bool((raw.loc[raw.SMILES_1.isna(), 'comp_0'] == 100).all())
rep['missing_Tg'] = int(raw['Tg(K)'].isna().sum())
rep['comp_sums_to_100'] = bool(np.allclose(raw.comp_0 + raw.comp_1, 100))
rep['x1_equals_comp0_over_100'] = bool(np.allclose(raw.x1 * 100, raw.comp_0))
rows = []
bad = 0
for r in raw.itertuples(index=False):
    s0 = canon(r.SMILES_0)
    s1 = canon(r.SMILES_1) if isinstance(r.SMILES_1, str) else None
    if s0 is None or (isinstance(r.SMILES_1, str) and s1 is None):
        bad += 1; continue
    x0 = r.comp_0 / 100.0
    if s1 is None or s1 == s0:
        a = b = s0; xa = 1.0
    else:
        a, b = sorted([s0, s1]); xa = x0 if a == s0 else 1 - x0
    rows.append(dict(id=int(r.ID), a=a, b=b, xa=round(float(xa), 6), tg=float(r[7])))
rep['unparseable_rows_dropped'] = bad
df = pd.DataFrame(rows)
ndup = int(df.duplicated(['a', 'b', 'xa']).sum())
rep['duplicate_structure_composition_rows'] = ndup
rep['duplicate_rows_tg_spread_K_max'] = float(df.groupby(['a', 'b', 'xa']).tg.agg(lambda v: v.max() - v.min()).max())
df = df.groupby(['a', 'b', 'xa'], as_index=False).agg(id=('id', 'first'), tg=('tg', 'mean'))
df['gkey'] = df.a + ' || ' + df.b
gids = {k: i for i, k in enumerate(sorted(df.gkey.unique()))}
df['g'] = df.gkey.map(gids)
df['pure'] = [a if xa == 1 else (b if xa == 0 else None) for a, b, xa in zip(df.a, df.b, df.xa)]
n = len(df)
monomers = sorted(set(df.a) | set(df.b))
rep.update(rows_used_copolymer_file=n, groups=len(gids), distinct_monomers=len(monomers),
           copolymer_systems=int((df.a != df.b).groupby(df.g).first().sum()),
           rows_true_copolymer=int(((df.xa > 0) & (df.xa < 1)).sum()), rows_homopolymer=int(df.pure.notna().sum()),
           tg_min=float(df.tg.min()), tg_max=float(df.tg.max()), tg_mean=float(df.tg.mean()))

# ---- compare with the homopolymer file (target only; Calculated* columns are never read)
hraw = pd.read_csv(homo_path)
smi_col = [c for c in hraw.columns if c.startswith('SMILES')][0]
h = pd.DataFrame({'smi': [canon(s) for s in hraw[smi_col]], 'tg': hraw['Experiment Tg (K)']}).dropna()
pure_tg = df[df.pure.notna()].groupby('pure').tg.mean()
both = h[h.smi.isin(pure_tg.index)]
dtg = (both.tg.values - pure_tg.loc[both.smi].values)
rep['homopolymer_file'] = dict(rows=int(len(hraw)), rows_with_experiment_tg=int(len(h)), distinct_structures=int(h.smi.nunique()),
    structures_already_in_copolymer_file_as_homopolymer=int(both.smi.nunique()),
    rows_not_in_copolymer_file=int((~h.smi.isin(pure_tg.index)).sum()),
    same_tg_within_0p5K=int((np.abs(dtg) <= 0.5).sum()), max_abs_tg_diff_K=float(np.abs(dtg).max()) if len(dtg) else None)
rep['rows_used_homopolymer_file'] = 0

# ---- features
A = np.array([describe(s)[0] for s in df.a]); B = np.array([describe(s)[0] for s in df.b])
FA = np.array([describe(s)[1] for s in df.a]); FB = np.array([describe(s)[1] for s in df.b])
rep['monomers_without_two_placeholders'] = int(sum(1 for s in monomers if describe(s)[2] != 2))

def base_block(a, b, xa):
    da, fa, _ = describe(a); db, fb, _ = describe(b)
    mean = xa * da + (1 - xa) * db
    wa = xa * da[0] / (xa * da[0] + (1 - xa) * db[0])  # weight fraction, assuming comp is mole percent
    return mean, np.abs(da - db), xa * fa + (1 - xa) * fb, wa

def fox_block(a, b, wa, g, lookup, fill):
    parts = [(a, wa)] if a == b else [(a, wa), (b, 1 - wa)]
    known_w = 0.0; s_lin = 0.0; s_inv = 0.0; ok = True
    for mono, w in parts:
        vals = [t for gg, t in lookup.get(mono, []) if gg != g]  # never the row's own system
        if vals:
            t = float(np.mean(vals)); known_w += w; s_lin += w * t; s_inv += w / t
        elif w > 1e-9:
            ok = False
    lin = s_lin / known_w if known_w > 1e-9 else fill
    fox = 1.0 / s_inv if (ok and s_inv > 0) else lin
    return [fox, 1.0 if (ok and known_w > 1e-9) else 0.0, known_w, lin]

def matrix(a_list, b_list, xa_list, g_list, train_idx, use_fp):
    lookup = {}
    for i in train_idx:
        p = df.pure.iat[i]
        if p is not None: lookup.setdefault(p, []).append((int(df.g.iat[i]), float(df.tg.iat[i])))
    fill = float(df.tg.values[train_idx].mean())
    X = []
    for a, b, xa, g in zip(a_list, b_list, xa_list, g_list):
        mean, diff, fp, wa = base_block(a, b, xa)
        row = list(mean) + list(diff) + [xa, 4 * xa * (1 - xa)] + fox_block(a, b, wa, g, lookup, fill)
        if use_fp: row += list(fp)
        X.append([float(f'{v:.6g}') for v in row])
    return X

NAMES = [f'mean_{d}' for d in DESC] + [f'absdiff_{d}' for d in DESC] + ['x_a', 'mix_4x(1-x)',
         'fox_tg_from_train', 'fox_available', 'homoTg_known_wfrac', 'homoTg_known_wmean']
FPN = [f'mean_morgan{k}' for k in range(NFP)]
y = df.tg.values; g = df.g.values

def task(tr, te, use_fp):
    tr = np.array(sorted(tr)); te = np.array(sorted(te))
    def side(idx):
        return {'X': matrix(df.a.values[idx], df.b.values[idx], df.xa.values[idx], g[idx], tr, use_fp),
                'y': [float(v) for v in y[idx]], 'g': [int(v) for v in g[idx]], 'idx': [int(v) for v in idx]}
    return {'train': side(tr), 'test': side(te)}

tr_main, te_main = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(np.zeros(n), y, g))
cv = [(tr_main[a_], tr_main[b_]) for a_, b_ in GroupKFold(n_splits=3).split(tr_main, y[tr_main], g[tr_main])]

# feature-set choice (Morgan bits or not): tree model, 3-fold grouped CV on training rows only
def gbr_cv(use_fp):
    err = []
    for tr, va in cv:
        t = task(tr, va, use_fp)
        m = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
        m.fit(np.array(t['train']['X']), t['train']['y'])
        err += list(np.abs(m.predict(np.array(t['test']['X'])) - np.array(t['test']['y'])))
    return float(np.mean(err))
cv_nofp, cv_fp = gbr_cv(False), gbr_cv(True)
use_fp = cv_fp < cv_nofp
rep['feature_set_choice'] = dict(gbr_cv_mae_descriptors_only=cv_nofp, gbr_cv_mae_with_morgan16=cv_fp, morgan_used=bool(use_fp))
feats = NAMES + (FPN if use_fp else [])

tasks = {f'cv{k}': task(tr, va, use_fp) for k, (tr, va) in enumerate(cv)}
tasks['main'] = task(tr_main, te_main, use_fp)
splits = {'main_seed42': {'train_idx': sorted(int(i) for i in tr_main), 'test_idx': sorted(int(i) for i in te_main)}}
for seed in range(1001, 1009):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(np.zeros(n), y, g))
    tasks[f's{seed}'] = task(tr, te, use_fp)
    splits[str(seed)] = {'train_idx': sorted(int(i) for i in tr), 'test_idx': sorted(int(i) for i in te)}

# harder split: hold out whole monomers; a test row has at least one monomer that no training row contains
rng = np.random.RandomState(7)
held = set(); in_test = np.zeros(n, bool)
for mono in rng.permutation(monomers):
    cand = in_test | (df.a.values == mono) | (df.b.values == mono)
    if cand.mean() > 0.25: continue
    held.add(mono); in_test = cand
    if in_test.mean() >= 0.20: break
tr_h, te_h = np.where(~in_test)[0], np.where(in_test)[0]
train_monos = set(df.a.values[tr_h]) | set(df.b.values[tr_h])
assert all((a not in train_monos) or (b not in train_monos) for a, b in zip(df.a.values[te_h], df.b.values[te_h]))
tasks['hard'] = task(tr_h, te_h, use_fp)
splits['hard_unseen_monomer'] = {'train_idx': [int(i) for i in tr_h], 'test_idx': [int(i) for i in te_h], 'held_out_monomers': sorted(held)}
rep['hard_split'] = dict(held_out_monomers=len(held), train_rows=int(len(tr_h)), test_rows=int(len(te_h)),
                         test_rows_true_copolymer=int(((df.xa.values[te_h] > 0) & (df.xa.values[te_h] < 1)).sum()))

# composition sweep for copolymer systems in the main test split (features use main training rows only)
sw = {'X': [], 'g': [], 'x': []}
for gid in sorted(set(g[te_main])):
    sub = df[df.g == gid]
    a, b = sub.a.iat[0], sub.b.iat[0]
    if a == b: continue
    xs = np.linspace(0, 1, 21)
    sw['X'] += matrix([a] * 21, [b] * 21, xs, [gid] * 21, tr_main, use_fp)
    sw['g'] += [int(gid)] * 21; sw['x'] += [float(v) for v in xs]
tasks['main']['extra'] = sw

main = tasks['main']
json.dump({'features': feats, 'train': {k: main['train'][k] for k in 'Xyg'}, 'test': {k: main['test'][k] for k in 'Xyg'}},
          open(f'{W}/nn_data.json', 'w'))
json.dump(splits, open(f'{W}/splits.json', 'w'))
json.dump({'features': feats, 'tasks': tasks}, open(f'{W}/tasks.json', 'w'))
df.drop(columns='gkey').to_json(f'{W}/rows.json', orient='records')
rep['main_split'] = dict(train_rows=int(len(tr_main)), test_rows=int(len(te_main)), train_groups=int(len(set(g[tr_main]))),
    test_groups=int(len(set(g[te_main]))), test_copolymer_systems=int(len(set(sw['g']))),
    test_rows_true_copolymer=int(((df.xa.values[te_main] > 0) & (df.xa.values[te_main] < 1)).sum()))
rep['features'] = feats
json.dump(rep, open(f'{W}/prepare_report.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != 'features'}, indent=1))
