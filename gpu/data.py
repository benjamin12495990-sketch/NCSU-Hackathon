"""Family loaders that read only repository files, through a guard that refuses sealed or full-data files.

Each loader returns a Dataset: training rows (X, y, groups), the family's fixed test rows (if any), grouped
CV folds over the training rows, the SHA-256 of the arrays, and the SHA-256 of every source file read.

  concrete      lab/data/train.json (train), reward.json (fixed test), monitor.json (reported separately)
  alloys        public/model/materials/nn_data.json  (+ domains/materials/work/cv_folds.json)
  insulators    public/model/insulators/nn_data.json (+ domains/insulators/work/cv_folds.json)
  polymers      public/model/polymers/nn_data.json   (GroupKFold(3) on the training groups)
  chemistry     domains/chemistry/work/prepare.py run on data_curated.csv into gpu/data_cache/chemistry/
  electrolytes  domains/electrolytes/work/data/processed.csv, train_idx.csv rows only (no test rows loaded)
  cleaning_pcmc, cleaning_tension
                domains/cleaning/work/cleaned/surfpro_clean.csv + surfpro_features.npz (SURFPRO surfactants; features exported
                once by domains/cleaning/work/export_features.py, so no RDKit is needed here); paper-held-out GroupKFold(5)
  corrosion     domains/corrosives/work/cleaned/corrdata_normalized.csv (NIST CORR-DATA; plain point-value rates only);
                reference-held-out GroupKFold(5)
"""
import csv
import io
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gpu.common import CACHE, REPO, sha256_bytes, sha256_file

# The sealed-data protocol (lab/README.md, lab/guard.mjs): never read the vault, and never read the full
# concrete files that contain the opened far-future rows.
BLOCKED_DIRS = [REPO / 'lab' / 'vault']
BLOCKED_FILES = [REPO / p for p in ('public/model/nn_data.json', 'public/model/known.json', 'public/model/test_points.json',
                                    'public/model/parity.json', 'train/data/concrete.csv')]


class SealedDataError(PermissionError):
    pass


def _norm(p):
    p = Path(p)
    return {Path(str(p.absolute())), p.resolve()}  # both the literal path and the target of any symlink


def guard_path(path):
    """Raises SealedDataError if path (or what it resolves to) is in lab/vault/ or is a blocked full-data file."""
    forms = _norm(path)
    blocked_files = set().union(*(_norm(f) for f in BLOCKED_FILES))
    blocked_dirs = set().union(*(_norm(d) for d in BLOCKED_DIRS))
    for f in forms:
        if f in blocked_files:
            raise SealedDataError(f'refused: {path} contains concrete rows from the sealed/opened future tier; use lab/data/')
        for d in blocked_dirs:
            if f == d or d in f.parents:
                raise SealedDataError(f'refused: {path} is inside lab/vault/, which is sealed')
    return Path(path)


def read_bytes(path):
    guard_path(path)
    return Path(path).read_bytes()


def read_json(path):
    return json.loads(read_bytes(path))


@dataclass
class Dataset:
    family: str
    features: list
    X: np.ndarray            # training rows
    y: np.ndarray
    groups: np.ndarray
    target: str
    X_test: np.ndarray = None  # the family's fixed test rows (None when only CV is allowed)
    y_test: np.ndarray = None
    groups_test: np.ndarray = None
    folds: np.ndarray = None   # CV fold id per training row
    folds_source: str = ''
    extra: dict = field(default_factory=dict)   # other fixed sets reported separately, name -> (X, y, groups)
    sources: dict = field(default_factory=dict)  # repo-relative path -> SHA-256 of the file read
    notes: list = field(default_factory=list)

    @property
    def data_sha256(self):
        parts = [json.dumps(self.features).encode(), np.ascontiguousarray(self.X, np.float64).tobytes(),
                 np.ascontiguousarray(self.y, np.float64).tobytes(), json.dumps([str(g) for g in self.groups]).encode(),
                 b'' if self.folds is None else np.asarray(self.folds, np.int64).tobytes()]
        if self.X_test is not None:
            parts += [np.ascontiguousarray(self.X_test, np.float64).tobytes(), np.ascontiguousarray(self.y_test, np.float64).tobytes()]
        for k in sorted(self.extra):
            parts += [k.encode(), np.ascontiguousarray(self.extra[k][0], np.float64).tobytes(), np.ascontiguousarray(self.extra[k][1], np.float64).tobytes()]
        return sha256_bytes(*parts)

    def summary(self):
        return {'family': self.family, 'target': self.target, 'features': len(self.features), 'train_rows': int(len(self.y)),
                'train_groups': int(len(set(map(str, self.groups)))), 'test_rows': None if self.y_test is None else int(len(self.y_test)),
                'cv_folds': None if self.folds is None else int(self.folds.max() + 1), 'folds_source': self.folds_source,
                'extra_sets': {k: int(len(v[1])) for k, v in self.extra.items()}, 'data_sha256': self.data_sha256,
                'sources': self.sources, 'notes': self.notes}


def _rel(p):
    # Always forward slashes, so the recorded sources are the same on Windows and Linux.
    return Path(p).resolve().relative_to(REPO.resolve()).as_posix()


def _src(ds_sources, p):
    guard_path(p)
    ds_sources[_rel(p)] = sha256_file(p)


def group_kfold(groups, k):
    from sklearn.model_selection import GroupKFold
    folds = np.zeros(len(groups), dtype=np.int64)
    for i, (_, va) in enumerate(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups)):
        folds[va] = i
    return folds


def _from_nn_data(family, path, target):
    d = read_json(path)
    tr, te = d['train'], d['test']
    ds = Dataset(family, d['features'], np.array(tr['X'], float), np.array(tr['y'], float), np.array(tr['g']), target,
                 np.array(te['X'], float), np.array(te['y'], float), np.array(te['g']))
    _src(ds.sources, path)
    return ds


def load_concrete():
    base = REPO / 'lab' / 'data'
    parts = {}
    for name in ('train', 'reward', 'monitor'):
        p = base / f'{name}.json'
        parts[name] = (read_json(p), p)
    tr = parts['train'][0]
    arr = lambda d: (np.array(d['X'], float), np.array(d['y'], float), np.array(d['g']))  # noqa: E731
    Xtr, ytr, gtr = arr(tr)
    Xre, yre, gre = arr(parts['reward'][0])
    ds = Dataset('concrete', tr['features'], Xtr, ytr, gtr, 'compressive strength, MPa (TRAIN = ages 1-7 d)', Xre, yre, gre,
                 extra={'monitor': arr(parts['monitor'][0])})
    for _, p in parts.values():
        _src(ds.sources, p)
    ds.folds = group_kfold(gtr, 3)
    ds.folds_source = 'GroupKFold(3) on TRAIN mix groups'
    ds.notes.append('fixed split: train on TRAIN (1-7 d), score on REWARD (14-28 d); MONITOR reported separately. '
                    'These data already informed project decisions, so every score is exploratory (lab/README.md).')
    return ds


def load_alloys():
    ds = _from_nn_data('alloys', REPO / 'public/model/materials/nn_data.json', 'Vickers hardness, HV')
    fp = REPO / 'domains/materials/work/cv_folds.json'
    folds = np.array(read_json(fp)['fold'], dtype=np.int64)
    assert len(folds) == len(ds.y)
    ds.folds, ds.folds_source = folds, 'domains/materials/work/cv_folds.json (3 grouped folds used by the family agent)'
    _src(ds.sources, fp)
    return ds


def load_insulators():
    ds = _from_nn_data('insulators', REPO / 'public/model/insulators/nn_data.json', 'log10(poly_total), DFT-calculated dielectric constant')
    fp = REPO / 'domains/insulators/work/cv_folds.json'
    folds = np.full(len(ds.y), -1, dtype=np.int64)
    for k, idx in enumerate(read_json(fp)['folds']):
        folds[np.array(idx, dtype=np.int64)] = k
    if (folds < 0).any():
        folds = group_kfold(ds.groups, 3)
        ds.folds_source = 'GroupKFold(3) on reduced-formula groups (cv_folds.json did not cover every row)'
    else:
        ds.folds_source = 'domains/insulators/work/cv_folds.json (3 grouped folds used by the family agent)'
        _src(ds.sources, fp)
    ds.folds = folds
    ds.notes.append('targets are DFT-calculated, not measured')
    return ds


def load_polymers():
    ds = _from_nn_data('polymers', REPO / 'public/model/polymers/nn_data.json', 'glass-transition temperature Tg, K')
    ds.folds, ds.folds_source = group_kfold(ds.groups, 3), 'GroupKFold(3) on copolymer-system groups'
    ds.notes.append('nn_data.json features fox_tg_from_train/homoTg_* were computed from the main split\'s training rows, '
                    'so CV on these fixed features is slightly optimistic compared with the agent\'s per-fold recomputation')
    return ds


def load_chemistry(rebuild=False):
    out = CACHE / 'chemistry'
    nn = out / 'nn_data.json'
    csv_path = REPO / 'domains/chemistry/work/data/data_curated.csv'
    prep = REPO / 'domains/chemistry/work/prepare.py'
    # The cache is git-ignored, so it must notice when its sources change (new data arriving by git pull): a stamp of the source
    # hashes sits beside it. A cache without a stamp (made by older code) is taken as current and stamped, never rebuilt blindly.
    stamp_path = out / 'source_stamp.json'
    stamp = {'data_curated.csv': sha256_file(guard_path(csv_path)), 'prepare.py': sha256_file(prep)}
    try:
        old_stamp = json.loads(stamp_path.read_text())
    except (OSError, ValueError):
        old_stamp = None
    if rebuild or not nn.exists() or (old_stamp is not None and old_stamp != stamp):
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-I', str(prep), str(csv_path), str(out)], check=True, stdout=subprocess.DEVNULL)
        old_stamp = None
    if old_stamp is None:
        stamp_path.write_text(json.dumps(stamp, indent=1))
    ds = _from_nn_data('chemistry', nn, 'aqueous solubility LogS, log10 mol/L')
    ds.sources = {}
    for p in (csv_path, prep):
        _src(ds.sources, p)
    ds.sources['gpu/data_cache/chemistry/nn_data.json'] = sha256_file(nn)
    rows = read_json(out / 'all_rows.json')
    ds.folds = np.array(rows['cv_fold'], dtype=np.int64)
    ds.folds_source = 'prepare.py cv_fold (GroupKFold(3) by InChIKey first block)'
    return ds


_EL = re.compile(r'([A-Z][a-z]?)(\d*\.?\d*)')


def parse_composition(s):
    """'Li7.0La3.0Zr2.0O12.0' -> {'Li': 7.0, ...}; empty amount = 1."""
    out = {}
    pos = 0
    for m in _EL.finditer(s):
        if m.start() != pos:
            raise ValueError(f'cannot parse composition {s!r}')
        pos = m.end()
        out[m.group(1)] = out.get(m.group(1), 0.0) + (float(m.group(2)) if m.group(2) else 1.0)
    if pos != len(s) or not out:
        raise ValueError(f'cannot parse composition {s!r}')
    return out


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def load_electrolytes(k=5):
    base = REPO / 'domains/electrolytes/work/data'
    pcsv, ptr = base / 'processed.csv', base / 'train_idx.csv'
    rows = list(csv.DictReader(io.StringIO(read_bytes(pcsv).decode('utf-8'))))
    train_ids = [r['ID'] for r in csv.DictReader(io.StringIO(read_bytes(ptr).decode('utf-8')))]
    by_id = {r['ID']: r for r in rows}
    assert len(by_id) == len(rows), 'duplicate IDs'
    assert len(set(train_ids)) == len(train_ids) and set(train_ids) <= set(by_id)
    keep = set(train_ids)
    tr = [r for r in rows if r['ID'] in keep]  # processed.csv order; the official test rows are never loaded here
    comps = [parse_composition(r['Composition']) for r in tr]
    elements = sorted({e for c in comps for e in c})
    X = np.zeros((len(tr), len(elements)))
    for i, c in enumerate(comps):
        tot = sum(c.values())
        for e, v in c.items():
            X[i, elements.index(e)] = v / tot
    y = np.log10(np.array([float(r['Ionic conductivity (S cm-1)']) for r in tr]))
    uf = _UF(len(tr))
    first = {}
    for i, r in enumerate(tr):
        # the same rule as domains/electrolytes/work/baseline.py: exact Composition string, DOI tokens split on '|'
        keys = ['C:' + r['Composition'].strip()] + ['D:' + t.strip().lower() for t in r['DOI'].split('|') if t.strip()]
        for key in keys:
            if key in first:
                uf.union(i, first[key])
            else:
                first[key] = i
    roots = [uf.find(i) for i in range(len(tr))]
    relabel = {r: j for j, r in enumerate(dict.fromkeys(roots))}  # 0, 1, ... in order of first appearance
    groups = np.array([relabel[r] for r in roots])
    ds = Dataset('electrolytes', [f'x_{e}' for e in elements], X, y, groups,
                 'log10 room-temperature ionic conductivity, log10(S/cm); OBELiX training partition only')
    for p in (pcsv, ptr):
        _src(ds.sources, p)
    ds.folds, ds.folds_source = group_kfold(groups, k), f'GroupKFold({k}) on connected components of a shared Composition string or DOI token (rule of domains/electrolytes/work/baseline.py)'
    ds.notes.append('formula-only element fractions; the official 121-row holdout is not loaded (labels are public, so it is not blind)')
    return ds


def _fill_missing(X, names):
    """Label-free handling of missing numbers: fill with the column median over all rows and add a 0/1 "was missing" column for every
    column that had any. (The trainers standardise on training rows only; this fill uses no target values.)"""
    X = np.array(X, float)
    miss = np.isnan(X)
    cols = np.where(miss.any(axis=0))[0]
    med = np.nanmedian(X, axis=0)
    X = np.where(miss, np.where(np.isnan(med), 0.0, med), X)
    if len(cols):
        X = np.hstack([X, miss[:, cols].astype(float)])
        names = list(names) + [f'{names[c]}__missing' for c in cols]
    return X, list(names)


def _load_cleaning(task, target, family):
    base = REPO / 'domains/cleaning/work/cleaned'
    pcsv, pnpz = base / 'surfpro_clean.csv', base / 'surfpro_features.npz'
    rows = list(csv.DictReader(io.StringIO(read_bytes(pcsv).decode('utf-8'))))
    guard_path(pnpz)
    z = np.load(io.BytesIO(read_bytes(pnpz)), allow_pickle=False)
    assert [int(r['Index']) for r in rows] == [int(i) for i in z['index']], 'surfpro_features.npz is not row-aligned with surfpro_clean.csv'
    keep = [i for i, r in enumerate(rows) if r[task] not in ('', None)]
    X, names = _fill_missing(z['X'][keep].astype(float), [str(f) for f in z['features']])
    varying = [j for j in range(X.shape[1]) if np.ptp(X[:, j]) > 0]   # drop columns that are constant within this task's rows
    X, names = X[:, varying], [names[j] for j in varying]
    y = np.array([float(rows[i][task]) for i in keep])
    groups = np.array([int(float(rows[i][f'group_meas_{task}'])) for i in keep])
    ds = Dataset(family, names, X, y, groups, target)
    for p in (pcsv, pnpz):
        _src(ds.sources, p)
    ds.folds, ds.folds_source = group_kfold(groups, 5), f'GroupKFold(5) on measurement-reference groups (group_meas_{task}; rule of domains/cleaning/work/prepare_surfpro.py)'
    ds.notes.append('SURFPRO single pure surfactants: a physical property, not cleaning performance. RDKit 2D descriptors + 512-bit Morgan counts + '
                    'type one-hot + CMC temperature; missing values filled with the column median plus a "was missing" flag')
    return ds


def load_cleaning_pcmc():
    return _load_cleaning('pCMC', 'pCMC = -log10(critical micelle concentration, mol/L); SURFPRO', 'cleaning_pcmc')


def load_cleaning_tension():
    return _load_cleaning('AW_ST_CMC', 'air-water surface tension at the CMC, mN/m; SURFPRO', 'cleaning_tension')


def load_corrosion(min_count=10):
    """NIST CORR-DATA: log10 corrosion rate (mm/yr) for plain point-value rates, the rows and inputs of
    domains/corrosives/work/baseline_corrdata.py. Categories seen fewer than min_count times are pooled as "infrequent"
    (counted over all rows; no target values are used)."""
    pcsv = REPO / 'domains/corrosives/work/cleaned/corrdata_normalized.csv'
    rows = list(csv.DictReader(io.StringIO(read_bytes(pcsv).decode('utf-8'))))

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float('nan')

    use = []
    for r in rows:
        if r['rate_status'] != 'numeric' or r['rate_qualifier'] != 'exact' or r['rate_columns_consistent'] == 'large_disagreement':
            continue
        rate = num(r['rate_mm_y'])
        if not rate > 0 or not r['reference_number'].strip():
            continue
        use.append((r, rate))
    cats = ['material_class', 'material_family', 'material', 'environment', 'conc_qualifier', 'temp_qualifier']
    cols, names = [], []
    for c in cats:
        vals = [(r[c].strip() or 'none') for r, _ in use]
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        levels = sorted(v for v, n in counts.items() if n >= min_count)
        pos = {v: j for j, v in enumerate(levels)}
        M = np.zeros((len(use), len(levels) + 1))
        for i, v in enumerate(vals):
            M[i, pos.get(v, len(levels))] = 1.0
        cols.append(M)
        names += [f'{c}={v}' for v in levels] + [f'{c}=(infrequent)']
    conc = np.array([num(r['conc_value']) if r['conc_unit'] == 'vol% (column default)' else float('nan') for r, _ in use])
    tv, tl, th = (np.array([num(r[k]) for r, _ in use]) for k in ('temp_C_value', 'temp_C_low', 'temp_C_high'))
    temp = np.where(np.isnan(tv), (tl + th) / 2, tv)
    dl, dh = (np.array([num(r[k]) for r, _ in use]) for k in ('duration_h_low', 'duration_h_high'))
    hours = np.log10(np.clip((dl + dh) / 2, 0.1, None))
    Xn, nn = _fill_missing(np.column_stack([conc, temp, hours]), ['conc_vol_pct', 'temp_C_mid', 'log10_hours'])
    X = np.hstack(cols + [Xn])
    names += nn
    varying = [j for j in range(X.shape[1]) if np.ptp(X[:, j]) > 0]
    X, names = X[:, varying], [names[j] for j in varying]
    y = np.log10(np.array([rate for _, rate in use]))
    refs = [r['reference_number'].strip() for r, _ in use]
    relabel = {g: j for j, g in enumerate(dict.fromkeys(refs))}
    groups = np.array([relabel[g] for g in refs])
    ds = Dataset('corrosion', names, X, y, groups, 'log10 corrosion rate, log10(mm/yr); NIST CORR-DATA point values')
    _src(ds.sources, pcsv)
    ds.folds, ds.folds_source = group_kfold(groups, 5), 'GroupKFold(5) on Reference # (rule of domains/corrosives/work/baseline_corrdata.py)'
    ds.notes.append('historical observations, not safety advice; bounds, ranges, zero rates and letter ratings are excluded; exploratory')
    return ds


LOADERS = {'concrete': load_concrete, 'alloys': load_alloys, 'insulators': load_insulators, 'polymers': load_polymers,
           'chemistry': load_chemistry, 'electrolytes': load_electrolytes, 'cleaning_pcmc': load_cleaning_pcmc,
           'cleaning_tension': load_cleaning_tension, 'corrosion': load_corrosion}

def load(family):
    if family not in LOADERS:
        raise SystemExit(f'unknown family {family!r}; choose from {sorted(LOADERS)}')
    return LOADERS[family]()


if __name__ == '__main__':
    for fam in (sys.argv[1:] or LOADERS):
        print(json.dumps(load(fam).summary(), indent=1))
