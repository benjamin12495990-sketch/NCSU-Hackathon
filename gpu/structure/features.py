"""Hand-built features for tree models: a stronger baseline than "element fractions + cell volume" for a crystal-graph network.

  composition statistics   mean, range and standard deviation (weighted by atomic fraction) of seven element properties
                           (Z, period, group, Pauling electronegativity, atomic weight, covalent radius, valence electrons;
                           gpu/structure/element_table.json)
  structure statistics     from the same neighbour lists the network sees (8 A cutoff, 12 nearest neighbours):
                           nearest-neighbour distance and its ratio to the sum of covalent radii (mean, min, max, std over atoms),
                           mean and std of all neighbour distances and radius ratios, coordination number within 1.2x the
                           nearest-neighbour distance (mean, std, min, max), a 16-bin radial histogram (0-8 A, per atom),
                           mean and max electronegativity difference over neighbour pairs, sorted lattice lengths and
                           angles, density and packing fraction

This is "composition + simple geometry", the kind of descriptor set used before graph networks. It is NOT a complete
structure fingerprint (no angles between bonds, no symmetry), so a network that still wins has something these trees lack.
Features are cached per record list in gpu/data_cache/jarvis/features/ (the neighbour search is the slow part).
"""
import hashlib
import json
import math
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from gpu.common import CACHE
from gpu.structure.graph import SYMBOLS, atoms_arrays, neighbor_graph

VERSION = 'features-v1'
_TABLE = json.loads(Path(__file__).with_name('element_table.json').read_text())
PROPS = ['Z', 'period', 'group', 'electronegativity', 'mass', 'covalent_radius', 'valence_electrons']
_P = np.zeros((len(SYMBOLS), len(PROPS)))
for _i, _s in enumerate(SYMBOLS):
    _row = _TABLE['elements'].get(_s)
    _P[_i] = [_row[p] for p in PROPS] if _row else np.nan
_P[np.isnan(_P[:, 0])] = np.nanmean(_P, axis=0)  # elements past Lr (none in JARVIS dft_3d): the mean row, so nothing is NaN
EN, RAD, MASS = PROPS.index('electronegativity'), PROPS.index('covalent_radius'), PROPS.index('mass')
RDF_BINS = np.arange(0.0, 8.5, 0.5)

NAMES = ([f'{s}_{p}' for p in PROPS for s in ('mean', 'range', 'std')]
         + [f'nn_dist_{s}' for s in ('mean', 'min', 'max', 'std')] + [f'nn_ratio_{s}' for s in ('mean', 'min', 'max', 'std')]
         + ['edge_dist_mean', 'edge_dist_std', 'edge_ratio_mean', 'edge_ratio_std']
         + [f'coord_{s}' for s in ('mean', 'std', 'min', 'max')] + [f'rdf_{lo:.1f}' for lo in RDF_BINS[:-1]]
         + ['en_diff_mean', 'en_diff_max'] + [f'lat_len_{i}' for i in range(3)] + [f'lat_ang_{i}' for i in range(3)] + ['density', 'packing'])


def one(atoms, cutoff=8.0, k=12, ties='first'):
    L, frac, Z = atoms_arrays(atoms)
    n = len(Z)
    p = _P[Z - 1]
    counts = np.bincount(Z - 1, minlength=len(SYMBOLS)).astype(float)
    w = counts[counts > 0] / n
    Pe = _P[np.where(counts > 0)[0]]
    mean = w @ Pe
    comp = np.empty(3 * len(PROPS))
    comp[0::3], comp[1::3], comp[2::3] = mean, Pe.max(0) - Pe.min(0), np.sqrt(w @ (Pe - mean) ** 2)
    src, dst, dist = neighbor_graph(L, frac, cutoff, k, ties)
    dist = dist.astype(float)
    rsum = (p[src, RAD] + p[dst, RAD]) / 100.0
    ratio = dist / rsum
    nn = np.full(n, cutoff)
    nnr = np.full(n, ratio.max() if len(ratio) else 1.0)
    if len(src):
        nn = np.full(n, np.inf)
        np.minimum.at(nn, src, dist)
        nn = np.where(np.isfinite(nn), nn, cutoff)
        nnr = np.full(n, np.inf)
        np.minimum.at(nnr, src, ratio)
        nnr = np.where(np.isfinite(nnr), nnr, ratio.max())
    near = (dist <= 1.2 * nn[src]).astype(float) if len(src) else np.zeros(0)
    cn = np.bincount(src, weights=near, minlength=n) if len(src) else np.zeros(n)
    rdf = np.histogram(dist, bins=RDF_BINS)[0] / n if len(dist) else np.zeros(len(RDF_BINS) - 1)
    de = np.abs(p[src, EN] - p[dst, EN]) if len(src) else np.zeros(1)
    lengths = np.linalg.norm(L, axis=1)
    cosang = [L[1] @ L[2] / (lengths[1] * lengths[2]), L[0] @ L[2] / (lengths[0] * lengths[2]), L[0] @ L[1] / (lengths[0] * lengths[1])]
    ang = np.sort(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    vol = abs(np.linalg.det(L))
    dens = p[:, MASS].sum() * 1.66054 / vol
    pack = (4.0 / 3.0 * math.pi * (p[:, RAD] / 100.0) ** 3).sum() / vol
    sd = lambda a: float(np.std(a)) if len(a) else 0.0  # noqa: E731
    st = [nn.mean(), nn.min(), nn.max(), sd(nn), nnr.mean(), nnr.min(), nnr.max(), sd(nnr),
          dist.mean() if len(dist) else cutoff, sd(dist), ratio.mean() if len(ratio) else 1.0, sd(ratio),
          cn.mean(), sd(cn), cn.min(), cn.max()]
    return np.concatenate([comp, st, rdf, [de.mean(), de.max()], np.sort(lengths), ang, [dens, pack]]).astype(np.float32)


def _chunk(args, ties='first'):
    return np.stack([one(a, ties=ties) for a in args])


def build(recs, threads=1, cache_dir=None, log=print, ties='first'):
    """Feature matrix [len(recs), len(NAMES)] for records with 'jid' and 'atoms'; cached on disk.
    ties: the neighbour-cut rule of gpu/structure/graph.py ('all' = whole shells, invariant; 'first' = the original rule).
    The cache key includes the rule, so features built under one rule are never reused for the other."""
    tag = VERSION if ties == 'first' else f'{VERSION}+ties-{ties}'   # the original key is unchanged, so older caches stay valid
    key = hashlib.sha256(json.dumps([tag, [r['jid'] for r in recs]]).encode()).hexdigest()[:24]
    path = Path(cache_dir or (CACHE / 'jarvis' / 'features')) / f'{key}.npy'
    if path.exists():
        return np.load(path)
    t0 = time.perf_counter()
    atoms = [r['atoms'] for r in recs]
    X = None
    if threads > 1 and len(atoms) > 3000:
        try:
            size = 500
            chunks = [atoms[i:i + size] for i in range(0, len(atoms), size)]
            with ProcessPoolExecutor(max_workers=threads) as ex:
                parts = []
                for j, part in enumerate(ex.map(partial(_chunk, ties=ties), chunks)):
                    parts.append(part)
                    if (j + 1) % 20 == 0:
                        log(f'  structure features {min(len(atoms), (j + 1) * size)}/{len(atoms)} ({time.perf_counter() - t0:.0f} s)')
            X = np.concatenate(parts)
        except Exception as e:  # noqa: BLE001 - fall back to one process
            log(f'  parallel feature build failed ({e!r}); using one process')
    if X is None:
        rows = []
        for i, a in enumerate(atoms):
            rows.append(one(a, ties=ties))
            if (i + 1) % 10000 == 0:
                log(f'  structure features {i + 1}/{len(atoms)} ({time.perf_counter() - t0:.0f} s)')
        X = np.stack(rows)
    assert X.shape[1] == len(NAMES), (X.shape, len(NAMES))
    assert np.isfinite(X).all(), 'non-finite structure feature'
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.npy')   # this process's own name: two jobs may build the same set at once
    np.save(tmp, X)
    try:
        tmp.replace(path)
    except OSError as e:   # Windows: the other job's identical file is open; keep using the features built here
        if not path.exists():
            raise
        log(f'  feature cache was written by another job at the same time ({e.__class__.__name__}); the unused copy is left at {tmp.name}')
    log(f'  structure features: {X.shape[0]} crystals x {X.shape[1]} in {time.perf_counter() - t0:.0f} s')
    return X
