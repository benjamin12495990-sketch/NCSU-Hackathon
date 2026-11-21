"""Synthetic crystals in the JARVIS-DFT record layout, for tests and for a CPU check when the real file is absent.

The target is NOT physics: `bulk_modulus_kv` here is a made-up smooth function of composition, interatomic
distances and density (a sum of pair terms per atom, scaled by 1 / volume per atom). It exists so the
pipeline can be checked end to end; a model that learns it shows the graph code works, nothing more.
"""
import numpy as np

POOL = ['Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'K', 'Ca', 'Ti', 'Fe', 'Ni', 'Cu', 'Zn', 'Zr', 'Mo', 'Sn', 'Ba', 'W']


def _stiff(el):
    return 0.6 + (sum(map(ord, el)) % 11) / 7.0


def make_records(n, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    while len(recs) < n:
        k = int(rng.integers(1, 4))
        els = list(rng.choice(POOL, size=k, replace=False))
        natoms = int(rng.integers(k, 9))
        species = els + list(rng.choice(els, size=natoms - k))
        abc = rng.uniform(3.0, 6.5, 3) * (natoms / 4) ** (1 / 3)
        ang = np.radians(rng.uniform(80, 100, 3))
        a = np.array([abc[0], 0, 0])
        b = abc[1] * np.array([np.cos(ang[2]), np.sin(ang[2]), 0])
        cx = np.cos(ang[1]); cy = (np.cos(ang[0]) - np.cos(ang[1]) * np.cos(ang[2])) / np.sin(ang[2])
        c = abc[2] * np.array([cx, cy, np.sqrt(max(1e-3, 1 - cx ** 2 - cy ** 2))])
        L = np.stack([a, b, c])
        frac = rng.random((natoms, 3))
        cart = frac @ L
        shifts = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing='ij'), -1).reshape(-1, 3) @ L
        d = np.linalg.norm(cart[None, None] + shifts[None, :, None] - cart[:, None, None], axis=-1)
        d[d < 1e-9] = np.inf
        if d.min() < 1.5:
            continue  # atoms too close; draw again
        s = np.array([_stiff(e) for e in species])
        pair = np.sqrt(s[:, None, None] * s[None, None, :]) * np.exp(-(d - 2.0)) * (d < 6.0)
        vpa = abs(np.linalg.det(L)) / natoms
        K = 40.0 * pair.sum() / natoms * (12.0 / vpa) * (1 + 0.03 * rng.standard_normal())
        recs.append({'jid': f'SYN-{len(recs)}', 'formula': ''.join(sorted(set(species))),
                     'atoms': {'lattice_mat': L.tolist(), 'coords': frac.tolist(), 'elements': species, 'cartesian': False},
                     'bulk_modulus_kv': float(K) if rng.random() > 0.05 else 'na'})
    return recs
