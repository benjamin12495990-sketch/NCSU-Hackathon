"""Periodic neighbour graphs from a lattice and atoms (numpy; no torch_geometric).

For every atom i, the neighbours are all periodic images of every atom j (j may equal i in another cell)
within `cutoff` angstrom, sorted by distance and cut to `max_neighbors` (CGCNN uses 8 A and 12).
"""
import math

import numpy as np

SYMBOLS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr '
           'Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb '
           'Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf '
           'Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()
Z_OF = {s: i + 1 for i, s in enumerate(SYMBOLS)}


def atoms_arrays(atoms):
    """JARVIS 'atoms' dict -> (lattice [3,3] rows a,b,c; fractional coords [N,3]; atomic numbers [N])."""
    L = np.asarray(atoms['lattice_mat'], float)
    X = np.asarray(atoms['coords'], float).reshape(-1, 3)
    frac = X @ np.linalg.inv(L) if atoms.get('cartesian', False) else X
    Z = np.array([Z_OF[e] for e in atoms['elements']], np.int64)
    return L, frac, Z


def neighbor_graph(lattice, frac, cutoff=8.0, max_neighbors=12):
    """Returns src, dst (int64 [E]) and dist (float32 [E]): edge i -> j with |r_j + T - r_i| < cutoff."""
    L = np.asarray(lattice, float)
    frac = np.asarray(frac, float) % 1.0
    cart = frac @ L
    vol = abs(np.linalg.det(L))
    if vol < 1e-8:
        raise ValueError('degenerate lattice')
    a, b, c = L
    heights = [vol / np.linalg.norm(np.cross(b, c)), vol / np.linalg.norm(np.cross(c, a)), vol / np.linalg.norm(np.cross(a, b))]
    n = [int(math.ceil(cutoff / h)) for h in heights]
    rng = [np.arange(-k, k + 1) for k in n]
    shifts = np.stack(np.meshgrid(*rng, indexing='ij'), -1).reshape(-1, 3)
    T = shifts @ L                                                   # [S, 3]
    N = len(cart)
    # d[i, s, j] = | cart[j] + T[s] - cart[i] |
    diff = cart[None, None, :, :] + T[None, :, None, :] - cart[:, None, None, :]
    d = np.sqrt((diff ** 2).sum(-1))
    zero = np.where((shifts == 0).all(1))[0][0]
    d[np.arange(N), zero, np.arange(N)] = np.inf                     # no self-edge in the home cell
    d = d.reshape(N, -1)
    src, dst, dist = [], [], []
    for i in range(N):
        cand = np.where(d[i] < cutoff)[0]
        if len(cand) > max_neighbors:
            cand = cand[np.argsort(d[i, cand], kind='stable')[:max_neighbors]]
        else:
            cand = cand[np.argsort(d[i, cand], kind='stable')]
        src.append(np.full(len(cand), i))
        dst.append(cand % N)
        dist.append(d[i, cand])
    return (np.concatenate(src).astype(np.int64), np.concatenate(dst).astype(np.int64),
            np.concatenate(dist).astype(np.float32))


def build_graph(atoms, cutoff=8.0, max_neighbors=12):
    L, frac, Z = atoms_arrays(atoms)
    src, dst, dist = neighbor_graph(L, frac, cutoff, max_neighbors)
    return {'Z': Z, 'src': src, 'dst': dst, 'dist': dist}


def collate(graphs):
    """Concatenates graphs into one disconnected graph; `batch` maps each atom to its crystal."""
    Z, src, dst, dist, batch = [], [], [], [], []
    off = 0
    for k, g in enumerate(graphs):
        Z.append(g['Z']); src.append(g['src'] + off); dst.append(g['dst'] + off); dist.append(g['dist'])
        batch.append(np.full(len(g['Z']), k))
        off += len(g['Z'])
    return {'Z': np.concatenate(Z), 'src': np.concatenate(src), 'dst': np.concatenate(dst),
            'dist': np.concatenate(dist), 'batch': np.concatenate(batch).astype(np.int64), 'n_graphs': len(graphs)}
