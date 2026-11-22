"""A CGCNN-style crystal graph network in plain PyTorch (index_add_ for message aggregation and pooling).

Xie and Grossman, Phys. Rev. Lett. 120, 145301 (2018): atoms carry learned element embeddings, edges a
Gaussian expansion of the interatomic distance; each convolution computes
  z_ij = W [h_i, h_j, e_ij] + b,   m_ij = sigmoid(z_f) * softplus(z_s),   h_i <- softplus(h_i + BN(sum_j m_ij)),
then a mean over atoms and a small MLP give one number per crystal.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ATOM_INPUTS = ('embedding', 'properties', 'both')


def element_feature_table(n_elements=119):
    """[n_elements, 32] per atomic number: 7 standardised element properties (Z, period, group, electronegativity, weight,
    covalent radius, valence electrons; gpu/structure/element_table.json) + one-hot group (18) + one-hot period (7). Row 0 is zero.
    Unlike a learned embedding, this gives an element never seen in training a sensible starting description."""
    t = json.loads((Path(__file__).with_name('element_table.json')).read_text())
    props = t['props']
    rows = {v['Z']: v for v in t['elements'].values()}
    X = torch.zeros(n_elements, len(props) + 18 + 7)
    cont = torch.tensor([[float(rows[z][p]) for p in props] for z in sorted(rows)])
    mean, std = cont.mean(0), cont.std(0).clamp(min=1e-6)
    for z in range(1, n_elements):
        r = rows.get(z)
        if r is None:  # beyond the table (none in JARVIS dft_3d): the average element
            X[z, :len(props)] = 0.0
            continue
        X[z, :len(props)] = (torch.tensor([float(r[p]) for p in props]) - mean) / std
        X[z, len(props) + int(r['group']) - 1] = 1.0
        X[z, len(props) + 18 + int(r['period']) - 1] = 1.0
    return X


class GaussianExpansion(nn.Module):
    def __init__(self, cutoff=8.0, n=40, width=None):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0, cutoff, n))
        self.gamma = 1.0 / (width or (cutoff / (n - 1))) ** 2

    def forward(self, d):
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


class CGConv(nn.Module):
    def __init__(self, f_atom, f_edge):
        super().__init__()
        self.lin = nn.Linear(2 * f_atom + f_edge, 2 * f_atom)
        self.bn1 = nn.BatchNorm1d(2 * f_atom)
        self.bn2 = nn.BatchNorm1d(f_atom)

    def forward(self, h, src, dst, e):
        z = self.bn1(self.lin(torch.cat([h[src], h[dst], e], dim=-1)))
        gate, core = z.chunk(2, dim=-1)
        msg = torch.sigmoid(gate) * F.softplus(core)
        agg = torch.zeros_like(h).index_add_(0, src, msg.to(h.dtype))  # under bf16 autocast msg is bf16
        return F.softplus(h + self.bn2(agg))


class CGCNN(nn.Module):
    def __init__(self, f_atom=64, n_conv=3, f_hidden=128, cutoff=8.0, n_gauss=40, n_elements=119, atom_input='embedding'):
        """atom_input: 'embedding' (a learned vector per element, the default; an element absent from training keeps its random start),
        'properties' (a linear map of element properties, so unseen elements are described by what they are), or 'both' (the sum,
        with the embedding starting at zero so an unseen element is described by its properties alone)."""
        super().__init__()
        if atom_input not in ATOM_INPUTS:
            raise ValueError(f'atom_input must be one of {ATOM_INPUTS}')
        self.atom_input = atom_input
        if atom_input in ('embedding', 'both'):
            self.embed = nn.Embedding(n_elements, f_atom)  # element embeddings, learned from scratch
            if atom_input == 'both':
                nn.init.zeros_(self.embed.weight)
        if atom_input in ('properties', 'both'):
            self.register_buffer('elem_feats', element_feature_table(n_elements))
            self.prop = nn.Linear(self.elem_feats.shape[1], f_atom)
        self.rbf = GaussianExpansion(cutoff, n_gauss)
        self.convs = nn.ModuleList([CGConv(f_atom, n_gauss) for _ in range(n_conv)])
        self.head = nn.Sequential(nn.Linear(f_atom, f_hidden), nn.Softplus(), nn.Linear(f_hidden, 1))

    def forward(self, Z, src, dst, dist, batch, n_graphs):
        h = self.embed(Z) if self.atom_input == 'embedding' else self.prop(self.elem_feats[Z]) + (self.embed(Z) if self.atom_input == 'both' else 0)
        e = self.rbf(dist)
        for conv in self.convs:
            h = conv(h, src, dst, e)
        h = h.float()  # pool in float32
        pooled = torch.zeros(n_graphs, h.shape[1], device=h.device).index_add_(0, batch, h)
        counts = torch.zeros(n_graphs, device=h.device).index_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
        return self.head(pooled / counts.clamp(min=1).unsqueeze(1)).squeeze(-1)
