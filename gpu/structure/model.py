"""A CGCNN-style crystal graph network in plain PyTorch (index_add_ for message aggregation and pooling).

Xie and Grossman, Phys. Rev. Lett. 120, 145301 (2018): atoms carry learned element embeddings, edges a
Gaussian expansion of the interatomic distance; each convolution computes
  z_ij = W [h_i, h_j, e_ij] + b,   m_ij = sigmoid(z_f) * softplus(z_s),   h_i <- softplus(h_i + BN(sum_j m_ij)),
then a mean over atoms and a small MLP give one number per crystal.
"""
import torch
from torch import nn
from torch.nn import functional as F


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
        agg = torch.zeros_like(h).index_add_(0, src, msg)
        return F.softplus(h + self.bn2(agg))


class CGCNN(nn.Module):
    def __init__(self, f_atom=64, n_conv=3, f_hidden=128, cutoff=8.0, n_gauss=40, n_elements=119):
        super().__init__()
        self.embed = nn.Embedding(n_elements, f_atom)  # element embeddings, learned from scratch
        self.rbf = GaussianExpansion(cutoff, n_gauss)
        self.convs = nn.ModuleList([CGConv(f_atom, n_gauss) for _ in range(n_conv)])
        self.head = nn.Sequential(nn.Linear(f_atom, f_hidden), nn.Softplus(), nn.Linear(f_hidden, 1))

    def forward(self, Z, src, dst, dist, batch, n_graphs):
        h = self.embed(Z)
        e = self.rbf(dist)
        for conv in self.convs:
            h = conv(h, src, dst, e)
        pooled = torch.zeros(n_graphs, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, batch, h)
        counts = torch.zeros(n_graphs, device=h.device, dtype=h.dtype).index_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
        return self.head(pooled / counts.clamp(min=1).unsqueeze(1)).squeeze(-1)
