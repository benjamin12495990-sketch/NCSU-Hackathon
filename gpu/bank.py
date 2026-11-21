"""A bank of K independent MLPs trained together with batched matmuls (torch.baddbmm).

Mirrors the Mlp class in src/lib/nnFamily.js:
  - He-normal weights (sd sqrt(2 / fan_in)), zero biases, ReLU hidden layers, one linear output;
  - Adam (b1 0.9, b2 0.999, eps 1e-8, bias-corrected) with L2 decay added to the weight gradient
    (not to biases), gradient = sum over the batch / batch size;
  - squared loss (gradient = error) or Huber with delta 1 on the standardised target (gradient = clip(error, -1, 1)).
Every member k has its own weights, optimiser state, learning rate, decay, loss and data order; the members
never interact, so one bank is K separate fits that share kernel launches. Exact bit parity with the
JavaScript trainer is not expected (different random streams, float32 by default, different summation order).
"""
import math

import torch

SQUARED_DELTA = 1e30  # a Huber delta this large is the squared loss


class MlpBank:
    def __init__(self, sizes, seeds, device, dtype=torch.float32, generators=None):
        self.sizes, self.K, self.device, self.dtype = list(sizes), len(seeds), device, dtype
        # One CPU generator per member: initial weights (and later its data order) come from it, so the
        # starting weights are the same on CPU and CUDA.
        self.gens = generators or [torch.Generator().manual_seed(int(s)) for s in seeds]
        self.W, self.b = [], []
        for l in range(len(sizes) - 1):
            n_in, n_out = sizes[l], sizes[l + 1]
            W = torch.stack([torch.randn(n_in, n_out, generator=g, dtype=torch.float64) * math.sqrt(2 / n_in) for g in self.gens])
            self.W.append(W.to(device=device, dtype=dtype).requires_grad_(True))
            self.b.append(torch.zeros(self.K, 1, n_out, device=device, dtype=dtype, requires_grad=True))
        self.params = self.W + self.b
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def n_weights(self):
        return sum(p[0].numel() for p in self.params)

    def forward(self, x):
        """x: [K, B, d] (or [B, d], shared by all members) -> [K, B]."""
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.K, -1, -1)
        h = x
        L = len(self.W)
        for l in range(L):
            h = torch.baddbmm(self.b[l], h, self.W[l])
            if l < L - 1:
                h = torch.relu(h)
        return h.squeeze(-1)

    def loss(self, pred, target, delta):
        """Sum over members of the batch-mean Huber loss; delta [K] (SQUARED_DELTA for squared)."""
        e = pred - target
        a = e.abs()
        d = delta.view(-1, 1)
        q = torch.minimum(a, d)
        per = 0.5 * q * q + d * (a - q)
        return per.mean(dim=1).sum(), (e.detach() ** 2).sum(dim=1)

    @torch.no_grad()
    def step(self, lr, decay):
        """One Adam update. lr, decay: tensors [K] on the device."""
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        lr3 = lr.view(-1, 1, 1)
        n_w = len(self.W)
        for i, (p, m, v) in enumerate(zip(self.params, self.m, self.v)):
            g = p.grad
            if i < n_w:
                g = g + decay.view(-1, 1, 1) * p
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            p.sub_(lr3 * (m / c1) / ((v / c2).sqrt() + eps))
            p.grad = None

    def state(self):
        return {'sizes': self.sizes, 'K': self.K, 't': self.t,
                'W': [w.detach().cpu() for w in self.W], 'b': [b.detach().cpu() for b in self.b],
                'm': [x.cpu() for x in self.m], 'v': [x.cpu() for x in self.v],
                'gens': [g.get_state() for g in self.gens]}

    def load_state(self, s):
        assert s['sizes'] == self.sizes and s['K'] == self.K, 'checkpoint bank shape does not match'
        with torch.no_grad():
            for dst, src in zip(self.W + self.b, s['W'] + s['b']):
                dst.copy_(src.to(dst))
            for dst, src in zip(self.m + self.v, s['m'] + s['v']):
                dst.copy_(src.to(dst))
        self.t = s['t']
        for g, st in zip(self.gens, s['gens']):
            g.set_state(st)


def epoch_orders(gens, n, device):
    """One shuffled row order per member, drawn from that member's CPU generator: [K, n] on device."""
    return torch.stack([torch.randperm(n, generator=g) for g in gens]).to(device, non_blocking=True)
