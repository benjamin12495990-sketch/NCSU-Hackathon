"""Throughput profile for banks of K independent MLPs trained together (batched matmul).

Synthetic data with the chemistry family's shape (8,000 rows x 17 inputs, layers 17-96-48-1, batch 32,
Huber loss, Adam with L2 decay, lr decaying linearly to 10%). It measures throughput and memory only:
no predictions are reported and no model is saved.

  python -m gpu.profile_batched_mlp --device cuda --counts 1,5,32,128 --epochs 10 --output gpu/runs/profile/cuda.json
Refuses to overwrite --output.
"""
import argparse
import time

from gpu.common import git_info, hardware, now_iso, peak_memory_mb, pick_device, refuse_existing, sync, write_json_new


def run_bank(K, X, y, hidden, epochs, batch, device, dtype, seed):
    import torch
    from gpu.bank import MlpBank, epoch_orders
    n, d = X.shape
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    bank = MlpBank([d, *hidden, 1], [seed + 101 * i for i in range(K)], device, dtype)
    lr0 = torch.full((K,), 0.004, device=device, dtype=dtype)
    decay = torch.full((K,), 1e-4, device=device, dtype=dtype)
    delta = torch.full((K,), 1.0, device=device, dtype=dtype)
    sync(device)
    t0 = time.perf_counter()
    last = None
    for ep in range(epochs):
        order = epoch_orders(bank.gens, n, device)
        lr = lr0 * (1 - 0.9 * ep / epochs)
        for s in range(0, n, batch):
            idx = order[:, s:s + batch]
            loss, _ = bank.loss(bank.forward(X[idx]), y[idx], delta)
            loss.backward()
            bank.step(lr, decay)
        last = loss.detach()
    final = float(last.item()) / K
    sync(device)
    wall = time.perf_counter() - t0
    return {'K': K, 'epochs': epochs, 'rows': n, 'wall_seconds': wall, 'seconds_per_epoch': wall / epochs,
            'model_rows_per_second': K * n * epochs / wall, 'weights_per_model': bank.n_weights(),
            'peak_memory_mb': peak_memory_mb(device), 'final_batch_loss_mean': final}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--counts', default='1,5,32,128', help='comma-separated bank sizes K')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--rows', type=int, default=8000)
    ap.add_argument('--inputs', type=int, default=17)
    ap.add_argument('--hidden', default='96,48')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--dtype', default='float32', choices=['float32', 'float64'])
    ap.add_argument('--threads', type=int, default=0, help='torch.set_num_threads (0 = leave default)')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--output', required=True, help='new JSON path (refused if it exists)')
    a = ap.parse_args(argv)
    refuse_existing(a.output)
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    device = pick_device(a.device)
    dtype = getattr(torch, a.dtype)
    g = torch.Generator().manual_seed(a.seed)
    X = torch.randn(a.rows, a.inputs, generator=g, dtype=torch.float64)
    w = torch.randn(a.inputs, generator=g, dtype=torch.float64)
    y = torch.tanh(X @ w / a.inputs ** 0.5) + 0.1 * torch.randn(a.rows, generator=g, dtype=torch.float64)
    X, y = X.to(device, dtype), ((y - y.mean()) / y.std()).to(device, dtype)
    hidden = [int(h) for h in a.hidden.split(',') if h]
    counts = [int(c) for c in a.counts.split(',')]
    run_bank(1, X, y, hidden, 1, a.batch, device, dtype, a.seed)  # warm-up
    rows = []
    for K in counts:
        r = run_bank(K, X, y, hidden, a.epochs, a.batch, device, dtype, a.seed)
        rows.append(r)
        print(f"K={K:4d}  wall {r['wall_seconds']:8.2f} s  {r['model_rows_per_second']:12.0f} model-rows/s  "
              f"peak {r['peak_memory_mb'] or float('nan'):8.1f} MB", flush=True)
    base = rows[0]['model_rows_per_second'] if rows else None
    for r in rows:
        r['speedup_vs_first_count'] = r['model_rows_per_second'] / base if base else None
    rep = {'created': now_iso(), 'git': git_info(), 'device': str(device), 'dtype': a.dtype,
           'hardware': hardware(device), 'threads': torch.get_num_threads(),
           'workload': {'rows': a.rows, 'inputs': a.inputs, 'layers': [a.inputs, *hidden, 1], 'batch': a.batch,
                        'loss': 'huber', 'data': 'synthetic (chemistry-family shape), not a model result'},
           'peak_memory_note': 'CUDA: torch.cuda.max_memory_allocated per bank; CPU: process peak RSS so far (Linux/macOS only)',
           'results': rows}
    write_json_new(a.output, rep)
    print('wrote', a.output)
    return rep


if __name__ == '__main__':
    main()
