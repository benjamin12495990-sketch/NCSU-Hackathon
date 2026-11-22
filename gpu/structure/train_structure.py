"""Structure-aware GPU workload: a CGCNN on a JARVIS-DFT 3D property (computed, OptB88vdW DFT; not measured).

Target (--target, default bulk_modulus_kv): the Voigt bulk or shear modulus as log10(GPa), the formation energy per atom
or the band gap as they are. Records whose value is missing ('na') or, for log targets, not positive are dropped; the
counts are reported. Split: grouped by reduced formula, each
formula assigned by SHA-256 to train (80%), val (10%) or test (10%), so no formula is on both sides and the
split does not depend on file order or --limit; the split list and its SHA-256 are saved with the run.
Baselines on the same split: the training mean, and ridge regression on element fractions.

  python -m gpu.structure.train_structure --device cuda --epochs 100
  python -m gpu.structure.train_structure --device cpu --limit 2000 --epochs 8      # a small CPU check
  python -m gpu.structure.train_structure ... --resume gpu/runs/structure/<run>     # same command + --resume
  python -m gpu.structure.train_structure --synthetic 600 ...                       # no download; pipeline test only

Each run writes a new folder gpu/runs/structure/<UTC time>_<tag>/ (never overwritten). The validation set
picks the best epoch; the test set is scored once at the end with that epoch's weights.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
import zipfile
from functools import reduce
from pathlib import Path

import numpy as np

from gpu.common import CACHE, RUNS, git_info, hardware, new_run_dir, now_iso, peak_memory_mb, pick_device, sha256_bytes, sha256_file, sync, write_json_new
from gpu.structure.graph import SYMBOLS, Z_OF, build_graph, collate

DEFAULT_ZIP = CACHE / 'jarvis' / 'jdft_3d-9-24-2025.json.zip'
TARGET = 'bulk_modulus_kv'  # the default target
# JARVIS-DFT 3D properties (all DFT-calculated). 'log10' targets must be positive and are modelled as log10(value);
# 'identity' targets are modelled as they are (the error is then in the property's own unit).
TARGETS = {
    'bulk_modulus_kv': {'transform': 'log10', 'label': 'log10(bulk modulus Kv / GPa), Voigt, OptB88vdW', 'unit': 'log10 GPa'},
    'shear_modulus_gv': {'transform': 'log10', 'label': 'log10(shear modulus Gv / GPa), Voigt, OptB88vdW', 'unit': 'log10 GPa'},
    'formation_energy_peratom': {'transform': 'identity', 'label': 'formation energy per atom, eV/atom, OptB88vdW', 'unit': 'eV/atom'},
    'optb88vdw_bandgap': {'transform': 'identity', 'label': 'band gap, eV, OptB88vdW (a known underestimate of real gaps)', 'unit': 'eV'},
}


def reduced_formula(elements):
    counts = {}
    for e in elements:
        counts[e] = counts.get(e, 0) + 1
    g = reduce(math.gcd, counts.values())
    return ''.join(f'{e}{counts[e] // g if counts[e] // g > 1 else ""}' for e in sorted(counts))


def load_records(a):
    if a.synthetic:
        from gpu.structure.synthetic import make_records
        recs, source = make_records(a.synthetic, seed=a.seed), {'synthetic': a.synthetic, 'seed': a.seed,
                                                                 'note': 'synthetic target, pipeline check only'}
    else:
        zp = Path(a.data)
        if not zp.exists():
            sys.exit(f'{zp} not found: run python -m gpu.structure.fetch_jarvis --download first (or use --synthetic N)')
        with zipfile.ZipFile(zp) as z:
            name = [n for n in z.namelist() if n.endswith('.json')][0]
            recs = json.loads(z.read(name))
        source = {'zip': str(zp), 'zip_sha256': sha256_file(zp), 'json_name': name}
    target = getattr(a, 'target', TARGET) or TARGET
    spec = TARGETS[target]
    counts = {'records': len(recs), 'missing_or_non_numeric': 0, 'non_positive': 0, 'target': target}
    kept = []
    for r in recs:
        v = r.get(target)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            counts['missing_or_non_numeric'] += 1
            continue
        if spec['transform'] == 'log10' and v <= 0:
            counts['non_positive'] += 1
            continue
        kept.append({'jid': r['jid'], 'formula': reduced_formula(r['atoms']['elements']), 'atoms': r['atoms'],
                     'y': math.log10(v) if spec['transform'] == 'log10' else float(v)})
    counts['kept'] = len(kept)
    kept.sort(key=lambda r: hashlib.sha256(r['jid'].encode()).hexdigest())  # file-order independent
    if a.limit:
        kept = kept[:a.limit]
    counts['used'] = len(kept)
    return kept, source, counts


SPLIT_MODES = ('formula', 'system')
SPLIT_RULES = {'formula': "sha256('split-v1:' + reduced formula) % 100: <80 train, <90 val, else test",
               'system': "sha256('split-system-v1:' + chemical system, i.e. the sorted set of elements joined by '-') % 100: <80 train, <90 val, else test"}


def chemical_system(elements):
    return '-'.join(sorted(set(elements)))


def group_of(rec, mode='formula'):
    """The unit that is never split across train / val / test: the reduced formula, or the whole chemical system (Fe-O, Li-Fe-P-O, ...)."""
    return rec['formula'] if mode == 'formula' else chemical_system(rec['atoms']['elements'])


def split_of(key, mode='formula'):
    salt = 'split-v1:' if mode == 'formula' else 'split-system-v1:'
    h = int(hashlib.sha256(f'{salt}{key}'.encode()).hexdigest(), 16) % 100
    return 'train' if h < 80 else 'val' if h < 90 else 'test'


def parts_of(recs, mode='formula'):
    return np.array([split_of(group_of(r, mode), mode) for r in recs])


def element_fractions(atoms):
    v = np.zeros(len(SYMBOLS))
    for e in atoms['elements']:
        v[Z_OF[e] - 1] += 1
    return v / v.sum()


def metrics(p, y):
    e = np.asarray(p) - np.asarray(y)
    sst = ((y - np.mean(y)) ** 2).sum()
    return {'n': int(len(y)), 'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt((e ** 2).mean())), 'r2': float(1 - (e ** 2).sum() / sst)}


def graphs_for(recs, a, key):
    """Builds (or loads from gpu/data_cache/jarvis/graphs/) the neighbour graphs."""
    import torch
    cache = CACHE / 'jarvis' / 'graphs' / f'{key}.pt'
    if cache.exists():
        return torch.load(cache, weights_only=False), 0.0, True
    t0 = time.perf_counter()
    gs = []
    for i, r in enumerate(recs):
        gs.append(build_graph(r['atoms'], a.cutoff, a.max_neighbors))
        if (i + 1) % 5000 == 0:
            print(f'  graphs {i + 1}/{len(recs)} ({time.perf_counter() - t0:.0f} s)', flush=True)
    sec = time.perf_counter() - t0
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix('.tmp')
    torch.save(gs, tmp)
    tmp.replace(cache)
    return gs, sec, False


def to_device(b, device):
    import torch
    return {k: (torch.as_tensor(v, device=device) if k != 'n_graphs' else v) for k, v in b.items()}


def save_predictions(path, recs, idx, y, preds, mode='formula'):
    """predictions.npz: for val and test, the record ids, reduced formulas, split groups (formula or chemical system), true and predicted values."""
    arrays = {}
    for p in ('val', 'test'):
        arrays[f'{p}_jid'] = np.array([recs[i]['jid'] for i in idx[p]])
        arrays[f'{p}_formula'] = np.array([recs[i]['formula'] for i in idx[p]])
        arrays[f'{p}_group'] = np.array([group_of(recs[i], mode) for i in idx[p]])
        arrays[f'{p}_y'] = np.asarray(y[idx[p]])
        arrays[f'{p}_pred'] = np.asarray(preds[p])
    np.savez_compressed(path, **arrays)


def run_model(model, graphs, idx, device, batch, amp):
    import torch
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        for s in range(0, len(idx), batch):
            b = to_device(collate([graphs[i] for i in idx[s:s + batch]]), device)
            b['dist'] = b['dist'].float()
            out.append(model(b['Z'], b['src'], b['dst'], b['dist'], b['batch'], b['n_graphs']).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(DEFAULT_ZIP))
    ap.add_argument('--synthetic', type=int, default=0, help='use N synthetic crystals instead of JARVIS (tests)')
    ap.add_argument('--limit', type=int, default=0, help='use the first N kept records (in SHA-256-of-jid order)')
    ap.add_argument('--target', default=TARGET, choices=sorted(TARGETS), help='JARVIS-DFT property to predict (default bulk_modulus_kv)')
    ap.add_argument('--split', default='formula', choices=SPLIT_MODES, help='formula: no reduced formula on both sides (default); system: no chemical system on both sides (a harder test)')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--atom-features', type=int, default=64)
    ap.add_argument('--conv', type=int, default=3)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--cutoff', type=float, default=8.0)
    ap.add_argument('--max-neighbors', type=int, default=12)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--amp', action='store_true', help='bfloat16 autocast (CUDA); off by default')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--stop-after-epochs', type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument('--runs-root', default=str(RUNS))
    ap.add_argument('--tag', default='')
    ap.add_argument('--resume')
    a = ap.parse_args(argv)
    import torch
    from gpu.structure.model import CGCNN
    if a.threads:
        torch.set_num_threads(a.threads)
    device = pick_device(a.device)
    t_start = time.perf_counter()
    recs, source, counts = load_records(a)
    t_load = time.perf_counter() - t_start
    parts = parts_of(recs, a.split)
    split_list = [[r['jid'], p] for r, p in zip(recs, parts)]
    split_sha = sha256_bytes(json.dumps(split_list))
    idx = {p: np.where(parts == p)[0] for p in ('train', 'val', 'test')}
    y = np.array([r['y'] for r in recs])
    ident_src = {k: v for k, v in source.items() if k != 'zip'}  # the file's hash, not where it sits on disk
    data_sha = sha256_bytes(json.dumps(ident_src, sort_keys=True), split_sha, y.tobytes())
    cfg = {k: getattr(a, k) for k in ('target', 'synthetic', 'limit', 'epochs', 'batch', 'lr', 'weight_decay', 'atom_features', 'conv',
                                      'hidden', 'cutoff', 'max_neighbors', 'seed', 'amp')}
    if a.split != 'formula':
        cfg['split'] = a.split  # left out for the default so older runs keep their identity
    ident = sha256_bytes(json.dumps(cfg, sort_keys=True), data_sha)
    print(f'records {counts["records"]}, missing/non-numeric {counts["missing_or_non_numeric"]}, non-positive {counts["non_positive"]}, '
          f'kept {counts["kept"]}, used {counts["used"]} | train/val/test {len(idx["train"])}/{len(idx["val"])}/{len(idx["test"])} '
          f'| split sha256 {split_sha[:16]}', flush=True)
    if a.resume:
        run = Path(a.resume)
        old = json.loads((run / 'run_config.json').read_text())
        if old['ident'] != ident:
            sys.exit(f'refusing to resume {run}: config or data hash differs')
        if (run / 'result.json').exists():
            sys.exit(f'{run} is already complete')
    else:
        run = new_run_dir(Path(a.runs_root) / 'structure', a.tag)
        write_json_new(run / 'run_config.json', {'ident': ident, 'config': cfg, 'data_sha256': data_sha, 'source': source,
                                                 'counts': counts, 'split_sha256': split_sha, 'argv': sys.argv, 'git': git_info(), 'created': now_iso()})
        write_json_new(run / 'split.json', {'rule': SPLIT_RULES[a.split],
                                            'sha256_of_list': split_sha, 'jid_part': split_list})

    # baselines
    ytr = y[idx['train']]
    base = {'mean': {p: metrics(np.full(len(idx[p]), ytr.mean()), y[idx[p]]) for p in ('val', 'test')}}
    from sklearn.linear_model import Ridge
    F = np.stack([element_fractions(r['atoms']) for r in recs])
    best = None
    for alpha in (1e-3, 1e-2, 0.1, 1.0, 10.0):
        m = Ridge(alpha=alpha).fit(F[idx['train']], ytr)
        v = metrics(m.predict(F[idx['val']]), y[idx['val']])
        if best is None or v['mae'] < best[1]['mae']:
            best = (alpha, v, m)
    base['ridge_element_fractions'] = {'alpha_chosen_on_val': best[0], 'val': best[1],
                                       'test': metrics(best[2].predict(F[idx['test']]), y[idx['test']])}

    gkey = sha256_bytes(json.dumps(ident_src, sort_keys=True), json.dumps([r['jid'] for r in recs]), a.cutoff, a.max_neighbors)[:24]
    graphs, t_graph, cached = graphs_for(recs, a, gkey)
    n_edges = sum(len(g['src']) for g in graphs)
    print(f'graphs: {len(graphs)} crystals, {sum(len(g["Z"]) for g in graphs)} atoms, {n_edges} edges, '
          f'{"from cache" if cached else f"built in {t_graph:.1f} s"}', flush=True)

    torch.manual_seed(a.seed)
    model = CGCNN(a.atom_features, a.conv, a.hidden, a.cutoff).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    ym, ys = float(ytr.mean()), float(ytr.std() or 1.0)
    rng = np.random.default_rng(a.seed)
    epoch, history, best_val, best_state, train_seconds = 0, [], math.inf, None, 0.0
    ck = run / 'ckpt' / 'last.pt'
    ck.parent.mkdir(exist_ok=True)
    if ck.exists():
        s = torch.load(ck, map_location='cpu', weights_only=False)
        if s['ident'] != ident:
            sys.exit('checkpoint belongs to a different config or data; refusing to resume')
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt'])
        epoch, history, best_val, best_state, train_seconds = s['epoch'], s['history'], s['best_val'], s['best_state'], s['train_seconds']
        rng.bit_generator.state = s['np_rng']; torch.set_rng_state(s['torch_rng']); random.setstate(s['py_rng'])
        if device.type == 'cuda' and s.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(s['cuda_rng'])
        print(f'resumed at epoch {epoch}', flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    ytr_dev = {}
    tr = idx['train']
    while epoch < a.epochs:
        model.train()
        for gparam in opt.param_groups:
            gparam['lr'] = a.lr * (1 - 0.9 * epoch / a.epochs)
        order = tr[rng.permutation(len(tr))]
        sync(device)
        t0 = time.perf_counter()
        tot, cnt = torch.zeros((), device=device), 0  # summed on the device: no per-step host sync
        for s in range(0, len(order), a.batch):
            bi = order[s:s + a.batch]
            if len(bi) < 2:
                continue  # BatchNorm needs more than one crystal
            b = to_device(collate([graphs[i] for i in bi]), device)
            target = torch.as_tensor((y[bi] - ym) / ys, device=device, dtype=torch.float32)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=a.amp):
                pred = model(b['Z'], b['src'], b['dst'], b['dist'].float(), b['batch'], b['n_graphs'])
            loss = torch.nn.functional.mse_loss(pred.float(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach() * len(bi); cnt += len(bi)
        sync(device)
        dt = time.perf_counter() - t0
        tot = float(tot)
        train_seconds += dt
        pv = run_model(model, graphs, idx['val'], device, 256, a.amp) * ys + ym
        mv = metrics(pv, y[idx['val']])
        epoch += 1
        history.append({'epoch': epoch, 'train_mse_std': tot / max(cnt, 1), 'val_mae': mv['mae'], 'seconds': dt,
                        'structures_per_second': len(tr) / dt})
        if mv['mae'] < best_val:
            best_val = mv['mae']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'epoch {epoch:3d}  train mse(std) {tot / max(cnt, 1):.4f}  val MAE {mv["mae"]:.4f}  {dt:.1f} s  '
              f'{len(tr) / dt:.0f} structures/s', flush=True)
        state = {'ident': ident, 'epoch': epoch, 'model': model.state_dict(), 'opt': opt.state_dict(), 'history': history,
                 'best_val': best_val, 'best_state': best_state, 'train_seconds': train_seconds, 'np_rng': rng.bit_generator.state,
                 'torch_rng': torch.get_rng_state(), 'py_rng': random.getstate(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None, 'config': cfg}
        torch.save(state, ck.with_suffix('.tmp'))
        ck.with_suffix('.tmp').replace(ck)
        if a.stop_after_epochs and epoch == a.stop_after_epochs and epoch < a.epochs:
            print('stopped (--stop-after-epochs); rerun with --resume', flush=True)
            raise SystemExit(3)
    model.load_state_dict(best_state)
    best_epoch = 1 + int(np.argmin([h['val_mae'] for h in history]))
    preds = {p: run_model(model, graphs, idx[p], device, 256, a.amp) * ys + ym for p in ('val', 'test')}
    res_model = {p: metrics(preds[p], y[idx[p]]) for p in ('val', 'test')}
    result = {'created': now_iso(), 'git': git_info(), 'device': str(device), 'hardware': hardware(device), 'threads': torch.get_num_threads(),
              'argv': sys.argv, 'config': cfg, 'source': source, 'counts': counts,
              'target': TARGETS[a.target]['label'] + ' (DFT-calculated)' if not a.synthetic else 'synthetic (pipeline check)',
              'split': {'sha256': split_sha, 'sizes': {p: int(len(v)) for p, v in idx.items()},
                        'mode': a.split, 'formulas': {p: int(len({recs[i]['formula'] for i in v})) for p, v in idx.items()},
                        'groups': {p: int(len({group_of(recs[i], a.split) for i in v})) for p, v in idx.items()}},
              'graphs': {'crystals': len(graphs), 'edges': n_edges, 'build_seconds': t_graph, 'from_cache': cached,
                         'cutoff': a.cutoff, 'max_neighbors': a.max_neighbors},
              'baselines': base, 'cgcnn': {'best_epoch_by_val': best_epoch, **res_model,
                                           'parameters': sum(p.numel() for p in model.parameters())},
              'timing': {'load_seconds': t_load, 'train_seconds': train_seconds, 'epochs': len(history),
                         'mean_epoch_seconds': train_seconds / max(1, len(history)),
                         'train_structures_per_second': len(tr) * len(history) / max(train_seconds, 1e-9),
                         'wall_seconds': time.perf_counter() - t_start, 'peak_memory_mb': peak_memory_mb(device)},
              'history': history}
    write_json_new(run / 'result.json', result)
    torch.save(best_state, run / 'best_model.pt')
    save_predictions(run / 'predictions.npz', recs, idx, y, preds, a.split)
    print(f'\n{TARGETS[a.target]["unit"]:<21s} val MAE   test MAE   test R2')
    print(f'mean baseline         {base["mean"]["val"]["mae"]:.4f}    {base["mean"]["test"]["mae"]:.4f}     {base["mean"]["test"]["r2"]:.3f}')
    rb = base['ridge_element_fractions']
    print(f'ridge (fractions)     {rb["val"]["mae"]:.4f}    {rb["test"]["mae"]:.4f}     {rb["test"]["r2"]:.3f}')
    print(f'CGCNN (epoch {best_epoch:3d})     {res_model["val"]["mae"]:.4f}    {res_model["test"]["mae"]:.4f}     {res_model["test"]["r2"]:.3f}')
    print(f'train {train_seconds:.1f} s, {result["timing"]["train_structures_per_second"]:.0f} structures/s on {device}; wrote {run / "result.json"}')
    return result, run


if __name__ == '__main__':
    main()
