"""PyTorch trainer for the tabular families, mirroring src/lib/nnFamily.js, with banks of networks on one device.

A run trains every (configuration x seed replica) member at once: members with the same layer shapes,
epochs, batch and transforms share one MlpBank (batched matmuls); lr, decay, loss and seed may differ inside
a bank. Seeds follow the JavaScript ensembles: seed + 101 * i.

  python -m gpu.train_tabular --family alloys --mode cv --device cuda --compare-reference
  python -m gpu.train_tabular --family alloys --grid '[{"decay":0.001},{"decay":0.01},{"hidden":[64,32]}]' --members 5
  python -m gpu.train_tabular --family alloys ... --resume gpu/runs/alloys/<run>   (same command + --resume)

Modes: cv    = grouped K-fold on the training rows (the family's own folds where it has them);
       fixed = train on the family's training rows, score its fixed test rows (concrete: REWARD, plus MONITOR);
       both  = cv then fixed.
The fixed test rows were already scored by the family agents, so new fixed-split numbers are exploratory.

Each run writes a new folder gpu/runs/<family>/<UTC time>_<tag>/ (never overwritten): run_config.json,
checkpoints in ckpt/, predictions.npz and result.json. Exact bit parity with the JavaScript trainer is not
expected (different random streams, float32 by default, different summation order); compare metrics.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from gpu.common import REPO, RUNS, git_info, hardware, new_run_dir, now_iso, peak_memory_mb, pick_device, sync, write_json_new

BASE = dict(hidden=[64, 32], epochs=300, batch=32, lr=0.004, decay=1e-3, loss='squared', inputClip=0,
            clampOutput=False, logTarget=False, log1pAuto=False)
FAMILY_DEFAULTS = {  # the settings each family agent chose (domains/<family>/work/result.json)
    'concrete': dict(),  # nnFamily.js defaults; lab/README "plain" baseline uses nn.js inputs instead
    'alloys': dict(hidden=[32, 16], epochs=300, lr=0.004, decay=0.01, loss='squared', inputClip=3, clampOutput=True),
    'insulators': dict(hidden=[64, 32], epochs=150, lr=0.002, decay=0.03, loss='huber'),
    'polymers': dict(hidden=[128, 64], epochs=200, lr=0.004, decay=0.01, loss='huber'),
    'chemistry': dict(hidden=[96, 48], epochs=30, lr=0.004, decay=1e-4, loss='huber', inputClip=5, log1pAuto=True),
    'electrolytes': dict(hidden=[32, 16], epochs=300, lr=0.004, decay=0.01, loss='huber', inputClip=3),  # no agent choice yet
    # added 2026-11-26, no agent choice yet: wide, partly sparse inputs (descriptors + fingerprint counts / one-hot categories)
    'cleaning_pcmc': dict(hidden=[128, 64], epochs=200, lr=0.002, decay=0.01, loss='huber', inputClip=5),
    'cleaning_tension': dict(hidden=[128, 64], epochs=200, lr=0.002, decay=0.01, loss='huber', inputClip=5),
    'corrosion': dict(hidden=[128, 64], epochs=60, lr=0.002, decay=0.01, loss='huber', inputClip=5),
}
BANK_KEYS = ('hidden', 'epochs', 'batch', 'inputClip', 'clampOutput', 'logTarget', 'log1pAuto')
MEMBER_KEYS = ('lr', 'decay', 'loss')


def metrics(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    e = p - y
    sst = float(((y - y.mean()) ** 2).sum())
    return {'n': int(len(y)), 'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt((e ** 2).mean())),
            'r2': float(1 - (e ** 2).sum() / sst) if sst > 0 else None}


class Prep:
    """Transforms fitted on training rows only (nnFamily.js: divide-by-n sd, sd 0 -> 1)."""

    def __init__(self, X, y, cfg):
        X = np.asarray(X, float)
        self.log1p = np.zeros(X.shape[1], bool)
        if cfg['log1pAuto']:  # chemistry rule: log1p if training min >= 0 and training skewness > 1
            m, s = X.mean(0), X.std(0)
            skew = np.where(s > 0, ((X - m) ** 3).mean(0) / np.where(s > 0, s, 1) ** 3, 0)
            self.log1p = (X.min(0) >= 0) & (skew > 1)
        Xt = self._log(X)
        self.mean, self.sd = Xt.mean(0), Xt.std(0)
        self.sd[self.sd == 0] = 1
        self.clip = cfg['inputClip'] if cfg['inputClip'] > 0 else np.inf
        self.logT = cfg['logTarget']
        t = np.log(y) if self.logT else np.asarray(y, float)
        self.ym, self.ys = t.mean(), t.std() or 1.0
        self.ylo, self.yhi = float(np.min(y)), float(np.max(y))
        self.clamp = cfg['clampOutput']

    def _log(self, X):
        X = np.array(X, float)
        X[:, self.log1p] = np.log1p(X[:, self.log1p])
        return X

    def x(self, X):
        return np.clip((self._log(X) - self.mean) / self.sd, -self.clip, self.clip)

    def y(self, y):
        t = np.log(y) if self.logT else np.asarray(y, float)
        return (t - self.ym) / self.ys

    def inv(self, z):
        p = z * self.ys + self.ym
        p = np.exp(p) if self.logT else p
        return np.clip(p, self.ylo, self.yhi) if self.clamp else p


def parse_value(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def build_plan(a):
    base = {**BASE, **FAMILY_DEFAULTS[a.family]}
    for kv in a.set or []:
        k, v = kv.split('=', 1)
        if k not in BASE:
            sys.exit(f'--set: unknown option {k}; options are {sorted(BASE)}')
        base[k] = parse_value(v)
    grid = [{}]
    if a.grid:
        grid = json.loads(Path(a.grid).read_text()) if Path(a.grid).suffix == '.json' and Path(a.grid).exists() else json.loads(a.grid)
    configs = []
    for i, g in enumerate(grid):
        bad = set(g) - set(BASE)
        if bad:
            sys.exit(f'--grid entry {i}: unknown options {sorted(bad)}')
        c = {**base, **g}
        c['hidden'] = [int(h) for h in c['hidden']]
        if c['loss'] not in ('squared', 'huber'):
            sys.exit(f"loss must be squared or huber, got {c['loss']}")
        configs.append(c)
    seeds = [a.seed + 101 * i for i in range(a.members)]
    banks = {}
    for ci, c in enumerate(configs):
        key = json.dumps({k: c[k] for k in BANK_KEYS}, sort_keys=True)
        banks.setdefault(key, []).extend((ci, s) for s in seeds)
    return configs, seeds, [{'shared': json.loads(k), 'members': v} for k, v in banks.items()]


def train_unit(unit_id, bank_spec, configs, Xtr, ytr, evals, device, dtype, ckpt_dir, ident, ckpt_every, stop_after):
    """Trains one bank on one training set; returns {eval name: predictions [K, n]} in target units."""
    import torch
    from gpu.bank import SQUARED_DELTA, MlpBank, epoch_orders
    shared, members = bank_spec['shared'], bank_spec['members']
    prep = Prep(Xtr, ytr, shared)
    X = torch.tensor(prep.x(Xtr), device=device, dtype=dtype)
    y = torch.tensor(prep.y(ytr), device=device, dtype=dtype)
    n, d = X.shape
    K = len(members)
    bank = MlpBank([d, *shared['hidden'], 1], [s for _, s in members], device, dtype)
    col = lambda k: torch.tensor([configs[ci][k] for ci, _ in members], device=device, dtype=dtype)  # noqa: E731
    lr0, decay = col('lr'), col('decay')
    delta = torch.tensor([1.0 if configs[ci]['loss'] == 'huber' else SQUARED_DELTA for ci, _ in members], device=device, dtype=dtype)
    E, B = shared['epochs'], shared['batch']
    ck = ckpt_dir / f'{unit_id}.pt'
    epoch, history, seconds = 0, [], 0.0
    if ck.exists():
        s = torch.load(ck, map_location='cpu', weights_only=False)
        if s['ident'] != ident:
            sys.exit(f'checkpoint {ck} was written for a different config or data hash; refusing to resume')
        if s.get('done'):
            return s['preds'], s['history'], s['seconds'], True
        bank.load_state(s['bank'])
        epoch, history, seconds = s['epoch'], s['history'], s['seconds']
        torch.set_rng_state(s['torch_rng'])
        print(f'  resumed {unit_id} at epoch {epoch}/{E}', flush=True)

    def predict():
        with torch.no_grad():
            out = {}
            for name, Xe in evals.items():
                xe = torch.tensor(prep.x(Xe), device=device, dtype=dtype)
                out[name] = prep.inv(bank.forward(xe).double().cpu().numpy())
            return out

    def save(done, preds=None):
        state = {'ident': ident, 'unit': unit_id, 'epoch': epoch, 'bank': bank.state(), 'history': history,
                 'seconds': seconds, 'torch_rng': torch.get_rng_state(), 'np_rng': np.random.get_state(),
                 'done': done, 'preds': preds, 'config': bank_spec}
        tmp = ck.with_suffix('.tmp')
        torch.save(state, tmp)
        tmp.replace(ck)  # a checkpoint replaces only its own previous checkpoint inside this run folder

    sync(device)
    t0 = time.perf_counter()
    while epoch < E:
        order = epoch_orders(bank.gens, n, device)
        lr = lr0 * (1 - 0.9 * epoch / E)
        sq = torch.zeros(K, device=device, dtype=dtype)
        for s in range(0, n, B):
            idx = order[:, s:s + B]
            loss, se = bank.loss(bank.forward(X[idx]), y[idx], delta)
            loss.backward()
            bank.step(lr, decay)
            sq += se
        epoch += 1
        if epoch % max(1, E // 20) == 0 or epoch == E:
            history.append({'epoch': epoch, 'train_mse_target_units': (sq / n * prep.ys ** 2).cpu().tolist()})
        if (ckpt_every and epoch % ckpt_every == 0 and epoch < E) or (stop_after and epoch == stop_after):
            sync(device)
            seconds += time.perf_counter() - t0
            save(False)
            t0 = time.perf_counter()
            if stop_after and epoch == stop_after:
                print(f'  stopped {unit_id} at epoch {epoch} (--stop-after-epochs); rerun with --resume', flush=True)
                raise SystemExit(3)
    sync(device)
    seconds += time.perf_counter() - t0
    preds = predict()
    save(True, preds)
    return preds, history, seconds, False


def reference_numbers(family, mode):
    p = REPO / 'domains' / {'alloys': 'materials'}.get(family, family) / 'work' / 'result.json'
    if family == 'concrete':
        return {'file': 'lab/README.md (smoke-test table)', 'select_mae_plain_network_MPa': 12.467,
                'note': 'plain nn.js network (64-32, its own base inputs, squared, 300 epochs, seed 11) on SELECT = REWARD + MONITOR; '
                        'inputs and scored set differ from this kit, so it is context only'}
    if family == 'electrolytes':
        q = REPO / 'domains/electrolytes/work/reproductions/obelix_group_cv.json'
        if not q.exists():
            return {'note': f'{q.relative_to(REPO).as_posix()} not in this checkout (it is on main after 1d1b90b); nothing to compare'}
        r = json.loads(q.read_text())
        m = r.get('models', {})
        return {'file': q.relative_to(REPO).as_posix(), 'parity_note': 'same groups and GroupKFold(5) folds; the mean baseline should match exactly',
                'cv_mean_baseline': {'mae': m.get('mean', {}).get('pooled_oof_mae'), 'rmse': m.get('mean', {}).get('pooled_oof_rmse'),
                                     'per_fold_mae': m.get('mean', {}).get('fold_mae')},
                'cv_selected_' + str(r.get('selected_model')): {'mae': r.get('selected_pooled_oof_mae'),
                                                                'rmse': m.get(r.get('selected_model'), {}).get('pooled_oof_rmse'),
                                                                'r2': m.get(r.get('selected_model'), {}).get('pooled_oof_r2')},
                'mode_run': mode}
    if not p.exists():
        return {'note': f'no result.json for {family}; nothing to compare'}
    r = json.loads(p.read_text())
    pick = lambda d, *ks: {k: d[k] for k in ks if k in d}  # noqa: E731
    out = {'file': p.relative_to(REPO).as_posix(), 'parity_note': 'JavaScript reference (float64, its own RNG); bit parity is not expected'}
    if family == 'alloys':
        out['cv_network_ensemble'] = pick(r['cv_on_training_rows']['network_ensemble'], 'mae', 'rmse', 'r2', 'per_fold_mae')
        out['cv_network_single'] = pick(r['cv_on_training_rows']['network_single_chosen'], 'mae', 'rmse', 'r2')
        out['cv_naive'] = pick(r['cv_on_training_rows']['naive_same_folds'], 'mae', 'rmse', 'r2')
        out['fixed_test_network_ensemble'] = r['b_test_split_once']['network']
    elif family == 'insulators':
        s = r['network_settings_chosen']
        out['cv_single_network_mae'] = s.get('single_net_cv_mae_log10')
        out['cv_note'] = s.get('members_reason')
        out['fixed_test_network_ensemble'] = pick(r['b_test_split_once']['network'], 'mae_log10', 'r2_log10')
    elif family == 'polymers':
        ch = r['network']['chosen_settings']
        for g in r['network'].get('cv_grid', []):
            c = g['config']
            if c['hidden'] == ch['hidden'] and c['decay'] == ch['decay'] and c['loss'] == ch['loss'] and c['epochs'] == ch['epochs']:
                out['cv_single_network_mae'] = g.get('cv_mae_single_net')
        out['fixed_test_network_ensemble'] = r['b_test_split_seed42']['network']
    elif family == 'chemistry':
        ch = r['network']['chosen_settings']
        for g in r['a_cv_on_training_rows']['grid_single_seed']:
            if g['hidden'] == ch['hidden'] and g['decay'] == ch['decay'] and g['loss'] == ch['loss']:
                out['cv_single_network'] = pick(g, 'cv_mae', 'cv_rmse')
        out['fixed_test_network_ensemble'] = r['b_test_once']['network_ensemble']
    out['mode_run'] = mode
    return out


def print_comparison(result):
    """The family's JavaScript reference numbers beside this run's first configuration."""
    ref = result['reference']
    print(f"\nreference: {ref.get('file', '')}  ({ref.get('parity_note') or ref.get('note', '')})")
    fmt = lambda v: f'{v:.4g}' if isinstance(v, (int, float)) else (v if isinstance(v, str) else '-')  # noqa: E731
    for mode, mres in result['modes'].items():
        sets = mres['configs'][0]['sets']
        for name, e in sets.items():
            en = e['seed_mean_ensemble']
            print(f"  this run  [{mode}/{name}] seed-mean ensemble MAE {fmt(en['mae'])} RMSE {fmt(en['rmse'])} R2 {fmt(en['r2'])} | "
                  f"members MAE {fmt(e['member_mae_mean'])} | train mean MAE {fmt(mres['naive_train_mean'][name]['mae'])}")
        prefix = 'cv_' if mode == 'cv' else 'fixed_'
        for k, v in ref.items():
            if not k.startswith(prefix):
                continue
            if isinstance(v, dict):
                mae = v.get('mae', v.get('cv_mae', v.get('mae_log10')))
                rmse = v.get('rmse', v.get('cv_rmse'))
                r2 = v.get('r2', v.get('r2_log10'))
                print(f'  reference [{k}] MAE {fmt(mae)} RMSE {fmt(rmse)} R2 {fmt(r2)}')
            else:
                print(f'  reference [{k}] {fmt(v)}')
    for k in ('select_mae_plain_network_MPa',):
        if k in ref:
            print(f'  reference [{k}] {ref[k]}')
    print('  (exact parity with the reference trainer is not expected; compare within seed noise)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--family', required=True, choices=sorted(FAMILY_DEFAULTS))
    ap.add_argument('--mode', default='cv', choices=['cv', 'fixed', 'both'])
    ap.add_argument('--device', default='auto')
    ap.add_argument('--dtype', default='float32', choices=['float32', 'float64'])
    ap.add_argument('--members', type=int, default=5, help='seed replicas per configuration (seeds seed + 101 i)')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--set', action='append', metavar='KEY=VALUE', help='override a family setting, e.g. --set epochs=50 --set hidden=[64,32]')
    ap.add_argument('--grid', help='JSON list of setting overrides (or a .json file); every entry x every seed is trained')
    ap.add_argument('--checkpoint-every', type=int, default=25, help='epochs between checkpoints (0 = only at the end)')
    ap.add_argument('--stop-after-epochs', type=int, default=0, help=argparse.SUPPRESS)  # for the resume test
    ap.add_argument('--runs-root', default=str(RUNS))
    ap.add_argument('--tag', default='')
    ap.add_argument('--resume', help='an existing run folder of this exact command; continues from its checkpoints')
    ap.add_argument('--compare-reference', action='store_true', help="print the family's JavaScript reference numbers")
    ap.add_argument('--threads', type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    from gpu.data import load
    device = pick_device(a.device)
    dtype = getattr(torch, a.dtype)
    ds = load(a.family)
    configs, seeds, banks = build_plan(a)
    if a.mode in ('fixed', 'both') and ds.X_test is None:
        sys.exit(f'{a.family} has no fixed test rows in this kit (CV only)')
    run_config = {'family': a.family, 'mode': a.mode, 'dtype': a.dtype, 'members': a.members, 'seeds': seeds,
                  'configs': configs, 'banks': banks, 'data_sha256': ds.data_sha256}
    ident = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    if a.resume:
        run = Path(a.resume)
        old = json.loads((run / 'run_config.json').read_text())
        if old['ident'] != ident:
            diff = {k: (old['run'].get(k), run_config.get(k)) for k in run_config if old['run'].get(k) != run_config.get(k)}
            sys.exit(f'refusing to resume {run}: config or data hash differs: {json.dumps(diff)[:2000]}')
        if (run / 'result.json').exists():
            sys.exit(f'{run} is already complete (result.json exists); start a new run instead')
    else:
        run = new_run_dir(Path(a.runs_root) / a.family, a.tag)
        write_json_new(run / 'run_config.json', {'ident': ident, 'run': run_config, 'created': now_iso(), 'git': git_info(),
                                                 'argv': sys.argv, 'data': ds.summary()})
    ckpt_dir = run / 'ckpt'
    ckpt_dir.mkdir(exist_ok=True)
    print(f'{a.family}: {len(ds.y)} training rows, {len(ds.features)} inputs | {len(configs)} config(s) x {a.members} seeds '
          f'in {len(banks)} bank(s) | device {device} {a.dtype} | run {run}', flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    t_all = time.perf_counter()
    result = {'created': now_iso(), 'git': git_info(), 'device': str(device), 'dtype': a.dtype, 'hardware': hardware(device),
              'threads': torch.get_num_threads(), 'argv': sys.argv, 'data': ds.summary(), 'configs': configs, 'seeds': seeds,
              'parity_note': 'PyTorch mirror of nnFamily.js; exact bit parity with the JavaScript trainer is not expected', 'modes': {}}
    preds_out = {}
    modes = ['cv', 'fixed'] if a.mode == 'both' else [a.mode]
    for mode in modes:
        if mode == 'cv':
            units = [(f'cv{f}', ds.folds != f, {'val': ds.folds == f}) for f in range(int(ds.folds.max()) + 1)]
            n_eval = {'val': len(ds.y)}
        else:
            units = [('fixed', np.ones(len(ds.y), bool), None)]
            n_eval = {'test': len(ds.y_test), **{k: len(v[1]) for k, v in ds.extra.items()}}
        # pooled predictions: [members of config c (seed order), rows]
        pooled = {name: np.full((len(configs), a.members, n), np.nan) for name, n in n_eval.items()}
        per_fold, seconds, naive = [], 0.0, {name: np.full(n, np.nan) for name, n in n_eval.items()}
        for uname, trmask, evmask in units:
            Xtr, ytr = ds.X[trmask], ds.y[trmask]
            if mode == 'cv':
                evals = {'val': ds.X[evmask['val']]}
                naive['val'][evmask['val']] = ytr.mean()
            else:
                evals = {'test': ds.X_test, **{k: v[0] for k, v in ds.extra.items()}}
                for k in evals:
                    naive[k][:] = ytr.mean()
            for bi, spec in enumerate(banks):
                p, hist, sec, cached = train_unit(f'{uname}-bank{bi}', spec, configs, Xtr, ytr, evals, device, dtype,
                                                  ckpt_dir, ident, a.checkpoint_every, a.stop_after_epochs)
                seconds += sec
                print(f'  {mode} {uname} bank{bi}: {len(spec["members"])} members, {spec["shared"]["epochs"]} epochs, '
                      f'{sec:.2f} s{" (from checkpoint)" if cached else ""}', flush=True)
                for k, (ci, s) in enumerate(spec['members']):
                    j = seeds.index(s)
                    for name in evals:
                        if mode == 'cv':
                            pooled[name][ci, j, evmask['val']] = p[name][k]
                        else:
                            pooled[name][ci, j] = p[name][k]
        truth = {'val': ds.y} if mode == 'cv' else {'test': ds.y_test, **{k: v[1] for k, v in ds.extra.items()}}
        mres = {'train_seconds': seconds, 'naive_train_mean': {k: metrics(naive[k], truth[k]) for k in truth}, 'configs': []}
        for ci, c in enumerate(configs):
            entry = {'config': c, 'sets': {}}
            for name in truth:
                mem = [metrics(pooled[name][ci, j], truth[name]) for j in range(a.members)]
                ens = pooled[name][ci].mean(0)
                e = {'members': mem, 'member_mae_mean': float(np.mean([m['mae'] for m in mem])),
                     'member_mae_sd': float(np.std([m['mae'] for m in mem])), 'seed_mean_ensemble': metrics(ens, truth[name])}
                if mode == 'cv':
                    e['per_fold_mae_ensemble'] = [float(np.abs(ens[ds.folds == f] - ds.y[ds.folds == f]).mean()) for f in range(int(ds.folds.max()) + 1)]
                entry['sets'][name] = e
            mres['configs'].append(entry)
        result['modes'][mode] = mres
        for name, arr in pooled.items():
            preds_out[f'{mode}_{name}'] = arr
    sync(device)
    result['wall_seconds'] = time.perf_counter() - t_all
    result['peak_memory_mb'] = peak_memory_mb(device)
    if a.compare_reference:
        result['reference'] = reference_numbers(a.family, a.mode)
    np.savez_compressed(run / 'predictions.npz', **preds_out)
    write_json_new(run / 'result.json', result)

    for mode, mres in result['modes'].items():
        print(f'\n[{mode}] training time {mres["train_seconds"]:.2f} s on {device}')
        for name, nm in mres['naive_train_mean'].items():
            print(f'  {name}: naive train-mean MAE {nm["mae"]:.4g}')
        for ci, entry in enumerate(mres['configs']):
            diff = {k: v for k, v in entry['config'].items() if configs[0].get(k) != v}
            for name, e in entry['sets'].items():
                en = e['seed_mean_ensemble']
                print(f'  config {ci} {json.dumps(diff) if ci else ""} {name}: members MAE {e["member_mae_mean"]:.4g} +- {e["member_mae_sd"]:.2g} | '
                      f'ensemble MAE {en["mae"]:.4g} RMSE {en["rmse"]:.4g} R2 {en["r2"]:.4f}')
    if a.compare_reference:
        print_comparison(result)
    print(f'\nwrote {run / "result.json"}  (wall {result["wall_seconds"]:.1f} s)')
    return result, run


if __name__ == '__main__':
    main()
