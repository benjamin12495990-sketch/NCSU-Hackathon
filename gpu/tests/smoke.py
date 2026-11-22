"""CPU-only end-to-end smoke test of the GPU kit on tiny settings:  python -m gpu.tests.smoke

Everything is written to a fresh temporary folder; nothing in the repository is changed.
"""
import contextlib
import io
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from gpu.common import REPO

RESULTS = []


def check(name):
    def deco(fn):
        t0 = time.perf_counter()
        try:
            msg = fn() or ''
            RESULTS.append((name, 'PASS', msg))
        except SkipTest as s:
            RESULTS.append((name, 'SKIP', str(s)))
        except BaseException as e:  # noqa: BLE001
            RESULTS.append((name, 'FAIL', f'{type(e).__name__}: {e}'))
            traceback.print_exc()
        print(f'[{RESULTS[-1][1]}] {name} ({time.perf_counter() - t0:.1f} s) {RESULTS[-1][2]}', flush=True)
        return fn
    return deco


class SkipTest(Exception):
    pass


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **k)


def expect_exit(fn, *a):
    try:
        quiet(fn, *a)
    except SystemExit as e:
        return e.code
    raise AssertionError('expected SystemExit')


def main():
    import torch
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    tmp = Path(tempfile.mkdtemp(prefix='gpu-smoke-'))
    runs = tmp / 'runs'
    print(f'smoke test, torch {torch.__version__}, outputs in {tmp}', flush=True)

    @check('sealed-data guard refuses vault and full concrete files')
    def _():
        from gpu.data import BLOCKED_FILES, SealedDataError, guard_path, load_concrete, read_bytes
        refused = 0
        for p in [*BLOCKED_FILES, REPO / 'lab/vault', REPO / 'lab/vault/future.json', REPO / 'lab/vault/SEALED.json',
                  REPO / 'lab/data/../vault/future.json', REPO / 'public/model/../model/nn_data.json']:
            try:
                guard_path(p)
            except SealedDataError:
                refused += 1
            else:
                raise AssertionError(f'guard let {p} through')
        link = tmp / 'innocent.json'
        try:
            os.symlink(REPO / 'public/model/nn_data.json', link)
        except OSError:  # Windows without Developer Mode cannot create symlinks; skip this sub-check
            link = None
        if link is not None:
            try:
                read_bytes(link)
                raise AssertionError('guard let a symlink to nn_data.json through')
            except SealedDataError:
                refused += 1
        guard_path(REPO / 'public/model/materials/nn_data.json')  # a family file is allowed
        ds = load_concrete()
        assert set(ds.sources) == {'lab/data/train.json', 'lab/data/reward.json', 'lab/data/monitor.json'}, ds.sources
        return f'{refused} paths refused; concrete loader read only {sorted(ds.sources)}'

    @check('guard unit tests (gpu/tests/test_guard.py)')
    def _():
        from gpu.tests import test_guard
        names = [n for n in dir(test_guard) if n.startswith('test_')]
        for n in names:
            getattr(test_guard, n)()
        return f'{len(names)} tests'

    @check('every family loader')
    def _():
        from gpu.data import LOADERS
        out = []
        for fam, fn in LOADERS.items():
            ds = fn()
            assert len(ds.X) == len(ds.y) == len(ds.groups) == len(ds.folds) and np.isfinite(ds.X).all() and np.isfinite(ds.y).all()
            for f in range(int(ds.folds.max()) + 1):  # grouped: no group on both sides of any fold
                assert not set(map(str, ds.groups[ds.folds == f])) & set(map(str, ds.groups[ds.folds != f])), fam
            out.append(f'{fam} {len(ds.y)}x{len(ds.features)}')
        return ', '.join(out)

    @check('check_env (CPU) and refusal to overwrite')
    def _():
        from gpu import check_env
        p = tmp / 'env.json'
        rep = quiet(check_env.main, ['--device', 'cpu', '--size', '256', '--reps', '2', '--output', str(p)])
        assert p.exists() and rep['matmul'][0]['checksum_finite']
        expect_exit(check_env.main, ['--device', 'cpu', '--output', str(p)])
        return f"torch {rep['hardware']['torch']}, {rep['matmul'][0]['tflops']:.3f} TFLOP/s fp32 at n=256"

    @check('batched MLP profiler')
    def _():
        from gpu import profile_batched_mlp as prof
        p = tmp / 'profile.json'
        rep = quiet(prof.main, ['--device', 'cpu', '--counts', '1,4', '--epochs', '1', '--rows', '800', '--output', str(p)])
        expect_exit(prof.main, ['--device', 'cpu', '--counts', '1', '--epochs', '1', '--output', str(p)])
        return ', '.join(f"K={r['K']}: {r['model_rows_per_second']:.0f} model-rows/s" for r in rep['results'])

    @check('bank members are independent (bank of 3 == 3 single fits, float64)')
    def _():
        from gpu.bank import MlpBank, epoch_orders
        g = torch.Generator().manual_seed(0)
        X = torch.randn(64, 5, generator=g, dtype=torch.float64)
        y = torch.randn(64, generator=g, dtype=torch.float64)
        dev = torch.device('cpu')

        def fit(seeds, lrs):
            bank = MlpBank([5, 8, 4, 1], seeds, dev, torch.float64)
            lr, dec = torch.tensor(lrs, dtype=torch.float64), torch.full((len(seeds),), 1e-3, dtype=torch.float64)
            delta = torch.ones(len(seeds), dtype=torch.float64)
            for ep in range(3):
                order = epoch_orders(bank.gens, 64, dev)
                for s in range(0, 64, 16):
                    idx = order[:, s:s + 16]
                    loss, _ = bank.loss(bank.forward(X[idx]), y[idx], delta)
                    loss.backward()
                    bank.step(lr * (1 - 0.9 * ep / 3), dec)
            return bank.forward(X).detach()
        together = fit([11, 112, 213], [0.004, 0.01, 0.002])
        alone = torch.cat([fit([s], [l]) for s, l in [(11, 0.004), (112, 0.01), (213, 0.002)]])
        err = (together - alone).abs().max().item()
        assert err < 1e-10, err
        return f'max difference {err:.1e}'

    from gpu import train_tabular as tt

    @check('tabular: alloys grouped CV (family folds) and fixed split, 2 seeds x 2 configs')
    def _():
        res, run = quiet(tt.main, ['--family', 'alloys', '--mode', 'both', '--device', 'cpu', '--members', '2', '--set', 'epochs=8',
                                   '--grid', '[{}, {"hidden": [16]}]', '--runs-root', str(runs), '--compare-reference'])
        cv = res['modes']['cv']
        ens = cv['configs'][0]['sets']['val']['seed_mean_ensemble']
        assert ens['mae'] < cv['naive_train_mean']['val']['mae'], 'did not beat the training mean'
        assert (run / 'result.json').exists() and (run / 'predictions.npz').exists() and list((run / 'ckpt').glob('*.pt'))
        return f"CV ensemble MAE {ens['mae']:.1f} HV vs naive {cv['naive_train_mean']['val']['mae']:.1f} (8 epochs)"

    @check('tabular: concrete through the guarded loader (TRAIN -> REWARD, MONITOR)')
    def _():
        res, run = quiet(tt.main, ['--family', 'concrete', '--mode', 'fixed', '--device', 'cpu', '--members', '2',
                                   '--set', 'epochs=5', '--runs-root', str(runs)])
        sets = res['modes']['fixed']['configs'][0]['sets']
        assert set(sets) == {'test', 'monitor'}
        assert set(res['data']['sources']) == {'lab/data/train.json', 'lab/data/reward.json', 'lab/data/monitor.json'}
        return f"REWARD MAE {sets['test']['seed_mean_ensemble']['mae']:.2f} MPa, MONITOR {sets['monitor']['seed_mean_ensemble']['mae']:.2f} (5 epochs)"

    @check('tabular: checkpoint resume is exact, and a changed config is refused')
    def _():
        args = ['--family', 'electrolytes', '--device', 'cpu', '--members', '2', '--set', 'epochs=12', '--checkpoint-every', '4',
                '--runs-root', str(runs)]
        _, full = quiet(tt.main, args + ['--tag', 'full'])
        code = expect_exit(tt.main, args + ['--tag', 'part', '--stop-after-epochs', '6'])
        assert code == 3
        part = next((runs / 'electrolytes').glob('*_part'))
        quiet(tt.main, args + ['--resume', str(part)])
        a, b = np.load(full / 'predictions.npz'), np.load(part / 'predictions.npz')
        assert all(np.array_equal(a[k], b[k]) for k in a.files), 'resumed predictions differ'
        code = expect_exit(tt.main, args + ['--set', 'lr=0.01', '--resume', str(part)])
        assert isinstance(code, str) and 'differs' in code, code
        expect_exit(tt.main, args + ['--resume', str(part)])  # complete runs are not reopened
        return 'interrupted-and-resumed run matches the uninterrupted run bit for bit'

    @check('structure: CGCNN on a small subset')
    def _():
        from gpu.structure import train_structure as ts
        if ts.DEFAULT_ZIP.exists():
            args, what = ['--limit', '300', '--epochs', '2'], 'JARVIS dft_3d subset (300)'
        else:
            print('  JARVIS zip not cached (python -m gpu.structure.fetch_jarvis --download); using synthetic crystals', flush=True)
            args, what = ['--synthetic', '240', '--epochs', '2'], 'synthetic crystals (JARVIS not cached)'
        res, run = quiet(ts.main, args + ['--device', 'cpu', '--batch', '32', '--runs-root', str(runs)])
        assert (run / 'split.json').exists() and (run / 'ckpt' / 'last.pt').exists()
        assert np.isfinite(res['cgcnn']['test']['mae'])
        return (f"{what}: val MAE {res['cgcnn']['val']['mae']:.3f} vs mean {res['baselines']['mean']['val']['mae']:.3f}, "
                f"{res['timing']['train_structures_per_second']:.0f} structures/s")

    @check('pipeline: plan builds; orchestrator fake-job tests (retry, resume, STOP, budget, thermal)')
    def _():
        from gpu.pipeline import plan as pl
        from gpu.pipeline.tests import test_orchestrator as to
        for prof in ('smoke', 'standard'):
            plan = pl.build_plan(prof, device='cpu')
            assert plan['jobs'] and pl.describe(plan)
            ids = [j['id'] for j in plan['jobs']]
            assert len(ids) == len(set(ids)), 'duplicate job ids'
        n = 0
        for name, fn in sorted(vars(to).items()):
            if name.startswith('test_') and callable(fn):
                fn()
                n += 1
        from gpu.pipeline.tests import test_analyze as ta, test_evolve as te
        from gpu.tests import test_features as tf, test_splits_finetune as tsf
        for mod in (ta, te, tf, tsf):
            for name, fn in sorted(vars(mod).items()):
                if name.startswith('test_') and callable(fn):
                    fn()
                    n += 1
        return f'{n} pipeline tests passed'

    print('\nsummary')
    for name, status, msg in RESULTS:
        print(f'  {status:4s} {name}: {msg}')
    fails = [r for r in RESULTS if r[1] == 'FAIL']
    print(f'\n{len(RESULTS) - len(fails)}/{len(RESULTS)} passed or skipped; outputs in {tmp}')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
