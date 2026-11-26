"""Leave-out-elements split, fine-tuning leak check, conformal intervals, nightly change detection, run lock.
python -m gpu.tests.test_splits_finetune   (the fine-tuning test trains two tiny CPU networks: ~20 s)"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpu.structure import train_structure as ts


def _recs(n=3000, target='formation_energy_peratom'):
    recs, _, _ = ts.load_records(SimpleNamespace(synthetic=0, data=str(ts.DEFAULT_ZIP), limit=n, seed=1, target=target))
    return recs


def test_element_split_never_shows_a_test_element_in_training_or_validation():
    if not ts.DEFAULT_ZIP.exists():
        return
    test_e, val_e = ts.element_sets()
    assert not test_e & val_e
    recs = _recs()
    parts = ts.parts_of(recs, 'element')
    for r, p in zip(recs, parts):
        els = set(r['atoms']['elements'])
        if p in ('train', 'val'):
            assert not els & test_e, (r['jid'], els)
        if p == 'train':
            assert not els & val_e
    assert {'train', 'val', 'test'} <= set(parts)


def test_system_split_keeps_every_chemical_system_on_one_side_and_formula_split_is_unchanged():
    if not ts.DEFAULT_ZIP.exists():
        return
    recs = _recs(2000, 'bulk_modulus_kv')
    parts = ts.parts_of(recs, 'system')
    side = {}
    for r, p in zip(recs, parts):
        side.setdefault(ts.chemical_system(r['atoms']['elements']), set()).add(p)
    assert all(len(v) == 1 for v in side.values())
    import hashlib
    h = int(hashlib.sha256(b'split-v1:NaCl').hexdigest(), 16) % 100
    assert ts.split_of('NaCl') == ('train' if h < 80 else 'val' if h < 90 else 'test')  # the formula rule of every earlier run


def test_fine_tuning_refuses_a_source_that_saw_the_test_crystals():
    if not ts.DEFAULT_ZIP.exists():
        return
    with tempfile.TemporaryDirectory() as d:
        base = ['--device', 'cpu', '--split', 'element', '--limit', '400', '--epochs', '1', '--runs-root', d]
        ts.main(['--target', 'formation_energy_peratom', '--atom-input', 'both', '--tag', 'src', *base])
        src = next(Path(d, 'structure').glob('*src'))
        res, run = ts.main(['--target', 'bulk_modulus_kv', '--init-from', str(src), '--tag', 'ok', *base])
        cfg = json.loads((run / 'run_config.json').read_text())['config']
        assert cfg['atom_input'] == 'both' and cfg['init_from']['run'] == src.name
        bad = Path(d, 'tampered')
        shutil.copytree(src, bad)
        sp = json.loads((bad / 'split.json').read_text())
        mine = {x[0] for x in json.loads((run / 'split.json').read_text())['jid_part'] if x[1] == 'test'}
        sp['jid_part'] = [[j, 'train' if j in mine else p] for j, p in sp['jid_part']]   # pretend the source trained on this run's test crystals
        (bad / 'split.json').write_text(json.dumps(sp))
        try:
            ts.main(['--target', 'bulk_modulus_kv', '--init-from', str(bad), '--tag', 'leak', *base])
        except SystemExit as e:
            assert 'refusing --init-from' in str(e), e
        else:
            raise AssertionError('a leaking source was accepted')
        try:
            ts.main(['--target', 'bulk_modulus_kv', '--init-from', str(src), '--tag', 'wrong', *base[:2], '--split', 'system', *base[4:]])
        except SystemExit as e:
            assert 'split' in str(e)
        else:
            raise AssertionError('a source from another split was accepted')


def test_conformal_intervals_cover_what_they_promise_on_exchangeable_data():
    from gpu.pipeline.analyze import conformal_cv, conformal_split
    rng = np.random.default_rng(5)
    val, test = rng.standard_t(4, 3000), rng.standard_t(4, 3000)
    out = conformal_split(val, test, np.ones(3000), np.ones(3000))
    for lv in ('0.80', '0.90', '0.95'):
        assert abs(out[lv]['coverage'] - float(lv)) < 0.025, (lv, out[lv])
    folds = np.arange(5000) % 5
    cv = conformal_cv(rng.normal(size=5000), folds)
    assert abs(cv['0.90']['coverage'] - 0.90) < 0.02, cv


def test_nightly_retrains_only_what_changed():
    from gpu.pipeline import nightly as N
    with tempfile.TemporaryDirectory() as d:
        reg = Path(d) / 'registry.json'
        kw = dict(device='cpu', smoke=True, ram_gb=64, cpu_threads=1, registry_path=reg, families=['alloys', 'concrete'], targets=['bulk_modulus_kv'])
        plan = N.build_nightly_plan(**kw)
        first = {j['id'] for j in plan['jobs']}
        assert {'tab_alloys', 'tab_concrete', 'base_alloys'} <= first
        keys = plan['nightly']['keys']
        reg.write_text(json.dumps({'tasks': {k: {'key': v, 'folder': d} for k, v in keys.items()}}))
        again = N.build_nightly_plan(**kw)
        assert again['jobs'] == [] and sorted(again['nightly']['skipped']) == sorted(keys)
        forced = N.build_nightly_plan(**kw, force=True)
        assert {j['id'] for j in forced['jobs']} == first


def test_registry_remembers_a_task_only_when_all_its_jobs_finished():
    from gpu.pipeline import nightly as N
    with tempfile.TemporaryDirectory() as d:
        pipe, reg = Path(d) / 'night', Path(d) / 'registry.json'
        pipe.mkdir()
        plan = {'jobs': [{'id': 'tab_alloys', 'family': 'alloys'}, {'id': 'base_alloys', 'family': 'alloys'}],
                'nightly': {'keys': {'tab:alloys': 'K1'}, 'retrain': ['tab:alloys'], 'skipped': [], 'registry': str(reg)}}
        analysis = {'tabular': [{'family': 'alloys', 'status': 'ok', 'target': 'HV', 'methods': {'trees': {'mae': 70.0}, 'network': {'mae': 75.0}}, 'fairness': {}}], 'intervals': {}}
        (pipe / 'state.json').write_text(json.dumps({'jobs': {'tab_alloys': {'status': 'done'}, 'base_alloys': {'status': 'skipped_budget'}}}))
        N.update_registry(pipe, plan, analysis)
        assert 'tab:alloys' not in N.load_registry(reg)['tasks'], 'a half-finished task must be retried, not remembered'
        lines = '\n'.join(N.registry_lines(plan, pipe))
        assert 'NOT finished' in lines and 'tab:alloys' in lines
        (pipe / 'state.json').write_text(json.dumps({'jobs': {'tab_alloys': {'status': 'done'}, 'base_alloys': {'status': 'done'}}}))
        N.update_registry(pipe, plan, analysis)
        assert N.load_registry(reg)['tasks']['tab:alloys']['key'] == 'K1'
        assert 'Retrained tonight: tab:alloys' in '\n'.join(N.registry_lines(plan, pipe))


def test_run_lock_refuses_a_live_beating_holder_and_takes_over_a_dead_or_silent_one():
    import time
    from gpu.run_pipeline import LockBusy, RunLock
    with tempfile.TemporaryDirectory() as d:
        lock = Path(d) / 'RUNNING.lock'
        p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            lock.write_text(json.dumps({'pid': p.pid, 'started': 'x', 'beat': time.time()}))
            try:
                with RunLock(d):
                    raise AssertionError('started beside a running pipeline')
            except LockBusy:
                pass
            # the same process id, alive, but the beat stopped long ago (a recycled pid after a reboot): taken over
            lock.write_text(json.dumps({'pid': p.pid, 'started': 'x', 'beat': time.time() - 3600}))
            RunLock.BEAT_S = 0.2
            try:
                with RunLock(d):
                    assert json.loads(lock.read_text())['pid'] is not None
            finally:
                RunLock.BEAT_S = 30.0
        finally:
            p.kill()
            p.wait()
        with RunLock(d):                                   # the holder is gone: taken over
            assert json.loads(lock.read_text())['pid'] is not None
        assert json.loads(lock.read_text())['pid'] is None   # released, not deleted


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
