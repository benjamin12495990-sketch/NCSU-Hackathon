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


def test_shear_modulus_recipe_recommends_trees():
    """DEC-021. Update this pin when a reviewed commit changes the recipe."""
    from gpu.pipeline import nightly as N
    assert N.load_recipes()['crystal']['shear_modulus_gv']['recommended'] == 'trees'


def test_a_crystal_recipe_records_the_recommended_models_error_and_interval():
    """The networks are trained for every crystal recipe; the registry records the recommended model's error and its interval
    (falling back to the network average's interval, labelled as such, when the recommended model has none). The recipe is
    patched by hand (no pytest fixture), so the plain-function runner below still works."""
    import copy
    from gpu.pipeline import nightly as N
    t = 'shear_modulus_gv'
    net = {'0.90': {'half_width': 0.34, 'coverage': 0.893}}
    full = {'network average': net, 'trees': {'0.90': {'half_width': 0.30, 'coverage': 0.889}}, 'blend': {'0.90': {'half_width': 0.31, 'coverage': 0.90}}}
    cases = [('trees', full, 0.154, 'trees', 0.30), ('trees', {'network average': net}, 0.154, 'network average', 0.34),
             ('blend', full, 0.1524, 'blend', 0.31), ('blend', {'network average': net}, 0.1524, 'network average', 0.34),
             ('network', full, 0.1677, 'network average', 0.34)]
    real, orig = N.load_recipes(), N.load_recipes
    try:
        for rec, mods, mae, imodel, hw in cases:
            r = copy.deepcopy(real)
            r['crystal'][t]['recommended'] = rec
            N.load_recipes = lambda r=r: r
            with tempfile.TemporaryDirectory() as d:
                pipe, reg = Path(d) / 'night', Path(d) / 'registry.json'
                pipe.mkdir()
                plan = {'jobs': [{'id': f'struct_{t}_sys_s11', 'target': t}, {'id': f'trees_{t}_sys', 'target': t}],
                        'nightly': {'keys': {f'crystal:{t}': 'K2'}, 'retrain': [f'crystal:{t}'], 'skipped': [], 'registry': str(reg)}}
                s = {'target': t, 'status': 'ok', 'label': 'shear', 'strongest_trees': 'rich', 'strongest_trees_key': 'pred_rich',
                     'trees': {'rich': {'test': {'mae': 0.154}}}, 'fairness': {'blend': {'mae': 0.1524}},
                     'compare': {'comparisons': {'pred_rich': {'network average of 3 runs': {'network_mae': 0.1677}}}}}
                iv = {'crystal': [{'task': t, 'split': 'system', 'models': mods}]}
                (pipe / 'state.json').write_text(json.dumps({'jobs': {f'struct_{t}_sys_s11': {'status': 'done'}, f'trees_{t}_sys': {'status': 'done'}}}))
                N.update_registry(pipe, plan, {'tabular': [], 'structure_system': [s], 'intervals': iv})
                e = N.load_registry(reg)['tasks'][f'crystal:{t}']
                assert e['recommended'] == rec and e['mae'] == mae and e['network_average_mae'] == 0.1677, (rec, e)
                assert e['interval_model'] == imodel and e['interval90_half_width'] == hw, (rec, e)
    finally:
        N.load_recipes = orig


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
