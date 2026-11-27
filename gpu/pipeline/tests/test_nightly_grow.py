"""Seed growth of unchanged crystal properties (DEC-023) and the benchmark hold-out of the TBmBJ task (DEC-022).
python -m pytest gpu/pipeline/tests/test_nightly_grow.py   (no data, no training: fake run folders only)"""
import json
import tempfile
from pathlib import Path

from gpu.pipeline import nightly as N


def _run(folder, job, files=('result.json', 'predictions.npz', 'run_config.json', 'split.json', 'best_model.pt'), ckpt=True, config=None):
    run = Path(folder) / 'jobs' / job / 'structure' / f'20261127T000000Z_{job}'
    run.mkdir(parents=True)
    for f in files:
        (run / f).write_text(json.dumps({'config': config or {'epochs': 100}}) if f == 'run_config.json' else f'{job}/{f}')
    if ckpt:
        (run / 'ckpt').mkdir()
        (run / 'ckpt' / 'last.pt').write_text('optimizer state, not carried')
    return run


def _trees(folder, t):
    d = Path(folder) / 'jobs' / f'trees_{t}_sys'
    d.mkdir(parents=True)
    (d / 'trees.json').write_text('{}')
    (d / 'trees_predictions.npz').write_text('npz')
    (d / 'stdout.log').write_text('not carried')
    return d


def test_completed_seeds_needs_result_and_predictions():
    with tempfile.TemporaryDirectory() as d:
        _run(d, 'struct_bulk_modulus_kv_sys_s11')
        _run(d, 'struct_bulk_modulus_kv_sys_s12', files=('result.json',))          # unfinished: no predictions
        _run(d, 'struct_bulk_modulus_kv_sys_s13')
        _run(d, 'struct_shear_modulus_gv_sys_s14')                                # another property
        assert N.completed_seeds(d, 'bulk_modulus_kv') == [11, 13]


def test_grow_plan_adds_two_new_seeds_after_the_highest():
    with tempfile.TemporaryDirectory() as d:
        for sd in (21, 22):
            _run(d, f'struct_formation_energy_peratom_sys_s{sd}')
        _trees(d, 'formation_energy_peratom')
        g, why = N.grow_plan('formation_energy_peratom', {'folder': d})
        assert why is None
        have, new, folder = g
        assert have == [21, 22] and new == [23, 24] and folder == Path(d)


def test_grow_plan_stops_at_the_cap_and_never_overshoots():
    with tempfile.TemporaryDirectory() as d:
        for sd in range(11, 20):                                                  # 9 seeds
            _run(d, f'struct_bulk_modulus_kv_sys_s{sd}')
        _trees(d, 'bulk_modulus_kv')
        (have, new, _), _ = N.grow_plan('bulk_modulus_kv', {'folder': d}, cap=10)
        assert len(have) == 9 and new == [20]                                     # one to reach 10, not two
        _run(d, 'struct_bulk_modulus_kv_sys_s20')
        assert N.grow_plan('bulk_modulus_kv', {'folder': d}, cap=10) == (None, None)   # full: silent skip


def test_grow_plan_refuses_without_trees_or_folder():
    with tempfile.TemporaryDirectory() as d:
        _run(d, 'struct_bulk_modulus_kv_sys_s11')
        g, why = N.grow_plan('bulk_modulus_kv', {'folder': d})
        assert g is None and 'tree baseline' in why
    g, why = N.grow_plan('bulk_modulus_kv', {'folder': '/no/such/folder'})
    assert g is None and 'not on this computer' in why
    g, why = N.grow_plan('bulk_modulus_kv', {})
    assert g is None


def test_carry_copies_only_what_the_analysis_reads_and_is_idempotent():
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as pipe:
        _run(src, 'struct_bulk_modulus_kv_sys_s11')
        _trees(src, 'bulk_modulus_kv')
        plan = {'carry': [{'target': 'bulk_modulus_kv', 'from': src,
                           'jobs': ['struct_bulk_modulus_kv_sys_s11', 'trees_bulk_modulus_kv_sys', 'struct_bulk_modulus_kv_sys_s99']}]}
        logs = []
        N.carry_runs(pipe, plan, log=logs.append)
        dst = Path(pipe) / 'jobs' / 'struct_bulk_modulus_kv_sys_s11'
        files = sorted(p.name for p in dst.rglob('*') if p.is_file())
        assert files == sorted(['best_model.pt', 'carried.json', 'predictions.npz', 'result.json', 'run_config.json', 'split.json'])
        assert not any('ckpt' in p.parts for p in dst.rglob('*'))
        tfiles = sorted(p.name for p in (Path(pipe) / 'jobs' / 'trees_bulk_modulus_kv_sys').iterdir())
        assert tfiles == ['carried.json', 'trees.json', 'trees_predictions.npz']
        assert any('s99 is missing' in m or 'struct_bulk_modulus_kv_sys_s99 is missing' in m for m in logs)
        assert N.completed_seeds(pipe, 'bulk_modulus_kv') == [11]
        # a second call (a resumed night) changes nothing and never overwrites
        marker = dst / 'structure' / '20261127T000000Z_struct_bulk_modulus_kv_sys_s11' / 'result.json'
        marker.write_text('tonight')
        N.carry_runs(pipe, plan, log=logs.append)
        assert marker.read_text() == 'tonight'
        assert not list(Path(pipe, 'jobs').glob('.carry*'))


def test_crystal_jobs_with_explicit_seeds_and_no_trees():
    rec = {'seeds': [11, 12, 13], 'epochs': 100, 'genes': 'default'}
    jobs = N.crystal_jobs('bulk_modulus_kv', rec, device='cuda', smoke=False, small_ram=False, cpu_threads=2, seeds=[14, 15], trees=False)
    assert [j['id'] for j in jobs] == ['struct_bulk_modulus_kv_sys_s14', 'struct_bulk_modulus_kv_sys_s15']
    assert all('--neighbor-ties' in j['argv'] for j in jobs)
    full = N.crystal_jobs('bulk_modulus_kv', rec, device='cuda', smoke=False, small_ram=False, cpu_threads=2)
    assert [j['id'] for j in full][-1] == 'trees_bulk_modulus_kv_sys' and len(full) == 4


def test_growth_settings_are_not_part_of_any_recipe():
    """A recipe entry is part of the task key: growth settings written there would retrain every property from scratch (DEC-023)."""
    rec = json.loads(N.RECIPES.read_text())
    for t, e in rec['crystal'].items():
        assert 'max_seeds' not in e and 'grow' not in e, t
    assert 'mbj_bandgap' in rec['crystal'] and 'mbj_bandgap' in N.NIGHTLY_EXTRA_TARGETS


def test_benchmark_holdout_file():
    from gpu.structure.train_structure import TARGETS
    spec = TARGETS['mbj_bandgap']
    f = Path(N.__file__).resolve().parents[1] / 'structure' / spec['exclude']
    d = json.loads(f.read_text())
    assert d['count'] == len(d['jids']) == len(set(d['jids'])) == 54
    assert all(j.startswith('JVASP-') for j in d['jids'])
    assert all('exclude' not in s for k, s in TARGETS.items() if k != 'mbj_bandgap')   # the other four tasks are unchanged


def test_hold_out_list_is_part_of_the_key_only_where_it_exists():
    """A changed hold-out list must retrain the TBmBJ task from scratch; the four older properties keep exactly their old keys."""
    assert N._exclude_sha('bulk_modulus_kv') == [] and N._exclude_sha('optb88vdw_bandgap') == []
    assert len(N._exclude_sha('mbj_bandgap')) == 1 and len(N._exclude_sha('mbj_bandgap')[0]) == 64
    rec = {'seeds': [11]}
    assert N._h('sha', rec, N.CODE_VERSION['crystal'], *N._exclude_sha('bulk_modulus_kv')) == N._h('sha', rec, N.CODE_VERSION['crystal'])


def test_grow_plan_refuses_seeds_trained_with_other_settings():
    with tempfile.TemporaryDirectory() as d:
        _run(d, 'struct_bulk_modulus_kv_sys_s11', config={'epochs': 100, 'limit': 0, 'neighbor_ties': 'all'})
        _run(d, 'struct_bulk_modulus_kv_sys_s12', config={'epochs': 3, 'limit': 800, 'neighbor_ties': 'all'})
        _trees(d, 'bulk_modulus_kv')
        g, why = N.grow_plan('bulk_modulus_kv', {'folder': d}, expect={'epochs': 100, 'limit': 0, 'neighbor_ties': 'all'})
        assert g is None and 'other settings' in why and 'seed 12' in why
        _run(d, 'struct_shear_modulus_gv_sys_s11', config={'epochs': 100, 'limit': None, 'neighbor_ties': 'all'})   # no limit stored as null
        _trees(d, 'shear_modulus_gv')
        g, why = N.grow_plan('shear_modulus_gv', {'folder': d}, expect={'epochs': 100, 'limit': 0, 'neighbor_ties': 'all'})
        assert why is None and g[1] == [12, 13]


def test_expected_config_matches_the_jobs_the_nightly_writes():
    rec = {'seeds': [21, 22], 'epochs': 100, 'genes': {'atom_features': 128, 'conv': 5, 'hidden': 192, 'lr': 0.004, 'weight_decay': 1e-05, 'batch': 64}}
    e = N.expected_config('formation_energy_peratom', rec, smoke=False, small_ram=False)
    assert e['epochs'] == 100 and e['limit'] == 0 and e['neighbor_ties'] == 'all' and e['conv'] == 5 and e['split'] == 'system'
    assert N.expected_config('formation_energy_peratom', rec, smoke=False, small_ram=True)['limit'] == 40000
    assert N.expected_config('bulk_modulus_kv', {'epochs': 100, 'genes': 'default'}, smoke=True, small_ram=False)['limit'] == 800


def test_carry_reports_incomplete_targets_and_never_leaves_a_matching_folder():
    with tempfile.TemporaryDirectory() as root:
        src, pipe = Path(root) / '20261127T000000Z_nightly', Path(root) / '20261128T000000Z_nightly'
        _run(src, 'struct_bulk_modulus_kv_sys_s11')
        (src / 'jobs' / 'struct_bulk_modulus_kv_sys_s12' / 'structure').mkdir(parents=True)        # a folder with nothing to carry
        _trees(src, 'bulk_modulus_kv')
        plan = {'carry': [{'target': 'bulk_modulus_kv', 'from': src.name,                         # stored by name: resolved beside tonight's
                           'jobs': ['struct_bulk_modulus_kv_sys_s11', 'struct_bulk_modulus_kv_sys_s12', 'trees_bulk_modulus_kv_sys']}]}
        logs = []
        bad = N.carry_runs(pipe, plan, log=logs.append)
        assert bad == {'bulk_modulus_kv'}
        assert N.completed_seeds(pipe, 'bulk_modulus_kv') == [11]
        assert sorted(p.name for p in (pipe / 'jobs').iterdir()) == ['struct_bulk_modulus_kv_sys_s11', 'trees_bulk_modulus_kv_sys']
        assert json.loads((pipe / 'jobs' / 'struct_bulk_modulus_kv_sys_s11' / 'carried.json').read_text())['from_folder'] == src.name


def test_analysis_ignores_folders_that_only_look_like_seed_jobs():
    from gpu.pipeline.analyze import _runs
    with tempfile.TemporaryDirectory() as d:
        _run(d, 'struct_bulk_modulus_kv_sys_s11')
        _run(d, 'struct_bulk_modulus_kv_sys_s11.carrying-1-2')   # a temp name of the first version of carry_runs
        assert [Path(r).parents[1].name for r in _runs(d, 'struct_bulk_modulus_kv_sys_s[0-9]*')] == ['struct_bulk_modulus_kv_sys_s11']
