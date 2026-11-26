"""The neighbour-rule comparison profile and the rule labels the report relies on.  python -m gpu.pipeline.tests.test_tiefix  (no data, no training)"""
import json
import tempfile
from pathlib import Path

from gpu.pipeline import plan as pl
from gpu.pipeline.analyze import SPLIT_SPECS, rule_of_run, rule_of_trees


def _strip_rule(argv):
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
        elif x == '--neighbor-ties':
            skip = True
        else:
            out.append(x)
    return out


def _existing(limit=None):
    """The held-out-systems jobs of a folder planned BEFORE the fix: what the holdout profile writes, without the rule flag it now adds."""
    jobs = pl.holdout_jobs(device='cuda', targets=['bulk_modulus_kv', 'formation_energy_peratom'], smoke=False, small_ram=False, cpu_threads=2, existing=None)
    for j in jobs:
        j['argv'] = _strip_rule(j['argv'])
        if limit:
            j['argv'] += ['--limit', str(limit)]
    return jobs


def _crystal(j):
    return any(m in j.get('argv', []) for m in pl.CRYSTAL_MODULES)


def test_the_comparison_repeats_the_same_seeds_epochs_and_records_and_only_changes_the_rule():
    old = _existing(limit=40000)
    new = pl.tiefix_jobs(device='cuda', smoke=False, cpu_threads=2, existing=old)
    ids = [j['id'] for j in new]
    assert len(ids) == len(set(ids)) and not set(ids) & {j['id'] for j in old}          # nothing existing is rerun or replaced
    for t, seeds in (('bulk_modulus_kv', [11, 12, 13]), ('formation_energy_peratom', [11, 12])):
        for sd in seeds:
            o = next(j for j in old if j['id'] == f'struct_{t}_sys_s{sd}')
            n = next(j for j in new if j['id'] == f'struct_{t}_sysT_s{sd}')
            val = lambda j, f: j['argv'][j['argv'].index(f) + 1]  # noqa: E731
            for flag in ('--target', '--split', '--epochs', '--seed', '--limit'):
                assert val(n, flag) == val(o, flag), (flag, n['id'])
            assert val(n, '--neighbor-ties') == 'all' and pl.job_rule(None, o) == 'first'   # an old line without the flag is the original rule
        trees = next(j for j in new if j['id'] == f'trees_{t}_sysT')
        assert trees['argv'][trees['argv'].index('--neighbor-ties') + 1] == 'all' and '--limit' in trees['argv'] and 'rich' in ' '.join(trees['argv'])
        inv = {j['id']: j for j in new if j['id'].startswith(f'inv_{t}_')}
        assert inv[f'inv_{t}_first']['after'] == [f'struct_{t}_sys_s11'] and inv[f'inv_{t}_all']['after'] == [f'struct_{t}_sysT_s11']
        assert '{RUN_OF:struct_' + t + '_sysT_s11}' in inv[f'inv_{t}_all']['argv']


def test_nothing_is_planned_without_the_original_runs_and_their_trees():
    assert pl.tiefix_jobs(device='cuda', smoke=False, cpu_threads=2, existing=[]) == []
    no_trees = [j for j in _existing() if not j['id'].startswith('trees_')]
    assert pl.tiefix_jobs(device='cuda', smoke=False, cpu_threads=2, existing=no_trees) == []
    plan = pl.build_plan('tiefix', device='cuda', existing=_existing())
    assert plan['jobs'] and pl.describe(plan)


def test_the_original_and_the_fixed_runs_are_never_read_as_one_set():
    import fnmatch
    for t in ('bulk_modulus_kv', 'formation_energy_peratom'):
        sys_pat, fix_pat = SPLIT_SPECS['system'][0].format(t=t), SPLIT_SPECS['systemT'][0].format(t=t)
        assert fnmatch.fnmatch(f'struct_{t}_sys_s11', sys_pat) and not fnmatch.fnmatch(f'struct_{t}_sysT_s11', sys_pat)
        assert fnmatch.fnmatch(f'struct_{t}_sysT_s11', fix_pat) and not fnmatch.fnmatch(f'struct_{t}_sys_s11', fix_pat)


def test_runs_and_tree_files_made_before_the_rule_was_recorded_count_as_the_original_rule():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, cfg in (('old', {'seed': 11}), ('new', {'seed': 11, 'neighbor_ties': 'all'})):
            (d / name).mkdir()
            (d / name / 'run_config.json').write_text(json.dumps({'config': cfg}))
        assert rule_of_run(d / 'old') == 'first' and rule_of_run(d / 'new') == 'all' and rule_of_run(d / 'missing') == 'unknown'
        for name, tj, want in (('t_old', {'feature_sets': {'rich': {}}}, 'first'), ('t_new', {'feature_sets': {'rich': {}}, 'neighbor_ties': 'all'}, 'all'),
                               ('t_comp', {'feature_sets': {'composition': {}}, 'neighbor_ties': None}, None)):
            (d / name).mkdir()
            (d / name / 'trees.json').write_text(json.dumps(tj))
            assert rule_of_trees(d / name / 'trees_predictions.npz') == want, name


def test_every_nightly_crystal_job_states_the_fixed_rule():
    from gpu.pipeline import nightly as N
    recipes = N.load_recipes()
    for t, rec in recipes['crystal'].items():
        for j in N.crystal_jobs(t, rec, device='cuda', smoke=False, small_ram=False, cpu_threads=2):
            assert j['argv'][j['argv'].index('--neighbor-ties') + 1] == 'all', j['id']
    assert N.CODE_VERSION['crystal'] != 'crystal-v1'      # the rule change retrains every crystal task once


def test_a_folder_planned_before_the_fix_keeps_its_rule_when_it_is_continued():
    """-Resume, -TieFix and a new cycle start a folder's unfinished jobs again; without the pin they would train under the new default."""
    jobs = _existing() + [{'id': 'ft_bulk_modulus_kv_s11', 'argv': ['-m', 'gpu.structure.train_structure', '--init-from', '{RUN_OF:x}']},
                          {'id': 'inv_x', 'argv': ['-m', 'gpu.structure.invariance', '--run', 'r']}, {'id': 'tab_alloys', 'argv': ['-m', 'gpu.train_tabular']}]
    plan = {'jobs': jobs, 'cycle': {'round': 'r2', 'targets': ['formation_energy_peratom']}, 'cycle_history': [{'targets': ['bulk_modulus_kv']}]}
    pinned = pl.pin_original_rule(plan)
    for j in plan['jobs']:
        if _crystal(j) and '--init-from' not in j['argv']:
            assert j['argv'][-2:] == ['--neighbor-ties', 'first'] and j['id'] in pinned, j['id']
    by = {j['id']: j for j in plan['jobs']}
    assert '--neighbor-ties' not in by['ft_bulk_modulus_kv_s11']['argv']         # a fine-tuned run takes the rule of the run it starts from
    assert '--neighbor-ties' not in by['inv_x']['argv'] and '--neighbor-ties' not in by['tab_alloys']['argv']
    assert plan['cycle']['neighbor_ties'] == 'first' and plan['cycle_history'][0]['neighbor_ties'] == 'first'
    assert plan['neighbor_rule_pins'][0]['rule'] == 'first'
    assert pl.pin_original_rule(plan) == [] and len(plan['neighbor_rule_pins']) == 1    # a second session changes nothing


def test_every_crystal_job_planned_now_states_its_rule_and_a_search_round_keeps_its_own():
    from gpu.pipeline import evolve as E
    plans = [pl.build_plan('standard', device='cuda'), pl.build_plan('smoke', device='cpu'), pl.build_plan('holdout', device='cuda'),
             pl.build_plan('cycle3', device='cuda', existing=pl.build_plan('standard', device='cuda')['jobs'] + pl.holdout_jobs(
                 device='cuda', targets=['formation_energy_peratom'], smoke=False, small_ram=False, cpu_threads=2, existing=None))]
    n = 0
    for plan in plans:
        for j in plan['jobs']:
            if _crystal(j):
                n += 1
                assert ('--neighbor-ties' in j['argv']) != ('--init-from' in j['argv']), j['id']
                assert pl.job_rule(None, j) in ('all', 'none') or '--init-from' in j['argv'], j['id']
    assert n > 20 and pl.pin_original_rule({'jobs': [j for p in plans for j in p['jobs']]}) == []   # nothing new is mistaken for an old line
    for ties in ('first', 'all'):
        sj = E.search_job('bulk_modulus_kv', 0, 0, dict(E.DEFAULT), None, device='cuda', epochs=2, limit=[], threads=0, smoke=True, ties=ties)
        fj = E.final_job('bulk_modulus_kv', 21, dict(E.DEFAULT), device='cuda', epochs=2, limit=[], threads=0, smoke=True, ties=ties)
        assert pl.job_rule(None, sj) == ties and pl.job_rule(None, fj) == ties


def _fake_run(pipe, jid, seed, ties=None):
    import numpy as np
    d = Path(pipe) / 'jobs' / jid / 'structure' / f'run_{jid}'
    d.mkdir(parents=True)
    cfg = {'seed': seed, **({'neighbor_ties': ties} if ties else {})}
    (d / 'run_config.json').write_text(json.dumps({'config': cfg}))
    (d / 'result.json').write_text('{}')
    np.savez(d / 'predictions.npz', test_jid=np.array(['a']))
    return d


def test_a_folder_whose_held_out_systems_runs_already_use_the_fixed_rule_has_nothing_to_compare():
    with tempfile.TemporaryDirectory() as d:
        old = _existing()
        for j in old:
            if j['id'].startswith('struct_bulk_modulus_kv_sys_s'):
                _fake_run(d, j['id'], int(j['id'].rsplit('_s', 1)[1]), ties='all')     # the cycle-2 jobs ran after the pull
        skipped = []
        jobs = pl.tiefix_jobs(device='cuda', smoke=False, cpu_threads=2, existing=old, pipe=d, skipped=skipped)
        assert not any(j['target'] == 'bulk_modulus_kv' for j in jobs) and any(j['target'] == 'formation_energy_peratom' for j in jobs)
        assert len(skipped) == 1 and 'bulk_modulus_kv' in skipped[0] and 'all' in skipped[0]
        fresh = pl.holdout_jobs(device='cuda', targets=['bulk_modulus_kv'], smoke=False, small_ram=False, cpu_threads=2, existing=None)
        skipped = []
        assert pl.tiefix_jobs(device='cuda', smoke=False, cpu_threads=2, existing=fresh, pipe=d, skipped=skipped) == [] and skipped   # planned after the fix


def test_the_comparison_refuses_runs_whose_recorded_rule_does_not_match_their_side():
    from gpu.pipeline.analyze import analyze_tiefix
    t = 'bulk_modulus_kv'
    with tempfile.TemporaryDirectory() as d:
        for sd in (11, 12):
            _fake_run(d, f'struct_{t}_sys_s{sd}', sd, ties='all')      # NOT original-rule runs, whatever their job name says
            _fake_run(d, f'struct_{t}_sysT_s{sd}', sd, ties='all')
        plan = {'jobs': [{'id': f'struct_{t}_sysT_s11', 'target': t}]}
        e = analyze_tiefix(d, plan)[0]
        assert e['status'] == 'error' and 'not made with the original rule' in e['reason'], e
        assert len(e['excluded_runs']) == 2 and all('(all)' in x for x in e['excluded_runs'])


def test_report_labels_invariance_rows_by_the_rule_the_run_recorded():
    from gpu.pipeline.report import tiefix_section
    inv = {'overall': {'share_changed_over_1pct_of_spread': 0.0, 'max_change': 1e-7, 'mean_change_as_share_of_test_mae': 0.0}, 'crystals': 40, 'unit': 'log10 GPa', 'seed': 11}
    items = [{'target': 'bulk_modulus_kv', 'status': 'error', 'reason': 'x', 'invariance': {'first': {**inv, 'neighbor_ties': 'all'}, 'all': {**inv, 'neighbor_ties': 'all'}}}]
    rows = [r for r in tiefix_section(items) if r.startswith('| ') and 'seed 11' in r]
    assert len(rows) == 2 and 'planned as the original rule' in rows[0] and rows[0].split('|')[2].strip().startswith('fixed rule'), rows
    assert 'planned as' not in rows[1]


def test_two_runs_building_the_same_graph_cache_at_once_both_finish():
    import threading
    from types import SimpleNamespace
    from gpu.structure import train_structure as ts
    atoms = {'lattice_mat': [[0, 2.82, 2.82], [2.82, 0, 2.82], [2.82, 2.82, 0]], 'coords': [[0, 0, 0], [.5, .5, .5]], 'elements': ['Na', 'Cl'], 'cartesian': False}
    recs = [{'jid': f'J{i}', 'atoms': atoms} for i in range(40)]
    a = SimpleNamespace(cutoff=8.0, max_neighbors=12, neighbor_ties='all')
    old = ts.CACHE
    with tempfile.TemporaryDirectory() as d:
        ts.CACHE = Path(d)
        try:
            errs, start = [], threading.Barrier(2)

            def go():
                try:
                    start.wait()
                    gs, _, _ = ts.graphs_for(recs, a, 'samekey')
                    assert len(gs) == 40
                except Exception as e:  # noqa: BLE001
                    errs.append(e)
            th = [threading.Thread(target=go) for _ in range(2)]
            [x.start() for x in th]
            [x.join() for x in th]
            assert not errs, errs
            assert (Path(d) / 'jarvis' / 'graphs' / 'samekey.pt').exists()
            assert not (Path(d) / 'jarvis' / 'graphs' / 'samekey.tmp').exists()   # no shared temporary name any more
        finally:
            ts.CACHE = old


def test_a_search_round_the_folder_already_has_keeps_its_rule_when_its_cycle_is_run_again():
    from gpu.run_pipeline import round_rule
    old = {'jobs': [], 'cycle': {'round': None, 'targets': ['bulk_modulus_kv']}}
    pl.pin_original_rule(old)                                                       # a round from before the fix
    assert round_rule(old, {'round': None, 'neighbor_ties': 'all'}) == 'first'       # -Cycle2 again: the same round continues under 'first'
    assert round_rule(old, {'round': 'r2', 'neighbor_ties': 'all'}) == 'all'         # -Cycle3: a new round uses the fixed rule
    later = {'jobs': [], 'cycle': {'round': 'r2', 'neighbor_ties': 'all'}, 'cycle_history': [{'round': None, 'neighbor_ties': 'first'}]}
    assert round_rule(later, {'round': None, 'neighbor_ties': 'all'}) == 'first' and round_rule(later, {'round': 'r2', 'neighbor_ties': 'all'}) == 'all'
    assert round_rule({'jobs': []}, {'round': None, 'neighbor_ties': 'all'}) == 'all'


def test_the_summary_keeps_every_held_out_property_and_names_each_rule():
    from gpu.pipeline.report import held_out_systems_summary
    item = lambda t, r: {'target': t, 'status': 'ok', 'neighbor_rules': {'networks': [r]}}  # noqa: E731
    orig = [item('bulk_modulus_kv', 'first'), item('shear_modulus_gv', 'first')]
    items, text, orig_first = held_out_systems_summary(orig, [item('bulk_modulus_kv', 'all')])   # shear's retrain did not finish
    assert [(t['target'], t['neighbor_rules']['networks']) for t in items] == [('bulk_modulus_kv', ['all']), ('shear_modulus_gv', ['first'])]
    assert 'fixed rule 1' in text and 'original rule 1' in text and orig_first
    items, text, orig_first = held_out_systems_summary([item('bulk_modulus_kv', 'all')], [])    # a folder made after the fix
    assert text == 'fixed rule' and not orig_first


def test_the_report_suggests_tiefix_only_where_it_can_run():
    from gpu.pipeline.report import tiefix_advice
    first = {'status': 'ok', 'neighbor_rules': {'networks': ['first']}}
    assert '-TieFix' in tiefix_advice({'structure_system': [first]}, False, False)
    assert '-TieFix -Rehearsal' in tiefix_advice({'structure_system': [first]}, True, False)
    assert '-TieFix' not in tiefix_advice({'structure': [first]}, False, False)                     # formula split only: tiefix has nothing to retrain
    assert '-TieFix' not in tiefix_advice({'structure_system': [{**first, 'neighbor_rules': {'networks': ['all']}}]}, False, False)
    assert 'measures it' in tiefix_advice({'structure_system': [first], 'tiefix': [{'status': 'ok'}]}, False, False)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
