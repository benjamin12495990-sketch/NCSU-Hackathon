"""Tests of the generation search's bookkeeping (no training).  python -m gpu.pipeline.tests.test_evolve"""
import json
import tempfile
from pathlib import Path

import numpy as np

from gpu.pipeline import evolve as E


def test_generation_zero_is_the_defaults_plus_distinct_random_settings_and_is_deterministic():
    a = E.propose([], 'bulk_modulus_kv', 0, 6, 4)
    b = E.propose([], 'bulk_modulus_kv', 0, 6, 4)
    assert [E.gkey(g) for g, _ in a] == [E.gkey(g) for g, _ in b]
    assert E.gkey(a[0][0]) == E.gkey(E.DEFAULT) and len(a) == 6 and len({E.gkey(g) for g, _ in a}) == 6
    other = E.propose([], 'shear_modulus_gv', 0, 6, 4)
    assert [E.gkey(g) for g, _ in a][1:] != [E.gkey(g) for g, _ in other][1:], 'targets should not share one random population'


def test_children_stay_in_the_search_space_differ_from_parents_and_never_repeat():
    rng = np.random.default_rng(3)
    g = dict(E.DEFAULT)
    for _ in range(300):
        c = E.mutate(g, rng)
        assert all(c[k] in E.SPACE[k] for k in E.SPACE) and E.gkey(c) != E.gkey(g)
        assert sum(c[k] != g[k] for k in E.SPACE) <= 2
    pop = E.propose([], 'bulk_modulus_kv', 0, 6, 4)
    arch = [{'id': f'i{k}', 'gen': 0, 'parent': None, 'genes': gg, 'val_mae': 0.2 - 0.01 * k} for k, (gg, _) in enumerate(pop)]
    kids = E.propose(arch, 'bulk_modulus_kv', 1, 6, 4)
    keys = [E.gkey(gg) for gg, _ in kids]
    assert len(kids) == 4 and len(set(keys)) == 4 and not set(keys) & {E.gkey(x['genes']) for x in arch}
    assert {par for _, par in kids} <= {'i5', 'i4'}, 'parents must be the two best (lowest validation MAE)'
    again = [E.gkey(gg) for gg, _ in E.propose(arch, 'bulk_modulus_kv', 1, 6, 4)]
    assert again == keys, 'the same archive must give the same children (resume safety)'


def test_archive_reads_only_validation_error_and_ignores_unfinished_runs():
    with tempfile.TemporaryDirectory() as d:
        job = E.search_job('bulk_modulus_kv', 0, 0, dict(E.DEFAULT), None, device='cpu', epochs=2, limit=[], threads=0, smoke=True)
        run = Path(d) / 'jobs' / job['id'] / 'structure' / '20260101T000000Z_x'
        run.mkdir(parents=True)
        arch = E.archive(d, [job], 'bulk_modulus_kv')
        assert arch[0]['val_mae'] is None                                  # nothing finished yet
        (run / 'result.json').write_text(json.dumps({'cgcnn': {'val': {'mae': 0.123}, 'test': {'mae': 9.99}}}))
        assert E.archive(d, [job], 'bulk_modulus_kv')[0]['val_mae'] is None  # predictions.npz missing: still unfinished
        (run / 'predictions.npz').write_bytes(b'x')
        a = E.archive(d, [job], 'bulk_modulus_kv')[0]
        assert a['val_mae'] == 0.123 and 'test' not in json.dumps(a)       # the test number never enters the archive


def test_jobs_use_the_system_split_and_the_search_seed_and_carry_their_settings():
    g = dict(E.DEFAULT, hidden=256, lr=1e-3)
    j = E.search_job('optb88vdw_bandgap', 2, 1, g, 'evo_x', device='cuda', epochs=40, limit=['--limit', '40000'], threads=3, smoke=False)
    s = ' '.join(j['argv'])
    assert '--split system' in s and f'--seed {E.SEARCH_SEED}' in s and '--hidden 256' in s and '--lr 0.001' in s and '--limit 40000' in s and '--threads 3' in s
    f = E.final_job('optb88vdw_bandgap', 21, g, device='cuda', epochs=100, limit=[], threads=0, smoke=False)
    assert f['id'] == 'evof_optb88vdw_bandgap_s21' and '--seed 21' in ' '.join(f['argv']) and 21 not in (E.SEARCH_SEED,)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
