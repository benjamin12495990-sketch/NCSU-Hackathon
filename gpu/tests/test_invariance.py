"""A crystal's graph, and the network's prediction, must not change when nothing physical changes.
python -m gpu.tests.test_invariance   (CPU, a few seconds; no data file needed)

Checked for the neighbour rule 'all' (gpu/structure/graph.py): relisting the atoms, shifting the cell origin, rotating the crystal,
reordering the lattice vectors, and doubling the cell to a 2x2x2 supercell. The original rule 'first' is shown to FAIL on a
perovskite, where 12 neighbours of two different elements sit at exactly the same distance from each oxygen atom."""
import hashlib
import json
from pathlib import Path

import numpy as np

from gpu.structure.graph import Z_OF, build_graph, collate

A = 3.905


def perovskite():   # SrTiO3, cubic: each O has 2 Ti at a/2, then 4 Sr + 8 O all at a/sqrt(2): the 12-cut lands inside that shell
    return {'lattice_mat': (np.eye(3) * A).tolist(), 'coords': [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
            'elements': ['Sr', 'Ti', 'O', 'O', 'O'], 'cartesian': False}


def rocksalt(a=5.64):
    return {'lattice_mat': [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]], 'coords': [[0, 0, 0], [.5, .5, .5]], 'elements': ['Na', 'Cl'], 'cartesian': False}


def wurtzite(a=3.25, c=5.2, u=0.382):   # ZnO: hexagonal, lower symmetry
    L = [[a, 0, 0], [-a / 2, a * 3 ** 0.5 / 2, 0], [0, 0, c]]
    return {'lattice_mat': L, 'coords': [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, .5], [1 / 3, 2 / 3, u], [2 / 3, 1 / 3, .5 + u]], 'elements': ['Zn', 'Zn', 'O', 'O'], 'cartesian': False}


def generic(seed=3, n=6):   # no symmetry: no ties, so both rules must give the same graph
    r = np.random.default_rng(seed)
    L = np.diag([5.1, 6.3, 7.4]) + r.normal(0, 0.4, (3, 3))
    return {'lattice_mat': L.tolist(), 'coords': r.random((n, 3)).tolist(), 'elements': list(r.choice(['Fe', 'O', 'Si', 'Li'], n)), 'cartesian': False}


STRUCTS = {'perovskite': perovskite, 'rocksalt': rocksalt, 'wurtzite': wurtzite, 'generic': generic}


def variants(a, seed=0):
    """name -> (atoms, perm) where perm[k] is the original atom index of the variant's atom k (None for the supercell)."""
    r = np.random.default_rng(seed)
    L, X, els = np.array(a['lattice_mat'], float), np.array(a['coords'], float), list(a['elements'])
    n = len(els)
    p = r.permutation(n)
    q, _ = np.linalg.qr(r.normal(size=(3, 3)))
    sh = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], float)
    return {
        'atoms relisted': ({'lattice_mat': L.tolist(), 'coords': X[p].tolist(), 'elements': [els[i] for i in p], 'cartesian': False}, p),
        'origin shifted': ({'lattice_mat': L.tolist(), 'coords': ((X + r.random(3)) % 1.0).tolist(), 'elements': els, 'cartesian': False}, np.arange(n)),
        'crystal rotated': ({'lattice_mat': (L @ q.T).tolist(), 'coords': X.tolist(), 'elements': els, 'cartesian': False}, np.arange(n)),
        'lattice vectors reordered': ({'lattice_mat': L[[1, 2, 0]].tolist(), 'coords': X[:, [1, 2, 0]].tolist(), 'elements': els, 'cartesian': False}, np.arange(n)),
        '2x2x2 supercell': ({'lattice_mat': (2 * L).tolist(), 'coords': np.concatenate([(X + s) / 2.0 for s in sh]).tolist(), 'elements': els * 8, 'cartesian': False}, None),
    }


def signatures(atoms, ties):
    """Per atom: (its element, the sorted list of (neighbour element, distance to 1e-4 A)). This is everything the network sees."""
    g = build_graph(atoms, 8.0, 12, ties)
    out = []
    for i in range(len(g['Z'])):
        m = g['src'] == i
        out.append((int(g['Z'][i]), tuple(sorted((int(g['Z'][j]), round(float(d), 4)) for j, d in zip(g['dst'][m], g['dist'][m])))))
    return out


def same_graph(a, b_atoms, perm, ties):
    sa, sb = signatures(a, ties), signatures(b_atoms, ties)
    if perm is None:                       # supercell: every original atom appears 8 times with the same surroundings
        return sorted(sa * 8) == sorted(sb)
    return [sa[i] for i in perm] == sb


def test_rule_all_gives_the_same_graph_however_the_crystal_is_written():
    for name, make in STRUCTS.items():
        a = make()
        for vname, (b, perm) in variants(a).items():
            assert same_graph(a, b, perm, 'all'), f'{name}: graph changed when {vname}'


def test_rule_first_is_order_dependent_on_a_perovskite():
    """The defect this file exists for (docs/DECISIONS.md LES-017). If this ever passes for 'first', the test has lost its teeth."""
    a = perovskite()
    changed = [v for seed in range(6) for v, (b, perm) in variants(a, seed).items() if not same_graph(a, b, perm, 'first')]
    assert changed, "the original rule unexpectedly gave identical graphs; the perovskite no longer exposes the tie"


def test_rules_agree_when_there_are_no_ties():
    a = generic()
    assert signatures(a, 'all') == signatures(a, 'first')


def test_rule_all_keeps_whole_shells():
    g = build_graph(perovskite(), 8.0, 12, 'all')
    per_atom = np.bincount(g['src']).tolist()
    assert per_atom == [12, 14, 14, 14, 14], per_atom       # Sr: 12 O; Ti: 6 O + 8 Sr; each O: 2 Ti + (4 Sr + 8 O)
    o = g['src'] == 2
    kinds = sorted(int(g['Z'][j]) for j in g['dst'][o])
    assert kinds.count(Z_OF['Ti']) == 2 and kinds.count(Z_OF['Sr']) == 4 and kinds.count(Z_OF['O']) == 8, kinds


def test_network_prediction_is_invariant_under_rule_all_and_not_under_rule_first():
    import torch
    from gpu.structure.model import CGCNN

    def predict(model, atoms, ties):
        b = collate([build_graph(atoms, 8.0, 12, ties)])
        with torch.no_grad():
            return float(model(torch.as_tensor(b['Z']), torch.as_tensor(b['src']), torch.as_tensor(b['dst']), torch.as_tensor(b['dist']).float(),
                               torch.as_tensor(b['batch']), 1))
    worst_all, worst_first, scale = 0.0, 0.0, []
    for seed in range(3):
        torch.manual_seed(seed)
        for atom_input in ('embedding', 'both'):
            m = CGCNN(64, 3, 128, 8.0, atom_input=atom_input).eval()
            for name, make in STRUCTS.items():
                a = make()
                p0 = {t: predict(m, a, t) for t in ('all', 'first')}
                scale.append(abs(p0['all']))
                for vname, (b, _) in variants(a, seed).items():
                    worst_all = max(worst_all, abs(predict(m, b, 'all') - p0['all']))
                    worst_first = max(worst_first, abs(predict(m, b, 'first') - p0['first']))
    typical = float(np.mean(scale)) or 1.0
    assert worst_all < 1e-4 * max(1.0, typical), (worst_all, typical)      # float32 noise only
    assert worst_first > 100 * worst_all, (worst_first, worst_all)          # the original rule changes predictions visibly
    print(f'    worst change of a prediction: rule all {worst_all:.2g}, rule first {worst_first:.2g}')


def test_cache_keys_differ_between_rules_and_the_original_key_is_unchanged():
    import json
    from gpu.common import sha256_bytes
    from gpu.structure.train_structure import graph_key
    recs, src = [{'jid': 'JVASP-1'}, {'jid': 'JVASP-2'}], {'zip_sha256': 'x'}
    old = sha256_bytes(json.dumps(src, sort_keys=True), json.dumps(['JVASP-1', 'JVASP-2']), 8.0, 12)[:24]   # the key every earlier run used
    assert graph_key(src, recs, 8.0, 12, 'first') == old
    assert graph_key(src, recs, 8.0, 12, 'all') != old


def test_feature_cache_is_separate_for_each_rule():
    import tempfile
    from gpu.structure import features as F
    recs = [{'jid': f'X-{i}', 'atoms': make()} for i, make in enumerate(STRUCTS.values())]
    with tempfile.TemporaryDirectory() as d:
        a = F.build(recs, cache_dir=d, log=lambda *_: None, ties='all')
        f = F.build(recs, cache_dir=d, log=lambda *_: None, ties='first')
        assert len(list(Path(d).glob('*.npy'))) == 2            # one file per rule, never shared
        assert not np.allclose(a, f)                            # the perovskite's neighbour statistics differ between the rules
        old_key = hashlib.sha256(json.dumps([F.VERSION, [r['jid'] for r in recs]]).encode()).hexdigest()[:24]
        assert (Path(d) / f'{old_key}.npy').exists()            # the original rule keeps the key every earlier tree run used


def test_new_runs_use_the_fixed_rule_and_resumed_or_fine_tuned_runs_keep_the_rule_they_started_with():
    import contextlib
    import io
    import tempfile
    from gpu.structure import invariance, train_structure as ts

    def run(*args):
        with contextlib.redirect_stdout(io.StringIO()):
            return ts.main(list(args))

    def refused(*args):
        try:
            run(*args)
        except SystemExit as e:
            return isinstance(e.code, str) and 'refusing' in e.code
        return False

    cfg_of = lambda r: json.loads((r / 'run_config.json').read_text())['config']  # noqa: E731
    with tempfile.TemporaryDirectory() as d:
        base = ['--device', 'cpu', '--synthetic', '150', '--epochs', '2', '--runs-root', d]
        res, new = run(*base, '--tag', 'new')
        assert cfg_of(new)['neighbor_ties'] == 'all' and res['graphs']['neighbor_ties'] == 'all'       # no flag: the fixed rule
        res_o, old = run(*base, '--tag', 'old', '--neighbor-ties', 'first')
        assert 'neighbor_ties' not in cfg_of(old) and res_o['graphs']['neighbor_ties'] == 'first'     # the original rule keeps the original identity
        # an interrupted original-rule run resumes under its own rule when the flag is not given, and refuses the other rule
        part = ['--device', 'cpu', '--synthetic', '150', '--epochs', '3', '--runs-root', d, '--seed', '5']
        try:
            run(*part, '--tag', 'cut', '--neighbor-ties', 'first', '--stop-after-epochs', '1')
        except SystemExit:
            pass
        cut = next(Path(d, 'structure').glob('*cut'))
        assert not (cut / 'result.json').exists()
        assert refused(*part, '--tag', 'cut', '--resume', str(cut), '--neighbor-ties', 'all')
        res_c, _ = run(*part, '--tag', 'cut', '--resume', str(cut))
        assert res_c['graphs']['neighbor_ties'] == 'first'
        # fine-tuning takes the rule of the network it starts from, and refuses a different one
        res_f, ft = run(*base, '--tag', 'ft', '--init-from', str(old))
        assert 'neighbor_ties' not in cfg_of(ft) and res_f['graphs']['neighbor_ties'] == 'first'
        assert refused(*base, '--tag', 'bad', '--init-from', str(old), '--neighbor-ties', 'all')
        _, ft2 = run(*base, '--tag', 'ft2', '--init-from', str(new))
        assert cfg_of(ft2)['neighbor_ties'] == 'all'
        # the invariance measurement rebuilds graphs with the run's own rule
        with contextlib.redirect_stdout(io.StringIO()):
            inv = invariance.main(['--run', str(new), '--output', str(Path(d) / 'inv_new.json'), '--n', '12'])
            inv_o = invariance.main(['--run', str(old), '--output', str(Path(d) / 'inv_old.json'), '--n', '12'])
        assert inv['neighbor_ties'] == 'all' and inv_o['neighbor_ties'] == 'first'
        assert inv['overall']['max_change'] < 1e-3 * max(1.0, inv['spread_of_predictions']), inv['overall']
        assert set(inv['variants']) >= {'atoms relisted', 'origin shifted', 'crystal rotated'}


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            r = fn()
            print('ok', name, '' if r is None else r)
