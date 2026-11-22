"""Checks the structure features against a crystal whose geometry is known.  python -m gpu.tests.test_features"""
import numpy as np

from gpu.structure import features as F


def rocksalt(a=5.64):
    return {'lattice_mat': [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]], 'coords': [[0, 0, 0], [0.5, 0.5, 0.5]],
            'elements': ['Na', 'Cl'], 'cartesian': False}


def test_rocksalt_geometry_and_density():
    f = dict(zip(F.NAMES, F.one(rocksalt())))
    assert abs(f['nn_dist_mean'] - 2.82) < 1e-3 and f['nn_dist_std'] < 1e-6, f['nn_dist_mean']      # Na-Cl nearest-neighbour distance a/2
    assert f['coord_mean'] == 6 and f['coord_min'] == 6 and f['coord_max'] == 6                      # octahedral: 6 within 1.2 x nearest
    assert abs(f['density'] - 2.165) < 0.02, f['density']                                            # NaCl density, g/cm3
    assert abs(f['mean_electronegativity'] - (0.93 + 3.16) / 2) < 1e-4 and abs(f['range_electronegativity'] - (3.16 - 0.93)) < 1e-4
    assert abs(f['en_diff_max'] - 2.23) < 1e-3
    assert abs(f['lat_ang_0'] - 60) < 1e-3                                                           # fcc primitive cell angles are 60 degrees


def test_features_do_not_depend_on_atom_order_or_cell_choice_for_composition():
    a = rocksalt()
    b = dict(a, elements=['Cl', 'Na'], coords=[[0.5, 0.5, 0.5], [0, 0, 0]])
    fa, fb = dict(zip(F.NAMES, F.one(a))), dict(zip(F.NAMES, F.one(b)))
    for k in F.NAMES:
        assert abs(fa[k] - fb[k]) < 1e-4, (k, fa[k], fb[k])


def test_build_is_cached_and_finite():
    import tempfile
    recs = [{'jid': f'x{i}', 'atoms': rocksalt(5.4 + 0.05 * i)} for i in range(5)]
    with tempfile.TemporaryDirectory() as d:
        X1 = F.build(recs, cache_dir=d, log=lambda m: None)
        X2 = F.build(recs, cache_dir=d, log=lambda m: None)
        assert X1.shape == (5, len(F.NAMES)) and np.array_equal(X1, X2) and np.isfinite(X1).all()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
