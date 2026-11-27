"""The site exporter (gpu/export_site.py): comparison wording and the fixed row sample.  python -m gpu.tests.test_export_site"""
from gpu.export_site import _ci, _r, _sample


def test_a_comparison_names_the_side_that_is_really_better():
    # the analysis words every comparison 'network'/'trees', whatever the two sides are; the export must name them
    c = {'diff': -0.0205, 'ci95': [-0.032, -0.0087], 'verdict': 'network clearly better'}
    out = _ci(c, 'blend', 'trees')
    assert out['better'] == 'blend' and out['verdict'] == 'blend clearly better'
    assert _ci({'diff': 0.1, 'ci95': [0.07, 0.13]}, 'network', 'trees')['verdict'] == 'trees clearly better'
    none = _ci({'diff': 0.002, 'ci95': [-0.02, 0.024]}, 'network', 'trees')
    assert none['better'] is None and none['verdict'] == 'no clear difference' and none['ci95'] == [-0.02, 0.024]
    assert _ci(None, 'a', 'b') is None


def test_the_row_sample_is_fixed_capped_and_not_a_block_of_ids():
    keys = [f'JVASP-{i}' for i in range(5000)]
    a, b = _sample(keys, 300), _sample(list(keys), 300)
    assert a == b and len(a) == 300 and a == sorted(a)
    assert max(a) - min(a) > 4000            # spread over the whole list, not the first 300 ids
    assert _sample(keys[:10], 300) == list(range(10))


def test_rounding_keeps_four_significant_figures():
    assert _r(0.123456) == 0.1235 and _r(12345.6) == 12350.0 and _r(None) is None and _r(0) == 0.0 and _r(float('nan')) is None


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
