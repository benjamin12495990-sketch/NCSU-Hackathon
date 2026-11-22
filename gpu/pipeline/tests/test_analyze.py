"""Tests of the fairness arithmetic in gpu/pipeline/analyze.py (synthetic numbers; no PyTorch).  python -m gpu.pipeline.tests.test_analyze"""
import numpy as np

from gpu.pipeline.analyze import crossfit, crossfit_topk, fairness, verdict


def test_crossfit_never_uses_the_scored_fold():
    rng = np.random.default_rng(0)
    y = rng.normal(size=60)
    folds = np.repeat(np.arange(3), 20)
    P = np.vstack([y + rng.normal(0, 1.0, 60) for _ in range(40)])      # 40 equally bad candidates
    cf, chosen = crossfit(P, y, folds)
    best_on_same = np.abs(P - y).mean(axis=1).min()
    assert np.abs(cf - y).mean() > best_on_same, 'cross-fitting must not beat the optimistic best-of-40'
    top = crossfit_topk(P, y, folds, 3)
    assert top.shape == y.shape


def test_averaging_independent_networks_beats_one_network_and_fairness_says_so():
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.normal(size=n)
    groups = np.arange(n) % 800
    singles = [y + rng.normal(0, 1.0, n) for _ in range(5)]              # five equally good independent networks
    trees = y + rng.normal(0, 0.8, n)                                     # one slightly better tree model
    f = fairness(y, groups, np.mean(singles, axis=0), singles, trees)
    avg_vs_trees = f['blend']['vs_network_average']                       # keys exist
    assert f['single']['n'] == 5 and f['single']['n_clearly_worse'] == 5, f['single']        # each single network loses to the trees
    assert f['blend']['mae'] < f['single']['mean_mae']
    from gpu.pipeline.analyze import _cmp
    assert verdict(*_cmp(np.mean(singles, axis=0), trees, y, groups)['ci95']) == 'network clearly better'  # but their average wins
    assert avg_vs_trees['verdict'] in ('network clearly better', 'no clear difference', 'trees clearly better')


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
