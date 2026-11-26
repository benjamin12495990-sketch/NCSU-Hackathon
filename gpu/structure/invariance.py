"""Measures, for one finished crystal-network run, how much its predictions change when nothing physical changes.

  python -m gpu.structure.invariance --run gpu/runs/structure/<run folder> --output invariance.json [--n 300] [--device cpu]

Takes --n test crystals of the run (ordered by a hash of the record id, so the choice is fixed and not a block of neighbouring ids), rewrites each one four ways (atoms relisted,
cell origin shifted, crystal rotated, cell doubled to 2x2x2 when it has at most --supercell-max-atoms atoms), rebuilds the graph with
the run's own neighbour rule, and predicts with the run's best weights. Reports the change against the unmodified crystal:
in the target's unit, as a share of the run's own test error, and the share of crystals that move by more than 1% of the spread of
predictions. A run made with the rule 'all' should show rounding noise only; a run made with 'first' shows the defect recorded in
docs/DECISIONS.md (LES-017). Reads the test crystals' structures and the run's weights; it does not train and does not look at labels
except to report the run's own test error for scale. Never overwrites --output: if it already holds the measurement of the same run with the
same settings (a session that ended before the pipeline recorded the job as done), it says so and exits 0; anything else there is refused. CPU is enough.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpu.common import now_iso, pick_device, write_json_new
from gpu.structure.graph import build_graph, collate
from gpu.structure.train_structure import DEFAULT_ZIP, TARGETS, load_records, parts_of


def variants(atoms, rng, supercell_max_atoms):
    L, X, els = np.array(atoms['lattice_mat'], float), np.array(atoms['coords'], float).reshape(-1, 3), list(atoms['elements'])
    if atoms.get('cartesian', False):
        X = X @ np.linalg.inv(L)
    n = len(els)
    p = rng.permutation(n)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    out = {'atoms relisted': {'lattice_mat': L.tolist(), 'coords': X[p].tolist(), 'elements': [els[i] for i in p], 'cartesian': False},
           'origin shifted': {'lattice_mat': L.tolist(), 'coords': ((X + rng.random(3)) % 1.0).tolist(), 'elements': els, 'cartesian': False},
           'crystal rotated': {'lattice_mat': (L @ q.T).tolist(), 'coords': X.tolist(), 'elements': els, 'cartesian': False}}
    if n <= supercell_max_atoms:
        sh = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], float)
        out['2x2x2 supercell'] = {'lattice_mat': (2 * L).tolist(), 'coords': np.concatenate([(X + s) / 2.0 for s in sh]).tolist(),
                                  'elements': els * 8, 'cartesian': False}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--data', default=str(DEFAULT_ZIP))
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--supercell-max-atoms', type=int, default=10)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    from gpu.structure.model import CGCNN

    run, out_path = Path(a.run), Path(a.output)
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
        except (OSError, ValueError):
            prev = {}
        same = (prev.get('run') == run.name and prev.get('settings') == {'n': a.n, 'supercell_max_atoms': a.supercell_max_atoms, 'seed': a.seed}
                and 'overall' in prev)
        if same:
            print(f'{out_path} already holds this measurement of {run.name}; nothing to do')
            return prev
        sys.exit(f'refusing to overwrite {out_path}')
    cfg = dict(json.loads((run / 'run_config.json').read_text())['config'])
    res = json.loads((run / 'result.json').read_text())
    ties, split, target = cfg.get('neighbor_ties', 'first'), cfg.get('split', 'formula'), cfg.get('target', 'bulk_modulus_kv')
    recs, _, _ = load_records(SimpleNamespace(synthetic=cfg.get('synthetic', 0), data=a.data, limit=cfg.get('limit', 0), seed=cfg.get('seed', 11), target=target))
    parts = parts_of(recs, split)
    y = np.array([r['y'] for r in recs])
    ytr = y[parts == 'train']
    ym, ys = float(ytr.mean()), float(ytr.std() or 1.0)
    # A hash order, not the text order of the ids: sorted as text, the first few hundred ids are one block (JVASP-10..., JVASP-11...) that had about
    # half the tie-sensitive share of the whole test set. The hash is fixed, so the same run always gets the same crystals.
    test = sorted((r for r, p in zip(recs, parts) if p == 'test'), key=lambda r: hashlib.sha256(str(r['jid']).encode()).hexdigest())[:a.n]
    device = pick_device(a.device)
    model = CGCNN(cfg['atom_features'], cfg['conv'], cfg['hidden'], cfg['cutoff'], atom_input=cfg.get('atom_input', 'embedding')).to(device)
    model.load_state_dict(torch.load(run / 'best_model.pt', map_location=device, weights_only=True))  # a plain dict of tensors
    model.eval()

    def predict(atoms_list):
        b = collate([build_graph(at, cfg['cutoff'], cfg['max_neighbors'], ties) for at in atoms_list])
        t = {k: (torch.as_tensor(v, device=device) if k != 'n_graphs' else v) for k, v in b.items()}
        with torch.no_grad():
            return model(t['Z'], t['src'], t['dst'], t['dist'].float(), t['batch'], t['n_graphs']).float().cpu().numpy() * ys + ym

    rng = np.random.default_rng(a.seed)
    base, changes = [], {}
    for r in test:
        vs = variants(r['atoms'], rng, a.supercell_max_atoms)
        p = predict([r['atoms']] + list(vs.values()))   # one batch: the unmodified crystal and its rewrites see the same BatchNorm statistics (eval mode)
        base.append(float(p[0]))
        for k, name in enumerate(vs, 1):
            changes.setdefault(name, []).append(abs(float(p[k]) - float(p[0])))
    spread = float(np.std(base)) or 1.0
    test_mae = res['cgcnn']['test']['mae']
    out = {'created': now_iso(), 'run': run.name, 'seed': cfg.get('seed'), 'target': target, 'unit': TARGETS[target]['unit'], 'split': split, 'neighbor_ties': ties,
           'settings': {'n': a.n, 'supercell_max_atoms': a.supercell_max_atoms, 'seed': a.seed}, 'crystal_order': 'sha256 of the record id',
           'crystals': len(test), 'spread_of_predictions': spread, 'run_test_mae': test_mae, 'variants': {}}
    for name, d in changes.items():
        d = np.array(d)
        out['variants'][name] = {'n': int(len(d)), 'max_change': float(d.max()), 'median_change': float(np.median(d)), 'mean_change': float(d.mean()),
                                 'mean_change_as_share_of_test_mae': float(d.mean() / test_mae) if test_mae else None,
                                 'share_changed_over_1pct_of_spread': float(np.mean(d > 0.01 * spread))}
    allc = np.concatenate([np.array(v) for v in changes.values()])
    out['overall'] = {'max_change': float(allc.max()), 'mean_change': float(allc.mean()), 'share_changed_over_1pct_of_spread': float(np.mean(allc > 0.01 * spread)),
                      'mean_change_as_share_of_test_mae': float(allc.mean() / test_mae) if test_mae else None}
    write_json_new(out_path, out)
    print(f'{run.name} (rule {ties}): {len(test)} test crystals; largest change {out["overall"]["max_change"]:.3g} {out["unit"]}, '
          f'{out["overall"]["share_changed_over_1pct_of_spread"]:.1%} of rewrites move a prediction by more than 1% of the spread; wrote {out_path}')
    return out


if __name__ == '__main__':
    main()
