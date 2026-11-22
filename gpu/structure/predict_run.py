"""Writes predictions.npz for a finished structure run that did not save one (runs made before predictions were saved).

  python -m gpu.structure.predict_run --run gpu/runs/structure/<run folder> [--device cuda]

Rebuilds the records, split and graphs exactly as the run did, loads its best_model.pt, predicts the validation and
test rows, checks the test MAE against the run's result.json, and writes predictions.npz into the run folder.
Refuses if the data file's SHA-256 differs from the run's or if predictions.npz already exists.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpu.common import pick_device, sha256_bytes
from gpu.structure.train_structure import DEFAULT_ZIP, graphs_for, load_records, metrics, parts_of, run_model, save_predictions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--data', default=str(DEFAULT_ZIP))
    ap.add_argument('--device', default='auto')
    ap.add_argument('--batch', type=int, default=256)
    a = ap.parse_args(argv)
    import torch
    from gpu.structure.model import CGCNN

    run = Path(a.run)
    if (run / 'predictions.npz').exists():
        sys.exit(f'{run} already has predictions.npz')
    old = json.loads((run / 'run_config.json').read_text())
    cfg = dict(old['config'])
    cfg.setdefault('target', 'bulk_modulus_kv')  # runs made before --target existed
    cfg.setdefault('split', 'formula')           # runs made before --split existed
    args = SimpleNamespace(**cfg, data=a.data)
    recs, source, counts = load_records(args)
    if not cfg.get('synthetic') and source['zip_sha256'] != old['source']['zip_sha256']:
        sys.exit('the data file differs from the one this run used (SHA-256 mismatch)')
    parts = parts_of(recs, cfg['split'])
    idx = {p: np.where(parts == p)[0] for p in ('train', 'val', 'test')}
    y = np.array([r['y'] for r in recs])
    ytr = y[idx['train']]
    ym, ys = float(ytr.mean()), float(ytr.std() or 1.0)
    ident_src = {k: v for k, v in source.items() if k != 'zip'}
    gkey = sha256_bytes(json.dumps(ident_src, sort_keys=True), json.dumps([r['jid'] for r in recs]), args.cutoff, args.max_neighbors)[:24]
    graphs, _, _ = graphs_for(recs, args, gkey)
    device = pick_device(a.device)
    model = CGCNN(cfg['atom_features'], cfg['conv'], cfg['hidden'], cfg['cutoff'], atom_input=cfg.get('atom_input', 'embedding')).to(device)
    model.load_state_dict(torch.load(run / 'best_model.pt', map_location=device, weights_only=False))
    preds = {p: run_model(model, graphs, idx[p], device, a.batch, bool(cfg.get('amp'))) * ys + ym for p in ('val', 'test')}
    got = {p: metrics(preds[p], y[idx[p]]) for p in ('val', 'test')}
    want = json.loads((run / 'result.json').read_text())['cgcnn']
    print(f'test MAE now {got["test"]["mae"]:.5f}, in result.json {want["test"]["mae"]:.5f}; '
          f'val MAE now {got["val"]["mae"]:.5f}, in result.json {want["val"]["mae"]:.5f}')
    if abs(got['test']['mae'] - want['test']['mae']) > 2e-3:
        sys.exit('refusing to write predictions: they do not reproduce the run\'s recorded test MAE')
    save_predictions(run / 'predictions.npz', recs, idx, y, preds, cfg['split'])
    print(f'wrote {run / "predictions.npz"}')


if __name__ == '__main__':
    main()
