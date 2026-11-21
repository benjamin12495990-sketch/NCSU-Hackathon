#!/usr/bin/env python3
"""Writes public/model/materials/alloys.json, the input file of the alloy generator.

Usage: python3 -I export_rows.py <MPEA_dataset.csv> <nn_data.json> <result.json> <out alloys.json>

It reuses parse(), descriptors() and the ELEMENTS table from prepare.py (in this folder), so the
browser can rebuild every training row exactly (scripts/alloy_check.mjs compares all 530 rows
with nn_data.json at 1e-6). Rows come out in nn_data.json order: train rows, then test rows.
The CSV is an untrusted download: run this with python3 -I and nothing from it is executed.
"""
import csv, importlib.util, json, math, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('prepare', os.path.join(HERE, 'prepare.py'))
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def main(csv_path, nn_path, result_path, out_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    # Same row filter as prepare.main(), in the same order.
    kept = []
    for r in rows:
        v = (r[prep.HV] or '').strip()
        if not v:
            continue
        hv = float(v)
        if not math.isfinite(hv) or hv <= 0:
            continue
        frac = prep.parse(r[prep.FORMULA])
        proc = (r[prep.PROC] or '').strip().upper() or 'UNKNOWN'
        kept.append((frac, proc, hv, ' '.join(r[prep.FORMULA].split())))

    el_rows = Counter(el for frac, *_ in kept for el in frac)
    elements = sorted(el for el, c in el_rows.items() if c >= 5)
    procs = sorted({p for _, p, _, _ in kept})
    desc_names = list(prep.descriptors(kept[0][0]).keys())
    features = ['x_' + el for el in elements] + ['x_other'] + ['proc_' + p for p in procs] + desc_names

    with open(nn_path) as f:
        nn = json.load(f)
    assert nn['features'] == features, 'feature list differs from nn_data.json'

    # Put the rows in nn_data.json order by matching (group, hv) in turn, without re-splitting.
    pool = {}
    for i, (frac, proc, hv, raw) in enumerate(kept):
        pool.setdefault((prep.norm_formula(frac), hv), []).append(i)
    order = []
    for split in ('train', 'test'):
        for g, y in zip(nn[split]['g'], nn[split]['y']):
            cands = pool[(g, y)]
            order.append((cands.pop(0), split))
    assert len(order) == len(kept) and len({i for i, _ in order}) == len(kept)

    used = sorted({el for frac, *_ in kept for el in frac})
    train_hv = nn['train']['y']
    with open(result_path) as f:
        res = json.load(f)
    out = {
        'about': 'Alloy generator inputs, written by domains/materials/export_rows.py from the MPEA dataset '
                 '(Borg et al., Scientific Data 7, 430 (2020)) with the parse/descriptor code of prepare.py. '
                 'Rows are in nn_data.json order: train rows, then test rows.',
        'features': features,
        'elements': elements,
        'procs': procs,
        # Routes a user may generate for. UNKNOWN and OTHER are data labels (route not reported, or
        # a mix of routes), not ways to make an alloy, so they stay featurizer inputs only.
        'routes_selectable': [p for p in procs if p not in ('UNKNOWN', 'OTHER')],
        'descriptors': desc_names,
        # [metallic radius pm, Pauling electronegativity, valence electrons, melting point K]
        'table': {el: list(prep.ELEMENTS[el]) for el in used},
        'clamp_HV': [min(train_hv), max(train_hv)],
        'q90_HV': res['d_interval90']['halfwidth_HV'],
        'q90_rule': res['d_interval90']['rule'] + ', five-network ensemble, from result.json',
        'rows': [
            {'f': kept[i][3], 'frac': dict(kept[i][0]),
             'proc': kept[i][1], 'hv': kept[i][2], 'g': prep.norm_formula(kept[i][0]), 'split': split}
            for i, split in order
        ],
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print('wrote', out_path, len(out['rows']), 'rows,', os.path.getsize(out_path), 'bytes; clamp', out['clamp_HV'])


if __name__ == '__main__':
    main(*sys.argv[1:5])
