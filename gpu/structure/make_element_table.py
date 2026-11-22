"""Writes gpu/structure/element_table.json: per-element properties for the structure-feature tree baseline.

  pip install mendeleev            # only needed for this script, not for training
  python -m gpu.structure.make_element_table

Columns: Z, period, group (lanthanides and actinides count as group 3), Pauling electronegativity, atomic weight (u),
covalent radius (pm, Cordero et al. 2008) and valence electrons (mendeleev's count; the group rule where it has none).
Values the library does not have (noble-gas electronegativities, some heavy-element radii) are filled with the mean of
the known values and listed under "filled". The file is committed, so training machines never need the library.
Refuses to overwrite.
"""
import json
import math
import sys
from pathlib import Path

from gpu.structure.graph import SYMBOLS

OUT = Path(__file__).with_name('element_table.json')
PROPS = ['Z', 'period', 'group', 'electronegativity', 'mass', 'covalent_radius', 'valence_electrons']


def main():
    if OUT.exists():
        sys.exit(f'{OUT} exists; refusing to overwrite')
    import mendeleev
    from mendeleev import element
    rows, filled = {}, {}
    for z in range(1, 104):
        e = element(z)
        sym = SYMBOLS[z - 1]
        assert e.symbol == sym, (e.symbol, sym)
        grp = e.group_id if e.group_id else 3
        try:
            nval = float(e.nvalence())
        except Exception:
            nval = float(grp if grp <= 12 else grp - 10)
        rows[sym] = {'Z': z, 'period': e.period, 'group': grp, 'electronegativity': e.electronegativity_pauling(), 'mass': e.atomic_weight,
                     'covalent_radius': e.covalent_radius_cordero, 'valence_electrons': nval}
    for p in ('electronegativity', 'mass', 'covalent_radius'):
        known = [r[p] for r in rows.values() if r[p] is not None and math.isfinite(r[p])]
        mean = sum(known) / len(known)
        for s, r in rows.items():
            if r[p] is None or not math.isfinite(r[p]):
                r[p] = round(mean, 4)
                filled.setdefault(s, []).append(p)
    # sanity: values I know
    assert rows['F']['electronegativity'] == 3.98 and rows['Cs']['electronegativity'] == 0.79 and rows['O']['electronegativity'] == 3.44
    assert abs(rows['Fe']['mass'] - 55.845) < 1e-3 and abs(rows['Si']['mass'] - 28.085) < 1e-2
    assert rows['Li']['covalent_radius'] < rows['Na']['covalent_radius'] < rows['K']['covalent_radius'] < rows['Rb']['covalent_radius'] < rows['Cs']['covalent_radius']
    assert rows['Na']['group'] == 1 and rows['Cl']['group'] == 17 and rows['Fe']['group'] == 8 and rows['La']['group'] == 3
    OUT.write_text(json.dumps({'source': f'mendeleev {mendeleev.__version__}: atomic_weight, electronegativity_pauling(), covalent_radius_cordero, group_id, period, nvalence()',
                               'props': PROPS, 'filled': filled, 'elements': rows}, indent=1))
    print(f'wrote {OUT}: {len(rows)} elements, {len(filled)} with filled values')


if __name__ == '__main__':
    main()
