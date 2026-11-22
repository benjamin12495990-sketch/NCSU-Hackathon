"""Writes gpu/structure/element_split.json: which elements are held out for the leave-out-elements split (--split element).

  python -m gpu.structure.make_element_split

Rule (fixed, so every property and every machine uses the same split):
  candidates  elements present in 0.05% to 3.5% of all JARVIS dft_3d crystals (O, Li, S, Ba, F and the like are never held out:
              each is in so many crystals that holding it out would gut the training set)
  order       candidates sorted by sha256('element-split-v1:' + symbol)
  test        take candidates in that order until the crystals containing ANY of them reach 10% of all crystals
  val         the next candidates, until crystals containing any of them (and no test element) reach a further 10%
A crystal containing a test element is in test; else one containing a validation element is in validation; else train.
So no test element appears in any training or validation crystal: the network meets those elements for the first time at test.
Counted over ALL crystals in the file, not per property, so the bulk-modulus and formation-energy splits hold out the same elements.
Refuses to overwrite.
"""
import collections
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from gpu.common import sha256_file
from gpu.structure.train_structure import DEFAULT_ZIP

OUT = Path(__file__).with_name('element_split.json')


def main():
    if OUT.exists():
        sys.exit(f'{OUT} exists; refusing to overwrite')
    with zipfile.ZipFile(DEFAULT_ZIP) as z:
        recs = json.loads(z.read([n for n in z.namelist() if n.endswith('.json')][0]))
    sets = [frozenset(r['atoms']['elements']) for r in recs]
    n = len(sets)
    freq = collections.Counter(e for s in sets for e in s)
    cand = sorted((e for e, c in freq.items() if 0.0005 <= c / n <= 0.035), key=lambda e: hashlib.sha256(f'element-split-v1:{e}'.encode()).hexdigest())
    test, val, it = [], [], iter(cand)
    covered = lambda els: sum(1 for s in sets if s & set(els))  # noqa: E731
    for e in it:
        test.append(e)
        if covered(test) >= 0.10 * n:
            break
    for e in it:
        val.append(e)
        if sum(1 for s in sets if (s & set(val)) and not (s & set(test))) >= 0.10 * n:
            break
    n_test = covered(test)
    n_val = sum(1 for s in sets if (s & set(val)) and not (s & set(test)))
    out = {'rule': __doc__.split('Rule (fixed')[1].split('Refuses')[0].strip(), 'test_elements': test, 'val_elements': val,
           'share_test': round(n_test / n, 4), 'share_val': round(n_val / n, 4), 'crystals': n,
           'element_frequency_percent': {e: round(freq[e] / n * 100, 2) for e in test + val}, 'source_zip_sha256': sha256_file(DEFAULT_ZIP)}
    OUT.write_text(json.dumps(out, indent=1))
    print(f'test elements {test} ({n_test / n:.1%} of crystals); validation elements {val} ({n_val / n:.1%}); wrote {OUT}')


if __name__ == '__main__':
    main()
