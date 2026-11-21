# Optional extra: Bemis-Murcko scaffold split (needs RDKit). Usage: python3 -I scaffold_split.py <csv> <all_rows.json> <out.json>
# Scaffold of the largest fragment; scaffold sets sorted by size, largest go to train until 80% (the usual deterministic scaffold split).
import csv, json, sys
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
a = json.load(open(sys.argv[2]))
smi = {}
with open(sys.argv[1], newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f): smi[r['ID']] = r['SMILES']
sets = defaultdict(list); failed = 0
for i, rid in enumerate(a['id']):
    m = Chem.MolFromSmiles(smi[rid])
    if m is None: failed += 1; sets['FAILED_' + rid].append(i); continue
    fr = max(Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumHeavyAtoms())
    try: sc = MurckoScaffold.MurckoScaffoldSmiles(mol=fr)
    except Exception: failed += 1; sc = 'FAILED_' + rid
    sets[sc].append(i)
order = sorted(sets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
n = len(a['id']); train, test = [], []
for sc, idx in order:
    (train if len(train) + len(idx) <= 0.8 * n else test).extend(idx)
g = a['g']; straddle = len(set(g[i] for i in train) & set(g[i] for i in test))
json.dump({'note': 'Murcko scaffold split', 'scaffolds': len(sets), 'acyclic_rows_all_in_train': len(sets.get('', [])), 'smiles_failed': failed,
           'inchikey_groups_on_both_sides': straddle, 'splits': [{'seed': 'scaffold', 'train': sorted(train), 'test': sorted(test)}]}, open(sys.argv[3], 'w'))
print('scaffolds', len(sets), 'acyclic', len(sets.get('', [])), 'failed', failed, 'train', len(train), 'test', len(test), 'straddling inchikey groups', straddle)
