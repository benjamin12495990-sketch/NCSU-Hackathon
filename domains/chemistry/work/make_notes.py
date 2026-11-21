# Adds the optional extras, caveats and file list. Usage: python3 -I make_notes.py <workdir> <notes.json>
import json, os, sys
W = sys.argv[1]; n = {}
p = os.path.join(W, 'scaffold', 'scaffold_result.json')
n['extra_scaffold_split_rdkit'] = json.load(open(p)) if os.path.exists(p) else 'not run'
p = os.path.join(W, 'cv_extra.json')
if os.path.exists(p):
    n['extra_cv_outside_grid_not_used'] = {'what': 'single-seed 3-fold grouped CV on training rows, run after the final model was fixed; compare with 0.7918 for the chosen setting',
                                           'rows': [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in json.load(open(p))]}
n['sanity_outcome'] = ('MolLogP: trend holds. Raising MolLogP from the 5th to the 95th training percentile with the other 16 inputs fixed lowers predicted LogS for 99.0% of the 2,000 test molecules '
    '(monotone at all 21 grid points for 91.1%; 96.7% fall locally around their own value); median slope about -0.52 LogS per logP unit. '
    'MolWt: weaker and mixed. Predicted LogS falls over the range 96 to 555 g/mol for 78.2% of molecules (median -1.18 LogS), is monotone for only 46.2%, and does not fall for the remaining 22%. '
    'Both sweeps are off-manifold: MolWt cannot really change while MolMR, HeavyAtomCount and LabuteASA stay fixed.')
n['caveats'] = [
    '17 bulk 2D descriptors cannot see 3D structure, crystal packing or melting point, polymorph, ionisation or pH, or temperature. Typical error is about 0.76 log units (a factor of about 6 in mol/L) and the 90% band is +-1.71 log units (a factor of about 50): a rough screen, not a replacement for measurement, and not able to rank close analogues.',
    'Labels are noisy and the split is only moderately strict. 78% of rows are single measurements (G1) of unknown quality; on rows measured at least twice with SD <= 0.5 (G3+G5, 371 test rows) MAE drops from 0.76 to 0.56, but those rows are also better-studied chemistry, so the gap is not a clean estimate of measurement noise. Grouping by the InChIKey first block keeps stereoisomers together, but different salts of one parent get different first blocks and close analogues sit on both sides; the scaffold split is the tougher figure.',
    'The comparison favours the network: it was tuned (8 settings by CV, 5-seed ensemble) while the tree model is one fixed untuned configuration, so the 0.07 MAE margin says this network beats this baseline, not that networks beat trees here. The chosen size is the largest in the grid. The eight extra splits are overlapping resamples of the same rows (the t interval is too narrow as a claim about new data) and their rows include main-test rows, with settings carried over rather than re-tuned. The 90% band has constant width and covered 89.3% of test rows, slightly under nominal.']
n['provenance'] = ('Every number here was computed in this session by the scripts listed under files, on this machine; none is quoted from the literature. '
    'The repository README does not define the G1-G5 labels; their meaning was checked against the SD and Occurrences columns of the CSV (table under e_reliable_rows_G3_G5). '
    'Test rows of the main split were predicted once, with settings fixed beforehand by the automated CV step. RDKit installed with pip (one attempt) and used only for the scaffold split.')
n['files'] = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(W) for f in fs if f not in ('start.txt',)) + [os.path.join(W, 'result.json')]
n['files'] = sorted(set(n['files']))
s = n['extra_scaffold_split_rdkit']
if isinstance(s, dict):
    a, b = s['network_ensemble'], s['trees_gbr']
    n['caveats'][1] += (' On the scaffold split (test ring systems never seen in training) MAE rises to %.2f for the network and %.2f for the trees, R2 about %.2f: '
                        'expect that level, not 0.76, on new chemical families.' % (a['mae'], b['mae'], a['r2']))
    n['caveats'][2] += (' On the scaffold split the two are tied (MAE %.3f network, %.3f trees), so the network lead is confined to molecules that resemble the training set.' % (a['mae'], b['mae']))
json.dump(n, open(sys.argv[2], 'w'), indent=1)
