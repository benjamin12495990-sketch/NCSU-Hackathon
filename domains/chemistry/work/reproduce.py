#!/usr/bin/env python3
"""Re-runs the AqSolDB pipeline end to end on CPU and writes a compact comparison report.

Usage (from the repository root):  python3 domains/chemistry/work/reproduce.py
All intermediate files (nn_data.json, all_rows.json, splits.json, predictions, the regenerated
result.json) go to domains/chemistry/work/runs/<run_id>/, which is gitignored; the committed
work/result.json is never touched. The compact report goes to work/reproductions/<date>_cpu.json
(a new name is used if that file already exists).
The main test split was already scored in the original report, so this is a reproduction,
not new evidence.
"""
import hashlib, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'data', 'data_curated.csv')
now = datetime.now(timezone.utc)
RUN = os.path.join(HERE, 'runs', now.strftime('%Y%m%dT%H%M%SZ'))
os.makedirs(os.path.join(RUN, 'scaffold'))
PY = [sys.executable, '-I']

def sha(p):
    h = hashlib.sha256(); h.update(open(p, 'rb').read()); return h.hexdigest()

steps, log = [], open(os.path.join(RUN, 'run.log'), 'w')
def run(name, cmd):
    t0 = time.time(); r = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    steps.append({'step': name, 'cmd': ' '.join(os.path.relpath(c, HERE) if os.path.isabs(c) and c.startswith(HERE) else c for c in cmd),
                  'seconds': round(time.time() - t0, 1), 'exit': r.returncode})
    print(steps[-1], flush=True)
    if r.returncode: sys.exit(f'{name} failed; see {RUN}/run.log')

R = lambda *p: os.path.join(RUN, *p)
t_all = time.time()
run('prepare', PY + ['prepare.py', CSV, RUN])
# determinism check: prepare.py again into a second folder, compare nn_data.json bytes
os.makedirs(R('prepare_again'))
run('prepare_again', PY + ['prepare.py', CSV, R('prepare_again')])
run('nn_cv', ['node', 'nn_run.js', 'cv', R('all_rows.json'), R('splits.json'), RUN])
run('nn_final', ['node', 'nn_run.js', 'final', R('all_rows.json'), R('splits.json'), RUN])
run('nn_splits', ['node', 'nn_run.js', 'splits', R('all_rows.json'), R('splits.json'), RUN])
run('trees', PY + ['trees.py', R('all_rows.json'), R('splits.json'), R('trees.json')])
run('trees_tuned', PY + ['trees_tuned.py', R('all_rows.json'), R('trees_tuned.json')])
run('scaffold_split', PY + ['scaffold_split.py', CSV, R('all_rows.json'), R('scaffold', 'scaffold_split.json')])
shutil.copy(R('cv_result.json'), R('scaffold', 'cv_result.json'))
run('scaffold_nn', ['node', 'nn_run.js', 'splits', R('all_rows.json'), R('scaffold', 'scaffold_split.json'), R('scaffold')])
run('scaffold_eval', PY + ['scaffold_eval.py', R('all_rows.json'), R('scaffold', 'scaffold_split.json'), R('scaffold', 'nn_splits.json'), R('scaffold', 'scaffold_result.json')])
run('make_notes', PY + ['make_notes.py', RUN, R('notes.json')])
run('evaluate', PY + ['evaluate.py', RUN, CSV, R('notes.json')])

new = json.load(open(R('result.json'))); old = json.load(open(os.path.join(HERE, 'result.json')))
tt = json.load(open(R('trees_tuned.json'))); sk = json.load(open(os.path.join(HERE, 'skeptic_check.json')))
pick = lambda r: {'test_network_ensemble_mae': r['b_test_once']['network_ensemble']['mae'],
                  'test_trees_fixed_mae': r['b_test_once']['trees_gbr']['mae'],
                  'test_naive_mean_mae': r['b_test_once']['naive_train_mean']['mae'],
                  'cv_network_ensemble_mae': r['a_cv_on_training_rows']['network_ensemble_cv']['mae'],
                  'cv_trees_fixed_mae': r['a_cv_on_training_rows']['trees_cv_same_folds']['mae'],
                  'chosen_network_settings': r['network']['chosen_settings'],
                  'eight_splits_mean_mae_network': r['c_eight_splits'].get('mean_mae_network'),
                  'eight_splits_mean_mae_trees': r['c_eight_splits'].get('mean_mae_trees'),
                  'scaffold_network_mae': (r['extra_scaffold_split_rdkit'] or {}).get('network_ensemble', {}).get('mae') if isinstance(r['extra_scaffold_split_rdkit'], dict) else None,
                  'scaffold_trees_mae': (r['extra_scaffold_split_rdkit'] or {}).get('trees_gbr', {}).get('mae') if isinstance(r['extra_scaffold_split_rdkit'], dict) else None,
                  'interval_90_test_coverage': r['d_interval_90']['test_coverage']}
o, n_ = pick(old), pick(new)
same = {k: o[k] == n_[k] for k in o}
nn_pred_old_hash = None
import numpy as np, sklearn
y = np.array(json.load(open(R('all_rows.json')))['y']); te = np.array(json.load(open(R('all_rows.json')))['main_test'])
p_nn = np.array(json.load(open(R('nn_final.json')))['pred']); d = np.abs(p_nn - y[te]) - np.abs(np.array(tt['test_pred']) - y[te])
report = {
    'run_id': os.path.basename(RUN), 'date_utc': now.isoformat(timespec='seconds'),
    'what': 'CPU reproduction of the AqSolDB LogS pipeline (prepare.py, nn_run.js cv/final/splits, trees.py, scaffold split, evaluate.py) plus the independent check\'s tuned-tree grid (trees_tuned.py)',
    'not_new_evidence': 'The main 2,000-row test split was scored in the original report; re-scoring it here only checks reproducibility.',
    'command': 'python3 domains/chemistry/work/reproduce.py',
    'data_sha256': {'data_curated.csv': sha(CSV)},
    'derived_sha256': {'nn_data.json': sha(R('nn_data.json')), 'nn_data.json_second_prepare': sha(R('prepare_again', 'nn_data.json')),
                       'all_rows.json': sha(R('all_rows.json')), 'splits.json': sha(R('splits.json'))},
    'prepare_is_deterministic': sha(R('nn_data.json')) == sha(R('prepare_again', 'nn_data.json')),
    'versions': {'python': platform.python_version(), 'node': subprocess.run(['node', '--version'], capture_output=True, text=True).stdout.strip(),
                 'numpy': np.__version__, 'sklearn': sklearn.__version__,
                 'rdkit': __import__('rdkit').__version__},
    'machine': {'cpus': os.cpu_count(), 'platform': platform.platform()},
    'folds': {'main_split': 'GroupShuffleSplit 80/20 by InChIKey first block, seed 42', 'cv': '3-fold GroupKFold on training rows',
              'extra_splits': '8 overlapping grouped 80/20 resamples (seeds 1001-1008)', 'scaffold': 'Bemis-Murcko scaffold split, largest sets to train'},
    'rows': {'train': new['data']['train_rows'], 'test': new['data']['test_rows'], 'groups': new['data']['groups_inchikey_first_block']},
    'original': o, 'reproduced': n_, 'identical_to_original': same,
    'tuned_trees': {'reproduced_grid': tt['grid_cv_3fold_grouped'], 'chosen': tt['chosen'], 'test_mae': tt['test_mae'],
                    'original_check_test_mae': sk.get('my_trees_tuned'), 'original_check_config': sk.get('tuned_trees_config'),
                    'network_minus_tuned_trees_test_mae': round(float(d.mean()), 4)},
    'selected_model_note': 'Network settings chosen by the original CV grid; tuned trees chosen by their own 3-fold grouped CV grid. Neither was selected on the test split.',
    'steps': steps, 'elapsed_seconds': round(time.time() - t_all, 1), 'run_dir_gitignored': os.path.relpath(RUN, HERE)}
os.makedirs(os.path.join(HERE, 'reproductions'), exist_ok=True)
out = os.path.join(HERE, 'reproductions', now.strftime('%Y%m%d') + '_cpu.json')
if os.path.exists(out): out = out[:-5] + '_' + report['run_id'] + '.json'
with open(out, 'x') as f: json.dump(report, f, indent=1)
print(json.dumps({k: report[k] for k in ('original', 'reproduced', 'identical_to_original', 'tuned_trees', 'prepare_is_deterministic')}, indent=1))
print('wrote', out)
