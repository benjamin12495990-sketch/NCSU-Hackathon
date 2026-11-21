# Integrity checks, then writes result.json. Usage: python3 -I finalize.py <work dir>
import sys, json, numpy as np
W = sys.argv[1]
r = json.load(open(f'{W}/result_core.json')); D = json.load(open(f'{W}/tasks.json')); rows = json.load(open(f'{W}/rows.json'))
nn = json.load(open(f'{W}/nn_data.json')); sp = json.load(open(f'{W}/splits.json'))
chk = {}
chk['no_group_on_both_sides_any_task'] = all(not (set(t['train']['g']) & set(t['test']['g'])) for t in D['tasks'].values())
chk['cv_folds_use_only_main_training_rows'] = all(set(D['tasks'][c]['train']['idx']) | set(D['tasks'][c]['test']['idx']) <= set(D['tasks']['main']['train']['idx']) for c in ('cv0', 'cv1', 'cv2'))
chk['nn_data_shape_ok'] = (list(nn.keys()) == ['features', 'train', 'test'] and all(len(x) == len(nn['features']) for x in nn['train']['X'] + nn['test']['X'])
                           and len(nn['train']['X']) == len(nn['train']['y']) == len(nn['train']['g']))
chk['splits_json_keys'] = list(sp.keys())
def unseen_share(t):
    tm = set(); [tm.update((rows[i]['a'], rows[i]['b'])) for i in t['train']['idx']]
    return round(float(np.mean([(rows[i]['a'] not in tm) or (rows[i]['b'] not in tm) for i in t['test']['idx']])), 3)
chk['main_test_rows_with_a_monomer_unseen_in_training_share'] = unseen_share(D['tasks']['main'])
chk['hard_test_rows_with_a_monomer_unseen_in_training_share'] = unseen_share(D['tasks']['hard'])
f = D['features']; iav = f.index('fox_available')
chk['main_train_rows_fox_available_share'] = round(float(np.mean([x[iav] for x in D['tasks']['main']['train']['X']])), 3)
chk['main_test_rows_fox_available_share'] = round(float(np.mean([x[iav] for x in D['tasks']['main']['test']['X']])), 3)
r['integrity_checks'] = chk
r['verdict_network_vs_trees'] = ('On the seed-42 test split the network ensemble has lower MAE than the trees (15.33 vs 17.54 K). Over the eight other grouped splits the mean difference is '
    f"{r['c_eight_splits']['mean_difference_network_minus_trees']} K with paired 95% t interval {r['c_eight_splits']['paired_t95_interval']}, which includes zero: the network is not shown to be better than the trees; it lost on 2 of 8 splits. On the unseen-monomer split they tie (18.51 vs 18.74 K).")
r['caveats'] = [
  'The headline error mixes two problems. Test rows that are real copolymers (93 rows, 23 systems) get MAE about 6 K because their monomers mostly also occur in training rows, often with a measured homopolymer Tg that feeds the Fox input; test homopolymer rows (59), most of them chemistry seen nowhere else, get about 30 K. Only 85 copolymer systems exist in the file, and the unseen-monomer split holds out mostly rare monomers (frequent ones were skipped to keep the test share near 20%), so it understates how hard a truly new common monomer would be.',
  'Network versus trees is not settled: the paired interval over eight splits includes zero, the eight splits reuse the same 666 rows so the interval is narrower than it should be, and the network settings and the Morgan-bits choice were picked on seed-42 training rows that overlap the other splits\' test rows. The trees are piecewise constant, so only 17% of their composition sweeps are strictly monotonic against 87% for the network (96% of the measured systems are).',
  'The 90% interval is one fixed width (plus or minus 40.4 K). It covers 88.8% of test rows overall but 100% of copolymer rows (far too wide for them) and only 71% of homopolymer rows (too narrow), and 87.3% on the unseen-monomer split. Labels are literature values with unknown method, heating rate and molar mass; the same structure differs by up to 54 K between the two source files. Whether composition is mole or weight percent is not stated in the file; mole percent was assumed.']
p = lambda n: f'{W}/{n}'
r['files_written'] = [p(n) for n in ('prepare.py', 'nn_generic.js', 'run_nn.js', 'evaluate.py', 'finalize.py', 'package.json', 'nn_data.json', 'splits.json',
    'tasks.json', 'rows.json', 'prepare_report.json', 'result_core.json', 'result.json', 'runs/evaluate.log', 'runs/final.json', 'runs/cv_best_ens5.json', 'runs/cv_cfg0.json ... runs/cv_cfg9.json', 'runs/smoke.json', '.t0')]
r['environment_note'] = 'rdkit 2026.09.2 was installed with pip into the system Python (the one install attempt allowed); nothing under /workspace/ncsu-hackathon or W/data was modified; no network, MCP or paid tools were used apart from that pip install'
json.dump(r, open(f'{W}/result.json', 'w'), indent=1)
print(json.dumps(chk, indent=1))
