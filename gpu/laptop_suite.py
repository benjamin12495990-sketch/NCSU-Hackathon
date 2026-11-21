"""Runs the whole first-day sequence on the laptop and collects what to share, in one command:

  python -m gpu.laptop_suite --device cuda                     # about 10-20 minutes
  python -m gpu.laptop_suite --device cuda --full-structure    # also the full JARVIS run (100 epochs)
  python -m gpu.laptop_suite --device cpu --quick              # a fast rehearsal of the suite itself

Steps: smoke test; check_env (CPU and the device); the batched-MLP profile (CPU and the device); every
tabular family with its agent's settings; a 32-seed alloy bank; the JARVIS download (if missing); and the
structure model on the same 3,000-record subset as gpu/reference_cpu/. Everything goes to a new folder
gpu/runs/<UTC time>_suite[_label]/. Its share/ subfolder holds the compact JSONs and summary.json, which sets each
result beside gpu/reference_cpu/. A failed step is recorded, and the suite moves on to the next one.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from gpu.common import GPU, RUNS, git_info, new_run_dir, now_iso, write_json_new

FAMILIES = [('alloys', 'cv'), ('electrolytes', 'cv'), ('insulators', 'cv'), ('polymers', 'cv'), ('chemistry', 'cv'), ('concrete', 'fixed')]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--label', default='', help='a short name for this machine, e.g. rtx5080')
    ap.add_argument('--full-structure', action='store_true', help='also train on all 31,897 JARVIS records (100 epochs)')
    ap.add_argument('--quick', action='store_true', help='tiny epochs, to rehearse the suite')
    ap.add_argument('--skip-cpu', action='store_true', help='skip the CPU comparison steps')
    a = ap.parse_args(argv)
    suite = new_run_dir(RUNS, 'suite' + (f'_{a.label}' if a.label else ''))
    runs, share = suite / 'runs', suite / 'share'
    share.mkdir()
    q = a.quick
    steps = [('smoke', ['gpu.tests.smoke'])]
    devices = ([] if a.skip_cpu or a.device == 'cpu' else ['cpu']) + [a.device]
    for d in devices:
        steps.append((f'env_{d}', ['gpu.check_env', '--device', d, '--output', str(share / f'env_{d}.json')]
                      + (['--size', '512', '--reps', '3'] if q else [])))
    for d in devices:
        counts = '1,5,32,128' if d == 'cpu' else '1,5,32,128,512'
        steps.append((f'profile_{d}', ['gpu.profile_batched_mlp', '--device', d, '--counts', '1,4' if q else counts,
                                       '--epochs', '1' if q else ('3' if d == 'cpu' else '10'), '--output', str(share / f'profile_{d}.json')]))
    ep = ['--set', 'epochs=3'] if q else []
    for fam, mode in FAMILIES:
        steps.append((f'tabular_{fam}_{mode}', ['gpu.train_tabular', '--family', fam, '--mode', mode, '--device', a.device,
                                               '--compare-reference', '--runs-root', str(runs), '--tag', f'{fam}_{mode}'] + ep))
    steps.append(('tabular_alloys_cv_seeds32', ['gpu.train_tabular', '--family', 'alloys', '--members', '32', '--device', a.device,
                                                '--runs-root', str(runs), '--tag', 'alloys_seeds32'] + ep))
    if not a.skip_cpu and a.device != 'cpu':
        steps.append(('tabular_alloys_cv_cpu', ['gpu.train_tabular', '--family', 'alloys', '--device', 'cpu',
                                                '--runs-root', str(runs), '--tag', 'alloys_cpu'] + ep))
    steps.append(('fetch_jarvis', ['gpu.structure.fetch_jarvis', '--download']))
    sub = ['--limit', '300', '--epochs', '2'] if q else ['--limit', '3000', '--epochs', '15']
    steps.append(('structure_3000', ['gpu.structure.train_structure', '--device', a.device, '--runs-root', str(runs), '--tag', 's3000'] + sub))
    if a.full_structure:
        steps.append(('structure_full', ['gpu.structure.train_structure', '--device', a.device, '--runs-root', str(runs),
                                         '--tag', 'sfull', '--epochs', '2' if q else '100']))

    log = []
    for name, cmd in steps:
        t0 = time.perf_counter()
        print(f'\n=== {name}: python -m {" ".join(cmd)}', flush=True)
        with open(suite / f'{name}.log', 'w', encoding='utf-8') as f:
            rc = subprocess.run([sys.executable, '-m', *cmd], cwd=GPU.parent, stdout=f, stderr=subprocess.STDOUT).returncode
        dt = time.perf_counter() - t0
        log.append({'step': name, 'command': ['python', '-m', *cmd], 'returncode': rc, 'seconds': round(dt, 1)})
        print(f'    {"ok" if rc == 0 else f"FAILED (exit {rc}), see " + str(suite / (name + ".log"))}  {dt:.0f} s', flush=True)

    for res in sorted(runs.glob('*/*/result.json')):
        fam, run = res.parent.parent.name, res.parent.name
        shutil.copy(res, share / f'{fam}__{run}.json')
    summary = {'created': now_iso(), 'git': git_info(), 'device': a.device, 'label': a.label, 'quick': q,
               'note': 'with --quick the epochs are tiny, so the comparison with reference_cpu means nothing' if q else '',
               'steps': log,
               'comparison_with_reference_cpu': compare(share)}
    write_json_new(share / 'summary.json', summary)
    print(f'\nshare this folder: {share}')
    print(json.dumps(summary['comparison_with_reference_cpu'], indent=1))
    return 0 if all(s['returncode'] == 0 for s in log) else 1


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def compare(share):
    ref = GPU / 'reference_cpu'
    out = {}
    pairs = [(f'{fam}_{mode}', f'tabular_{fam}_{mode}.json', f'{fam}__*_{fam}_{mode}.json', mode) for fam, mode in FAMILIES]
    pairs.append(('alloys_seeds32', 'tabular_alloys_cv_seeds32.json', 'alloys__*_alloys_seeds32.json', 'cv'))
    for key, rname, pattern, mode in pairs:
        r, new = _load(ref / rname), [_load(p) for p in share.glob(pattern)]
        if not r or not new or not new[0]:
            continue
        n = new[0]
        pick = lambda d: (d['modes'][mode]['configs'][0]['sets'][('val' if mode == 'cv' else 'test')]['seed_mean_ensemble']['mae'],  # noqa: E731
                          d['modes'][mode]['train_seconds'])
        (rm, rt), (nm, nt) = pick(r), pick(n)
        out[key] = {'ensemble_mae_reference_cpu': rm, 'ensemble_mae_this_run': nm, 'train_seconds_reference_cpu': rt,
                    'train_seconds_this_run': nt, 'speedup': rt / nt if nt else None, 'device': n['device']}
    r, new = _load(ref / 'structure_jarvis_bulk_3000.json'), [_load(p) for p in share.glob('structure__*_s3000.json')]
    if r and new and new[0]:
        n = new[0]
        out['structure_3000'] = {'test_mae_reference_cpu': r['cgcnn']['test']['mae'], 'test_mae_this_run': n['cgcnn']['test']['mae'],
                                 'split_sha256_matches': r['split']['sha256'] == n['split']['sha256'],
                                 'structures_per_second_reference_cpu': r['timing']['train_structures_per_second'],
                                 'structures_per_second_this_run': n['timing']['train_structures_per_second'], 'device': n['device']}
    r = _load(ref / 'profile_cpu.json')
    for p in share.glob('profile_*.json'):
        n = _load(p)
        if r and n:
            ref_k = {x['K']: x['model_rows_per_second'] for x in r['results']}
            out[p.stem] = {str(x['K']): {'model_rows_per_second': x['model_rows_per_second'],
                                         'vs_reference_cpu': x['model_rows_per_second'] / ref_k[x['K']] if x['K'] in ref_k else None}
                           for x in n['results']}
    return out


if __name__ == '__main__':
    sys.exit(main())
