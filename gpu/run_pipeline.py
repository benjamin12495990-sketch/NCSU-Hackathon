"""The unattended training pipeline: start it, leave, read REPORT.md in the morning.

  python -m gpu.run_pipeline --profile smoke --device cuda --tag rehearsal     # a few minutes: checks everything works
  python -m gpu.run_pipeline --profile standard --device cuda --tag overnight  # the real run (default budget 10 hours)
  python -m gpu.run_pipeline --profile standard --dry-run                      # list the jobs, run nothing
  python -m gpu.run_pipeline --resume gpu/runs/pipeline/<folder>               # continue after a crash, reboot or stop
  python -m gpu.pipeline.report --pipeline gpu/runs/pipeline/<folder>          # rebuild the report from what finished

What it does: trains a network sweep for every tabular family and crystal-graph networks for four JARVIS properties on the GPU, one
job at a time, while CPU baselines (tuned trees, linear models) run beside them; checkpoints every job so a crash or reboot loses
minutes, not hours; keeps Windows awake; watches GPU temperature; and at the end writes REPORT.md, summary.json and share.zip in
its folder. It only runs gpu.* modules, which read data through the sealed-data guard in gpu/data.py. It never deletes anything.

Exit codes: 0 everything finished; 1 finished but some jobs failed; 2 stopped early (STOP, Ctrl+C or the time budget; resume to continue);
3 the pipeline program itself crashed (rerun with --resume); 4 not started (a check failed).

To stop: create a file named STOP in the pipeline folder (finish the running job, start no more) or STOP_NOW, or press Ctrl+C.
Keep the laptop plugged in with its lid open (a closed lid sleeps the laptop whatever this program asks).
"""
import argparse
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

from gpu.common import REPO, RUNS, git_info, hardware, new_run_dir, now_iso, write_json_new
from gpu.pipeline.orchestrator import Orchestrator, Telemetry, atomic_write
from gpu.pipeline.plan import build_plan, describe

VAULT = REPO / 'lab' / 'vault'
JARVIS_ZIP = REPO / 'gpu' / 'data_cache' / 'jarvis' / 'jdft_3d-9-24-2025.json.zip'


def total_ram_gb():
    try:
        if os.name == 'nt':
            class MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                            ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('sullAvailExtendedVirtual', ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(ms)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullTotalPhys / 2 ** 30
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    return int(line.split()[1]) / 2 ** 20
    except Exception:
        return None


def vault_snapshot():
    """Sizes and modification times of the sealed folder's files. stat() only: the files' contents are never read."""
    snap = {}
    if VAULT.exists():
        for p in sorted(VAULT.rglob('*')):
            if p.is_file():
                s = p.stat()
                snap[str(p.relative_to(VAULT))] = (s.st_size, s.st_mtime_ns)
    return snap


def disk_need_gb(plan):
    """Rough disk need: crystal graph caches (shared by the seeds of a target) + checkpoints and results."""
    need = 1.5
    for t in sorted({j['target'] for j in plan['jobs'] if j['id'].startswith('struct_')}):
        big = t in ('formation_energy_peratom', 'optb88vdw_bandgap')
        capped = any('--limit' in j['argv'] for j in plan['jobs'] if j.get('target') == t)
        need += 2.5 if (not big or capped) else 6.5
    return need


def preflight(a, plan):
    """Returns (machine info, list of problems that stop the run, list of warnings)."""
    stop, warn = [], []
    info = {'python': sys.version.split()[0], 'created': now_iso(), 'git': git_info(), 'ram_gb': total_ram_gb(), 'hardware': hardware()}
    if a.device.startswith('cuda'):
        try:
            import torch
            if not torch.cuda.is_available():
                stop.append('--device cuda was asked for, but PyTorch cannot see a CUDA GPU. Install the CUDA build of PyTorch (gpu/README.md section 1), '
                            'or rehearse on the CPU with --device cpu.')
            else:
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
                info['torch'] = torch.__version__
        except ImportError:
            stop.append('PyTorch is not installed in this Python environment.')
    free = shutil.disk_usage(RUNS.parent if RUNS.parent.exists() else REPO).free / 2 ** 30
    info['disk_free_gb'] = round(free, 1)
    need = disk_need_gb(plan)
    info['disk_need_gb_estimate'] = need
    if free < 3:
        stop.append(f'only {free:.1f} GB of disk space is free; at least 3 GB is needed.')
    elif free < need:
        warn.append(f'{free:.1f} GB of disk space is free but this plan may need about {need:.0f} GB (crystal graph caches). Jobs that run out of space fail and the report says which; free some space or use --no-structure / --targets.')
    return info, stop, warn


def ensure_jarvis(plan, warn):
    """The crystal jobs need the JARVIS file. Fetch it if missing; if that fails, drop the crystal jobs and say so."""
    if not any(j['id'].startswith(('struct_', 'trees_')) for j in plan['jobs']) or JARVIS_ZIP.exists():
        return
    print('fetching the JARVIS-DFT file (about 46 MB) ...', flush=True)
    r = subprocess.run([sys.executable, '-m', 'gpu.structure.fetch_jarvis', '--download'], cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0 or not JARVIS_ZIP.exists():
        warn.append('The JARVIS file could not be downloaded, so the crystal-structure jobs were dropped from this run. ' + (r.stderr or r.stdout)[-300:])
        plan['jobs'] = [j for j in plan['jobs'] if not j['id'].startswith(('struct_', 'trees_'))]
        plan['notes'].append('crystal-structure jobs dropped: JARVIS download failed')


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', default='standard', choices=['smoke', 'standard'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tag', default='')
    ap.add_argument('--budget-hours', type=float, default=None, help='stop starting new jobs after this many hours (default 10 for standard)')
    ap.add_argument('--cpu-workers', type=int, default=1, help='CPU baseline jobs running beside the GPU queue; 0 runs everything one at a time')
    ap.add_argument('--cpu-threads', type=int, default=2, help='threads per CPU job (kept small so the GPU jobs keep a core to themselves)')
    ap.add_argument('--families', help='comma-separated subset of the tabular families')
    ap.add_argument('--targets', help='comma-separated subset of the JARVIS properties')
    ap.add_argument('--no-tabular', action='store_true')
    ap.add_argument('--no-structure', action='store_true')
    ap.add_argument('--max-attempts', type=int, default=2)
    ap.add_argument('--hot-c', type=float, default=92.0, help='stop a GPU job to cool down when the card stays at or above this temperature')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    ap.add_argument('--resume', help='an existing pipeline folder to continue')
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--runs-root', default=str(RUNS / 'pipeline'))
    a = ap.parse_args(argv)
    import json

    plan_args = dict(profile=a.profile, device=a.device, families=a.families.split(',') if a.families else None,
                     targets=a.targets.split(',') if a.targets else None, tabular=not a.no_tabular, structure=not a.no_structure,
                     ram_gb=total_ram_gb(), cpu_threads=a.cpu_threads)
    budget = a.budget_hours if a.budget_hours is not None else (None if a.profile == 'smoke' else 10.0)

    if a.resume:
        pipe = Path(a.resume)
        plan = json.loads((pipe / 'plan.json').read_text())
        info, stop, warn = preflight(a, plan)
    else:
        plan = build_plan(**plan_args)
        if a.dry_run:
            print(describe(plan))
            return 0
        info, stop, warn = preflight(a, plan)
    for w in warn:
        print('WARNING:', w)
    if stop:
        print('\nNOT STARTED:\n  ' + '\n  '.join(stop))
        return 4
    if not a.resume:
        ensure_jarvis(plan, warn)
        pipe = new_run_dir(Path(a.runs_root), a.tag)
        write_json_new(pipe / 'plan.json', {**plan, 'machine': info, 'warnings': warn, 'budget_hours': budget})
        for fam, grid in plan['grids'].items():
            write_json_new(pipe / 'grids' / f'{fam}.json', grid)
        print(describe(plan))
    atomic_write(Path(a.runs_root) / 'LAST_PIPELINE.txt', str(pipe.resolve()))  # a pointer, so launchers can find the folder to resume
    print(f'\npipeline folder: {pipe}\nstatus file:     {pipe / "STATUS.txt"}   (refreshed every 30 s)\nstop: create a file named STOP in that folder (or press Ctrl+C)\n', flush=True)

    before = vault_snapshot()
    tel = Telemetry(pipe / 'telemetry.csv')
    tel.start()
    orch = Orchestrator(pipe, plan['jobs'], budget_hours=budget, cpu_workers=a.cpu_workers, telemetry=tel, hot_c=a.hot_c,
                        max_attempts=a.max_attempts, retry_failed=a.retry_failed, log=lambda m: print(m, flush=True))
    rc = orch.run()
    tel.stop()
    after = vault_snapshot()
    integ = ('the sealed concrete folder (lab/vault/) is unchanged: same files, sizes and modification times; this pipeline never reads it.' if before == after
             else 'WARNING: the files in lab/vault/ changed while this pipeline ran. Investigate before trusting any result.')
    state_path = pipe / 'state.json'
    st = json.loads(state_path.read_text())
    st['integrity'] = integ
    atomic_write(state_path, json.dumps(st, indent=1))
    from gpu.pipeline.report import generate
    try:
        rep, z = generate(pipe)
        print(f'\nreport: {rep}\nshare this one file with whoever needs the results: {z}')
    except Exception as e:  # the jobs' own results are safe on disk even if the report step fails
        print(f'\nthe report could not be written ({e!r}); rebuild it with: python -m gpu.pipeline.report --pipeline "{pipe}"')
    return rc


def main(argv=None):
    try:
        return run(argv)
    except KeyboardInterrupt:
        return 2
    except Exception:  # noqa: BLE001 - the launcher resumes after this
        import traceback
        traceback.print_exc()
        print('\nthe pipeline program crashed; finished jobs are safe on disk. Continue with: python -m gpu.run_pipeline --resume <the pipeline folder>')
        return 3


if __name__ == '__main__':
    sys.exit(main())
