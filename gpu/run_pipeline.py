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
import time
from pathlib import Path

from gpu.common import REPO, RUNS, git_info, hardware, new_run_dir, now_iso, write_json_new
from gpu.pipeline.orchestrator import Orchestrator, Telemetry, atomic_write


def write_json_atomic(path, obj):
    import json as _json
    atomic_write(path, _json.dumps(obj, indent=1))

from gpu.pipeline.plan import STRUCT_ORDER_EARLY, STRUCT_ORDER_LATE, build_plan, describe, limit_for, pin_original_rule

VAULT = REPO / 'lab' / 'vault'
JARVIS_ZIP = REPO / 'gpu' / 'data_cache' / 'jarvis' / 'jdft_3d-9-24-2025.json.zip'


def plan_is_smoke(plan):
    return any(str(x).endswith('smoke') for x in [plan.get('profile'), *plan.get('profile_history', [])])


def say_pins(pinned):
    if pinned:
        print(f'this folder was planned before the neighbour-rule fix: {len(pinned)} of its crystal jobs or search rounds did not state a rule, so they keep '
              "the original one ('first') if they start again (" + ', '.join(pinned[:6]) + (', ...' if len(pinned) > 6 else '') + '). Finished runs are unchanged.')


def round_rule(plan, cfg):
    """The neighbour rule of the search round cfg describes: a round this folder already has (now or in cycle_history) is being continued
    and keeps the rule it began with ('first' for a round from before the fix); a round new to the folder uses cfg's own ('all')."""
    prev = next((c for c in [plan.get('cycle')] + list(plan.get('cycle_history') or []) if c and c.get('round') == cfg.get('round')), None)
    return prev.get('neighbor_ties', 'first') if prev is not None else cfg.get('neighbor_ties', 'all')


def find_pipeline(root, smoke):
    """The newest pipeline folder under root that has run, is (or is not) a rehearsal, and is not a nightly folder. Used by
    --extend latest / latest-smoke so a rehearsal or a nightly can never be mistaken for the main run: their job ids are the same."""
    import json
    for d in sorted(Path(root).glob('*/'), reverse=True):
        try:
            plan = json.loads((d / 'plan.json').read_text())
        except (OSError, ValueError):
            continue
        if (d / 'state.json').exists() and plan_is_smoke(plan) == smoke and not plan.get('nightly'):
            return d
    return None


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


def pid_alive(pid):
    """True if a process with this id is running. On Windows os.kill(pid, 0) would TERMINATE it, so ask the kernel instead."""
    if pid <= 0:
        return False
    if os.name == 'nt':
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class LockBusy(Exception):
    pass


class RunLock:
    """gpu/runs/pipeline/RUNNING.lock while a pipeline runs: a second one (say, the nightly task while a cycle is still going) refuses to start.
    The holder rewrites a "beat" time every 30 s. A lock counts as held only if its process is alive AND its beat is fresh, so a lock
    left by a crash or reboot is taken over even if Windows has since given the same process id to something else.
    On exit the lock is marked released (overwritten, not deleted)."""
    BEAT_S, STALE_S = 30.0, 180.0

    def __init__(self, root):
        self.path = Path(root) / 'RUNNING.lock'
        self._stop = None

    def _read(self):
        import json
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def _held(self, old):
        if not old.get('pid') or old.get('pid') == os.getpid() or not pid_alive(int(old['pid'])):
            return False
        return (time.time() - float(old.get('beat', 0))) < self.STALE_S

    def _write(self):
        import json
        atomic_write(self.path, json.dumps({'pid': os.getpid(), 'started': self.started, 'beat': time.time()}))

    def __enter__(self):
        import threading
        old = self._read()
        if not self._held(old) and old.get('pid') and old.get('pid') != os.getpid() and pid_alive(int(old['pid'])):
            # alive but silent: a laptop that just woke up may not have beaten yet. Give it one beat before taking over.
            time.sleep(self.BEAT_S + 15)
            old = self._read()
        if self._held(old):
            raise LockBusy(f'another pipeline is running (process {old["pid"]}, started {old.get("started")}); not starting a second one on the same GPU. '
                           f'Wait for it to finish, or stop it with a STOP file in its folder. (Lock file: {self.path})')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = now_iso()
        self._write()
        self._stop = threading.Event()

        def beat():
            while not self._stop.wait(self.BEAT_S):
                try:
                    self._write()
                except OSError:
                    pass
        threading.Thread(target=beat, daemon=True).start()
        return self

    def __exit__(self, *exc):
        import json
        if self._stop is not None:
            self._stop.set()
        try:
            atomic_write(self.path, json.dumps({'pid': None, 'released': now_iso()}))
        except OSError:
            pass
        return False


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', default='standard', choices=['smoke', 'standard', 'holdout', 'holdout-smoke', 'cycle2', 'cycle2-smoke', 'cycle3', 'cycle3-smoke', 'tiefix', 'tiefix-smoke',
                                                           'nightly', 'nightly-smoke'],
                    help='cycle2 = the holdout jobs plus generations of crystal networks; tiefix = the held-out-systems networks and trees again under the fixed '
                         'neighbour rule. Both need --extend <an earlier pipeline folder>')
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
    ap.add_argument('--extend', help='an existing pipeline folder to ADD the jobs of --profile holdout / cycle2 to (nothing already there is changed)')
    ap.add_argument('--gpu-slots', type=int, default=None, help='GPU jobs running at once (default 2 for cycle2, else 1). The card was ~20%% busy with one job.')
    ap.add_argument('--generations', type=int, default=None)
    ap.add_argument('--population', type=int, default=None, help='individuals in generation 0 (the defaults plus random ones)')
    ap.add_argument('--children', type=int, default=None, help='children per generation')
    ap.add_argument('--search-epochs', type=int, default=None)
    ap.add_argument('--final-epochs', type=int, default=None)
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--force', action='store_true', help='nightly: retrain every task even if its data and recipe did not change')
    ap.add_argument('--runs-root', default=str(RUNS / 'pipeline'))
    a = ap.parse_args(argv)
    import json

    cycle = a.profile.startswith(('cycle2', 'cycle3'))
    tiefix = a.profile.startswith('tiefix')
    smoke_run = a.profile.endswith('smoke') or a.profile == 'smoke'
    ram = total_ram_gb()
    if a.gpu_slots is not None:
        gpu_slots = a.gpu_slots
    elif (cycle or tiefix) and ram is not None and ram >= 28:
        gpu_slots = 2
    else:
        gpu_slots = 1
        if cycle or tiefix:
            print(f'using 1 GPU job at a time: this computer reports {"an unknown amount of" if ram is None else f"{ram:.0f} GB of"} memory, and two full-size crystal jobs each hold several GB '
                  'of graphs. Pass --gpu-slots 2 to run two at once if you have the memory to spare.')
    plan_profile = {'cycle2': 'holdout', 'cycle2-smoke': 'holdout-smoke'}.get(a.profile, a.profile)  # cycle3 has its own plan
    plan_args = dict(profile=plan_profile, device=a.device, families=a.families.split(',') if a.families else None,
                     targets=a.targets.split(',') if a.targets else None, tabular=not a.no_tabular, structure=not a.no_structure,
                     ram_gb=total_ram_gb(), cpu_threads=a.cpu_threads)
    nightly = a.profile.startswith('nightly')
    budget = a.budget_hours if a.budget_hours is not None else (None if smoke_run else (12.0 if cycle else 8.0 if nightly else 10.0))
    if a.extend in ('latest', 'latest-smoke'):
        found = find_pipeline(a.runs_root, smoke=(a.extend == 'latest-smoke'))
        if found is None:
            print('no ' + ('rehearsal' if a.extend == 'latest-smoke' else 'real (non-rehearsal)') + ' pipeline folder found under ' + str(a.runs_root)
                  + ('; run the rehearsal first: .\\gpu\\run_overnight.ps1 -Rehearsal' if a.extend == 'latest-smoke' else '.'))
            return 4
        a.extend = str(found)
    if (cycle or tiefix) and not (a.extend or a.resume):
        print(f'--profile {a.profile} adds to an earlier pipeline folder (it compares against that folder\'s results): '
              'use --extend <folder>, for example the folder in gpu\\runs\\pipeline\\LAST_PIPELINE.txt.')
        return 4

    if a.resume:
        pipe = Path(a.resume)
        plan = json.loads((pipe / 'plan.json').read_text())
        pinned = pin_original_rule(plan)   # jobs planned before the fix keep the rule they were planned under
        if pinned and not a.dry_run:
            shutil.copy2(pipe / 'plan.json', pipe / f'plan.before_rule_pin_{time.strftime("%Y%m%dT%H%M%S")}.json')  # the old plan is kept
            write_json_atomic(pipe / 'plan.json', plan)
        say_pins(pinned)
        if plan.get('cycle'):  # a cycle remembers its own settings; the budget clock restarts with each session
            if a.gpu_slots is None:
                gpu_slots = plan['cycle'].get('gpu_slots', gpu_slots)
            if a.budget_hours is None:
                budget = plan['cycle'].get('budget_hours', budget)
        info, stop, warn = preflight(a, plan)
    elif a.extend:
        pipe = Path(a.extend)
        plan = json.loads((pipe / 'plan.json').read_text())
        if plan.get('nightly'):
            print(f'refusing: {pipe} is a nightly folder; cycles extend the main run folder. Use --extend latest, or give the main folder.')
            return 4
        if plan_is_smoke(plan) != smoke_run:
            print(f'refusing: {pipe} is {"a rehearsal" if plan_is_smoke(plan) else "a real run"} but --profile {a.profile} is {"a rehearsal" if smoke_run else "real"}. '
                  'Their job names are the same, so mixing them would pass rehearsal results off as real ones.')
            return 4
        say_pins(pin_original_rule(plan))   # in memory; written below together with the new jobs (a copy of the old plan is kept)
        new = build_plan(**plan_args, existing=plan['jobs'], pipe=pipe)
        for s_ in new.get('skipped', []):
            print('left out:', s_)
        if tiefix and not new['jobs']:
            if new.get('skipped'):
                print(f'nothing to compare: the held-out-systems runs in {pipe} were not made with the original rule (see "left out" above), so there is no '
                      'original-rule set to compare them with; runs made with the fixed rule already give fixed-rule numbers. Nothing was started.')
            else:
                print(f'nothing to compare: {pipe} has no held-out-systems networks with their trees (jobs struct_<property>_sys_s<seed> and trees_<property>_sys). '
                      'The comparison needs a folder whose second cycle ran before the fix; a folder made after the fix already uses the fixed rule.')
            return 4
        if a.dry_run:
            print(describe(new))
            return 0
        info, stop, warn = preflight(a, {**plan, 'jobs': plan['jobs'] + new['jobs']})
    else:
        if nightly:
            from gpu.pipeline.nightly import build_nightly_plan
            if not JARVIS_ZIP.exists() and not a.dry_run:   # a fresh clone: without the file the crystal tasks would be dropped without a word
                print('fetching the JARVIS-DFT file (about 46 MB) ...', flush=True)
                r = subprocess.run([sys.executable, '-m', 'gpu.structure.fetch_jarvis', '--download'], cwd=str(REPO), capture_output=True, text=True)
                if r.returncode != 0 or not JARVIS_ZIP.exists():
                    print('WARNING: the JARVIS file could not be downloaded; tonight trains the table families only. ' + (r.stderr or r.stdout)[-200:])
            plan = build_nightly_plan(device=a.device, smoke=smoke_run, ram_gb=ram, cpu_threads=a.cpu_threads, force=a.force,
                                      registry_path=Path(a.runs_root) / ('registry_smoke.json' if smoke_run else 'registry.json'),
                                      families=plan_args['families'], targets=plan_args['targets'])
            if a.tag == '':
                a.tag = 'nightly'
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
    if a.extend and not a.resume:
        if not (pipe / 'state.json').exists():
            print(f'{pipe} has no state.json: it is not a pipeline folder that has run.')
            return 4
        shutil.copy2(pipe / 'plan.json', pipe / f'plan.before_extend_{time.strftime("%Y%m%dT%H%M%S")}.json')  # the old plan is kept
        have = {j['id'] for j in plan['jobs']}
        plan['jobs'] += [j for j in new['jobs'] if j['id'] not in have]
        plan['notes'] = list(plan.get('notes', [])) + new['notes']
        for fam, grid in (new.get('grids') or {}).items():   # sweeps added by this extension need their grid files
            plan.setdefault('grids', {})[fam] = grid
            if not (pipe / 'grids' / f'{fam}.json').exists():
                write_json_new(pipe / 'grids' / f'{fam}.json', grid)
        if new.get('members') and not plan.get('members'):
            plan['members'] = new['members']
        plan['profile_history'] = list(plan.get('profile_history', [plan['profile']])) + [a.profile]
        if cycle:
            small_ram = (total_ram_gb() or 99) < 24
            threads = max(1, min(4, ((os.cpu_count() or 8) - a.cpu_threads) // gpu_slots))
            common = {'device': a.device, 'budget_hours': budget, 'smoke': smoke_run, 'gpu_slots': gpu_slots, 'threads': threads,
                      'neighbor_ties': 'all'}   # a round begun now uses the fixed rule; evolve.py passes it to every individual and finalist
            if a.profile.startswith('cycle3'):
                from gpu.pipeline import evolve as E
                from gpu.pipeline.plan import PRETRAIN, SPACE_WIDE
                r1 = E.best_of(E.archive(pipe, plan['jobs'], PRETRAIN, None))
                base = dict(r1['genes']) if r1 else dict(E.DEFAULT)
                space = {k: sorted(set(v) | {base[k]}) for k, v in SPACE_WIDE.items()}
                r1_finals = any(j['id'].startswith(f'evof_{PRETRAIN}_s') for j in plan['jobs'])
                cfg = {'round': 'r2', 'targets': [PRETRAIN], 'split': 'system', 'space': space, 'seed_mode': 'mutants', 'bases': {PRETRAIN: base},
                       'generations': a.generations or (2 if smoke_run else 3), 'population': a.population or (3 if smoke_run else 5),
                       'children': a.children or (2 if smoke_run else 3), 'search_epochs': a.search_epochs or (2 if smoke_run else 40),
                       # the same number of epochs as the runs the finalists are compared with: otherwise longer training alone looks like a win
                       'final_epochs': a.final_epochs or (3 if smoke_run else int((plan.get('cycle') or {}).get('final_epochs') or 100)), 'final_seeds': {PRETRAIN: [31, 32]},
                       'compare_to': 'evof_{t}_s[0-9]*' if r1_finals else 'struct_{t}_sys_s[0-9]*',
                       'compare_label': 'the round-1 winner' if r1_finals else 'the default settings',
                       'limits': {PRETRAIN: limit_for(plan['jobs'], PRETRAIN, smoke_run, small_ram)}, **common}
            else:
                targets = a.targets.split(',') if a.targets else ['bulk_modulus_kv'] if smoke_run else STRUCT_ORDER_EARLY + STRUCT_ORDER_LATE
                cfg = {'targets': targets, 'generations': a.generations or (2 if smoke_run else 4), 'population': a.population or (3 if smoke_run else 6),
                       'children': a.children or (2 if smoke_run else 4), 'search_epochs': a.search_epochs or (2 if smoke_run else 40),
                       'final_epochs': a.final_epochs or (3 if smoke_run else 100), 'limits': {t: limit_for(plan['jobs'], t, smoke_run, small_ram) for t in targets},
                       'compare_to': 'struct_{t}_sys_s[0-9]*', 'compare_label': 'the default settings', **common}
            if plan.get('cycle') and plan['cycle'].get('round') != cfg.get('round'):
                plan['cycle_history'] = list(plan.get('cycle_history', [])) + [plan['cycle']]  # the earlier round's settings are kept
            cfg['neighbor_ties'] = round_rule(plan, cfg)
            plan['cycle'] = cfg
        write_json_atomic(pipe / 'plan.json', plan)
        print(describe(new))
        if cycle:
            from gpu.pipeline.evolve import forecast_hours
            c = plan['cycle']
            fh, per, sp = forecast_hours(pipe, plan, c, new['jobs'])
            per_text = ', '.join(f'{t.split("_")[0]} {v:.1f} h' for t, v in per.items())
            print(f'\nrough forecast for the whole cycle on the GPU queue: about {fh:.1f} h ({per_text} of GPU time, {c["gpu_slots"]} job(s) at a time, '
                  f'assuming {sp}x speed-up from sharing). Based on how long the same jobs took earlier in this folder; the report records the real times.')
            print(f'\nplus {"search round " + c["round"] if c.get("round") else "generations"} of crystal networks on held-out chemical systems: {len(c["targets"])} properties x {c["generations"]} generations '
                  f'(population {c["population"]}, {c["children"]} children each, {c["search_epochs"]} epochs per search run), then finalists at {c["final_epochs"]} epochs on fresh seeds, '
                  f'compared with {c.get("compare_label", "the default settings")}.')
    elif not a.resume:
        ensure_jarvis(plan, warn)
        pipe = new_run_dir(Path(a.runs_root), a.tag)
        write_json_new(pipe / 'plan.json', {**plan, 'machine': info, 'warnings': warn, 'budget_hours': budget})
        for fam, grid in plan['grids'].items():
            write_json_new(pipe / 'grids' / f'{fam}.json', grid)
        print(describe(plan))
    # Pointers, so launchers can find the folder to resume. One per kind, so a nightly or a rehearsal never hides the main run:
    #   LAST_PIPELINE.txt  the main run (standard / cycle folders)     LAST_NIGHTLY.txt, LAST_REHEARSAL.txt  the newest of those
    #   LAST_SESSION.txt   whatever ran last (used only to resume the same folder after a crash of the pipeline program)
    kind = 'LAST_NIGHTLY.txt' if plan.get('nightly') else 'LAST_REHEARSAL.txt' if plan_is_smoke(plan) else 'LAST_PIPELINE.txt'
    atomic_write(Path(a.runs_root) / kind, str(pipe.resolve()))
    atomic_write(Path(a.runs_root) / 'LAST_SESSION.txt', str(pipe.resolve()))
    print(f'\npipeline folder: {pipe}\nstatus file:     {pipe / "STATUS.txt"}   (refreshed every 30 s)\nstop: create a file named STOP in that folder (or press Ctrl+C)\n', flush=True)

    before = vault_snapshot()
    tel = Telemetry(pipe / 'telemetry.csv')
    tel.start()
    t_session = time.time()

    def make_orch(plan_, hours_left):
        return Orchestrator(pipe, plan_['jobs'], budget_hours=hours_left, cpu_workers=a.cpu_workers, gpu_slots=gpu_slots, telemetry=tel, hot_c=a.hot_c,
                            max_attempts=a.max_attempts, retry_failed=a.retry_failed, log=lambda m: print(m, flush=True))

    last_profile = str((plan.get('profile_history') or [plan.get('profile')])[-1])
    if plan.get('cycle') and not last_profile.startswith('tiefix'):   # a tiefix session (and its resumes) only runs the plan's jobs: it never grows a search
        from gpu.pipeline.evolve import run_cycle
        c = plan['cycle']
        gpu_slots = max(gpu_slots, 1)
        rc = run_cycle(pipe, plan, make_orch=make_orch, targets=c['targets'], cfg={**c, 'gpu_slots': gpu_slots}, budget_hours=budget,
                       elapsed_hours=lambda: (time.time() - t_session) / 3600.0, say=lambda m: print(m, flush=True))
    else:
        if plan.get('cycle') and last_profile.startswith('tiefix'):
            c = plan['cycle']
            if not any(j.get('group') == 'evolve_final' and j.get('round') == c.get('round') for j in plan['jobs']):
                print('note: this folder has a search round without finalists; a tiefix session does not grow it. To continue it (under the rule it began with), '
                      'run its cycle again: .\\gpu\\run_overnight.ps1 ' + ('-Cycle3' if c.get('round') else '-Cycle2') + (' -Rehearsal' if plan_is_smoke(plan) else ''), flush=True)
        rc = make_orch(plan, budget).run()
    tel.stop()
    after = vault_snapshot()
    integ = ('the sealed concrete folder (lab/vault/) is unchanged: same files, sizes and modification times; this pipeline never reads it.' if before == after
             else 'WARNING: the files in lab/vault/ changed while this pipeline ran. Investigate before trusting any result.')
    state_path = pipe / 'state.json'
    st = json.loads(state_path.read_text())
    st['integrity'] = integ
    atomic_write(state_path, json.dumps(st, indent=1))
    if plan.get('nightly'):
        try:
            from gpu.pipeline.analyze import analyze_all
            from gpu.pipeline.nightly import update_registry
            update_registry(pipe, plan, analyze_all(pipe, plan))
        except Exception as e:  # noqa: BLE001 - the night's results are on disk either way
            print(f'the registry could not be updated ({e!r})')
    from gpu.pipeline.report import generate
    try:
        rep, z = generate(pipe)
        print(f'\nreport: {rep}\nshare this one file with whoever needs the results: {z}')
    except Exception as e:  # the jobs' own results are safe on disk even if the report step fails
        print(f'\nthe report could not be written ({e!r}); rebuild it with: python -m gpu.pipeline.report --pipeline "{pipe}"')
    return rc


def main(argv=None):
    try:
        if argv is None:
            argv = sys.argv[1:]
        if '--dry-run' in argv:
            return run(argv)
        root = argv[argv.index('--runs-root') + 1] if '--runs-root' in argv else str(RUNS / 'pipeline')
        with RunLock(root):
            return run(argv)
    except LockBusy as e:
        print(f'NOT STARTED: {e}')
        return 4
    except KeyboardInterrupt:
        return 2
    except Exception:  # noqa: BLE001 - the launcher resumes after this
        import traceback
        traceback.print_exc()
        print('\nthe pipeline program crashed; finished jobs are safe on disk. Continue with: python -m gpu.run_pipeline --resume <the pipeline folder>')
        return 3


if __name__ == '__main__':
    sys.exit(main())
