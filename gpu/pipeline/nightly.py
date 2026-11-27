"""The nightly run: retrain whatever changed, re-check it, and keep a registry of the current models.

  python -m gpu.run_pipeline --profile nightly          (or .\\gpu\\run_overnight.ps1 -Nightly, which first does git pull --ff-only)

Each night gets its own folder. For every task (9 table families, 4 crystal properties) the nightly computes a key from
  the data       the SHA-256 of the family's data arrays, or of the JARVIS file
  the recipe     its entry in gpu/recipes.json (what to train, which model is recommended)
  the code       a version string below (bump it when training code changes in a way that changes results)
and trains the task only if the key differs from the one in the local registry (gpu/runs/pipeline/registry.json), or with --force.
So a night with no new data and no recipe change trains nothing and says so; new data in a family (after git pull) retrains just
that family. Recipes change only by a reviewed commit: a cycle that finds a better model is evidence for editing recipes.json.

Table families are retrained with the standard sweep and baselines (about 2-14 GPU minutes each on the RTX 5070 Laptop); the report
then shows the recommended model (trees or the no-fit blend of networks and trees). Crystal properties (recommended: network, trees or the
blend; the networks are trained either way, for the comparison) are trained on the
held-out-chemical-systems split with the recipe's settings and seeds, beside the tree baselines on the same split, so every night's
numbers are honest test numbers and come with conformal prediction intervals. Crystal graphs and structure features use the invariant
neighbour rule (--neighbor-ties all), so a prediction does not depend on how a crystal is written down.

A crystal property whose key did NOT change is not left idle (DEC-023): the night adds GROW_SEEDS new network seeds to it, until its
ensemble holds MAX_SEEDS. The seeds already trained are carried into tonight's folder from the registry's folder (only the files the
analysis reads, plus the trained model; hard links where the disk allows), so the night's analysis, report and site export see the whole
ensemble and every folder stays self-contained. The tree baseline is carried too: it is deterministic and would give the same numbers.
Seeds are pooled only under an unchanged key: same data, same recipe, same code, same test split. The table families are not grown yet.
"""
import glob
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from gpu.common import REPO

RECIPES = REPO / 'gpu' / 'recipes.json'
# crystal-v2 (2026-11-26): neighbour lists are cut by the invariant rule (gpu/structure/graph.py, ties='all'), which changes every crystal
# network and the structure-feature trees, so each crystal task is retrained once under the new rule.
CODE_VERSION = {'tabular': 'tab-v1', 'crystal': 'crystal-v2'}
NEIGHBOR_TIES = 'all'   # stated in every nightly crystal job, so a night's folder never depends on a default
# Properties the nightly trains beyond the four of the standard profile (gpu/pipeline/plan.py). DEC-022.
NIGHTLY_EXTRA_TARGETS = ['mbj_bandgap']
# Seed growth for unchanged crystal properties (DEC-023). Kept here, not in recipes.json: a recipe entry is part of the task key, so
# writing these there would retrain every property from scratch.
GROW_SEEDS = 2
MAX_SEEDS = 10
CARRY_FILES = {'result.json', 'predictions.npz', 'run_config.json', 'split.json', 'best_model.pt', 'trees.json', 'trees_predictions.npz'}


def load_recipes():
    return json.loads(RECIPES.read_text())


def _h(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def task_keys(recipes, families, targets, jarvis_sha):
    from gpu.data import load
    keys = {}
    for f in families:
        keys[f'tab:{f}'] = _h(load(f).data_sha256, recipes['tabular'].get(f), CODE_VERSION['tabular'])
    for t in targets:
        keys[f'crystal:{t}'] = _h(jarvis_sha, recipes['crystal'].get(t), CODE_VERSION['crystal'], *_exclude_sha(t))
    return keys


def _exclude_sha(t):
    """The SHA-256 of a target's hold-out list (train_structure TARGETS 'exclude'), as an extra key part; nothing for the other targets, so
    their keys are what they were. A changed list changes which crystals are trained and tested, so the task retrains from scratch instead
    of pooling seeds trained on different crystals (DEC-023)."""
    from gpu.common import sha256_file
    from gpu.structure.train_structure import TARGETS
    ex = TARGETS.get(t, {}).get('exclude')
    return [sha256_file(Path(__file__).resolve().parents[1] / 'structure' / ex)] if ex else []


def load_registry(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {'tasks': {}}


def crystal_jobs(t, recipe, *, device, smoke, small_ram, cpu_threads, seeds=None, trees=True):
    from gpu.pipeline.evolve import genes_args
    from gpu.pipeline.plan import BIG_TARGETS, STRUCT_EST_MIN, TREES_EST_MIN, _struct_job
    lim = ['--limit', '800'] if smoke else (['--limit', '40000'] if (t in BIG_TARGETS and small_ram) else [])
    genes = recipe.get('genes', 'default')
    extra = ([] if genes == 'default' else genes_args(genes)) + ['--neighbor-ties', NEIGHBOR_TIES]
    seeds = seeds if seeds is not None else [11, 12] if smoke else recipe['seeds']
    epochs = 3 if smoke else recipe.get('epochs', 100)
    big = genes != 'default' and (genes.get('conv', 3) > 3 or genes.get('atom_features', 64) > 64)
    jobs = [_struct_job(f'struct_{t}_sys_s{sd}', t, 'system', sd, device=device, epochs=epochs, lim=lim, smoke=smoke, extra=extra,
                        est=int(STRUCT_EST_MIN[t] * (1.8 if big else 1.0))) for sd in seeds]
    if not trees:
        return jobs
    jobs.append({'id': f'trees_{t}_sys', 'queue': 'cpu', 'group': 'baseline', 'target': t, 'split': 'system',
                 'argv': ['-m', 'gpu.structure.baseline_trees', '--target', t, '--threads', str(cpu_threads), '--split', 'system',
                          '--feature-sets', 'composition,comp+cell,rich', '--neighbor-ties', NEIGHBOR_TIES, '--output', '{JOB_DIR}/trees.json', *lim],
                 'expect': ['trees.json', 'trees_predictions.npz'], 'env': {'OMP_NUM_THREADS': str(cpu_threads)}, 'timeout_s': 5 * 3600,
                 'est_minutes': 1 if smoke else TREES_EST_MIN[t] + 5})
    return jobs


def completed_seeds(folder, t):
    """Network seeds of property t whose held-out-systems run finished (result.json and predictions.npz) in a pipeline folder."""
    out = []
    for job in sorted(glob.glob(str(Path(folder) / 'jobs' / f'struct_{t}_sys_s[0-9]*'))):
        done = [r for r in glob.glob(str(Path(job) / 'structure' / '*' / 'result.json')) if (Path(r).parent / 'predictions.npz').exists()]
        tail = Path(job).name.rsplit('_s', 1)[-1]
        if done and tail.isdigit():
            out.append(int(tail))
    return sorted(out)


def _run_config(folder, t, seed):
    for r in sorted(glob.glob(str(Path(folder) / 'jobs' / f'struct_{t}_sys_s{seed}' / 'structure' / '*' / 'run_config.json'))):
        try:
            return json.loads(Path(r).read_text()).get('config') or {}
        except (OSError, ValueError):
            pass
    return None


def grow_plan(t, entry, *, cap=MAX_SEEDS, step=GROW_SEEDS, expect=None):
    """For a crystal property whose key did not change: (carried seeds, new seeds, source folder) or (None, reason) when it cannot grow.
    expect: settings tonight's new runs will use ({config field: value}); every carried run must have been trained with the same ones,
    or the property is not grown (seeds trained on different settings or crystals must never be pooled)."""
    folder = Path(entry['folder']) if entry.get('folder') else None
    if folder is None or not folder.is_dir():
        return None, f'crystal:{t} was not grown: its last folder ({folder.name if folder else "none"}) is not on this computer'
    have = completed_seeds(folder, t)
    if not have:
        return None, f'crystal:{t} was not grown: no finished network run in {folder.name}'
    if not (folder / 'jobs' / f'trees_{t}_sys' / 'trees_predictions.npz').exists():
        return None, f'crystal:{t} was not grown: its tree baseline is missing in {folder.name}'
    if len(have) >= cap:
        return None, None   # full: nothing to say every night
    for sd in have:
        cfg = _run_config(folder, t, sd)
        if cfg is None:
            return None, f'crystal:{t} was not grown: seed {sd} in {folder.name} has no readable run_config.json'
        diff = {k: (cfg.get(k), v) for k, v in (expect or {}).items() if (int(cfg.get(k) or 0) if k == 'limit' else cfg.get(k)) != v}
        if diff:
            return None, f'crystal:{t} was not grown: seed {sd} in {folder.name} was trained with other settings than tonight\'s {diff}'
    new = list(range(max(have) + 1, max(have) + 1 + min(step, cap - len(have))))
    return (have, new, folder), None


def _carry_source(c, pipe):
    """The folder a carry entry copies from: stored by name when it sits beside tonight's folder (no local path in plan.json), else as given."""
    f = Path(c['from'])
    return f if f.is_absolute() else Path(pipe).parent / f


def carry_runs(pipe, plan, log=print):
    """Copies the runs a grown property keeps (plan['carry']) into tonight's folder, before any job starts, and returns the targets whose
    carry is incomplete. Only CARRY_FILES are taken (not the training checkpoints); a hard link is used where the file system allows, else a
    copy. Each job is assembled in a short hidden folder (jobs/.carry<pid>_<n>, which no analysis pattern can match and which keeps paths
    short on Windows) and renamed into place in one step, with retries for a file another program briefly holds. Idempotent: a job folder
    already there is left as it is, so a resumed night does not copy twice. Nothing is ever deleted or overwritten, and nothing here can
    stop the night: a property whose carry fails is not grown tonight (update_registry checks the seeds it finds)."""
    incomplete = set()
    jobs_dir = Path(pipe) / 'jobs'
    n_tmp = 0
    for c in plan.get('carry', []):
        src_folder = _carry_source(c, pipe)
        for job in c['jobs']:
            src, dst = src_folder / 'jobs' / job, jobs_dir / job
            if dst.exists():
                continue
            files = [f for f in sorted(src.rglob('*')) if f.is_file() and f.name in CARRY_FILES and 'ckpt' not in f.relative_to(src).parts] \
                if src.is_dir() else []
            if not files:
                log(f'carry: {job} is missing in {src_folder.name}; {c["target"]} is not grown tonight')
                incomplete.add(c['target'])
                continue
            try:
                n_tmp += 1
                tmp = jobs_dir / f'.carry{os.getpid()}_{n_tmp}'
                while tmp.exists():
                    n_tmp += 1
                    tmp = jobs_dir / f'.carry{os.getpid()}_{n_tmp}'
                tmp.mkdir(parents=True)
                for f in files:
                    out = tmp / f.relative_to(src)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(f, out)
                    except OSError:
                        shutil.copy2(f, out)
                (tmp / 'carried.json').write_text(json.dumps({'from_folder': src_folder.name, 'job': job, 'files': len(files)}, indent=1))
                for attempt in range(6):
                    try:
                        tmp.replace(dst)
                        break
                    except OSError:
                        if attempt == 5:
                            raise
                        time.sleep(1)
                log(f'carried {job} from {src_folder.name} ({len(files)} files)')
            except OSError as e:
                log(f'carry: {job} could not be carried ({type(e).__name__}: {e.strerror or e}); {c["target"]} is not grown tonight')
                incomplete.add(c['target'])
    return incomplete


def expected_config(t, recipe, *, smoke, small_ram):
    """The run_config fields tonight's network runs of property t will have (from the same rules crystal_jobs uses)."""
    from gpu.pipeline.plan import BIG_TARGETS
    out = {'target': t, 'split': 'system', 'neighbor_ties': NEIGHBOR_TIES, 'epochs': 3 if smoke else recipe.get('epochs', 100),
           'limit': 800 if smoke else (40000 if (t in BIG_TARGETS and small_ram) else 0)}
    genes = recipe.get('genes', 'default')
    if genes != 'default':
        out.update({k: v for k, v in genes.items() if k in ('atom_features', 'conv', 'hidden', 'lr', 'weight_decay', 'batch')})
    return out


def build_nightly_plan(*, device, smoke, ram_gb, cpu_threads, registry_path, force=False, families=None, targets=None):
    """Returns (plan, keys, skipped): plan with only the tasks whose key changed."""
    from gpu.common import sha256_file
    from gpu.pipeline.plan import STRUCT_ORDER_EARLY, STRUCT_ORDER_LATE, TAB_ORDER, build_plan
    from gpu.structure.train_structure import DEFAULT_ZIP
    recipes = load_recipes()
    families = families or (['alloys', 'electrolytes', 'concrete'] if smoke else TAB_ORDER)
    targets = targets or (['bulk_modulus_kv'] if smoke else STRUCT_ORDER_EARLY + STRUCT_ORDER_LATE + NIGHTLY_EXTRA_TARGETS)
    jarvis_sha = sha256_file(DEFAULT_ZIP) if DEFAULT_ZIP.exists() else None
    keys = task_keys(recipes, families, targets if jarvis_sha else [], jarvis_sha)
    reg = load_registry(registry_path)['tasks']
    todo_f = [f for f in families if force or reg.get(f'tab:{f}', {}).get('key') != keys[f'tab:{f}']]
    todo_t = [t for t in targets if jarvis_sha and (force or reg.get(f'crystal:{t}', {}).get('key') != keys[f'crystal:{t}'])]
    grow, carry, grow_notes = {}, [], []
    for t in targets:
        if not jarvis_sha or force or t in todo_t:
            continue
        g, why = grow_plan(t, reg.get(f'crystal:{t}', {}), expect=expected_config(t, recipes['crystal'][t], smoke=smoke,
                                                                               small_ram=ram_gb is not None and ram_gb < 24))
        if why:
            grow_notes.append(why)
        if g:
            have, new, folder = g
            grow[t] = {'carried_seeds': have, 'new_seeds': new, 'from_folder': folder.name}
            # by name when the folder sits beside tonight's (the runs root): plan.json then holds no local path
            beside = folder.resolve().parent == Path(registry_path).resolve().parent
            carry.append({'target': t, 'from': folder.name if beside else str(folder),
                          'jobs': [f'struct_{t}_sys_s{sd}' for sd in have] + [f'trees_{t}_sys']})
    skipped = [k for k in keys if (k.startswith('tab:') and k[4:] not in todo_f) or (k.startswith('crystal:') and k[8:] not in todo_t and k[8:] not in grow)]
    plan = {'profile': 'nightly-smoke' if smoke else 'nightly', 'device': device, 'members': 2 if smoke else 5, 'jobs': [], 'grids': {}, 'notes': []}
    if todo_f:
        tab = build_plan('smoke' if smoke else 'standard', device=device, families=todo_f, structure=False, ram_gb=ram_gb, cpu_threads=cpu_threads)
        plan['jobs'] += tab['jobs']
        plan['grids'] = tab['grids']
        plan['members'] = tab['members']
    for t in todo_t:
        plan['jobs'] += crystal_jobs(t, recipes['crystal'][t], device=device, smoke=smoke, small_ram=ram_gb is not None and ram_gb < 24, cpu_threads=cpu_threads)
    for t, g in grow.items():
        plan['jobs'] += crystal_jobs(t, recipes['crystal'][t], device=device, smoke=smoke, small_ram=ram_gb is not None and ram_gb < 24, cpu_threads=cpu_threads,
                                     seeds=g['new_seeds'], trees=False)
    if carry:
        plan['carry'] = carry
    plan['notes'].append(f'Nightly: {len(todo_f)} table famil{"y" if len(todo_f) == 1 else "ies"} and {len(todo_t)} crystal propert{"y" if len(todo_t) == 1 else "ies"} changed and are retrained; '
                         f'{len(skipped)} task(s) unchanged since their last nightly: {", ".join(skipped) or "none"}.')
    for t, g in grow.items():
        n0, n1 = len(g['carried_seeds']), len(g['carried_seeds']) + len(g['new_seeds'])
        plan['notes'].append(f'Growing crystal:{t} (data, recipe and code unchanged): {n0} -> {n1} network seeds; seeds {g["carried_seeds"]} and its tree '
                             f'baseline carried from {g["from_folder"]}, new seeds {g["new_seeds"]} trained tonight (DEC-023).')
    plan['notes'] += grow_notes
    if not jarvis_sha:
        plan['notes'].append('The JARVIS file is missing (gpu/data_cache/jarvis/), so NO crystal property was planned tonight; '
                             'run: python -m gpu.structure.fetch_jarvis --download')
    if smoke:
        plan['notes'].append('Smoke sizes: the numbers mean nothing.')
    plan['nightly'] = {'keys': keys, 'retrain': [f'tab:{f}' for f in todo_f] + [f'crystal:{t}' for t in todo_t] + [f'crystal:{t}' for t in grow],
                       'grow': {f'crystal:{t}': g for t, g in grow.items()}, 'skipped': skipped,
                       'registry': str(registry_path), 'recipes_version': recipes.get('version')}
    return plan


def _recommended_tab(t, rec):
    m = t['methods']
    f = t.get('fairness', {})
    if rec == 'blend' and 'blend' in f:
        return 'blend', f['blend']['mae']
    if rec == 'network':
        return 'network average', m['network']['mae']
    return 'trees', m['trees']['mae']


def update_registry(pipe, plan, analysis):
    """Writes the registry entries of every task this night retrained successfully. Unchanged tasks keep their old entries."""
    nightly = plan.get('nightly')
    if not nightly:
        return None
    path = Path(nightly['registry'])
    reg = load_registry(path)
    recipes = load_recipes()
    iv = analysis.get('intervals') or {}
    ivt = {x['task']: x for x in iv.get('tabular', []) if 'models' in x}
    ivc = {x['task']: x for x in iv.get('crystal', []) if 'models' in x and x.get('split') == 'system'}
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        state = json.loads((Path(pipe) / 'state.json').read_text())['jobs']
    except (OSError, ValueError, KeyError):
        state = {}

    def all_done(pred):
        # A task counts as trained only when EVERY one of its jobs finished: a night cut short (budget, STOP, a failed seed) must be
        # retried the next night, not remembered as done with fewer runs than the recipe asks for.
        ids = [j['id'] for j in plan['jobs'] if pred(j)]
        return bool(ids) and all(state.get(i, {}).get('status') == 'done' for i in ids)

    def newer_registered(k):
        # a resumed OLDER night must not replace what a newer night registered under the same key (it would shrink a grown ensemble)
        e = reg['tasks'].get(k) or {}
        return e.get('key') == nightly['keys'][k] and Path(e.get('folder', '')).name > Path(pipe).name

    for t in analysis.get('tabular', []):
        k = f'tab:{t["family"]}'
        if k not in nightly['retrain'] or t.get('status') != 'ok' or not all_done(lambda j, f=t['family']: j.get('family') == f):
            continue
        if newer_registered(k):
            continue
        rec = recipes['tabular'].get(t['family'], {}).get('recommended', 'trees')
        name, mae = _recommended_tab(t, rec)
        model_iv = (ivt.get(t['family'], {}).get('models') or {}).get(name, {}).get('0.90', {})
        reg['tasks'][k] = {'key': nightly['keys'][k], 'folder': str(pipe), 'date': stamp, 'recommended': name, 'mae': mae, 'unit': t['target'],
                           'trees_mae': t['methods']['trees']['mae'], 'interval90_half_width': model_iv.get('half_width'), 'interval90_coverage': model_iv.get('coverage')}
    for s in analysis.get('structure_system', []):
        k = f'crystal:{s["target"]}'
        if k not in nightly['retrain'] or s.get('status') != 'ok' or not all_done(lambda j, tg=s['target']: j.get('target') == tg):
            continue
        if newer_registered(k):
            continue
        seeds = completed_seeds(pipe, s['target'])
        g = (nightly.get('grow') or {}).get(k)
        if g and seeds != sorted(g['carried_seeds'] + g['new_seeds']):
            continue   # the carry was incomplete: keep last night's entry; tonight's runs stay on disk and the next night tries again
        if s.get('n_runs') is not None and s['n_runs'] != len(seeds):
            continue   # the analysis pooled a different set of runs than the seeds found: never register that
        rec = recipes['crystal'][s['target']]['recommended']
        f = s.get('fairness', {})
        avg = [c for kk, c in s['compare']['comparisons'][s['strongest_trees_key']].items() if kk.startswith('network average')]
        net_mae = avg[0]['network_mae'] if avg else f.get('single', {}).get('mean_mae')
        trees_mae = s['trees'][s['strongest_trees']]['test']['mae']
        # The networks are trained for every crystal recipe (the night's comparison needs them); the recipe only says which model is recommended.
        mae = f.get('blend', {}).get('mae') if rec == 'blend' else trees_mae if rec == 'trees' else net_mae
        mods = (ivc.get(s['target'], {}).get('models') or {})
        imodel = rec if rec in ('blend', 'trees') and rec in mods else 'network average'
        model_iv = mods.get(imodel, {}).get('0.90', {})
        reg['tasks'][k] = {'key': nightly['keys'][k], 'folder': str(pipe), 'date': stamp, 'recommended': rec, 'mae': mae, 'network_average_mae': net_mae,
                           'strongest_trees': s['strongest_trees'], 'trees_mae': trees_mae, 'unit': s['label'],
                           'interval90_half_width': model_iv.get('half_width'), 'interval90_coverage': model_iv.get('coverage'),
                           'interval_model': imodel, 'seeds': completed_seeds(pipe, s['target'])}
    reg['updated'] = stamp
    tmp = path.with_suffix('.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(reg, indent=1))
    tmp.replace(path)
    return reg


def registry_lines(plan, pipe=None):
    """Report lines: what is current for every task, retrained tonight or not."""
    nightly = plan.get('nightly')
    if not nightly:
        return []
    reg = load_registry(nightly['registry'])['tasks']
    here = None if pipe is None else str(pipe)
    ok = [k for k in nightly['retrain'] if reg.get(k, {}).get('key') == nightly['keys'][k] and (here is None or reg[k].get('folder') == here)]
    not_ok = [k for k in nightly['retrain'] if k not in ok]
    L = ['## Nightly: the current model for every task', '',
         f'Retrained tonight: {", ".join(ok) or ("nothing" if not_ok else "nothing (no data or recipe changed)")}. '
         + (f'**Planned but NOT finished (will be retried next night): {", ".join(not_ok)}.** ' if not_ok else '')
         + f'Unchanged: {", ".join(nightly["skipped"]) or "none"}.'
         + ''.join(f' {k} grew from {len(g["carried_seeds"])} to {len(g["carried_seeds"]) + len(g["new_seeds"])} network seeds (same data, recipe and code; DEC-023).'
                   for k, g in (nightly.get('grow') or {}).items() if k in ok), '',
         '| Task | Recommended model | Test MAE | Strongest trees MAE | 90% interval half-width / coverage | Trained |', '| --- | --- | --- | --- | --- | --- |']
    for k in sorted(nightly['keys']):
        e = reg.get(k)
        if not e:
            L.append(f'| {k} | - | not trained yet | - | - | - |')
            continue
        if e.get('key') != nightly['keys'][k]:
            L.append(f'| {k} | OUT OF DATE: the data or recipe changed and the retrain did not finish; the last finished model ({e.get("date")}) was {e.get("recommended")}, MAE {e.get("mae")} | - | - | - | - |')
            continue
        hw = e.get('interval90_half_width', e.get('interval90_half_width_network'))
        cov = e.get('interval90_coverage', e.get('interval90_coverage_network'))
        f3 = lambda x: '-' if x is None else f'{x:.4g}'  # noqa: E731
        L.append(f'| {k} | {e.get("recommended")} | {f3(e.get("mae"))} | {f3(e.get("trees_mae"))} | {f3(hw)} / {f3(cov)} | {e.get("date")} ({Path(e["folder"]).name}) |')
    L += ['', 'Intervals are for the recommended model (for crystal runs made before trees saved validation predictions, for the network average). Recipes: `gpu/recipes.json`.', '']
    return L
