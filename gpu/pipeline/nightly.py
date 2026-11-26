"""The nightly run: retrain whatever changed, re-check it, and keep a registry of the current models.

  python -m gpu.run_pipeline --profile nightly          (or .\\gpu\\run_overnight.ps1 -Nightly, which first does git pull --ff-only)

Each night gets its own folder. For every task (6 table families, 4 crystal properties) the nightly computes a key from
  the data       the SHA-256 of the family's data arrays, or of the JARVIS file
  the recipe     its entry in gpu/recipes.json (what to train, which model is recommended)
  the code       a version string below (bump it when training code changes in a way that changes results)
and trains the task only if the key differs from the one in the local registry (gpu/runs/pipeline/registry.json), or with --force.
So a night with no new data and no recipe change trains nothing and says so; new data in a family (after git pull) retrains just
that family. Recipes change only by a reviewed commit: a cycle that finds a better model is evidence for editing recipes.json.

Table families are retrained with the standard sweep and baselines (about 2-14 GPU minutes each on the RTX 5070 Laptop); the report
then shows the recommended model (trees or the no-fit blend of networks and trees). Crystal properties are trained on the
held-out-chemical-systems split with the recipe's settings and seeds, beside the tree baselines on the same split, so every night's
numbers are honest test numbers and come with conformal prediction intervals. Crystal graphs and structure features use the invariant
neighbour rule (--neighbor-ties all), so a prediction does not depend on how a crystal is written down.
"""
import hashlib
import json
import time
from pathlib import Path

from gpu.common import REPO

RECIPES = REPO / 'gpu' / 'recipes.json'
# crystal-v2 (2026-11-26): neighbour lists are cut by the invariant rule (gpu/structure/graph.py, ties='all'), which changes every crystal
# network and the structure-feature trees, so each crystal task is retrained once under the new rule.
CODE_VERSION = {'tabular': 'tab-v1', 'crystal': 'crystal-v2'}
NEIGHBOR_TIES = 'all'   # stated in every nightly crystal job, so a night's folder never depends on a default


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
        keys[f'crystal:{t}'] = _h(jarvis_sha, recipes['crystal'].get(t), CODE_VERSION['crystal'])
    return keys


def load_registry(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {'tasks': {}}


def crystal_jobs(t, recipe, *, device, smoke, small_ram, cpu_threads):
    from gpu.pipeline.evolve import genes_args
    from gpu.pipeline.plan import BIG_TARGETS, STRUCT_EST_MIN, TREES_EST_MIN, _struct_job
    lim = ['--limit', '800'] if smoke else (['--limit', '40000'] if (t in BIG_TARGETS and small_ram) else [])
    genes = recipe.get('genes', 'default')
    extra = ([] if genes == 'default' else genes_args(genes)) + ['--neighbor-ties', NEIGHBOR_TIES]
    seeds = [11, 12] if smoke else recipe['seeds']
    epochs = 3 if smoke else recipe.get('epochs', 100)
    big = genes != 'default' and (genes.get('conv', 3) > 3 or genes.get('atom_features', 64) > 64)
    jobs = [_struct_job(f'struct_{t}_sys_s{sd}', t, 'system', sd, device=device, epochs=epochs, lim=lim, smoke=smoke, extra=extra,
                        est=int(STRUCT_EST_MIN[t] * (1.8 if big else 1.0))) for sd in seeds]
    jobs.append({'id': f'trees_{t}_sys', 'queue': 'cpu', 'group': 'baseline', 'target': t, 'split': 'system',
                 'argv': ['-m', 'gpu.structure.baseline_trees', '--target', t, '--threads', str(cpu_threads), '--split', 'system',
                          '--feature-sets', 'composition,comp+cell,rich', '--neighbor-ties', NEIGHBOR_TIES, '--output', '{JOB_DIR}/trees.json', *lim],
                 'expect': ['trees.json', 'trees_predictions.npz'], 'env': {'OMP_NUM_THREADS': str(cpu_threads)}, 'timeout_s': 5 * 3600,
                 'est_minutes': 1 if smoke else TREES_EST_MIN[t] + 5})
    return jobs


def build_nightly_plan(*, device, smoke, ram_gb, cpu_threads, registry_path, force=False, families=None, targets=None):
    """Returns (plan, keys, skipped): plan with only the tasks whose key changed."""
    from gpu.common import sha256_file
    from gpu.pipeline.plan import STRUCT_ORDER_EARLY, STRUCT_ORDER_LATE, TAB_ORDER, build_plan
    from gpu.structure.train_structure import DEFAULT_ZIP
    recipes = load_recipes()
    families = families or (['alloys', 'electrolytes', 'concrete'] if smoke else TAB_ORDER)
    targets = targets or (['bulk_modulus_kv'] if smoke else STRUCT_ORDER_EARLY + STRUCT_ORDER_LATE)
    jarvis_sha = sha256_file(DEFAULT_ZIP) if DEFAULT_ZIP.exists() else None
    keys = task_keys(recipes, families, targets if jarvis_sha else [], jarvis_sha)
    reg = load_registry(registry_path)['tasks']
    todo_f = [f for f in families if force or reg.get(f'tab:{f}', {}).get('key') != keys[f'tab:{f}']]
    todo_t = [t for t in targets if jarvis_sha and (force or reg.get(f'crystal:{t}', {}).get('key') != keys[f'crystal:{t}'])]
    skipped = [k for k in keys if (k.startswith('tab:') and k[4:] not in todo_f) or (k.startswith('crystal:') and k[8:] not in todo_t)]
    plan = {'profile': 'nightly-smoke' if smoke else 'nightly', 'device': device, 'members': 2 if smoke else 5, 'jobs': [], 'grids': {}, 'notes': []}
    if todo_f:
        tab = build_plan('smoke' if smoke else 'standard', device=device, families=todo_f, structure=False, ram_gb=ram_gb, cpu_threads=cpu_threads)
        plan['jobs'] += tab['jobs']
        plan['grids'] = tab['grids']
        plan['members'] = tab['members']
    for t in todo_t:
        plan['jobs'] += crystal_jobs(t, recipes['crystal'][t], device=device, smoke=smoke, small_ram=ram_gb is not None and ram_gb < 24, cpu_threads=cpu_threads)
    plan['notes'].append(f'Nightly: {len(todo_f)} table famil{"y" if len(todo_f) == 1 else "ies"} and {len(todo_t)} crystal propert{"y" if len(todo_t) == 1 else "ies"} changed and are retrained; '
                         f'{len(skipped)} task(s) unchanged since their last nightly: {", ".join(skipped) or "none"}.')
    if not jarvis_sha:
        plan['notes'].append('The JARVIS file is missing (gpu/data_cache/jarvis/), so NO crystal property was planned tonight; '
                             'run: python -m gpu.structure.fetch_jarvis --download')
    if smoke:
        plan['notes'].append('Smoke sizes: the numbers mean nothing.')
    plan['nightly'] = {'keys': keys, 'retrain': [f'tab:{f}' for f in todo_f] + [f'crystal:{t}' for t in todo_t], 'skipped': skipped,
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

    for t in analysis.get('tabular', []):
        k = f'tab:{t["family"]}'
        if k not in nightly['retrain'] or t.get('status') != 'ok' or not all_done(lambda j, f=t['family']: j.get('family') == f):
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
        rec = recipes['crystal'][s['target']]['recommended']
        f = s.get('fairness', {})
        avg = [c for kk, c in s['compare']['comparisons'][s['strongest_trees_key']].items() if kk.startswith('network average')]
        net_mae = avg[0]['network_mae'] if avg else f.get('single', {}).get('mean_mae')
        mae = f.get('blend', {}).get('mae') if rec == 'blend' else net_mae
        mods = (ivc.get(s['target'], {}).get('models') or {})
        model_iv = (mods.get('blend') if rec == 'blend' and 'blend' in mods else mods.get('network average', {})).get('0.90', {})
        reg['tasks'][k] = {'key': nightly['keys'][k], 'folder': str(pipe), 'date': stamp, 'recommended': rec, 'mae': mae, 'network_average_mae': net_mae,
                           'strongest_trees': s['strongest_trees'], 'trees_mae': s['trees'][s['strongest_trees']]['test']['mae'], 'unit': s['label'],
                           'interval90_half_width': model_iv.get('half_width'), 'interval90_coverage': model_iv.get('coverage'),
                           'interval_model': 'blend' if rec == 'blend' and 'blend' in mods else 'network average'}
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
         + f'Unchanged: {", ".join(nightly["skipped"]) or "none"}.', '',
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
