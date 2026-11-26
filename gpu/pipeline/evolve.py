"""Generations of crystal networks: a small evolutionary search over network settings, under the HARD split (held-out chemical systems).

For each JARVIS property:
  generation 0   the default settings plus random ones (a population)
  generation g   children: copies of the best two individuals so far ("parents") with two settings nudged to a neighbouring value
                 (occasionally a random jump). Every individual is trained as one ordinary run (gpu.structure.train_structure),
                 on the same seed, for a short number of epochs.
  reward         the validation MAE on held-out chemical systems. Test rows are never used to choose, mutate or rank anything.
  finalists      the best individual by validation MAE is retrained for the full number of epochs on FRESH seeds, and only then
                 scored on the test systems, against (a) the default settings trained the same way earlier in this pipeline
                 and (b) the strongest tree baseline. Selecting the winner of ~20 candidates on validation rows and then
                 re-measuring it on new seeds and unseen test rows removes the best-of-many optimism.
Nothing is ever deleted: every individual's run folder (config, split, predictions, weights) stays, and lineage.json lists every
individual with its parent, settings and validation error. The search is deterministic: the same results give the same children,
so an interrupted cycle resumes into exactly the same population.
"""
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

SPACE = {'atom_features': [32, 48, 64, 96, 128], 'conv': [2, 3, 4, 5], 'hidden': [64, 128, 192, 256],
         'lr': [5e-4, 1e-3, 2e-3, 4e-3], 'weight_decay': [0.0, 1e-5, 1e-4, 1e-3], 'batch': [32, 64, 128]}
DEFAULT = {'atom_features': 64, 'conv': 3, 'hidden': 128, 'lr': 2e-3, 'weight_decay': 0.0, 'batch': 64}   # train_structure's defaults
FLAGS = {'atom_features': '--atom-features', 'conv': '--conv', 'hidden': '--hidden', 'lr': '--lr', 'weight_decay': '--weight-decay', 'batch': '--batch'}
SEARCH_SEED = 11
FINAL_SEEDS = {'bulk_modulus_kv': [21, 22, 23], 'shear_modulus_gv': [21, 22, 23], 'formation_energy_peratom': [21, 22], 'optb88vdw_bandgap': [21, 22]}


def gkey(genes):
    return json.dumps({k: genes[k] for k in sorted(genes)})


def genes_args(genes):
    return [x for k in FLAGS for x in (FLAGS[k], str(genes[k]))]


def describe_genes(genes, base=DEFAULT):
    """Plain words: only what differs from the defaults."""
    diff = [f'{k} {genes[k]}' for k in FLAGS if genes[k] != base[k]]
    return ', '.join(diff) if diff else 'the defaults'


def rng_for(target, gen, salt=''):
    return np.random.default_rng(int(hashlib.sha256(f'evolve:{target}:{gen}:{salt}'.encode()).hexdigest()[:12], 16))


def random_genes(rng, space=SPACE):
    return {k: v[int(rng.integers(len(v)))] for k, v in space.items()}


def mutate(genes, rng, n_change=2, jump=0.2, space=SPACE):
    child = dict(genes)
    for k in rng.choice(list(space), size=n_change, replace=False):
        vals, i = space[k], space[k].index(child[k])
        if rng.random() < jump:
            opts = [j for j in range(len(vals)) if j != i]
        else:
            opts = [j for j in (i - 1, i + 1) if 0 <= j < len(vals)]
        child[k] = vals[int(rng.choice(opts))]
    return child


def prefix(rnd):
    return 'evo' if rnd is None else f'evo_{rnd}'


def archive(pipe, plan_jobs, target, rnd=None):
    """Every individual of one search round for a target, with its settings, parent and (if finished) validation MAE.
    Reads result.json only for the validation error. Round None is the first search (job ids evo_<target>_...)."""
    out = []
    for j in plan_jobs:
        if j.get('group') != 'evolve' or j.get('target') != target or j.get('round') != rnd:
            continue
        val = None
        hits = sorted(glob.glob(str(Path(pipe) / 'jobs' / j['id'] / 'structure' / '*' / 'result.json')))
        if hits and glob.glob(str(Path(pipe) / 'jobs' / j['id'] / 'structure' / '*' / 'predictions.npz')):
            try:
                val = float(json.loads(Path(hits[-1]).read_text())['cgcnn']['val']['mae'])
            except (OSError, KeyError, ValueError):
                val = None
        out.append({'id': j['id'], 'gen': j['gen'], 'parent': j.get('parent'), 'genes': j['genes'], 'val_mae': val})
    return out


def propose(arch, target, gen, population, children, *, space=SPACE, base=DEFAULT, seed_mode='random', rnd=None):
    """Settings for generation `gen`: [(genes, parent id or None)]. Deterministic given the archive.
    Generation 0 is `base` plus random settings (seed_mode 'random') or plus mutants of `base` (seed_mode 'mutants', used to continue
    from an earlier winner)."""
    seen = {gkey(a['genes']) for a in arch}
    out, rng = [], rng_for(target, gen, '' if rnd is None else rnd)
    if gen == 0:
        cands = [(dict(base), None)]
        tries = 0
        while len(cands) < population and tries < 1000:
            tries += 1
            g = mutate(base, rng, space=space) if seed_mode == 'mutants' else random_genes(rng, space)
            if gkey(g) not in seen and gkey(g) not in {gkey(c[0]) for c in cands}:
                cands.append((g, None))
        return cands
    ranked = sorted((a for a in arch if a['val_mae'] is not None), key=lambda a: a['val_mae'])
    parents = ranked[:2]
    if not parents:
        return []
    tries = 0
    while len(out) < children and tries < 400:
        tries += 1
        par = parents[len(out) % len(parents)]
        g = mutate(par['genes'], rng, space=space)
        if gkey(g) in seen or gkey(g) in {gkey(c[0]) for c in out}:
            continue
        out.append((g, par['id']))
    return out


def search_job(target, gen, idx, genes, parent, *, device, epochs, limit, threads, smoke, rnd=None, split='system'):
    jid = f'{prefix(rnd)}_{target}_g{gen}_c{idx}'
    argv = ['-m', 'gpu.structure.train_structure', '--device', device, '--target', target, '--split', split, '--epochs', str(epochs),
            '--seed', str(SEARCH_SEED), '--runs-root', '{JOB_DIR}', '--tag', jid, *genes_args(genes), *limit]
    if threads:
        argv += ['--threads', str(threads)]
    return {'id': jid, 'queue': 'gpu', 'group': 'evolve', 'target': target, 'split': split, 'gen': gen, 'parent': parent, 'genes': genes, 'argv': argv, 'round': rnd,
            'expect': ['structure/*/result.json', 'structure/*/predictions.npz'], 'resumable': True, 'resume_glob': 'structure/*/run_config.json',
            'timeout_s': 3 * 3600, 'est_minutes': 1 if smoke else 5}


def final_job(target, seed, genes, *, device, epochs, limit, threads, smoke, rnd=None, split='system', winner_id=None):
    jid = f'evof_{target}_s{seed}' if rnd is None else f'evof_{rnd}_{target}_s{seed}'
    argv = ['-m', 'gpu.structure.train_structure', '--device', device, '--target', target, '--split', split, '--epochs', str(epochs),
            '--seed', str(seed), '--runs-root', '{JOB_DIR}', '--tag', jid, *genes_args(genes), *limit]
    if threads:
        argv += ['--threads', str(threads)]
    return {'id': jid, 'queue': 'gpu', 'group': 'evolve_final', 'target': target, 'split': split, 'genes': genes, 'argv': argv, 'round': rnd, 'winner_id': winner_id,
            'expect': ['structure/*/result.json', 'structure/*/predictions.npz'], 'resumable': True, 'resume_glob': 'structure/*/run_config.json',
            'timeout_s': 6 * 3600, 'est_minutes': 1 if smoke else 15}


def best_of(arch):
    done = [a for a in arch if a['val_mae'] is not None]
    return min(done, key=lambda a: a['val_mae']) if done else None


def write_lineage(pipe, plan_jobs, targets, rnd=None, bases=None):
    """lineage.json (lineage_<round>.json for later rounds): every individual ever trained, never pruned.
    bases: {target: the settings this round started from} (default: train_structure's defaults)."""
    out = {}
    for t in targets:
        arch = archive(pipe, plan_jobs, t, rnd)
        base = (bases or {}).get(t, DEFAULT)
        default = next((a for a in arch if gkey(a['genes']) == gkey(base)), None)
        best = best_of(arch)
        out[t] = {'individuals': arch, 'default_id': default['id'] if default else None, 'best_id': best['id'] if best else None,
                  'best_by_generation': [min([a['val_mae'] for a in arch if a['gen'] <= g and a['val_mae'] is not None] or [None]) for g in range(max([a['gen'] for a in arch] or [-1]) + 1)]}
    text = json.dumps(out, indent=1)
    name = 'lineage.json' if rnd is None else f'lineage_{rnd}.json'
    tmp = Path(pipe) / (name + '.tmp')
    tmp.write_text(text)
    tmp.replace(Path(pipe) / name)
    return out


def extend_plan(pipe, plan, new_jobs):
    """Adds jobs (never replaces or removes) to plan.json. Returns the jobs that were actually new."""
    from gpu.pipeline.orchestrator import atomic_write
    have = {j['id'] for j in plan['jobs']}
    added = [j for j in new_jobs if j['id'] not in have]
    if added:
        plan['jobs'] += added
        atomic_write(Path(pipe) / 'plan.json', json.dumps(plan, indent=1))
    return added


def _avg_minutes(pipe, ids):
    try:
        st = json.loads((Path(pipe) / 'state.json').read_text())['jobs']
    except (OSError, ValueError):
        return None
    secs = [st[i].get('seconds', 0) for i in ids if i in st and st[i].get('status') == 'done' and st[i].get('seconds')]
    return (sum(secs) / len(secs) / 60.0) if secs else None


def stop_requested(pipe):
    return any((Path(pipe) / n).exists() for n in ('STOP', 'STOP_NOW'))


def run_cycle(pipe, plan, *, make_orch, targets, cfg, elapsed_hours, budget_hours, say=print):
    """Generations, then finalists. make_orch(plan, hours_left) returns an Orchestrator; each call runs whatever is still pending, so a
    resumed cycle picks up exactly where it stopped. Returns 0 (all well), 1 (some jobs failed) or 2 (stopped, or out of time)."""
    pipe = Path(pipe)
    worst = 0
    slots = max(1, cfg.get('gpu_slots', 1))
    rnd, split = cfg.get('round'), cfg.get('split', 'system')
    space, seed_mode = cfg.get('space') or SPACE, cfg.get('seed_mode', 'random')
    bases = cfg.get('bases') or {}
    base_of = lambda t: bases.get(t, DEFAULT)  # noqa: E731
    final_seeds = cfg.get('final_seeds') or FINAL_SEEDS

    def left():
        return None if budget_hours is None else max(0.0, budget_hours - elapsed_hours())

    def run_pending():
        nonlocal worst
        rc = make_orch(plan, left()).run()
        worst = max(worst, rc)
        _lineage()
        return rc

    def lim(t):
        return cfg['limits'].get(t, [])

    def _lineage():
        write_lineage(pipe, plan['jobs'], targets, rnd, {t: base_of(t) for t in targets})

    def has_finals(t):
        return any(j.get('group') == 'evolve_final' and j.get('round') == rnd and j.get('target') == t for j in plan['jobs'])

    search_targets = [t for t in targets if not has_finals(t)]
    if len(search_targets) < len(targets):
        # The finalists of a round are chosen once. A later session (resume, --retry-failed, a fresh time budget) must not add
        # individuals afterwards: the report would then name a "best individual" that the finalist numbers do not belong to.
        say('finalists already chosen for: ' + ', '.join(t for t in targets if has_finals(t)) + '; no further generations for them')
    for g in range(cfg['generations']):
        if not search_targets:
            break
        if g > 0 and budget_hours is not None:  # would one more generation plus the finalists still fit?
            est = 0.0
            for t in search_targets:
                ids = [a['id'] for a in archive(pipe, plan['jobs'], t, rnd)]
                m = _avg_minutes(pipe, ids)
                if m is None:
                    continue
                fin = len(final_seeds[t]) * m * cfg['final_epochs'] / max(1, cfg['search_epochs'])
                est += (cfg['children'] * m + fin) / 60.0 / slots
            if est and elapsed_hours() + est > budget_hours:
                say(f'time budget: another generation plus the finalists would need about {est:.1f} h and {left():.1f} h are left; going straight to the finalists')
                break
        new = []
        for t in search_targets:
            arch = archive(pipe, plan['jobs'], t, rnd)
            if any(a['gen'] == g for a in arch):
                continue  # this generation was created before an interruption; its pending jobs run below
            for i, (genes, parent) in enumerate(propose(arch, t, g, cfg['population'], cfg['children'], space=space, base=base_of(t), seed_mode=seed_mode, rnd=rnd)):
                new.append(search_job(t, g, i, genes, parent, device=cfg['device'], epochs=cfg['search_epochs'], limit=lim(t), threads=cfg['threads'],
                                      smoke=cfg['smoke'], rnd=rnd, split=split))
        added = extend_plan(pipe, plan, new)
        say(f'{"" if rnd is None else rnd + " "}generation {g}: {len(added)} new individuals' + ''.join(f' ({t}: {sum(1 for j in added if j["target"] == t)})' for t in targets))
        if run_pending() == 2 or stop_requested(pipe):
            return 2
    finals = []
    for t in search_targets:
        best = best_of(archive(pipe, plan['jobs'], t, rnd))
        if best is None:
            say(f'{t}: no finished individual; no finalist')
        elif gkey(best['genes']) == gkey(base_of(t)):
            say(f'{t}: the starting settings were still the best by validation error, so there is nothing new to retrain')
        else:
            say(f'{t}: best by validation error is {best["id"]} ({describe_genes(best["genes"], base_of(t))}); retraining on fresh seeds')
            seeds = [21, 22] if cfg['smoke'] else final_seeds[t]
            for sd in seeds:
                finals.append(final_job(t, sd, best['genes'], device=cfg['device'], epochs=cfg['final_epochs'], limit=lim(t), threads=cfg['threads'],
                                        smoke=cfg['smoke'], rnd=rnd, split=split, winner_id=best['id']))
    added = extend_plan(pipe, plan, finals)
    if added or any(j['group'] == 'evolve_final' and j.get('round') == rnd for j in plan['jobs']):
        run_pending()
    _lineage()
    return 2 if (worst == 2 or stop_requested(pipe)) else worst


def forecast_hours(pipe, plan, cfg, static_jobs=()):
    """Rough wall-clock hours: the not-yet-done GPU jobs in static_jobs plus the search round and its finalists, priced with the minutes
    this machine actually took for comparable jobs earlier in the folder (falls back to guesses). Two parallel GPU jobs are assumed
    to be 1.5x as fast as one, not 2x. A planning guess; the report records the real times."""
    import fnmatch
    from gpu.pipeline.plan import STRUCT_EST_MIN
    try:
        st = json.loads((Path(pipe) / 'state.json').read_text())['jobs']
    except (OSError, ValueError):
        st = {}

    def epochs_of(j):
        a = j.get('argv', [])
        return float(a[a.index('--epochs') + 1]) if '--epochs' in a else 100.0

    def per100(pattern):
        vals = [st[j['id']]['seconds'] / 60.0 * 100.0 / epochs_of(j) for j in plan['jobs']
                if fnmatch.fnmatch(j['id'], pattern) and st.get(j['id'], {}).get('status') == 'done' and st[j['id']].get('seconds')]
        return sum(vals) / len(vals) if vals else None

    targets = sorted({j.get('target') for j in static_jobs if j.get('target')} | set(cfg.get('targets', [])))
    m100 = {t: per100(f'struct_{t}_s[0-9]*') or float(STRUCT_EST_MIN.get(t, 15)) for t in targets}
    parts = {t: 0.0 for t in targets}
    for j in static_jobs:
        if j['queue'] == 'gpu' and st.get(j['id'], {}).get('status') != 'done' and j.get('target') in m100:
            parts[j['target']] += m100[j['target']] * epochs_of(j) / 100.0 / 60.0
    seeds = cfg.get('final_seeds') or FINAL_SEEDS
    for t in cfg.get('targets', []):
        mb = (per100(cfg['compare_to'].format(t=t)) if cfg.get('compare_to') and cfg.get('round') else None) or m100[t]
        n_search = cfg['population'] + (cfg['generations'] - 1) * cfg['children']
        parts[t] += (n_search * mb * cfg['search_epochs'] + len(seeds.get(t, [0, 0])) * mb * cfg['final_epochs']) / 100.0 / 60.0
    speed = 1.0 if cfg.get('gpu_slots', 1) <= 1 else 1.5
    return sum(parts.values()) / speed, parts, speed
