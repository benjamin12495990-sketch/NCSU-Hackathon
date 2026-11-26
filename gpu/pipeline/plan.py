"""What the pipeline runs. Two profiles:

  smoke     a rehearsal of the whole thing in a few minutes (tiny grids, 800 crystals): proves the machinery works
  standard  the real run: a 288-setting sweep for every tabular family (5 seeds, grouped cross-validation), strong CPU
            baselines for each, and crystal-graph networks for four JARVIS-DFT properties with several seeds each

The order puts the most useful and cheapest GPU work first, so a run cut short by the time budget still leaves a useful
report: crystal bulk and shear modulus (3 seeds each), then every tabular family, then formation energy and band gap.
CPU jobs (the baselines) run beside the GPU queue. Time figures are the minutes measured on one RTX 5070 Laptop GPU and are used only for display;
the report lists what each job actually took.
"""
import glob
import json
import time
from pathlib import Path

from gpu.train_tabular import BASE, FAMILY_DEFAULTS

TAB_ORDER = ['alloys', 'polymers', 'electrolytes', 'insulators', 'concrete', 'chemistry', 'cleaning_pcmc', 'cleaning_tension', 'corrosion']
# minutes MEASURED on an RTX 5070 Laptop GPU (the first overnight run); other machines will differ
TAB_EST_MIN = {'alloys': 6, 'polymers': 5, 'electrolytes': 14, 'insulators': 6, 'concrete': 2, 'chemistry': 10,
               'cleaning_pcmc': 12, 'cleaning_tension': 10, 'corrosion': 15}  # the last three are guesses (never run on the GPU yet)
STRUCT_SEEDS = {'bulk_modulus_kv': [11, 12, 13], 'shear_modulus_gv': [11, 12, 13], 'formation_energy_peratom': [11, 12], 'optb88vdw_bandgap': [11, 12]}
STRUCT_ORDER_EARLY = ['bulk_modulus_kv', 'shear_modulus_gv']
STRUCT_ORDER_LATE = ['formation_energy_peratom', 'optb88vdw_bandgap']
STRUCT_EST_MIN = {'bulk_modulus_kv': 9, 'shear_modulus_gv': 8, 'formation_energy_peratom': 20, 'optb88vdw_bandgap': 18}
TREES_EST_MIN = {'bulk_modulus_kv': 3, 'shear_modulus_gv': 2, 'formation_energy_peratom': 16, 'optb88vdw_bandgap': 19}
BIG_TARGETS = ('formation_energy_peratom', 'optb88vdw_bandgap')  # 94,000 crystals; the graphs need several GB of memory
# Every crystal job a planner writes from now on states its neighbour rule (docs/DECISIONS.md DEC-020). A crystal job WITHOUT the flag can
# therefore only come from a plan written before the fix, and pin_original_rule() gives it the rule it was planned under.
TIES_ALL = ['--neighbor-ties', 'all']


def tabular_grid(family, profile):
    base = {**BASE, **FAMILY_DEFAULTS[family]}
    if profile == 'smoke':
        return [{'hidden': [16, 8], 'epochs': 10, 'lr': 0.004, 'decay': 1e-3, 'loss': 'huber'},
                {'hidden': [32, 16], 'epochs': 10, 'lr': 0.002, 'decay': 1e-2, 'loss': 'squared'}]
    e = base['epochs']
    epochs = sorted({max(10, e // 2), e, e * 2})
    grid = []
    for hidden in ([32, 16], [64, 32], [128, 64], [256, 128]):
        for ep in epochs:
            for lr in (0.001, 0.002, 0.004):
                for decay in (1e-4, 1e-3, 1e-2, 3e-2):
                    for loss in ('squared', 'huber'):
                        grid.append({'hidden': hidden, 'epochs': ep, 'lr': lr, 'decay': decay, 'loss': loss})
    return grid


def limit_used(existing, target):
    """The --limit an earlier plan used for a target's crystal jobs (so new jobs use the very same records), or None if it had none."""
    for j in (existing or []):
        if j.get('target') == target and '--limit' in j['argv']:
            return j['argv'][j['argv'].index('--limit') + 1]
    return None


def limit_for(existing, target, smoke, small_ram):
    """--limit arguments for a target's crystal jobs: the same records an earlier plan used, else the profile's rule."""
    if existing is not None and any(j.get('target') == target for j in existing):
        lim = limit_used(existing, target)
        return ['--limit', lim] if lim else []
    if smoke:
        return ['--limit', '800']
    return ['--limit', '40000'] if (target in BIG_TARGETS and small_ram) else []


def holdout_jobs(*, device, targets, smoke, small_ram, cpu_threads, existing):
    """Crystal-network and tree runs on a split that holds out whole chemical systems, plus the stronger structure-feature trees
    on the original formula split (the networks for that split already exist in the pipeline folder being extended)."""
    jobs = []
    for t in targets:
        lim = limit_for(existing, t, smoke, small_ram)
        seeds = [11, 12] if smoke else STRUCT_SEEDS[t]
        env = {'OMP_NUM_THREADS': str(cpu_threads)}
        jobs.append({'id': f'trees_{t}_rich', 'queue': 'cpu', 'group': 'baseline', 'target': t, 'split': 'formula',
                     'argv': ['-m', 'gpu.structure.baseline_trees', '--target', t, '--threads', str(cpu_threads), '--feature-sets', 'rich', *TIES_ALL,
                              '--output', '{JOB_DIR}/trees.json', *lim],
                     'expect': ['trees.json', 'trees_predictions.npz'], 'env': env, 'timeout_s': 4 * 3600, 'est_minutes': 1 if smoke else (30 if t in BIG_TARGETS else 8)})
        jobs.append({'id': f'trees_{t}_sys', 'queue': 'cpu', 'group': 'baseline', 'target': t, 'split': 'system',
                     'argv': ['-m', 'gpu.structure.baseline_trees', '--target', t, '--threads', str(cpu_threads), '--split', 'system',
                              '--feature-sets', 'composition,comp+cell,rich', *TIES_ALL, '--output', '{JOB_DIR}/trees.json', *lim],
                     'expect': ['trees.json', 'trees_predictions.npz'], 'env': env, 'timeout_s': 5 * 3600, 'est_minutes': 1 if smoke else (40 if t in BIG_TARGETS else 12)})
        for sd in seeds:
            jobs.append({'id': f'struct_{t}_sys_s{sd}', 'queue': 'gpu', 'group': 'structure', 'target': t, 'split': 'system',
                         'argv': ['-m', 'gpu.structure.train_structure', '--device', device, '--target', t, '--split', 'system',
                                  '--epochs', '3' if smoke else '100', '--seed', str(sd), '--runs-root', '{JOB_DIR}', '--tag', f'{t}_sys_s{sd}', *TIES_ALL, *lim],
                         'expect': ['structure/*/result.json', 'structure/*/predictions.npz'], 'resumable': True, 'resume_glob': 'structure/*/run_config.json',
                         'timeout_s': (6 if t in BIG_TARGETS else 3) * 3600, 'est_minutes': 1 if smoke else STRUCT_EST_MIN[t]})
    return jobs


def build_plan(profile='standard', *, device='cuda', families=None, targets=None, tabular=True, structure=True, ram_gb=None,
               cpu_threads=2, members=None, existing=None, pipe=None):
    if profile in ('cycle3', 'cycle3-smoke'):
        smoke = profile == 'cycle3-smoke'
        tab_jobs, grids = new_family_jobs(device=device, smoke=smoke, cpu_threads=cpu_threads, existing=existing, members=members)
        jobs = tab_jobs + cycle3_jobs(device=device, smoke=smoke, small_ram=ram_gb is not None and ram_gb < 24, cpu_threads=cpu_threads, existing=existing)
        return {'profile': profile, 'device': device, 'members': members or (2 if smoke else 5), 'jobs': jobs, 'grids': grids,
                'notes': ['Cycle 3: network sweeps for the table families this folder had not trained yet (surfactants, corrosion), the leave-out-elements split (with and without element-property atom inputs), bulk and shear moduli fine-tuned from the '
                          'formation-energy network, and a wider formation-energy search.' + (' Smoke sizes: the numbers mean nothing.' if smoke else '')]}
    if profile in ('tiefix', 'tiefix-smoke'):
        smoke = profile == 'tiefix-smoke'
        skipped = []
        jobs = tiefix_jobs(device=device, smoke=smoke, cpu_threads=cpu_threads, existing=existing, targets=targets, pipe=pipe, skipped=skipped)
        return {'profile': profile, 'device': device, 'members': 0, 'jobs': jobs, 'grids': {}, 'skipped': skipped,
                'notes': ['Neighbour-rule check: the held-out-systems networks and trees of this folder trained again, same seeds, settings and crystals, with the '
                          'invariant neighbour rule (gpu/structure/graph.py), plus a measurement of how much one old and one new network change their '
                          'predictions when a crystal is only rewritten.' + (' Smoke sizes: the numbers mean nothing.' if smoke else '')]}
    if profile in ('holdout', 'holdout-smoke'):
        smoke = profile == 'holdout-smoke'
        targets = targets or (['bulk_modulus_kv'] if smoke else STRUCT_ORDER_EARLY + STRUCT_ORDER_LATE)
        small_ram = ram_gb is not None and ram_gb < 24
        jobs = holdout_jobs(device=device, targets=targets, smoke=smoke, small_ram=small_ram, cpu_threads=cpu_threads, existing=existing)
        return {'profile': profile, 'device': device, 'members': 0, 'jobs': jobs, 'grids': {},
                'notes': ['Holdout profile: whole chemical systems are held out of training, and structure-feature trees are added on both splits.'
                          + (' Smoke sizes: the numbers mean nothing.' if smoke else '')]}
    if profile not in ('smoke', 'standard'):
        raise SystemExit(f'unknown profile {profile!r}; choose smoke, standard, holdout, cycle3, tiefix or their -smoke versions')
    smoke = profile == 'smoke'
    families = families or (['alloys', 'electrolytes', 'concrete'] if smoke else TAB_ORDER)
    targets = targets or (['bulk_modulus_kv'] if smoke else STRUCT_ORDER_EARLY + STRUCT_ORDER_LATE)
    members = members or (2 if smoke else 5)
    notes, jobs, grids = [], [], {}
    limit = []
    if smoke:
        limit = ['--limit', '800']
        notes.append('Smoke profile: tiny settings, only for checking that the machinery works. Its numbers mean nothing.')
    small_ram = ram_gb is not None and ram_gb < 24

    def tab(fam):
        grids[fam] = tabular_grid(fam, profile)
        mode = 'fixed' if fam == 'concrete' else 'cv'
        jobs.append({'id': f'tab_{fam}', 'queue': 'gpu', 'group': 'tabular', 'family': fam,
                     'argv': ['-m', 'gpu.train_tabular', '--family', fam, '--mode', mode, '--device', device, '--members', str(members),
                              '--grid', '{PIPE_DIR}/grids/' + fam + '.json', '--runs-root', '{JOB_DIR}', '--tag', 'sweep', '--checkpoint-every', '25'],
                     'expect': ['*/*/result.json', '*/*/predictions.npz'], 'resumable': True, 'resume_glob': '*/*/run_config.json',
                     'timeout_s': 5 * 3600, 'est_minutes': 2 if smoke else TAB_EST_MIN[fam]})
        jobs.append({'id': f'base_{fam}', 'queue': 'cpu', 'group': 'baseline', 'family': fam,
                     'argv': ['-m', 'gpu.baselines_tabular', '--family', fam, '--threads', str(cpu_threads), '--output', '{JOB_DIR}/baselines.json'],
                     'expect': ['baselines.json', 'baselines_predictions.npz'], 'env': {'OMP_NUM_THREADS': str(cpu_threads)},
                     'timeout_s': 3 * 3600, 'est_minutes': 1 if smoke else 6})

    def struct(t):
        lim = ['--limit', '40000'] if (t in BIG_TARGETS and small_ram and not smoke) else limit
        if lim and not smoke:
            notes.append(f'{t}: only {lim[1]} randomly chosen crystals are used (this computer reports under 24 GB of memory; the full set needs several GB).')
        seeds = [11, 12] if smoke else STRUCT_SEEDS[t]
        for s in seeds:
            jobs.append({'id': f'struct_{t}_s{s}', 'queue': 'gpu', 'group': 'structure', 'target': t,
                         'argv': ['-m', 'gpu.structure.train_structure', '--device', device, '--target', t, '--epochs', '3' if smoke else '100',
                                  '--seed', str(s), '--runs-root', '{JOB_DIR}', '--tag', f'{t}_s{s}', *TIES_ALL, *lim],
                         'expect': ['structure/*/result.json', 'structure/*/predictions.npz'], 'resumable': True, 'resume_glob': 'structure/*/run_config.json',
                         'timeout_s': (6 if t in BIG_TARGETS else 3) * 3600, 'est_minutes': 1 if smoke else STRUCT_EST_MIN[t]})
        jobs.append({'id': f'trees_{t}', 'queue': 'cpu', 'group': 'baseline', 'target': t,
                     'argv': ['-m', 'gpu.structure.baseline_trees', '--target', t, '--threads', str(cpu_threads), *TIES_ALL, '--output', '{JOB_DIR}/trees.json', *lim],
                     'expect': ['trees.json', 'trees_predictions.npz'], 'env': {'OMP_NUM_THREADS': str(cpu_threads)},
                     'timeout_s': 3 * 3600, 'est_minutes': 1 if smoke else TREES_EST_MIN[t]})

    early = [t for t in targets if t in STRUCT_ORDER_EARLY]
    late = [t for t in targets if t not in STRUCT_ORDER_EARLY]
    if structure:
        for t in early:
            struct(t)
    if tabular:
        for f in families:
            tab(f)
    if structure:
        for t in late:
            struct(t)
    if not jobs:
        raise SystemExit('nothing to run: both tabular and structure are switched off')
    return {'profile': profile, 'device': device, 'members': members, 'jobs': jobs, 'grids': grids, 'notes': notes}


def describe(plan):
    lines = [f'profile {plan["profile"]}, device {plan["device"]}, {plan["members"]} seeds per setting']
    gpu = [j for j in plan['jobs'] if j['queue'] == 'gpu']
    cpu = [j for j in plan['jobs'] if j['queue'] != 'gpu']
    for title, js in (('GPU queue', gpu), ('CPU queue (beside the GPU queue)', cpu)):
        lines.append(f'\n{title}: {len(js)} jobs, rough guess {sum(j.get("est_minutes", 0) for j in js) / 60:.1f} h')
        for j in js:
            extra = f', {len(plan["grids"][j["family"]])} settings' if j['id'].startswith('tab_') else ''
            lines.append(f'  {j["id"]:34s} ~{j.get("est_minutes", 0):4.0f} min{extra}')
    lines += [''] + [f'note: {n}' for n in plan['notes']]
    return '\n'.join(lines)


# ---- cycle 3: leave-out-elements, fine-tuning from formation energy, and a wider formation-energy search ---------------------------
PRETRAIN = 'formation_energy_peratom'
FT_TARGETS = ('bulk_modulus_kv', 'shear_modulus_gv')
SPACE_WIDE = {'atom_features': [64, 96, 128, 160, 192, 256], 'conv': [3, 4, 5, 6, 7, 8], 'hidden': [128, 192, 256, 384],
              'lr': [1e-3, 2e-3, 4e-3, 6e-3, 8e-3], 'weight_decay': [0.0, 1e-5, 1e-4], 'batch': [32, 64, 128]}


def _struct_job(jid, t, split, seed, *, device, epochs, lim, smoke, extra=(), after=None, tag=None, est=None, group='structure'):
    if '--init-from' not in extra and '--neighbor-ties' not in extra:   # a fine-tuned run takes the rule of the run it starts from
        extra = [*extra, *TIES_ALL]
    job = {'id': jid, 'queue': 'gpu', 'group': group, 'target': t, 'split': split,
           'argv': ['-m', 'gpu.structure.train_structure', '--device', device, '--target', t, '--split', split, '--epochs', str(epochs),
                    '--seed', str(seed), '--runs-root', '{JOB_DIR}', '--tag', tag or jid, *extra, *lim],
           'expect': ['structure/*/result.json', 'structure/*/predictions.npz'], 'resumable': True, 'resume_glob': 'structure/*/run_config.json',
           'timeout_s': (8 if t in BIG_TARGETS else 4) * 3600, 'est_minutes': 1 if smoke else (est or STRUCT_EST_MIN[t])}
    if after:
        job['after'] = list(after)
    return job


def new_family_jobs(*, device, smoke, cpu_threads, existing, members):
    """Sweep + baseline jobs for every table family the folder has never trained (so one folder covers every simulation)."""
    have = {j['id'] for j in (existing or [])}
    todo = [f for f in TAB_ORDER if f'tab_{f}' not in have]
    if smoke:
        todo = [f for f in todo if f in ('cleaning_tension', 'corrosion')][:2] or todo[:1]
    if not todo:
        return [], {}
    tab = build_plan('smoke' if smoke else 'standard', device=device, families=todo, structure=False, cpu_threads=cpu_threads, members=members)
    return tab['jobs'], tab['grids']


def cycle3_jobs(*, device, smoke, small_ram, cpu_threads, existing):
    """Static jobs of cycle 3 (the search round is added by gpu/pipeline/evolve.py)."""
    targets = ['bulk_modulus_kv'] if smoke else STRUCT_ORDER_EARLY + STRUCT_ORDER_LATE
    if smoke and PRETRAIN not in targets:
        targets = targets + [PRETRAIN]
    epochs = 3 if smoke else 100
    seeds_e = [11, 12]
    jobs = []
    env = {'OMP_NUM_THREADS': str(cpu_threads)}
    for t in targets:
        lim = limit_for(existing, t, smoke, small_ram)
        jobs.append({'id': f'trees_{t}_elem', 'queue': 'cpu', 'group': 'baseline', 'target': t, 'split': 'element',
                     'argv': ['-m', 'gpu.structure.baseline_trees', '--target', t, '--threads', str(cpu_threads), '--split', 'element',
                              '--feature-sets', 'composition,comp+cell,rich', *TIES_ALL, '--output', '{JOB_DIR}/trees.json', *lim],
                     'expect': ['trees.json', 'trees_predictions.npz'], 'env': env, 'timeout_s': 5 * 3600, 'est_minutes': 1 if smoke else TREES_EST_MIN[t] + 5})
        for sd in seeds_e:  # the learned element vectors (default): blind to elements never seen in training
            jobs.append(_struct_job(f'struct_{t}_elem_s{sd}', t, 'element', sd, device=device, epochs=epochs, lim=lim, smoke=smoke))
        for sd in seeds_e:  # element properties as the atom input: the fix
            jobs.append(_struct_job(f'struct_{t}_elemP_s{sd}', t, 'element', sd, device=device, epochs=epochs, lim=lim, smoke=smoke,
                                    extra=['--atom-input', 'both']))
    sources = {'formula': f'struct_{PRETRAIN}_s11', 'system': f'struct_{PRETRAIN}_sys_s11', 'element': f'struct_{PRETRAIN}_elemP_s11'}
    have = {j['id'] for j in (existing or [])} | {j['id'] for j in jobs}
    for t in FT_TARGETS:
        if smoke and t not in targets:
            continue  # rehearsal: only the properties it also trains from scratch
        lim = limit_for(existing, t, smoke, small_ram)
        for split, seeds, tag in (('formula', [11, 12, 13], 'ft'), ('system', [11, 12, 13], 'ft_sys'), ('element', [11, 12], 'ft_elemP')):
            src = sources[split]
            if src not in have:
                continue  # no pretrained source for this split in this folder
            for sd in ([11, 12] if smoke else seeds):
                jobs.append(_struct_job(f'{tag}_{t}_s{sd}', t, split, sd, device=device, epochs=epochs, lim=lim, smoke=smoke,
                                        extra=['--init-from', '{RUN_OF:' + src + '}', '--lr', '1e-3'], after=[src], group='finetune'))
    return jobs


# ---- the neighbour-rule fix: the same held-out-systems runs again under the invariant rule (docs/DECISIONS.md DEC-020, LES-017) ----
def _argv_value(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


CRYSTAL_MODULES = ('gpu.structure.train_structure', 'gpu.structure.baseline_trees')


def pin_original_rule(plan):
    """For a plan written before the neighbour-rule fix: give every crystal job that does not state a rule the rule it was planned
    under ('first'), so a job that starts after the fix (never started, skipped, retried from scratch, or a new generation of an
    unfinished search) is trained the same way as its siblings. Jobs with --init-from are left alone: they take the rule of the run
    they start from. The run identity of an original-rule run does not include the rule, so finished and half-finished runs are
    unaffected. Search rounds without a recorded rule get 'first' too. Returns the ids changed (empty if nothing needed it)."""
    pinned = []
    for j in plan.get('jobs', []):
        argv = j.get('argv', [])
        if not any(m in argv for m in CRYSTAL_MODULES) or '--neighbor-ties' in argv or '--init-from' in argv:
            continue
        j['argv'] = [*argv, '--neighbor-ties', 'first']
        pinned.append(j['id'])
    for c in [plan.get('cycle')] + list(plan.get('cycle_history', [])):
        if c and 'neighbor_ties' not in c:
            c['neighbor_ties'] = 'first'
            pinned.append(f'search round {c.get("round") or "r1"}')
    if pinned:
        plan.setdefault('neighbor_rule_pins', []).append({'when': time.strftime('%Y-%m-%dT%H:%M:%S'), 'rule': 'first', 'jobs': pinned})
    return pinned


def job_rule(pipe, job):
    """The neighbour rule a crystal job's results were made with: what its run (or tree file) recorded if it has started, else what its
    command line says ('first' for a line written before the fix). 'none' for trees without structure features (no rule involved),
    'mixed' if its folder holds runs made under both rules."""
    argv = job.get('argv', [])
    if pipe is not None:
        jd = Path(pipe) / 'jobs' / job['id']
        if 'gpu.structure.baseline_trees' in argv and (jd / 'trees.json').exists():
            try:
                tj = json.loads((jd / 'trees.json').read_text())
                return (tj.get('neighbor_ties') or 'first') if 'rich' in tj.get('feature_sets', {}) else 'none'
            except (OSError, ValueError):
                return 'unknown'
        cfgs = sorted(glob.glob(str(jd / 'structure' / '*' / 'run_config.json')))
        if cfgs:
            rules = set()
            for f in cfgs:
                try:
                    rules.add(json.loads(Path(f).read_text())['config'].get('neighbor_ties', 'first'))
                except (OSError, ValueError, KeyError):
                    rules.add('unknown')
            return rules.pop() if len(rules) == 1 else 'mixed'
    if 'gpu.structure.baseline_trees' in argv and 'rich' not in _argv_value(argv, '--feature-sets', 'composition,comp+cell'):
        return 'none'
    return _argv_value(argv, '--neighbor-ties', 'first')


def tiefix_jobs(*, device, smoke, cpu_threads, existing, targets=None, pipe=None, skipped=None):
    """For every property whose held-out-systems networks (struct_<t>_sys_s<seed>) and trees (trees_<t>_sys) are in the folder AND were made
    with the original rule: the same networks (same seeds, epochs, record limit, default settings) and the same trees again with
    --neighbor-ties all, under new job ids (struct_<t>_sysT_s<seed>, trees_<t>_sysT), and two invariance measurements (one original-rule
    network, one fixed-rule). Existing jobs are not replaced; unfinished ones still run, under the rule they were planned with.
    A property whose held-out-systems runs already use the fixed rule (a folder made after the fix) has nothing to compare: it is left
    out and a sentence saying why goes to `skipped` (a list), so no fixed-rule run is ever labelled 'original'."""
    existing = existing or []
    skipped = skipped if skipped is not None else []
    by_id = {j['id']: j for j in existing}
    jobs = []
    env = {'OMP_NUM_THREADS': str(cpu_threads)}
    for t in STRUCT_ORDER_EARLY + STRUCT_ORDER_LATE:
        if targets and t not in targets:
            continue
        olds = sorted((j for j in existing if j['id'].startswith(f'struct_{t}_sys_s') and j['id'][len(f'struct_{t}_sys_s'):].isdigit()),
                      key=lambda j: int(j['id'].rsplit('_s', 1)[1]))
        if not olds or f'trees_{t}_sys' not in by_id:
            continue
        rules = {o['id']: job_rule(pipe, o) for o in olds}
        tree_rule = job_rule(pipe, by_id[f'trees_{t}_sys'])
        if set(rules.values()) != {'first'} or tree_rule not in ('first', 'none'):
            skipped.append(f'{t}: the held-out-systems runs here were not all made with the original rule ('
                           + ', '.join(f'{k} {v}' for k, v in [*rules.items(), (f'trees_{t}_sys', tree_rule)]) + '), so there is no original set to compare with.')
            continue
        lim = limit_for(existing, t, smoke, False)
        for o in olds:
            sd = int(o['id'].rsplit('_s', 1)[1])
            extra = [x for f in ('--atom-features', '--conv', '--hidden', '--lr', '--weight-decay', '--batch') if f in o['argv'] for x in (f, _argv_value(o['argv'], f))]
            jobs.append(_struct_job(f'struct_{t}_sysT_s{sd}', t, 'system', sd, device=device, epochs=_argv_value(o['argv'], '--epochs', '100'), lim=lim, smoke=smoke,
                                    extra=[*extra, '--neighbor-ties', 'all'], est=int(STRUCT_EST_MIN[t] * 1.3), group='tiefix'))
        feats = _argv_value(by_id[f'trees_{t}_sys']['argv'], '--feature-sets', 'composition,comp+cell,rich')
        jobs.append({'id': f'trees_{t}_sysT', 'queue': 'cpu', 'group': 'baseline', 'target': t, 'split': 'system',
                     'argv': ['-m', 'gpu.structure.baseline_trees', '--target', t, '--threads', str(cpu_threads), '--split', 'system',
                              '--feature-sets', feats, '--neighbor-ties', 'all', '--output', '{JOB_DIR}/trees.json', *lim],
                     'expect': ['trees.json', 'trees_predictions.npz'], 'env': env, 'timeout_s': 5 * 3600, 'est_minutes': 1 if smoke else (40 if t in BIG_TARGETS else 12)})
        first = int(olds[0]['id'].rsplit('_s', 1)[1])
        for rule, src in (('first', f'struct_{t}_sys_s{first}'), ('all', f'struct_{t}_sysT_s{first}')):
            jobs.append({'id': f'inv_{t}_{rule}', 'queue': 'cpu', 'group': 'invariance', 'target': t, 'after': [src],
                         'argv': ['-m', 'gpu.structure.invariance', '--run', '{RUN_OF:' + src + '}', '--output', '{JOB_DIR}/invariance.json', '--n', '40' if smoke else '300'],
                         'expect': ['invariance.json'], 'env': env, 'timeout_s': 2 * 3600, 'est_minutes': 1 if smoke else (6 if t in BIG_TARGETS else 3)})
    return jobs
