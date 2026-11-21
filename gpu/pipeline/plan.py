"""What the pipeline runs. Two profiles:

  smoke     a rehearsal of the whole thing in a few minutes (tiny grids, 800 crystals): proves the machinery works
  standard  the real run: a 288-setting sweep for every tabular family (5 seeds, grouped cross-validation), strong CPU
            baselines for each, and crystal-graph networks for four JARVIS-DFT properties with several seeds each

The order puts the most useful and cheapest GPU work first, so a run cut short by the time budget still leaves a useful
report: crystal bulk and shear modulus (3 seeds each), then every tabular family, then formation energy and band gap.
CPU jobs (the baselines) run beside the GPU queue. All time figures here are rough guesses used only for display;
the report lists what each job actually took.
"""
from gpu.train_tabular import BASE, FAMILY_DEFAULTS

TAB_ORDER = ['alloys', 'polymers', 'electrolytes', 'insulators', 'concrete', 'chemistry']
TAB_EST_MIN = {'alloys': 10, 'polymers': 12, 'electrolytes': 12, 'insulators': 15, 'concrete': 8, 'chemistry': 60}
STRUCT_SEEDS = {'bulk_modulus_kv': [11, 12, 13], 'shear_modulus_gv': [11, 12, 13], 'formation_energy_peratom': [11, 12], 'optb88vdw_bandgap': [11, 12]}
STRUCT_ORDER_EARLY = ['bulk_modulus_kv', 'shear_modulus_gv']
STRUCT_ORDER_LATE = ['formation_energy_peratom', 'optb88vdw_bandgap']
STRUCT_EST_MIN = {'bulk_modulus_kv': 12, 'shear_modulus_gv': 12, 'formation_energy_peratom': 35, 'optb88vdw_bandgap': 35}
TREES_EST_MIN = {'bulk_modulus_kv': 3, 'shear_modulus_gv': 3, 'formation_energy_peratom': 15, 'optb88vdw_bandgap': 15}
BIG_TARGETS = ('formation_energy_peratom', 'optb88vdw_bandgap')  # 94,000 crystals; the graphs need several GB of memory


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


def build_plan(profile='standard', *, device='cuda', families=None, targets=None, tabular=True, structure=True, ram_gb=None,
               cpu_threads=2, members=None):
    if profile not in ('smoke', 'standard'):
        raise SystemExit(f'unknown profile {profile!r}; choose smoke or standard')
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
                                  '--seed', str(s), '--runs-root', '{JOB_DIR}', '--tag', f'{t}_s{s}', *lim],
                         'expect': ['structure/*/result.json', 'structure/*/predictions.npz'], 'resumable': True, 'resume_glob': 'structure/*/run_config.json',
                         'timeout_s': (6 if t in BIG_TARGETS else 3) * 3600, 'est_minutes': 1 if smoke else STRUCT_EST_MIN[t]})
        jobs.append({'id': f'trees_{t}', 'queue': 'cpu', 'group': 'baseline', 'target': t,
                     'argv': ['-m', 'gpu.structure.baseline_trees', '--target', t, '--threads', str(cpu_threads), '--output', '{JOB_DIR}/trees.json', *lim],
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
    for title, js in (('GPU queue (one job at a time)', gpu), ('CPU queue (beside the GPU queue)', cpu)):
        lines.append(f'\n{title}: {len(js)} jobs, rough guess {sum(j.get("est_minutes", 0) for j in js) / 60:.1f} h')
        for j in js:
            extra = f', {len(plan["grids"][j["family"]])} settings' if j['id'].startswith('tab_') else ''
            lines.append(f'  {j["id"]:34s} ~{j.get("est_minutes", 0):4.0f} min{extra}')
    lines += [''] + [f'note: {n}' for n in plan['notes']]
    return '\n'.join(lines)
