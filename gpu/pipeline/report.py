"""Writes REPORT.md, summary.json and share.zip for a pipeline folder.

  python -m gpu.pipeline.report --pipeline gpu/runs/pipeline/<folder>

Safe to run at any time, also while the pipeline is still going or after it was interrupted: it reports what finished.
An earlier REPORT.md is renamed, not overwritten.
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

from gpu.pipeline.analyze import analyze_all
from gpu.pipeline.orchestrator import atomic_write, telemetry_summary

TAB_NAMES = {'alloys': 'Alloys (hardness)', 'polymers': 'Polymers (glass transition)', 'electrolytes': 'Solid electrolytes (conductivity)',
             'insulators': 'Insulators (dielectric constant, calculated)', 'concrete': 'Concrete (strength, ages 14-28 d from 1-7 d)',
             'chemistry': 'Chemistry (water solubility)'}
STRUCT_NAMES = {'bulk_modulus_kv': 'Bulk modulus (log10 GPa)', 'shear_modulus_gv': 'Shear modulus (log10 GPa)',
                'formation_energy_peratom': 'Formation energy (eV/atom)', 'optb88vdw_bandgap': 'Band gap (eV)'}


def g(x, n=4):
    return '-' if x is None else f'{x:.{n}g}'


def ci(d):
    return f'{d["diff"]:+.3g} [{d["ci95"][0]:+.3g}, {d["ci95"][1]:+.3g}]'


def single_text(f):
    """Plain words for 'one network at a time against the trees'."""
    sg = f['single']
    if sg['n_clearly_better'] == sg['n']:
        return f'clearly better in all {sg["n"]} runs'
    if sg['n_clearly_worse'] == sg['n']:
        return f'trees clearly better in all {sg["n"]} runs'
    return f'clearly better in {sg["n_clearly_better"]} of {sg["n"]} runs' + (f', clearly worse in {sg["n_clearly_worse"]}' if sg['n_clearly_worse'] else '')


def single_bucket(f):
    sg = f['single']
    return 'network' if sg['n_clearly_better'] == sg['n'] else 'trees' if sg['n_clearly_worse'] == sg['n'] else 'unclear'


def fairness_lines(f, avg_verdict, avg_name='network average'):
    L = [f'  - **Fairness checks** (the headline compares the {avg_name} with ONE tree model):',
         f'    - One network at a time: MAE {g(f["single"]["mean_mae"])} on average; against the trees it is {single_text(f)}.']
    if 'trees_top3' in f:
        t3 = f['trees_top3']
        L.append(f'    - Average of the 3 best tree models: MAE {g(t3["mae"])}; the {avg_name} against it: {ci(t3["network_average_vs_it"])}, {t3["network_average_vs_it"]["verdict"]}.')
    b = f['blend']
    L.append(f'    - Half {avg_name}, half trees (no fitted weight): MAE {g(b["mae"])}; against the trees {ci(b["vs_trees"])} ({b["vs_trees"]["verdict"]}); '
             f'against the {avg_name} {ci(b["vs_network_average"])} ({b["vs_network_average"]["verdict"]}).')
    m = f['median_ae']
    L.append('    - Median error (the typical row, not pulled by a few bad ones): ' + ', '.join(f'{k} {g(v)}' for k, v in m.items()) + '.')
    if avg_verdict == 'network clearly better' and single_bucket(f) != 'network':
        L.append('    - Reading: if the average wins but single networks do not, the gain comes from averaging several networks, not from one network being better than trees.')
    return L


def read_log_tail(path, n=12):
    try:
        return ''.join(Path(path).read_text(encoding='utf-8', errors='replace').splitlines(True)[-n:]).rstrip()
    except OSError:
        return '(no log)'


def read_telemetry(path):
    rows = []
    if Path(path).exists():
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                def num(k):
                    try:
                        return float(r[k])
                    except (ValueError, KeyError, TypeError):
                        return None
                rows.append([r['time'], num('util_pct'), num('mem_mib'), num('power_w'), num('temp_c'), num('sm_clock_mhz')])
    return rows


def tabular_section(items):
    ok = [t for t in items if t['status'] == 'ok']
    L = ['## Table-based families: network sweep against tuned trees', '',
         'Each network number is the best of a large sweep, **chosen without looking at the rows it is scored on** (cross-fitted); the trees and the '
         'linear models get exactly the same treatment. MAE = typical size of the miss, in the unit shown (lower is better).', '']
    if ok:
        L += ['| Family | Predicts | Rows / groups | Guess the average | Best linear | Best trees | Network: average of seeds | Network: one seed | Average minus trees (95% interval) | Verdict: average | Verdict: one network at a time |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for t in ok:
            m, f = t['methods'], t['fairness']
            L.append(f'| {TAB_NAMES.get(t["family"], t["family"])} | {t["target"]} | {t["rows"]} / {t["groups"]} | {g(m["mean"]["mae"])} | {g(m["linear"]["mae"])} | '
                     f'{g(m["trees"]["mae"])} | {g(m["network"]["mae"])} | {g(f["single"]["mean_mae"])} | {ci(t["network_vs_trees"])} | **{t["network_vs_trees"]["verdict"]}** | {single_text(f)} |')
        L.append('')
    for t in items:
        if t['status'] != 'ok':
            L.append(f'- **{TAB_NAMES.get(t["family"], t["family"])}**: no result ({t["reason"]}).')
    for t in ok:
        L += [f'### {TAB_NAMES.get(t["family"], t["family"])}', '']
        if t['mode'] == 'fixed':
            L.append(f'- **Exploratory.** {t["note"]} The network setting was chosen on {t["selection_set"]}; scores below are on {t["evaluation_set"]}.')
            for k, v in t['methods'].items():
                L.append(f'  - {k}: MAE {g(v["mae"])}, R2 {g(v["r2"], 3)}')
            for k in ('trees', 'linear', 'ridge_ln_age_only'):
                if f'network_vs_{k}' in t:
                    L.append(f'- Network versus {k}: {ci(t[f"network_vs_{k}"])}, {t[f"network_vs_{k}"]["verdict"]}.')
            L += fairness_lines(t['fairness'], t['network_vs_trees']['verdict'])
        else:
            L.append(f'- {t["n_network_settings"]} network settings x {t["n_seeds"]} seeds, {t["n_trees_models"]} tree and {t["n_linear_models"]} linear baseline models, '
                     f'{t["folds"]}-fold grouped cross-validation ({t["rows"]} rows).')
            L.append(f'- Cross-fitted: network MAE {g(t["methods"]["network"]["mae"])} (R2 {g(t["methods"]["network"]["r2"], 3)}), trees {g(t["methods"]["trees"]["mae"])} '
                     f'(R2 {g(t["methods"]["trees"]["r2"], 3)}), linear {g(t["methods"]["linear"]["mae"])}, guess-the-average {g(t["methods"]["mean"]["mae"])}.')
            o = t['optimistic_best_of_all']
            L.append(f'- Without the correction (best setting picked on the same folds it is scored on, **optimistic**): network {g(o["network"])}, trees {g(o["trees"])}, linear {g(o["linear"])}.')
            L.append(f'- One seed instead of the average of {t["n_seeds"]}: network MAE {g(t["network_single_seed"]["mae"])}.')
            L.append(f'- Network versus linear models: {ci(t["network_vs_linear"])}, {t["network_vs_linear"]["verdict"]}.')
            L.append(f'- Settings chosen on each fold: ' + '; '.join(sorted(set(t['chosen']['network']))[:3]) + ('; ...' if len(set(t['chosen']['network'])) > 3 else ''))
            L += fairness_lines(t['fairness'], t['network_vs_trees']['verdict'])
        L.append('')
    return L


def structure_section(items):
    ok = [t for t in items if t['status'] == 'ok']
    L = ['## Crystal-structure networks against trees', '',
         'A crystal-graph network learns from the atom positions. The trees get either the element fractions, or the fractions plus volume per atom and atoms per cell '
         '(two cheap structure facts). Test error is on formulas the models never saw (split by exact formula). All labels are **DFT calculations**, not measurements.', '']
    if ok:
        L += ['| Property | Crystals (train / test) | Guess the average | Linear | Trees: fractions | Trees: + cell | Network: mean of single runs | Network: average of runs | Average minus the stronger trees (95% interval) | Verdict: average | Verdict: single runs | Speed |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for t in ok:
            tr = t['trees']
            v = t['verdicts'].get('pred_comp_cell') or next(iter(t['verdicts'].values()))
            sp = [r['structures_per_second'] for r in t['network_runs']]
            nm = [r['test']['mae'] for r in t['network_runs']]
            key = 'pred_comp_cell' if 'pred_comp_cell' in t['compare']['comparisons'] else next(iter(t['compare']['comparisons']))
            avg = [c for k2, c in t['compare']['comparisons'][key].items() if k2.startswith('network average')]
            avg_mae = avg[0]['network_mae'] if avg else sum(nm) / len(nm)
            f = t.get('fairness')
            L.append(f'| {STRUCT_NAMES.get(t["target"], t["target"])} | {t["split_sizes"]["train"]} / {t["split_sizes"]["test"]} | {g(t["mean_baseline_test_mae"])} | {g(t["ridge_test_mae"])} | '
                     f'{g(tr["composition"]["test"]["mae"])} | {g(tr["comp+cell"]["test"]["mae"])} | {g(sum(nm) / len(nm))} ({len(nm)} runs) | {g(avg_mae)} | {ci(v)} | **{v["verdict"]}** | '
                     f'{single_text(f) if f else "-"} | {sum(sp) / len(sp):.0f} crystals/s |')
        L.append('')
    for t in items:
        if t['status'] != 'ok':
            L.append(f'- **{STRUCT_NAMES.get(t["target"], t["target"])}**: no result ({t["reason"]}).')
    for t in ok:
        L += [f'### {STRUCT_NAMES.get(t["target"], t["target"])}', '', '```'] + t['compare_lines'] + ['```', '']
        if t.get('fairness'):
            L += fairness_lines(t['fairness'], (t['verdicts'].get('pred_comp_cell') or next(iter(t['verdicts'].values())))['verdict'], 'average of runs')
            z = t['fairness'].get('by_zero_gap')
            if z:
                L.append(f'  - Band gap split: {z["share_exact_zero"]:.0%} of test crystals have gap exactly 0. On those, network MAE {g(z["network_mae_zero"])} vs trees {g(z["trees_mae_zero"])}; '
                         f'on the rest, network {g(z["network_mae_nonzero"])} vs trees {g(z["trees_mae_nonzero"])}.')
            L.append('')
    return L


def build(pipe):
    pipe = Path(pipe)
    plan = json.loads((pipe / 'plan.json').read_text())
    state = json.loads((pipe / 'state.json').read_text())
    tel = telemetry_summary(read_telemetry(pipe / 'telemetry.csv'))
    A = analyze_all(pipe, plan)
    jobs = plan['jobs']
    st = state['jobs']
    done = [j for j in jobs if st.get(j['id'], {}).get('status') == 'done']
    bad = [j for j in jobs if st.get(j['id'], {}).get('status') in ('failed', 'timeout')]
    other = [j for j in jobs if st.get(j['id'], {}).get('status') not in ('done', 'failed', 'timeout')]
    gpu_h = sum(st[j['id']].get('seconds', 0) for j in done if j['queue'] == 'gpu') / 3600
    tab_ok = [t for t in A['tabular'] if t['status'] == 'ok']
    str_ok = [t for t in A['structure'] if t['status'] == 'ok']
    tv = [t['network_vs_trees']['verdict'] for t in tab_ok]
    tb = [single_bucket(t['fairness']) for t in tab_ok]
    sv = [(t['verdicts'].get('pred_comp_cell') or next(iter(t['verdicts'].values())))['verdict'] for t in str_ok]
    sb = [single_bucket(t['fairness']) if t.get('fairness') else 'unclear' for t in str_ok]
    smoke = plan.get('profile') == 'smoke'
    L = [f'# Pipeline report: {pipe.name}', '', f'Generated {time.strftime("%Y-%m-%d %H:%M:%S")}. Folder: `{pipe}`', '']
    if smoke:
        L += ['> **REHEARSAL (smoke profile).** Tiny networks, a few hundred crystals, a few epochs. This run only checks that the machinery works; '
              'its accuracy numbers and verdicts mean nothing. Run `--profile standard` for real results.', '']
    L += ['## At a glance', '',
          f'- Jobs: **{len(done)} finished**, {len(bad)} failed or timed out, {len(other)} not run (stopped or out of time) of {len(jobs)}. GPU job time: {gpu_h:.2f} h.',
          f'- Table-based families ({len(tab_ok)} with results). Average of the network seeds against one tuned tree model: network clearly better on {tv.count("network clearly better")}, trees clearly better on {tv.count("trees clearly better")}, no clear difference on {tv.count("no clear difference")}. '
          f'**One network at a time**: clearly better in every seed on {tb.count("network")}, trees clearly better in every seed on {tb.count("trees")}, mixed or unclear on {tb.count("unclear")}.',
          f'- Crystal-structure properties ({len(str_ok)} with results), against the stronger trees. Average of runs: network clearly better on {sv.count("network clearly better")}, trees clearly better on {sv.count("trees clearly better")}, no clear difference on {sv.count("no clear difference")}. '
          f'**Single runs**: clearly better in every run on {sb.count("network")}, mixed or unclear on {sb.count("unclear")}, trees clearly better in every run on {sb.count("trees")}.']
    if tel['samples']:
        L.append(f'- GPU: peak temperature {g(tel["peak_temp_c"], 3)} C, peak power {g(tel["peak_power_w"], 3)} W, peak memory {g(tel["peak_mem_mib"], 4)} MiB, average utilisation {g(tel["mean_util_pct"], 3)}% ({tel["samples"]} samples).')
    integ = state.get('integrity')
    if integ:
        L.append(f'- Sealed data: {integ}')
    L += ['', '**Averages versus single networks.** The headline "average" verdict compares several networks averaged together with ONE tree model, which favours the network just by averaging. '
          'The "one network at a time" verdict removes that. If the two disagree, the win is an ensemble effect; the fairness checks under each family also show a no-fit blend of networks and trees.', '']
    L += ['**How to read the verdicts.** "Network clearly better" means the whole 95% bootstrap interval of (network error minus trees error) is below zero; "trees clearly better" means it is '
          'entirely above zero; anything else is "no clear difference". The bootstrap resamples whole groups (mixes, alloy systems, papers, formulas), never single rows.', '']
    L += tabular_section(A['tabular']) + structure_section(A['structure'])
    L += ['## What each job did', '', '| Job | Status | Minutes | Attempts |', '| --- | --- | --- | --- |']
    for j in jobs:
        s = st.get(j['id'], {})
        L.append(f'| {j["id"]} | {s.get("status", "-")} | {s.get("seconds", 0) / 60:.1f} | {len(s.get("attempts", []))} |')
    L.append('')
    if bad or other:
        L += ['## Problems', '']
        for j in bad:
            L += [f'### {j["id"]}: {st[j["id"]]["status"]}', '```', read_log_tail(pipe / 'logs' / f'{j["id"]}.log'), '```', '']
        if other:
            L.append('Not run: ' + ', '.join(j['id'] for j in other) + '.')
            L.append('')
        L += [f'To continue: `python -m gpu.run_pipeline --resume "{pipe}"` (finished jobs are skipped; add `--retry-failed` to try failed ones again).', '']
    L += ['## What this does and does not show', '',
          '- Every number is from historical data or DFT calculations. None of it is a new measurement, so none of it is evidence of a discovery; a sealed set of new measurements is still the missing evidence.',
          '- Tabular families: the old fixed test splits were already looked at by earlier work, so this report uses cross-validation with cross-fitted selection instead. Concrete is exploratory.',
          '- Crystal properties: the split is by exact formula, so crystals of the same structure type with different elements can sit on both sides. Expect the real error on genuinely new chemistries to be larger. A split that holds out whole chemical systems is the next test.',
          '- Each comparison is one of many; with a 95% interval, a few "clear" results are expected by chance. Treat single clear wins as leads to confirm, not as findings.',
          '- Rows of a family that share a source (paper, alloy system, mix) were always kept together in a fold.', '']
    nxt = []
    if sv.count('network clearly better'):
        names = [STRUCT_NAMES.get(t['target'], t['target']) for t, v in zip(str_ok, sv) if v == 'network clearly better']
        nxt.append('The average of crystal-network runs clearly beats the cell-feature trees on: ' + ', '.join(names) + '. Check the "single runs" column before crediting the network itself, '
                   'and validate on a split that holds out whole chemical systems and against trees given richer structure features.')
    elif str_ok:
        nxt.append('No crystal property shows a clear win over trees that are given the cell volume. Either test harder splits, or use trees for these properties.')
    if tv.count('network clearly better'):
        names = [TAB_NAMES.get(t['family'], t['family']) for t, v in zip(tab_ok, tv) if v == 'network clearly better']
        nxt.append('The average of network seeds clearly beats the tuned trees on: ' + ', '.join(names) + '. Single networks may not; see the "one network at a time" column.')
    elif tab_ok:
        nxt.append('On the table-based families the network did not clearly beat tuned trees, even with the whole sweep and the correction for picking the best setting. '
                   'That points to the amount and kind of data, not the compute, as the limit [likely, not proven]; more rows or better descriptors are the likelier lever than a bigger sweep.')
    if bad or other:
        nxt.append('Some jobs did not finish: see Problems above.')
    if smoke:
        nxt = ['This was a rehearsal: nothing above is a result. If no jobs failed, start the real run with `--profile standard`.'] + (['Some rehearsal jobs did not finish: see Problems above.'] if bad or other else [])
    L += ['## Suggested next steps', ''] + [f'{i + 1}. {n}' for i, n in enumerate(nxt)] + ['']
    return '\n'.join(L), {'analysis': A, 'telemetry': tel, 'jobs': {j['id']: st.get(j['id'], {}) for j in jobs}, 'notes': plan.get('notes', []),
                          'integrity': integ}


def share_bundle(pipe):
    pipe = Path(pipe)
    share = pipe / 'share'
    share.mkdir(exist_ok=True)
    for name in ('REPORT.md', 'summary.json', 'plan.json', 'state.json', 'telemetry.csv', 'pipeline.log'):
        if (pipe / name).exists():
            shutil.copy2(pipe / name, share / name)
    (share / 'logs').mkdir(exist_ok=True)
    for f in (pipe / 'logs').glob('*.log'):
        shutil.copy2(f, share / 'logs' / f.name)
    for pat in ('jobs/*/**/result.json', 'jobs/*/baselines.json', 'jobs/*/trees.json'):
        for f in pipe.glob(pat):
            dst = share / 'results' / f.relative_to(pipe / 'jobs')
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
    return shutil.make_archive(str(pipe / 'share'), 'zip', root_dir=str(share))


def generate(pipe):
    pipe = Path(pipe)
    text, summary = build(pipe)
    for name in ('REPORT.md', 'summary.json'):
        old = pipe / name
        if old.exists():
            old.rename(pipe / f'{old.stem}.{time.strftime("%Y%m%dT%H%M%S", time.gmtime(old.stat().st_mtime))}.old{old.suffix}')
    atomic_write(pipe / 'REPORT.md', text)
    atomic_write(pipe / 'summary.json', json.dumps(summary, indent=1, default=str))
    return pipe / 'REPORT.md', share_bundle(pipe)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline', required=True)
    a = ap.parse_args(argv)
    rep, z = generate(a.pipeline)
    print(f'wrote {rep}\nwrote {z}')


if __name__ == '__main__':
    main()
