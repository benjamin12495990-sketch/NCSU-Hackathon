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
             'chemistry': 'Chemistry (water solubility)', 'cleaning_pcmc': 'Surfactants (critical micelle concentration)',
             'cleaning_tension': 'Surfactants (surface tension at the CMC)', 'corrosion': 'Corrosion (rate, NIST CORR-DATA, exploratory)'}
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


def primary(t):
    """The verdict against the tree feature set with the lowest validation error (the strongest opponent chosen without the test rows)."""
    return t['verdicts'].get(t.get('strongest_trees_key')) or next(iter(t['verdicts'].values()))


RULE_WORDS = {'first': 'the original rule (ties among the 12 nearest neighbours broken by listing order; not invariant)',
              'all': 'the fixed rule (every neighbour tied with the 12th is kept; invariant)', 'unknown': 'an unrecorded rule'}


RULE_SHORT = {'first': 'original rule', 'all': 'fixed rule', 'unknown': 'unrecorded rule', 'mixed': 'both rules', None: 'no rule (no structure features)'}


def rules_text(rs):
    return ' and '.join(RULE_SHORT.get(r, str(r)) for r in rs) if rs else '-'


def rule_note(ok):
    """One line saying which neighbour rule the networks and the structure-feature trees of a section were built with."""
    nets = sorted({r for t in ok for r in (t.get('neighbor_rules') or {}).get('networks', [])})
    trees = sorted({r for t in ok for r in (t.get('neighbor_rules') or {}).get('trees', [])})
    if not nets:
        return []
    txt = 'Neighbour rule: networks built with ' + ' and '.join(RULE_WORDS.get(r, r) for r in nets)
    if trees:
        txt += '; structure-feature trees with ' + ('the same' if trees == nets else ' and '.join(RULE_WORDS.get(r, r) for r in trees))
    warn = ''
    if len(nets) > 1 or (trees and trees != nets):
        warn = ' **The rules differ inside this section, so its comparisons include the effect of the rule change.**'
    elif 'first' in nets:
        warn = ' Under this rule the same crystal written two ways can get two predictions (`docs/DECISIONS.md`, LES-017).'
    return [txt + '.' + warn, '']


def mixed_rule_warning(c, names=('the first set', 'the second set')):
    """A sentence for a comparison whose sides were built under different neighbour rules ('' if they were not): the two sets of
    networks, or either set against structure-feature trees made with the other rule."""
    c = c or {}
    r = c.get('neighbor_rules') or {}
    parts = []
    if c.get('mixed_neighbor_rules'):
        parts.append(f'{names[0]} used the {rules_text(r.get("new"))}, {names[1]} the {rules_text(r.get("old"))}')
    if c.get('mixed_new_vs_trees') or c.get('mixed_old_vs_trees'):
        parts.append(f'the structure-feature trees used the {RULE_SHORT.get(r.get("trees"), r.get("trees"))}')
    if not parts:
        return ''
    return (' **Different neighbour rules: ' + '; '.join(parts) + ', so the comparisons that mix them include the effect of the rule change, '
            'not only what the row says it compares.**')


def tiefix_section(items):
    ok = [e for e in items if e.get('status') == 'ok']
    L = ['## Neighbour-rule fix: did it change accuracy, and did it remove the defect?', '',
         'Until this fix each atom kept exactly its 12 nearest neighbours; when several neighbours sat at the same distance (to within rounding), which ones were kept '
         'depended on the order the atoms were listed in. The same crystal written two ways could then get two different predictions. The fixed rule keeps every '
         'neighbour within 0.001 angstrom of the 12th, so the graph depends on the geometry only. Here the held-out-systems networks and trees are trained again under '
         'the fixed rule, with the same seeds, settings, crystals and split, and compared with the originals on the same test crystals. Each side is chosen by the rule '
         'its runs recorded, not by job name. Negative differences mean the fixed rule has the lower error.', '']
    if ok:
        L += ['| Property | Seeds on both sides | Original rule: single / average | Fixed rule: single / average | Fixed minus original, average of runs (95%, test crystals only) | On these networks | Per seed: fixed minus original | Trees, strongest set: original / fixed | Trees, same set under both rules: fixed minus original (95%) | Fixed network average minus fixed trees | Edges vs original | Training time vs original |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for e in ok:
            tr = e.get('trees')
            tr1 = f'{g(tr["original_mae"])} ({tr["original_strongest"]}) / {g(tr["fixed_mae"])} ({tr["fixed_strongest"]})' if tr else '-'
            if tr and tr.get('fixed_vs_original'):
                tr2 = f'{tr["compared_set"]}: {ci(tr["fixed_vs_original"])}, {tr["fixed_vs_original"]["verdict"]}'
            elif tr:
                tr2 = 'no feature set on both sides that uses the rule'
            else:
                tr2 = e.get('trees_missing') or 'trees missing'
            nt = f'{ci(e["new_vs_trees"])}: {e["new_vs_trees"]["verdict"].replace("fixed rule", "network")}' if tr else '-'
            ps = e.get('per_seed') or []
            pst = ', '.join(f's{x["seed"]} {x["diff"]:+.3g}' for x in ps) + (f' ({e["seeds_lower_fixed"]} of {len(ps)} lower under the fixed rule)' if ps else '')
            L.append(f'| {STRUCT_NAMES.get(e["target"], e["target"])} | {", ".join(str(x) for x in e["seeds"])} | {g(e["old_single_mae_mean"])} / {g(e["old_average_mae"])} | '
                     f'{g(e["new_single_mae_mean"])} / {g(e["new_average_mae"])} | {ci(e["new_vs_old"])} | **{e["new_vs_old"]["verdict"]}** | {pst or "-"} | {tr1} | {tr2} | {nt} | '
                     f'x{e["edges_ratio"]:.2f} ({g(e.get("edges_per_atom_fixed"), 3)} per atom) | x{e["train_seconds_ratio"]:.2f}' + (' (different devices: not comparable)' if len(e.get('devices', [])) > 1 else '') + ' |')
        sds = [f'{STRUCT_NAMES.get(e["target"], e["target"]).split(" (")[0]}: original {g(e["seed_sd"]["original"], 2)}, fixed {g(e["seed_sd"]["fixed"], 2)}'
               for e in ok if (e.get('seed_sd') or {}).get('original') is not None]
        L += ['', '**How far this goes.** The 95% interval resamples test crystals with these particular trained networks held fixed. It does not include which seeds happened '
              'to be trained, and with two or three seeds per side that is the larger uncertainty: a "clearly better" here is a statement about these networks on these '
              'test crystals, not proof that the rule itself makes networks more accurate. The per-seed column shows whether every seed moved the same way'
              + ('; for scale, the seed-to-seed spread (standard deviation) of single-run test error is ' + '; '.join(sds) if sds else '') + '. '
              '"No clear difference" means the fix did not move the error by more than the interval can resolve, not that the two sets of networks are identical. '
              'Training time is each run\'s own training clock and depends on what else the GPU was doing. Every column compares like with like: fixed '
              'against original is the rule change itself, and the network-against-trees column uses the fixed-rule trees.', '']
    for e in items:
        if e.get('excluded_runs'):
            L.append(f'- {STRUCT_NAMES.get(e.get("target"), e.get("target"))}: left out because their recorded rule does not match their side: ' + ', '.join(e['excluded_runs']) + '.')
        if e.get('status') != 'ok':
            L.append(f'- {STRUCT_NAMES.get(e.get("target"), e.get("target"))}: no accuracy comparison ({e.get("reason") or "the fixed-rule runs, the original runs or the trees did not finish"}; '
                     f'{e.get("n_new_runs_finished", 0)} retrained and {e.get("n_old_runs_finished", 0)} held-out-systems runs found).')
    inv = [(e, r, e['invariance'][r]) for e in items for r in ('first', 'all') if r in (e.get('invariance') or {})]
    if inv:
        L += ['', '**Does a prediction change when the crystal is only rewritten?** One network of each kind predicts test crystals four ways besides the original: atoms relisted, '
              'origin shifted, crystal rotated, and (small cells) the cell doubled in every direction. Nothing physical changes, so every change below is error from the method. '
              'Each row is labelled with the rule the measured run recorded.', '',
              '| Property | Network | Test crystals | Rewrites that move the prediction by more than 1% of the spread of predictions | Largest change | Average change, as a share of the run\'s test error |',
              '| --- | --- | --- | --- | --- | --- |']
        for e, r, v in inv:
            o = v['overall']
            share = '-' if o.get('mean_change_as_share_of_test_mae') is None else f'{o["mean_change_as_share_of_test_mae"]:.2%}'
            rec = v.get('neighbor_ties', 'unknown')
            label = RULE_SHORT.get(rec, rec) + ('' if rec == r else f' (**planned as the {RULE_SHORT.get(r, r)}**)')
            L.append(f'| {STRUCT_NAMES.get(e["target"], e["target"])} | {label} (seed {v.get("seed", "?")}) | {v["crystals"]} | '
                     f'{o["share_changed_over_1pct_of_spread"]:.1%} | {g(o["max_change"], 3)} {v["unit"]} | {share} |')
        L += ['', 'Under the fixed rule the remaining changes should be rounding noise (around 0.00001 of the unit). The fix does not make the cut smooth: a neighbour sitting '
              'just outside the 0.001 angstrom tolerance is still dropped whole, so two crystals that differ by a tiny real distortion can still get slightly different graphs.', '']
    L.append('')
    return L


def rule_caveat(A, smoke=False, nightly=False):
    """A bullet for 'what this does and does not show' when crystal results in the folder were made with the original neighbour rule."""
    rules = {r for k in ('structure', 'structure_system', 'structure_element', 'structure_elementP', 'structure_systemT') for t in A.get(k, [])
             if t.get('status') == 'ok' for r in (t.get('neighbor_rules') or {}).get('networks', [])}
    if 'first' not in rules:
        return []
    return ['- Crystal networks marked "original rule" were trained before the neighbour-rule fix: when several neighbours are equally far, which ones the network sees depends on '
            'how the crystal is written down, so the same crystal can get slightly different predictions. Their test errors are real measurements of those networks, but a '
            'comparison between an original-rule set and a fixed-rule set also contains the rule change' + tiefix_advice(A, smoke, nightly)]


def tiefix_advice(A, smoke, nightly):
    """How the rule change is (or can be) measured for this folder. -TieFix is suggested only where it can run: the folder has finished
    held-out-systems networks made with the original rule (the only thing it can retrain) and no finished comparison yet."""
    if any(e.get('status') == 'ok' for e in A.get('tiefix') or []):
        return '; the "Neighbour-rule fix" section measures it.'
    can = any(t.get('status') == 'ok' and (t.get('neighbor_rules') or {}).get('networks') == ['first'] for t in A.get('structure_system', []))
    if can and not nightly:
        return '. Run `.\\gpu\\run_overnight.ps1 -TieFix' + (' -Rehearsal' if smoke else '') + '` to measure it.'
    return ('. This folder has no original-rule held-out-systems runs to retrain, so the rule change is not measured here; runs made with the fixed rule '
            '(a new run, or the nightly) give fixed-rule numbers.')


def structure_section(items, split='formula', ref=None, tiefix=None):
    """ref: for split 'systemT', the original held-out-systems items, so the header can say whether the seeds really are the same;
    tiefix: the 'Neighbour-rule fix' items, so it mentions the seed-by-seed comparison only where that comparison exists."""
    ok = [t for t in items if t['status'] == 'ok']
    if split == 'formula':
        head = ['## Crystal-structure networks against trees (split by exact formula)', '',
                'A crystal-graph network learns from the atom positions. The trees get element fractions, fractions plus cell volume, or (when that job ran) element-property '
                'statistics plus neighbour-geometry statistics; the opponent in each verdict is the tree set with the lowest *validation* error. Test error is on formulas the '
                'models never saw. All labels are **DFT calculations**, not measurements.', '']
    elif split in ('element', 'elementP'):
        import json as _json
        from pathlib import Path as _P
        es = _json.loads((_P(__file__).resolve().parents[1] / 'structure' / 'element_split.json').read_text())
        what = ('networks that describe atoms by **element properties** (electronegativity, radius, group, period, ...; `--atom-input both`)' if split == 'elementP' else
                'the default networks, which learn **one vector per element** (an element never seen in training keeps its random starting vector)')
        head = [f'## Crystal-structure networks on HELD-OUT ELEMENTS ({"element-property inputs" if split == "elementP" else "learned element vectors"})', '',
                f'The hardest test here. Every crystal containing {", ".join(es["test_elements"])} is test ({es["share_test"]:.0%} of crystals); every crystal containing '
                f'{", ".join(es["val_elements"])} (and none of those) is validation; neither set of elements appears in training. This section shows {what}. '
                'Trees on element fractions cannot use an unseen element either; trees on element-property statistics can. All labels are **DFT calculations**.', '']
    elif split == 'systemT':
        seeds = lambda t: sorted(r['seed'] for r in t.get('network_runs', []))  # noqa: E731
        ref_ok = {t['target']: t for t in (ref or []) if t.get('status') == 'ok'}
        short = lambda t: STRUCT_NAMES.get(t['target'], t['target']).split(' (')[0]  # noqa: E731
        differ = [short(t) + f' (seeds {seeds(t)} here, {seeds(ref_ok[t["target"]])} there)' for t in ok if t['target'] in ref_ok and seeds(t) != seeds(ref_ok[t['target']])]
        noref = [short(t) for t in ok if t['target'] not in ref_ok]
        compared = {e['target'] for e in (tiefix or []) if e.get('status') == 'ok'}
        notcmp = [short(t) for t in ok if t['target'] not in compared]
        rich = any((t.get('neighbor_rules') or {}).get('trees') for t in ok)
        head = ['## Crystal-structure networks on HELD-OUT CHEMICAL SYSTEMS, fixed neighbour rule', '',
                'The same test as the held-out-systems section (same crystals, split and settings), with networks' + (' and structure-feature trees' if rich else '')
                + ' built under the fixed neighbour rule, which keeps every neighbour tied with the 12th so a prediction no longer depends on how the crystal is written down. '
                + ('Not every seed finished on both sides: ' + '; '.join(differ) + ', so the two sections are not exactly the same set of runs. ' if differ else '')
                + ('No held-out-systems result to compare with for: ' + ', '.join(noref) + '. ' if noref else '')
                + ('Same seeds as there. ' if ok and not differ and not noref else '')
                + 'These are the numbers to carry forward' + ('; the section "Neighbour-rule fix" compares them with the originals seed by seed' if compared else '')
                + (' (not for ' + ', '.join(notcmp) + ': see that section)' if compared and notcmp else '; that comparison did not finish, see "Neighbour-rule fix"' if not compared else '')
                + '. All labels are **DFT calculations**.', '']
    else:
        head = ['## Crystal-structure networks on HELD-OUT CHEMICAL SYSTEMS', '',
                'The harder test. Every crystal whose element combination (Fe-O, Li-Fe-P-O, ...) lands in the test set is absent from training, so the test error says how the '
                'models do on element combinations they have never seen. Same models, same trees, same verdict rules as above. Expect larger errors than the formula split; '
                'what matters is the comparison. All labels are **DFT calculations**.', '']
    L = list(head)
    L += rule_note(ok)
    if ok:
        L += ['| Property | Crystals (train / test) | Guess the average | Linear | Trees: fractions | Trees: + cell | Trees: + structure features | Network: mean of single runs | Network: average of runs | Average minus the strongest trees (95% interval) | Verdict: average | Verdict: single runs | Speed |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for t in ok:
            tr, v = t['trees'], primary(t)
            sp = [r['structures_per_second'] for r in t['network_runs']]
            nm = [r['test']['mae'] for r in t['network_runs']]
            avg = [c for k2, c in t['compare']['comparisons'][t['strongest_trees_key']].items() if k2.startswith('network average')]
            avg_mae = avg[0]['network_mae'] if avg else sum(nm) / len(nm)
            f = t.get('fairness')
            mae = lambda k: g(tr[k]['test']['mae']) if k in tr else '-'  # noqa: E731
            L.append(f'| {STRUCT_NAMES.get(t["target"], t["target"])} | {t["split_sizes"]["train"]} / {t["split_sizes"]["test"]} | {g(t["mean_baseline_test_mae"])} | {g(t["ridge_test_mae"])} | '
                     f'{mae("composition")} | {mae("comp+cell")} | {mae("rich")} | {g(sum(nm) / len(nm))} ({len(nm)} runs) | {g(avg_mae)} | {ci(v)} vs {t["strongest_trees"]} | **{v["verdict"]}** | '
                     f'{single_text(f) if f else "-"} | {sum(sp) / len(sp):.0f} crystals/s |')
        L.append('')
    for t in items:
        if t['status'] != 'ok':
            L.append(f'- **{STRUCT_NAMES.get(t["target"], t["target"])}**: no result ({t["reason"]}).')
    for t in ok:
        L += [f'### {STRUCT_NAMES.get(t["target"], t["target"])}' + {'formula': '', 'system': ' (held-out systems)', 'element': ' (held-out elements, learned vectors)', 'elementP': ' (held-out elements, element properties)', 'systemT': ' (held-out systems, fixed neighbour rule)'}[split], '', '```'] + t['compare_lines'] + ['```', '']
        vals = ', '.join(f'{k}: validation MAE {g(v["val"]["mae"])}' for k, v in t['trees'].items())
        L.append(f'- Strongest trees by validation error: {t["strongest_trees"]} ({vals}).')
        if t.get('fairness'):
            L += fairness_lines(t['fairness'], primary(t)['verdict'], 'average of runs')
            z = t['fairness'].get('by_zero_gap')
            if z:
                L.append(f'  - Band gap split: {z["share_exact_zero"]:.0%} of test crystals have gap exactly 0. On those, network MAE {g(z["network_mae_zero"])} vs trees {g(z["trees_mae_zero"])}; '
                         f'on the rest, network {g(z["network_mae_nonzero"])} vs trees {g(z["trees_mae_nonzero"])}.')
            L.append('')
    return L


def evolution_section(items):
    ok = [e for e in items if e.get('status') == 'ok']
    L = ['## Generations of crystal networks (held-out chemical systems)', '',
         'A small evolutionary search per property. Generation 0 is the default settings plus random ones; each later generation is children of the two best individuals so far, '
         'with two settings nudged. Every individual is an ordinary training run on the harder split; its **reward is the validation error on held-out systems**. The test rows '
         'never choose anything. Nothing is deleted: `lineage.json` and every run folder keep all individuals.', '',
         '**What counts as evidence:** the validation curve below is optimistic (it is the best of many candidates on the same rows). The only fair test is the finalist line: '
         'the winner retrained on fresh seeds and scored on test systems it never influenced, against the default settings trained the same way earlier in this pipeline.', '']
    for e in items:
        if e.get('status') != 'ok':
            L.append(f'- **{STRUCT_NAMES.get(e["target"], e["target"])}**: no result ({e.get("reason", "no individual finished")}).')
    for e in ok:
        name = STRUCT_NAMES.get(e['target'], e['target'])
        curve = ' -> '.join(g(v) for v in e['best_by_generation'])
        start = 'The default settings' if e.get('round', 'r1') == 'r1' else f'The starting point ({e.get("start_label")}: {e.get("start_changes")})'
        sr = e.get('search_rules') or []
        fr = (e.get('finalists') or {}).get('neighbor_rules') or {}
        rule_line = (f'- Neighbour rule: search individuals {rules_text(sr)}' + (f'; finalists {rules_text(fr.get("new"))}; compared runs {rules_text(fr.get("old"))}' if fr else '')
                     + ('. **This round mixes rules, so its ranking and comparisons include the effect of the rule change.**' if len(sr) > 1 or (fr and (fr.get('new') != fr.get('old') or (sr and fr.get('new') != sr))) else '.'))
        L += [f'### {name}' + ('' if e.get('round', 'r1') == 'r1' else f' — search round {e["round"]} (wider settings, starting from the round-1 winner; finalists trained {e.get("final_epochs")} epochs, the same as the runs they are compared with)'), '',
              f'- {e["n_finished"]} of {e["n_individuals"]} individuals finished over {e["generations"]} generations.', rule_line,
              f'- Best validation MAE so far, generation by generation: {curve}. {start} scored {g(e["default_val_mae"])} in the same protocol.']
        if e['default_won']:
            L.append('- The starting settings were still the best by validation error: the search found nothing better, so there is no finalist retrain.')
        else:
            gain = (e['default_val_mae'] - e['best_val_mae']) / e['default_val_mae'] * 100 if e['default_val_mae'] else None
            L.append(f'- Best individual {e["best_id"]}: {e["best_changes"]} (validation MAE {g(e["best_val_mae"])}' + (f', {gain:.1f}% below the starting settings' if gain is not None else '') + ').')
        if e.get('finalist_differs_from_best'):
            L.append(f'- **Note:** the finalists below were trained with {e.get("finalist_changes")} (the best when they were chosen). A later individual now has a lower '
                     'validation error; it was not retrained, so the finalist numbers are NOT for the "best individual" named above.')
        elif e.get('finalist_changes') and not e['default_won']:
            L.append(f'- Finalist settings: {e["finalist_changes"]}.')
        fi = e.get('finalists')
        if fi:
            d = fi['evolved_vs_default']
            old = 'default' if e.get('round', 'r1') == 'r1' else 'previous winner'
            L += [f'- **Finalists on test systems** ({fi["n_evolved_runs"]} evolved runs on fresh seeds vs {fi["n_default_runs"]} runs of {e.get("compare_label", "the default settings")}): single-run MAE evolved {g(fi["evolved_single_mae_mean"])} vs {old} {g(fi["default_single_mae_mean"])}; '
                  f'average of runs evolved {g(fi["evolved_average_mae"])} vs {old} {g(fi["default_average_mae"])}; strongest trees ({fi["strongest_trees"]}) {g(fi["trees_mae"])}.',
                  f'  - Evolved minus {old} (average of runs): {ci(d)}, **{d["verdict"]}**.' + mixed_rule_warning(fi, ('the evolved runs', f'the {old} runs')),
                  f'  - Evolved against the strongest trees: {ci(fi["evolved_vs_trees"])}, {fi["evolved_vs_trees"]["verdict"]}; {old} against the strongest trees: {ci(fi["default_vs_trees"])}, {fi["default_vs_trees"]["verdict"]}.']
        elif not e['default_won']:
            L.append('- Finalist runs are missing or did not finish' + (f' ({e["finalists_error"]})' if e.get('finalists_error') else '') + '.')
        top = sorted([x for x in e['individuals'] if x['val_mae'] is not None], key=lambda x: x['val_mae'])[:6]
        L += ['', '| Rank | Individual | Generation | Parent | Differs from the defaults | Validation MAE |', '| --- | --- | --- | --- | --- | --- |']
        for k, x in enumerate(top, 1):
            L.append(f'| {k} | {x["id"]} | {x["gen"]} | {x["parent"] or "-"} | {x["changes"]} | {g(x["val_mae"])} |')
        L.append('')
    return L


def finetune_section(items):
    ok = [e for e in items if e.get('status') == 'ok']
    L = ['## Moduli networks that start from the formation-energy network', '',
         'Bulk and shear modulus have about 30,000 labelled crystals; formation energy has about 94,000. Each fine-tuned network starts from a formation-energy network '
         '(every layer except the output) trained on the **same split**, and the trainer refuses to start if that network ever trained on, or chose its epoch with, '
         'this run\'s validation or test crystals. It is then compared with the same architecture trained from scratch on the same rows, and with the strongest trees. '
         'Two things differ besides the starting weights, so read the verdict as "this fine-tuning recipe against the from-scratch recipe": fine-tuned runs use learning rate 1e-3 '
         '(from-scratch runs 2e-3), and every fine-tuned seed of a split starts from the same formation-energy network (seed 11), so their average is an average of less '
         'independent networks than the from-scratch average.', '']
    if ok:
        L += ['| Property | Split | Neighbour rule: fine-tuned / scratch / trees | Fine-tuned: single / average | From scratch: single / average | Fine-tuned minus scratch (95%) | Verdict | Fine-tuned minus strongest trees | Verdict vs trees | Single fine-tuned runs vs trees |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for e in ok:
            sv = e['single_new_vs_trees']
            single = f'{sum(v.endswith("fine-tuned clearly better") for v in sv)} of {len(sv)} clearly better, {sum(v == "trees clearly better" for v in sv)} clearly worse'
            nr = e.get('neighbor_rules') or {}
            L.append(f'| {STRUCT_NAMES.get(e["target"], e["target"])} | {e["split"]} | {rules_text(nr.get("new"))} / {rules_text(nr.get("old"))} / {RULE_SHORT.get(nr.get("trees"), nr.get("trees"))} | {g(e["new_single_mae_mean"])} / {g(e["new_average_mae"])} | {g(e["old_single_mae_mean"])} / {g(e["old_average_mae"])} | '
                     f'{ci(e["new_vs_old"])} | **{e["new_vs_old"]["verdict"]}** | {ci(e["new_vs_trees"])} ({e["strongest_trees"]}, MAE {g(e["trees_mae"])}) | {e["new_vs_trees"]["verdict"]} | {single} |')
        L.append('')
        for e in ok:
            w = mixed_rule_warning(e, ('the fine-tuned runs', 'the from-scratch runs'))
            if w:
                L.append(f'- {STRUCT_NAMES.get(e["target"], e["target"])} ({e["split"]}):' + w)
    for e in items:
        if e.get('status') != 'ok':
            L.append(f'- {STRUCT_NAMES.get(e.get("target"), e.get("target"))} ({e.get("split", "-")}): no result ({e.get("reason", "runs missing")}).')
    L.append('')
    return L


def intervals_section(iv):
    if not iv or iv.get('error'):
        return ['## Prediction intervals', '', f'- Not computed ({(iv or {}).get("error", "no data")}).', '']
    L = ['## Prediction intervals (conformal)', '',
         'For every prediction, an interval "value +/- w" built so that, on rows like the test rows, it contains the true value 90% of the time. w is calibrated on '
         'rows the model never trained on (validation crystals; for table families, the other cross-validation folds) and then **checked** on rows the calibration never saw. '
         '"Coverage" is how often the interval really contained the truth: 0.90 for a 90% interval means it can be trusted; well below means it is too narrow. '
         'For log10 targets a half-width w means "within a factor 10^w". Scaled intervals use the spread between network seeds, so confident rows get narrow intervals.', '']
    rows = []
    for x in iv.get('tabular', []):
        for m, v in (x.get('models') or {}).items():
            rows.append((TAB_NAMES.get(x['task'], x['task']), '-', m, v))
    for x in iv.get('crystal', []):
        for m, v in (x.get('models') or {}).items():
            sp = {'systemT': 'system'}.get(x['split'], x['split']) + (f', {rules_text(x["neighbor_rules"])}' if x.get('neighbor_rules') else '')
            rows.append((STRUCT_NAMES.get(x['task'], x['task']), sp, m + (f' ({x["n_runs"]} network runs)' if x.get('n_runs') and m != 'trees' else ''), v))
    if rows:
        L += ['| Task | Split | Model | 90% half-width | 90% coverage | 80% coverage | 95% coverage | Scaled 90%: median half-width / coverage |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for task, split, m, v in rows:
            sc = v['0.90'].get('scaled')
            L.append(f'| {task} | {split} | {m} | {g(v["0.90"]["half_width"])} | {v["0.90"]["coverage"]:.3f} | {v["0.80"]["coverage"]:.3f} | {v["0.95"]["coverage"]:.3f} | '
                     + (f'{g(sc["median_half_width"])} / {sc["coverage"]:.3f}' if sc else '-') + ' |')
        L.append('')
    for x in iv.get('tabular', []) + iv.get('crystal', []):
        if x.get('error'):
            L.append(f'- {x["task"]}: not computed ({x["error"]}).')
    L += ['', '- These are **marginal** guarantees (true on average over rows like the test rows), not a promise for each single material. A material unlike anything in the '
          'data (a new element, a far-off composition) can be outside its interval more often; the held-out-elements rows show how much.', '']
    return L


def held_out_systems_summary(sys_orig, sys_fixed):
    """(items to quote, rule text, whether the held-out-systems networks are all original-rule). Property by property: the fixed-rule result
    where it finished, else the held-out-systems one, so a property whose retrain failed is still counted, under the rule it recorded."""
    fixed_by = {t['target']: t for t in sys_fixed}
    items = [fixed_by.get(t['target'], t) for t in sys_orig] + [t for t in sys_fixed if t['target'] not in {x['target'] for x in sys_orig}]
    by_rule = {}
    for t in items:
        by_rule.setdefault(rules_text((t.get('neighbor_rules') or {}).get('networks', [])), []).append(t)
    text = '; '.join(f'{k} {len(v)}' for k, v in by_rule.items()) if len(by_rule) > 1 else next(iter(by_rule), '-')
    return items, text, bool(sys_orig) and all((t.get('neighbor_rules') or {}).get('networks') == ['first'] for t in sys_orig)


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
    sys_orig = [t for t in A.get('structure_system', []) if t['status'] == 'ok']
    sys_fixed = [t for t in A.get('structure_systemT', []) if t['status'] == 'ok']
    # The held-out-systems numbers to quote, property by property: the fixed-rule result where it finished, else the held-out-systems one,
    # each named with the rule its networks recorded (so a property whose retrain failed is still counted, under its own rule)
    sys_ok, sys_rules, orig_is_first = held_out_systems_summary(sys_orig, sys_fixed)
    yv = [primary(t)['verdict'] for t in sys_ok]
    yb = [single_bucket(t['fairness']) if t.get('fairness') else 'unclear' for t in sys_ok]
    tv = [t['network_vs_trees']['verdict'] for t in tab_ok]
    tb = [single_bucket(t['fairness']) for t in tab_ok]
    sv = [primary(t)['verdict'] for t in str_ok]
    sb = [single_bucket(t['fairness']) if t.get('fairness') else 'unclear' for t in str_ok]
    smoke = any(str(x).endswith('smoke') for x in [plan.get('profile'), *plan.get('profile_history', [])])
    L = [f'# Pipeline report: {pipe.name}', '', f'Generated {time.strftime("%Y-%m-%d %H:%M:%S")}. Folder: `{pipe}`', '']
    if smoke:
        L += ['> **REHEARSAL (smoke profile).** Tiny networks, a few hundred crystals, a few epochs. This run only checks that the machinery works; '
              'its accuracy numbers and verdicts mean nothing. Run `--profile standard` for real results.', '']
    if plan.get('nightly'):
        from gpu.pipeline.nightly import registry_lines
        L += registry_lines(plan, pipe)
    L += ['## At a glance', '',
          f'- Jobs: **{len(done)} finished**, {len(bad)} failed or timed out, {len(other)} not run (stopped or out of time) of {len(jobs)}. GPU job time: {gpu_h:.2f} h.',
          f'- Table-based families ({len(tab_ok)} with results). Average of the network seeds against one tuned tree model: network clearly better on {tv.count("network clearly better")}, trees clearly better on {tv.count("trees clearly better")}, no clear difference on {tv.count("no clear difference")}. '
          f'**One network at a time**: clearly better in every seed on {tb.count("network")}, trees clearly better in every seed on {tb.count("trees")}, mixed or unclear on {tb.count("unclear")}.',
          f'- Crystal-structure properties ({len(str_ok)} with results), against the stronger trees. Average of runs: network clearly better on {sv.count("network clearly better")}, trees clearly better on {sv.count("trees clearly better")}, no clear difference on {sv.count("no clear difference")}. '
          f'**Single runs**: clearly better in every run on {sb.count("network")}, mixed or unclear on {sb.count("unclear")}, trees clearly better in every run on {sb.count("trees")}.']
    evo_ok = [e for e in (A.get('evolution') or []) if e.get('status') == 'ok']
    if evo_ok:
        r1 = [e for e in evo_ok if e.get('round', 'r1') == 'r1']
        fin = [e for e in r1 if e.get('finalists')]
        better = [STRUCT_NAMES.get(e['target'], e['target']) for e in fin if e['finalists']['evolved_vs_default']['verdict'] == 'evolved clearly better']
        won = [STRUCT_NAMES.get(e['target'], e['target']) for e in r1 if e['default_won']]
        if r1:
            L.append(f'- **Generations**: {len(r1)} properties searched; on fresh seeds and test systems the evolved settings clearly beat the defaults on {len(better)}' + (f' ({", ".join(better)})' if better else '') + f'; the defaults stayed best on {len(won)}' + (f' ({", ".join(won)})' if won else '') + '.')
        for e in evo_ok:
            if e.get('round', 'r1') == 'r1':
                continue
            name = STRUCT_NAMES.get(e['target'], e['target'])
            if e['default_won']:
                txt = 'nothing beat the starting settings (the previous winner)'
            elif e.get('finalists'):
                txt = f'finalists against {e.get("compare_label", "the previous winner")}: {e["finalists"]["evolved_vs_default"]["verdict"]}'
            else:
                txt = 'finalists not finished'
            L.append(f'- **Search round {e["round"]}** ({name}, wider settings): {txt}.')
    for key, label in (('structure_element', 'Held-out elements, learned element vectors'), ('structure_elementP', 'Held-out elements, element-property inputs')):
        ok_e = [t for t in A.get(key, []) if t['status'] == 'ok']
        if ok_e:
            ve = [primary(t)['verdict'] for t in ok_e]
            L.append(f'- **{label}** ({len(ok_e)} properties), against the strongest trees. Average of runs: network clearly better on {ve.count("network clearly better")} '
                     f'({", ".join(STRUCT_NAMES.get(t["target"], t["target"]) for t, v in zip(ok_e, ve) if v == "network clearly better") or "none"}), trees clearly better on {ve.count("trees clearly better")}, no clear difference on {ve.count("no clear difference")}.')
    ft_ok = [e for e in (A.get('finetune') or []) if e.get('status') == 'ok']
    if ft_ok:
        L.append('- **Fine-tuning from formation energy** (fine-tuned vs from scratch, average of runs): ' + '; '.join(
            f'{STRUCT_NAMES.get(e["target"], e["target"]).split(" (")[0]} {e["split"]}: {e["new_vs_old"]["verdict"]}' for e in ft_ok) + '.')
    tf_ok = [e for e in (A.get('tiefix') or []) if e.get('status') == 'ok']
    if tf_ok:
        L.append('- **Neighbour-rule fix** (same seeds retrained under the fixed rule; fixed minus original test MAE, average of runs, 95% interval over test crystals '
                 'only, not over which seeds were trained): ' + '; '.join(
            f'{STRUCT_NAMES.get(e["target"], e["target"]).split(" (")[0]} {ci(e["new_vs_old"])}, lower in {e.get("seeds_lower_fixed", "?")} of {len(e.get("per_seed") or e["seeds"])} seeds'
            for e in tf_ok) + '. See "Neighbour-rule fix".')
    ivs = A.get('intervals') or {}
    if ivs and not ivs.get('error'):
        cov = [v['0.90']['coverage'] for x in ivs.get('tabular', []) + ivs.get('crystal', []) for v in (x.get('models') or {}).values()]
        if cov:
            L.append(f'- **90% prediction intervals**: coverage between 0.87 and 0.93 for {sum(0.87 <= c <= 0.93 for c in cov)} of {len(cov)} task/model rows '
                     f'(lowest {min(cov):.2f}); see "Prediction intervals".')
    if sys_ok:
        L.append(f'- **Held-out chemical systems** ({len(sys_ok)} properties; networks by rule: {sys_rules}' + ('; the fixed-rule result is used where it finished, and the original-rule set is in its own section, for reference' if sys_fixed and orig_is_first else '')
                 + f'), against the strongest trees. Average of runs: network clearly better on {yv.count("network clearly better")}, trees clearly better on {yv.count("trees clearly better")}, no clear difference on {yv.count("no clear difference")}. '
                 f'Single runs: clearly better in every run on {yb.count("network")}, mixed or unclear on {yb.count("unclear")}, trees clearly better in every run on {yb.count("trees")}.')
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
    if A.get('structure_system'):
        L += structure_section(A['structure_system'], 'system')
    for key, split in (('structure_element', 'element'), ('structure_elementP', 'elementP')):
        if A.get(key):
            L += structure_section(A[key], split)
    if A.get('structure_systemT'):
        L += structure_section(A['structure_systemT'], 'systemT', ref=A.get('structure_system'), tiefix=A.get('tiefix'))
    if A.get('tiefix'):
        L += tiefix_section(A['tiefix'])
    if A.get('finetune'):
        L += finetune_section(A['finetune'])
    if A.get('evolution'):
        L += evolution_section(A['evolution'])
    if A.get('intervals'):
        L += intervals_section(A['intervals'])
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
          '- Crystal properties on the formula split: crystals of the same structure type with different elements can sit on both sides, so expect the real error on genuinely new chemistries to be larger. The held-out-systems section (when present) is the harder test; even it keeps every element and most sub-systems in training, so it is not a "new element" test.',
          '- Each comparison is one of many; with a 95% interval, a few "clear" results are expected by chance. Treat single clear wins as leads to confirm, not as findings.',
          *rule_caveat(A, smoke, bool(plan.get('nightly'))),
          '- Rows of a family that share a source (paper, alloy system, mix) were always kept together in a fold.', '']
    nxt = []
    if sv.count('network clearly better'):
        names = [STRUCT_NAMES.get(t['target'], t['target']) for t, v in zip(str_ok, sv) if v == 'network clearly better']
        nxt.append('On the formula split the average of crystal-network runs clearly beats the strongest trees on: ' + ', '.join(names) + '. Check the "single runs" column before crediting the network itself'
                   + ('' if sys_ok else ', and test on held-out chemical systems against structure-feature trees') + '.')
    elif str_ok:
        nxt.append('On the formula split no crystal property shows a clear win over the strongest trees. Use trees for these properties, or change what the network is given.')
    if sys_ok:
        nxt.append(f'Held-out chemical systems (networks by rule: {sys_rules}): network average clearly better on ' + (', '.join(STRUCT_NAMES.get(t['target'], t['target']) for t, v in zip(sys_ok, yv) if v == 'network clearly better') or 'none')
                   + '; trees clearly better on ' + (', '.join(STRUCT_NAMES.get(t['target'], t['target']) for t, v in zip(sys_ok, yv) if v == 'trees clearly better') or 'none')
                   + '. This is the comparison to quote' + ('; the original-rule section is kept for reference' if sys_fixed and orig_is_first else '')
                   + ('. Properties marked original rule have no finished fixed-rule retrain in this folder, so their verdict still includes the order-dependent graphs.' if 'original rule' in sys_rules else '.'))
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
