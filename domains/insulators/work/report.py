#!/usr/bin/env python3
"""Combine network (nn_out.json) and tree (trees.json) outputs into result.json.
Targets are DFT-CALCULATED dielectric constants, not lab measurements.
Usage: python3 -I report.py <work_dir>
"""
import json, math, sys
import numpy as np
from scipy import stats

W = sys.argv[1]
J = lambda f: json.load(open(f"{W}/{f}"))
nn, tr, data, databg, allr = J('nn_out.json'), J('trees.json'), J('nn_data.json'), J('nn_data_bandgap.json'), J('all_rows.json')
ytr = np.array(data['train']['y']); yte = np.array(data['test']['y'])
bg_te = np.array(databg['test']['X'])[:, -1]
r4 = lambda v: round(float(v), 4)


def metrics(pred, y=yte):
    pred = np.asarray(pred, float); e = pred - y
    ae_units = np.abs(10 ** pred - 10 ** y)
    return {"mae_log10": r4(np.abs(e).mean()), "r2_log10": r4(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "median_abs_error_dielectric_units": r4(np.median(ae_units)), "mean_abs_error_dielectric_units": r4(ae_units.mean()),
            "typical_factor_off_10^mae": r4(10 ** np.abs(e).mean())}


naive = np.full(len(yte), tr['naive_train_mean'])
res = {}
res["IMPORTANT"] = ("All target values are CALCULATED with density functional theory (DFPT, Materials Project; Petousis et al.), "
                    "NOT measured in a lab. Every error below is error against that calculation, not against experiment. "
                    "The model is a fast stand-in for a slow calculation.")
res["numbers_run_by"] = "Every number in this file was computed in this session by the scripts listed under files_written (no number is quoted from a paper)."
res["data"] = {"source": "Hugging Face foundry-ml/dielectric_constant_v1-1 (CC BY 4.0), Petousis et al., DOI 10.18126/racd-go9m",
               "rows_used": len(allr['y']), "groups_reduced_formulas": len(allr['group_names']),
               "train_rows": len(ytr), "test_rows": len(yte),
               "train_groups": len(set(data['train']['g'])), "test_groups": len(set(data['test']['g'])),
               "target": "y = log10(poly_total); poly_total is the DFT-calculated polycrystalline total dielectric constant",
               "split": "GroupShuffleSplit 80/20 by reduced formula, random_state=42; 8 more grouped splits seeds 1001-1008"}
res["features_main"] = data['features']
res["features_second_model"] = "features_main + band_gap_eV_calculated (Materials Project calculated band gap)"
res["excluded_inputs"] = "n, poly_electronic, pot_ferroelectric (outputs of the same calculation as the target); crystal-structure JSON not used"
res["element_table_source"] = ("Typed by hand in prepare.py for the 63 elements present, from memory (nothing downloaded): atomic number, period, group = periodic table; "
                               "Pauling electronegativity as in the CRC Handbook / Wikipedia data page; IUPAC abridged standard atomic weights; covalent radii "
                               "from Cordero et al., Dalton Trans. 2008 (C sp3; Mn/Fe/Co low-spin); valence electrons DERIVED from group (group for 1-12, group-10 for 13-17). "
                               "Checked in code: molar masses of NaCl, SiO2, BaTiO3, Al2O3, Mg(OH)2 to 0.02 g/mol; Z/period/group ordering; electronegativity and radius trends. "
                               "Not independently verified against a reference file tonight.")

ch = dict(nn['chosen'])
res["network_settings_chosen"] = ch
res["cv_on_training_rows_3fold_grouped"] = {
    "grid_single_network": [{k: g[k] for k in ('name', 'hidden', 'lr', 'decay', 'epochs', 'loss', 'cv_mae_log10', 'seconds')} for g in nn['cv_grid']],
    "network_ensemble5_cv_mae_log10": r4(nn['ensemble_cv']['cv_mae_log10']),
    "trees_cv_mae_log10_fixed_settings": r4(tr['cv_mae_log10_main']), "naive_cv_mae_log10": r4(tr['cv_mae_log10_naive'])}
for g in res["cv_on_training_rows_3fold_grouped"]["grid_single_network"]:
    g['cv_mae_log10'] = r4(g['cv_mae_log10'])

if 'final' in nn:
    pn = np.array(nn['final']['test_pred']); pt = np.array(tr['test_pred_main'])
    res["b_test_split_once"] = {"network": metrics(pn), "trees_gradient_boosting": metrics(pt), "naive_training_mean": metrics(naive)}
    hi = 10 ** yte > 30
    res["b_test_split_once"]["high_tail_rows_true_value_above_30"] = {
        "rows": int(hi.sum()), "network_mae_log10": r4(np.abs(pn - yte)[hi].mean()), "trees_mae_log10": r4(np.abs(pt - yte)[hi].mean()),
        "network_median_predicted_over_true": r4(np.median(10 ** (pn - yte)[hi])), "trees_median_predicted_over_true": r4(np.median(10 ** (pt - yte)[hi]))}
    res["final_network"] = {"training_seconds": nn['final']['train_seconds'], "layers": nn['final']['layers'], "weights_total": nn['final']['weights'],
                            "members": nn['final']['members'], "runtime": "Node, single thread, niced, on a shared 4-CPU machine",
                            "tree_fit_seconds": tr['fit_seconds_main']}
    # (d) 90% interval from CV absolute errors on training rows (split-conformal style quantile)
    oof = np.array(nn['final']['oof_pred']); a = np.sort(np.abs(oof - ytr)); n = len(a)
    k = min(n, math.ceil((n + 1) * 0.9)); q = a[k - 1]
    inside = np.abs(pn - yte) <= q
    res["d_interval_90"] = {"method": "absolute-error quantile of 3-fold grouped out-of-fold errors on training rows (rank ceil((n+1)*0.9) of n), applied as prediction +/- q in log10 units around the final network",
                            "half_width_log10": r4(q), "multiplicative_factor": r4(10 ** q),
                            "reads_as": f"calculated value expected between prediction/{10 ** q:.2f} and prediction*{10 ** q:.2f}",
                            "test_coverage": r4(inside.mean()), "test_rows_inside": int(inside.sum()), "test_rows": len(yte), "nominal": 0.9,
                            "test_coverage_rows_true_value_above_30": r4(inside[hi].mean()), "rows_above_30": int(hi.sum()),
                            "test_coverage_rows_true_value_30_or_below": r4(inside[~hi].mean()),
                            "largest_test_prediction_dielectric_units": r4(10 ** pn.max()), "largest_test_true_value": r4(10 ** yte.max()),
                            "warning": "coverage is an average over compounds like those in this dataset; it fails badly on the high-dielectric tail, which the network systematically under-predicts"}

if nn.get('splits'):
    a = np.array([s['mae_log10_network'] for s in nn['splits']])
    b = np.array([s['mae_log10_trees'] for s in tr['splits']][:len(a)])
    nv = np.array([s['mae_log10_naive'] for s in tr['splits']][:len(a)])
    d = a - b; m = d.mean(); se = d.std(ddof=1) / math.sqrt(len(d)); tcrit = stats.t.ppf(0.975, len(d) - 1)
    res["c_eight_grouped_splits"] = {"seeds": [s['seed'] for s in nn['splits']], "network_members_per_split": nn.get('splits_members'),
                                     "mae_log10_network": [r4(v) for v in a], "mae_log10_trees": [r4(v) for v in b], "mae_log10_naive": [r4(v) for v in nv],
                                     "mean_network": r4(a.mean()), "mean_trees": r4(b.mean()), "mean_naive": r4(nv.mean()),
                                     "mean_difference_network_minus_trees": r4(m), "paired_t_95_interval": [r4(m - tcrit * se), r4(m + tcrit * se)],
                                     "splits_completed": len(a), "network_better_on_splits": int((d < 0).sum()),
                                     "reading": ("negative difference = network better. " + ("Interval excludes zero." if (m - tcrit * se) * (m + tcrit * se) > 0 else "Interval includes zero: no clear winner.")),
                                     "caveat": "the eight test sets overlap, so the splits are not independent and the t interval is somewhat too narrow"}

if 'bandgap' in nn:
    pnb = np.array(nn['bandgap']['test_pred']); ptb = np.array(tr['test_pred_bandgap'])
    res["e_second_model_with_band_gap"] = {"network": metrics(pnb), "trees_gradient_boosting": metrics(ptb),
                                           "network_settings": "same as the main network (not re-tuned)", "tree_band_gap_importance": r4(tr['band_gap_importance'])}
    if 'final' in nn:
        res["e_second_model_with_band_gap"]["gain_mae_log10_network"] = r4(metrics(pn)['mae_log10'] - metrics(pnb)['mae_log10'])
        res["e_second_model_with_band_gap"]["gain_mae_log10_trees"] = r4(metrics(pt)['mae_log10'] - metrics(ptb)['mae_log10'])

    def sweep_stats(sw):
        sw = np.array(sw)  # grid x rows
        steps = np.diff(sw, axis=0)
        return {"share_rows_prediction_lower_at_6.5eV_than_0.5eV": r4((sw[-1] < sw[0]).mean()),
                "share_rows_falling_at_every_step": r4((steps < 0).all(axis=0).mean()),
                "share_rows_never_rising_at_any_step": r4((steps <= 1e-12).all(axis=0).mean()),
                "share_of_all_steps_falling": r4((steps < 0).mean()),
                "median_change_log10_per_eV": r4(np.median((sw[-1] - sw[0]) / (6.5 - 0.5)))}
    sp = lambda p: r4(stats.spearmanr(p, bg_te).statistic)
    res["sanity_check_band_gap_trend"] = {
        "expectation": "dielectric constant should fall as band gap rises",
        "reference_true_target_vs_band_gap_spearman_test": sp(yte),
        "main_model_no_band_gap_input": {"network_prediction_vs_true_band_gap_spearman": sp(pn) if 'final' in nn else None,
                                         "trees_prediction_vs_true_band_gap_spearman": sp(pt) if 'final' in nn else None},
        "second_model_sweep": {"grid_eV": nn['bandgap']['sweep_grid_eV'], "rows": len(yte),
                               "training_band_gap_quantiles_5_50_95_eV": tr['train_band_gap_quantiles_5_50_95'],
                               "network": sweep_stats(nn['bandgap']['sweep_pred']), "trees": sweep_stats(tr['sweep_pred_bandgap']),
                               "note": "band gap is swept on real test rows with every other input held fixed; the top of the grid (above about 5.4 eV) is beyond 95% of training rows"}}

res["nn_run_complete"] = bool(nn.get('complete'))
res["nn_total_seconds_all_stages"] = nn.get('total_seconds')
import os
if os.path.exists(f"{W}/ablation.json"):
    res["ablation_trees_cv_training_rows_only"] = J('ablation.json')
if os.path.exists(f"{W}/nn_cv_extra.json"):
    ex = J('nn_cv_extra.json')
    res["followup_cv_grid_training_rows_only_not_used_for_choice"] = {"note": ex['note'], "grid": [{"name": g['name'], "cv_mae_log10": g['cv_mae_log10']} for g in ex['grid']]}
res["files_written"] = [f"{W}/{f}" for f in ['prepare.py', 'nn_generic.js', 'run_nn.js', 'trees.py', 'report.py', 'cv_extra.js', 'ablation.py', 'package.json',
                                             'nn_data.json', 'nn_data_bandgap.json', 'splits.json', 'cv_folds.json', 'all_rows.json', 'trees.json', 'nn_out.json',
                                             'nn_cv_extra.json', 'ablation.json', 'extras.json', 'trees.log', 'nn.log', 'cv_extra.log', 'report.log', 'result.json', '.t0']]
extra = sys.argv[2] if len(sys.argv) > 2 else None
if extra:
    res.update(json.load(open(extra)))
json.dump(res, open(f"{W}/result.json", 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('features_main', 'files_written')}, indent=1))
