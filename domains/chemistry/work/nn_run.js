// Usage: node nn_run.js <cv|final|splits> <all_rows.json> <splits.json> <outdir>
// cv:     3-fold grouped CV on main-training rows only; picks settings; never reads test rows' targets.
// final:  trains the chosen ensemble on main-training rows, predicts the held-out test rows, runs the sweeps.
// splits: retrains the chosen ensemble on each of the eight extra grouped splits.
import fs from 'node:fs'
import { createGenericEnsemble } from './nn_generic.js'
const [mode, allPath, splitsPath, outDir] = process.argv.slice(2)
const all = JSON.parse(fs.readFileSync(allPath, 'utf8'))
const fit = (X, y, cfg) => { const t0 = Date.now(); const m = createGenericEnsemble(X, y, cfg); m.runEpochs(cfg.epochs); return { model: m, sec: (Date.now() - t0) / 1000 } }
const tr = all.main_train, te = all.main_test

if (mode === 'cv') {
  const fold = all.cv_fold
  const cv = (cfg) => {
    const oof = new Array(tr.length); let sec = 0
    for (let f = 0; f < 3; f++) {
      const a = [], b = []
      tr.forEach((ri, k) => (fold[k] === f ? b : a).push(k))
      const r = fit(a.map((k) => all.X[tr[k]]), a.map((k) => all.y[tr[k]]), cfg); sec += r.sec
      for (const k of b) oof[k] = r.model.predict(all.X[tr[k]])
    }
    let abs = 0, sq = 0
    oof.forEach((p, k) => { const e = p - all.y[tr[k]]; abs += Math.abs(e); sq += e * e })
    return { cfg, mae: abs / tr.length, rmse: Math.sqrt(sq / tr.length), sec, oof }
  }
  const base = { epochs: 30, batch: 32, lr: 0.004, loss: 'huber', log1p: 'auto', clip: 5, members: 1, seed: 11 }
  const table = []
  const run = (extra) => { const r = cv({ ...base, ...extra }); table.push({ cfg: r.cfg, mae: r.mae, rmse: r.rmse, sec: r.sec }); console.log(JSON.stringify(table[table.length - 1])); return r }
  let best = null
  for (const hidden of [[32, 16], [64, 32], [96, 48]]) for (const decay of [1e-4, 1e-3]) { const r = run({ hidden, decay }); if (!best || r.mae < best.mae) best = r }
  const b1 = { hidden: best.cfg.hidden, decay: best.cfg.decay }
  for (const extra of [{ ...b1, log1p: 'none' }, { ...b1, loss: 'squared' }]) { const r = run(extra); if (r.mae < best.mae) best = r }
  const chosen = { ...best.cfg, members: 5 }
  const ens = cv(chosen)
  const absErr = ens.oof.map((p, k) => Math.abs(p - all.y[tr[k]])).sort((a, b) => a - b)
  const n = absErr.length, k90 = Math.min(n, Math.ceil((n + 1) * 0.9))
  fs.writeFileSync(outDir + '/cv_result.json', JSON.stringify({ table, chosen, single_net_cv: { mae: best.mae, rmse: best.rmse },
    ensemble_cv: { mae: ens.mae, rmse: ens.rmse, sec: ens.sec }, q90_abs_error: absErr[k90 - 1], n_cv_rows: n, oof: ens.oof }))
  console.log('chosen', JSON.stringify(chosen), 'ens cv', ens.mae, ens.rmse, 'q90', absErr[k90 - 1])
}

if (mode === 'final') {
  const { chosen } = JSON.parse(fs.readFileSync(outDir + '/cv_result.json', 'utf8'))
  const X = tr.map((i) => all.X[i]), y = tr.map((i) => all.y[i])
  const { model, sec } = fit(X, y, chosen)
  const pred = te.map((i) => model.predict(all.X[i]))
  const member_pred = model.nets.map((t) => te.map((i) => t.predict(all.X[i])))
  const sweeps = {}
  for (const name of ['MolLogP', 'MolWt']) {
    const j = all.features.indexOf(name)
    const col = X.map((r) => r[j]).sort((a, b) => a - b)
    const lo = col[Math.floor(0.05 * (col.length - 1))], hi = col[Math.floor(0.95 * (col.length - 1))]
    const G = 21, grid = Array.from({ length: G }, (_, g) => lo + ((hi - lo) * g) / (G - 1))
    const h = name === 'MolLogP' ? 0.5 : 25
    const meanCurve = new Array(G).fill(0), net = []
    let down = 0, mono = 0, monoUp = 0, local = 0
    for (const i of te) {
      const row = all.X[i].slice()
      const p = grid.map((v) => { row[j] = v; return model.predict(row) })
      p.forEach((v, g) => { meanCurve[g] += v / te.length })
      const d = p[G - 1] - p[0]; net.push(d)
      if (d < 0) down++
      if (p.every((v, g) => g === 0 || v <= p[g - 1] + 1e-9)) mono++
      if (p.every((v, g) => g === 0 || v >= p[g - 1] - 1e-9)) monoUp++
      row[j] = all.X[i][j] + h; const up = model.predict(row); row[j] = all.X[i][j] - h; const dn = model.predict(row)
      if (up < dn) local++
    }
    net.sort((a, b) => a - b)
    sweeps[name] = { range_5_to_95_pct_of_training: [lo, hi], grid_points: G, molecules: te.length,
      share_net_decrease_over_range: down / te.length, share_monotone_nonincreasing: mono / te.length, share_monotone_nondecreasing: monoUp / te.length,
      share_local_decrease_at_own_value: local / te.length, local_step: h,
      median_net_change_logS: net[Math.floor(net.length / 2)], median_slope_per_unit: net[Math.floor(net.length / 2)] / (hi - lo), mean_curve: meanCurve }
  }
  fs.writeFileSync(outDir + '/nn_final.json', JSON.stringify({ chosen, train_seconds: sec, train_rows: tr.length, weights: model.weights, layers: model.layers,
    log1p_columns: all.features.filter((_, i) => model.scaler.logCols[i]), scaler: { mean: model.scaler.mean, sd: model.scaler.sd, clip: model.scaler.clip },
    pred, member_pred, sweeps }))
  console.log('final trained in', sec, 's; log1p columns:', all.features.filter((_, i) => model.scaler.logCols[i]).join(','))
}

if (mode === 'splits') {
  const { chosen } = JSON.parse(fs.readFileSync(outDir + '/cv_result.json', 'utf8'))
  const { splits } = JSON.parse(fs.readFileSync(splitsPath, 'utf8'))
  const out = []
  for (const s of splits) {
    const { model, sec } = fit(s.train.map((i) => all.X[i]), s.train.map((i) => all.y[i]), chosen)
    out.push({ seed: s.seed, sec, pred: s.test.map((i) => model.predict(all.X[i])) })
    fs.writeFileSync(outDir + '/nn_splits.json', JSON.stringify(out))
    console.log('split', s.seed, 'done in', sec)
  }
}
