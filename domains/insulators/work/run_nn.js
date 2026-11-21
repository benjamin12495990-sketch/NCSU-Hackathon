// Trains and evaluates the from-scratch network for the insulators family.
// Targets are DFT-CALCULATED dielectric constants (log10 poly_total), not lab measurements.
// Usage: node run_nn.js <work_dir> <budget_seconds>
import fs from 'node:fs'
import path from 'node:path'
import { train } from './nn_generic.js'

const W = process.argv[2]
const BUDGET = Number(process.argv[3] || 400)
const T0 = Date.now()
const secs = () => (Date.now() - T0) / 1000
const read = (f) => JSON.parse(fs.readFileSync(path.join(W, f), 'utf8'))
const out = { note: 'Targets are DFT-calculated dielectric constants, not lab measurements. Predictions are log10(poly_total).' }
const save = () => fs.writeFileSync(path.join(W, 'nn_out.json'), JSON.stringify(out))
const mae = (p, y) => p.reduce((s, v, i) => s + Math.abs(v - y[i]), 0) / y.length
const pick = (a, idx) => idx.map((i) => a[i])

const data = read('nn_data.json')
const folds = read('cv_folds.json').folds
const Xtr = data.train.X, ytr = data.train.y, Xte = data.test.X, yte = data.test.y
const n = Xtr.length

// ---- (a) 3-fold grouped cross-validation on TRAINING rows only, single network per setting ----
const grid = [
  { name: 'A 64-32 squared wd1e-3', hidden: [64, 32], lr: 0.004, decay: 1e-3, epochs: 150, loss: 'squared' },
  { name: 'B 64-32 huber wd1e-3', hidden: [64, 32], lr: 0.004, decay: 1e-3, epochs: 150, loss: 'huber' },
  { name: 'C 64-32 huber wd1e-2', hidden: [64, 32], lr: 0.004, decay: 1e-2, epochs: 150, loss: 'huber' },
  { name: 'D 32-16 huber wd1e-2', hidden: [32, 16], lr: 0.004, decay: 1e-2, epochs: 150, loss: 'huber' },
  { name: 'E 32 huber wd1e-2', hidden: [32], lr: 0.004, decay: 1e-2, epochs: 150, loss: 'huber' },
  { name: 'F 64-32 huber wd3e-2 lr.002', hidden: [64, 32], lr: 0.002, decay: 3e-2, epochs: 150, loss: 'huber' },
]
const cvPredict = (cfg, members) => {
  const oof = new Array(n).fill(0)
  for (const fold of folds) {
    const held = new Set(fold)
    const trIdx = []
    for (let i = 0; i < n; i++) if (!held.has(i)) trIdx.push(i)
    const model = train(pick(Xtr, trIdx), pick(ytr, trIdx), { ...cfg, batch: 32, seed: 11, members })
    for (const i of fold) oof[i] = model.predict(Xtr[i])
  }
  return oof
}
out.cv_grid = []
for (const cfg of grid) {
  const t = secs()
  const oof = cvPredict(cfg, 1)
  const r = { ...cfg, cv_mae_log10: mae(oof, ytr), seconds: +(secs() - t).toFixed(2) }
  out.cv_grid.push(r)
  console.log('cv', r.name, r.cv_mae_log10.toFixed(4), r.seconds + 's')
}
const best = out.cv_grid.reduce((a, b) => (b.cv_mae_log10 < a.cv_mae_log10 ? b : a))
const { name, cv_mae_log10, seconds, ...cfg } = best
out.chosen = { ...cfg, batch: 32, seed: 11, chosen_by: '3-fold grouped CV MAE (log10) on training rows, single network', single_net_cv_mae_log10: cv_mae_log10, name }
save()

// ---- out-of-fold predictions of the 5-seed ensemble with the chosen settings (for the 90% interval) ----
{
  const t = secs()
  const oof = cvPredict(cfg, 5)
  out.ensemble_cv = { members: 5, cv_mae_log10: mae(oof, ytr), oof_pred: oof, seconds: +(secs() - t).toFixed(2) }
  console.log('ensemble cv', out.ensemble_cv.cv_mae_log10.toFixed(4), out.ensemble_cv.seconds + 's')
}
const members = out.ensemble_cv.cv_mae_log10 <= cv_mae_log10 ? 5 : 1
out.chosen.members = members
out.chosen.members_reason = `ensemble-of-5 CV MAE ${out.ensemble_cv.cv_mae_log10.toFixed(4)} vs single ${cv_mae_log10.toFixed(4)}`
save()

// ---- final network on all training rows; the test split is predicted once ----
{
  const t = secs()
  const model = train(Xtr, ytr, { ...cfg, batch: 32, seed: 11, members })
  out.final = { train_seconds: +(secs() - t).toFixed(2), layers: model.layers, weights: model.weights, members,
    test_pred: Xte.map((x) => model.predict(x)) }
  if (members === 5) { // if the ensemble was chosen its OOF predictions feed the interval; else redo single OOF
    out.final.oof_pred = out.ensemble_cv.oof_pred
  } else {
    out.final.oof_pred = cvPredict(cfg, 1)
  }
  console.log('final test mae', mae(out.final.test_pred, yte).toFixed(4), out.final.train_seconds + 's')
  save()
}

// ---- second model: same settings, inputs plus calculated band gap ----
{
  const bg = read('nn_data_bandgap.json')
  const t = secs()
  const model = train(bg.train.X, bg.train.y, { ...cfg, batch: 32, seed: 11, members })
  const grid = Array.from({ length: 9 }, (_, i) => 0.5 + 0.75 * i) // eV, same grid as trees.py
  const last = bg.features.length - 1
  out.bandgap = { train_seconds: +(secs() - t).toFixed(2), test_pred: bg.test.X.map((x) => model.predict(x)), sweep_grid_eV: grid,
    sweep_pred: grid.map((v) => bg.test.X.map((x) => { const r = x.slice(); r[last] = v; return model.predict(r) })) }
  console.log('bandgap test mae', mae(out.bandgap.test_pred, bg.test.y).toFixed(4))
  save()
}

// ---- eight repeated grouped splits ----
{
  const all = read('all_rows.json')
  const splits = read('splits.json').splits
  // per-fit cost estimate from the final fit; drop to fewer members only if the time budget would be blown
  const perNet = out.final.train_seconds / members
  const left = BUDGET - secs()
  let m = members
  if (perNet * m * splits.length > left) m = Math.max(1, Math.min(members, Math.floor(left / (perNet * splits.length))))
  out.splits_members = m
  out.splits = []
  for (const s of splits) {
    const model = train(pick(all.X, s.train), pick(all.y, s.train), { ...cfg, batch: 32, seed: 11, members: m })
    const p = s.test.map((i) => model.predict(all.X[i]))
    const r = { seed: s.seed, mae_log10_network: mae(p, pick(all.y, s.test)) }
    out.splits.push(r)
    console.log('split', s.seed, r.mae_log10_network.toFixed(4), secs().toFixed(0) + 's')
    save()
  }
}
out.total_seconds = +secs().toFixed(1)
out.complete = true
save()
console.log('done', out.total_seconds)
