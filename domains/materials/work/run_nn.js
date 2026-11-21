// Usage: node run_nn.js cv <dir>      settings search by 3-fold grouped CV on training rows only
//        node run_nn.js final <dir>   one evaluation on the held-out test split, the eight extra splits, the Al sweep
import { readFileSync, writeFileSync } from 'node:fs'
import { createTrainer, createEnsemble, metrics } from './nn_generic.js'

const [mode, dir] = process.argv.slice(2)
const read = (name) => JSON.parse(readFileSync(`${dir}/${name}`, 'utf8'))
const data = read('nn_data.json')

function fit(train, settings, members) {
  const model = members > 1 ? createEnsemble(train, { ...settings, members }) : createTrainer(train, settings)
  model.runEpochs(settings.epochs)
  return model
}

// Out-of-fold predictions for the training rows.
function crossValidate(settings, members, folds) {
  const { X, y } = data.train
  const pred = new Array(y.length)
  const perFold = []
  for (let f = 0; f < 3; f++) {
    const tr = { X: [], y: [] }, held = []
    for (let i = 0; i < y.length; i++) {
      if (folds[i] === f) held.push(i)
      else { tr.X.push(X[i]); tr.y.push(y[i]) }
    }
    const model = fit(tr, settings, members)
    for (const i of held) pred[i] = model.predict(X[i])
    perFold.push(metrics(held.map((i) => pred[i]), held.map((i) => y[i])).mae)
  }
  return { ...metrics(pred, y), perFold, pred }
}

if (mode === 'cv') {
  const folds = read('cv_folds.json').fold
  // A first pass without guards (log kept in cv_unguarded.log) gave absurd
  // out-of-fold predictions for a few unusual alloys, so the guards are on here.
  const guard = process.argv[4] === 'unguarded' ? {} : { inputClip: 3, clampOutput: true }
  const base = { epochs: 300, batch: 32, lr: 0.004, seed: 11, loss: 'squared', ...guard }
  const grid = []
  for (const hidden of [[32, 16], [64, 32], [128, 64]])
    for (const decay of [1e-3, 1e-2, 5e-2])
      for (const logTarget of [false, true]) grid.push({ ...base, hidden, decay, logTarget })
  const table = []
  for (const s of grid) {
    const r = crossValidate(s, 1, folds)
    table.push({ settings: s, members: 1, cv_mae: r.mae, cv_rmse: r.rmse, cv_r2: r.r2, per_fold_mae: r.perFold })
    console.log(JSON.stringify(s.hidden), s.decay, s.logTarget ? 'log' : 'raw', 'MAE', r.mae.toFixed(1), 'R2', r.r2.toFixed(3))
  }
  let best = table.reduce((a, b) => (b.cv_mae < a.cv_mae ? b : a))
  // One more question for the winner: does Huber loss help?
  const hub = { ...best.settings, loss: 'huber' }
  const rh = crossValidate(hub, 1, folds)
  const hubRow = { settings: hub, members: 1, cv_mae: rh.mae, cv_rmse: rh.rmse, cv_r2: rh.r2, per_fold_mae: rh.perFold }
  table.push(hubRow)
  console.log('huber on winner: MAE', rh.mae.toFixed(1))
  if (hubRow.cv_mae < best.cv_mae) best = hubRow
  // Ensemble of five with the chosen settings: its out-of-fold errors give the 90% interval.
  const re = crossValidate(best.settings, 5, folds)
  const absErr = re.pred.map((p, i) => Math.abs(p - data.train.y[i])).sort((a, b) => a - b)
  const n = absErr.length, k = Math.min(n, Math.ceil((n + 1) * 0.9))
  const out = {
    chosen: best.settings, members: 5,
    chosen_single_net_cv: { mae: best.cv_mae, rmse: best.cv_rmse, r2: best.cv_r2 },
    ensemble_cv: { mae: re.mae, rmse: re.rmse, r2: re.r2, per_fold_mae: re.perFold },
    interval90_halfwidth_HV: absErr[k - 1], interval_rule: `the ${k}th smallest of ${n} out-of-fold absolute errors (ceil((n+1)*0.9))`,
    oof_pred: re.pred, grid: table,
  }
  writeFileSync(`${dir}/cv_results.json`, JSON.stringify(out))
  console.log('chosen', JSON.stringify(best.settings), 'ensemble CV MAE', re.mae.toFixed(1), 'R2', re.r2.toFixed(3), 'q90', absErr[k - 1].toFixed(1))
}

// The chosen settings under a different fold file (training rows only), e.g. whole alloy systems held out.
if (mode === 'cvfile') {
  const cv = read('cv_results.json')
  const r = crossValidate(cv.chosen, cv.members, read(process.argv[4]).fold)
  writeFileSync(`${dir}/cv_system_nn.json`, JSON.stringify({ folds: process.argv[4], settings: cv.chosen, members: cv.members, mae: r.mae, rmse: r.rmse, r2: r.r2, per_fold_mae: r.perFold }))
  console.log('cvfile', process.argv[4], 'MAE', r.mae.toFixed(1), 'R2', r.r2.toFixed(3), r.perFold.map((v) => v.toFixed(1)).join(' '))
}

if (mode === 'final') {
  const cv = read('cv_results.json')
  const settings = cv.chosen, members = cv.members
  const t0 = performance.now()
  const model = fit(data.train, settings, members)
  const seconds = (performance.now() - t0) / 1000
  const pred = data.test.X.map((x) => model.predict(x))
  const q = cv.interval90_halfwidth_HV
  const inside = pred.filter((p, i) => Math.abs(p - data.test.y[i]) <= q).length
  const trainPred = data.train.X.map((x) => model.predict(x))

  const all = read('splits.json')
  const splits = all.splits.map((s) => {
    const tr = { X: s.train.map((i) => all.X[i]), y: s.train.map((i) => all.y[i]) }
    const m = fit(tr, settings, members)
    const p = s.test.map((i) => m.predict(all.X[i]))
    return { seed: s.seed, ...metrics(p, s.test.map((i) => all.y[i])) }
  })

  const sw = read('sweep.json')
  const sweep = sw.sweep.map((r) => ({ al_x: r.al_x, al_fraction: r.al_fraction, hv_pred: model.predict(r.features) }))

  writeFileSync(`${dir}/nn_final.json`, JSON.stringify({
    settings, members, layers: model.layers, weights: model.weights, train_seconds: seconds,
    test: metrics(pred, data.test.y), train_fit: metrics(trainPred, data.train.y), test_pred: pred,
    interval90_halfwidth_HV: q, test_inside: inside, test_rows: pred.length, test_coverage: inside / pred.length,
    splits, sweep,
  }))
  console.log('test', JSON.stringify(metrics(pred, data.test.y)), 'seconds', seconds.toFixed(1), 'coverage', inside, '/', pred.length)
  console.log('splits MAE', splits.map((s) => s.mae.toFixed(1)).join(' '))
  console.log('sweep', sweep.map((s) => `${s.al_x}:${s.hv_pred.toFixed(0)}`).join(' '))
}
