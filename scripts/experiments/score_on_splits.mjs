// Scores a network design on the eight SELECTION splits (80/20 by mix over all 1,005 lab results).
// Usage: node scripts/experiments/score_on_splits.mjs <path to nn.js> '<config JSON>' <splits.json>
//   config.members > 1 uses createEnsemble, otherwise createTrainer. Everything else is passed through.
// The network trains on each split's training rows only and is scored on that split's held-out rows.
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import { dirname, join } from 'node:path'
const here = dirname(new URL(import.meta.url).pathname)
const nn = await import(pathToFileURL(process.argv[2]).href)
const cfg = JSON.parse(process.argv[3] || '{}')
const splits = JSON.parse(readFileSync(process.argv[4]))
const d = JSON.parse(readFileSync(join(here, '../../public/model/nn_data.json')))
const X = [...d.train.X, ...d.test.X], y = [...d.train.y, ...d.test.y]
const maes = []; const t0 = performance.now()
for (const mask of splits) {
  const tr = { X: [], y: [] }, te = { X: [], y: [] }
  mask.forEach((m, i) => { const t = m ? te : tr; t.X.push(X[i]); t.y.push(y[i]) })
  const model = (cfg.members > 1 ? nn.createEnsemble : nn.createTrainer)({ features: d.features, train: tr, test: te }, cfg)
  while (!model.done) model.runEpochs(25)
  let abs = 0; for (let i = 0; i < te.y.length; i++) abs += Math.abs(model.predict(te.X[i]) - te.y[i])
  maes.push(abs / te.y.length)
}
const mean = maes.reduce((s, v) => s + v, 0) / maes.length
console.log(JSON.stringify({ config: cfg, per_split: maes.map((v) => +v.toFixed(3)), mean: +mean.toFixed(3), seconds_per_split: +(((performance.now() - t0) / 1000) / maes.length).toFixed(1) }))
