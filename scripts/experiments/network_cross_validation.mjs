// Chooses the network's settings by cross-validation on TRAINING rows only (whole mixes kept together).
import { readFileSync } from 'node:fs'
const root = process.argv[2]
const { createTrainer, createEnsemble } = await import(root + '/src/lib/nn.js')
const data = JSON.parse(readFileSync(root + '/public/model/nn_data.json'))
const { X, y, g } = data.train
const groups = [...new Set(g)]
// deterministic shuffle of groups, then 3 folds
let a = 12345; const rnd = () => { a = (a * 1664525 + 1013904223) >>> 0; return a / 4294967296 }
for (let i = groups.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [groups[i], groups[j]] = [groups[j], groups[i]] }
const K = 3, foldOf = new Map(groups.map((gr, i) => [gr, i % K]))
const fold = (k) => { const tr = { X: [], y: [] }, va = { X: [], y: [] }; X.forEach((x, i) => { const t = foldOf.get(g[i]) === k ? va : tr; t.X.push(x); t.y.push(y[i]) }); return { features: data.features, train: tr, test: va } }
const configs = JSON.parse(process.argv[3])
for (const cfg of configs) {
  const t0 = performance.now(); const maes = []
  for (let k = 0; k < K; k++) { const t = (cfg.members ? createEnsemble : createTrainer)(fold(k), cfg); while (!t.done) t.runEpochs(50); maes.push(t.history[t.history.length - 1].mae) }
  console.log(`${JSON.stringify(cfg).padEnd(74)} CV MAE ${(maes.reduce((s, v) => s + v, 0) / K).toFixed(3)}  [${maes.map((v) => v.toFixed(2)).join(', ')}]  ${((performance.now() - t0) / 1000).toFixed(0)} s`)
}
