import { readFileSync } from 'node:fs'
const root = process.argv[2]
const { createEnsemble, createTrainer } = await import(root + '/src/lib/nn.js')
const d = JSON.parse(readFileSync(root + '/public/model/nn_data.json')), splits = JSON.parse(readFileSync(process.argv[3]))
const X = [...d.train.X, ...d.test.X], y = [...d.train.y, ...d.test.y]
const ens = [], single = []
for (const mask of splits) {
  const tr = { X: [], y: [] }, te = { X: [], y: [] }
  mask.forEach((m, i) => { const t = m ? te : tr; t.X.push(X[i]); t.y.push(y[i]) })
  const data = { features: d.features, train: tr, test: te }
  const e = createEnsemble(data, { features: 'rich', decay: 0.003, loss: 'huber', members: 5 }); while (!e.done) e.runEpochs(50)
  const s = createTrainer(data, {}); while (!s.done) s.runEpochs(50)
  ens.push(e.history[e.history.length - 1].mae); single.push(s.history[s.history.length - 1].mae)
  console.log(`split ${ens.length}: test rows ${te.y.length} | network ensemble ${ens[ens.length - 1].toFixed(2)} | single network ${single[single.length - 1].toFixed(2)}`)
}
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length
console.log(`NETWORK ENSEMBLE mean ${mean(ens).toFixed(3)} | SINGLE NETWORK mean ${mean(single).toFixed(3)}`)
console.log('ENS ' + JSON.stringify(ens.map((v) => +v.toFixed(3))))
