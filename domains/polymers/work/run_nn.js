// node run_nn.js <tasks.json> <out.json> <task,task,...> '<options json>'
// Trains one network (or ensemble) per task and writes its predictions.
import fs from 'node:fs'
import { createGenericEnsemble } from './nn_generic.js'
const [tasksPath, outPath, names, optJson] = process.argv.slice(2)
const all = JSON.parse(fs.readFileSync(tasksPath, 'utf8')).tasks
const opts = JSON.parse(optJson)
const out = {}
for (const name of names.split(',')) {
  const t = all[name]
  const t0 = performance.now()
  const model = createGenericEnsemble(t, opts)
  model.runEpochs(model.totalEpochs)
  const seconds = (performance.now() - t0) / 1000
  out[name] = {
    seconds, weights: model.weights, layers: model.layers,
    pred: t.test.X.map((r) => model.predict(r)),
    train_pred: t.train.X.map((r) => model.predict(r)),
    extra_pred: t.extra ? t.extra.X.map((r) => model.predict(r)) : null,
  }
}
fs.writeFileSync(outPath, JSON.stringify(out))
