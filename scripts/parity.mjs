// npm run parity: checks that the browser code predicts the same strength as
// Python. For each of the 20 rows in parity.json it rebuilds the features from
// the mix and age, predicts with both models (expected strength and the
// cautious lower estimate), and compares with the Python predictions.
// Fails if any differs by more than 0.01 MPa.
import { readFileSync } from 'node:fs'
import { featurize } from '../src/lib/featurize.js'
import { predict } from '../src/lib/predict.js'

const read = (name) => JSON.parse(readFileSync(new URL('../public/model/' + name, import.meta.url)))
const spec = read('features.json'), model = read('model.json'), modelLow = read('model_low.json'), parity = read('parity.json')
let worst = 0, failed = 0
for (const row of parity) {
  const js = predict(model, featurize(row.mix, row.age, spec))
  const fromStored = predict(model, row.features)
  const jsLow = predict(modelLow, featurize(row.mix, row.age, spec))
  const diff = Math.max(Math.abs(js - row.prediction), Math.abs(fromStored - row.prediction), Math.abs(jsLow - row.prediction_low))
  worst = Math.max(worst, diff)
  if (diff > 0.01) failed++
  console.log(`${diff > 0.01 ? 'FAIL' : 'ok  '} age ${String(row.age).padStart(3)} d  python ${row.prediction.toFixed(4)}  javascript ${js.toFixed(4)} MPa`)
}
console.log(`\n${parity.length - failed}/${parity.length} match within 0.01 MPa (largest difference ${worst.toExponential(2)} MPa)`)
process.exit(failed ? 1 : 0)
