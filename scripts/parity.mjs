// npm run parity: checks that the browser code gives the same Tc as Python.
// For each of the 20 rows in parity.json it predicts twice, once from the
// stored feature vector and once from the formula text, and compares both
// with the Python prediction. Fails if any differs by more than 0.01 K.
import { readFileSync } from 'node:fs'
import { parseFormula } from '../src/lib/parseFormula.js'
import { featurize } from '../src/lib/featurize.js'
import { predict } from '../src/lib/predict.js'

const read = (name) => JSON.parse(readFileSync(new URL('../public/model/' + name, import.meta.url)))
const elements = read('elements.json'), model = read('model.json'), parity = read('parity.json')
let worst = 0, failed = 0
for (const row of parity) {
  const fromFeatures = predict(model, row.features)
  const fromFormula = predict(model, featurize(parseFormula(row.formula, elements), elements))
  const diff = Math.max(Math.abs(fromFeatures - row.prediction), Math.abs(fromFormula - row.prediction))
  worst = Math.max(worst, diff)
  const ok = diff <= 0.01
  if (!ok) failed++
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${row.formula.padEnd(34)} py ${row.prediction.toFixed(4)}  js ${fromFormula.toFixed(4)}`)
}
console.log(`\n${parity.length - failed}/${parity.length} match within 0.01 K (largest difference ${worst.toExponential(2)} K)`)
process.exit(failed ? 1 : 0)
