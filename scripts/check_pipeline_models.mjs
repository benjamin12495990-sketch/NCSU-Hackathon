// Checks that the browser walks the exported tree models exactly as scikit-learn predicted them (gpu/export_site.py writes
// `checks` rows with scikit-learn's own predictions). Run: npm run pipeline-check
import { readFileSync, existsSync } from 'node:fs'
import { predictTrees } from '../src/lib/pipeline.js'

const dir = new URL('../public/pipeline/', import.meta.url).pathname
let failed = 0
function check(file, toVector) {
  const path = dir + file
  if (!existsSync(path)) { console.log(`${file}: not exported, skipped`); return }
  const m = JSON.parse(readFileSync(path, 'utf8'))
  let worst = 0
  for (const c of m.checks) worst = Math.max(worst, Math.abs(predictTrees(m, toVector(m, c)) - c.pred))
  const ok = worst < 1e-6
  if (!ok) failed++
  console.log(`${file}: ${m.checks.length} check rows, largest difference ${worst.toExponential(2)} ${ok ? 'ok' : 'FAILED'}`)
}
check('corrosion_model.json', (m, c) => m.features.map((n) => c.x_nonzero[n] ?? 0))
check('electrolyte_model.json', (m, c) => m.elements.map((e) => c.fractions[e] ?? 0))
process.exit(failed ? 1 : 0)
