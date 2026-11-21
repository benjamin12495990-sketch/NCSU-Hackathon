// npm run lab:parity: trains src/lib/nn.js createTrainer and lab/net.mjs createNet side by side on the page's
// own split (public/model/nn_data.json) and requires identical predictions on every train and test row.
// Checks options {} (the page default), 'rich', and PANEL_DESIGN (the switch). Nothing here is archived.
import './allow_full_data.mjs'
import './guard.mjs'
import { readFileSync } from 'node:fs'
import { createTrainer, PANEL_DESIGN } from '../src/lib/nn.js'
import { createNet, fromWeights, PANEL_DESIGN as LAB_PANEL } from './net.mjs'

const nn = JSON.parse(readFileSync(new URL('../public/model/nn_data.json', import.meta.url), 'utf8'))
let failed = 0
const sameDesign = JSON.stringify(LAB_PANEL) === JSON.stringify(PANEL_DESIGN)
if (!sameDesign) failed++
console.log(`${sameDesign ? 'ok  ' : 'FAIL'} net.mjs PANEL_DESIGN is identical to src/lib/nn.js PANEL_DESIGN`)
for (const [name, options] of [['{} (page default)', {}], ["{ features: 'rich' }", { features: 'rich' }], ['PANEL_DESIGN', PANEL_DESIGN]]) {
  const t0 = Date.now()
  const page = createTrainer(nn, options); page.runEpochs(1e9)
  const lab = createNet(nn.train, options); lab.train()
  const rebuilt = fromWeights(JSON.parse(JSON.stringify(lab.getWeights())))
  let worst = 0, worstRebuilt = 0, abs = 0
  const rows = [...nn.train.X, ...nn.test.X]
  for (const r of rows) {
    const a = page.predict(r), b = lab.predict(r), c = rebuilt.predict(r)
    worst = Math.max(worst, Math.abs(a - b)); worstRebuilt = Math.max(worstRebuilt, Math.abs(b - c))
  }
  nn.test.X.forEach((r, i) => { abs += Math.abs(lab.predict(r) - nn.test.y[i]) })
  const pass = worst === 0 && worstRebuilt === 0
  if (!pass) failed++
  console.log(`${pass ? 'ok  ' : 'FAIL'} ${name}: ${rows.length} rows, largest |nn.js - net.mjs| = ${worst}, after save/load ${worstRebuilt}; page test MAE ${(abs / nn.test.y.length).toFixed(4)} MPa (nn.js history ${page.history.at(-1).mae.toFixed(4)}); ${((Date.now() - t0) / 1000).toFixed(1)} s`)
}
process.exit(failed ? 1 : 0)
