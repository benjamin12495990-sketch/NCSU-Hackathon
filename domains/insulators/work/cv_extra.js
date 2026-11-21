// Follow-up cross-validation on TRAINING rows only, run AFTER the test split had been scored once.
// It does not touch the test split and was NOT used to choose the reported network.
// Purpose: the chosen setting sat on the edge of the first grid, so this shows whether more
// regularisation would help. Targets are DFT-calculated dielectric constants, not measurements.
// Usage: node cv_extra.js <work_dir>
import fs from 'node:fs'
import path from 'node:path'
import { train } from './nn_generic.js'

const W = process.argv[2]
const read = (f) => JSON.parse(fs.readFileSync(path.join(W, f), 'utf8'))
const data = read('nn_data.json'), folds = read('cv_folds.json').folds
const X = data.train.X, y = data.train.y, n = X.length
const grid = [
  { name: 'G 64-32 huber wd1e-1 lr.002', hidden: [64, 32], lr: 0.002, decay: 1e-1, epochs: 150, loss: 'huber' },
  { name: 'H 32 huber wd3e-2 lr.002', hidden: [32], lr: 0.002, decay: 3e-2, epochs: 150, loss: 'huber' },
  { name: 'I 32 huber wd3e-2 lr.004', hidden: [32], lr: 0.004, decay: 3e-2, epochs: 150, loss: 'huber' },
  { name: 'J 64-32 huber wd3e-2 lr.002 300ep', hidden: [64, 32], lr: 0.002, decay: 3e-2, epochs: 300, loss: 'huber' },
  { name: 'K 64-32 squared wd3e-2 lr.002', hidden: [64, 32], lr: 0.002, decay: 3e-2, epochs: 150, loss: 'squared' },
  { name: 'L 16 huber wd1e-2 lr.004', hidden: [16], lr: 0.004, decay: 1e-2, epochs: 150, loss: 'huber' },
]
const out = { note: 'CV on training rows only, run after the single test evaluation; not used for model choice; no test numbers here. Targets are DFT-calculated, not measured.', grid: [] }
for (const cfg of grid) {
  let abs = 0
  for (const fold of folds) {
    const held = new Set(fold), idx = []
    for (let i = 0; i < n; i++) if (!held.has(i)) idx.push(i)
    const m = train(idx.map((i) => X[i]), idx.map((i) => y[i]), { ...cfg, batch: 32, seed: 11, members: 1 })
    for (const i of fold) abs += Math.abs(m.predict(X[i]) - y[i])
  }
  out.grid.push({ ...cfg, cv_mae_log10: +(abs / n).toFixed(4) })
  console.log(cfg.name, (abs / n).toFixed(4))
}
fs.writeFileSync(path.join(W, 'nn_cv_extra.json'), JSON.stringify(out))
