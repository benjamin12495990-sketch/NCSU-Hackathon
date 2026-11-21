// Scores four reference models, trained on TRAIN and scored on SELECT with the same reward as evolution:
//  (a) plain: the page's network, options {} (base inputs, 64-32, squared loss, 300 epochs, seed 11)
//  (b) physics: PANEL_DESIGN, the site's switch (four textbook laws as inputs, Huber, decay 0.003)
//  (c) trees: scikit-learn GradientBoostingRegressor with the site's settings (lab/trees.py)
//  (d) curve: no learning. Each mix's strength at age t is s28 * f(t), f(t) = exp(0.25 (1 - sqrt(28 / t)))
//      (fib/CEB-FIP hardening curve for ordinary cement). s28 is the least-squares fit of that one number to
//      the mix's own TRAIN rows (sum y f / sum f^2); a mix with no TRAIN rows gets s28 fitted to all TRAIN rows.
//      Also reported: curve_global, where every mix gets the all-TRAIN s28. (The literal "mean TRAIN strength
//      at the same age" cannot be computed: TRAIN has no rows at 14 or 28 days.)
// Models are saved once under lab/results/models/ and their SHA-256 written to lab/results/baselines_select.json.
import './guard.mjs'
import path from 'node:path'
import { readFileSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { gzipSync } from 'node:zlib'
import { LAB } from './guard.mjs'
import { writeSame, sha256 } from './io.mjs'
import { createNet, PANEL_DESIGN } from './net.mjs'
import { reward } from './reward.mjs'

const TRAIN = JSON.parse(readFileSync(path.join(LAB, 'data', 'train.json'), 'utf8'))
const SELECT = JSON.parse(readFileSync(path.join(LAB, 'data', 'select.json'), 'utf8'))
const MODELS = path.join(LAB, 'results', 'models')
const out = { note: 'Trained on TRAIN (age <= 7 d, 253 rows), scored on SELECT (14-28 d, 481 rows). reward = -MAE - lambda * violations.', models: {} }

function saveNet(name, net) {
  const text = JSON.stringify(net.getWeights())
  writeSame(path.join(MODELS, `${name}.json.gz`), gzipSync(text, { level: 9 }))
  return sha256(text)
}
for (const [name, options] of [['plain', {}], ['physics', PANEL_DESIGN]]) {
  const t0 = Date.now()
  const net = createNet(TRAIN, options); net.train()
  out.models[name] = { kind: 'network', options, file: `results/models/${name}.json.gz`, sha256: saveNet(name, net), ...reward(net.predictMany, TRAIN, SELECT) }
  console.log(`${name}: SELECT MAE ${out.models[name].selectMae.toFixed(3)} MPa, violations ${out.models[name].violations.toFixed(3)}, reward ${out.models[name].reward.toFixed(3)} (${((Date.now() - t0) / 1000).toFixed(1)} s)`)
}

// (c) trees
const py = (args, input) => {
  const r = spawnSync('python3', ['-I', path.join(LAB, 'trees.py'), ...args], { input, encoding: 'utf8', maxBuffer: 1 << 26 })
  if (r.status !== 0) throw new Error(`trees.py ${args[0]} failed: ${r.stderr}`)
  return JSON.parse(r.stdout)
}
const fit = py(['fit'])
const treesPredict = (rows) => py(['predict', fit.sha256], JSON.stringify({ X: rows }))
out.models.trees = { kind: 'trees', file: 'results/models/trees.pkl', sha256: fit.sha256, params: fit.params, sklearn: fit.sklearn, ...reward(treesPredict, TRAIN, SELECT) }
console.log(`trees: SELECT MAE ${out.models.trees.selectMae.toFixed(3)} MPa, violations ${out.models.trees.violations.toFixed(3)}, reward ${out.models.trees.reward.toFixed(3)}`)

// (d) curve
const f = (t) => Math.exp(0.25 * (1 - Math.sqrt(28 / t)))
const fitS28 = (idx) => idx.reduce((s, i) => s + TRAIN.y[i] * f(TRAIN.X[i][7]), 0) / idx.reduce((s, i) => s + f(TRAIN.X[i][7]) ** 2, 0)
const byMix = new Map()
TRAIN.mix.forEach((m, i) => (byMix.get(m) || byMix.set(m, []).get(m)).push(i))
const curve = { format: 'lab-curve-1', curve: 'exp(0.25 * (1 - sqrt(28 / t)))', globalS28: fitS28(TRAIN.y.map((_, i) => i)), perMixS28: Object.fromEntries([...byMix].map(([m, idx]) => [m, fitS28(idx)])) }
const mixOf = (x) => x.slice(0, 7).join(',')
export const curvePredict = (c, perMix) => (rows) => rows.map((x) => (perMix && mixOf(x) in c.perMixS28 ? c.perMixS28[mixOf(x)] : c.globalS28) * f(x[7]))
const curveText = JSON.stringify(curve)
writeSame(path.join(MODELS, 'curve.json'), curveText)
for (const [name, perMix] of [['curve', true], ['curve_global', false]]) {
  out.models[name] = { kind: 'curve', perMix, file: 'results/models/curve.json', sha256: sha256(curveText), ...reward(curvePredict(curve, perMix), TRAIN, SELECT) }
  console.log(`${name}: SELECT MAE ${out.models[name].selectMae.toFixed(3)} MPa, violations ${out.models[name].violations.toFixed(3)}, reward ${out.models[name].reward.toFixed(3)}`)
}
const seenShare = SELECT.mix.filter((m) => byMix.has(m)).length / SELECT.y.length
out.selectRowsWhoseMixIsInTrain = Number(seenShare.toFixed(4))
writeSame(path.join(LAB, 'results', 'baselines_select.json'), JSON.stringify(out, null, 1) + '\n')
console.log(`SELECT rows whose exact mix has TRAIN rows: ${(100 * seenShare).toFixed(1)}%. Wrote lab/results/baselines_select.json`)
