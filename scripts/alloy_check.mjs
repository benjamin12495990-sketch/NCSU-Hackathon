// Checks for the alloy generator (npm run alloy-check).
// Part 1: featurize() rebuilds all 530 rows of the alloy network's nn_data.json from alloys.json (1e-6).
// Part 2: trains one network (nn_generic.js, the materials agent's settings, seed 11), runs the
//         generator for three targets plus one adversarial request, and asserts every rule on every
//         returned candidate.
// nn_data.json and nn_generic.js are read from public/model/materials/ and src/lib/ when another
// branch has put them there, otherwise from MATERIALS_SIM_DIR (default: tonight's training folder).
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { featurize } from '../src/lib/alloyFeatures.js'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const SIM = process.env.MATERIALS_SIM_DIR ||
  '/tmp/claude-0/-home-user/c96b0b28-08aa-572a-ab28-a200b90c8d07/scratchpad/burn/sim/materials'
const pick = (...paths) => paths.find((p) => existsSync(p))
const nnPath = pick(join(root, 'public/model/materials/nn_data.json'), join(SIM, 'nn_data.json'))
const resultPath = join(SIM, 'result.json')
const data = JSON.parse(readFileSync(join(root, 'public/model/materials/alloys.json'), 'utf8'))
const nn = JSON.parse(readFileSync(nnPath, 'utf8'))
let failures = 0
const fail = (msg) => { failures++; console.log('FAIL', msg) }

// ---- Part 1: feature parity -------------------------------------------------------------------
console.log(`Part 1: feature parity (alloys.json vs ${nnPath})`)
if (JSON.stringify(nn.features) !== JSON.stringify(data.features)) fail('feature lists differ')
const nnRows = [...nn.train.X.map((x, i) => ({ x, y: nn.train.y[i], g: nn.train.g[i], split: 'train' })),
  ...nn.test.X.map((x, i) => ({ x, y: nn.test.y[i], g: nn.test.g[i], split: 'test' }))]
if (nnRows.length !== data.rows.length) fail(`row count ${data.rows.length} vs ${nnRows.length}`)
let worst = 0, worstAt = ''
nnRows.forEach((ref, i) => {
  const row = data.rows[i]
  if (row.hv !== ref.y || row.g !== ref.g || row.split !== ref.split) fail(`row ${i}: hv/g/split differ`)
  const x = featurize(row.frac, row.proc, data)
  x.forEach((v, k) => {
    const d = Math.abs(v - ref.x[k])
    if (d > worst) { worst = d; worstAt = `row ${i} ${data.features[k]}` }
  })
})
const PARITY_TOL = 1e-6 + 1e-12
console.log(`  ${data.rows.length} rows x ${data.features.length} features; largest difference ${worst.toExponential(2)}${worst ? ' at ' + worstAt : ''}`)
if (worst > PARITY_TOL) fail(`feature parity: ${worst} > 1e-6`)
else console.log('  PASS feature parity at 1e-6')
const trainHv = nn.train.y
if (data.clamp_HV[0] !== Math.min(...trainHv) || data.clamp_HV[1] !== Math.max(...trainHv)) fail('clamp range is not the training HV range')
else console.log(`  PASS clamp range ${data.clamp_HV[0]}-${data.clamp_HV[1]} HV is the training range`)
if (failures) { console.log(`${failures} failure(s); part 2 not run`); process.exit(1) }

export { root, SIM, data, nn, resultPath, fail, pick, pathToFileURL }
