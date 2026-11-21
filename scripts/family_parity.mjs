// Parity gate for the family pages: trains each family's five-network ensemble twice on the same
// public/model/<family>/nn_data.json, once with src/lib/nnFamily.js and once with the family agent's
// own trainer (domains/<family>/nn_generic.js), and requires identical test predictions (max
// difference 1e-9) and the test MAE reported in that family's result.json (family.json test.network).
// Run: npm run family-parity [-- --family=materials]
import { readFileSync } from 'node:fs'
import { createFamilyEnsemble, createFamilyTrainer, familyMetrics } from '../src/lib/nnFamily.js'

const root = new URL('../', import.meta.url)
const read = (p) => JSON.parse(readFileSync(new URL(p, root), 'utf8'))
const only = (process.argv.find((a) => a.startsWith('--family=')) || '').split('=')[1]

// How each agent's trainer is called (their call shapes differ), and the MAE precision result.json reports.
const AGENT = {
  materials: { tol: 0.001, make: (m, data, o) => m.createEnsemble(data.train, o) },
  insulators: { tol: 0.00005, make: (m, data, o) => m.fitEnsemble(data.train.X, data.train.y, o) },
  polymers: { tol: 0.005, make: (m, data, o) => m.createGenericEnsemble(data, o) },
}

let failed = 0
for (const id of Object.keys(AGENT)) {
  if (only && only !== id) continue
  const fam = read(`public/model/${id}/family.json`)
  const data = read(`public/model/${id}/nn_data.json`)
  const opts = { ...fam.options, members: fam.members }
  const t0 = Date.now()
  const ours = createFamilyEnsemble(data, opts)
  ours.runEpochs(ours.totalEpochs)
  const t1 = Date.now()
  const agentModule = await import(new URL(`domains/${id}/nn_generic.js`, root))
  const theirs = AGENT[id].make(agentModule, data, opts)
  theirs.runEpochs(theirs.totalEpochs)
  const pOurs = data.test.X.map((x) => ours.predict(x))
  const pTheirs = data.test.X.map((x) => theirs.predict(x))
  let maxDiff = 0
  for (let i = 0; i < pOurs.length; i++) maxDiff = Math.max(maxDiff, Math.abs(pOurs[i] - pTheirs[i]))
  const mae = familyMetrics(pOurs, data.test.y).mae
  const want = fam.test.network
  // The single network the browser trains by default (seed 11), for the README and the page check.
  const single = createFamilyTrainer(data, { ...fam.options, seed: fam.options.seed })
  single.runEpochs(single.totalEpochs)
  const singleMae = familyMetrics(data.test.X.map((x) => single.predict(x)), data.test.y).mae
  const okPred = maxDiff <= 1e-9
  const okMae = Math.abs(mae - want) <= AGENT[id].tol
  if (!okPred || !okMae) failed++
  console.log(`${okPred && okMae ? 'ok  ' : 'FAIL'} ${id.padEnd(10)} max prediction difference ${maxDiff.toExponential(2)} | ensemble test MAE ${mae.toFixed(6)} (result.json ${want}, tolerance ${AGENT[id].tol}) | single seed-${fam.options.seed} network ${singleMae.toFixed(6)} | nnFamily ${((t1 - t0) / 1000).toFixed(1)} s`)
}
console.log(failed ? `\n${failed} family(ies) failed parity` : '\nall families match')
process.exit(failed ? 1 : 0)
