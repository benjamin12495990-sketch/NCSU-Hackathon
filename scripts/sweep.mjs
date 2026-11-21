// npm run sweep: trains the page's default network from scratch (add --physics
// for the design behind the "textbook strength laws" switch), then checks on a
// spread of tested mixes whether its predictions follow two known trends:
// more water for the same binder gives weaker concrete, and older concrete
// is stronger. Prints the tree model beside it. Exits with an error if the
// network gets the overall direction wrong on most mixes.
import { readFileSync } from 'node:fs'
import { createTrainer, PANEL_DESIGN } from '../src/lib/nn.js'
import { featurize } from '../src/lib/featurize.js'
import { predict } from '../src/lib/predict.js'
import { mixName } from '../src/lib/name.js'
import { AGES, waterSteps, trend } from '../src/lib/sweep.js'

const read = (name) => JSON.parse(readFileSync(new URL('../public/model/' + name, import.meta.url)))
const nnData = read('nn_data.json'), spec = read('features.json'), space = read('space.json'), trees = read('model.json'), known = read('known.json')
const physics = process.argv.includes('--physics')
const net = createTrainer(nnData, physics ? PANEL_DESIGN : {})
while (!net.done) net.runEpochs(50)
console.log(`Network trained (${physics ? 'with the textbook laws' : 'default design, as the page trains it'}): ${net.history.at(-1).mae.toFixed(2)} MPa average miss on the held-out mixes.\n`)

const models = { network: (x) => net.predict(x), trees: (x) => predict(trees, x) }
const total = { network: { water: [0, 0], age: [0, 0], waterRight: 0, ageRight: 0 }, trees: { water: [0, 0], age: [0, 0], waterRight: 0, ageRight: 0 } }
let mixes = 0
for (let i = 0; i < known.length; i += Math.floor(known.length / 24)) {
  const mix = known[i].m, steps = waterSteps(mix, space)
  if (!steps) continue
  mixes++
  const line = [mixName(mix).padEnd(34)]
  for (const [name, f] of Object.entries(models)) {
    const w = trend(steps.map((s) => f(featurize(s.mix, 28, spec))))
    const a = trend(AGES.map((age) => f(featurize(mix, age, spec))))
    const t = total[name]
    t.water[0] += w.down; t.water[1] += w.steps; t.age[0] += a.up; t.age[1] += a.steps
    if (w.last < w.first) t.waterRight++
    if (a.last > a.first) t.ageRight++
    line.push(`${name}: water ${w.first.toFixed(0)}→${w.last.toFixed(0)} MPa (${w.down}/${w.steps} steps down), age ${a.first.toFixed(0)}→${a.last.toFixed(0)} MPa (${a.up}/${a.steps} up)`)
  }
  console.log(line.join(' | '))
}
console.log()
for (const [name, t] of Object.entries(total)) {
  console.log(`${name}: more water lowered strength overall on ${t.waterRight} of ${mixes} mixes (${Math.round((100 * t.water[0]) / t.water[1])}% of single steps); more age raised it on ${t.ageRight} of ${mixes} (${Math.round((100 * t.age[0]) / t.age[1])}% of single steps).`)
}
const ok = total.network.waterRight > mixes / 2 && total.network.ageRight > mixes / 2
console.log(ok ? '\nPASS: the network follows both trends on most mixes.' : '\nFAIL: the network gets a trend wrong on most mixes.')
process.exit(ok ? 0 : 1)
