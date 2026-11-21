// Evolution without deletion. Generation 0 is N random genomes; every later generation makes K children.
// Each child's parent is drawn from the WHOLE archive (every individual ever trained) with probability
// proportional to exp(reward / T): better networks get more children, nothing is removed. A child whose
// layer shapes match its parent's starts from the parent's trained weights (Lamarckian), then trains on
// TRAIN; otherwise it starts fresh from its own seed. Deterministic for a run seed; rerun to resume.
//   node lab/evolve.mjs [--archive lab/archive] [--seed 1] [--N 24] [--K 12] [--generations 10] [--T 0.5] [--lambda 2]
import './guard.mjs'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { LAB } from './guard.mjs'
import { sha256 } from './io.mjs'
import { readFileSync } from 'node:fs'
import { createNet, seeded, sameShape } from './net.mjs'
import { randomGenome, mutate, genomeId, toOptions } from './genome.mjs'
import { reward, LAMBDA } from './reward.mjs'
import { openArchive } from './archive.mjs'

const arg = (name, dflt) => { const i = process.argv.indexOf('--' + name); return i > 0 ? process.argv[i + 1] : dflt }
const cfg = {
  archive: path.resolve(arg('archive', path.join(LAB, 'archive'))),
  runSeed: Number(arg('seed', 1)), N: Number(arg('N', 24)), K: Number(arg('K', 12)),
  generations: Number(arg('generations', 10)), // generations after generation 0
  T: Number(arg('T', 0.5)), // selection temperature in MPa: a parent 0.5 MPa better is e = 2.7 times as likely
  lambda: Number(arg('lambda', LAMBDA)),
}
const trainText = readFileSync(path.join(LAB, 'data', 'train.json')), selectText = readFileSync(path.join(LAB, 'data', 'select.json'))
const TRAIN = JSON.parse(trainText), SELECT = JSON.parse(selectText)
const genSeed = (g) => parseInt(createHash('sha256').update(`${cfg.runSeed}:${g}`).digest('hex').slice(0, 8), 16)

const A = openArchive(cfg.archive)
const header = { type: 'run', runSeed: cfg.runSeed, N: cfg.N, K: cfg.K, T: cfg.T, lambda: cfg.lambda, trainSha256: sha256(trainText), selectSha256: sha256(selectText), selection: 'p(parent) proportional to exp(reward / T) over the whole archive as it stood at the start of the generation' }
const old = A.header()
if (!old) A.append({ ...header, created: new Date().toISOString() })
else for (const k of Object.keys(header)) if (JSON.stringify(old[k]) !== JSON.stringify(header[k])) { console.error(`evolve: archive ${cfg.archive} was made with ${k}=${JSON.stringify(old[k])}, not ${JSON.stringify(header[k])}. Use another --archive.`); process.exit(1) }

// The plan for generation g depends only on the run seed and on generations < g, so a resumed run makes the same children.
function plan(g, before) {
  const rng = seeded(genSeed(g))
  if (g === 0) return Array.from({ length: cfg.N }, () => ({ genome: randomGenome(rng), parent: null }))
  const best = Math.max(...before.map((r) => r.reward))
  const w = before.map((r) => Math.exp((r.reward - best) / cfg.T)), total = w.reduce((s, v) => s + v, 0)
  return Array.from({ length: cfg.K }, () => {
    let u = rng() * total, k = 0
    while (k < w.length - 1 && (u -= w[k]) > 0) k++
    return { genome: mutate(before[k].genome, rng), parent: before[k] }
  })
}

for (let g = 0; g <= cfg.generations; g++) {
  const all = A.individuals()
  const before = all.filter((r) => r.generation < g)
  const done = new Map(all.filter((r) => r.generation === g).map((r) => [r.index, r]))
  const children = plan(g, before)
  for (let i = 0; i < children.length; i++) {
    const { genome, parent } = children[i]
    const id = `g${String(g).padStart(3, '0')}-${String(i).padStart(3, '0')}-${genomeId(genome)}`
    if (done.has(i)) { if (done.get(i).id !== id) throw new Error(`resume mismatch at generation ${g} child ${i}: ${done.get(i).id} vs ${id}`); continue }
    const t0 = process.hrtime.bigint()
    const net = createNet(TRAIN, toOptions(genome))
    const inherit = parent && sameShape(parent.layers, net.layers)
    if (inherit) net.setWeights(A.loadWeights(parent.id, parent.weightsSha256))
    net.pretrain(genome.pretrain ? [] : null)
    net.train()
    const trainSeconds = Number(process.hrtime.bigint() - t0) / 1e9
    const score = reward(net.predictMany, TRAIN, SELECT, cfg.lambda)
    const weightsSha256 = A.saveWeights(id, net.getWeights())
    A.append({ type: 'individual', id, parent: parent ? parent.id : null, generation: g, index: i, genome, layers: net.layers, inherited: !!inherit, weightCount: net.weightCount, ...score, trainSeconds: Number(trainSeconds.toFixed(3)), weightsSha256, created: new Date().toISOString() })
    console.log(`gen ${g} #${i} ${id} ${genome.features}/${genome.loss} [${genome.hidden}] ${genome.epochs}ep ${inherit ? 'warm' : 'fresh'}: reward ${score.reward.toFixed(3)} (SELECT MAE ${score.selectMae.toFixed(3)} MPa, violations ${score.violations.toFixed(3)} = falling ${score.fallingShare.toFixed(3)} + negative ${score.negativeShare.toFixed(3)}) ${trainSeconds.toFixed(1)} s`)
  }
  const now = A.individuals()
  const best = now.reduce((a, b) => (b.reward > a.reward ? b : a))
  console.log(`after generation ${g}: archive ${now.length} individuals, best reward ${best.reward.toFixed(3)} (${best.id}, SELECT MAE ${best.selectMae.toFixed(3)} MPa, violations ${best.violations.toFixed(3)})`)
}
