// Phase-2 evolution (the lab's methods research, section 7) and its random-search control.
//   node lab/evolve2.mjs [--archive lab/archive] [--seed 1] [--children 20] [--generations 25] [--workers 3]
//   node lab/evolve2.mjs --random [--archive lab/archive_random] [--total <individuals>] [--seed 1]
// Evolution: generation 0 = 20 genomes from the prior; then each generation makes 20 children, each with ONE mutation.
//  - 16 children: parent drawn from the whole archive (eligible individuals only: violation rate <= 5%) with
//    p_i ~ exp(-(rank_i - 1) / tau) / (1 + c_i)^0.5, rank by fitness (1 = best), tau = max(3, 0.1 N), N = archive
//    size, c_i = children i already has (updated as the generation is planned).
//  - 4 children (20%): parent drawn uniformly from the elites of the MAP-Elites cells
//    (size bucket x pre-training kind x feature set), elite = best fitness in the cell.
//  - Partial Lamarckian inheritance: when layers, feature set and pre-training kind are unchanged, with p 0.5 the
//    child starts from the parent's weights and trains 30% of its epochs; otherwise it trains from scratch
//    (from the cached pre-trained weights when it pre-trains).
//  - Champion (Ladder rule, Blum & Hardt 2015): after each generation, its eligible individuals are tried in order
//    of rmse + LAMBDA*V; one becomes champion only if that beats the champion's by more than the paired bootstrap
//    SE of their RMSE difference.
//  - Stop: (1) 25 generations; (2) Ladder: no new champion for 5 generations; (3) monitor gap: over the last 5
//    generations, the best-so-far individual's MONITOR RMSE has not improved while its REWARD RMSE has.
//    MONITOR is used for nothing else. The stopping reason is written to the lineage.
// Random search: the same number of individuals as the evolution archive, all from the prior, in batches of 20,
// scored by the same reward, champion by the same Ladder rule; no other stopping rule.
// Determinism: each generation's plan depends only on the run seed and earlier generations; jobs are pure
// functions run by 3 worker processes; results are appended in planned order. Rerun the command to resume.
import './guard.mjs'
import path from 'node:path'
import { fork } from 'node:child_process'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { LAB } from './guard.mjs'
import { sha256 } from './io.mjs'
import { seeded, sameShape } from './net.mjs'
import { randomGenome2, mutate2, genomeId2, sizeBucket } from './genome2.mjs'
import { scorer, layersOf, cachedPretrain, SEEDS, PRETRAIN_HP } from './evo2_core.mjs'
import { LAMBDA, MAX_VIOLATION_RATE } from './reward2.mjs'
import { openArchive } from './archive.mjs'

const arg = (name, dflt) => { const i = process.argv.indexOf('--' + name); return i > 0 ? process.argv[i + 1] : dflt }
const RANDOM = process.argv.includes('--random')
const cfg = {
  mode: RANDOM ? 'random' : 'evolution',
  archive: path.resolve(arg('archive', path.join(LAB, RANDOM ? 'archive_random' : 'archive'))),
  runSeed: Number(arg('seed', 1)), children: Number(arg('children', 20)), generations: Number(arg('generations', 25)),
  workers: Number(arg('workers', 3)), patience: 5,
}
if (RANDOM) {
  const evo = openArchive(path.join(LAB, 'archive'))
  cfg.total = Number(arg('total', evo.individuals().length))
  if (!cfg.total) { console.error('evolve2 --random: run the evolution first, or pass --total'); process.exit(1) }
}
const hashFile = (p) => sha256(readFileSync(p))
const A = openArchive(cfg.archive)
const header = {
  type: 'run', version: 2, mode: cfg.mode, runSeed: cfg.runSeed, children: cfg.children, generations: cfg.generations, patience: cfg.patience,
  seedsPerIndividual: SEEDS, lambda: LAMBDA, maxViolationRate: MAX_VIOLATION_RATE, pretrainHp: PRETRAIN_HP, ...(RANDOM ? { total: cfg.total } : {}),
  data: Object.fromEntries(['train', 'reward', 'monitor'].map((n) => [n, hashFile(path.join(LAB, 'data', `${n}.json`))])),
  boxcrete: hashFile(path.join(LAB, 'external', 'boxcrete_data.csv')),
}
const old = A.header()
if (!old) A.append({ ...header, created: new Date().toISOString() })
else for (const k of Object.keys(header)) if (JSON.stringify(old[k]) !== JSON.stringify(header[k])) { console.error(`evolve2: ${cfg.archive} was made with ${k}=${JSON.stringify(old[k])}, not ${JSON.stringify(header[k])}`); process.exit(1) }
if (A.records().some((r) => r.type === 'stop')) { console.log(`evolve2: ${cfg.archive} already stopped: ${A.records().find((r) => r.type === 'stop').reason}`); process.exit(0) }

const genRng = (g) => seeded(parseInt(createHash('sha256').update(`${cfg.mode}:${cfg.runSeed}:${g}`).digest('hex').slice(0, 8), 16))
const byFitness = (a, b) => a.fitness - b.fitness || (a.id < b.id ? -1 : 1)
const loss = (r) => r.rewardRmse + LAMBDA * (r.vAge + r.vWb)

function plan(g, before) {
  const rng = genRng(g)
  if (g === 0 || RANDOM) {
    const n = RANDOM ? Math.min(cfg.children, cfg.total - before.length) : cfg.children
    return Array.from({ length: n }, () => ({ genome: randomGenome2(rng), parent: null, selection: RANDOM ? 'random' : 'init', mutation: null, inherit: false }))
  }
  const eligible = before.filter((r) => r.eligible).sort(byFitness)
  const pool = eligible.length ? eligible : [...before].sort(byFitness) // fallback only if nothing is eligible
  const N = before.length, tau = Math.max(3, 0.1 * N)
  const kids = new Map(before.map((r) => [r.id, 0]))
  for (const r of before) if (r.parent && kids.has(r.parent)) kids.set(r.parent, kids.get(r.parent) + 1)
  const cells = new Map()
  for (const r of pool) { const c = `${sizeBucket(r.weightCount)}|${r.genome.pretrain}|${r.genome.features}`; if (!cells.has(c)) cells.set(c, r) }
  const elites = [...cells.keys()].sort().map((c) => cells.get(c))
  const out = []
  for (let k = 0; k < cfg.children; k++) {
    let parent, selection
    if (k >= Math.round(0.8 * cfg.children)) { parent = elites[Math.floor(rng() * elites.length)]; selection = 'elite' }
    else {
      const w = pool.map((r, i) => Math.exp(-i / tau) / Math.sqrt(1 + kids.get(r.id)))
      let u = rng() * w.reduce((s, v) => s + v, 0), i = 0
      while (i < w.length - 1 && (u -= w[i]) > 0) i++
      parent = pool[i]; selection = 'rank'
    }
    kids.set(parent.id, kids.get(parent.id) + 1)
    const { genome, mutation } = mutate2(parent.genome, rng)
    const coin = rng()
    const inherit = sameShape(layersOf(genome), parent.layers) && genome.features === parent.genome.features && genome.pretrain === parent.genome.pretrain && coin < 0.5
    out.push({ genome, parent, selection, mutation, inherit })
  }
  return out
}

// --- worker pool ---
const workers = Array.from({ length: cfg.workers }, () => fork(path.join(LAB, 'evo2_worker.mjs'), [], { stdio: ['ignore', 'inherit', 'inherit', 'ipc'] }))
const idle = [...workers], waiting = [], pending = new Map()
let nextId = 0
const pump = () => {
  while (idle.length && waiting.length) {
    const w = idle.pop(), job = waiting.shift()
    pending.set(job.id, { ...job, w }); w.send(job.msg)
  }
}
for (const w of workers) w.on('message', (m) => {
  const job = pending.get(m.id); pending.delete(m.id); idle.push(w); pump()
  if (m.ok) job.resolve(m.res); else job.reject(new Error(m.error))
})
const run = (msg) => new Promise((resolve, reject) => { const id = nextId++; waiting.push({ id, msg: { ...msg, id }, resolve, reject }); pump() })

function generationSummary(g, all, state) {
  const mine = all.filter((r) => r.generation === g)
  const eligibleAll = all.filter((r) => r.eligible).sort(byFitness)
  const best = eligibleAll[0] || [...all].sort(byFitness)[0]
  let newChampion = false
  for (const c of mine.filter((r) => r.eligible).sort((a, b) => loss(a) - loss(b) || (a.id < b.id ? -1 : 1))) {
    if (!state.champion) { state.champion = c; newChampion = true; continue }
    const se = scorer.pairedSe(c.rowSE, state.champion.rowSE)
    if (loss(c) < loss(state.champion) - se) { state.champion = c; newChampion = true }
  }
  const ch = state.champion
  return {
    type: 'generation', generation: g, individuals: all.length, eligibleShare: Number((mine.filter((r) => r.eligible).length / mine.length).toFixed(3)),
    genBestFitness: Math.min(...mine.map((r) => r.fitness)), genMedianRewardRmse: median(mine.map((r) => r.rewardRmse)),
    bestSoFar: { id: best.id, fitness: best.fitness, rewardRmse: best.rewardRmse, monitorRmse: best.monitorRmse },
    champion: ch ? { id: ch.id, loss: loss(ch), rewardRmse: ch.rewardRmse, monitorRmse: ch.monitorRmse } : null, newChampion,
  }
}
const median = (v) => { const s = [...v].sort((a, b) => a - b); return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2 }

async function main() {
  const state = { champion: null }
  const summaries = []
  for (let g = 0; ; g++) {
    if (!RANDOM && g > cfg.generations) break
    let all = A.individuals()
    if (RANDOM && all.filter((r) => r.generation < g).length >= cfg.total) break
    const before = all.filter((r) => r.generation < g)
    const done = new Map(all.filter((r) => r.generation === g).map((r) => [r.index, r]))
    const kids = plan(g, before)
    if (!kids.length) break
    const t0 = Date.now()
    // 1. pre-train what the cache lacks (one job per missing key), 2. train every child.
    const need = new Map()
    kids.forEach((k, i) => { if (!done.has(i) && !k.inherit && k.genome.pretrain !== 'none') for (let r = 0; r < SEEDS; r++) if (!cachedPretrain(k.genome, r)) need.set(JSON.stringify([layersOf(k.genome), k.genome.features, k.genome.pretrain, r]), { genome: k.genome, r }) })
    await Promise.all([...need.values()].map((n) => run({ type: 'pretrain', genome: n.genome, r: n.r })))
    const results = new Array(kids.length)
    let flushed = 0
    const flush = () => {
      while (flushed < kids.length && (done.has(flushed) || results[flushed])) {
        const i = flushed++, k = kids[i]
        const id = `g${String(g).padStart(3, '0')}-${String(i).padStart(3, '0')}-${genomeId2(k.genome)}`
        if (done.has(i)) { if (done.get(i).id !== id) throw new Error(`resume mismatch at ${g}/${i}`); continue }
        const { weights, metrics, trainSeconds, layers, weightCount } = results[i]
        const weightsSha256 = A.saveWeights(id, weights)
        A.append({ type: 'individual', id, parent: k.parent ? k.parent.id : null, generation: g, index: i, selection: k.selection, mutation: k.mutation, genome: k.genome, layers, inherited: k.inherit, weightCount, ...metrics, trainSeconds: Number(trainSeconds.toFixed(3)), weightsSha256, created: new Date().toISOString() })
      }
    }
    await Promise.all(kids.map((k, i) => {
      if (done.has(i)) return null
      const parent = k.inherit ? { archive: cfg.archive, id: k.parent.id, sha: k.parent.weightsSha256 } : null
      return run({ type: 'train', genome: k.genome, parent }).then((res) => { results[i] = res; flush() })
    }))
    flush()
    all = A.individuals()
    const s = generationSummary(g, all, state)
    summaries.push(s)
    if (!A.records().some((r) => r.type === 'generation' && r.generation === g)) A.append({ ...s, seconds: (Date.now() - t0) / 1000 })
    console.log(`gen ${g}: ${all.length} individuals, eligible ${(100 * s.eligibleShare).toFixed(0)}%, gen best fitness ${s.genBestFitness.toFixed(3)}, best-so-far ${s.bestSoFar.id} REWARD ${s.bestSoFar.rewardRmse.toFixed(3)} MONITOR ${s.bestSoFar.monitorRmse.toFixed(3)}, champion ${s.champion ? s.champion.id + ' ' + s.champion.loss.toFixed(3) : '-'}${s.newChampion ? ' (new)' : ''} (${((Date.now() - t0) / 1000).toFixed(1)} s)`)
    if (RANDOM) continue
    let reason = null
    if (g >= cfg.generations) reason = `generation cap (${cfg.generations})`
    else if (g >= cfg.patience && summaries.slice(-cfg.patience).every((x) => !x.newChampion)) reason = `ladder: no new champion for ${cfg.patience} generations`
    else if (g >= cfg.patience) {
      const minMon = (upto) => Math.min(...summaries.slice(0, upto + 1).map((x) => x.bestSoFar.monitorRmse))
      const nowS = summaries[g], thenS = summaries[g - cfg.patience]
      if (minMon(g) >= minMon(g - cfg.patience) && nowS.bestSoFar.rewardRmse < thenS.bestSoFar.rewardRmse) reason = `monitor gap: best-so-far MONITOR RMSE not improved for ${cfg.patience} generations while REWARD RMSE improved`
    }
    if (reason) { A.append({ type: 'stop', generation: g, reason, champion: state.champion && state.champion.id, created: new Date().toISOString() }); console.log(`stop after generation ${g}: ${reason}`); break }
  }
  if (RANDOM && !A.records().some((r) => r.type === 'stop')) A.append({ type: 'stop', reason: `random search complete (${A.individuals().length} individuals)`, champion: state.champion && state.champion.id, created: new Date().toISOString() })
  for (const w of workers) w.kill()
}
main().catch((e) => { console.error(e); for (const w of workers) w.kill(); process.exit(1) })
