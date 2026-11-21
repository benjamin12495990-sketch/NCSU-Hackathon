// Phase-2 baselines and champions, all trained on TRAIN only, scored on REWARD and MONITOR, and saved once with
// their SHA-256 for pre-registration (lab/results/models2/, lab/results/baselines2.json). Run after evolve2.mjs
// and evolve2.mjs --random. Networks over 10 seeds use seed index r = 0..9 as in evo2_core.trainReplica
// (fine-tune seed genome.seed + 7919 r, pre-training seed 1000 + r); "plain" with r = 0 is the page's network.
import './guard.mjs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { gzipSync } from 'node:zlib'
import { LAB } from './guard.mjs'
import { writeSame, sha256 } from './io.mjs'
import { TRAIN, REWARD, MONITOR, scorer, trainReplica } from './evo2_core.mjs'
import { calibrate, meta as physicsMeta, fibS } from './physics/synthetic.mjs'
import { boxcreteRows } from './physics/boxcrete.mjs'
import { openArchive } from './archive.mjs'
import { predictions, scoreAll } from './scoring2.mjs'

const DIR = path.join(LAB, 'results', 'models2')
const OUT = path.resolve(process.argv.includes('--out') ? process.argv[process.argv.indexOf('--out') + 1] : path.join(LAB, 'results', 'baselines2_v2.json'))
const rel = (p) => path.relative(LAB, p)
const SEEDS10 = 10
const evo = openArchive(path.join(LAB, 'archive')), rnd = openArchive(path.join(LAB, 'archive_random'))
const stopOf = (A) => A.records().find((r) => r.type === 'stop')
const evoStop = stopOf(evo), rndStop = stopOf(rnd)
if (!evoStop || !rndStop) { console.error('baselines2: run evolve2.mjs and evolve2.mjs --random to completion first'); process.exit(1) }
const byId = (A) => new Map(A.individuals().map((r) => [r.id, r]))
const evoIds = byId(evo), rndIds = byId(rnd)
const champ = evoIds.get(evoStop.champion), rchamp = rndIds.get(rndStop.champion)

const models = []
const PLAIN = { hidden: [64, 32], lr: 0.004, decay: 1e-3, loss: 'squared', epochs: 300, features: 'base', pretrain: 'none', seed: 11 }
const PANEL = { ...PLAIN, decay: 0.003, loss: 'huber', features: 'physics' } // = PANEL_DESIGN of src/lib/nn.js
function seedsModel(role, genome, note) {
  const t0 = Date.now()
  const nets = Array.from({ length: SEEDS10 }, (_, r) => trainReplica(genome, r))
  const text = JSON.stringify({ format: 'lab-net-replicas-1', genome, replicas: nets.map((n) => n.getWeights()) })
  const file = path.join(DIR, `${role}.json.gz`)
  writeSame(file, gzipSync(text, { level: 9 }))
  models.push({ role, kind: 'seeds', file: rel(file), sha256: sha256(text), genome, note })
  console.log(`${role}: 10 seeds trained (${((Date.now() - t0) / 1000).toFixed(1)} s)`)
}
const archived = (role, A, r, note) => models.push({ role, kind: 'archived', id: r.id, file: rel(path.join(A.dir, 'weights', `${r.id}.json.gz`)), sha256: r.weightsSha256, genome: r.genome, note })

// 1. physics equation alone (each generator calibrated on TRAIN)
const cal = calibrate(TRAIN)
for (const gen of ['powers', 'fib']) {
  const text = JSON.stringify({ generator: gen, scale: cal[gen].scale, calibratedOn: 'TRAIN' })
  const file = path.join(DIR, `physics_${gen}.json`)
  writeSame(file, text)
  models.push({ role: `physics_${gen}`, kind: 'physics', file: rel(file), sha256: sha256(text), note: `${gen} generator, scale ${cal[gen].scale.toFixed(4)} fitted on TRAIN` })
}
// 1b. per-mix age curve: s28 of each TRAIN mix by least squares on its own TRAIN rows, f(t) = exp(s (1 - sqrt(28/t)))
// with s = 0.20 + 0.35 p_SCM; fallback for a mix with no TRAIN rows: the TRAIN-calibrated fib generator (physics_fib).
{
  const f = (x) => Math.exp(fibS(x) * (1 - Math.sqrt(28 / x[7])))
  const by = new Map()
  TRAIN.X.forEach((x, i) => { const k = x.slice(0, 7).join(','); (by.get(k) || by.set(k, []).get(k)).push(i) })
  const perMixS28 = Object.fromEntries([...by].map(([k, idx]) => [k, idx.reduce((s, i) => s + TRAIN.y[i] * f(TRAIN.X[i]), 0) / idx.reduce((s, i) => s + f(TRAIN.X[i]) ** 2, 0)]))
  const text = JSON.stringify({ curve: 's28_mix * exp((0.20 + 0.35 p_SCM)(1 - sqrt(28/t)))', perMixS28, fallback: 'physics_fib generator (scale fitted on TRAIN)', fallbackScale: cal.fib.scale })
  const file = path.join(DIR, 'curve_permix.json')
  writeSame(file, text)
  models.push({ role: 'curve_permix', kind: 'curve', file: rel(file), sha256: sha256(text), note: `per-mix fib curve, s28 fitted on each mix's TRAIN rows (${by.size} mixes); other mixes use the TRAIN-calibrated fib generator` })
}
// 2-3. trees, forest, log-age ridge
for (const name of ['gbt', 'rf', 'ridge']) {
  const r = spawnSync('python3', ['-I', path.join(LAB, 'sk_models.py'), 'fit', name], { encoding: 'utf8' })
  if (r.status !== 0) throw new Error(r.stderr)
  const info = JSON.parse(r.stdout)
  models.push({ role: name, kind: 'sklearn', name, file: info.file, sha256: info.sha256, note: `${info.model}, scikit-learn ${info.sklearn}` })
}
// 4-8. networks over 10 seeds
seedsModel('plain', PLAIN, 'page default network (options {})')
seedsModel('panel_design', PANEL, "PANEL_DESIGN, the site's physics switch")
seedsModel('plain_physics_pretrain', { ...PLAIN, pretrain: 'physics' }, 'page default network with physics pre-training only')
seedsModel('champion_retrained', champ.genome, `evolution champion ${champ.id} retrained from scratch`)
seedsModel('champion_no_pretrain', { ...champ.genome, pretrain: 'none' }, `evolution champion ${champ.id} with pretrain 'none'`)
seedsModel('random_champion_retrained', rchamp.genome, `random-search champion ${rchamp.id} retrained from scratch`)
// 9. champions as archived, random-search champion, top-10 ensemble
archived('champion', evo, champ, `evolution champion by the Ladder rule (${evoStop.reason})`)
archived('random_champion', rnd, rchamp, 'random-search champion by the same Ladder rule')
const top10 = evo.individuals().filter((r) => r.eligible).sort((a, b) => a.fitness - b.fitness || (a.id < b.id ? -1 : 1)).slice(0, 10)
models.push({ role: 'top10_ensemble', kind: 'ensemble', members: top10.map((r) => ({ id: r.id, file: rel(path.join(evo.dir, 'weights', `${r.id}.json.gz`)), sha256: r.weightsSha256 })), note: 'mean prediction of the 10 eligible evolution individuals with the best fitness' })

// Score on REWARD and MONITOR (mix-grouped bootstrap CIs) plus the physics violations on the query grid.
const out = { note: 'Trained on TRAIN only. REWARD was used by evolution for selection (optimistic for evolved models); MONITOR was not.', champion: { id: champ.id, genome: champ.genome, stop: evoStop }, randomChampion: { id: rchamp.id, genome: rchamp.genome }, physics: physicsMeta(TRAIN), boxcrete: boxcreteRows().meta, models: [] }
const rewardScores = scoreAll(models, REWARD, {}, { championRole: 'champion' }), monitorScores = scoreAll(models, MONITOR, {}, { championRole: 'champion' })
for (let i = 0; i < models.length; i++) {
  const m = models[i]
  const v = scorer.score([(rows) => { const P = predictions(m, rows); return P[0].map((_, k) => P.reduce((s, p) => s + p[k], 0) / P.length) }])
  out.models.push({ ...m, reward: rewardScores[i], monitor: monitorScores[i], violations: { vAge: v.vAge, vWb: v.vWb, rate: v.violationRate, eligible: v.eligible } })
  const f = (s) => (s.medianOverSeeds ? `${s.medianOverSeeds.rmse.toFixed(3)} (median of 10; mean-of-seeds ${s.overall.rmse.toFixed(3)})` : s.overall.rmse.toFixed(3))
  console.log(`${m.role.padEnd(24)} REWARD RMSE ${f(rewardScores[i])}  MONITOR RMSE ${f(monitorScores[i])}  V ${(v.vAge + v.vWb).toFixed(3)} MPa, rate ${(100 * v.violationRate).toFixed(1)}%`)
}
writeSame(OUT, JSON.stringify(out, null, 1) + '\n')
console.log(`wrote ${path.relative(LAB, OUT)}`)
