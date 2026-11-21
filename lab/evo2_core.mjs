// Shared by evolve2.mjs, its workers and baselines2.mjs: the data tiers, the pre-training rows, the cache of
// pre-trained weights, and training one individual (2 seeds) from a genome.
//  - Pre-training rows: 'physics' = 1,800 synthetic Powers rows (physics/synthetic.mjs, seed 2026); 'boxcrete' =
//    the 395 BOxCrete rows; 'both' = the two together. Fixed per kind, so the shared scaling is fixed per kind.
//  - Pre-training: 40 epochs, lr 0.004, decay 1e-4, batch 32, squared loss, init and shuffles from the pre-training
//    seed 1000 + seed index. Cached in lab/cache/pretrain/ per (layers, features, kind, pre-training seed).
//  - Seed r of an individual (r = 0, 1) fine-tunes with seed genome.seed + 7919 r.
//  - Inheritance (Lamarckian): seed r starts from the parent's seed-r weights and trains 30% of the genome's
//    epochs. Allowed only when layers, feature set and pre-training kind are all unchanged (the kind fixes the scaling).
import './guard.mjs'
import path from 'node:path'
import { existsSync, readFileSync } from 'node:fs'
import { gzipSync, gunzipSync } from 'node:zlib'
import { LAB } from './guard.mjs'
import { readJson, writeSame, sha256 } from './io.mjs'
import { createNet, layersFor, fromWeights } from './net.mjs'
import { toOptions2 } from './genome2.mjs'
import { synthesize } from './physics/synthetic.mjs'
import { boxcreteRows } from './physics/boxcrete.mjs'
import { makeScorer } from './reward2.mjs'

export const TRAIN = readJson(path.join(LAB, 'data', 'train.json'))
export const REWARD = readJson(path.join(LAB, 'data', 'reward.json'))
export const MONITOR = readJson(path.join(LAB, 'data', 'monitor.json'))
export const scorer = makeScorer(TRAIN, REWARD, MONITOR)
export const PRETRAIN_HP = { epochs: 40, lr: 0.004, decay: 1e-4, batch: 32 }
export const SEEDS = 2
export const CACHE = path.join(LAB, 'cache', 'pretrain')

const rowsMemo = new Map()
export function pretrainRows(kind) {
  if (kind === 'none') return null
  if (!rowsMemo.has(kind)) {
    const phys = () => synthesize(TRAIN, { kind: 'powers', mixes: 300, agesPerMix: 6, seed: 2026 })
    const box = () => { const b = boxcreteRows(); return { X: b.X, y: b.y } }
    let rows
    if (kind === 'physics') rows = phys()
    else if (kind === 'boxcrete') rows = box()
    else if (kind === 'both') { const a = phys(), b = box(); rows = { X: [...a.X, ...b.X], y: [...a.y, ...b.y] } }
    else throw new Error(`unknown pretrain kind ${kind}`)
    rowsMemo.set(kind, rows)
  }
  return rowsMemo.get(kind)
}

const optsFor = (genome, extra = {}) => ({ ...toOptions2(genome), ...extra })
export const layersOf = (genome) => layersFor(toOptions2(genome))
export const pretrainKey = (genome, r) => sha256(JSON.stringify([layersOf(genome), genome.features, genome.pretrain, 1000 + r, PRETRAIN_HP])).slice(0, 24)
export const cachedPretrain = (genome, r) => existsSync(path.join(CACHE, `${pretrainKey(genome, r)}.json.gz`))

// Builds (or confirms) the cached pre-trained weights for seed index r of this genome's shape.
export function buildPretrain(genome, r) {
  const file = path.join(CACHE, `${pretrainKey(genome, r)}.json.gz`)
  if (existsSync(file)) return JSON.parse(gunzipSync(readFileSync(file)).toString())
  const rows = pretrainRows(genome.pretrain)
  const net = createNet(TRAIN, optsFor(genome, { seed: 1000 + r }), { scaleWith: rows })
  net.pretrain(rows, { ...PRETRAIN_HP, seed: 1000 + r })
  const w = net.getWeights()
  writeSame(file, gzipSync(JSON.stringify(w), { level: 6 }))
  return w
}

// Trains seed r. parentWeights (one replica) => inherit and train 30% of the epochs.
export function trainReplica(genome, r, parentWeights = null) {
  const rows = pretrainRows(genome.pretrain)
  const epochs = parentWeights ? Math.max(1, Math.round(0.3 * genome.epochs)) : genome.epochs
  const net = createNet(TRAIN, optsFor(genome, { seed: genome.seed + 7919 * r, epochs }), rows ? { scaleWith: rows } : {})
  if (parentWeights) net.setWeights(parentWeights)
  else if (rows) net.setWeights(buildPretrain(genome, r))
  net.train()
  return net
}

// One individual: SEEDS replicas, scored. Returns the weights object (all replicas) and the metrics.
export function trainIndividual(genome, parentReplicas = null) {
  const t0 = process.hrtime.bigint()
  const nets = Array.from({ length: SEEDS }, (_, r) => trainReplica(genome, r, parentReplicas ? parentReplicas[r] : null))
  const trainSeconds = Number(process.hrtime.bigint() - t0) / 1e9
  const metrics = scorer.score(nets.map((n) => n.predictMany))
  const weights = { format: 'lab-net-replicas-1', replicas: nets.map((n) => n.getWeights()) }
  return { weights, metrics, trainSeconds, layers: nets[0].layers, weightCount: nets[0].weightCount }
}

// Prediction of a saved multi-replica model: the mean of its replicas.
export function replicasPredict(weights) {
  const nets = weights.replicas.map((w) => fromWeights(w))
  return (rows) => { const P = nets.map((n) => n.predictMany(rows)); return rows.map((_, i) => P.reduce((s, p) => s + p[i], 0) / P.length) }
}
