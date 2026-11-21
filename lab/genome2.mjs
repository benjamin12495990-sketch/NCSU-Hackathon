// Phase-2 genome: { hidden: [h1, h2] (8-64), lr, decay, loss ('squared' | 'huber' | 'mae'), epochs (40-400),
// features ('base' | 'rich' | 'physics'), pretrain ('none' | 'physics' | 'boxcrete' | 'both'), seed }.
// The 64-unit cap keeps every network at a few thousand weights (at most 5,445).
// mutate2() makes exactly ONE mutation per child (Real et al. 2019), PBT-style (Jaderberg et al. 2017):
//   operator family: continuous 0.5, layer size 0.2, discrete 0.3 (design choices, not confirmed)
//   continuous (lr, decay or epochs, one chosen uniformly): with p 0.2 resample from the prior, else multiply
//              by 0.8 or 1.2; clipped to the prior range
//   layer size (one layer chosen uniformly): multiply by 0.8 or 1.25, rounded, change at least 1 unit, clipped
//   discrete (loss, feature set or pre-training kind, one chosen uniformly): switch to a different value
// Not implemented from the research note: the activation toggle (Mlp is copied unchanged and is ReLU only).
// The seed is always redrawn.
import './guard.mjs'
import { createHash } from 'node:crypto'

export const BOUNDS = { hidden: [8, 64], lr: [3e-4, 3e-2], decay: [1e-5, 1e-2], epochs: [40, 400] }
export const LOSSES = ['squared', 'huber', 'mae']
export const FEATURE_SETS = ['base', 'rich', 'physics']
export const PRETRAIN = ['none', 'physics', 'boxcrete', 'both']

const clamp = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v))
const logU = (rng, [lo, hi]) => Math.exp(Math.log(lo) + rng() * (Math.log(hi) - Math.log(lo)))
const pick = (rng, list) => list[Math.floor(rng() * list.length)]
const round = (v) => Number(v.toPrecision(6))
const newSeed = (rng) => 1 + Math.floor(rng() * 2 ** 30)
const prior = {
  lr: (rng) => round(logU(rng, BOUNDS.lr)),
  decay: (rng) => round(logU(rng, BOUNDS.decay)),
  epochs: (rng) => Math.round(BOUNDS.epochs[0] + rng() * (BOUNDS.epochs[1] - BOUNDS.epochs[0])),
  hidden: (rng) => Math.round(logU(rng, BOUNDS.hidden)),
}

export function randomGenome2(rng) {
  return { hidden: [prior.hidden(rng), prior.hidden(rng)], lr: prior.lr(rng), decay: prior.decay(rng), loss: pick(rng, LOSSES), epochs: prior.epochs(rng), features: pick(rng, FEATURE_SETS), pretrain: pick(rng, PRETRAIN), seed: newSeed(rng) }
}

export function mutate2(parent, rng) {
  const g = { ...parent, hidden: [...parent.hidden] }
  const u = rng()
  let what
  if (u < 0.5) {
    const key = pick(rng, ['lr', 'decay', 'epochs'])
    if (rng() < 0.2) { g[key] = prior[key](rng); what = `${key} resampled` }
    else {
      const f = rng() < 0.5 ? 0.8 : 1.2
      g[key] = key === 'epochs' ? Math.round(clamp(g[key] * f, BOUNDS.epochs)) : round(clamp(g[key] * f, BOUNDS[key]))
      what = `${key} x${f}`
    }
  } else if (u < 0.7) {
    const i = rng() < 0.5 ? 0 : 1, f = rng() < 0.5 ? 0.8 : 1.25
    let h = Math.round(g.hidden[i] * f)
    if (h === g.hidden[i]) h += f > 1 ? 1 : -1
    g.hidden[i] = clamp(h, BOUNDS.hidden)
    what = `hidden[${i}] x${f}`
  } else {
    const key = pick(rng, ['loss', 'features', 'pretrain'])
    const list = { loss: LOSSES, features: FEATURE_SETS, pretrain: PRETRAIN }[key]
    g[key] = pick(rng, list.filter((v) => v !== g[key]))
    what = `${key} -> ${g[key]}`
  }
  g.seed = newSeed(rng)
  return { genome: g, mutation: what }
}

const canonical = (g) => JSON.stringify([g.hidden, g.lr, g.decay, g.loss, g.epochs, g.features, g.pretrain, g.seed])
export const genomeId2 = (g) => createHash('sha256').update(canonical(g)).digest('hex').slice(0, 12)

export function toOptions2(g) {
  const o = { hidden: g.hidden, lr: g.lr, decay: g.decay, loss: g.loss, epochs: g.epochs, features: g.features, seed: g.seed }
  if (g.features === 'physics') Object.assign(o, { physicsOn: 'base', physics: ['abrams', 'bolomey', 'feret', 'bolomeyAge'] })
  return o
}
// Parameter-count bucket for the MAP-Elites grid: < 1,000 / 1,000-2,999 / >= 3,000 weights.
export const sizeBucket = (weightCount) => (weightCount < 1000 ? 0 : weightCount < 3000 ? 1 : 2)
