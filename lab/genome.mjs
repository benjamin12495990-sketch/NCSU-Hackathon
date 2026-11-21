// A genome is the recipe for one network: { hidden: [h1, h2], lr, decay, loss, epochs, features, pretrain, seed }.
// randomGenome(rng) draws one; mutate(genome, rng) changes a few fields by the step sizes below; genomeId()
// is a short SHA-256 of the canonical genome. toOptions() turns a genome into net.mjs createNet options.
import './guard.mjs'
import { createHash } from 'node:crypto'

export const BOUNDS = {
  hidden: [8, 128], // units per hidden layer, integer
  lr: [3e-4, 3e-2], // Adam learning rate, log scale
  decay: [1e-5, 1e-2], // L2 weight decay, log scale
  epochs: [40, 400], // passes over TRAIN
}
export const LOSSES = ['squared', 'huber']
export const FEATURE_SETS = ['base', 'rich', 'physics']

const clamp = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v))
const logUniform = (rng, [lo, hi]) => Math.exp(Math.log(lo) + rng() * (Math.log(hi) - Math.log(lo)))
const normal = (rng) => Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng())
const pick = (rng, list) => list[Math.floor(rng() * list.length)]
const newSeed = (rng) => 1 + Math.floor(rng() * 2 ** 31)
const round = (v, digits = 6) => Number(v.toPrecision(digits))

export function randomGenome(rng) {
  return {
    hidden: [0, 1].map(() => Math.round(logUniform(rng, BOUNDS.hidden))),
    lr: round(logUniform(rng, BOUNDS.lr)),
    decay: round(logUniform(rng, BOUNDS.decay)),
    loss: pick(rng, LOSSES),
    epochs: Math.round(BOUNDS.epochs[0] + rng() * (BOUNDS.epochs[1] - BOUNDS.epochs[0])),
    features: pick(rng, FEATURE_SETS),
    pretrain: false, // reserved for physics pre-training; not used yet
    seed: newSeed(rng),
  }
}

// Step sizes. Each of these fields changes with probability MUTATE_P (independently):
//   hidden[i]: multiplied by exp(N(0, 0.35)), rounded, kept in 8..128 (about +-35% per step)
//   lr, decay: multiplied by exp(N(0, 0.5)) (about x0.6 to x1.6), kept in bounds
//   epochs:    plus round(N(0, 60)), kept in 40..400
//   loss:      flipped to the other loss
//   features:  replaced by a different input set
// The seed is always redrawn, so a child never repeats its parent's shuffles. A child whose hidden sizes and
// features are unchanged keeps the parent's layer shapes and so inherits its trained weights (see evolve.mjs).
export const MUTATE_P = { hidden: 0.25, lr: 0.4, decay: 0.3, epochs: 0.3, loss: 0.1, features: 0.1 }
export function mutate(parent, rng) {
  const g = { ...parent, hidden: [...parent.hidden] }
  for (let i = 0; i < g.hidden.length; i++) if (rng() < MUTATE_P.hidden) g.hidden[i] = Math.round(clamp(g.hidden[i] * Math.exp(0.35 * normal(rng)), BOUNDS.hidden))
  if (rng() < MUTATE_P.lr) g.lr = round(clamp(g.lr * Math.exp(0.5 * normal(rng)), BOUNDS.lr))
  if (rng() < MUTATE_P.decay) g.decay = round(clamp(g.decay * Math.exp(0.5 * normal(rng)), BOUNDS.decay))
  if (rng() < MUTATE_P.epochs) g.epochs = Math.round(clamp(g.epochs + 60 * normal(rng), BOUNDS.epochs))
  if (rng() < MUTATE_P.loss) g.loss = g.loss === 'squared' ? 'huber' : 'squared'
  if (rng() < MUTATE_P.features) g.features = pick(rng, FEATURE_SETS.filter((f) => f !== g.features))
  g.seed = newSeed(rng)
  return g
}

const canonical = (g) => JSON.stringify({ hidden: g.hidden, lr: g.lr, decay: g.decay, loss: g.loss, epochs: g.epochs, features: g.features, pretrain: g.pretrain, seed: g.seed })
export const genomeId = (g) => createHash('sha256').update(canonical(g)).digest('hex').slice(0, 12)

// 'physics' uses the same four laws as the site's switch (PANEL_DESIGN in src/lib/nn.js).
export function toOptions(g) {
  const o = { hidden: g.hidden, lr: g.lr, decay: g.decay, loss: g.loss, epochs: g.epochs, features: g.features, seed: g.seed }
  if (g.features === 'physics') Object.assign(o, { physicsOn: 'base', physics: ['abrams', 'bolomey', 'feret', 'bolomeyAge'] })
  return o
}
