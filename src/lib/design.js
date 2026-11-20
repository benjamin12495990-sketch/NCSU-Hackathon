import { featurize } from './featurize.js'
import { predict } from './predict.js'
import { co2PerM3, absoluteVolume } from './co2.js'
import { mixKey, nearestTested, resultUpTo, scaledDistance } from './nearest.js'

// Small seeded random generator, so the same request always gives the same
// candidates (a demo must not change between clicks).
function seeded(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

// Makes one new mix from a parent mix: change the binder amount, the share of
// slag and fly ash, the water/binder ratio and the superplasticizer dose, then
// rescale both aggregates so the mix keeps the parent's volume (the "absolute
// volume method"). Returns null if any amount leaves the range labs tested.
function variant(parent, rand, step, opts, data) {
  const { space, factors, spec } = data
  const L = space.limits
  const [c0, s0, f0, w0, sp0, ca0, fa0] = parent
  const b0 = c0 + s0 + f0
  const binder = clamp(b0 * (1 + step * (rand() * 0.3 - 0.2)), L.binder[0], L.binder[1])
  let slagFrac = opts.allowSlag ? clamp(s0 / b0 + step * (rand() - 0.5) * 0.5, 0, L.slag_fraction_max) : 0
  let flyFrac = opts.allowFlyAsh ? clamp(f0 / b0 + step * (rand() - 0.5) * 0.5, 0, L.flyash_fraction_max) : 0
  if (slagFrac < 0.05) slagFrac = 0 // nobody batches a few kg of slag
  if (flyFrac < 0.05) flyFrac = 0
  const scm = slagFrac + flyFrac
  if (scm > L.scm_fraction_max) {
    slagFrac *= L.scm_fraction_max / scm
    flyFrac *= L.scm_fraction_max / scm
  }
  const wb = clamp((w0 / b0) * (1 + step * (rand() - 0.5) * 0.2), L.water_binder_ratio[0], L.water_binder_ratio[1])
  const spPerBinder = clamp(sp0 / b0 + step * (rand() - 0.5) * 0.01, 0, L.superplasticizer_per_binder_max)

  const slag = Math.round(binder * slagFrac)
  const fly = Math.round(binder * flyFrac)
  const cement = Math.round(binder) - slag - fly
  const water = Math.round(wb * binder)
  const sp = Math.round(spPerBinder * binder * 10) / 10
  const paste = [cement, slag, fly, water, sp, 0, 0]
  const aggregateVolume = absoluteVolume(parent, factors, spec.ingredients) - absoluteVolume(paste, factors, spec.ingredients)
  const k = aggregateVolume / absoluteVolume([0, 0, 0, 0, 0, ca0, fa0], factors, spec.ingredients)
  const mix = [cement, slag, fly, water, sp, Math.round(ca0 * k), Math.round(fa0 * k)]
  for (let i = 0; i < mix.length; i++) {
    const [lo, hi] = space.ranges[spec.ingredients[i]]
    if (!(mix[i] >= lo && mix[i] <= hi)) return null
  }
  return mix
}

// Inverse design. Given a target strength (MPa) at an age (days):
//   1. make thousands of new mixes by varying every tested mix,
//   2. drop any that is already in the dataset,
//   3. keep those whose CAUTIOUS strength estimate reaches the target
//      (or, if the user turns caution off, whose expected strength does),
//   4. rank by embodied CO2,
//   5. check each against its three nearest lab-tested mixes: it counts as
//      "lab-supported" only if at least two of those three are close to it
//      and actually reached the target in the lab.
export function design(request, data, limit = 10) {
  const { spec, model, modelLow, known, metrics, factors } = data
  const opts = { allowSlag: true, allowFlyAsh: true, onlySupported: true, cautious: true, ...request }
  const { targetMPa, ageDays } = opts
  const rand = seeded(20261120)
  const stats = { generated: 0, removedKnown: 0, meetTarget: 0, hiddenUnsupported: 0 }
  const seen = new Set()
  const feasible = []

  const consider = (mix) => {
    if (!mix) return
    const key = mixKey(mix)
    if (seen.has(key)) return
    seen.add(key)
    stats.generated++
    if (known.keys.has(key)) return void stats.removedKnown++
    const x = featurize(mix, ageDays, spec)
    const strength = predict(model, x)
    if (strength < targetMPa) return
    const low = predict(modelLow, x)
    if (opts.cautious && low < targetMPa) return
    stats.meetTarget++
    feasible.push({ mix, strength, low, co2: co2PerM3(mix, factors, spec.ingredients) })
  }

  const usable = (mix) => (opts.allowSlag || mix[1] === 0) && (opts.allowFlyAsh || mix[2] === 0)
  for (const e of known.entries) for (let i = 0; i < 40; i++) consider(variant(e.mix, rand, 1, opts, data))
  // Second pass: fine-tune around the best mixes found so far.
  feasible.sort((a, b) => a.co2 - b.co2)
  for (const c of feasible.slice(0, 40)) for (let i = 0; i < 60; i++) consider(variant(c.mix, rand, 0.4, opts, data))
  feasible.sort((a, b) => a.co2 - b.co2)

  // Tested mixes that have a lab result at this age or younger. Concrete keeps
  // gaining strength, so a result at a younger age is a safe comparison.
  const tested = []
  for (const entry of known.entries) {
    const result = resultUpTo(entry, ageDays)
    if (result) tested.push({ entry, result })
  }
  const bestTested = (allowed) => {
    let best = null
    for (const { entry, result } of tested) {
      if (!allowed(entry.mix) || result[1] < targetMPa) continue
      const co2 = co2PerM3(entry.mix, factors, spec.ingredients)
      if (!best || co2 < best.co2) best = { mix: entry.mix, co2, age: result[0], strength: result[1] }
    }
    return best
  }
  const baseline = bestTested(usable)
  const cementOnly = bestTested((mix) => mix[1] === 0 && mix[2] === 0)

  // Walk down the ranking, skipping near-copies of mixes already listed.
  const top = []
  for (const c of feasible) {
    if (top.length >= limit) break
    if (top.some((t) => scaledDistance(t.mix, c.mix, known.scale) < 0.1)) continue
    const neighbours = nearestTested(known, tested, c.mix, 3)
    const backing = neighbours.filter((n) => n.distance <= metrics.low_confidence_distance && n.strength >= targetMPa)
    const supported = backing.length >= 2
    if (!supported && opts.onlySupported) { stats.hiddenUnsupported++; continue }
    top.push({
      ...c,
      waterBinder: c.mix[3] / (c.mix[0] + c.mix[1] + c.mix[2]),
      vsBaseline: baseline ? c.co2 / baseline.co2 - 1 : null,
      neighbours,
      supported,
    })
  }
  return { targetMPa, ageDays, top, baseline, cementOnly, stats }
}
