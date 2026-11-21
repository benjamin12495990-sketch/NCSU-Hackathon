// Proposes untested alloy compositions and ranks them by the network's lower 90% edge.
// Rules (see the critique for why each one exists):
//  - Only element SETS that have been tested; each element varies inside the range of fractions
//    seen in the tested alloys of that set (never widened), capped by the user's maximum.
//    So no element goes below 0.05 unless the tested alloys of that set have it lower.
//  - Fractions are whole multiples of 0.001 that sum to exactly 1000/1000.
//  - Distance = L1 on element fractions to the nearest tested alloy (any processing route).
//    Under 0.10: not new, excluded. 0.10-0.20: "interpolation between tested alloys".
//    0.20-0.30: "untested composition near tested ones". 0.30-0.40: "far from tested alloys", the
//    least reliable tier. Over 0.40 (about the 90th percentile of
//    the spacing between tested alloys, measured by spacing()): dropped as too far from the data.
//  - A tested alloy with one or more elements left out and the rest rescaled (L1 under 0.10 to
//    that projection) is excluded as a trivial copy.
//  - A prediction at the clamp limits (the training HV range) is dropped: it means the network is
//    outside its data.
//  - Cautious estimate = prediction - q. Kept if it reaches the target; ranked by cautious
//    estimate, then by smaller distance. Deterministic for a given seed.
//  - Neighbour evidence: every kept candidate carries its 3 nearest DISTINCT tested compositions
//    (L1, any route; repeat measurements of one composition are one neighbour, with the mean of its
//    measurements, how many there were and the routes they were made by), how many of them
//    measured at least the target on that mean, and the gap "predicted minus the nearest tested
//    alloy's measured HV". With onlySupported (and a target) a candidate is kept only if at least
//    2 of the 3 reached the target; the ones it removes are counted as excluded.unsupported.
//  - gapLimit (HV, optional): candidates the network rates more than this above their nearest
//    tested alloy are flagged farAbove and ranked after all the others.
//  - routeMustBeTested (default true): only element sets that were tested by the chosen route.
//  - proc must be one of data.routes_selectable (UNKNOWN and OTHER are data labels, not routes).
import { featurize } from './alloyFeatures.js'

export const NOT_NEW = 0.1
export const INTERPOLATION = 0.2
export const TOO_FAR = 0.4
export const UNITS = 1000 // fractions in steps of 0.001
export const TIER_INTERPOLATION = 'interpolation between tested alloys'
export const TIER_NEAR = 'untested composition near tested ones'
export const FAR = 0.3
export const TIER_FAR = 'far from tested alloys (least reliable)'
export const tierOf = (d) => (d < INTERPOLATION ? TIER_INTERPOLATION : d < FAR ? TIER_NEAR : TIER_FAR)

// Seeded random numbers (the same generator as src/lib/nn.js, copied so nn.js stays untouched).
export function seeded(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const setKey = (frac) => Object.keys(frac).filter((el) => frac[el] > 0).sort().join('-')

// L1 distance between two compositions given as { El: x }.
export function l1(a, b) {
  let d = 0
  for (const el in a) d += Math.abs(a[el] - (b[el] || 0))
  for (const el in b) if (!(el in a)) d += b[el]
  return d
}

// The tested compositions as one dense matrix, built once per alloys.json.
const denseCache = new WeakMap()
function denseRows(data) {
  if (!denseCache.has(data)) {
    const els = [...new Set(data.rows.flatMap((r) => Object.keys(r.frac)))]
    const col = new Map(els.map((el, i) => [el, i]))
    const m = new Float64Array(data.rows.length * els.length), sums = new Float64Array(data.rows.length)
    data.rows.forEach((r, i) => {
      for (const [el, x] of Object.entries(r.frac)) { m[i * els.length + col.get(el)] = x; sums[i] += x }
    })
    denseCache.set(data, { col, m, sums, w: els.length })
  }
  return denseCache.get(data)
}

export const NEIGHBOURS = 3
export const SUPPORT_NEEDED = 2

// Tested rows grouped by composition (g): the row indices of each distinct composition.
const groupCache = new WeakMap()
function groups(data) {
  if (!groupCache.has(data)) {
    const byG = new Map()
    data.rows.forEach((r, i) => { if (!byG.has(r.g)) byG.set(r.g, []); byG.get(r.g).push(i) })
    groupCache.set(data, byG)
  }
  return groupCache.get(data)
}

// The k nearest DISTINCT tested compositions (any route), nearest first; ties keep the lower row
// index. Rows of one composition share a distance, so each composition enters once, by its first row.
// Each neighbour: { formula, g, index (first row), distance, hv (mean of its measurements), n, hvs, routes,
// byRoute {route: mean HV of the measurements made by that route} }.
export function nearestK(frac, data, k = NEIGHBOURS) {
  const { col, m, sums, w } = denseRows(data)
  const byG = groups(data)
  const els = Object.keys(frac)
  const idx = els.map((el) => col.get(el)), xs = els.map((el) => frac[el])
  const best = [] // [{ distance, index }], sorted
  for (let r = 0; r < sums.length; r++) {
    let d = 0, cover = 0
    for (let j = 0; j < idx.length; j++) {
      const y = idx[j] === undefined ? 0 : m[r * w + idx[j]]
      d += Math.abs(xs[j] - y); cover += y
    }
    d += sums[r] - cover
    if (best.some((b) => b.g === data.rows[r].g)) continue
    if (best.length < k || d < best[best.length - 1].distance) {
      let at = best.length
      while (at > 0 && best[at - 1].distance > d) at--
      best.splice(at, 0, { distance: d, index: r, g: data.rows[r].g })
      if (best.length > k) best.pop()
    }
  }
  return best.map(({ distance, index, g }) => {
    const rows = byG.get(g).map((i) => data.rows[i])
    const hvs = rows.map((t) => t.hv)
    const routes = [...new Set(rows.map((t) => t.proc))].sort()
    const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length
    const byRoute = Object.fromEntries(routes.map((p) => [p, mean(rows.filter((t) => t.proc === p).map((t) => t.hv))]))
    return { formula: data.rows[index].f, g, index, distance, hv: mean(hvs), n: hvs.length, hvs, routes, byRoute }
  })
}

// Nearest tested alloy (train and test rows, any route) to a composition. The L1 distance is
// sum over the candidate's elements of |x - y| plus the tested alloy's fraction in other elements.
export function nearestTested(frac, data) {
  const { col, m, sums, w } = denseRows(data)
  const els = Object.keys(frac)
  const idx = els.map((el) => col.get(el)), xs = els.map((el) => frac[el])
  let best = Infinity, at = -1
  for (let r = 0; r < sums.length; r++) {
    let d = 0, cover = 0
    for (let k = 0; k < idx.length; k++) {
      const y = idx[k] === undefined ? 0 : m[r * w + idx[k]]
      d += Math.abs(xs[k] - y); cover += y
    }
    d += sums[r] - cover
    if (d < best) { best = d; at = r }
  }
  return { distance: best, index: at }
}

const quantile = (sorted, p) => {
  const h = (sorted.length - 1) * p, lo = Math.floor(h)
  return sorted[lo] + (h - lo) * ((sorted[lo + 1] ?? sorted[lo]) - sorted[lo])
}

// Spacing of the tested alloys: L1 from each distinct composition to its nearest other one.
export function spacing(data) {
  const distinct = new Map()
  for (const r of data.rows) if (!distinct.has(r.g)) distinct.set(r.g, r.frac)
  const fr = [...distinct.values()]
  const nn = fr.map((a, i) => Math.min(...fr.map((b, j) => (i === j ? Infinity : l1(a, b))))).sort((x, y) => x - y)
  const q = {}
  for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) q[`q${Math.round(p * 100)}`] = quantile(nn, p)
  return { compositions: fr.length, ...q }
}

// The tested element sets and the range of each element's fraction inside each set.
export function testedSets(data) {
  const sets = new Map()
  data.rows.forEach((r, i) => {
    const key = setKey(r.frac)
    if (!sets.has(key)) sets.set(key, { key, els: key.split('-'), rows: [], lo: {}, hi: {}, routes: [] })
    const s = sets.get(key)
    s.rows.push(i)
    if (!s.routes.includes(r.proc)) s.routes.push(r.proc)
    for (const el of s.els) {
      s.lo[el] = Math.min(s.lo[el] ?? Infinity, r.frac[el])
      s.hi[el] = Math.max(s.hi[el] ?? -Infinity, r.frac[el])
    }
  })
  return [...sets.values()]
}

export const compositionString = (frac) =>
  Object.keys(frac).sort().map((el) => `${el}${(frac[el] * 100).toFixed(1)}`).join(' ')

// options: targetHV (null = as high as possible), allowed { El: maxFraction }, proc, limit,
//          seed, perSet (draws per element set), routeMustBeTested (only sets tested by proc)
// data: alloys.json; predict(row of 37 inputs) -> HV; q: half-width of the 90% interval in HV.
//          onlySupported (keep a candidate only if 2 of its 3 nearest tested alloys reached the target)
// generate() runs in one go; generateAsync() gives the browser a turn between element sets.
export function generate(options, data, predict, q) {
  const run = generateSteps(options, data, predict, q)
  let step = run.next()
  while (!step.done) step = run.next()
  return step.value
}

// Same result as generate(); yields to the browser (setTimeout 0) between element sets whenever
// more than sliceMs has passed since the last yield, and reports the time taken and the yields.
export async function generateAsync(options, data, predict, q, { sliceMs = 40 } = {}) {
  const t0 = performance.now()
  let last = t0, yields = 0
  const run = generateSteps(options, data, predict, q)
  let step = run.next()
  while (!step.done) {
    if (performance.now() - last > sliceMs) {
      await new Promise((r) => setTimeout(r, 0))
      yields++
      last = performance.now()
    }
    step = run.next()
  }
  step.value.stats.timing = { ms: performance.now() - t0, yields }
  return step.value
}

function* generateSteps(options, data, predict, q) {
  const { targetHV = null, allowed = {}, proc = 'CAST', limit = 20, seed = 1, perSet = 300, routeMustBeTested = true, onlySupported = false, gapLimit = null } = options
  if (data.routes_selectable && !data.routes_selectable.includes(proc)) {
    throw new Error(`route ${proc} is not one to generate for (choose one of ${data.routes_selectable.join(', ')})`)
  }
  const rand = seeded(seed)
  const own = new Set(data.elements)
  const [clampLo, clampHi] = data.clamp_HV
  const sets = { tested: 0, pooledElement: 0, singleComposition: 0, notAllowed: 0, routeNotTested: 0, offGrid: 0, cappedOut: 0, used: 0 }
  const excluded = { duplicate: 0, notNew: 0, tooFar: 0, subsetCopy: 0, clamped: 0, belowTarget: 0, unsupported: 0 }
  const all = testedSets(data)
  sets.tested = all.length
  let draws = 0, bestCautious = -Infinity, bestMeasured = null
  const kept = []
  for (const s of all) {
    yield
    if (s.els.some((el) => !own.has(el))) { sets.pooledElement++; continue }
    if (new Set(s.rows.map((i) => data.rows[i].g)).size < 2) { sets.singleComposition++; continue }
    if (s.els.some((el) => !(allowed[el] > 0))) { sets.notAllowed++; continue }
    if (routeMustBeTested && !s.routes.includes(proc)) { sets.routeNotTested++; continue }
    const lo = s.els.map((el) => Math.ceil(s.lo[el] * UNITS - 1e-6))
    const hiData = s.els.map((el) => Math.floor(s.hi[el] * UNITS + 1e-6))
    const hi = s.els.map((el, i) => Math.min(hiData[i], Math.floor(Math.min(1, allowed[el]) * UNITS + 1e-6)))
    const sum = (a) => a.reduce((t, v) => t + v, 0)
    const fitsData = lo.every((v, i) => v <= hiData[i]) && sum(lo) <= UNITS && sum(hiData) >= UNITS
    if (!fitsData) { sets.offGrid++; continue }
    if (!(lo.every((v, i) => v <= hi[i]) && sum(hi) >= UNITS)) { sets.cappedOut++; continue }
    sets.used++
    // The hardest tested alloy in the usable sets (by the chosen route when the route rule is on).
    for (const i of s.rows) {
      const t = data.rows[i]
      if ((!routeMustBeTested || t.proc === proc) && (!bestMeasured || t.hv > bestMeasured.hv)) bestMeasured = { hv: t.hv, formula: t.f, proc: t.proc }
    }
    // Tested alloys that contain every element of this set and more, projected onto this set.
    const supers = []
    for (const r of data.rows) {
      const k = Object.keys(r.frac)
      if (k.length > s.els.length && s.els.every((el) => r.frac[el] > 0)) {
        const t = s.els.reduce((a, el) => a + r.frac[el], 0)
        supers.push(Object.fromEntries(s.els.map((el) => [el, r.frac[el] / t])))
      }
    }
    const seen = new Set()
    const n = s.els.length
    for (let d = 0; d < perSet; d++) {
      draws++
      const order = Array.from({ length: n }, (_, i) => i)
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]] }
      const u = new Array(n)
      let rem = UNITS
      for (let p = 0; p < n - 1; p++) {
        const i = order[p]
        let restLo = 0, restHi = 0
        for (let r = p + 1; r < n; r++) { restLo += lo[order[r]]; restHi += hi[order[r]] }
        const a = Math.max(lo[i], rem - restHi), b = Math.min(hi[i], rem - restLo)
        u[i] = a + Math.floor(rand() * (b - a + 1))
        rem -= u[i]
      }
      u[order[n - 1]] = rem
      const key = u.join(',')
      if (seen.has(key)) { excluded.duplicate++; continue }
      seen.add(key)
      const frac = Object.fromEntries(s.els.map((el, i) => [el, u[i] / UNITS]))
      const near = nearestTested(frac, data)
      if (near.distance < NOT_NEW) { excluded.notNew++; continue }
      if (near.distance > TOO_FAR) { excluded.tooFar++; continue }
      if (supers.some((sup) => l1(frac, sup) < NOT_NEW)) { excluded.subsetCopy++; continue }
      const predicted = predict(featurize(frac, proc, data))
      if (predicted <= clampLo + 1e-9 || predicted >= clampHi - 1e-9) { excluded.clamped++; continue }
      const cautious = predicted - q
      bestCautious = Math.max(bestCautious, cautious)
      if (targetHV != null && cautious < targetHV) { excluded.belowTarget++; continue }
      const neighbours = nearestK(frac, data)
      // With the route rule on, only measurements made by the chosen route count as support, and the
      // gap is taken against the nearest composition's measurements by that route when it has any.
      const evidence = (nb) => (routeMustBeTested ? nb.byRoute[proc] : nb.hv)
      const reached = targetHV == null ? null : neighbours.filter((nb) => evidence(nb) != null && evidence(nb) >= targetHV).length
      if (onlySupported && targetHV != null && reached < SUPPORT_NEEDED) { excluded.unsupported++; continue }
      const r = data.rows[near.index], nb0 = neighbours[0]
      const sameRoute = routeMustBeTested && nb0.byRoute[proc] != null
      const ref = { formula: nb0.formula, proc: sameRoute ? proc : nb0.routes.join('/'), hv: sameRoute ? nb0.byRoute[proc] : nb0.hv,
        n: sameRoute ? data.rows.filter((t) => t.g === nb0.g && t.proc === proc).length : nb0.n, split: r.split, index: nb0.index }
      kept.push({
        composition: compositionString(frac), frac, units: u, set: s.key, proc,
        // Routes under which this element set was tested; false means the network has never seen
        // this set made by the chosen route, which the page should say next to the row.
        setRoutes: [...s.routes].sort(), routeTestedInSet: s.routes.includes(proc),
        predicted, cautious, distance: near.distance,
        tier: tierOf(near.distance),
        nearest: ref,
        // Predicted minus the nearest tested alloy's measured HV: a large positive gap means the
        // network rates this composition far above the closest thing anyone measured.
        gapHV: predicted - ref.hv,
        farAbove: gapLimit != null && predicted - ref.hv > gapLimit,
        neighbours, neighboursReached: reached,
      })
    }
  }
  kept.sort((a, b) => a.farAbove - b.farAbove || b.cautious - a.cautious || a.distance - b.distance || (a.composition < b.composition ? -1 : 1))
  const tiers = { [TIER_INTERPOLATION]: 0, [TIER_NEAR]: 0, [TIER_FAR]: 0 }
  for (const c of kept) tiers[c.tier]++
  return {
    candidates: kept.slice(0, limit),
    stats: {
      sets, draws, unique: draws - excluded.duplicate, excluded, kept: kept.length,
      returned: Math.min(limit, kept.length), tiers,
      keptRouteNotTested: kept.filter((c) => !c.routeTestedInSet).length,
      keptFarAbove: kept.filter((c) => c.farAbove).length, bestMeasured,
      // The highest cautious estimate any candidate that passed every other rule reached, and the
      // highest the clamp allows at all: a target above either cannot be met.
      bestCautious: Number.isFinite(bestCautious) ? bestCautious : null, ceilingCautious: clampHi - q, keptSets: new Set(kept.map((c) => c.set)).size,
      rules: { notNew: NOT_NEW, interpolation: INTERPOLATION, tooFar: TOO_FAR, step: 1 / UNITS, clamp: data.clamp_HV, q, targetHV, proc, seed, perSet, routeMustBeTested, onlySupported, gapLimit, neighbours: NEIGHBOURS, supportNeeded: SUPPORT_NEEDED },
    },
  }
}
