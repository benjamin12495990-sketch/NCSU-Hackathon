// Checks for the alloy generator (npm run alloy-check).
// Part 1: featurize() rebuilds all 530 rows of the alloy network's nn_data.json from alloys.json (1e-6).
// Part 2: trains one network (nn_generic.js, the materials agent's settings, seed 11), runs the
//         generator for three targets plus one adversarial request, and asserts every rule on every
//         returned candidate.
// nn_data.json and nn_generic.js are read from public/model/materials/ and src/lib/ when another
// branch has put them there; result.json and nn_generic.js from domains/materials/work/ (or MATERIALS_SIM_DIR).
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { featurize } from '../src/lib/alloyFeatures.js'
import { generate, generateAsync, spacing, testedSets, TOO_FAR, NOT_NEW, INTERPOLATION, TIER_INTERPOLATION, TIER_NEAR, TIER_FAR, FAR, NEIGHBOURS, SUPPORT_NEEDED } from '../src/lib/alloyGen.js'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const SIM = process.env.MATERIALS_SIM_DIR || join(root, 'domains/materials/work')
const pick = (...paths) => paths.find((p) => existsSync(p))
const nnPath = pick(join(root, 'public/model/materials/nn_data.json'), join(SIM, 'nn_data.json'))
const resultPath = join(SIM, 'result.json')
const data = JSON.parse(readFileSync(join(root, 'public/model/materials/alloys.json'), 'utf8'))
const nn = JSON.parse(readFileSync(nnPath, 'utf8'))
let failures = 0
const fail = (msg) => { failures++; console.log('FAIL', msg) }

// ---- Part 1: feature parity -------------------------------------------------------------------
console.log(`Part 1: feature parity (alloys.json vs ${nnPath})`)
if (JSON.stringify(nn.features) !== JSON.stringify(data.features)) fail('feature lists differ')
const nnRows = [...nn.train.X.map((x, i) => ({ x, y: nn.train.y[i], g: nn.train.g[i], split: 'train' })),
  ...nn.test.X.map((x, i) => ({ x, y: nn.test.y[i], g: nn.test.g[i], split: 'test' }))]
if (nnRows.length !== data.rows.length) fail(`row count ${data.rows.length} vs ${nnRows.length}`)
let worst = 0, worstAt = ''
nnRows.forEach((ref, i) => {
  const row = data.rows[i]
  if (row.hv !== ref.y || row.g !== ref.g || row.split !== ref.split) fail(`row ${i}: hv/g/split differ`)
  const x = featurize(row.frac, row.proc, data)
  x.forEach((v, k) => {
    const d = Math.abs(v - ref.x[k])
    if (d > worst) { worst = d; worstAt = `row ${i} ${data.features[k]}` }
  })
})
const PARITY_TOL = 1e-6 + 1e-12
console.log(`  ${data.rows.length} rows x ${data.features.length} features; largest difference ${worst.toExponential(2)}${worst ? ' at ' + worstAt : ''}`)
if (worst > PARITY_TOL) fail(`feature parity: ${worst} > 1e-6`)
else console.log('  PASS feature parity at 1e-6')
const trainHv = nn.train.y
if (data.clamp_HV[0] !== Math.min(...trainHv) || data.clamp_HV[1] !== Math.max(...trainHv)) fail('clamp range is not the training HV range')
else console.log(`  PASS clamp range ${data.clamp_HV[0]}-${data.clamp_HV[1]} HV is the training range`)
if (failures) { console.log(`${failures} failure(s); part 2 not run`); process.exit(1) }


// ---- Part 2: generator rules on a trained network ---------------------------------------------
const genericPath = join(SIM, 'nn_generic.js')
const { createTrainer, metrics } = await import(pathToFileURL(genericPath).href)
const result = JSON.parse(readFileSync(resultPath, 'utf8'))
const settings = { ...result.network.settings, seed: 11 }
console.log(`\nPart 2: one network trained with ${genericPath}`)
console.log('  settings', JSON.stringify(settings))
const t0 = Date.now()
const net = createTrainer(nn.train, settings)
net.runEpochs(settings.epochs)
const predict = (row) => net.predict(row)
const testMae = metrics(nn.test.X.map(predict), nn.test.y).mae
console.log(`  trained in ${((Date.now() - t0) / 1000).toFixed(1)} s; single-network test MAE ${testMae.toFixed(2)} HV ` +
  `(file: five-network ensemble ${result.b_test_split_once.network.mae.toFixed(2)} HV)`)
const q = data.q90_HV
if (q !== result.d_interval90.halfwidth_HV) fail('q in alloys.json differs from result.json')
console.log(`  q (90% half-width from the file) = ${q.toFixed(1)} HV; clamp ${data.clamp_HV.join('-')} HV`)

const sp = spacing(data)
console.log(`  spacing of tested alloys (L1 to nearest other distinct composition, n=${sp.compositions}): ` +
  ['q10', 'q25', 'q50', 'q75', 'q90'].map((k) => `${k} ${sp[k].toFixed(3)}`).join(', '))
if (Math.abs(sp.q90 - TOO_FAR) > 0.02) fail(`90th percentile spacing ${sp.q90.toFixed(3)} is not near the 0.40 cut`)

// Independent re-implementations for the assertions (dense vectors, brute force).
const allEls = [...new Set(data.rows.flatMap((r) => Object.keys(r.frac)))].sort()
const dense = (f) => allEls.map((el) => f[el] || 0)
const testedDense = data.rows.map((r) => dense(r.frac))
const dist = (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0)
const sets = new Map(testedSets(data).map((s) => [s.key, s]))
const minHv = Math.min(...nn.train.y), maxHv = Math.max(...nn.train.y)

function checkRequest(label, opts) {
  const res = generate({ ...opts, limit: Infinity }, data, predict, q) // assert on every kept candidate
  const again = generate({ ...opts, limit: Infinity }, data, predict, q)
  const shown = generate(opts, data, predict, q)
  if (JSON.stringify(shown.candidates) !== JSON.stringify(res.candidates.slice(0, opts.limit))) fail(`${label}: limit changes the ranking`)
  const st = res.stats, ex = st.excluded
  console.log(`\n${label}: target ${opts.targetHV ?? 'as high as possible'} HV, route ${opts.proc}, ` +
    `only sets tested by this route ${res.stats.rules.routeMustBeTested}, only supported ${res.stats.rules.onlySupported}, seed ${opts.seed}, ` +
    `elements ${Object.keys(opts.allowed).length === data.elements.length ? 'all 21' : Object.keys(opts.allowed).join(' ')}`)
  console.log(`  element sets: ${st.sets.tested} tested, ${st.sets.used} used; skipped: ${st.sets.pooledElement} contain an element pooled in x_other, ` +
    `${st.sets.singleComposition} have only one tested composition, ${st.sets.notAllowed} use an element not allowed, ` +
    `${st.sets.routeNotTested} never tested as ${opts.proc} (switch on), ` +
    `${st.sets.offGrid} cannot be varied on the 0.001 grid without widening, ${st.sets.cappedOut} ruled out by the maximum fractions`)
  const setTotal = Object.entries(st.sets).reduce((a, [k, v]) => a + (k === 'tested' ? 0 : v), 0)
  if (setTotal !== st.sets.tested) fail(`${label}: set counts do not add up`)
  console.log(`  best cautious estimate of any candidate passing the other rules: ${st.bestCautious?.toFixed(0)} HV ` +
    `(clamp ceiling ${st.ceilingCautious.toFixed(0)} HV)`)
  if (opts.targetHV != null && !st.kept && !(st.bestCautious < opts.targetHV)) fail(`${label}: empty although best cautious reaches the target`)
  console.log(`  generated ${st.draws}; excluded: ${ex.duplicate} duplicates, ${ex.notNew} under 0.10 from a tested alloy (not new), ` +
    `${ex.tooFar} over 0.40 (too far), ${ex.subsetCopy} a tested alloy with an element left out, ${ex.clamped} prediction at the clamp, ` +
    `${ex.belowTarget} cautious estimate below target, ${ex.unsupported} fewer than ${SUPPORT_NEEDED} of ${NEIGHBOURS} nearest tested alloys reached the target (unsupported); kept ${st.kept} (${st.tiers[TIER_INTERPOLATION]} interpolation, ${st.tiers[TIER_NEAR]} near-tested, ${st.tiers[TIER_FAR]} far) from ${st.keptSets} sets; returned ${st.returned}`)
  const total = ex.duplicate + ex.notNew + ex.tooFar + ex.subsetCopy + ex.clamped + ex.belowTarget + ex.unsupported + st.kept
  if (total !== st.draws) fail(`${label}: counts do not add up (${total} vs ${st.draws})`)
  if (JSON.stringify(res) !== JSON.stringify(again)) fail(`${label}: not deterministic for the same seed`)
  const offRoute = res.candidates.filter((c) => !c.routeTestedInSet).length
  if (offRoute !== st.keptRouteNotTested) fail(`${label}: keptRouteNotTested count is wrong`)
  console.log(`  kept candidates whose element set was never tested as ${opts.proc}: ${offRoute} of ${res.candidates.length}`)
  let checked = 0
  res.candidates.forEach((c, i) => {
    const where = `${label} #${i + 1} ${c.composition}`
    const s = sets.get(c.set)
    const els = Object.keys(c.frac).sort()
    if (!s || els.join('-') !== c.set) fail(`${where}: element set is not a tested set`)
    for (const el of els) {
      const x = c.frac[el]
      if (!(opts.allowed[el] > 0) || x > opts.allowed[el] + 1e-12) fail(`${where}: ${el} not allowed or above its maximum`)
      if (s && (x < s.lo[el] - 1e-9 || x > s.hi[el] + 1e-9)) fail(`${where}: ${el}=${x} outside tested range [${s.lo[el]}, ${s.hi[el]}] (widened)`)
      if (s && x < 0.05 - 1e-12 && !(s.lo[el] < 0.05)) fail(`${where}: ${el} below 0.05 although the tested alloys of the set never are`)
    }
    const unitSum = c.units.reduce((a, b) => a + b, 0)
    const fracSum = els.reduce((a, el) => a + c.frac[el], 0)
    if (unitSum !== 1000 || Math.abs(fracSum - 1) > 1e-12) fail(`${where}: fractions do not sum to 1 (${unitSum}/1000, ${fracSum})`)
    const cd = dense(c.frac)
    let best = Infinity, at = -1
    testedDense.forEach((t, j) => { const d = dist(cd, t); if (d < best) { best = d; at = j } })
    if (Math.abs(best - c.distance) > 1e-9) fail(`${where}: distance ${c.distance} but brute force gives ${best}`)
    // The named nearest alloy must be at the brute-force minimum (ties within 1e-9 may name either).
    const named = data.rows[c.nearest.index]
    if (named.f !== c.nearest.formula || Math.abs(dist(cd, testedDense[c.nearest.index]) - best) > 1e-9)
      fail(`${where}: nearest tested alloy is not at the minimum distance (brute force: ${data.rows[at].f})`)
    if (best < NOT_NEW || best > TOO_FAR) fail(`${where}: distance ${best} outside 0.10-0.40`)
    const tier = best < INTERPOLATION ? TIER_INTERPOLATION : best < FAR ? TIER_NEAR : TIER_FAR
    if (tier !== c.tier) fail(`${where}: tier ${c.tier} should be ${tier}`)
    const p = predict(featurize(c.frac, opts.proc, data))
    if (Math.abs(p - c.predicted) > 1e-9) fail(`${where}: prediction does not reproduce`)
    if (!(p > minHv + 1e-9 && p < maxHv - 1e-9)) fail(`${where}: prediction ${p} at the clamp`)
    if (Math.abs(c.cautious - (p - q)) > 1e-9) fail(`${where}: cautious is not prediction - q`)
    if (opts.targetHV != null && c.cautious < opts.targetHV) fail(`${where}: cautious ${c.cautious} below target`)
    // Neighbour evidence: the 3 nearest DISTINCT tested compositions by brute force (rows grouped by
    // g, each with the mean of its measurements), the count that reached the target, the support
    // rule, and the gap to the nearest tested alloy.
    const seenG = new Set()
    const order = testedDense.map((t, j) => ({ d: dist(cd, t), j })).sort((a, b) => a.d - b.d || a.j - b.j)
      .filter(({ j }) => !seenG.has(data.rows[j].g) && seenG.add(data.rows[j].g)).slice(0, NEIGHBOURS)
    if (!Array.isArray(c.neighbours) || c.neighbours.length !== NEIGHBOURS) fail(`${where}: expected ${NEIGHBOURS} neighbours`)
    else {
      c.neighbours.forEach((nb, k) => {
        if (Math.abs(nb.distance - order[k].d) > 1e-9) fail(`${where}: neighbour ${k + 1} distance ${nb.distance} but brute force gives ${order[k].d}`)
        const row = data.rows[nb.index]
        const same = data.rows.filter((t) => t.g === row.g)
        const mean = same.reduce((a, t) => a + t.hv, 0) / same.length
        const routes = [...new Set(same.map((t) => t.proc))].sort()
        if (row.f !== nb.formula || row.g !== nb.g || Math.abs(mean - nb.hv) > 1e-9 || nb.n !== same.length || JSON.stringify(routes) !== JSON.stringify(nb.routes))
          fail(`${where}: neighbour ${k + 1} does not match its tested composition`)
        if (data.rows[order[k].j].g !== nb.g && Math.abs(order[k].d - nb.distance) > 1e-9) fail(`${where}: neighbour ${k + 1} is not the brute-force one`)
        if (Math.abs(dist(cd, testedDense[nb.index]) - nb.distance) > 1e-9) fail(`${where}: neighbour ${k + 1} distance is not its own`)
      })
      if (Math.abs(c.neighbours[0].distance - c.distance) > 1e-9) fail(`${where}: first neighbour is not the nearest tested alloy`)
      if (new Set(c.neighbours.map((nb) => nb.g)).size !== NEIGHBOURS) fail(`${where}: neighbours repeat a composition`)
      // With the route rule on, only measurements made by the chosen route count as support.
      const ev = (nb) => {
        if (!st.rules.routeMustBeTested) return nb.hv
        const by = data.rows.filter((t) => t.g === nb.g && t.proc === st.rules.proc).map((t) => t.hv)
        return by.length ? by.reduce((a, b) => a + b, 0) / by.length : null
      }
      const reached = opts.targetHV == null ? null : c.neighbours.filter((nb) => ev(nb) != null && ev(nb) >= opts.targetHV).length
      if (reached !== c.neighboursReached) fail(`${where}: neighboursReached ${c.neighboursReached} should be ${reached}`)
      if (st.rules.onlySupported && opts.targetHV != null && !(c.neighboursReached >= SUPPORT_NEEDED)) fail(`${where}: unsupported candidate kept`)
    }
    if (Math.abs(c.gapHV - (c.predicted - c.nearest.hv)) > 1e-9) fail(`${where}: gap is not predicted minus nearest measured HV`)
    {
      // The nearest cell is the nearest distinct composition: the mean of its measurements by the chosen
      // route when the route rule is on and it has any, otherwise the mean of all its measurements.
      const g0 = c.neighbours[0].g, all = data.rows.filter((t) => t.g === g0), by = all.filter((t) => t.proc === st.rules.proc)
      const use = st.rules.routeMustBeTested && by.length ? by : all
      const mean = use.reduce((a, t) => a + t.hv, 0) / use.length
      if (Math.abs(mean - c.nearest.hv) > 1e-9 || c.nearest.n !== use.length || c.nearest.formula !== c.neighbours[0].formula) fail(`${where}: nearest cell does not match its composition's measurements`)
    }
    if (st.rules.routeMustBeTested && !c.routeTestedInSet) fail(`${where}: set never tested by ${opts.proc} although the route rule is on`)
    const far = opts.gapLimit != null && c.gapHV > opts.gapLimit
    if (c.farAbove !== far) fail(`${where}: farAbove ${c.farAbove} should be ${far}`)
    if (i > 0) {
      const b = res.candidates[i - 1]
      if (b.farAbove && !c.farAbove) fail(`${where}: ranked below a candidate the network rates far above its neighbours`)
      else if (b.farAbove === c.farAbove && (b.cautious < c.cautious || (b.cautious === c.cautious && b.distance > c.distance))) fail(`${where}: ranked out of order`)
    }
    checked++
  })
  console.log(`  asserted every rule on all ${checked} kept candidates (this list returns the first ${opts.limit}; the page shows 10); deterministic for seed ${opts.seed}: ${JSON.stringify(res) === JSON.stringify(again)}`)
  console.log('  top five:')
  for (const c of res.candidates.slice(0, 5)) {
    console.log(`    ${c.composition.padEnd(44)} pred ${c.predicted.toFixed(0)} HV, cautious ${c.cautious.toFixed(0)} HV, ` +
      `distance ${c.distance.toFixed(3)} (${c.tier}); nearest tested: ${c.nearest.formula} ${c.nearest.proc} ${c.nearest.hv} HV, gap ${c.gapHV >= 0 ? '+' : ''}${c.gapHV.toFixed(0)} HV; ` +
      `neighbours ${c.neighbours.map((nb) => nb.hv).join('/')} HV` + (c.neighboursReached == null ? '' : `, ${c.neighboursReached} of ${NEIGHBOURS} reached the target`) +
      (c.routeTestedInSet ? '' : `; set never tested as ${c.proc} (only ${c.setRoutes.join('/')})`))
  }
  if (!res.candidates.length) console.log('    (none)')
  return res
}

const all = Object.fromEntries(data.elements.map((el) => [el, 1]))
const defaults = generate({ targetHV: 400, allowed: all }, data, predict, q).stats.rules
if (defaults.routeMustBeTested !== true) fail('routeMustBeTested should default to true')
if (defaults.onlySupported !== false) fail('onlySupported should default to false in the library (the page turns it on with a target)')
for (const target of [400, 600, 800]) checkRequest(`Request ${target}`, { targetHV: target, allowed: all, proc: 'CAST', limit: 20, seed: 1 })

// The neighbour-support rule: the same request with onlySupported, and how many it removed.
const plain400 = generate({ targetHV: 400, allowed: all, proc: 'CAST', limit: Infinity, seed: 1 }, data, predict, q)
for (const target of [400, 600, 800]) {
  const sup = checkRequest(`Request ${target}, only supported`, { targetHV: target, allowed: all, proc: 'CAST', limit: 20, seed: 1, onlySupported: true })
  if (target === 400) {
    const full = generate({ targetHV: 400, allowed: all, proc: 'CAST', limit: Infinity, seed: 1, onlySupported: true }, data, predict, q)
    const removed = full.stats.excluded.unsupported
    if (full.stats.kept !== plain400.stats.kept - removed) fail('onlySupported: kept + removed does not equal the plain request')
    const plainSupported = plain400.candidates.filter((c) => c.neighboursReached >= SUPPORT_NEEDED).map((c) => c.composition)
    if (JSON.stringify(plainSupported) !== JSON.stringify(full.candidates.map((c) => c.composition))) fail('onlySupported keeps a different set than filtering the plain request')
    console.log(`  onlySupported at 400 HV removed ${removed} of ${plain400.stats.kept} candidates; ${full.stats.kept} kept` +
      ` (supported candidates in the plain list: ${plainSupported.length})`)
    void sup
  }
}
const noTarget = generate({ targetHV: null, allowed: all, proc: 'CAST', limit: 5, seed: 1, onlySupported: true }, data, predict, q)
if (noTarget.stats.excluded.unsupported !== 0 || noTarget.candidates.some((c) => c.neighboursReached !== null)) fail('onlySupported without a target should change nothing')
else console.log('\nonlySupported without a target removes nothing and neighboursReached is null: true')

// Only real routes can be generated for: UNKNOWN and OTHER are data labels.
if (JSON.stringify(data.routes_selectable) !== JSON.stringify(data.procs.filter((p) => p !== 'UNKNOWN' && p !== 'OTHER'))) fail('routes_selectable should be procs without UNKNOWN and OTHER')
for (const bad of ['UNKNOWN', 'OTHER']) {
  let threw = false
  try { generate({ targetHV: 400, allowed: all, proc: bad, seed: 1 }, data, predict, q) } catch { threw = true }
  if (!threw) fail(`generate accepted route ${bad}`)
  else console.log(`\nRoute ${bad} is refused (routes to generate for: ${data.routes_selectable.join(', ')})`)
}

// The page's ranking: candidates rated more than one familiar-system error (family.json) above
// their nearest tested alloy go below the others.
const famFile = JSON.parse(readFileSync(join(root, 'public/model/materials/family.json'), 'utf8'))
const gapRun = checkRequest(`Request 400, only supported, gap limit ${famFile.test.network.toFixed(1)} HV (the page's settings)`,
  { targetHV: 400, allowed: all, proc: 'CAST', limit: Infinity, seed: 1, onlySupported: true, gapLimit: famFile.test.network })
console.log(`  ${gapRun.stats.keptFarAbove} of ${gapRun.stats.kept} kept candidates are rated more than ${famFile.test.network.toFixed(1)} HV above their nearest tested alloy and ranked last`)

// The user's maximum fractions: Al at most 0.10, Ti at most 0.15, everything else allowed.
checkRequest('Capped request (Al <= 0.10, Ti <= 0.15)', { targetHV: 400, allowed: { ...all, Al: 0.1, Ti: 0.15 }, proc: 'CAST', limit: 20, seed: 1 })

// The optional switch: only element sets that were tested by the chosen route.
const sameRoute = checkRequest('Request 400, only sets tested as CAST', { targetHV: 400, allowed: all, proc: 'CAST', limit: 20, seed: 1, routeMustBeTested: true })
if (sameRoute.candidates.some((c) => !c.routeTestedInSet)) fail('routeMustBeTested let an untested route through')
checkRequest('Request 400, route rule switched off', { targetHV: 400, allowed: all, proc: 'CAST', limit: 20, seed: 1, routeMustBeTested: false })

// The async version (the page's) gives exactly the same answer as the synchronous one.
for (const opts of [{ targetHV: 400, onlySupported: true }, { targetHV: 800, onlySupported: true }, { targetHV: null, routeMustBeTested: false }]) {
  const o = { allowed: all, proc: 'CAST', limit: 10, seed: 1, ...opts }
  const a = await generateAsync(o, data, predict, q, { sliceMs: 5 })
  const { timing, ...rest } = a.stats
  const b = generate(o, data, predict, q)
  if (JSON.stringify({ ...a, stats: rest }) !== JSON.stringify(b)) fail(`generateAsync differs from generate for ${JSON.stringify(opts)}`)
  else console.log(`generateAsync matches generate for ${JSON.stringify(opts)}: ${timing.ms.toFixed(0)} ms, ${timing.yields} yields at a 5 ms slice`)
}

// The clamp rule fires: a predictor stuck at either training limit must leave nothing.
for (const stuck of data.clamp_HV) {
  const r = generate({ targetHV: null, allowed: all, proc: 'CAST', seed: 1 }, data, () => stuck, q)
  if (r.stats.kept !== 0 || r.stats.excluded.clamped === 0) fail(`clamp rule: prediction ${stuck} was not dropped`)
  else console.log(`\nClamp rule: a predictor stuck at ${stuck} HV leaves 0 candidates (${r.stats.excluded.clamped} dropped at the clamp)`)
}

// Adversarial: only the best-studied five elements. A candidate must not be a tested alloy rescaled,
// nor a tested alloy with some elements left out and the rest rescaled.
const five = { Al: 1, Co: 1, Cr: 1, Fe: 1, Ni: 1 }
const adv = checkRequest('Adversarial (Al Co Cr Fe Ni only)', { targetHV: null, allowed: five, proc: 'CAST', limit: 50, seed: 1 })
let closest = Infinity, closestAt = ''
for (const c of adv.candidates) {
  const els = Object.keys(c.frac)
  if (els.some((el) => !(el in five))) fail(`adversarial: ${c.composition} uses an element outside Al Co Cr Fe Ni`)
  for (const r of data.rows) {
    if (!els.every((el) => r.frac[el] > 0)) continue // tested alloy lacks one of the candidate's elements
    const t = els.reduce((a, el) => a + r.frac[el], 0)
    const d = els.reduce((a, el) => a + Math.abs(c.frac[el] - r.frac[el] / t), 0)
    if (d < closest) { closest = d; closestAt = `${c.composition} vs ${r.f}` }
    if (d < NOT_NEW) fail(`adversarial: ${c.composition} is ${r.f} rescaled (L1 ${d.toFixed(3)})`)
  }
}
console.log(`  rescaled-copy test over ${adv.candidates.length} candidates: closest rescaled tested alloy L1 ${closest.toFixed(3)} (${closestAt})`)

// How much the answer depends on the seed, and how long one request takes.
const seedRows = [1, 2, 3].map((seed) => {
  const t = Date.now()
  const r = generate({ targetHV: 400, allowed: all, proc: 'CAST', limit: 20, seed }, data, predict, q)
  return `seed ${seed}: kept ${r.stats.kept}, best cautious ${r.stats.bestCautious.toFixed(0)} HV, top ${r.candidates[0]?.composition} (${Date.now() - t} ms)`
})
console.log(`\nSeed sensitivity, target 400 HV, all elements, CAST:\n  ${seedRows.join('\n  ')}`)

console.log(failures ? `\n${failures} failure(s)` : '\nALL CHECKS PASSED')
process.exit(failures ? 1 : 0)
