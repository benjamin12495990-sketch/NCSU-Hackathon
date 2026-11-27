import { useEffect, useMemo, useRef, useState } from 'react'
import { loadPipeline, predictTrees, fmt } from '../../lib/pipeline.js'
import './ElectrolyteDesigner.css'

// Solid-electrolyte designer (see README.md in this folder). The visitor names the room-temperature ionic conductivity a lithium
// solid electrolyte must reach. The page starts from the most conductive measured compositions in electrolyte_model.json, makes
// every nudge of them (in a fixed order, with no random numbers, so the same inputs always give the same list), predicts each with
// the exported trees, ranks them by the cautious lower estimate (prediction minus the 90% half-width) and shows the nearest measured
// compositions. Every number shown comes from that file; the constants below are the generator's settings, and the page states them.
// A nudge works on element fractions only: it does not balance charge, and the page says so wherever it shows a new formula.

const PRESETS = [-4, -3, -2] // target chips, log10(S/cm)
const SEED_COUNTS = [10, 25, 50] // how many measured compositions to start from
const STEP = 0.005 // a move takes 2% to 5% of all atoms, in steps of 0.5%, and never more than half of the donor element
const MOVE_MIN = 4
const MOVE_MAX = 10
const MINOR = 0.15 // an element (not Li) below this atom fraction may be swapped
const SUBS_MAX = 8 // a minor element is swapped for at most this many others, the most frequent first
const SAME_L1 = 0.01 // closer than this to a measured composition counts as that composition
const CLOSE_L1 = 0.2 // a "close" measured neighbour differs in at most 10% of the atoms (L1 / 2)
const SHOWN = 30 // rows in the ranked table
const PER_SEED_SHOWN = 2 // at most this many rows from one starting composition (when that option is on)
const SEEDS_FOLDED = 10
const SAME_F = 2e-6 // the export rounds fractions to 6 decimals: closer than this is the same number

// Short option texts, so a select never cuts them off; the form's note says what each chemistry means (the rule is chemOf).
const CHEM = [
  { id: 'any', label: 'Any' },
  { id: 'sulfide', label: 'Sulfides' },
  { id: 'oxide', label: 'Oxides' },
  { id: 'halide', label: 'Halides' },
  { id: 'other', label: 'Other' },
]
const chemOf = (f) => (f.S ? 'sulfide' : f.O ? 'oxide' : f.F || f.Cl || f.Br || f.I ? 'halide' : 'other')
// A move and a swap keep the kind of element: an anion-forming non-metal for another, anything else for anything else. H is a
// kind of its own (it is an anion in a hydride and a cation in a hydroxide), so it is never moved or swapped. The kinds are fixed
// lists, not a reading of each compound (the phosphorus of a phosphide counts as a cation), and the page says so.
const ANIONS = new Set(['N', 'O', 'F', 'S', 'Cl', 'Se', 'Br', 'Te', 'I'])
const ANION_TEXT = [...ANIONS].slice(0, -1).join(', ') + ' and ' + [...ANIONS].at(-1)
const kindOf = (e) => (ANIONS.has(e) ? 'anion' : e === 'H' ? 'H' : 'cation')
// Said beside every list of new formulas and in the trust list. It needs no number.
const NUDGE_CAVEAT = 'A nudge changes element fractions only. It does not keep the formula charge-balanced or check that the composition forms a stable phase, so read a formula as a direction in composition space: its lithium content may need adjusting before anyone could make it.'

const byKey = (a, b) => (a < b ? -1 : a > b ? 1 : 0)
const median = (sorted) => {
  const n = sorted.length
  return n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2
}
const keyOf = (f) => Object.keys(f).sort().map((e) => e + f[e].toFixed(4)).join('')
const atoms = (n) => (n >= 0.01 ? String(Number(n.toFixed(2))) : n.toPrecision(1))
const pct = (share) => `${Number((share * 100).toFixed(1))}%`
const differs = (d) => `${Number(((d / 2) * 100).toFixed(1))}%`
const thousands = (x) => Number(x.toPrecision(2)).toLocaleString('en-US')
const verb = (n, one, many) => (n === 1 ? one : many)
const minus = (text) => text.replace('-', '−')
const sameFractions = (a, b) => {
  const keys = Object.keys(a)
  return keys.length === Object.keys(b).length && keys.every((e) => Math.abs(a[e] - (b[e] ?? NaN)) <= SAME_F)
}
const reducedMotion = () => typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

// "Li6.4La3Zr1.4Ta0.6O12" -> [['Li', 6.4], ['La', 3], ...] and its number of atoms; null if any character is not part of a token.
function parseComposition(text) {
  const s = String(text)
  const parts = []
  let used = 0
  for (const m of s.matchAll(/([A-Z][a-z]?)(\d+(?:\.\d+)?)?/g)) {
    used += m[0].length
    parts.push([m[1], m[2] === undefined ? 1 : Number(m[2])])
  }
  if (!parts.length || used !== s.length) return null
  return { parts, natoms: parts.reduce((sum, [, n]) => sum + n, 0) }
}

function validModel(m) {
  return Array.isArray(m?.elements) && Array.isArray(m?.trees) && m.trees.length > 0 && Array.isArray(m?.records) &&
    m.records.length > 0 && Number.isFinite(m?.init) && Number.isFinite(m?.cv?.interval90_half_width) && Number.isFinite(m?.cv?.mae)
}

function vectorOf(elIndex, size, f) {
  const x = new Float64Array(size)
  for (const e of Object.keys(f)) {
    const i = elIndex.get(e)
    if (i !== undefined) x[i] = f[e]
  }
  return x
}

// Everything that depends only on the model file: measured compositions grouped by their fractions (the same composition can be
// written Li7La3Zr2O12 or Li56La24Zr16O96, and can be measured several times), and the browser-versus-exporter check.
function prepare(model) {
  const elements = model.elements
  const elIndex = new Map(elements.map((e, i) => [e, i]))
  const hw = model.cv.interval90_half_width
  const groups = new Map()
  for (const r of model.records) {
    if (!r?.fractions || !Number.isFinite(r.y)) continue
    const key = keyOf(r.fractions)
    if (!groups.has(key)) groups.set(key, { key, f: r.fractions, entries: Object.entries(r.fractions), rows: [] })
    groups.get(key).rows.push(r)
  }
  const uniq = [...groups.values()]
  for (const g of uniq) {
    let best = null
    for (const r of g.rows) {
      const p = parseComposition(r.composition)
      if (p && (!best || p.natoms < best.natoms)) best = p
    }
    g.natoms = best ? best.natoms : 100
    g.order = best ? best.parts.map(([e]) => e) : Object.keys(g.f).sort((a, b) => g.f[b] - g.f[a])
    g.parts = best ? best.parts.map(([e, n]) => [e, atoms(n)]) : g.order.map((e) => [e, atoms(g.f[e] * 100)])
    g.ys = g.rows.map((r) => r.y).sort((a, b) => a - b)
    g.medY = median(g.ys)
    const oofs = g.rows.map((r) => r.oof).filter(Number.isFinite).sort((a, b) => a - b)
    g.medOof = oofs.length ? median(oofs) : null
    g.oofVaries = oofs.length > 1 && oofs[oofs.length - 1] - oofs[0] > 1e-9
    // The export's folds keep one composition string (and one paper) together, not one set of fractions. The trees give one input
    // one value, so two measurements with the same fractions but different out-of-fold values were predicted by different fold
    // models, and each of those models had the other measurement in its training rows. It happens when the data writes one
    // composition two ways (Li3N and Li6N2). Such an out-of-fold value is not a clean test, and the page says so beside it.
    // Only a composition written more than one way counts, so every sentence about "spellings" is true of the group it is shown for.
    g.spellings = [...new Set(g.rows.map((r) => String(r.composition)))]
    const split = new Set()
    for (let i = 0; g.spellings.length > 1 && i < g.rows.length; i++) {
      for (let j = i + 1; j < g.rows.length; j++) {
        const a = g.rows[i]
        const b = g.rows[j]
        if (Number.isFinite(a.oof) && Number.isFinite(b.oof) && Math.abs(a.oof - b.oof) > 1e-9 && sameFractions(a.fractions, b.fractions)) split.add(a).add(b)
      }
    }
    g.split = split
    g.leaky = split.size > 0
    g.ids = g.rows.map((r) => r.id)
    g.chem = chemOf(g.f)
    g.x = vectorOf(elIndex, elements.length, g.f)
    g.idx = nonzero(g.x)
  }
  const allY = model.records.map((r) => r.y).filter(Number.isFinite).sort((a, b) => a - b)
  // The exporter sets the half-width to the ceil((n + 1) * 0.9)-th smallest out-of-fold miss of these same rows (_gbm_oof in
  // gpu/export_site.py). The page checks that against the records (which are rounded to 3 decimals) before saying so.
  const misses = model.records.filter((r) => Number.isFinite(r.y) && Number.isFinite(r.oof)).map((r) => Math.abs(r.oof - r.y)).sort((a, b) => a - b)
  const nScored = misses.length
  const nInside = misses.filter((v) => v <= hw).length
  const hwFromMisses = nScored > 0 && Math.abs(misses[Math.min(nScored - 1, Math.ceil((nScored + 1) * 0.9) - 1)] - hw) < 2e-3
  // The trust list says how many compositions the data writes more than one way, and for how many of those the spellings were split.
  const leaky = uniq.filter((g) => g.leaky).sort((a, b) => a.spellings.join('').length - b.spellings.join('').length || byKey(a.key, b.key))
  const respelled = uniq.filter((g) => g.spellings.length > 1).length
  const chemCounts = { any: uniq.length }
  for (const g of uniq) chemCounts[g.chem] = (chemCounts[g.chem] ?? 0) + 1
  const checks = Array.isArray(model.checks) ? model.checks : []
  let checkWorst = 0
  for (const c of checks) checkWorst = Math.max(checkWorst, Math.abs(predictTrees(model, vectorOf(elIndex, elements.length, c.fractions ?? {})) - c.pred))
  return {
    elements, elIndex, elSet: new Set(elements), uniq, byKey: new Map(uniq.map((g) => [g.key, g])), allY, nScored, nInside, hwFromMisses,
    leaky, respelled, leakyRows: leaky.reduce((n, g) => n + g.split.size, 0),
    minY: allY[0], maxY: allY[allY.length - 1], chemCounts, nChecks: checks.length, checkWorst,
  }
}

// Non-zero element indices of a dense fraction vector.
const nonzero = (x) => {
  const idx = []
  for (let i = 0; i < x.length; i++) if (x[i] > 0) idx.push(i)
  return Int32Array.from(idx)
}

// The k measured compositions nearest to fractions f by L1 distance (sum of |difference| over elements). Works on dense vectors
// over the model's elements, because it runs for every new composition.
function nearest(prep, f, k, skipKey, x = vectorOf(prep.elIndex, prep.elements.length, f)) {
  const idx = nonzero(x)
  const best = []
  for (const u of prep.uniq) {
    if (u.key === skipKey) continue
    const ux = u.x
    let d = 0
    for (let a = 0; a < idx.length; a++) d += Math.abs(x[idx[a]] - ux[idx[a]])
    const uidx = u.idx
    for (let b = 0; b < uidx.length; b++) if (x[uidx[b]] === 0) d += ux[uidx[b]]
    if (best.length === k && d >= best[k - 1].d) continue
    let at = best.length
    while (at > 0 && best[at - 1].d > d) at--
    best.splice(at, 0, { d, u })
    if (best.length > k) best.pop()
  }
  return best
}

// Elements that may replace a minor element of a starting composition: elements of the same kind (kindOf) that appear in measured
// compositions containing all of the starting composition's other elements, the SUBS_MAX most frequent first. If no such
// composition brings one and the starting composition has four or more elements, compositions containing all but one of its other
// elements are used instead. The page states this rule; keep the two in step.
function substitutes(prep, seed, minor, allowed) {
  const rest = seed.order.filter((e) => e !== minor)
  const kind = kindOf(minor)
  const count = (need) => {
    const c = new Map()
    for (const u of prep.uniq) {
      let shared = 0
      for (const e of rest) if (u.f[e] !== undefined) shared++
      if (shared < need) continue
      for (const [e] of u.entries) if (seed.f[e] === undefined && kindOf(e) === kind && allowed(e)) c.set(e, (c.get(e) ?? 0) + 1)
    }
    return [...c].sort((a, b) => b[1] - a[1] || byKey(a[0], b[0])).slice(0, SUBS_MAX).map(([e]) => e)
  }
  const strict = count(rest.length)
  return strict.length || rest.length <= 2 ? strict : count(rest.length - 1)
}

// Every nudge of each starting composition, in a fixed order: each move of 2% to 5% of the atoms (steps of 0.5%, never more than
// half of the donor) from one of its elements to another of the same kind, and each swap of a minor element for one of its
// substitutes. A move between the two kinds (iodine to phosphorus, say) is left out: it changes the ratio of anions to cations,
// which a substitution does not. What is left is still not charge-balanced (NUDGE_CAVEAT).
function nudges(s, subs) {
  const out = []
  for (const from of s.order) {
    for (const to of s.order) {
      if (to === from || kindOf(to) !== kindOf(from)) continue
      for (let k = MOVE_MIN; k <= MOVE_MAX && s.f[from] >= 2 * k * STEP; k++) out.push({ kind: 'move', from, to, share: k * STEP })
    }
  }
  for (const [from, list] of subs) for (const to of list) out.push({ kind: 'swap', from, to })
  return out
}

function generate(model, prep, ask) {
  const avoid = new Set(ask.avoid)
  const allowed = (e) => prep.elSet.has(e) && !avoid.has(e)
  const fits = (f) => Object.keys(f).every(allowed) && (ask.chem === 'any' || chemOf(f) === ask.chem)
  const seeds = prep.uniq.filter((u) => fits(u.f)).sort((a, b) => b.medY - a.medY || byKey(a.key, b.key)).slice(0, ask.seedCount)
  const measuredKeys = new Set(prep.uniq.map((u) => u.key))
  const seen = new Set()
  const stats = { moves: 0, swaps: 0, outside: 0, repeats: 0, measured: 0 }
  const made = []
  for (const s of seeds) {
    const subs = new Map()
    for (const e of s.order) {
      if (e === 'Li' || s.f[e] >= MINOR) continue
      const list = substitutes(prep, s, e, allowed)
      if (list.length) subs.set(e, list)
    }
    for (const how of nudges(s, subs)) {
      const f = { ...s.f }
      let order = s.order
      if (how.kind === 'swap') {
        f[how.to] = f[how.from]
        delete f[how.from]
        order = s.order.map((e) => (e === how.from ? how.to : e))
      } else {
        f[how.from] -= how.share
        f[how.to] += how.share
      }
      // Renormalise: fractions stay non-negative and sum to 1.
      const total = Object.values(f).reduce((a, v) => a + Math.max(0, v), 0)
      for (const e of Object.keys(f)) {
        const v = Math.max(0, f[e]) / total
        if (v > 0) f[e] = v
        else delete f[e]
      }
      if (!fits(f)) { stats.outside++; continue }
      const key = keyOf(f)
      if (measuredKeys.has(key)) { stats.measured++; continue }
      if (seen.has(key)) { stats.repeats++; continue }
      seen.add(key)
      made.push({ key, f, order, seed: s, how })
    }
  }
  const cands = []
  const perSeed = new Map()
  for (const c of made) {
    const x = vectorOf(prep.elIndex, prep.elements.length, c.f)
    const near = nearest(prep, c.f, 5, null, x)
    if (near[0] && near[0].d < SAME_L1) { stats.measured++; continue }
    c.near = near
    c.pred = predictTrees(model, x)
    c.parts = c.order.filter((e) => c.f[e] > 0).map((e) => [e, atoms(c.f[e] * c.seed.natoms)])
    stats[c.how.kind === 'swap' ? 'swaps' : 'moves']++
    const ps = perSeed.get(c.seed.key) ?? { made: 0, best: null }
    ps.made++
    if (!ps.best || c.pred > ps.best.pred) ps.best = c
    perSeed.set(c.seed.key, ps)
    cands.push(c)
  }
  // The same order as the cautious lower estimate (pred - half-width). The trees give many compositions the same value; a tie goes
  // to the composition closer to a measured one.
  cands.sort((a, b) => b.pred - a.pred || a.near[0].d - b.near[0].d || byKey(a.key, b.key))
  return { ask, seeds, cands, byKey: new Map(cands.map((c) => [c.key, c])), perSeed, stats }
}

// What the table shows for a target: ranked by the cautious lower estimate, optionally only those whose expected value reaches it,
// and at most PER_SEED_SHOWN rows from one starting composition.
function rank(gen, t, hw, twoPerSeed, onlyReach) {
  let nExpected = 0
  let nCautious = 0
  for (const c of gen.cands) {
    if (c.pred >= t) nExpected++
    if (c.pred - hw >= t) nCautious++
  }
  const rows = []
  const per = new Map()
  for (const c of gen.cands) {
    if (rows.length >= SHOWN || (onlyReach && c.pred < t)) break
    const n = per.get(c.seed.key) ?? 0
    if (twoPerSeed && n >= PER_SEED_SHOWN) continue
    per.set(c.seed.key, n + 1)
    rows.push(c)
  }
  return { rows, nExpected, nCautious }
}

// How the model's own out-of-fold predictions did at this target. Each comes from trees that never trained on that measurement,
// nor on any other sharing its composition string or its paper (the export's fold rule; see g.leaky in prepare for its gap).
function honesty(model, t, hw) {
  const h = { pos: 0, predPos: 0, tp: 0, cautiousPos: 0, cautiousTp: 0 }
  for (const r of model.records) {
    if (!Number.isFinite(r.y) || !Number.isFinite(r.oof)) continue
    const a = r.y >= t
    const b = r.oof >= t
    if (a) h.pos++
    if (b) h.predPos++
    if (a && b) h.tp++
    if (r.oof - hw >= t) {
      h.cautiousPos++
      if (a) h.cautiousTp++
    }
  }
  return h
}

// Whether the three nearest measured compositions back a target: at least two of them close (CLOSE_L1) and at or above it. When
// they do not, the flag says how many reached it, or that the ones reaching it are not close.
function backing(c, t) {
  const near = c.near.slice(0, 3)
  const reach = near.filter((n) => n.u.medY >= t).length
  const ok = near.filter((n) => n.d <= CLOSE_L1 && n.u.medY >= t).length >= 2
  const text = ok ? 'Measured neighbours reach it'
    : reach >= 2 ? 'Neighbours reach it, but are not close'
      : `${reach} of ${near.length} neighbours ${verb(reach, 'reaches', 'reach')} it`
  return { ok, text }
}

// 10^log as a two-figure mantissa and an exponent: -2.68 -> { m: 2.1, e: -3 }.
function mantissa(log) {
  const e = Math.floor(log)
  const m = Number((10 ** (log - e)).toPrecision(2))
  return m >= 10 ? { m: 1, e: e + 1 } : { m, e }
}
// Conductivity 10^log S/cm as "2.1 × 10⁻³ S/cm" (and "2.1 S/cm" when the power is zero). The raised power is for the eye only: the
// hidden caret makes a copy read "10^−3" instead of "10−3", and the hidden words are what a screen reader says.
function Sigma({ log, unit = true }) {
  if (!Number.isFinite(log)) return <span>–</span>
  const { m, e } = mantissa(log)
  const tail = unit ? ' S/cm' : ''
  if (e === 0) return <span className="ed-sigma">{m}{tail}</span>
  return (
    <span className="ed-sigma">
      <span aria-hidden="true">{m === 1 ? '' : `${m} × `}10<span className="ed-hide">^</span><sup>{e < 0 ? `−${-e}` : e}</sup>{tail}</span>
      <span className="ed-hide ed-said">{m === 1 ? '' : `${m} times `}10 to the {e < 0 ? `minus ${-e}` : e}{tail}</span>
    </span>
  )
}
function sigmaText(log) {
  const { m, e } = mantissa(log)
  return e === 0 ? `${m} S/cm` : `${m}e${e} S/cm`
}

const partsText = (parts) => parts.map(([e, n]) => (n === '1' ? e : e + n)).join('')
function Chem({ parts }) {
  return (
    <span className="ed-formula" aria-label={partsText(parts)}>
      {parts.map(([e, n]) => <span key={e}>{e}{n !== '1' && <sub>{n}</sub>}</span>)}
    </span>
  )
}

// "the 25 most conductive measured compositions", or all of them when fewer match than were asked for.
const seedsText = (n, asked) => (n >= asked ? `the ${n} most conductive measured compositions` : n === 1 ? 'the one measured composition' : `all ${n} measured compositions`)

const howText = (how) => (how.kind === 'swap' ? `${how.from} swapped for ${how.to}` : `${pct(how.share)} of the atoms moved from ${how.from} to ${how.to}`)

function Measured({ u }) {
  return (
    <>
      <Sigma log={u.medY} unit={false} />
      {u.ys.length > 1 && <span className="label">median of {u.ys.length}</span>}
    </>
  )
}

// The model's out-of-fold value for a measured composition; a median when its measurements got different values, and marked when
// the data's two spellings of it fell in different folds (g.leaky in prepare).
function Oof({ u }) {
  if (u.medOof === null) return '–'
  return (
    <>
      <Sigma log={u.medOof} unit={false} />
      {u.oofVaries && <span className="label">{u.leaky ? 'median; not a clean test' : 'median'}</span>}
    </>
  )
}

// A composition as the data writes it ("Li6.0N2.0" shown as Li6N2, in the source's element order).
function Spelling({ text }) {
  const p = parseComposition(text)
  return p ? <Chem parts={p.parts.map(([e, n]) => [e, atoms(n)])} /> : <span className="ed-formula">{text}</span>
}

// A table that scrolls sideways inside its own box when it is wider than the page, with a line saying so only while it is.
// "plain" drops the border, for the small tables inside the detail panel.
function ScrollTable({ hint = 'Scroll the table sideways for its other columns.', plain = false, children }) {
  const ref = useRef(null)
  const [over, setOver] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return undefined
    const ro = new ResizeObserver(() => setOver(el.scrollWidth > el.clientWidth + 1))
    ro.observe(el)
    if (el.firstElementChild) ro.observe(el.firstElementChild)
    return () => ro.disconnect()
  }, [])
  return (
    <>
      {over && <p className="fine ed-scrollhint">{hint}</p>}
      <div className={plain ? 'sim-scroll' : 'tablewrap sim-scroll'} ref={ref}>{children}</div>
    </>
  )
}

// A log-scale strip: the 90% range of a prediction, its expected value, every measured value in the data, the nearest measured
// compositions and the target. Nothing on it moves.
function RangeStrip({ prep, center, hw, target, measured, near }) {
  const W = 360
  const hasMeasured = Number.isFinite(measured)
  const lo = Math.floor(Math.min(prep.minY, center - hw, target, hasMeasured ? measured : Infinity))
  const hi = Math.ceil(Math.max(prep.maxY, center + hw, target, hasMeasured ? measured : -Infinity))
  const x = (v) => 12 + ((v - lo) / (hi - lo)) * (W - 24)
  const step = hi - lo > 12 ? 4 : 2
  const ticks = []
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v)
  const rug = useMemo(() => prep.allY.map((y) => `M${(12 + ((y - lo) / (hi - lo)) * (W - 24)).toFixed(1)} 56V68`).join(''), [prep, lo, hi])
  return (
    <svg className="ed-strip" viewBox={`0 0 ${W} 116`} role="img"
      aria-label={`${hasMeasured ? `Measured ${sigmaText(measured)}; model` : 'Expected'} ${sigmaText(center)}, 90% range ${sigmaText(center - hw)} to ${sigmaText(center + hw)}; target ${sigmaText(target)}`}>
      <rect className="ed-band" x={x(center - hw)} y={20} width={x(center + hw) - x(center - hw)} height={16} rx={2} />
      <line className="ed-center" x1={x(center)} x2={x(center)} y1={16} y2={40} />
      {hasMeasured && <line className="ed-measured" x1={x(measured)} x2={x(measured)} y1={12} y2={44} />}
      <path className="ed-rug" d={rug} />
      {near.map((n) => <circle key={n.u.key} className="ed-near" cx={x(n.u.medY)} cy={62} r={3.5} />)}
      <line className="ed-target" x1={x(target)} x2={x(target)} y1={8} y2={74} />
      <text x={Math.min(x(target) + 4, W - 34)} y={11}>target</text>
      <line className="ed-axis" x1={12} x2={W - 12} y1={78} y2={78} />
      {ticks.map((v) => (
        <g key={v}>
          <line className="ed-axis" x1={x(v)} x2={x(v)} y1={78} y2={82} />
          <text x={x(v)} y={96} textAnchor="middle">10<tspan className="ed-exp" dy={-5}>{v < 0 ? `−${-v}` : v}</tspan></text>
        </g>
      ))}
      <text x={W / 2} y={112} textAnchor="middle">conductivity, S/cm (log scale)</text>
    </svg>
  )
}

// On a phone the "differs in" column is folded under the formula (ElectrolyteDesigner.css), so the three that matter fit.
function NearTable({ near }) {
  return (
    <ScrollTable plain>
      <table className="sim-table">
        <thead>
          <tr><th>Measured composition</th><th className="num ed-wide">Differs in (share of atoms)</th><th className="num">Measured (S/cm)</th><th className="num">Model, out of fold (S/cm)</th></tr>
        </thead>
        <tbody>
          {near.map((n) => (
            <tr key={n.u.key}>
              <td className="ed-name"><Chem parts={n.u.parts} /> <span className="label">{n.u.chem}<span className="ed-phone"> · {differs(n.d)} of the atoms differ</span></span></td>
              <td className="num ed-wide">{differs(n.d)}</td>
              <td className="num"><Measured u={n.u} /></td>
              <td className="num"><Oof u={n.u} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollTable>
  )
}

function Detail({ item, prep, hw, target }) {
  const legend = (first, other) => (
    <p className="fine">{first} Grey ticks: all {prep.allY.length} measured values in the data. Circles: the three nearest {other ? 'other ' : ''}measured compositions. Dashed: your target.</p>
  )
  if (item.kind === 'measured') {
    const u = item.u
    return (
      <div className="ed-detail">
        <span className="label">Measured composition{u.ys.length > 1 ? `, ${u.ys.length} measurements` : ''}</span>
        <h2><Chem parts={u.parts} /></h2>
        <p className="ed-how"><span className="flag ok">measured</span> {u.chem} · record {u.ids.join(', ')}</p>
        <div className="stats">
          <div>
            <span className="label">Measured{u.ys.length > 1 ? ' (median)' : ''}</span>
            <span className="big"><Sigma log={u.medY} /></span>
            {u.ys.length > 1 && <span className="label">range <Sigma log={u.ys[0]} /> to <Sigma log={u.ys[u.ys.length - 1]} /></span>}
          </div>
          <div>
            <span className="label">Model, out of fold{u.oofVaries ? ' (median)' : ''}</span>
            <span className="big">{u.medOof === null ? '–' : <Sigma log={u.medOof} />}</span>
            {u.medOof !== null && (
              <span className="label">
                {Math.abs(u.medOof - u.medY) < 0.005 ? 'on the measured value' : `${fmt(Math.abs(u.medOof - u.medY), 3)} log units too ${u.medOof < u.medY ? 'low' : 'high'}`}
              </span>
            )}
          </div>
        </div>
        {u.medOof !== null && (
          <p className="ed-how">
            {u.leaky ? (
              <>
                <span className="flag low">not a clean test</span> The data writes this composition {u.spellings.length} ways
                ({u.spellings.map((t, i) => <span key={t}>{i > 0 && ', '}<Spelling text={t} /></span>)}) and their measurements fell in
                different folds, so the trees behind each out-of-fold value had trained on another spelling of it.
              </>
            ) : 'Out of fold: predicted by trees that never trained on this measurement, nor on any other sharing its composition string or its paper.'}
          </p>
        )}
        {u.medOof !== null && (
          <>
            <RangeStrip prep={prep} center={u.medOof} hw={hw} target={target} measured={u.medY} near={item.near.slice(0, 3)} />
            {legend('Band and accent line: the out-of-fold prediction with its 90% range. Taller plain line: the measured value.', true)}
          </>
        )}
        {u.rows.length > 1 && (
          <>
            <h3>Its {u.rows.length} measurements</h3>
            <ScrollTable plain>
              <table className="sim-table">
                <thead>
                  <tr><th className="ed-wide">Record</th><th>Written in the data as</th><th className="num">Measured (S/cm)</th><th className="num">Model, out of fold (S/cm)</th></tr>
                </thead>
                <tbody>
                  {u.rows.map((r, i) => (
                    <tr key={`${r.id} ${i}`}>
                      <td className="ed-wide">{r.id}</td>
                      <td className="ed-name"><Spelling text={String(r.composition)} /><span className="label ed-phone">record {r.id}</span></td>
                      <td className="num"><Sigma log={r.y} unit={false} /></td>
                      <td className="num">
                        {Number.isFinite(r.oof) ? <Sigma log={r.oof} unit={false} /> : '–'}
                        {u.split.has(r) && <span className="label">not a clean test</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ScrollTable>
          </>
        )}
        <h3>Nearest other measured compositions</h3>
        <NearTable near={item.near} />
      </div>
    )
  }
  const c = item.c
  const s = c.seed
  const els = [...s.order]
  if (c.how.kind === 'swap') els.splice(els.indexOf(c.how.from) + 1, 0, c.how.to)
  return (
    <div className="ed-detail">
      <span className="label">New composition (not in the data)</span>
      <h2><Chem parts={c.parts} /></h2>
      <p className="ed-how">
        <span className="flag low">untested prediction</span> made from <Chem parts={s.parts} /> (measured <Sigma log={s.medY} />{s.ys.length > 1 ? ', median' : ''}): {howText(c.how)}.
      </p>
      <div className="stats">
        <div>
          <span className="label">Expected (prediction)</span>
          <span className="big"><Sigma log={c.pred} /></span>
          <span className="label">log₁₀ σ = {minus(fmt(c.pred, 3))}</span>
        </div>
        <div>
          <span className="label">Cautious lower estimate</span>
          <span className="big"><Sigma log={c.pred - hw} /></span>
          <span className="label">90% range up to <Sigma log={c.pred + hw} /></span>
        </div>
      </div>
      <RangeStrip prep={prep} center={c.pred} hw={hw} target={target} near={c.near.slice(0, 3)} />
      {legend('Band: the 90% range of the prediction. Accent line: the expected value.', false)}
      <h3>Atoms per formula (same total as the starting formula)</h3>
      <div className="sim-scroll">
        <table className="sim-table">
          <thead><tr><th>Element</th><th className="num">Starting (measured)</th><th className="num">This one</th><th className="num">Change, % of all atoms</th></tr></thead>
          <tbody>
            {els.map((e) => {
              const a = s.f[e] ?? 0
              const b = c.f[e] ?? 0
              const d = (b - a) * 100
              const moved = Math.abs(d) >= 0.05
              return (
                <tr key={e}>
                  <td className={moved ? 'strong' : ''}>{e}</td>
                  <td className="num">{a ? atoms(a * s.natoms) : '–'}</td>
                  <td className="num">{b ? atoms(b * s.natoms) : '–'}</td>
                  <td className={moved ? 'num strong' : 'num'}>{moved ? `${d > 0 ? '+' : '−'}${Number(Math.abs(d).toFixed(1))}` : '0'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="fine ed-after">
        The nudge changes element fractions only: this formula is not checked for charge balance or for forming a stable phase, so its
        lithium content may need adjusting before anyone could make it.
      </p>
      <h3>Nearest measured compositions</h3>
      <NearTable near={c.near} />
    </div>
  )
}

function parseTarget(text, prep) {
  const t = String(text).trim().replace(/\s*s\s*\/\s*cm$/i, '').replace(/\s+/g, '').replace(',', '.').replace(/[−–]/g, '-')
  let v = NaN
  if (/^(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(t)) v = Number(t)
  else {
    const m = t.match(/^(?:(\d+\.?\d*|\.\d+)[x×*])?10\^\(?([-+]?\d+(?:\.\d+)?)\)?$/)
    if (m) v = (m[1] ? Number(m[1]) : 1) * 10 ** Number(m[2])
  }
  if (!(v > 0) || !Number.isFinite(v)) return { error: 'Type the conductivity in S/cm as a number, for example 1e-3 or 0.001.' }
  const lo = Math.floor(prep.minY)
  const hi = Math.ceil(prep.maxY) + 1
  const log = Math.log10(v)
  if (log < lo || log > hi) {
    const p = (e) => (e === 0 ? '1' : `1e${e}`)
    return { error: `Choose a conductivity between ${p(lo)} and ${p(hi)} S/cm. The measured values behind the model run from ${sigmaText(prep.minY)} to ${sigmaText(prep.maxY)}.` }
  }
  return { log }
}

function parseAvoid(text, prep) {
  const list = [...new Set(String(text).split(/[\s,;]+/).filter(Boolean).map((s) => s[0].toUpperCase() + s.slice(1).toLowerCase()))]
  const unknown = list.filter((e) => !prep.elSet.has(e))
  if (unknown.length) {
    return { error: `${unknown.join(', ')} ${unknown.length > 1 ? 'are' : 'is'} not among the ${prep.elements.length} elements the model knows (${prep.elements.join(', ')}).` }
  }
  if (list.includes('Li')) return { error: 'Every composition here contains lithium, so Li cannot be left out.' }
  return { list }
}

export default function ElectrolyteDesigner({ tasks = [] }) {
  const [model, setModel] = useState(undefined)
  const [targetText, setTargetText] = useState('1e-3')
  const [chem, setChem] = useState('any')
  const [avoidText, setAvoidText] = useState('')
  const [seedCount, setSeedCount] = useState(25)
  const [ask, setAsk] = useState(null)
  const [targetLog, setTargetLog] = useState(null)
  const [twoPerSeed, setTwoPerSeed] = useState(true)
  const [onlyReach, setOnlyReach] = useState(true)
  const [sel, setSel] = useState(null)
  const [allSeeds, setAllSeeds] = useState(false)
  const [error, setError] = useState('')
  const resultRef = useRef(null)

  // No AbortController: loadPipeline keeps one shared request per file, and under StrictMode (npm run dev) React runs this effect
  // twice, so aborting in the first cleanup would leave the second run waiting on a cancelled request for good. If another caller
  // cancelled the shared request, ask once more; a second failure ends in the plain "not exported" note.
  useEffect(() => {
    let live = true
    const set = (m) => { if (live) setModel(m ?? null) }
    loadPipeline('electrolyte_model.json').then(set, () => loadPipeline('electrolyte_model.json').then(set, () => set(null)))
    return () => { live = false }
  }, [])

  const prep = useMemo(() => {
    if (!model) return null
    try {
      return validModel(model) ? prepare(model) : { broken: true }
    } catch {
      return { broken: true }
    }
  }, [model])
  const ready = !!prep && !prep.broken
  const hw = ready ? model.cv.interval90_half_width : null
  const gen = useMemo(() => (ready && ask ? generate(model, prep, ask) : null), [ready, model, prep, ask])
  const view = useMemo(() => (gen && targetLog !== null ? rank(gen, targetLog, hw, twoPerSeed, onlyReach) : null), [gen, targetLog, hw, twoPerSeed, onlyReach])
  const trust = useMemo(() => (ready && targetLog !== null ? honesty(model, targetLog, hw) : null), [ready, model, targetLog, hw])
  const item = useMemo(() => {
    if (!gen || !view) return null
    const measured = (u) => ({ kind: 'measured', u, near: nearest(prep, u.f, 5, u.key) })
    if (sel?.kind === 'measured' && prep.byKey.has(sel.key)) return measured(prep.byKey.get(sel.key))
    if (sel?.kind === 'new' && gen.byKey.has(sel.key)) return { kind: 'new', c: gen.byKey.get(sel.key) }
    if (view.rows[0]) return { kind: 'new', c: view.rows[0] }
    return gen.seeds[0] ? measured(gen.seeds[0]) : null
  }, [gen, view, sel, prep])

  if (model === undefined) return <section className="card fam-section"><p className="note">Loading the electrolyte model…</p></section>
  if (!ready) {
    return (
      <section className="card fam-section">
        <h2>Design a solid electrolyte</h2>
        <p className="note">
          {model === null ? 'The electrolyte model has not been exported to the site yet.' : 'The electrolyte model file is incomplete, so the designer cannot run.'}
          {' '}On the training laptop: <code>python -m gpu.export_site --browser-models-only</code>, then commit <code>public/pipeline/electrolyte_model.json</code>.
        </p>
      </section>
    )
  }

  const run = (text = targetText) => {
    setTargetText(text)
    const t = parseTarget(text, prep)
    if (t.error) return setError(t.error)
    const av = parseAvoid(avoidText, prep)
    if (av.error) return setError(av.error)
    setError('')
    setTargetLog(t.log)
    const next = { chem, avoid: [...av.list].sort(), seedCount }
    setAsk((prev) => (prev && JSON.stringify(prev) === JSON.stringify(next) ? prev : next))
    setSel(null)
  }
  // The detail panel is at the top of the result block (on a phone, above the picks), so scroll to its start.
  const show = (next) => {
    setSel(next)
    resultRef.current?.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'start' })
  }

  const cv = model.cv
  const spread = view && view.rows.length > 1 ? view.rows[0].pred - view.rows[view.rows.length - 1].pred : null
  const bestNew = view?.rows[0] ?? null
  // With "only those that reach the target" unticked, the first row can fall short of it: the pick must say so.
  const bestShort = !!bestNew && bestNew.pred < targetLog
  const bestMeasured = gen?.seeds[0] ?? null
  const measuredReach = targetLog === null ? 0 : prep.uniq.filter((u) => u.medY >= targetLog).length
  const seedsShown = gen ? (allSeeds ? gen.seeds : gen.seeds.slice(0, SEEDS_FOLDED)) : []
  const isSel = (kind, key) => !!item && item.kind === kind && (kind === 'new' ? item.c.key : item.u.key) === key
  const bigFactor = thousands(10 ** hw)
  const coverPct = prep.nScored ? Math.round((prep.nInside / prep.nScored) * 100) : null
  const leakyShown = seedsShown.some((u) => u.leaky)
  // The results keep the request they were made with until Propose is pressed again; say so when the form no longer matches it.
  const formChanged = !!gen && targetLog !== null && (() => {
    const t = parseTarget(targetText, prep)
    const av = parseAvoid(avoidText, prep)
    if (t.error || av.error) return true
    return Math.abs(t.log - targetLog) > 1e-9 || chem !== gen.ask.chem || seedCount !== gen.ask.seedCount || [...av.list].sort().join(' ') !== gen.ask.avoid.join(' ')
  })()

  return (
    <>
      <section className="card fam-section ed">
        <h2>Design a solid electrolyte</h2>
        <form onSubmit={(e) => { e.preventDefault(); run() }}>
          <div className="ed-form">
            <div>
              <label htmlFor="ed-target">Conductivity needed (S/cm, room temperature)</label>
              <input id="ed-target" value={targetText} onChange={(e) => setTargetText(e.target.value)} autoComplete="off" autoCapitalize="none" spellCheck={false} placeholder="e.g. 1e-3" />
            </div>
            <div>
              <label htmlFor="ed-chem">Chemistry</label>
              <select id="ed-chem" value={chem} onChange={(e) => setChem(e.target.value)}>
                {CHEM.map((c) => <option key={c.id} value={c.id}>{c.label} · {prep.chemCounts[c.id] ?? 0} measured</option>)}
              </select>
            </div>
            <div>
              <label htmlFor="ed-avoid">Leave out elements (optional)</label>
              <input id="ed-avoid" value={avoidText} onChange={(e) => setAvoidText(e.target.value)} autoComplete="off" spellCheck={false} placeholder="e.g. Ge, Pb" />
            </div>
            <div>
              <label htmlFor="ed-seeds">Measured compositions to start from</label>
              <select id="ed-seeds" value={seedCount} onChange={(e) => setSeedCount(Number(e.target.value))}>
                {SEED_COUNTS.map((n) => <option key={n} value={n}>The {n} most conductive</option>)}
              </select>
            </div>
          </div>
          <div className="ed-actions">
            <button type="submit">Propose compositions</button>
          </div>
          <div className="examples">
            <span>Try:</span>
            {PRESETS.map((p) => (
              <button type="button" key={p} className="chip" aria-pressed={!!ask && targetLog === p} onClick={() => run(`1e${p}`)}>
                <Sigma log={p} />
              </button>
            ))}
          </div>
          <p className="note small">
            Compositions are atom fractions over the {prep.elements.length} elements the model knows; new ones use only those. Sulfides
            contain S; oxides contain O and no S; halides contain F, Cl, Br or I and no O or S; the count beside each is its measured
            compositions. Change a choice, then press Propose again. The three targets are round numbers, not requirements from a
            standard.
          </p>
        </form>
        {error && <p className="error" role="alert">{error}{gen && view ? ' The results below are still for your previous request.' : ''}</p>}
      </section>

      {gen && view && gen.seeds.length === 0 && (
        <section className="card fam-section"><p className="note">No measured composition matches these choices. Choose another chemistry or leave out fewer elements.</p></section>
      )}

      {gen && view && gen.seeds.length > 0 && (
        <>
          <section className="result ed-result" ref={resultRef}>
            <div className="stagewrap">{item && <Detail item={item} prep={prep} hw={hw} target={targetLog} />}</div>
            <div className="readout">
              <h2>Target: <Sigma log={targetLog} /> at room temperature</h2>
              <p className="ed-ask">
                Chemistry: {(CHEM.find((c) => c.id === gen.ask.chem)?.label ?? gen.ask.chem).toLowerCase()}. Left out: {gen.ask.avoid.join(', ') || 'none'}.
                {' '}Starting from {seedsText(gen.seeds.length, gen.ask.seedCount)} matching these choices.
              </p>
              {formChanged && <p className="flag low ed-stale">The form above has changed since. Press Propose compositions to update these results.</p>}
              <div className="picks">
                <button type="button" className="pick" disabled={!bestNew} aria-pressed={!!bestNew && isSel('new', bestNew.key)}
                  onClick={() => bestNew && show({ kind: 'new', key: bestNew.key })}>
                  <span className="label">{bestShort ? 'Closest new composition (untested prediction)' : 'Best new composition (untested prediction)'}</span>
                  {bestNew ? (
                    <>
                      <span className="mixname"><Chem parts={bestNew.parts} /></span>
                      <span className="big"><Sigma log={bestNew.pred} /> <small>{bestShort ? 'expected, below your target' : 'expected'}</small></span>
                      <span className="label">cautious lower estimate <Sigma log={bestNew.pred - hw} />; {howText(bestNew.how)} in <Chem parts={bestNew.seed.parts} /></span>
                    </>
                  ) : <span className="label">{gen.cands.length === 0 ? 'no nudge gives a new composition with these choices' : 'none has an expected conductivity at the target with these choices'}</span>}
                </button>
                <button type="button" className="pick" disabled={!bestMeasured} aria-pressed={!!bestMeasured && isSel('measured', bestMeasured.key)}
                  onClick={() => bestMeasured && show({ kind: 'measured', key: bestMeasured.key })}>
                  <span className="label">Most conductive measured composition matching your choices</span>
                  {bestMeasured && (
                    <>
                      <span className="mixname"><Chem parts={bestMeasured.parts} /></span>
                      <span className="big"><Sigma log={bestMeasured.medY} /> <small>measured{bestMeasured.ys.length > 1 ? `, median of ${bestMeasured.ys.length}` : ''}</small></span>
                      <span className="label">
                        the model&apos;s out-of-fold prediction for it: {bestMeasured.medOof === null ? '–' : <Sigma log={bestMeasured.medOof} />}
                        {bestMeasured.oofVaries && (bestMeasured.leaky ? ' (median; not a clean test)' : ' (median)')}
                      </span>
                    </>
                  )}
                </button>
              </div>
              <p className="note">
                {measuredReach} of the {prep.uniq.length} measured compositions in the data {verb(measuredReach, 'reaches', 'reach')} the target (median of
                their measurements). {gen.cands.length === 0 ? 'No new composition was made with these choices.' : (
                  <>
                    Of the {gen.cands.length.toLocaleString()} new {verb(gen.cands.length, 'composition', 'compositions')}, {view.nExpected.toLocaleString()} {verb(view.nExpected, 'has', 'have')} an
                    expected conductivity at or above it and {view.nCautious.toLocaleString()} {verb(view.nCautious, 'has', 'have')} a cautious lower estimate that reaches it.
                  </>
                )}
              </p>
              <p className="warn">
                The 90% range of a prediction is ±{fmt(hw, 3)} log units: a factor of about {bigFactor} either way. The model can point in a
                direction, towards the kinds of composition that measured well, but it cannot rank close candidates. Read the list as
                directions worth considering, not as results: a new formula is not checked for charge balance or for forming a stable
                phase.
              </p>
            </div>
          </section>

          <section className="card fam-section">
            <h2>New compositions, ranked by the cautious lower estimate</h2>
            <p className="note">
              {gen.cands.length.toLocaleString()} new {verb(gen.cands.length, 'composition was', 'compositions were')} made in your browser from {seedsText(gen.seeds.length, gen.ask.seedCount)}
              {' '}matching your choices, by trying every nudge: {gen.stats.moves.toLocaleString()} by a move and {gen.stats.swaps.toLocaleString()} by a swap.
              A move shifts {pct(MOVE_MIN * STEP)} to {pct(MOVE_MAX * STEP)} of the atoms (in steps of {pct(STEP)}, never more than half of an element)
              from one element to another of the same kind. A swap replaces a minor element (under {pct(MINOR)} of the atoms, not Li) with
              another of the same kind found in measured compositions that contain all of the starting composition&apos;s other elements
              or, when none is found there and it has four or more elements, all but one of them (at most the {SUBS_MAX} most frequent are
              tried). The kinds are fixed lists: {ANION_TEXT} count as anions, hydrogen is never moved or swapped, and every other
              element counts as a cation, whatever its role in the compound.
              {gen.stats.measured > 0 && ` ${gen.stats.measured.toLocaleString()} ${verb(gen.stats.measured, 'nudge', 'nudges')} matched a measured composition and ${verb(gen.stats.measured, 'was', 'were')} removed.`}
              {gen.stats.outside > 0 && ` ${gen.stats.outside.toLocaleString()} left your chemistry and ${verb(gen.stats.outside, 'was', 'were')} dropped.`}
              {gen.cands.length > 0 && ' The trees give many compositions the same value; a tie goes to the one closer to a measured composition.'}
            </p>
            {gen.cands.length > 0 && (
              <p className="note">
                Every one is an <strong>untested prediction</strong>: not in this dataset, though it may exist elsewhere in the
                literature. {NUDGE_CAVEAT}
              </p>
            )}
            <div className="options ed-options">
              <label><input type="checkbox" checked={onlyReach} onChange={(e) => setOnlyReach(e.target.checked)} /> Only those whose expected conductivity reaches the target</label>
              <label><input type="checkbox" checked={twoPerSeed} onChange={(e) => setTwoPerSeed(e.target.checked)} /> At most {PER_SEED_SHOWN} from one starting composition</label>
            </div>
            {view.rows.length === 0 ? (
              <p className="note">
                {gen.cands.length === 0 ? (
                  'None of these nudges gives a new composition from the starting compositions below. Widen the chemistry or leave out fewer elements.'
                ) : (
                  <>
                    No new composition has an expected conductivity of <Sigma log={targetLog} /> with these choices. Lower the target, widen the
                    chemistry, or untick the first option to see the closest ones.
                  </>
                )}
              </p>
            ) : (
              <>
                {spread !== null && (
                  <p className="note">
                    {spread < cv.mae
                      ? `The ${view.rows.length} rows below span ${fmt(spread, 3)} log units of expected conductivity, less than the model's average miss of ${fmt(cv.mae, 3)}: their order says little.`
                      : `The ${view.rows.length} rows below span ${fmt(spread, 3)} log units of expected conductivity. Rows closer together than the model's average miss (${fmt(cv.mae, 3)} log units) cannot be told apart.`}
                  </p>
                )}
                <ScrollTable hint="Scroll the table sideways for the measured neighbours and the Details buttons.">
                  <table className="ranked ed-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>New composition and how it was made</th>
                        <th className="num ed-wide">Expected (S/cm)</th>
                        <th className="num ed-wide">Cautious lower estimate (S/cm)</th>
                        <th className="ed-wide">Reaches the target</th>
                        <th>Three nearest measured (S/cm), share of atoms that differ</th>
                        <th><span className="sr">Details</span></th>
                      </tr>
                    </thead>
                    <tbody>
                      {view.rows.map((c, i) => {
                        const back = backing(c, targetLog)
                        const lowOk = c.pred - hw >= targetLog
                        const reach = lowOk ? 'flag ok' : c.pred >= targetLog ? 'flag low' : 'flag plain'
                        return (
                          <tr key={c.key} className={isSel('new', c.key) ? 'on' : ''}>
                            <td>{i + 1}</td>
                            <td className="ed-wrap ed-main">
                              <strong><Chem parts={c.parts} /></strong>
                              {/* On a phone the next three columns are folded into this cell (ElectrolyteDesigner.css). */}
                              <span className="ed-phone">Expected <Sigma log={c.pred} /></span>
                              <span className="ed-phone strong">Cautious estimate <Sigma log={c.pred - hw} /></span>
                              <span className="ed-phone">
                                <span className={reach}>
                                  {lowOk ? 'Reaches the target: cautious estimate' : c.pred >= targetLog ? 'Reaches the target: expected only' : 'Does not reach the target'}
                                </span>
                              </span>
                              <span className="label">untested prediction · from <Chem parts={c.seed.parts} />: {howText(c.how)}</span>
                            </td>
                            <td className="num ed-wide"><Sigma log={c.pred} unit={false} /></td>
                            <td className="num strong ed-wide"><Sigma log={c.pred - hw} unit={false} /></td>
                            <td className="ed-wide">
                              <span className={reach}>{lowOk ? 'Cautious estimate' : c.pred >= targetLog ? 'Expected only' : 'No'}</span>
                            </td>
                            <td className="ed-wrap">
                              <span className={back.ok ? 'flag ok' : 'flag plain'}>{back.text}</span>
                              <span className="label">
                                {c.near.slice(0, 3).map((n, k) => (
                                  <span key={n.u.key}>{k > 0 && '; '}<Sigma log={n.u.medY} unit={false} /> ({differs(n.d)})</span>
                                ))}
                              </span>
                            </td>
                            <td>
                              <button type="button" className="chip" aria-pressed={isSel('new', c.key)} aria-label={`Show details of composition ${i + 1}`}
                                onClick={() => show({ kind: 'new', key: c.key })}>Details</button>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </ScrollTable>
              </>
            )}
            <dl className="readme">
              <div><dt>Expected</dt><dd>The trees&apos; prediction of the room-temperature ionic conductivity of the composition. A prediction, not a measurement.</dd></div>
              <div>
                <dt>Cautious lower estimate</dt>
                <dd>
                  The expected value minus {fmt(hw, 3)} log units (divided by about {bigFactor}). The table is ranked by it.
                  {prep.hwFromMisses
                    ? ` The ±${fmt(hw, 3)} is the 90th percentile of the model's own out-of-fold misses, so it is how the range was set, not a separate test of it.`
                    : coverPct !== null && ` ${prep.nInside} of the ${prep.nScored} out-of-fold misses (${coverPct}%) were within ±${fmt(hw, 3)}.`}
                </dd>
              </div>
              <div><dt>Reaches the target</dt><dd>&quot;Cautious estimate&quot;: even the lower estimate reaches it. &quot;Expected only&quot;: the prediction reaches it but the lower estimate falls short.</dd></div>
              <div>
                <dt>Measured neighbours reach it</dt>
                <dd>
                  At least two of the three nearest measured compositions differ in at most {differs(CLOSE_L1)} of the atoms and reached the
                  target in the lab (median of their measurements). Otherwise the flag says how many of the three reached it, or that the
                  ones reaching it are not that close. The new composition itself is untested.
                </dd>
              </div>
            </dl>
          </section>

          <section className="card fam-section">
            <h2>The measured starting compositions</h2>
            <p className="note">
              Measured values from the data, beside the model&apos;s out-of-fold prediction for each (made by trees that never trained on that
              measurement, nor on any other sharing its composition string or its paper): how close it came where the answer is known.
            </p>
            <ScrollTable hint="Scroll the table sideways for the new compositions made from each and the Details buttons.">
              <table className="compact ed-table">
                <thead>
                  <tr>
                    <th>#</th><th>Measured composition</th><th className="ed-wide">Chemistry</th><th className="num">Measured (S/cm)</th>
                    <th className="num">Model, out of fold (S/cm)</th><th className="num">New ones made</th><th className="num">Best of them, expected (S/cm)</th>
                    <th><span className="sr">Details</span></th>
                  </tr>
                </thead>
                <tbody>
                  {seedsShown.map((u, i) => {
                    const ps = gen.perSeed.get(u.key)
                    return (
                      <tr key={u.key} className={isSel('measured', u.key) ? 'on' : ''}>
                        <td>{i + 1}</td>
                        <td className="ed-name"><Chem parts={u.parts} /><span className="label ed-phone">{u.chem}</span></td>
                        <td className="ed-wide">{u.chem}</td>
                        <td className="num"><Measured u={u} /></td>
                        <td className="num"><Oof u={u} /></td>
                        <td className="num">{ps?.made ?? 0}</td>
                        <td className="num">{ps?.best ? <Sigma log={ps.best.pred} unit={false} /> : '–'}</td>
                        <td>
                          <button type="button" className="chip" aria-pressed={isSel('measured', u.key)} aria-label={`Show details of measured composition ${i + 1}`}
                            onClick={() => show({ kind: 'measured', key: u.key })}>Details</button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </ScrollTable>
            {leakyShown && (
              <p className="fine ed-after">
                &quot;Not a clean test&quot;: the data writes this composition more than one way and the spellings fell in different folds, so the
                trees behind its out-of-fold value had trained on another spelling of it. Open its details for the measurements.
              </p>
            )}
            {gen.seeds.length > SEEDS_FOLDED && (
              <p className="more">
                <button type="button" className="chip" onClick={() => setAllSeeds((v) => !v)}>
                  {allSeeds ? `Show the first ${SEEDS_FOLDED}` : `Show all ${gen.seeds.length}`}
                </button>
              </p>
            )}
          </section>
        </>
      )}

      <section className="card fam-section">
        <h2>How far to trust this designer</h2>
        <ul className="ed-list">
          <li>
            Cross-validated on {model.records.length} measurements, grouped so that measurements sharing a composition string (as written in
            the source) or a paper are never split between training and testing: the expected value misses the measured one
            by <strong>{fmt(cv.mae, 3)} log units on average</strong> (a factor of about {thousands(10 ** cv.mae)}).
            {Number.isFinite(cv.mean_baseline_mae) && ` Predicting the average of the data for every composition misses by ${fmt(cv.mean_baseline_mae, 3)} on average.`}
          </li>
          <li>
            The 90% range is ±{fmt(hw, 3)} log units around the prediction.
            {prep.hwFromMisses
              ? ` That width is the 90th percentile of the out-of-fold misses on these same ${prep.nScored} measurements, so ${prep.nInside} of them (${coverPct}%) fall inside it by construction: the figure shows how the range was set and is not a separate test of it.`
              : coverPct !== null && ` ${prep.nInside} of the ${prep.nScored} out-of-fold misses (${coverPct}%) were within it.`}
            {' '}The cautious lower estimate is its lower edge.
          </li>
          {prep.leaky.length > 0 && (
            <li>
              {prep.respelled} of the {prep.uniq.length} measured compositions {verb(prep.respelled, 'is', 'are')} written more than one way in
              the data (for example <Spelling text={prep.leaky[0].spellings[0]} /> and <Spelling text={prep.leaky[0].spellings[1]} />).
              {' '}For {prep.leaky.length === prep.respelled ? verb(prep.respelled, 'it', `all ${prep.respelled}`) : `${prep.leaky.length} of them`} the
              spellings fell in different folds: the same fractions got different out-of-fold values, which only different folds
              give. For those {prep.leakyRows} measurements the trees behind the out-of-fold value had trained on the same composition
              under another spelling, so it is not a clean test. The average miss and the 90% range above include them.
            </li>
          )}
          {trust && prep.nScored > 0 && (
            <li>
              At your target (<Sigma log={targetLog} />): of the {trust.pos} {verb(trust.pos, 'measurement that reaches', 'measurements that reach')} it, the out-of-fold prediction also
              reached it for {trust.tp}. {trust.predPos > 0 ? `Of the ${trust.predPos} it predicted to reach it, ${trust.tp} really did.` : 'It predicted none to reach it.'}
              {' '}{trust.cautiousPos > 0
                ? `The cautious lower estimate reached it for ${trust.cautiousPos}, of which ${trust.cautiousTp} really did.`
                : 'The cautious lower estimate reached it for none.'}
            </li>
          )}
          <li>
            New compositions sit close to measured ones the trees trained on, so their predictions lean on those measurements. The errors
            above come from measurements held out of training together with every other one sharing their composition string or
            paper, which is the fairer test{prep.leaky.length > 0 ? ' (apart from the spellings noted above)' : ''}.
          </li>
          <li>
            This is a compact model trained for this page ({model.kind ?? 'gradient-boosted trees'}
            {model.params ? `, ${model.params.n_estimators} trees of depth ${model.params.max_depth}` : ''}), separate from the pipeline&apos;s
            network and trees{tasks.length > 0 ? ', whose results follow below' : ''}. Its only inputs are element fractions: it cannot see crystal
            structure, processing, density or grain boundaries.
          </li>
          <li>{NUDGE_CAVEAT}</li>
          {prep.nChecks > 0 && (
            <li>
              {prep.checkWorst < 1e-6
                ? `The page walks the exported trees itself; on the export's ${prep.nChecks} check compositions it reproduces the exporter's predictions to within ${prep.checkWorst.toExponential(1)} log units.`
                : `Warning: on the export's ${prep.nChecks} check compositions the page's predictions differ from the exporter's by up to ${prep.checkWorst.toExponential(1)} log units. Treat them with care.`}
            </li>
          )}
        </ul>
        <p className="fine">Data: {model.source ?? '–'}. Target: {model.unit ?? '–'}. Folds: {cv.folds ?? '–'}. Model file generated {model.generated ?? '–'}.</p>
      </section>
    </>
  )
}
