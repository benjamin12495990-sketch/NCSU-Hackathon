import { fmt } from '../../lib/pipeline.js'

// The logic of the crystal finder (CrystalFinder.jsx), kept free of React so it can be checked on its own.
// Input: the crystal tasks of index.json and their rows files (crystal_<target>.json). Each rows file lists held-out crystals with the
// DFT-calculated value `y` and the three predictions (network average, strongest trees, blend) in the target's own units; bulk and
// shear modulus are log10(GPa) there, so the page converts to GPa for people and back to log10 for every comparison.
// Nothing here invents a number: thresholds come from the visitor, everything else from the files.

export const MODEL_KEYS = ['network', 'trees', 'blend']

// Names and units people read, for the targets the pipeline trains. `convert` (set in prepare) is true when the export's target is a
// log10 of this unit. `positive` marks a property that can never be below zero (a stiffness, a gap), so nobody needs to type a minus
// sign for it. A target not listed here is shown as the export names it, in the export's own units.
const KNOWN = {
  bulk_modulus_kv: { label: 'Bulk modulus', unit: 'GPa', order: 0, positive: true },
  shear_modulus_gv: { label: 'Shear modulus', unit: 'GPa', order: 1, positive: true },
  formation_energy_peratom: { label: 'Formation energy', unit: 'eV/atom', order: 2 },
  optb88vdw_bandgap: { label: 'Band gap (OptB88vdW)', unit: 'eV', order: 3, positive: true },
  mbj_bandgap: { label: 'Band gap (TBmBJ)', unit: 'eV', order: 4, positive: true },
}

// The chemical elements, to tell "Li Xe" (two elements, one of them in no crystal here) from text such as "JVASP".
const SYMBOLS = new Set(('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd ' +
  'Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm ' +
  'Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split(' '))

const isNum = (v) => typeof v === 'number' && Number.isFinite(v)

// "blend", "trees", "network" or a longer name starting with "network" in the registry; null for anything else.
function namedModel(rec) {
  if (typeof rec !== 'string') return null
  if (rec.startsWith('network')) return 'network'
  return rec === 'trees' || rec === 'blend' ? rec : null
}
// The model that screens a property when the visitor asks for the recommended one: the registry's, or the blend when the export
// carries no usable recommendation (prepare() records which, as `recKnown`, so the page never calls that fallback "recommended").
export const modelKey = (rec) => namedModel(rec) ?? 'blend'

// The value at fraction p of a sorted list (linear interpolation between neighbours).
function quantile(sorted, p) {
  if (!sorted.length) return null
  const k = (sorted.length - 1) * p
  const f = Math.floor(k)
  const c = Math.min(f + 1, sorted.length - 1)
  return sorted[f] + (sorted[c] - sorted[f]) * (k - f)
}

// "C11H18O4Zn" -> [["C", 11], ["H", 18], ["O", 4], ["Zn", 1]]; null when the text is not element symbols with optional counts.
// Spaces, commas, hyphens and middle dots may separate the symbols ("Li O", "As-Lu-O").
export function parseFormula(text) {
  const out = []
  const re = /([A-Z][a-z]?)(\d+(?:\.\d+)?)?|[\s,·-]+/y
  let at = 0
  while (at < text.length) {
    re.lastIndex = at
    const m = re.exec(text)
    if (!m || m[0].length === 0) return null
    if (m[1]) out.push([m[1], m[2] ? Number(m[2]) : 1])
    at = re.lastIndex
  }
  return out.length ? out : null
}

// Element -> share of the atoms, so "Fe2O3" and "Fe4O6" compare equal.
function shares(parts) {
  const m = new Map()
  let total = 0
  for (const [el, n] of parts) {
    m.set(el, (m.get(el) ?? 0) + n)
    total += n
  }
  if (!(total > 0)) return null
  for (const [el, n] of m) m.set(el, n / total)
  return m
}

function sameShares(a, b) {
  if (!a || !b || a.size !== b.size) return false
  for (const [el, v] of a) if (!b.has(el) || Math.abs(b.get(el) - v) > 1e-6) return false
  return true
}

// Everything that depends only on the files: the properties (one per crystal task whose rows loaded) and the crystals, merged by
// JARVIS id across the rows files. A crystal is only in the test set of some properties, so it may lack the others.
export function prepare(tasks, files) {
  const props = []
  const dropped = []
  for (const t of tasks) {
    const d = files[t.name]
    const title = String(t.title ?? t.name).replace(/\s*\(DFT\)\s*$/, '')
    if (!d || !Array.isArray(d.rows)) {
      dropped.push({ key: t.name, label: KNOWN[t.name]?.label ?? title, why: `its rows file (${t.rows_file}) is missing from this export or cannot be read` })
      continue
    }
    const iv = d.interval90 ?? t.interval90 ?? {}
    const hw = {}
    const coverage = {}
    for (const m of MODEL_KEYS) {
      hw[m] = isNum(iv[m]?.half_width) && iv[m].half_width > 0 ? iv[m].half_width : null
      coverage[m] = isNum(iv[m]?.coverage) ? iv[m].coverage : null
    }
    if (MODEL_KEYS.some((m) => hw[m] === null)) {
      dropped.push({ key: t.name, label: KNOWN[t.name]?.label ?? title, why: 'its 90% ranges are missing from this export' })
      continue
    }
    const known = KNOWN[t.name]
    const unitText = String(d.unit ?? t.unit ?? '')
    const convert = !!known && /log10/i.test(unitText)
    props.push({
      key: t.name, task: t, label: known?.label ?? title, unit: known ? known.unit : '', unitText, convert, order: known?.order ?? 99,
      hw, coverage, rec: modelKey(t.recommended), recKnown: namedModel(t.recommended) !== null,
      // `signed`: a limit on it can be negative (formation energy; a target this file does not know). `floorZero`: the property cannot
      // be below zero, yet its range (prediction ± half-width, in its own units) can run there (band gap).
      signed: !convert && !known?.positive, floorZero: !convert && !!known?.positive,
      rowsShown: d.rows.length, rowsTotal: isNum(d.rows_total) ? d.rows_total : null,
      sampleRule: d.sample_rule ?? null, split: d.split ?? null,
    })
  }
  props.sort((a, b) => a.order - b.order || (a.label < b.label ? -1 : 1))

  const byId = new Map()
  for (const p of props) {
    const ys = []
    for (const r of files[p.key].rows) {
      if (!r || !isNum(r.y) || !MODEL_KEYS.every((m) => isNum(r[m]))) continue
      const jid = String(r.jid ?? '')
      if (!jid) continue
      let c = byId.get(jid)
      if (!c) {
        const formula = String(r.formula ?? '')
        const system = String(r.system ?? '')
        const parts = parseFormula(formula)
        const els = system ? system.split('-').filter(Boolean) : parts ? [...new Set(parts.map((x) => x[0]))] : []
        const id = jid.toLowerCase()
        // For the text search: the id is split at its first hyphen, so the "JVASP" every id starts with is not searched by accident.
        const find = { formula: formula.toLowerCase(), system: system.toLowerCase(), id, idTail: id.slice(id.indexOf('-') + 1), idKey: id.replace(/[^a-z0-9]/g, '') }
        c = { jid, formula, system, els, shares: parts ? shares(parts) : null, find, v: {}, i: byId.size }
        byId.set(jid, c)
      }
      if (c.v[p.key]) continue // a repeated id in one file: the first row stands
      c.v[p.key] = { y: r.y, network: r.network, trees: r.trees, blend: r.blend }
      ys.push(r.y)
    }
    ys.sort((a, b) => a - b)
    p.n = ys.length
    p.stats = ys.length ? { min: ys[0], q1: quantile(ys, 0.25), median: quantile(ys, 0.5), q3: quantile(ys, 0.75), max: ys[ys.length - 1] } : null
  }
  const crystals = [...byId.values()]
  const elements = new Set()
  for (const c of crystals) for (const e of c.els) elements.add(e)
  const complete = crystals.filter((c) => props.every((p) => c.v[p.key])).length
  return { props, dropped, crystals, elements, complete }
}

// Between the units people type and read (GPa for the moduli) and the units of the files (log10 GPa for the moduli).
export const toShown = (p, v) => (p.convert ? 10 ** v : v)
export const toData = (p, v) => (p.convert ? Math.log10(v) : v)
// A value of the files in the units people read, to three significant figures.
export const show = (p, v) => (isNum(v) ? fmt(toShown(p, v), 3) : '–')

// One limit typed by the visitor, in the units people read. Blank means no limit. Returns the limit in the units of the files
// (`value`) and as typed (`shown`). A typographic minus sign and a decimal comma ("67,1", "-0,535") are accepted. A comma followed
// by exactly three digits after a number that does not start with 0 ("1,000", "67,100") can be a thousands comma or a decimal comma,
// so it is not guessed: the visitor is asked.
// `pending` marks a number that is only half typed.
export function parseBound(text, p, side) {
  const raw = String(text ?? '').trim().replace(/[\u2212\u2013]/g, '-')
  const none = (error) => ({ value: null, shown: null, error })
  if (raw === '') return none('')
  const what = `${p.label.toLowerCase()}${p.unit ? ` in ${p.unit}` : ''}`
  if (/^[+-]?[1-9]\d{0,2}(,\d{3})+(\.\d+)?$/.test(raw)) {
    const where = `${p.label}, ${side === 'min' ? 'at least' : 'at most'}`
    const plain = raw.replace(/,/g, '')
    const single = /^[+-]?\d+,\d+$/.test(raw)
    return none(single
      ? `${where}: “${raw}” could mean ${plain} or ${String(Number(raw.replace(',', '.')))}. Type it without the comma, or with a decimal point.`
      : `${where}: type ${plain} without ${raw.split(',').length > 2 ? 'commas' : 'the comma'}.`)
  }
  const t = raw.replace(',', '.')
  if (!/^[+-]?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?$/i.test(t)) {
    // "-", "." or "1e-" is a number still being typed: not a mistake yet, but not a limit either.
    if (/^[+-]?\d*\.?\d*(e[+-]?)?$/i.test(t)) return { ...none(''), pending: true }
    return none(`Type the ${side === 'min' ? 'lowest' : 'highest'} ${what} as a plain number, or leave it blank for no limit.`)
  }
  const n = Number(t)
  if (!Number.isFinite(n)) return none(`That ${what} is too large to use.`)
  if (p.convert && !(n > 0)) return none(`A ${what} must be above 0.`)
  return { value: toData(p, n), shown: n, error: '' }
}

// How the search box was read. Element symbols typed with capitals ("Li O", "LiFePO4") ask for crystals that contain all of them
// (JARVIS writes formulas with the elements in alphabetical order, so matching the typed text would miss most of them). `missing`
// lists the typed elements that no crystal of the export contains: the answer is then "none", not a text search. Anything else is
// matched as text inside the formula, the chemical system or the JARVIS id.
export function readQuery(prep, text) {
  const q = String(text ?? '').trim()
  if (!q) return { mode: 'all', text: '' }
  const parts = parseFormula(q)
  if (parts && parts.every(([el]) => prep.elements.has(el) || SYMBOLS.has(el))) {
    const elements = [...new Set(parts.map((x) => x[0]))]
    return { mode: 'elements', text: q, elements, missing: elements.filter((el) => !prep.elements.has(el)), shares: shares(parts) }
  }
  return { mode: 'text', text: q }
}

// Text search, whatever the capitals: inside the formula or the chemical system; in the id, inside the part after "JVASP-", or from
// the start of the whole id when two letters or more are typed ("jv", "jvasp-41", "jvasp 41603"), or anywhere in it when the text has
// a digit ("sp-41603"). Without that rule "as" (meant as arsenic) or "s" would match the "JVASP" of every id.
function textMatch(c, q, qKey) {
  const f = c.find
  if (f.formula.includes(q) || f.system.includes(q) || f.idTail.includes(q)) return true
  if (/\d/.test(q) && f.id.includes(q)) return true
  return qKey.length > 1 && (f.id.startsWith(q) || f.idKey.startsWith(qKey))
}

function pool(prep, query) {
  if (query.mode === 'elements') return query.missing?.length ? [] : prep.crystals.filter((c) => query.elements.every((e) => c.els.includes(e)))
  if (query.mode === 'text') {
    const q = query.text.toLowerCase()
    const qKey = q.replace(/[^a-z0-9]/g, '')
    return prep.crystals.filter((c) => textMatch(c, q, qKey))
  }
  return prep.crystals
}

// One requirement on one crystal. The judged range is the prediction itself, or with `cautious` the whole 90% range of the
// prediction (prediction ± half-width), which must sit inside the limits. `inside` is the room left to the nearer limit, in the
// units of the file; `norm` is that room in half-widths, so properties with different units compare.
export function judge(p, cell, req, model, cautious) {
  const pred = cell[model]
  const hw = p.hw[model]
  const lo = cautious ? pred - hw : pred
  const hi = cautious ? pred + hw : pred
  let inside = Infinity
  if (req.min !== null) inside = Math.min(inside, lo - req.min)
  if (req.max !== null) inside = Math.min(inside, req.max - hi)
  return { ok: inside >= 0, inside, norm: inside / hw }
}

const truthMeets = (cell, req) => (req.min === null || cell.y >= req.min) && (req.max === null || cell.y <= req.max)

// The screen. `ask` = { reqs: { [property key]: { min, max } } (limits in the units of the files, null = none), model: 'rec' or a
// model key, cautious, query (from readQuery) }.
// Returns the crystals predicted to meet every requirement (most room first), the ones whose DFT-calculated values meet them but
// the prediction does not (missed), how many could not be judged because the export has no value of a required property for them,
// and the honest test: on the judged crystals, what the predictions alone would have picked against what the DFT-calculated values
// say, at the cautious edge and on the prediction itself.
export function screen(prep, ask) {
  const set = prep.props.filter((p) => ask.reqs[p.key] && (ask.reqs[p.key].min !== null || ask.reqs[p.key].max !== null))
  const modelOf = {}
  for (const p of prep.props) modelOf[p.key] = ask.model === 'rec' ? p.rec : ask.model
  const all = pool(prep, ask.query)
  const exact = (c) => ask.query.mode === 'elements' && sameShares(c.shares, ask.query.shares)
  const lacking = {}
  const hits = []
  const missed = []
  const test = { n: 0, real: 0, cautious: { picked: 0, right: 0 }, plain: { picked: 0, right: 0 } }
  let judged = 0
  for (const c of all) {
    const miss = set.filter((p) => !c.v[p.key])
    if (miss.length) {
      for (const p of miss) lacking[p.key] = (lacking[p.key] ?? 0) + 1
      continue
    }
    judged++
    if (!set.length) {
      hits.push({ c, j: {}, tight: null, norm: null, truth: null, exact: exact(c) })
      continue
    }
    const j = {}
    let tight = null
    let okCautious = true
    let okPlain = true
    let truth = true
    for (const p of set) {
      const req = ask.reqs[p.key]
      const cell = c.v[p.key]
      const m = modelOf[p.key]
      const cautiousJ = judge(p, cell, req, m, true)
      const plainJ = judge(p, cell, req, m, false)
      if (!cautiousJ.ok) okCautious = false
      if (!plainJ.ok) okPlain = false
      const t = truthMeets(cell, req)
      if (!t) truth = false
      j[p.key] = { ...(ask.cautious ? cautiousJ : plainJ), truth: t }
      if (tight === null || j[p.key].norm < j[tight].norm) tight = p.key
    }
    test.n++
    if (truth) test.real++
    if (okCautious) { test.cautious.picked++; if (truth) test.cautious.right++ }
    if (okPlain) { test.plain.picked++; if (truth) test.plain.right++ }
    const row = { c, j, tight, norm: j[tight].norm, truth, exact: exact(c) }
    if (ask.cautious ? okCautious : okPlain) hits.push(row)
    else if (truth) missed.push(row)
  }
  const byName = (a, b) => (a.c.formula < b.c.formula ? -1 : a.c.formula > b.c.formula ? 1 : a.c.jid < b.c.jid ? -1 : a.c.jid > b.c.jid ? 1 : 0)
  const order = (a, b) => Number(b.exact) - Number(a.exact) || (set.length ? b.norm - a.norm : 0) || byName(a, b)
  hits.sort(order)
  missed.sort(order)
  return { set, modelOf, pool: all.length, judged, lacking, hits, missed, test }
}
