// Pure helpers of CorrosionDesigner.jsx (no React, so a node script can test them): rebuilding the model's input vector in the browser
// exactly as gpu/data.py load_corrosion builds it (the recipe is in README.md of this folder), predicting every material in one
// environment, and indexing the measured records. Every number comes from public/pipeline/corrosion_model.json.
import { predictTrees } from '../../lib/pipeline.js'

const NUMBERS = ['conc_vol_pct', 'temp_C_mid', 'log10_hours']
const finite = (v) => typeof v === 'number' && Number.isFinite(v)

export function validModel(m) {
  return Array.isArray(m?.features) && m.features.length > 0 && Array.isArray(m?.trees) && m.trees.length > 0 && finite(m?.init) &&
    Array.isArray(m?.materials) && m.materials.length > 0 && Array.isArray(m?.environments) && m.environments.length > 0 &&
    Array.isArray(m?.records) && finite(m?.cv?.mae) && finite(m?.cv?.interval90_half_width) && m.cv.interval90_half_width > 0 &&
    NUMBERS.every((n) => m.features.includes(n) && finite(m?.numeric_medians?.[n]))
}

// The column a category value switches on: its own column "<category>=<value>" (an empty value counts as "none"), else the pooled
// "<category>=(infrequent)" column, else none (-1). `own` says whether the model has a column of its own for the value.
function slot(index, category, value) {
  const text = String(value ?? '').trim() || 'none'
  const own = index.get(`${category}=${text}`)
  if (own !== undefined) return { i: own, own: true }
  const pooled = index.get(`${category}=(infrequent)`)
  return { i: pooled === undefined ? -1 : pooled, own: false }
}

const median = (sorted) => (sorted.length ? (sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2) : null)
const span = (values) => {
  const s = values.filter(finite).sort((a, b) => a - b)
  return s.length ? { n: s.length, min: s[0], max: s[s.length - 1], median: median(s) } : null
}

// Everything that depends only on the model file.
export function prepare(model) {
  const index = new Map(model.features.map((f, i) => [f, i]))
  const col = (name) => (index.has(name) ? index.get(name) : -1)
  const cols = {
    conc: col('conc_vol_pct'), temp: col('temp_C_mid'), hours: col('log10_hours'),
    concMissing: col('conc_vol_pct__missing'), tempMissing: col('temp_C_mid__missing'), hoursMissing: col('log10_hours__missing'),
  }
  // Materials. Many share every input (same class and family, no column of their own), so they are predicted once per group.
  const groups = new Map()
  const materials = model.materials.map((m, id) => {
    const c = slot(index, 'material_class', m.class)
    const f = slot(index, 'material_family', m.family)
    const s = slot(index, 'material', m.name)
    const key = `${c.i}|${f.i}|${s.i}`
    if (!groups.has(key)) groups.set(key, { g: groups.size, on: [c.i, f.i, s.i].filter((i) => i >= 0) })
    return {
      id, name: String(m.name), cls: String(m.class ?? '').trim() || 'none', family: String(m.family ?? '').trim() || 'none',
      rows: finite(m.rows) ? m.rows : 0, g: groups.get(key).g, own: s.own, ownFamily: f.own, ownClass: c.own, envs: new Map(),
    }
  })
  const byName = new Map(materials.map((m) => [m.name, m]))
  const classCounts = new Map()
  for (const m of materials) classCounts.set(m.cls, (classCounts.get(m.cls) ?? 0) + 1)
  const classes = [...classCounts].map(([name, n]) => ({ name, n })).sort((a, b) => b.n - a.n || (a.name < b.name ? -1 : 1))

  // Measured records, by environment and then by material. A record's rate is 10^y mm/yr.
  const byEnv = new Map()
  for (const r of model.records) {
    if (!r || !finite(r.y)) continue
    const mat = byName.get(String(r.material))
    if (!mat) continue
    const env = String(r.environment)
    if (!byEnv.has(env)) byEnv.set(env, { all: [], byMat: new Map() })
    const e = byEnv.get(env)
    const rec = {
      mat, y: r.y, conc: finite(r.conc_vol_pct) ? r.conc_vol_pct : null, temp: finite(r.temp_C) ? r.temp_C : null,
      hours: finite(r.hours) ? r.hours : null, ref: String(r.reference ?? '').trim(),
    }
    e.all.push(rec)
    if (!e.byMat.has(mat.id)) e.byMat.set(mat.id, [])
    e.byMat.get(mat.id).push(rec)
    mat.envs.set(env, (mat.envs.get(env) ?? 0) + 1)
  }
  const environments = model.environments.map((e) => {
    const name = String(e.name)
    const s = slot(index, 'environment', name)
    const recs = byEnv.get(name) ?? { all: [], byMat: new Map() }
    const ys = recs.all.map((r) => r.y).sort((a, b) => a - b)
    return {
      name, rows: finite(e.rows) ? e.rows : recs.all.length, i: s.i, own: s.own, records: recs.all, byMat: recs.byMat, ys,
      conc: span(recs.all.map((r) => r.conc)), temp: span(recs.all.map((r) => r.temp)), hours: span(recs.all.map((r) => r.hours)),
    }
  }).sort((a, b) => b.rows - a.rows || (a.name < b.name ? -1 : 1))
  const envByName = new Map(environments.map((e) => [e.name, e]))

  // The summary of each material's measured rates in each environment (log10 mm/yr, sorted).
  for (const e of environments) {
    e.stats = new Map()
    for (const [id, recs] of e.byMat) {
      const ys = recs.map((r) => r.y).sort((a, b) => a - b)
      e.stats.set(id, { n: recs.length, ys, min: ys[0], max: ys[ys.length - 1], median: median(ys), temp: span(recs.map((r) => r.temp)), conc: span(recs.map((r) => r.conc)) })
    }
  }

  const qual = {
    concExact: slot(index, 'conc_qualifier', 'exact').i, concNone: slot(index, 'conc_qualifier', 'none').i,
    tempExact: slot(index, 'temp_qualifier', 'exact').i, tempNone: slot(index, 'temp_qualifier', 'none').i,
    tempRoom: index.has('temp_qualifier=room') ? index.get('temp_qualifier=room') : -1,
    tempBoiling: index.has('temp_qualifier=boiling') ? index.get('temp_qualifier=boiling') : -1,
  }

  // The export's check rows: scikit-learn's own predictions for 12 training rows, with their non-zero inputs.
  const checks = Array.isArray(model.checks) ? model.checks : []
  let checkWorst = 0
  for (const c of checks) {
    const x = new Float64Array(model.features.length)
    for (const [name, v] of Object.entries(c?.x_nonzero ?? {})) if (index.has(name)) x[index.get(name)] = v
    checkWorst = Math.max(checkWorst, Math.abs(predictTrees(model, x) - c.pred))
  }

  const everything = environments.flatMap((e) => e.records)
  const all = {
    n: everything.length, conc: span(everything.map((r) => r.conc)), temp: span(everything.map((r) => r.temp)),
    hours: span(everything.map((r) => r.hours)), y: span(everything.map((r) => r.y)),
  }

  return {
    size: model.features.length, index, cols, qual, materials, byName, groups: [...groups.values()], classes, environments, envByName, all,
    medians: model.numeric_medians, hw: model.cv.interval90_half_width, nRecords: model.records.length,
    nOwnMaterials: materials.filter((m) => m.own).length, nOwnEnvironments: environments.filter((e) => e.own).length,
    nChecks: checks.length, checkWorst,
  }
}

// The part of the input vector that describes the exposure (everything but the material), as the recipe in README.md says:
// a number the visitor gives is used as it is and sets its qualifier to "exact"; a blank number takes the median of the data and
// switches its "__missing" column on, with the qualifier "none" (or "room" / "boiling" for a temperature given in those words,
// which is how the data records them: without a number).
//   ask: { environment, conc: number|null, temp: number|null, tempWord: 'room'|'boiling'|null, hours: number|null }
export function exposureVector(prep, ask) {
  const x = new Float64Array(prep.size)
  const set = (i, v) => { if (i >= 0) x[i] = v }
  const env = slot(prep.index, 'environment', ask.environment)
  set(env.i, 1)
  const { cols, qual, medians } = prep
  if (ask.conc !== null && ask.conc !== undefined) {
    set(qual.concExact, 1)
    set(cols.conc, ask.conc)
  } else {
    set(qual.concNone, 1)
    set(cols.conc, medians.conc_vol_pct)
    set(cols.concMissing, 1)
  }
  if (ask.temp !== null && ask.temp !== undefined) {
    set(qual.tempExact, 1)
    set(cols.temp, ask.temp)
  } else {
    const word = ask.tempWord === 'room' ? qual.tempRoom : ask.tempWord === 'boiling' ? qual.tempBoiling : -1
    set(word >= 0 ? word : qual.tempNone, 1)
    set(cols.temp, medians.temp_C_mid)
    set(cols.tempMissing, 1)
  }
  if (ask.hours !== null && ask.hours !== undefined) {
    set(cols.hours, Math.log10(Math.max(ask.hours, 0.1)))
  } else {
    set(cols.hours, medians.log10_hours)
    set(cols.hoursMissing, 1)
  }
  return x
}

// The full input vector of one material in an exposure (used by the tests; the page predicts by group, below).
export function inputVector(prep, ask, materialName) {
  const x = exposureVector(prep, ask)
  const m = prep.byName.get(materialName)
  if (m) for (const i of prep.groups[m.g].on) x[i] = 1
  return x
}

// log10(mm/yr) predicted for every material in one exposure: one walk of the trees per group of materials that share all inputs.
export function predictAll(model, prep, ask) {
  const x = exposureVector(prep, ask)
  const byGroup = new Float64Array(prep.groups.length)
  for (let g = 0; g < prep.groups.length; g++) {
    const on = prep.groups[g].on
    for (const i of on) x[i] = 1
    byGroup[g] = predictTrees(model, x)
    for (const i of on) x[i] = 0
  }
  return byGroup
}

// The export rounds each log10 rate to four significant figures, so a rate measured at exactly the limit can sit a hair above it
// (0.5 mm/yr is stored as -0.301, which is 0.50003). Measured rates are compared with this much slack, in log10 units.
export const MEASURED_SLACK = 0.0005

// Rows of the ranked table for one environment: every material (of one class, if asked), lowest prediction first, which is also the
// order of the cautious upper estimate because every prediction gets the same 90% range. With `fold`, materials that the model gives
// the same inputs and that have no record in this environment share one row: nothing on the page could tell them apart. A tie goes
// to the row with more measured records here, then to the name.
export function rank(prep, preds, env, cls, fold) {
  const rows = []
  const folds = new Map()
  for (const mat of prep.materials) {
    if (cls !== 'any' && mat.cls !== cls) continue
    const st = env.stats.get(mat.id) ?? null
    if (!st && fold) {
      let row = folds.get(mat.g)
      if (!row) {
        row = { key: `g${mat.g}`, mats: [], pred: preds[mat.g], st: null }
        folds.set(mat.g, row)
        rows.push(row)
      }
      row.mats.push(mat)
    } else {
      rows.push({ key: `m${mat.id}`, mats: [mat], pred: preds[mat.g], st })
    }
  }
  rows.sort((a, b) => a.pred - b.pred || (b.st?.n ?? 0) - (a.st?.n ?? 0) || (a.mats[0].name < b.mats[0].name ? -1 : a.mats[0].name > b.mats[0].name ? 1 : 0))
  return rows
}

// One row against a limit (log10 mm/yr): is the cautious upper estimate within it, is the prediction, how many of the material's
// measured rates in this environment are, and do those measured neighbours agree (the prediction is within the limit and so is every
// measured rate) or disagree (the prediction is within it, at least one measured rate is not).
export function judge(row, limit, hw) {
  const cautious = row.pred + hw <= limit
  const expected = row.pred <= limit
  let within = 0
  if (row.st) for (const y of row.st.ys) if (y <= limit + MEASURED_SLACK) within++
  const n = row.st ? row.st.n : 0
  return { cautious, expected, n, within, agree: expected && n > 0 && within === n, disagree: expected && n > 0 && within < n }
}

// A material's records in one environment, ordered for the visitor: closest to the conditions asked for first. The distance adds, for
// each number the visitor gave, the difference divided by the span of that number over the environment's records (hours on a log
// scale); a record without the number counts as a whole span away. With no number given: highest measured rate first.
export function orderRecords(recs, env, ask) {
  const dims = []
  const width = (s) => (s && s.max > s.min ? s.max - s.min : 1)
  if (ask.conc !== null && ask.conc !== undefined) {
    const w = width(env.conc)
    dims.push((r) => (r.conc === null ? 1 : Math.abs(r.conc - ask.conc) / w))
  }
  if (ask.temp !== null && ask.temp !== undefined) {
    const w = width(env.temp)
    dims.push((r) => (r.temp === null ? 1 : Math.abs(r.temp - ask.temp) / w))
  }
  if (ask.hours !== null && ask.hours !== undefined) {
    const lg = (h) => Math.log10(Math.max(h, 0.1))
    const w = env.hours && env.hours.max > env.hours.min ? lg(env.hours.max) - lg(env.hours.min) : 1
    dims.push((r) => (r.hours === null ? 1 : Math.abs(lg(r.hours) - lg(ask.hours)) / w))
  }
  if (!dims.length) return { closest: false, list: [...recs].sort((a, b) => b.y - a.y) }
  const scored = recs.map((r) => ({ r, d: dims.reduce((sum, f) => sum + f(r), 0) }))
  scored.sort((a, b) => a.d - b.d || b.r.y - a.r.y)
  return { closest: true, list: scored.map((x) => x.r) }
}
