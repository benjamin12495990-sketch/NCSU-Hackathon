// Shared helpers for the pages built on the GPU pipeline's exports (public/pipeline/, written on the laptop by
// `python -m gpu.export_site` after a nightly run). Every number shown comes from those files; nothing is typed here.
//
// Files (see gpu/export_site.py for how each value is computed):
//   index.json                 one entry per nightly task: models' errors, comparisons with 95% intervals, 90% intervals
//   tab_<family>.json          held-out rows of a table family (cross-validated, out of fold): truth, network, trees, blend
//   crystal_<target>.json      held-out-systems test crystals: truth, network, trees, blend
//   structures.json            unit cells of those crystals, for the 3D view
//   corrosion_model.json, electrolyte_model.json, surfactants.json   small models / tables for the designers

const cache = new Map()

// Fetches public/pipeline/<name> once per page load. Resolves to null when the file is absent (no export yet).
// The request is shared by every caller, so it never takes a caller's AbortSignal: aborting it for one component (React
// StrictMode unmounts and remounts every component once under `npm run dev`) would fail it for all the others. Callers
// that unmount ignore the result instead. The second argument is accepted and ignored, for older callers.
// eslint-disable-next-line no-unused-vars
export function loadPipeline(name, _signal) {
  if (!cache.has(name)) {
    cache.set(name, fetch(`${import.meta.env.BASE_URL}pipeline/${name}`)
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => { cache.delete(name); return null }))
  }
  return cache.get(name)
}

// Walks gradient-boosted regression trees exported by gpu/export_site.py: prediction = init + sum of leaf values.
// Each tree: f (feature per node, -2 at leaves), t (threshold), l/r (children, -1 at leaves), v (leaf value, learning rate applied).
export function predictTrees(model, x) {
  let s = model.init
  for (const t of model.trees) {
    let n = 0
    while (t.l[n] !== -1) n = x[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n]
    s += t.v[n]
  }
  return s
}

// Number formatting: 4 significant figures, trailing zeros dropped; a sign for differences.
export function fmt(x, sig = 4) {
  if (x === null || x === undefined || !Number.isFinite(x)) return '–'
  if (x === 0) return '0'
  return String(Number(x.toPrecision(sig)))
}
export const signed = (x, sig = 3) => (x === null || x === undefined ? '–' : (x > 0 ? '+' : '') + fmt(x, sig))
export const interval = (ci) => (ci ? `[${signed(ci[0])}, ${signed(ci[1])}]` : '')

// A paired comparison from the export ({diff, ci95, first, second, better, verdict}) in the project's wording:
// "X clearly better" only when the whole 95% interval is on one side of zero; otherwise "no clear difference" WITH its interval.
export function comparisonText(c) {
  if (!c) return 'not available'
  return c.better ? `${c.better} clearly better, ${signed(c.diff)} ${interval(c.ci95)}` : `no clear difference, ${signed(c.diff)} ${interval(c.ci95)}`
}

// Log10 targets are shown in their own units too (a factor of 10^w for a half-width w).
export const factor = (w) => (w === null || w === undefined ? '–' : `×${fmt(10 ** w, 2)}`)

// The fixed set of models every export carries, with the words the pages use for them.
export const MODELS = [
  { key: 'network', label: 'Network (average of runs)' },
  { key: 'trees', label: 'Strongest tree model' },
  { key: 'blend', label: 'Half network, half trees (blend)' },
]

// Rows of one task, filtered by a free-text query over the VALUES of their label fields (formula, material, SMILES,
// composition, ...). Field names and JSON punctuation are not searched, so "material" or a quote mark matches nothing.
export function matchRows(rows, query) {
  const q = query.trim().toLowerCase()
  if (!q) return rows
  const text = (r) => Object.values(r.label ?? { f: r.formula, s: r.system, j: r.jid })
    .filter((v) => v !== null && v !== undefined).map(String).join(' \u0001 ').toLowerCase()
  return rows.filter((r) => text(r).includes(q))
}
