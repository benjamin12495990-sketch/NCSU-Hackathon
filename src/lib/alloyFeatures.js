// The 37 model inputs of the alloy-hardness network, rebuilt in the browser.
// A JavaScript copy of row_features() and descriptors() in domains/materials/prepare.py;
// scripts/alloy_check.mjs checks it against all 530 rows of nn_data.json at 1e-6.
//   frac  { El: atomic fraction }, fractions summing to 1
//   proc  processing route, one of table.procs (e.g. 'CAST')
//   table public/model/materials/alloys.json (features, elements, procs, table)

// Python's round(v, 6) on the value's exact decimal expansion (toFixed does the same, except
// that an exact tie goes up instead of to even, which does not happen for these values).
const r6 = (v) => Number(v.toFixed(6))

export function descriptors(frac, table) {
  const els = Object.keys(frac).filter((el) => frac[el] > 0)
  const xs = els.map((el) => frac[el])
  const props = els.map((el) => {
    const p = table.table[el]
    if (!p) throw new Error(`element ${el} is not in the property table`)
    return p
  })
  const dot = (k) => xs.reduce((s, x, i) => s + x * props[i][k], 0)
  const rbar = dot(0), cbar = dot(1), vbar = dot(2)
  const spread = (k, bar) => Math.sqrt(xs.reduce((s, x, i) => s + x * (props[i][k] - bar) ** 2, 0))
  return {
    n_elements: els.length,
    mix_entropy_R: -xs.reduce((s, x) => s + x * Math.log(x), 0),
    radius_mean_pm: rbar,
    radius_mismatch_pct: 100 * Math.sqrt(xs.reduce((s, x, i) => s + x * (1 - props[i][0] / rbar) ** 2, 0)),
    eneg_mean: cbar,
    eneg_spread: spread(1, cbar),
    vec_mean: vbar,
    vec_spread: spread(2, vbar),
    tmelt_mean_K: dot(3),
  }
}

export function featurize(frac, proc, table) {
  if (!table.procs.includes(proc)) throw new Error(`unknown processing route ${proc}`)
  const own = new Set(table.elements)
  const d = descriptors(frac, table)
  let other = 0
  for (const [el, x] of Object.entries(frac)) if (!own.has(el)) other += x
  return [
    ...table.elements.map((el) => r6(frac[el] || 0)),
    r6(other),
    ...table.procs.map((p) => (p === proc ? 1 : 0)),
    ...table.descriptors.map((k) => r6(d[k])),
  ]
}
