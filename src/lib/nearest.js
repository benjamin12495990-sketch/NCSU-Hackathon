import { compositionKey } from './featurize.js'

// Prepares known.json for fast lookups: a key -> entry map for the
// "already known" check, and each entry's squared length for distances.
export function prepareKnown(knownRaw) {
  const entries = knownRaw.map(([formula, tc, c]) => {
    const idx = [], val = []
    let norm2 = 0
    for (let k = 0; k < c.length; k += 2) {
      idx.push(c[k]); val.push(c[k + 1]); norm2 += c[k + 1] * c[k + 1]
    }
    return { formula, tc, idx, val, norm2 }
  })
  const byKey = new Map()
  for (const e of entries) {
    byKey.set(e.idx.map((j, k) => j + ':' + e.val[k].toFixed(4)).join('|'), e)
  }
  return { entries, byKey }
}

// Returns the dataset entry with exactly this composition, or null.
export function findKnown(known, x, nElements) {
  return known.byKey.get(compositionKey(x, nElements)) ?? null
}

// Finds the closest known material by straight-line distance between
// element-fraction vectors. A large distance means the model is extrapolating.
export function nearest(known, x, nElements) {
  let xNorm2 = 0
  for (let k = 0; k < nElements; k++) xNorm2 += x[k] * x[k]
  let best = null, bestD2 = Infinity
  for (const e of known.entries) {
    let dot = 0
    for (let k = 0; k < e.idx.length; k++) dot += x[e.idx[k]] * e.val[k]
    const d2 = xNorm2 + e.norm2 - 2 * dot
    if (d2 < bestD2) { bestD2 = d2; best = e }
  }
  return { entry: best, distance: Math.sqrt(Math.max(0, bestD2)) }
}
