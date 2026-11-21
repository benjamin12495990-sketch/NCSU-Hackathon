// "Known" mixes are the distinct lab-tested mixes in known.json. Distances compare
// the seven ingredient amounts, each divided by its range in the dataset, so
// no single ingredient dominates.
export function prepareKnown(knownRaw, space, ingredients) {
  const scale = ingredients.map((n) => space.ranges[n][1] - space.ranges[n][0])
  const entries = knownRaw.map((k) => ({ mix: k.m, results: k.s, scaled: k.m.map((v, i) => v / scale[i]) }))
  const keys = new Set(entries.map((e) => mixKey(e.mix)))
  return { entries, scale, keys }
}

// Two mixes count as the same when every amount matches to the nearest kg.
export const mixKey = (mix) => mix.map((v) => Math.round(v)).join('|')

export function scaledDistance(a, b, scale) {
  let d2 = 0
  for (let i = 0; i < a.length; i++) d2 += ((a[i] - b[i]) / scale[i]) ** 2
  return Math.sqrt(d2)
}

// The latest lab result of a mix at or before the given age, or null.
export function resultUpTo(entry, age) {
  let best = null
  for (const r of entry.results) if (r[0] <= age && (!best || r[0] > best[0])) best = r
  return best
}

// The k closest tested mixes among `pool` (entries paired with one lab
// result each), nearest first, with their distance to the candidate.
export function nearestTested(known, pool, mix, k = 3) {
  const s = mix.map((v, i) => v / known.scale[i])
  const out = []
  for (const item of pool) {
    let d2 = 0
    for (let i = 0; i < s.length; i++) d2 += (s[i] - item.entry.scaled[i]) ** 2
    if (out.length < k || d2 < out[out.length - 1].d2) {
      out.push({ d2, item })
      out.sort((x, y) => x.d2 - y.d2)
      if (out.length > k) out.pop()
    }
  }
  return out.map(({ d2, item }) => ({ mix: item.entry.mix, age: item.result[0], strength: item.result[1], distance: Math.sqrt(d2) }))
}
