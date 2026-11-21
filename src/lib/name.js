// Gives a mix a name built only from its own recipe, so the same mix always
// gets the same name. C, S and F are the shares of cement, slag and fly ash in
// the binder by mass (percent); W is water divided by binder, times 100.
// Example: "Ternary blend C30-S35-F35-W40". These are labels for this page,
// not standard designations.
export function mixName(mix) {
  const [cement, slag, fly, water] = mix
  const binder = cement + slag + fly
  const s = Math.round((slag / binder) * 100)
  const f = Math.round((fly / binder) * 100)
  const kind = s > 0 && f > 0 ? 'Ternary blend' : s > 0 ? 'Slag blend' : f > 0 ? 'Fly ash blend' : 'Cement-only mix'
  const shares = [`C${100 - s - f}`, s > 0 && `S${s}`, f > 0 && `F${f}`].filter(Boolean).join('-')
  return `${kind} ${shares}-W${Math.round((water / binder) * 100)}`
}

// Names a ranked list. If two mixes in the list would share a name (same
// shares, different aggregate amounts), the later ones get "(b)", "(c)", ...
export function nameAll(items) {
  const used = new Map()
  return items.map((item) => {
    const base = mixName(item.mix)
    const n = (used.get(base) ?? 0) + 1
    used.set(base, n)
    return { ...item, name: n === 1 ? base : `${base} (${String.fromCharCode(96 + n)})` }
  })
}
