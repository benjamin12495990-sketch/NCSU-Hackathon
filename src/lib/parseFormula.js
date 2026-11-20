// Turns a formula string such as "YBa2Cu3O7" or "La1.85Sr0.15CuO4" into
// ordered [symbol, amount] pairs. Throws an Error with a readable message
// for empty input, junk, or symbols the model does not know.
export function parseFormula(text, elements) {
  const s = String(text ?? '').replace(/\s+/g, '')
  if (!s) throw new Error('Type a chemical formula, for example YBa2Cu3O7.')
  const known = new Set(elements)
  let i = 0

  const readNumber = () => {
    const m = /^(\d+\.?\d*|\.\d+)/.exec(s.slice(i))
    if (!m) return 1
    i += m[0].length
    return Number(m[0])
  }

  const readGroup = (depth) => {
    const parts = [] // [symbol, amount]
    while (i < s.length) {
      const ch = s[i]
      if (ch === '(' || ch === '[') {
        i++
        const inner = readGroup(depth + 1)
        if (s[i] !== ')' && s[i] !== ']') throw new Error('A bracket is opened but never closed.')
        i++
        const mult = readNumber()
        for (const [el, n] of inner) parts.push([el, n * mult])
      } else if (ch === ')' || ch === ']') {
        if (depth === 0) throw new Error('A bracket is closed but was never opened.')
        return parts
      } else {
        const m = /^[A-Z][a-z]?/.exec(s.slice(i))
        if (!m) {
          throw new Error(
            `Could not read "${s.slice(i, i + 6)}". Element symbols start with a capital letter, like Cu or O.`,
          )
        }
        if (!known.has(m[0])) {
          throw new Error(`"${m[0]}" is not an element this model knows (it covers H to Rn).`)
        }
        i += m[0].length
        parts.push([m[0], readNumber()])
      }
    }
    return parts
  }

  const raw = readGroup(0)
  // Merge repeats, keeping the order of first appearance.
  const merged = new Map()
  for (const [el, n] of raw) merged.set(el, (merged.get(el) ?? 0) + n)
  const pairs = [...merged].filter(([, n]) => n > 0)
  if (!pairs.length) throw new Error('The formula has no atoms in it.')
  return pairs
}

// Writes [symbol, amount] pairs back as a formula string.
export function formatFormula(pairs) {
  return pairs.map(([el, n]) => el + (n === 1 ? '' : String(Number(n.toFixed(4))))).join('')
}
