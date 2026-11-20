import { featurize } from './featurize.js'
import { formatFormula } from './parseFormula.js'
import { predict } from './predict.js'
import { findKnown, nearest } from './nearest.js'

// "Chemically similar" = same periodic-table group, plus lanthanide for
// lanthanide. Noble gases and elements with no stable isotope (Tc, Pm, Po,
// At, Rn) are left out because nobody could synthesize those candidates.
export const SIMILAR_SETS = [
  ['Li', 'Na', 'K', 'Rb', 'Cs'],
  ['Be', 'Mg', 'Ca', 'Sr', 'Ba'],
  ['Sc', 'Y', 'La', 'Lu'],
  ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
  ['Ti', 'Zr', 'Hf'], ['V', 'Nb', 'Ta'], ['Cr', 'Mo', 'W'], ['Mn', 'Re'],
  ['Fe', 'Ru', 'Os'], ['Co', 'Rh', 'Ir'], ['Ni', 'Pd', 'Pt'], ['Cu', 'Ag', 'Au'],
  ['Zn', 'Cd', 'Hg'],
  ['B', 'Al', 'Ga', 'In', 'Tl'],
  ['C', 'Si', 'Ge', 'Sn', 'Pb'],
  ['N', 'P', 'As', 'Sb', 'Bi'],
  ['O', 'S', 'Se', 'Te'],
  ['F', 'Cl', 'Br', 'I'],
]

export function similarElements(el) {
  const out = new Set()
  for (const set of SIMILAR_SETS) if (set.includes(el)) for (const other of set) out.add(other)
  out.delete(el)
  return [...out]
}

// Makes one-swap candidates, drops any whose composition is already in the
// dataset, predicts Tc for the rest and returns the top `limit` with their
// nearest known material and a confidence flag.
export function generate(pairs, data, limit = 10) {
  const { elements, model, known, metrics } = data
  const n = elements.length
  const seen = new Set()
  const candidates = []
  let removedKnown = 0
  pairs.forEach(([el], pos) => {
    for (const sub of similarElements(el)) {
      // Swap one element, keep every number; merge if `sub` is already present.
      const merged = new Map()
      pairs.forEach(([e, amt], k) => {
        const name = k === pos ? sub : e
        merged.set(name, (merged.get(name) ?? 0) + amt)
      })
      const swapped = [...merged]
      const formula = formatFormula(swapped)
      if (seen.has(formula)) continue
      seen.add(formula)
      const x = featurize(swapped, elements)
      if (findKnown(known, x, n)) { removedKnown++; continue }
      candidates.push({ formula, swap: `${el} → ${sub}`, x, tc: predict(model, x) })
    }
  })
  candidates.sort((a, b) => b.tc - a.tc)
  const top = candidates.slice(0, limit).map((c) => {
    const near = nearest(known, c.x, n)
    return {
      formula: c.formula,
      swap: c.swap,
      elements: elements.filter((_, k) => c.x[k] > 0),
      tc: c.tc,
      nearestFormula: near.entry.formula,
      nearestTc: near.entry.tc,
      distance: near.distance,
      lowConfidence: near.distance > metrics.low_confidence_distance,
    }
  })
  return { top, generated: candidates.length + removedKnown, removedKnown }
}
