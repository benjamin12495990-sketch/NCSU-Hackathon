import { parseFormula, formatFormula } from './parseFormula.js'
import { featurize } from './featurize.js'
import { predict } from './predict.js'
import { findKnown, nearest } from './nearest.js'
import { generate } from './generate.js'

// The whole pipeline for one typed formula: parse, predict, look it up in
// the dataset, and generate ranked candidates. Throws on bad input.
export function analyze(text, data, limit = 10) {
  const { elements, model, known, metrics } = data
  const pairs = parseFormula(text, elements)
  const x = featurize(pairs, elements)
  const near = nearest(known, x, elements.length)
  return {
    formula: formatFormula(pairs),
    tc: predict(model, x),
    known: findKnown(known, x, elements.length),
    nearestFormula: near.entry.formula,
    nearestTc: near.entry.tc,
    lowConfidence: near.distance > metrics.low_confidence_distance,
    ...generate(pairs, data, limit),
  }
}
