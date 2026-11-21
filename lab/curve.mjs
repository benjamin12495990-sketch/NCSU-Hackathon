// The no-learning age-curve baseline: strength(t) = s28 * f(t), f(t) = exp(0.25 * (1 - sqrt(28 / t))), the
// fib/CEB-FIP hardening curve for ordinary cement. fitCurve() fits s28 per mix (least squares on that mix's
// TRAIN rows) and once for all TRAIN rows; curvePredict() uses the mix's own s28 when it has one (perMix).
import './guard.mjs'

export const ageFactor = (t) => Math.exp(0.25 * (1 - Math.sqrt(28 / t)))
export const mixOf = (x) => x.slice(0, 7).join(',')
export function fitCurve(train) {
  const s28 = (idx) => idx.reduce((s, i) => s + train.y[i] * ageFactor(train.X[i][7]), 0) / idx.reduce((s, i) => s + ageFactor(train.X[i][7]) ** 2, 0)
  const byMix = new Map()
  train.X.forEach((x, i) => (byMix.get(mixOf(x)) || byMix.set(mixOf(x), []).get(mixOf(x))).push(i))
  return { format: 'lab-curve-1', curve: 'exp(0.25 * (1 - sqrt(28 / t)))', globalS28: s28(train.y.map((_, i) => i)), perMixS28: Object.fromEntries([...byMix].map(([m, idx]) => [m, s28(idx)])) }
}
export const curvePredict = (c, perMix) => (rows) => rows.map((x) => (perMix && mixOf(x) in c.perMixS28 ? c.perMixS28[mixOf(x)] : c.globalS28) * ageFactor(x[7]))
