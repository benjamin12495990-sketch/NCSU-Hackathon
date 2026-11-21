// Phase-2 reward, following the lab's methods research (section 7.5), constants fixed before any run:
//   rmse      = RMSE on REWARD, pooled over the individual's 2 seeds (root of the mean squared error over rows and seeds)
//   se        = bootstrap standard error of rmse: 200 resamples of REWARD mix groups, fixed for the whole run
//   V_age     = mean over query pairs of ReLU(f(t) - f(2t)), MPa: strength must not fall with age
//   V_wb      = mean over query pairs of ReLU(f(w/b + 0.05) - f(w/b)), MPa: more water for the same binder must not help
//   fitness   = rmse + se + LAMBDA * (V_age + V_wb)   (lower is better; the "upper confidence bound" used to rank parents)
//   reward    = -(rmse + LAMBDA * (V_age + V_wb))      (reported)
//   eligible  = violation rate <= 5%: the share of query pairs that break either rule by more than 0.01 MPa.
//               An ineligible individual stays in the archive but never has children.
// LAMBDA = 1 MPa per MPa. Query points are unlabeled: every distinct TRAIN mix crossed with ages
// t = 1, 2, 3, 5, 7, 14, 28, 56, 90, 180 (paired with 2t) and, for w/b, ages 1, 3, 7, 14, 28, 56, 90, 180, 365.
// Violations use the mean prediction of the 2 seeds. No term uses any labelled row older than 28 days.
import './guard.mjs'
import { seeded } from './net.mjs'

export const LAMBDA = 1
export const MAX_VIOLATION_RATE = 0.05
const TOL = 0.01
const AGE_PAIRS = [1, 2, 3, 5, 7, 14, 28, 56, 90, 180]
const WB_AGES = [1, 3, 7, 14, 28, 56, 90, 180, 365]

export function makeScorer(TRAIN, REWARD, MONITOR) {
  const mixes = [...new Map(TRAIN.X.map((x) => [x.slice(0, 7).join(','), x])).values()]
  const ageRows = [], wbRows = []
  for (const m of mixes) {
    for (const t of AGE_PAIRS) { const a = [...m], b = [...m]; a[7] = t; b[7] = 2 * t; ageRows.push(a, b) }
    for (const t of WB_AGES) {
      const binder = m[0] + m[1] + m[2], wb = m[3] / binder
      const a = [...m], b = [...m]; a[7] = t; b[7] = t
      b[3] = (wb + 0.05) * binder; b[8] = wb + 0.05
      wbRows.push(a, b)
    }
  }
  // Fixed bootstrap resamples of REWARD mix groups (the same for every individual, so comparisons are paired).
  const groups = [...new Set(REWARD.g)], rowsOf = new Map(groups.map((g) => [g, []]))
  REWARD.g.forEach((g, i) => rowsOf.get(g).push(i))
  const rng = seeded(4242)
  const resamples = Array.from({ length: 200 }, () => Array.from({ length: groups.length }, () => rowsOf.get(groups[Math.floor(rng() * groups.length)])).flat())
  const rmseOf = (rowSE, idx) => Math.sqrt(idx.reduce((s, i) => s + rowSE[i], 0) / idx.length)
  const sd = (v) => { const m = v.reduce((s, x) => s + x, 0) / v.length; return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - 1)) }

  // predictors: one predictMany function per seed.
  function score(predictors) {
    const pr = predictors.map((f) => f(REWARD.X)), pm = predictors.map((f) => f(MONITOR.X))
    const rowSE = REWARD.y.map((y, i) => pr.reduce((s, p) => s + (p[i] - y) ** 2, 0) / pr.length)
    const all = REWARD.y.map((_, i) => i)
    const rmse = rmseOf(rowSE, all)
    const se = sd(resamples.map((idx) => rmseOf(rowSE, idx)))
    const pooled = (P, set) => {
      let sq = 0, ab = 0
      for (const p of P) set.y.forEach((y, i) => { sq += (p[i] - y) ** 2; ab += Math.abs(p[i] - y) })
      return { rmse: Math.sqrt(sq / (P.length * set.y.length)), mae: ab / (P.length * set.y.length) }
    }
    const mean = (rows) => { const P = predictors.map((f) => f(rows)); return rows.map((_, i) => P.reduce((s, p) => s + p[i], 0) / P.length) }
    const pa = mean(ageRows), pw = mean(wbRows)
    let vAge = 0, vWb = 0, bad = 0, neg = 0
    for (let k = 0; k < pa.length; k += 2) { const d = pa[k] - pa[k + 1]; vAge += Math.max(0, d); if (d > TOL) bad++ }
    for (let k = 0; k < pw.length; k += 2) { const d = pw[k + 1] - pw[k]; vWb += Math.max(0, d); if (d > TOL) bad++ }
    for (const v of [...pa, ...pw]) if (v < 0) neg++
    vAge /= pa.length / 2; vWb /= pw.length / 2
    const rate = bad / ((pa.length + pw.length) / 2)
    const r = pooled(pr, REWARD), m = pooled(pm, MONITOR)
    return {
      rewardRmse: rmse, rewardMae: r.mae, rewardSe: se, monitorRmse: m.rmse, monitorMae: m.mae,
      vAge, vWb, violationRate: rate, negativeShare: neg / (pa.length + pw.length),
      fitness: rmse + se + LAMBDA * (vAge + vWb), reward: -(rmse + LAMBDA * (vAge + vWb)), eligible: rate <= MAX_VIOLATION_RATE,
      rowSE: rowSE.map((v) => Number(v.toFixed(4))),
    }
  }
  // Paired bootstrap SE of the RMSE difference between two individuals (from their stored rowSE).
  const pairedSe = (a, b) => sd(resamples.map((idx) => rmseOf(a, idx) - rmseOf(b, idx)))
  return { score, pairedSe, queryPairs: (ageRows.length + wbRows.length) / 2, mixes: mixes.length }
}
