// reward = -(MAE on SELECT, MPa) - LAMBDA * violations, where
//   violations = share of TRAIN mixes whose predicted strength falls anywhere along the ages
//                1, 3, 7, 14, 28, 56, 90, 180, 365 d (each step must not drop by more than 1e-6 MPa)
//              + share of all those sweep predictions (mixes x 9 ages) that are below zero.
// LAMBDA = 2 MPa per unit share: a network that breaks the age rule on every training mix pays 2 MPa, the
// same as being 2 MPa less accurate; one that breaks it on 10% of mixes pays 0.2 MPa. Both parts are reported.
import './guard.mjs'

export const LAMBDA = 2
export const SWEEP_AGES = [1, 3, 7, 14, 28, 56, 90, 180, 365]
const iAge = 7, iWb = 8

// One row per distinct TRAIN mix (its seven amounts), with age blank. water_binder_ratio does not depend on age.
export function trainMixes(train) {
  const seen = new Map()
  train.X.forEach((x, i) => { if (!seen.has(train.mix[i])) seen.set(train.mix[i], x) })
  return [...seen.values()]
}

export function physicsViolations(predictMany, mixes) {
  let falling = 0, negative = 0, total = 0
  const rows = []
  for (const m of mixes) for (const t of SWEEP_AGES) { const r = [...m]; r[iAge] = t; r[iWb] = m[iWb]; rows.push(r) }
  const p = predictMany(rows)
  for (let k = 0; k < mixes.length; k++) {
    let falls = false
    for (let a = 0; a < SWEEP_AGES.length; a++) {
      const v = p[k * SWEEP_AGES.length + a]
      total++
      if (v < 0) negative++
      if (a > 0 && v < p[k * SWEEP_AGES.length + a - 1] - 1e-6) falls = true
    }
    if (falls) falling++
  }
  return { fallingShare: falling / mixes.length, negativeShare: negative / total, mixes: mixes.length, sweepPredictions: total }
}

export function maeOn(predictMany, set) {
  const p = predictMany(set.X)
  let abs = 0, sq = 0
  const m = set.y.reduce((s, v) => s + v, 0) / set.y.length
  let tot = 0
  set.y.forEach((y, i) => { abs += Math.abs(p[i] - y); sq += (p[i] - y) ** 2; tot += (y - m) ** 2 })
  return { mae: abs / set.y.length, r2: 1 - sq / tot }
}

// Everything the archive stores about one model's score.
export function reward(predictMany, train, select, lambda = LAMBDA) {
  const { mae, r2 } = maeOn(predictMany, select)
  const v = physicsViolations(predictMany, trainMixes(train))
  const violations = v.fallingShare + v.negativeShare
  return { reward: -mae - lambda * violations, selectMae: mae, selectR2: r2, violations, fallingShare: v.fallingShare, negativeShare: v.negativeShare, lambda }
}
