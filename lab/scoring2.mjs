// Phase-2 scoring of pre-registered models on any labelled set (FUTURE at the opening, MONITOR in the dry run,
// REWARD and MONITOR in baselines2.mjs). Every model file is re-hashed before use; a mismatch throws.
// Model kinds:
//   'seeds'    gzipped { replicas: [weights...] }: one network per seed; metrics per seed, their median, and the
//              mean prediction of all seeds ('ensemble' row)
//   'archived' an archive individual (its 2 seeds); scored as the mean of its seeds
//   'ensemble' members = archived individuals; prediction = mean of the members' predictions
//   'sklearn'  lab/sk_models.py pickle;  'physics'  a calibrated generator from physics/synthetic.mjs
//   'curve'    per-mix age curve: s28_mix * exp(s(p_SCM) (1 - sqrt(28/t))), s = 0.20 + 0.35 p_SCM, s28_mix fitted on
//              that mix's TRAIN rows; a mix with no TRAIN rows falls back to the TRAIN-calibrated fib generator
// Bootstrap (pre-registered): 2,000 resamples of mix groups g with replacement, seeded(909); 95% percentile
// interval = sorted values at 0-based positions floor(0.025 B) and ceil(0.975 B) - 1.
import './guard.mjs'
import path from 'node:path'
import { readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { gunzipSync } from 'node:zlib'
import { LAB } from './guard.mjs'
import { fromWeights } from './net.mjs'
import { powersRaw, fibRaw, fibS } from './physics/synthetic.mjs'
import { seeded } from './net.mjs'

const sha256 = (b) => createHash('sha256').update(b).digest('hex')
export function checkedFile(file, expect, gz) {
  const bytes = readFileSync(path.join(LAB, file))
  const body = gz ? gunzipSync(bytes) : bytes
  if (sha256(body) !== expect) throw new Error(`${file}: SHA-256 ${sha256(body)} does not match the registered ${expect}`)
  return body
}
const meanOf = (P) => P[0].map((_, i) => P.reduce((s, p) => s + p[i], 0) / P.length)
const replicaPreds = (file, sha, X) => JSON.parse(checkedFile(file, sha, true).toString()).replicas.map((w) => fromWeights(w).predictMany(X))

// Returns one prediction array per seed (a single array for single models).
export function predictions(m, X) {
  if (m.kind === 'seeds') return replicaPreds(m.file, m.sha256, X)
  if (m.kind === 'archived') return [meanOf(replicaPreds(m.file, m.sha256, X))]
  if (m.kind === 'ensemble') return [meanOf(m.members.map((mm) => meanOf(replicaPreds(mm.file, mm.sha256, X))))]
  if (m.kind === 'physics') {
    const p = JSON.parse(checkedFile(m.file, m.sha256, false).toString())
    return [X.map((x) => p.scale * (p.generator === 'powers' ? powersRaw(x, x[7]) : fibRaw(x, x[7])))]
  }
  if (m.kind === 'curve') {
    const c = JSON.parse(checkedFile(m.file, m.sha256, false).toString())
    return [X.map((x) => { const k = x.slice(0, 7).join(','); return k in c.perMixS28 ? c.perMixS28[k] * Math.exp(fibS(x) * (1 - Math.sqrt(28 / x[7]))) : c.fallbackScale * fibRaw(x, x[7]) })]
  }
  if (m.kind === 'sklearn') {
    checkedFile(m.file, m.sha256, false)
    const r = spawnSync('python3', ['-I', path.join(LAB, 'sk_models.py'), 'predict', m.name, m.sha256], { input: JSON.stringify({ X }), encoding: 'utf8', maxBuffer: 1 << 26 })
    if (r.status !== 0) throw new Error(`sk_models.py predict ${m.name} failed: ${r.stderr}`)
    return [JSON.parse(r.stdout)]
  }
  throw new Error(`unknown model kind ${m.kind}`)
}

export function metrics(p, y, idx) {
  if (!idx.length) return { rows: 0 }
  const ys = idx.map((i) => y[i]), mean = ys.reduce((s, v) => s + v, 0) / ys.length
  let abs = 0, sq = 0, tot = 0
  idx.forEach((i, k) => { const e = p[i] - ys[k]; abs += Math.abs(e); sq += e * e; tot += (ys[k] - mean) ** 2 })
  return { rows: idx.length, rmse: Math.sqrt(sq / idx.length), mae: abs / idx.length, r2: tot > 0 ? 1 - sq / tot : null }
}
const median = (v) => { const s = [...v].sort((a, b) => a - b); return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2 }

// Mix-grouped bootstrap (by g), fixed resamples so every model and every paired difference uses the same ones.
export const BOOTSTRAP = { resamples: 2000, seed: 909, unit: 'mix group g' }
export function groupResamples(g, B = BOOTSTRAP.resamples, seed = BOOTSTRAP.seed) {
  const groups = [...new Set(g)], rowsOf = new Map(groups.map((x) => [x, []]))
  g.forEach((x, i) => rowsOf.get(x).push(i))
  const rng = seeded(seed)
  return Array.from({ length: B }, () => Array.from({ length: groups.length }, () => rowsOf.get(groups[Math.floor(rng() * groups.length)])).flat())
}
const rmseIdx = (p, y, idx) => Math.sqrt(idx.reduce((s, i) => s + (p[i] - y[i]) ** 2, 0) / idx.length)
const ci = (v) => { const s = [...v].sort((a, b) => a - b); return [s[Math.floor(0.025 * s.length)], s[Math.ceil(0.975 * s.length) - 1]] }

// set = { X, y, g }; groups = named row-index lists (seen / unseen mixes, age buckets ...).
export function scoreAll(models, set, groups, { championRole = null } = {}) {
  const all = set.y.map((_, i) => i)
  const res = groups && groups.resamples ? groups.resamples : groupResamples(set.g)
  const out = [], point = new Map()
  for (const m of models) {
    const P = predictions(m, set.X), pm = P.length > 1 ? meanOf(P) : P[0]
    point.set(m.role, pm)
    const perSeed = P.length > 1 ? P.map((p) => metrics(p, set.y, all)) : null
    const r = {
      role: m.role, kind: m.kind, file: m.file || null, sha256: m.sha256 || null,
      overall: metrics(pm, set.y, all),
      medianOverSeeds: perSeed ? { seeds: perSeed.length, rmse: median(perSeed.map((x) => x.rmse)), mae: median(perSeed.map((x) => x.mae)), rmseRange: [Math.min(...perSeed.map((x) => x.rmse)), Math.max(...perSeed.map((x) => x.rmse))] } : null,
      rmseCi95: ci(res.map((idx) => rmseIdx(pm, set.y, idx))),
      subsets: Object.fromEntries(Object.entries(groups || {}).filter(([k]) => k !== 'resamples').map(([k, idx]) => [k, metrics(pm, set.y, idx)])),
      negativePredictions: pm.filter((v) => v < 0).length,
    }
    out.push(r)
  }
  if (championRole && point.has(championRole)) {
    const pc = point.get(championRole)
    for (const r of out) {
      if (r.role === championRole) continue
      const pb = point.get(r.role), d = res.map((idx) => rmseIdx(pc, set.y, idx) - rmseIdx(pb, set.y, idx))
      r.championMinusThis = { rmse: rmseIdx(pc, set.y, all) - rmseIdx(pb, set.y, all), ci95: ci(d) }
    }
  }
  return out
}
