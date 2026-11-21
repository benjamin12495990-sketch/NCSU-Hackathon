// Scores a list of pre-registered models on one set of rows. Used by open_vault.mjs (on FUTURE, once) and by
// rehearse_scoring.mjs (on SELECT, any time, to test this code without the vault). Every model file is
// re-hashed before it is used; a mismatch throws.
import './guard.mjs'
import path from 'node:path'
import { readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { gunzipSync } from 'node:zlib'
import { LAB } from './guard.mjs'
import { fromWeights } from './net.mjs'
import { curvePredict } from './curve.mjs'

const sha256 = (b) => createHash('sha256').update(b).digest('hex')

// Returns the model's bytes after checking them against the pre-registered SHA-256.
export function checkedModel(m) {
  const bytes = readFileSync(path.join(LAB, m.file))
  const body = m.kind === 'network' ? gunzipSync(bytes) : bytes
  const h = sha256(body)
  if (h !== m.sha256) throw new Error(`${m.role}: ${m.file} hash ${h} does not match the pre-registered ${m.sha256}`)
  return body
}

export function predictWith(m, X) {
  const body = checkedModel(m)
  if (m.kind === 'network') return fromWeights(JSON.parse(body.toString())).predictMany(X)
  if (m.kind === 'curve') return curvePredict(JSON.parse(body.toString()), m.perMix)(X)
  if (m.kind === 'trees') {
    const r = spawnSync('python3', ['-I', path.join(LAB, 'trees.py'), 'predict', m.sha256], { input: JSON.stringify({ X }), encoding: 'utf8', maxBuffer: 1 << 26 })
    if (r.status !== 0) throw new Error(`trees prediction failed: ${r.stderr}`)
    return JSON.parse(r.stdout)
  }
  throw new Error(`unknown model kind ${m.kind}`)
}

export function metrics(p, y, idx) {
  if (!idx.length) return { rows: 0 }
  const ys = idx.map((i) => y[i]), mean = ys.reduce((s, v) => s + v, 0) / ys.length
  let abs = 0, sq = 0, tot = 0
  idx.forEach((i, k) => { const e = p[i] - ys[k]; abs += Math.abs(e); sq += e * e; tot += (ys[k] - mean) ** 2 })
  return { rows: idx.length, mae: abs / idx.length, rmse: Math.sqrt(sq / idx.length), r2: tot > 0 ? 1 - sq / tot : null }
}

// set = { X, y }; split = { seenMixRows, unseenMixRows, seenGroupRows, unseenGroupRows } (row indices into set).
export function scoreModels(models, set, split) {
  const all = set.y.map((_, i) => i)
  const band = (lo, hi) => all.filter((i) => set.X[i][7] > lo && set.X[i][7] <= hi)
  return models.map((m) => {
    const p = predictWith(m, set.X)
    return {
      role: m.role, id: m.id || null, file: m.file, sha256: m.sha256,
      overall: metrics(p, set.y, all),
      seenMix: metrics(p, set.y, split.seenMixRows), unseenMix: metrics(p, set.y, split.unseenMixRows),
      seenGroup: metrics(p, set.y, split.seenGroupRows), unseenGroup: metrics(p, set.y, split.unseenGroupRows),
      byAge: { 'up to 28 d': metrics(p, set.y, band(0, 28)), '29-56 d': metrics(p, set.y, band(28, 56)), '57-100 d': metrics(p, set.y, band(56, 100)), '101-365 d': metrics(p, set.y, band(100, 365)) },
      negativePredictions: p.filter((v) => v < 0).length,
    }
  })
}
