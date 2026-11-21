// Splits SELECT (14-28 d, 481 rows) by mix group (g) into REWARD (about 2/3 of groups, the only rows the reward
// sees) and MONITOR (about 1/3, never used for selection; read only to detect over-use of REWARD and to stop).
// Deterministic (seeded shuffle of the group ids, seed 2027). Writes lab/data/reward.json and monitor.json once.
import './guard.mjs'
import path from 'node:path'
import { LAB } from './guard.mjs'
import { readJson, writeSame } from './io.mjs'
import { seeded } from './net.mjs'

const S = readJson(path.join(LAB, 'data', 'select.json'))
const groups = [...new Set(S.g)].sort((a, b) => a - b)
const rng = seeded(2027)
for (let i = groups.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [groups[i], groups[j]] = [groups[j], groups[i]] }
const rewardGroups = new Set(groups.slice(0, Math.round((2 / 3) * groups.length)))
const pick = (keep) => {
  const idx = S.y.map((_, i) => i).filter((i) => rewardGroups.has(S.g[i]) === keep)
  return { features: S.features, n: idx.length, X: idx.map((i) => S.X[i]), y: idx.map((i) => S.y[i]), g: idx.map((i) => S.g[i]), mix: idx.map((i) => S.mix[i]) }
}
const R = pick(true), M = pick(false)
writeSame(path.join(LAB, 'data', 'reward.json'), JSON.stringify(R))
writeSame(path.join(LAB, 'data', 'monitor.json'), JSON.stringify(M))
const ages = (d) => JSON.stringify(Object.fromEntries([14, 28].map((t) => [t, d.X.filter((x) => x[7] === t).length])))
console.log(`REWARD ${R.n} rows from ${new Set(R.g).size} mix groups (ages ${ages(R)}); MONITOR ${M.n} rows from ${new Set(M.g).size} groups (ages ${ages(M)})`)
