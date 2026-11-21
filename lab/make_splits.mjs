// Splits the 1,005 lab results by curing age into three tiers and seals the oldest one.
// TRAIN = age <= 7 d, SELECT = 14-28 d, FUTURE = age > 28 d. FUTURE goes to lab/vault/ with its SHA-256;
// the other tiers to lab/data/. Run once: a second run only checks that the seal still matches.
// This script and open_vault.mjs are the only ones that touch the vault, so it does not import guard.mjs.
import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const LAB = path.dirname(fileURLToPath(import.meta.url))
const VAULT_DIR = path.join(LAB, 'vault')
const FUTURE = path.join(VAULT_DIR, 'future.json'), SEALED = path.join(VAULT_DIR, 'SEALED.json')
const RECEIPT = path.join(LAB, 'seal_receipt.json')
const sha256 = (s) => createHash('sha256').update(s).digest('hex')

if (existsSync(SEALED)) {
  const seal = JSON.parse(readFileSync(SEALED, 'utf8'))
  const now = sha256(readFileSync(FUTURE))
  if (now !== seal.sha256) { console.error(`make_splits: vault hash changed! sealed ${seal.sha256}, now ${now}`); process.exit(1) }
  console.log(`make_splits: already sealed at ${seal.created} (${seal.rows} rows, sha256 ${seal.sha256.slice(0, 16)}...). Nothing written.`)
  process.exit(0)
}

const nn = JSON.parse(readFileSync(path.join(LAB, '..', 'public', 'model', 'nn_data.json'), 'utf8'))
const iAge = nn.features.indexOf('age')
// A mix is its seven ingredient amounts (kg/m3); the same mix is usually tested at several ages.
const mixKey = (x) => x.slice(0, 7).join(',')
const all = []
for (const part of ['train', 'test']) nn[part].X.forEach((x, i) => all.push({ x, y: nn[part].y[i], g: nn[part].g[i], pagePart: part }))
const tier = (r) => (r.x[iAge] <= 7 ? 'train' : r.x[iAge] <= 28 ? 'select' : 'future')
const pack = (rows) => ({ features: nn.features, n: rows.length, X: rows.map((r) => r.x), y: rows.map((r) => r.y), g: rows.map((r) => r.g), mix: rows.map((r) => mixKey(r.x)) })
const tiers = { train: [], select: [], future: [] }
for (const r of all) tiers[tier(r)].push(r)

const seenMix = new Set([...tiers.train, ...tiers.select].map((r) => mixKey(r.x)))
const seenGroup = new Set([...tiers.train, ...tiers.select].map((r) => r.g))
const future = pack(tiers.future)
future.seenMix = tiers.future.map((r) => seenMix.has(mixKey(r.x))) // same seven amounts tested in TRAIN or SELECT
future.seenGroup = tiers.future.map((r) => seenGroup.has(r.g)) // same mix group id (near-identical recipes)

mkdirSync(path.join(LAB, 'data'), { recursive: true }); mkdirSync(VAULT_DIR, { recursive: true })
const futureText = JSON.stringify(future)
writeFileSync(path.join(LAB, 'data', 'train.json'), JSON.stringify(pack(tiers.train)), { flag: 'wx' })
writeFileSync(path.join(LAB, 'data', 'select.json'), JSON.stringify(pack(tiers.select)), { flag: 'wx' })
writeFileSync(FUTURE, futureText, { flag: 'wx' })
const idx = (flags, want) => flags.map((f, i) => (f === want ? i : -1)).filter((i) => i >= 0)
const seal = {
  file: 'future.json', sha256: sha256(futureText), rows: future.n, created: new Date().toISOString(),
  rule: 'FUTURE = age > 28 d; TRAIN = age <= 7 d; SELECT = 14-28 d; source public/model/nn_data.json train+test',
  // Row indices only (no strengths), so the seen/unseen split of the final score is fixed now.
  seenMixRows: idx(future.seenMix, true), unseenMixRows: idx(future.seenMix, false),
  seenGroupRows: idx(future.seenGroup, true), unseenGroupRows: idx(future.seenGroup, false),
}
writeFileSync(SEALED, JSON.stringify(seal, null, 1), { flag: 'wx' })
// The receipt is a copy of the seal kept outside the vault, so preregister.mjs can name the hash without opening the vault.
writeFileSync(RECEIPT, JSON.stringify(seal, null, 1), { flag: 'wx' })
console.log(`TRAIN ${tiers.train.length}  SELECT ${tiers.select.length}  FUTURE ${future.n} (sealed, sha256 ${seal.sha256.slice(0, 16)}...)`)
console.log(`FUTURE rows whose exact mix is in TRAIN or SELECT: ${seal.seenMixRows.length}; not: ${seal.unseenMixRows.length}`)
console.log(`FUTURE rows whose mix group is in TRAIN or SELECT: ${seal.seenGroupRows.length}; not: ${seal.unseenGroupRows.length}`)
