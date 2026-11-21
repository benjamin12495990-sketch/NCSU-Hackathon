// THE ONE-TIME OPENING. Written, never run so far. It refuses unless lab/preregistration.json exists and every
// hash matches: the vault's sealed SHA-256, the receipt, the pre-registration, and each listed model file. It
// then scores each listed model once on FUTURE (overall; rows whose exact mix was or was not tested in TRAIN or
// SELECT; by mix group; by age band), writes lab/results/vault_opening.json and the marker
// lab/results/VAULT_OPENED, and on any later run only prints the stored result. The marker is written BEFORE
// scoring, so a crash or a disliked result cannot be followed by a second attempt.
//   node lab/open_vault.mjs [--prereg lab/preregistration.json]
// It reads the vault before importing guard.mjs (scoring.mjs and io.mjs load it), so its static imports are
// node builtins only. The scoring code is tested without the vault by lab/rehearse_scoring.mjs.
import { readFileSync, existsSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { gunzipSync } from 'node:zlib'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const LAB = path.dirname(fileURLToPath(import.meta.url))
const arg = (name, dflt) => { const i = process.argv.indexOf('--' + name); return i > 0 ? process.argv[i + 1] : dflt }
const PREREG = path.resolve(arg('prereg', path.join(LAB, 'preregistration.json')))
const RESULT = path.join(LAB, 'results', 'vault_opening.json')
const MARKER = path.join(LAB, 'results', 'VAULT_OPENED')
const sha256 = (b) => createHash('sha256').update(b).digest('hex')
const die = (msg) => { console.error(`open_vault: ${msg}`); process.exit(1) }

// 0. Second run: print what is stored and stop. The vault is not read again.
if (existsSync(MARKER)) {
  console.log(`open_vault: the vault was already opened (${readFileSync(MARKER, 'utf8').trim()}). It is never scored twice. Stored result:`)
  console.log(existsSync(RESULT) ? readFileSync(RESULT, 'utf8') : '(no result file: the opening did not finish; by protocol it is not repeated)')
  process.exit(0)
}

// 1. Check every hash before anything is scored.
if (!existsSync(PREREG)) die(`no pre-registration at ${PREREG}. Run lab/preregister.mjs first.`)
const preregText = readFileSync(PREREG)
const prereg = JSON.parse(preregText)
if (!prereg.models || !prereg.models.length) die('the pre-registration lists no models')
const receipt = JSON.parse(readFileSync(path.join(LAB, 'seal_receipt.json'), 'utf8'))
if (receipt.sha256 !== prereg.vault.sha256) die('the pre-registration names a different vault hash than lab/seal_receipt.json')
for (const m of prereg.models) {
  const bytes = readFileSync(path.join(LAB, m.file))
  const h = sha256(m.kind === 'network' ? gunzipSync(bytes) : bytes)
  if (h !== m.sha256) die(`${m.role}: ${m.file} hash ${h} does not match the pre-registered ${m.sha256}`)
}
const VAULT_DIR = path.join(LAB, 'vault')
const seal = JSON.parse(readFileSync(path.join(VAULT_DIR, 'SEALED.json'), 'utf8'))
if (seal.sha256 !== receipt.sha256) die('SEALED.json and lab/seal_receipt.json disagree')
const futureText = readFileSync(path.join(VAULT_DIR, 'future.json'))
if (sha256(futureText) !== seal.sha256) die('future.json no longer matches its seal')
const future = JSON.parse(futureText)
if (future.y.length !== seal.rows) die('row count does not match the seal')

// 2. Only now load the lab modules (this activates guard.mjs; FUTURE is already in memory).
const { writeNew } = await import('./io.mjs')
const { scoreModels } = await import('./scoring.mjs')
const openedAt = new Date().toISOString()
writeNew(MARKER, `opened ${openedAt} with pre-registration sha256 ${sha256(preregText)}\n`)

// 3. Score each listed model once.
const results = scoreModels(prereg.models, future, seal)
const out = { openedAt, preregistrationSha256: sha256(preregText), vaultSha256: seal.sha256, rows: future.y.length, seenMixRows: seal.seenMixRows.length, unseenMixRows: seal.unseenMixRows.length, results }
writeNew(RESULT, JSON.stringify(out, null, 1) + '\n')
for (const r of results) console.log(`${r.role}: FUTURE MAE ${r.overall.mae.toFixed(3)} MPa (seen mixes ${r.seenMix.mae?.toFixed(3)}, unseen ${r.unseenMix.mae?.toFixed(3) ?? '-'}), R2 ${r.overall.r2.toFixed(3)}`)
console.log(`open_vault: wrote ${path.relative(LAB, RESULT)}. The vault is now spent; later runs only print this result.`)
