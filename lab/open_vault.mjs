// THE ONE-TIME OPENING (phase 2). The real opening has never been run.
//   node lab/open_vault.mjs             the real opening: once, at the very end
//   node lab/open_vault.mjs --dry-run   the whole opening path on a DECOY vault built from MONITOR rows in a fresh
//                                       temporary folder; loads guard.mjs first, so it cannot read lab/vault/
// Order: (0) if a stored result or the marker exists, print the stored result and stop; (1) the pre-registration
// must exist and name the vault hash in lab/seal_receipt.json; (2) every listed model file must match its hash;
// (3) the vault (or decoy) must match its seal; (4) score each model once (lab/scoring2.mjs); (5) save the result;
// (6) only then write the marker. A crash before (5) leaves nothing behind, so it cannot lock the vault; once (5)
// is done, every later run only prints the result.
// For the real opening the vault is read before any lab module is imported (they load guard.mjs), so the static
// imports here are node builtins only.
import { readFileSync, existsSync, mkdtempSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { gunzipSync } from 'node:zlib'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const DRY = process.argv.includes('--dry-run')
if (DRY) await import('./guard.mjs') // from here on, any read of lab/vault/ throws
const LAB = path.dirname(fileURLToPath(import.meta.url))
const sha256 = (b) => createHash('sha256').update(b).digest('hex')
const readJ = (p) => JSON.parse(readFileSync(p, 'utf8'))
const die = (msg) => { console.error(`open_vault${DRY ? ' --dry-run' : ''}: ${msg}`); process.exit(1) }
const arg = (name, dflt) => { const i = process.argv.indexOf('--' + name); return i > 0 ? process.argv[i + 1] : dflt }
const PREREG = path.resolve(arg('prereg', path.join(LAB, 'preregistration.json')))

let vaultDir, resultsDir
if (DRY) {
  // Decoy: MONITOR rows play FUTURE; "seen" = exact mix tested in TRAIN or REWARD.
  const { writeNew } = await import('./io.mjs')
  const tmp = mkdtempSync(path.join(os.tmpdir(), 'lab-vault-dry-run-'))
  vaultDir = path.join(tmp, 'vault'); resultsDir = path.join(tmp, 'results')
  const M = readJ(path.join(LAB, 'data', 'monitor.json')), T = readJ(path.join(LAB, 'data', 'train.json')), R = readJ(path.join(LAB, 'data', 'reward.json'))
  const seenMix = new Set([...T.mix, ...R.mix]), seenG = new Set([...T.g, ...R.g])
  const text = JSON.stringify(M)
  const idx = (f) => M.y.map((_, i) => i).filter(f)
  writeNew(path.join(vaultDir, 'future.json'), text)
  writeNew(path.join(vaultDir, 'SEALED.json'), JSON.stringify({ decoy: true, sha256: sha256(text), rows: M.y.length, seenMixRows: idx((i) => seenMix.has(M.mix[i])), unseenMixRows: idx((i) => !seenMix.has(M.mix[i])), seenGroupRows: idx((i) => seenG.has(M.g[i])), unseenGroupRows: idx((i) => !seenG.has(M.g[i])) }))
  console.log(`dry run: decoy vault of ${M.y.length} MONITOR rows in ${tmp}`)
} else {
  vaultDir = path.join(LAB, 'vault'); resultsDir = path.join(LAB, 'results')
}
const RESULT = path.join(resultsDir, 'vault_opening.json'), MARKER = path.join(resultsDir, 'VAULT_OPENED')

function printStored() {
  console.log(`open_vault: the vault was already opened${existsSync(MARKER) ? ` (${readFileSync(MARKER, 'utf8').trim()})` : ''}. It is never scored twice. Stored result:`)
  const r = readJ(RESULT)
  for (const m of r.results) console.log(`  ${m.role}: RMSE ${m.overall.rmse.toFixed(3)} MPa`)
  console.log(`  success criterion met: ${r.primary.success}`)
}
// (0)
if (existsSync(RESULT) || existsSync(MARKER)) {
  if (!existsSync(RESULT)) die('a marker exists without a result; by protocol nothing is repeated')
  printStored(); process.exit(0)
}
// (1)
if (!existsSync(PREREG)) die(`no pre-registration at ${PREREG}; run lab/preregister.mjs first`)
const preregText = readFileSync(PREREG), prereg = JSON.parse(preregText)
const receipt = readJ(path.join(LAB, 'seal_receipt.json'))
if (prereg.version !== 2 || !prereg.models || !prereg.models.length) die('not a phase-2 pre-registration')
if (prereg.vault.sha256 !== receipt.sha256) die('the pre-registration names a different vault hash than lab/seal_receipt.json')
// (2)
const fileHash = (file, gz) => { const b = readFileSync(path.join(LAB, file)); return sha256(gz ? gunzipSync(b) : b) }
for (const m of prereg.models) {
  const list = m.kind === 'ensemble' ? m.members.map((x) => ({ ...x, gz: true })) : [{ file: m.file, sha256: m.sha256, gz: m.kind === 'seeds' || m.kind === 'archived' }]
  for (const x of list) if (fileHash(x.file, x.gz) !== x.sha256) die(`${m.role}: ${x.file} does not match its pre-registered hash`)
}
// (3)
const seal = readJ(path.join(vaultDir, 'SEALED.json'))
const futureText = readFileSync(path.join(vaultDir, 'future.json'))
if (sha256(futureText) !== seal.sha256) die('the vault file does not match its seal')
if (!DRY && seal.sha256 !== receipt.sha256) die('SEALED.json and lab/seal_receipt.json disagree')
const future = JSON.parse(futureText)
if (future.y.length !== seal.rows) die('row count does not match the seal')

// (4) Only now load the lab modules (for the real opening this activates guard.mjs; FUTURE is already in memory).
const { writeNew } = await import('./io.mjs')
const { scoreAll } = await import('./scoring2.mjs')
const { makeScorer } = await import('./reward2.mjs')
const { predictions } = await import('./scoring2.mjs')
const age = (lo, hi) => future.y.map((_, i) => i).filter((i) => future.X[i][7] >= lo && future.X[i][7] <= hi)
const groups = {
  seenMix: seal.seenMixRows, unseenMix: seal.unseenMixRows, seenGroup: seal.seenGroupRows, unseenGroup: seal.unseenGroupRows,
  'age 56 d': age(56, 56), 'age 90-100 d': age(90, 100), 'age 120-180 d': age(120, 180), 'age 270-365 d': age(270, 365),
}
const results = scoreAll(prereg.models, future, groups, { championRole: prereg.primary.championRole })
const grid = makeScorer(future, { X: future.X.slice(0, 2), y: future.y.slice(0, 2), g: [0, 1] }, { X: future.X.slice(0, 2), y: future.y.slice(0, 2) })
for (const r of results) {
  const m = prereg.models.find((x) => x.role === r.role)
  const v = grid.score([(rows) => { const P = predictions(m, rows); return P[0].map((_, k) => P.reduce((s, p) => s + p[k], 0) / P.length) }])
  r.violationsOnFutureMixGrid = { vAge: v.vAge, vWb: v.vWb, rate: v.violationRate }
}
const against = results.filter((r) => prereg.primary.against.includes(r.role)).sort((a, b) => a.overall.rmse - b.overall.rmse)
const champ = results.find((r) => r.role === prereg.primary.championRole), bestBase = against[0]
const success = !!(champ && bestBase && bestBase.championMinusThis && bestBase.championMinusThis.ci95[1] < 0)
const out = {
  dryRun: DRY, openedAt: new Date().toISOString(), preregistrationSha256: sha256(preregText), vaultSha256: seal.sha256, rows: future.y.length,
  primary: { champion: prereg.primary.championRole, bestBaseline: bestBase && bestBase.role, difference: bestBase && bestBase.championMinusThis, success, criterion: prereg.primary.successCriterion },
  results,
}
// (5) save, (6) then mark
writeNew(RESULT, JSON.stringify(out, null, 1) + '\n')
writeNew(MARKER, `opened ${out.openedAt} with pre-registration sha256 ${out.preregistrationSha256}${DRY ? ' (DRY RUN on a decoy)' : ''}\n`)
for (const r of results) console.log(`${r.role.padEnd(24)} RMSE ${r.overall.rmse.toFixed(3)} [${r.rmseCi95.map((v) => v.toFixed(2)).join(', ')}]${r.medianOverSeeds ? ` median of 10 seeds ${r.medianOverSeeds.rmse.toFixed(3)}` : ''}  seen ${r.subsets.seenMix.rmse?.toFixed(3)} (${r.subsets.seenMix.rows})  unseen ${r.subsets.unseenMix.rows ? r.subsets.unseenMix.rmse.toFixed(3) : '-'} (${r.subsets.unseenMix.rows})`)
console.log(`primary: ${out.primary.champion} minus best of baselines (${out.primary.bestBaseline}) = ${bestBase.championMinusThis.rmse.toFixed(3)} MPa, 95% CI [${bestBase.championMinusThis.ci95.map((v) => v.toFixed(3)).join(', ')}]: success ${success}`)
console.log(`open_vault${DRY ? ' --dry-run' : ''}: wrote ${RESULT} and then the marker.`)
if (DRY) {
  // Prove the two remaining properties: a second run only prints, and the real vault stayed unreadable.
  console.log('\ndry run, second invocation of step (0):'); printStored()
  let blocked = false
  try { readFileSync(path.join(LAB, 'vault', 'future.json')) } catch (e) { blocked = /sealed/.test(String(e.message)) }
  console.log(`dry run: reading lab/vault/future.json now ${blocked ? 'throws (guard active): the real vault was never read' : 'DID NOT THROW'}`)
  if (!blocked) process.exit(1)
}
