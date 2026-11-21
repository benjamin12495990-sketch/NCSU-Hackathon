// THE ONE-TIME OPENING (pre-registration format 3). The real opening has never been run.
//   node lab/open_vault.mjs             the real opening: once, at the very end. Afterwards COMMIT
//                                       lab/results/OPENING_STARTED.json, vault_opening.json and VAULT_OPENED at once.
//   node lab/open_vault.mjs --dry-run   every step below on DECOY vaults built from MONITOR rows in a temporary
//                                       folder, with guard.mjs loaded first so lab/vault/ cannot be read, plus the
//                                       refusal cases (second run, crash after start, changed code)
// Steps: (0) if OPENING_STARTED.json, the result or the marker exists, the vault counts as opened: refuse and print
// whatever exists; (1) the pre-registration must be format 3 and name the vault hash in lab/seal_receipt.json;
// (2) the code manifest (codehash.mjs: SHA-256 of every lab file in the import closure, the data tiers, and the
// Node / Python / scikit-learn / NumPy / SciPy versions) must equal the registered one; (3) every model file must
// match its hash; (4) write OPENING_STARTED.json (time, pre-registration hash, code hash) BEFORE reading the vault;
// (5) read the vault and check its seal; (6) score every model once; (7) write the result as soon as scoring ends,
// then the marker. Files in lab/results/ are written with flag 'wx' (never overwritten).
// For the real opening the vault is read before any guarded lab module is imported, so the static imports here are
// node builtins and codehash.mjs (builtins only, no guard).
import { readFileSync, existsSync, mkdtempSync, mkdirSync, writeFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { gunzipSync } from 'node:zlib'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { codeManifest, compareManifest } from './codehash.mjs'

const DRY = process.argv.includes('--dry-run')
if (DRY) await import('./guard.mjs') // from here on, any read of lab/vault/ throws
const LAB = path.dirname(fileURLToPath(import.meta.url))
const sha256 = (b) => createHash('sha256').update(b).digest('hex')
const readJ = (p) => JSON.parse(readFileSync(p, 'utf8'))
const arg = (name, dflt) => { const i = process.argv.indexOf('--' + name); return i > 0 ? process.argv[i + 1] : dflt }
const PREREG = path.resolve(arg('prereg', path.join(LAB, 'preregistration.json')))
class Refusal extends Error {}
const refuse = (msg) => { throw new Refusal(msg) }
const writeOnce = (p, text) => { mkdirSync(path.dirname(p), { recursive: true }); writeFileSync(p, text, { flag: 'wx' }) }

function printExisting(started, result, marker) {
  if (existsSync(started)) console.log(`  OPENING_STARTED.json: ${readFileSync(started, 'utf8').trim()}`)
  if (existsSync(marker)) console.log(`  marker: ${readFileSync(marker, 'utf8').trim()}`)
  if (existsSync(result)) {
    const r = readJ(result)
    for (const m of r.results) console.log(`  ${m.role}: RMSE ${m.overall.rmse.toFixed(3)} MPa`)
    console.log(`  primary: ${JSON.stringify(r.primary.summary)}`)
  } else console.log('  (no result file: the opening started but did not finish; by protocol it is not repeated)')
}

async function openOnce({ vaultDir, resultsDir, registeredCode = null, label }) {
  const STARTED = path.join(resultsDir, 'OPENING_STARTED.json'), RESULT = path.join(resultsDir, 'vault_opening.json'), MARKER = path.join(resultsDir, 'VAULT_OPENED')
  // (0)
  if (existsSync(STARTED) || existsSync(RESULT) || existsSync(MARKER)) {
    console.log(`[${label}] refused: the vault counts as opened. What exists:`)
    printExisting(STARTED, RESULT, MARKER)
    return 'refused-opened'
  }
  // (1)
  if (!existsSync(PREREG)) refuse(`no pre-registration at ${PREREG}`)
  const preregText = readFileSync(PREREG), prereg = JSON.parse(preregText)
  if (prereg.formatVersion !== 3) refuse('not a format-3 pre-registration')
  const receipt = readJ(path.join(LAB, 'seal_receipt.json'))
  if (prereg.vault.sha256 !== receipt.sha256) refuse('the pre-registration names a different vault hash than lab/seal_receipt.json')
  // (2)
  const current = codeManifest(), diffs = compareManifest(registeredCode || prereg.code, current)
  if (diffs.length) { console.log(`[${label}] refused: code or versions differ from the pre-registration:\n  ${diffs.join('\n  ')}`); return 'refused-code' }
  // (3)
  const fileHash = (file, gz) => { const b = readFileSync(path.join(LAB, file)); return sha256(gz ? gunzipSync(b) : b) }
  for (const m of prereg.models) {
    const list = m.kind === 'ensemble' ? m.members.map((x) => ({ ...x, gz: true })) : [{ file: m.file, sha256: m.sha256, gz: m.kind === 'seeds' || m.kind === 'archived' }]
    for (const x of list) if (fileHash(x.file, x.gz) !== x.sha256) refuse(`${m.role}: ${x.file} does not match its pre-registered hash`)
  }
  // (4)
  const startedAt = new Date().toISOString()
  writeOnce(STARTED, JSON.stringify({ startedAt, preregistrationSha256: sha256(preregText), codeSha256: current.codeSha256, dryRun: DRY }) + '\n')
  // (5)
  const seal = readJ(path.join(vaultDir, 'SEALED.json'))
  const futureText = readFileSync(path.join(vaultDir, 'future.json'))
  if (sha256(futureText) !== seal.sha256) refuse('the vault file does not match its seal')
  if (!DRY && seal.sha256 !== receipt.sha256) refuse('SEALED.json and lab/seal_receipt.json disagree')
  const future = JSON.parse(futureText)
  if (future.y.length !== seal.rows) refuse('row count does not match the seal')
  // (6) Only now load the guarded lab modules (for the real opening this activates guard.mjs; FUTURE is in memory).
  const { scoreAll, predictions, BOOTSTRAP } = await import('./scoring2.mjs')
  const { makeScorer } = await import('./reward2.mjs')
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
  // Pre-registered criterion: best = lowest point RMSE on all FUTURE rows among "against"; success = upper end of
  // the 95% percentile interval of RMSE(champion) - RMSE(best) below 0. Secondary: the same against every model.
  const against = results.filter((r) => prereg.primary.against.includes(r.role)).sort((a, b) => a.overall.rmse - b.overall.rmse)
  const best = against[0]
  const success = best.championMinusThis.ci95[1] < 0
  const summary = { champion: prereg.primary.championRole, best: best.role, difference: best.championMinusThis.rmse, ci95: best.championMinusThis.ci95, success }
  const secondary = Object.fromEntries(against.map((r) => [r.role, { rmse: r.overall.rmse, championMinusThis: r.championMinusThis }]))
  const out = { dryRun: DRY, startedAt, finishedAt: new Date().toISOString(), preregistrationSha256: sha256(preregText), codeSha256: current.codeSha256, vaultSha256: seal.sha256, rows: future.y.length, bootstrap: BOOTSTRAP, primary: { summary, secondary, criterion: prereg.primary.successCriterion }, results }
  // (7) the result as soon as scoring ends, then the marker
  writeOnce(RESULT, JSON.stringify(out, null, 1) + '\n')
  writeOnce(MARKER, `opened ${out.finishedAt} with pre-registration sha256 ${out.preregistrationSha256}${DRY ? ' (DRY RUN on a decoy)' : ''}\n`)
  console.log(`[${label}] opened ${future.y.length} rows:`)
  for (const r of results) console.log(`  ${r.role.padEnd(26)} RMSE ${r.overall.rmse.toFixed(3)} [${r.rmseCi95.map((v) => v.toFixed(2)).join(', ')}]${r.medianOverSeeds ? `  median of 10 seeds ${r.medianOverSeeds.rmse.toFixed(3)}` : ''}  seen ${r.subsets.seenMix.rmse?.toFixed(3)} (${r.subsets.seenMix.rows})  unseen ${r.subsets.unseenMix.rows ? r.subsets.unseenMix.rmse.toFixed(3) : '-'} (${r.subsets.unseenMix.rows})`)
  console.log(`  primary: ${summary.champion} - ${summary.best} = ${summary.difference.toFixed(3)} MPa, 95% CI [${summary.ci95.map((v) => v.toFixed(3)).join(', ')}], success ${success}`)
  for (const [role, s] of Object.entries(secondary)) if (role !== best.role) console.log(`  secondary: ${summary.champion} - ${role} = ${s.championMinusThis.rmse.toFixed(3)} [${s.championMinusThis.ci95.map((v) => v.toFixed(3)).join(', ')}]`)
  console.log(`  wrote ${RESULT}, then the marker${DRY ? '' : '. Commit lab/results/OPENING_STARTED.json, vault_opening.json and VAULT_OPENED now.'}`)
  return 'opened'
}

function buildDecoy(tmp, withVault = true) {
  const vaultDir = path.join(tmp, 'vault'), resultsDir = path.join(tmp, 'results')
  if (withVault) {
    const M = readJ(path.join(LAB, 'data', 'monitor.json')), T = readJ(path.join(LAB, 'data', 'train.json')), R = readJ(path.join(LAB, 'data', 'reward.json'))
    const seenMix = new Set([...T.mix, ...R.mix]), seenG = new Set([...T.g, ...R.g]), text = JSON.stringify(M)
    const idx = (f) => M.y.map((_, i) => i).filter(f)
    writeOnce(path.join(vaultDir, 'future.json'), text)
    writeOnce(path.join(vaultDir, 'SEALED.json'), JSON.stringify({ decoy: true, sha256: sha256(text), rows: M.y.length, seenMixRows: idx((i) => seenMix.has(M.mix[i])), unseenMixRows: idx((i) => !seenMix.has(M.mix[i])), seenGroupRows: idx((i) => seenG.has(M.g[i])), unseenGroupRows: idx((i) => !seenG.has(M.g[i])) }))
  }
  return { vaultDir, resultsDir }
}

try {
  if (!DRY) {
    const status = await openOnce({ vaultDir: path.join(LAB, 'vault'), resultsDir: path.join(LAB, 'results'), label: 'real opening' })
    process.exit(status === 'opened' || status === 'refused-opened' ? 0 : 1)
  }
  const tmp = mkdtempSync(path.join(os.tmpdir(), 'lab-vault-dry-run-'))
  console.log(`dry run in ${tmp} (decoy vaults of MONITOR rows; the real vault stays unreadable)\n`)
  const checks = []
  const A = buildDecoy(path.join(tmp, 'A'))
  checks.push(['A: full opening on a decoy', await openOnce({ ...A, label: 'A' }) === 'opened'])
  checks.push(['A: OPENING_STARTED.json, result and marker written', ['OPENING_STARTED.json', 'vault_opening.json', 'VAULT_OPENED'].every((f) => existsSync(path.join(A.resultsDir, f)))])
  checks.push(['B: a second run on the same results folder is refused and prints the stored result', await openOnce({ ...A, label: 'B' }) === 'refused-opened'])
  const C = buildDecoy(path.join(tmp, 'C'), false) // no vault at all: a refusal must come before any vault read
  writeOnce(path.join(C.resultsDir, 'OPENING_STARTED.json'), JSON.stringify({ startedAt: 'simulated crash after step 4' }) + '\n')
  checks.push(['C: OPENING_STARTED.json without a result (crash) is refused without reading any vault', await openOnce({ ...C, label: 'C' }) === 'refused-opened'])
  const D = buildDecoy(path.join(tmp, 'D'))
  const prereg = readJ(PREREG), doctored = { ...prereg.code, files: { ...prereg.code.files, 'net.mjs': '0'.repeat(64) } }
  checks.push(['D: a changed code file is refused before OPENING_STARTED.json is written', await openOnce({ ...D, registeredCode: doctored, label: 'D' }) === 'refused-code' && !existsSync(path.join(D.resultsDir, 'OPENING_STARTED.json'))])
  let blocked = false
  try { readFileSync(path.join(LAB, 'vault', 'future.json')) } catch (e) { blocked = /sealed/.test(String(e.message)) }
  checks.push(['E: reading lab/vault/future.json throws during the dry run (guard active)', blocked])
  console.log('\ndry-run checks:')
  for (const [name, ok] of checks) console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}`)
  process.exit(checks.every(([, ok]) => ok) ? 0 : 1)
} catch (e) {
  if (e instanceof Refusal) { console.error(`open_vault${DRY ? ' --dry-run' : ''}: refused: ${e.message}`); process.exit(1) }
  throw e
}
