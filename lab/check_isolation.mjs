// Proves the vault and the archive are protected. Runtime part: with guard.mjs loaded, every fs read
// function throws on a lab/vault/ path, the full dataset files are refused, and an existing archive file
// cannot be overwritten, deleted, renamed or truncated. Static part: scans every code file in lab/.
import './guard.mjs'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import path from 'node:path'
import { LAB, FULL_DATA } from './guard.mjs'

let failed = 0
const ok = (cond, msg) => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${msg}`); if (!cond) failed++ }
const throws = (fn) => { try { fn(); return false } catch { return true } }
const rejects = async (fn) => { try { await fn(); return false } catch { return true } }

// --- runtime: reading the vault ---
const vaultDir = path.join(LAB, 'vault'), sealedFile = path.join(vaultDir, 'future.json')
ok(throws(() => readFileSync(sealedFile)), 'readFileSync on the sealed file throws')
ok(throws(() => fs.readFileSync(path.relative(process.cwd(), sealedFile))), 'readFileSync with a relative path throws')
ok(throws(() => fs.readFileSync(new URL('file://' + sealedFile))), 'readFileSync with a file: URL throws')
ok(throws(() => fs.readFile(sealedFile, () => {})), 'readFile throws')
ok(throws(() => fs.createReadStream(sealedFile)), 'createReadStream throws')
ok(throws(() => fs.openSync(sealedFile, 'r')), 'openSync throws')
ok(throws(() => fs.readdirSync(vaultDir)), 'readdirSync on the vault folder throws')
ok(await rejects(() => fs.promises.readFile(sealedFile)), 'promises.readFile rejects')
ok(await rejects(() => fsp.readFile(sealedFile)), 'node:fs/promises readFile rejects')
ok(throws(() => fs.copyFileSync(sealedFile, path.join(LAB, 'copy.json'))), 'copyFileSync out of the vault throws')
for (const f of FULL_DATA) ok(throws(() => readFileSync(f)), `full dataset ${path.relative(path.dirname(LAB), f)} is refused`)
ok(!throws(() => readFileSync(path.join(LAB, 'data', 'train.json'))), 'lab/data/train.json can be read')

// --- runtime: the archive is append-only ---
const tmpArchive = path.join(LAB, 'archive_isolation_check')
fs.mkdirSync(tmpArchive, { recursive: true })
const probe = path.join(tmpArchive, 'probe.txt')
if (!fs.existsSync(probe)) fs.writeFileSync(probe, 'written once\n', { flag: 'wx' })
ok(throws(() => fs.writeFileSync(probe, 'x')), 'writeFileSync over an existing archive file throws')
ok(throws(() => fs.unlinkSync(probe)), 'unlinkSync on an archive file throws')
ok(throws(() => fs.rmSync(tmpArchive, { recursive: true })), 'rmSync on an archive folder throws')
ok(throws(() => fs.renameSync(probe, probe + '.moved')), 'renameSync of an archive file throws')
ok(throws(() => fs.truncateSync(probe)), 'truncateSync on an archive file throws')
ok(throws(() => fs.openSync(probe, 'w')), 'openSync(..., "w") on an existing archive file throws')
ok(!throws(() => fs.appendFileSync(path.join(tmpArchive, 'append.log'), '.')), 'appendFileSync to an archive file is allowed')

// --- static scan of every code file in lab/ ---
const walk = (d) => readdirSync(d).flatMap((f) => {
  const p = path.join(d, f)
  if (['vault', 'node_modules', 'data', 'results'].includes(f) || f.startsWith('archive')) return []
  return statSync(p).isDirectory() ? walk(p) : [p]
})
const code = walk(LAB).filter((p) => /\.(mjs|js|cjs|py|sh)$/.test(p))
const VAULT_OK = new Set(['open_vault.mjs', 'make_splits.mjs', 'guard.mjs', 'check_isolation.mjs']) // guard blocks it, this file tests the block
const FULL_OK = new Set(['make_splits.mjs', 'net_parity.mjs', 'allow_full_data.mjs', 'guard.mjs', 'check_isolation.mjs'])
const vaultRe = /['"`/\\]vault(['"`/\\]|$)|\bVAULT_DIR\b|SEALED\.json|future\.json/m
const fullRe = /nn_data\.json|known\.json|test_points\.json|parity\.json|concrete\.csv/
const destroyRe = /\b(unlinkSync|unlink|rmSync|rmdirSync|rmdir|renameSync|rename|truncateSync|truncate|ftruncateSync)\s*\(|\.rm\s*\(|\bshutil\.rmtree|\bos\.(remove|unlink|rename|replace)\b/
for (const p of code) {
  const name = path.relative(LAB, p), text = readFileSync(p, 'utf8')
  if (!VAULT_OK.has(name)) ok(!vaultRe.test(text), `${name}: does not mention the vault`)
  if (!FULL_OK.has(name)) ok(!fullRe.test(text), `${name}: does not mention the full dataset files`)
  if (name !== 'check_isolation.mjs') ok(!destroyRe.test(text), `${name}: no unlink, rm, rename or truncate`)
  if (name !== 'guard.mjs' && name !== 'io.mjs' && name !== 'make_splits.mjs' && name !== 'check_isolation.mjs') {
    // Everything else must write through io.mjs (writeNew / writeSame) or append; no raw overwrite.
    ok(!/\bwriteFile(Sync)?\s*\(|createWriteStream\s*\(|open\(.*['"]w/.test(text), `${name}: writes only through io.mjs or appendFileSync`)
  }
  if (name !== 'guard.mjs' && name !== 'check_isolation.mjs' && name !== 'make_splits.mjs' && name !== 'open_vault.mjs' && /\.mjs$/.test(name)) {
    const firstImport = text.match(/^import .*$/m)
    ok(firstImport && /['"](\.\.?\/)+guard\.mjs['"]/.test(firstImport[0]) || (name === 'net_parity.mjs' && /^import '\.\/allow_full_data\.mjs'\nimport '\.\/guard\.mjs'/m.test(text)) || name === 'allow_full_data.mjs', `${name}: imports guard.mjs first`)
  }
}
console.log(`\n${failed ? 'FAILED' : 'PASSED'}: ${failed} failure(s)`)
process.exit(failed ? 1 : 0)
