// Imported first by every lab script except open_vault.mjs (and make_splits.mjs, which writes the vault).
// It wraps node:fs so that (1) opening any path inside lab/vault/ throws, (2) the full dataset files that
// contain the FUTURE rows (public/model/nn_data.json and friends) cannot be read unless a script set
// globalThis.LAB_ALLOW_FULL_DATA before importing this file (only net_parity.mjs does), and (3) nothing
// can delete, rename, truncate or overwrite an existing file under lab/archive*/ (appending is allowed).
import fs from 'node:fs'
import path from 'node:path'
import { syncBuiltinESMExports } from 'node:module'
import { fileURLToPath } from 'node:url'

const LAB = path.dirname(fileURLToPath(import.meta.url))
const VAULT = path.join(LAB, 'vault') + path.sep
const REPO = path.dirname(LAB)
// Files outside lab/ that hold rows older than 28 days (the FUTURE tier).
export const FULL_DATA = ['public/model/nn_data.json', 'public/model/known.json', 'public/model/test_points.json', 'public/model/parity.json', 'train/data/concrete.csv'].map((p) => path.join(REPO, p))

function toPath(p) {
  if (typeof p === 'number') return null // a file descriptor was opened earlier, through this guard
  if (p instanceof URL) return fileURLToPath(p)
  if (Buffer.isBuffer(p)) return p.toString()
  return typeof p === 'string' ? p : null
}
function real(p) {
  const abs = path.resolve(p)
  try { return fs.realpathSync.native(abs) } catch { return abs }
}
export function isVaultPath(p) {
  const s = toPath(p)
  if (s === null) return false
  const abs = path.resolve(s), r = real(s)
  return abs.startsWith(VAULT) || r.startsWith(VAULT) || abs === VAULT.slice(0, -1) || r === VAULT.slice(0, -1)
}
function isFullData(p) {
  const s = toPath(p)
  if (s === null) return false
  const r = real(s), abs = path.resolve(s)
  return FULL_DATA.includes(r) || FULL_DATA.includes(abs)
}
function isArchivePath(p) {
  const s = toPath(p)
  if (s === null) return false
  const rel = path.relative(LAB, path.resolve(s))
  return /^archive[^/]*\//.test(rel)
}
function checkRead(p, fn) {
  if (isVaultPath(p)) throw new Error(`guard: ${fn}(${toPath(p)}) refused: lab/vault/ is sealed until open_vault.mjs`)
  if (isFullData(p) && !globalThis.LAB_ALLOW_FULL_DATA) throw new Error(`guard: ${fn}(${toPath(p)}) refused: this file contains FUTURE rows; read lab/data/ instead`)
}
function checkWrite(p, fn, flags) {
  if (isVaultPath(p)) throw new Error(`guard: ${fn}(${toPath(p)}) refused: lab/vault/ is sealed`)
  if (!isArchivePath(p)) return
  const appendOnly = typeof flags === 'string' && flags.startsWith('a')
  const exists = fs.existsSync(toPath(p))
  if (exists && !appendOnly) throw new Error(`guard: ${fn}(${toPath(p)}) refused: archive files are never overwritten or deleted`)
}
const flagOf = (o) => (typeof o === 'string' ? undefined : o && o.flag)
const isWriteFlag = (f) => typeof f === 'string' && /[wa+]/.test(f)

const wrap = (obj, name, check) => {
  const orig = obj[name]
  if (typeof orig !== 'function' || orig.__labGuarded) return
  const w = function (...args) { check(args, name); return orig.apply(this, args) }
  w.__labGuarded = true
  obj[name] = w
}
const readCheck = (args, name) => checkRead(args[0], name)
for (const name of ['readFileSync', 'readFile', 'createReadStream', 'readdirSync', 'readdir', 'opendirSync', 'copyFileSync', 'copyFile', 'cpSync']) wrap(fs, name, readCheck)
for (const name of ['readFile', 'readdir', 'opendir', 'copyFile', 'cp']) wrap(fs.promises, name, readCheck)
const openCheck = (args, name) => {
  checkRead(args[0], name)
  const f = args[1]
  if (isWriteFlag(f)) checkWrite(args[0], name, f)
}
wrap(fs, 'openSync', openCheck); wrap(fs, 'open', openCheck); wrap(fs.promises, 'open', openCheck)
const writeCheck = (args, name) => checkWrite(args[0], name, flagOf(args[2]) || 'w')
for (const name of ['writeFileSync', 'writeFile']) wrap(fs, name, writeCheck)
wrap(fs.promises, 'writeFile', writeCheck)
wrap(fs, 'createWriteStream', (args, name) => checkWrite(args[0], name, flagOf(args[1]) || 'w'))
const appendCheck = (args, name) => checkWrite(args[0], name, 'a')
for (const name of ['appendFileSync', 'appendFile']) wrap(fs, name, appendCheck)
wrap(fs.promises, 'appendFile', appendCheck)
// Deleting, renaming onto, or truncating anything under lab/archive*/ or lab/vault/ is refused outright.
const destroyCheck = (args, name) => {
  for (const a of name.startsWith('rename') ? [args[0], args[1]] : [args[0]]) {
    if (isVaultPath(a) || isArchivePath(a)) throw new Error(`guard: ${name}(${toPath(a)}) refused: archive and vault files are never deleted, moved or truncated`)
  }
}
for (const name of ['unlinkSync', 'unlink', 'rmSync', 'rm', 'rmdirSync', 'rmdir', 'renameSync', 'rename', 'truncateSync', 'truncate']) wrap(fs, name, destroyCheck)
for (const name of ['unlink', 'rm', 'rmdir', 'rename', 'truncate']) wrap(fs.promises, name, destroyCheck)
syncBuiltinESMExports() // so `import { readFileSync } from 'node:fs'` in later modules gets the wrapped function

export const GUARD_ACTIVE = true
export { LAB, REPO, VAULT }
