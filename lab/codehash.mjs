// The code manifest that pins a pre-registration: the SHA-256 of every lab file that can affect a prediction or a
// score, found as the import closure of the scripts that train, score and open (static imports, dynamic
// import('...'), forked workers and the Python scripts they spawn), plus the data tiers, the BOxCrete file and the
// seal receipt; and the Node, Python, scikit-learn, NumPy and SciPy versions. codeSha256 is the SHA-256 of the
// canonical JSON of all of it.
// This module imports node builtins only and does NOT load guard.mjs, because open_vault.mjs runs it before it
// reads the vault (check_isolation.mjs verifies both properties). It reads lab code and data files only.
import { readFileSync, existsSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const LAB = path.dirname(fileURLToPath(import.meta.url))
const sha256 = (b) => createHash('sha256').update(b).digest('hex')
export const ROOTS = ['open_vault.mjs', 'scoring2.mjs', 'preregister.mjs', 'baselines2.mjs', 'evolve2.mjs', 'evo2_worker.mjs', 'split_select.mjs', 'make_splits.mjs', 'codehash.mjs', 'sk_models.py', 'trees.py']
export const DATA = ['data/train.json', 'data/select.json', 'data/reward.json', 'data/monitor.json', 'external/boxcrete_data.csv', 'seal_receipt.json']

function closure() {
  const seen = new Set(), todo = [...ROOTS]
  while (todo.length) {
    const rel = path.normalize(todo.pop())
    if (seen.has(rel)) continue
    if (!existsSync(path.join(LAB, rel))) throw new Error(`codehash: ${rel} is missing`)
    seen.add(rel)
    if (!rel.endsWith('.mjs')) continue
    const text = readFileSync(path.join(LAB, rel), 'utf8'), dir = path.dirname(rel)
    const refs = [...text.matchAll(/(?:from\s+|import\s*\(\s*|import\s+)['"](\.{1,2}\/[^'"]+)['"]/g)].map((m) => m[1])
    const spawned = [...text.matchAll(/['"]([\w./-]+\.(?:py|mjs))['"]/g)].map((m) => m[1]).filter((f) => !f.startsWith('.') && existsSync(path.join(LAB, dir, f)))
    for (const r of refs) { const p = path.normalize(path.join(dir, r)); if (p.startsWith('..')) throw new Error(`codehash: ${rel} imports ${r} from outside lab/`); todo.push(p) }
    for (const f of spawned) todo.push(path.normalize(path.join(dir, f)))
  }
  return [...seen].sort()
}

export function versions() {
  const r = spawnSync('python3', ['-I', '-c', 'import sys, sklearn, numpy, scipy; print(sys.version.split()[0], sklearn.__version__, numpy.__version__, scipy.__version__)'], { encoding: 'utf8' })
  const [python, sklearn, numpy, scipy] = r.status === 0 ? r.stdout.trim().split(' ') : [null, null, null, null]
  return { node: process.version, python, sklearn, numpy, scipy }
}

export function codeManifest() {
  const files = Object.fromEntries(closure().map((f) => [f, sha256(readFileSync(path.join(LAB, f)))]))
  const data = Object.fromEntries(DATA.map((f) => [f, sha256(readFileSync(path.join(LAB, f)))]))
  const m = { roots: ROOTS, files, data, versions: versions() }
  return { ...m, codeSha256: sha256(JSON.stringify(m)) }
}

// Lists every difference between a registered manifest and the current one (empty = identical).
export function compareManifest(registered, current) {
  const diffs = []
  for (const part of ['files', 'data', 'versions']) {
    const a = registered[part] || {}, b = current[part] || {}
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) if (a[k] !== b[k]) diffs.push(`${part}/${k}: registered ${a[k] ?? '(absent)'}, now ${b[k] ?? '(absent)'}`)
  }
  if (registered.codeSha256 !== current.codeSha256) diffs.push(`codeSha256: registered ${registered.codeSha256}, now ${current.codeSha256}`)
  return diffs
}
