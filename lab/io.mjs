// Small file helpers shared by the lab scripts. Nothing here deletes or overwrites: writeNew() refuses an
// existing path (flag 'wx'), writeSame() accepts an existing file only if its bytes are identical.
import './guard.mjs'
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'node:fs'
import { createHash } from 'node:crypto'
import path from 'node:path'

export const sha256 = (data) => createHash('sha256').update(data).digest('hex')
export const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'))
export function writeNew(p, data) {
  mkdirSync(path.dirname(p), { recursive: true })
  writeFileSync(p, data, { flag: 'wx' })
}
export function writeSame(p, data) {
  mkdirSync(path.dirname(p), { recursive: true })
  if (existsSync(p)) {
    const old = readFileSync(p)
    if (!old.equals(Buffer.from(data))) throw new Error(`${p} exists with different content; refusing to overwrite (choose a new name)`)
    return false
  }
  writeFileSync(p, data, { flag: 'wx' })
  return true
}
