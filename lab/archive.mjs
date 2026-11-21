// The archive: lineage.jsonl (append-only, one JSON line per individual, first line describes the run) and
// weights/<id>.json.gz. Nothing is ever deleted or overwritten: lines are only appended, and a weights file
// is written with flag 'wx' (or accepted if an identical one is already there, after a crash mid-write).
import './guard.mjs'
import { readFileSync, existsSync, appendFileSync, mkdirSync } from 'node:fs'
import { gzipSync, gunzipSync } from 'node:zlib'
import path from 'node:path'
import { sha256, writeSame } from './io.mjs'

export function openArchive(dir) {
  mkdirSync(path.join(dir, 'weights'), { recursive: true })
  const lineage = path.join(dir, 'lineage.jsonl')
  const read = () => (existsSync(lineage) ? readFileSync(lineage, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l)) : [])
  return {
    dir, lineage,
    records: read,
    header: () => read().find((r) => r.type === 'run') || null,
    individuals: () => read().filter((r) => r.type === 'individual'),
    append(record) { appendFileSync(lineage, JSON.stringify(record) + '\n') },
    // Returns the SHA-256 of the uncompressed weights JSON (the hash pre-registration names).
    saveWeights(id, weights) {
      const text = JSON.stringify(weights)
      writeSame(path.join(dir, 'weights', `${id}.json.gz`), gzipSync(text, { level: 9 }))
      return sha256(text)
    },
    loadWeights(id, expectSha) {
      const text = gunzipSync(readFileSync(path.join(dir, 'weights', `${id}.json.gz`))).toString()
      if (expectSha && sha256(text) !== expectSha) throw new Error(`weights ${id}: SHA-256 does not match lineage`)
      return JSON.parse(text)
    },
  }
}
