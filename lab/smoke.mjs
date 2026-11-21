// npm run lab:smoke: make_splits (does nothing if already sealed), baselines, a tiny evolution (N=6, K=4,
// 2 generations after generation 0, run seed 1) into a fresh temporary archive outside the repo, then
// check_isolation. It also checks the run is deterministic: the new lineage must equal the committed
// lab/archive_smoke/lineage.jsonl apart from timestamps and training times.
import './guard.mjs'
import path from 'node:path'
import os from 'node:os'
import { mkdtempSync, readFileSync, existsSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { LAB } from './guard.mjs'

const run = (script, args = []) => {
  console.log(`\n$ node lab/${script} ${args.join(' ')}`)
  const r = spawnSync('nice', ['-n', '10', process.execPath, path.join(LAB, script), ...args], { stdio: 'inherit' })
  if (r.status !== 0) { console.error(`smoke: ${script} failed`); process.exit(1) }
}
const tmp = mkdtempSync(path.join(os.tmpdir(), 'lab-smoke-'))
run('make_splits.mjs')
run('baselines.mjs')
run('evolve.mjs', ['--archive', tmp, '--seed', '1', '--N', '6', '--K', '4', '--generations', '2'])
run('check_isolation.mjs')

const strip = (file) => readFileSync(file, 'utf8').trim().split('\n').map((l) => { const r = JSON.parse(l); delete r.created; delete r.trainSeconds; return JSON.stringify(r) })
const fresh = strip(path.join(tmp, 'lineage.jsonl'))
const committed = path.join(LAB, 'archive_smoke', 'lineage.jsonl')
if (existsSync(committed)) {
  const same = JSON.stringify(fresh) === JSON.stringify(strip(committed))
  console.log(`\n${same ? 'ok  ' : 'FAIL'} fresh smoke lineage equals lab/archive_smoke/lineage.jsonl (deterministic for run seed 1)`)
  if (!same) process.exit(1)
}
const inds = fresh.map((l) => JSON.parse(l)).filter((r) => r.type === 'individual')
const best = inds.reduce((a, b) => (b.reward > a.reward ? b : a))
console.log(`smoke: ${inds.length} individuals, best reward ${best.reward.toFixed(3)} (${best.id}: SELECT MAE ${best.selectMae.toFixed(3)} MPa, violations ${best.violations.toFixed(3)}); archive at ${tmp}`)
