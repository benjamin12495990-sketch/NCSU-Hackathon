// Run once, after evolution and before the vault is opened. Names exactly which models open_vault.mjs will
// score on FUTURE, with each model's SHA-256 and the sealed hash of FUTURE (read from lab/seal_receipt.json, so
// nothing sealed is opened), and writes lab/preregistration.json. It refuses to overwrite an existing file.
//   node lab/preregister.mjs [--archive lab/archive] [--out lab/preregistration.json] [--dry-run]
// Models: the champion by reward; the best individual without 'physics' inputs; the best individual that did
// not inherit weights (trained from its own seed); and every baseline in lab/results/baselines_select.json.
import './guard.mjs'
import path from 'node:path'
import { readFileSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { LAB } from './guard.mjs'
import { readJson, sha256, writeNew } from './io.mjs'
import { openArchive } from './archive.mjs'
import { checkedModel } from './scoring.mjs'

const arg = (name, dflt) => { const i = process.argv.indexOf('--' + name); return i > 0 ? process.argv[i + 1] : dflt }
const archiveDir = path.resolve(arg('archive', path.join(LAB, 'archive')))
const outFile = path.resolve(arg('out', path.join(LAB, 'preregistration.json')))
const A = openArchive(archiveDir)
const all = A.individuals()
if (!all.length) { console.error(`preregister: no individuals in ${archiveDir}`); process.exit(1) }
const best = (list) => list.reduce((a, b) => (b.reward > a.reward ? b : a), list[0])
const picks = [
  ['champion', 'highest reward in the archive', all],
  ['best_without_physics_inputs', "highest reward among genomes with features 'base' or 'rich'", all.filter((r) => r.genome.features !== 'physics')],
  ['best_without_inherited_weights', 'highest reward among individuals trained from their own seed (inherited: false)', all.filter((r) => !r.inherited)],
]
const rel = (p) => path.relative(LAB, p)
const models = []
for (const [role, rule, list] of picks) {
  if (!list.length) { console.log(`preregister: no individual for ${role}; skipped`); continue }
  const r = best(list)
  models.push({ role, rule, kind: 'network', id: r.id, file: rel(path.join(archiveDir, 'weights', `${r.id}.json.gz`)), sha256: r.weightsSha256, hash: 'sha256 of the uncompressed weights JSON', genome: r.genome, inherited: r.inherited, generation: r.generation, select: { reward: r.reward, selectMae: r.selectMae, violations: r.violations } })
}
const baselines = readJson(path.join(LAB, 'results', 'baselines_select.json'))
for (const [name, b] of Object.entries(baselines.models)) {
  models.push({ role: `baseline_${name}`, rule: 'baseline from lab/baselines.mjs', kind: b.kind, perMix: b.perMix, file: b.file, sha256: b.sha256, hash: b.kind === 'network' ? 'sha256 of the uncompressed weights JSON' : 'sha256 of the file', select: { reward: b.reward, selectMae: b.selectMae, violations: b.violations } })
}
for (const m of models) checkedModel(m) // every file must still match the hash it is registered under
const receipt = readJson(path.join(LAB, 'seal_receipt.json'))
const lineageText = readFileSync(A.lineage)
const git = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: LAB, encoding: 'utf8' })
const prereg = {
  created: new Date().toISOString(),
  commit: git.status === 0 ? git.stdout.trim() : null,
  vault: { sha256: receipt.sha256, rows: receipt.rows, sealed: receipt.created },
  archive: { dir: rel(archiveDir), individuals: all.length, lineageBytes: lineageText.length, lineageSha256: sha256(lineageText) },
  plan: 'Each model below is scored exactly once on FUTURE (age > 28 d) by lab/open_vault.mjs: MAE, RMSE and R2 overall, on rows whose exact mix was tested in TRAIN or SELECT (seen) and on the others (unseen), the same by mix group, and by age band. No model is retrained, tuned or added after this file exists.',
  models,
}
const text = JSON.stringify(prereg, null, 1) + '\n'
if (process.argv.includes('--dry-run')) { console.log(text); process.exit(0) }
writeNew(outFile, text)
console.log(`preregister: wrote ${outFile} with ${models.length} models; FUTURE sha256 ${receipt.sha256.slice(0, 16)}...`)
for (const m of models) console.log(`  ${m.role}: ${m.id || m.file} (SELECT reward ${m.select.reward.toFixed(3)}, MAE ${m.select.selectMae.toFixed(3)})`)
