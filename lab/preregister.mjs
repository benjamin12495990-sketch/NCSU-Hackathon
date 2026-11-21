// Phase 2. Run once, after evolve2.mjs, evolve2.mjs --random and baselines2.mjs, before the vault is opened.
// Names every model open_vault.mjs will score on FUTURE (lab/results/baselines2.json: the physics equations, trees,
// forest, ridge, plain network x10 seeds, PANEL_DESIGN, plain + physics pre-training, champion retrained x10,
// champion without pre-training, the archived champion, the random-search champion, the top-10 ensemble) with
// their SHA-256, the sealed FUTURE hash (from lab/seal_receipt.json; the vault is not opened), both archives'
// lineage hashes, the metrics and the success criterion. Writes lab/preregistration.json once (never overwritten).
//   node lab/preregister.mjs [--out lab/preregistration.json] [--dry-run]
import './guard.mjs'
import path from 'node:path'
import { readFileSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { LAB } from './guard.mjs'
import { readJson, sha256, writeNew } from './io.mjs'
import { openArchive } from './archive.mjs'
import { checkedFile } from './scoring2.mjs'

const arg = (name, dflt) => { const i = process.argv.indexOf('--' + name); return i > 0 ? process.argv[i + 1] : dflt }
const outFile = path.resolve(arg('out', path.join(LAB, 'preregistration.json')))
const b2 = readJson(path.join(LAB, 'results', 'baselines2.json'))
const models = b2.models.map(({ reward, monitor, violations, ...m }) => ({ ...m, selectionSetScores: { rewardRmse: reward.overall.rmse, monitorRmse: monitor.overall.rmse, rewardMedianOverSeeds: reward.medianOverSeeds && reward.medianOverSeeds.rmse, monitorMedianOverSeeds: monitor.medianOverSeeds && monitor.medianOverSeeds.rmse } }))
for (const m of models) {
  if (m.kind === 'ensemble') for (const mm of m.members) checkedFile(mm.file, mm.sha256, true)
  else checkedFile(m.file, m.sha256, m.kind === 'seeds' || m.kind === 'archived')
}
const archive = (dir) => {
  const A = openArchive(path.join(LAB, dir)), text = readFileSync(A.lineage)
  return { dir, individuals: A.individuals().length, lineageBytes: text.length, lineageSha256: sha256(text), stop: A.records().find((r) => r.type === 'stop') }
}
const receipt = readJson(path.join(LAB, 'seal_receipt.json'))
const git = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: LAB, encoding: 'utf8' })
const prereg = {
  version: 2,
  created: new Date().toISOString(),
  commit: git.status === 0 ? git.stdout.trim() : null,
  vault: { sha256: receipt.sha256, rows: receipt.rows, sealed: receipt.created, seenMixRows: receipt.seenMixRows.length, unseenMixRows: receipt.unseenMixRows.length },
  data: Object.fromEntries(['train', 'reward', 'monitor'].map((n) => [n, sha256(readFileSync(path.join(LAB, 'data', `${n}.json`)))])),
  archives: { evolution: archive('archive'), randomSearch: archive('archive_random') },
  championRule: 'The evolution champion is the individual holding the Ladder title when evolve2.mjs stopped (eligible individuals only; a challenger must beat REWARD RMSE + 1 x (V_age + V_wb) of the champion by more than the paired bootstrap SE of their RMSE difference). champion_retrained is that genome trained from scratch with seed indices 0-9.',
  primary: {
    championRole: 'champion_retrained',
    against: ['physics_powers', 'physics_fib', 'gbt', 'rf', 'ridge', 'plain', 'random_champion'],
    successCriterion: 'Success if the RMSE on FUTURE of champion_retrained (mean prediction of its 10 seeds) minus the RMSE of the best model in "against" has a 95% mix-grouped paired bootstrap interval (1,000 resamples of FUTURE mix groups, seed 909) entirely below zero. Otherwise the headline is that evolution did not beat the best simple baseline.',
  },
  metrics: 'For every model, once: RMSE (primary), MAE and R2 on all FUTURE rows; for multi-seed models also the median and range over seeds; mix-grouped bootstrap 95% CI of RMSE; the same on rows whose exact mix was or was not tested in TRAIN or SELECT, by mix group, and by age bucket (56 d, 90-100 d, 120-180 d, 270-365 d); paired RMSE difference to champion_retrained with its CI; physics violations (V_age, V_wb, rate) on the query grid of FUTURE mixes (inputs only).',
  honestReporting: 'REWARD scores of evolved models are optimistic (they were selected on REWARD). Every result is published, including failures. The vault is opened exactly once.',
  models,
}
const text = JSON.stringify(prereg, null, 1) + '\n'
if (process.argv.includes('--dry-run')) { console.log(text); process.exit(0) }
writeNew(outFile, text)
console.log(`preregister: wrote ${path.relative(path.dirname(LAB), outFile)}, sha256 ${sha256(text)}; ${models.length} models; FUTURE sha256 ${receipt.sha256}`)
