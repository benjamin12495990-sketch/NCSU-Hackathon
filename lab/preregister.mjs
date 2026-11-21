// Pre-registration format 3 (supersedes preregistration_v1.json, which is kept unchanged). Run after evolve2.mjs,
// evolve2.mjs --random and baselines2.mjs, with the code final, before the vault is opened. It records:
//  - every model to be scored on FUTURE (lab/results/baselines2_v2.json) with its SHA-256, and which are primary;
//  - the code manifest (codehash.mjs): SHA-256 of every lab file in the import closure of the training, scoring
//    and opening scripts, the data tiers, and the Node / Python / scikit-learn / NumPy / SciPy versions;
//  - the sealed FUTURE hash (from lab/seal_receipt.json; the vault is not opened), both lineage hashes;
//  - the exact success criterion, the metrics and the disclosures.
//   node lab/preregister.mjs [--dry-run]
import './guard.mjs'
import path from 'node:path'
import { readFileSync, existsSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { LAB } from './guard.mjs'
import { readJson, sha256, writeNew, replaceArchived } from './io.mjs'
import { openArchive } from './archive.mjs'
import { checkedFile, BOOTSTRAP } from './scoring2.mjs'
import { codeManifest } from './codehash.mjs'

const OUT = path.join(LAB, 'preregistration.json'), V1 = path.join(LAB, 'preregistration_v1.json')
const b2 = readJson(path.join(LAB, 'results', 'baselines2_v2.json'))
const AGAINST = ['physics_powers', 'physics_fib', 'curve_permix', 'gbt', 'rf', 'ridge', 'plain', 'panel_design', 'plain_physics_pretrain', 'random_champion_retrained']
const DESCRIPTIVE = ['champion_no_pretrain', 'champion', 'random_champion', 'top10_ensemble']
const models = b2.models.map(({ reward, monitor, violations, ...m }) => ({
  ...m, use: m.role === 'champion_retrained' ? 'primary champion' : AGAINST.includes(m.role) ? 'primary comparison ("against")' : 'descriptive only',
  selectionSetScores: { rewardRmse: reward.overall.rmse, monitorRmse: monitor.overall.rmse, rewardMedianOverSeeds: reward.medianOverSeeds && reward.medianOverSeeds.rmse, monitorMedianOverSeeds: monitor.medianOverSeeds && monitor.medianOverSeeds.rmse, violationRate: violations.rate },
}))
const roles = models.map((m) => m.role)
for (const r of ['champion_retrained', ...AGAINST, ...DESCRIPTIVE]) if (!roles.includes(r)) throw new Error(`missing model ${r}`)
for (const m of models) {
  if (m.kind === 'ensemble') for (const mm of m.members) checkedFile(mm.file, mm.sha256, true)
  else checkedFile(m.file, m.sha256, m.kind === 'seeds' || m.kind === 'archived')
}
const archive = (dir) => {
  const A = openArchive(path.join(LAB, dir)), text = readFileSync(A.lineage)
  return { dir, individuals: A.individuals().length, lineageBytes: text.length, lineageSha256: sha256(text), stop: A.records().find((r) => r.type === 'stop') }
}
const receipt = readJson(path.join(LAB, 'seal_receipt.json'))
const git = (args) => { const r = spawnSync('git', args, { cwd: LAB, encoding: 'utf8' }); return r.status === 0 ? r.stdout.trim() : null }
const prereg = {
  formatVersion: 3,
  registration: 2,
  supersedes: { file: 'preregistration_v1.json', sha256: sha256(readFileSync(V1)), why: 'an independent audit found the strongest baselines missing from the comparison, an unfair random-search comparison, unpinned code and missing disclosures' },
  created: new Date().toISOString(),
  commit: git(['rev-parse', 'HEAD']), workingTreeClean: git(['status', '--porcelain', '--', '.']) === '',
  vault: { sha256: receipt.sha256, rows: receipt.rows, sealed: receipt.created, seenMixRows: receipt.seenMixRows.length, unseenMixRows: receipt.unseenMixRows.length },
  code: codeManifest(),
  archives: { evolution: archive('archive'), randomSearch: archive('archive_random') },
  championRule: 'The evolution champion is the individual holding the Ladder title when evolve2.mjs stopped (eligible individuals only; a challenger must beat REWARD RMSE + 1 x (V_age + V_wb) of the champion by more than the paired bootstrap SE of their RMSE difference). champion_retrained is that genome trained from scratch with seed indices 0-9; its prediction is the mean of the 10 seeds.',
  primary: {
    championRole: 'champion_retrained',
    against: AGAINST,
    successCriterion: `"Best" is the model in "against" with the lowest point RMSE (mean prediction of its seeds, for multi-seed models) on all ${receipt.rows} FUTURE rows. The paired bootstrap resamples FUTURE mix groups g (near-identical recipes) with replacement, ${BOOTSTRAP.resamples} resamples, random stream seeded(${BOOTSTRAP.seed}) as in lab/scoring2.mjs; in each resample both models are scored on the same rows. The 95% interval is the percentile interval of RMSE(champion_retrained) - RMSE(best): the sorted values at 0-based positions floor(0.025 B) and ceil(0.975 B) - 1. Success if and only if its upper end is below 0. The same interval is reported against every model in "against" as secondary results. Otherwise the headline is that evolution did not beat the best baseline.`,
  },
  descriptiveOnly: DESCRIPTIVE,
  metrics: 'For every model, once: RMSE (primary), MAE and R2 on all FUTURE rows; for multi-seed models also the median and range over seeds; mix-group bootstrap 95% percentile interval of RMSE; the same on rows whose exact mix was or was not tested in TRAIN or SELECT, by mix group, and by age (56 d, 90-100 d, 120-180 d, 270-365 d); RMSE(champion_retrained) - RMSE(model) with its paired interval; physics violations (V_age, V_wb, rate) on the query grid of FUTURE mixes (inputs only).',
  disclosures: [
    'The input sets base, rich and physics, PANEL_DESIGN and the page network\'s default settings (64-32, lr 0.004, decay 1e-3, 300 epochs) were designed earlier in this project on the full UCI data, which includes ages over 28 days. Prior knowledge of those ages therefore entered the design of the search space and of several baselines, although no FUTURE row was used by any lab code.',
    'The REWARD and MONITOR scores of every model were known when this comparison set and criterion were written (they are listed under selectionSetScores). This registration was written after an audit of the first one, also with those scores known.',
    'The seal is procedural. The page data file (nn_data in public/model) and the sealed FUTURE file in the lab vault folder are both in git, so anyone could have read FUTURE. The seal and the guard prove only that no lab code read the vault before the opening.',
    'Gradient-boosted trees and the random forest cannot extrapolate in age beyond the 7-day TRAIN rows, so they are weak baselines by construction.',
    `${receipt.seenMixRows.length} of ${receipt.rows} FUTURE rows come from mixes already tested at earlier ages (in TRAIN or SELECT). The headline is therefore about predicting later ages of known mixes. The ${receipt.unseenMixRows.length} unseen-mix rows are descriptive only.`,
    'The per-mix curve uses the fallback (the TRAIN-calibrated fib generator) for mixes with no TRAIN rows; changing w/b changes the mix, so its w/b violation rate (17% on the TRAIN query grid) mostly reflects that switch.',
    'The REWARD scores of evolved models are optimistic: evolution selected on REWARD.',
  ],
  openingProtocol: 'Run node lab/open_vault.mjs once. It writes lab/results/OPENING_STARTED.json before reading the vault; the opener must commit OPENING_STARTED.json, vault_opening.json and VAULT_OPENED immediately after the run, so the opening counts as done in every clone.',
  honestReporting: 'Every result is published, including failures. The vault is opened exactly once.',
  models,
}
const text = JSON.stringify(prereg, null, 1) + '\n'
if (process.argv.includes('--dry-run')) { console.log(text); process.exit(0) }
if (existsSync(OUT)) replaceArchived(OUT, V1, text)
else writeNew(OUT, text)
console.log(`preregister: wrote lab/preregistration.json (format 3), sha256 ${sha256(text)}; ${models.length} models; code sha256 ${prereg.code.codeSha256}; FUTURE sha256 ${receipt.sha256}`)
