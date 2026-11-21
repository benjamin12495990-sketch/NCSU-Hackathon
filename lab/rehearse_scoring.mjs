// Tests the scoring code that open_vault.mjs will use, on SELECT instead of FUTURE, so a bug cannot waste the
// single opening. It never touches the vault and writes nothing. SELECT rows count as "seen" when their exact
// mix has TRAIN rows. (On SELECT these numbers are not new: evolution already chose by them.)
//   node lab/rehearse_scoring.mjs --prereg <a pre-registration file>
import './guard.mjs'
import path from 'node:path'
import { LAB } from './guard.mjs'
import { readJson } from './io.mjs'
import { scoreModels } from './scoring.mjs'

const i = process.argv.indexOf('--prereg')
const prereg = readJson(path.resolve(i > 0 ? process.argv[i + 1] : path.join(LAB, 'preregistration.json')))
const train = readJson(path.join(LAB, 'data', 'train.json')), select = readJson(path.join(LAB, 'data', 'select.json'))
const seenMix = new Set(train.mix), seenGroup = new Set(train.g)
const idx = (f) => select.y.map((_, k) => k).filter(f)
const split = {
  seenMixRows: idx((k) => seenMix.has(select.mix[k])), unseenMixRows: idx((k) => !seenMix.has(select.mix[k])),
  seenGroupRows: idx((k) => seenGroup.has(select.g[k])), unseenGroupRows: idx((k) => !seenGroup.has(select.g[k])),
}
for (const r of scoreModels(prereg.models, select, split)) {
  const want = prereg.models.find((m) => m.role === r.role).select.selectMae
  const same = Math.abs(r.overall.mae - want) < 1e-9
  console.log(`${same ? 'ok  ' : 'DIFF'} ${r.role}: SELECT MAE ${r.overall.mae.toFixed(4)} (pre-registered ${want.toFixed(4)}), seen mixes ${r.seenMix.mae.toFixed(3)} on ${r.seenMix.rows}, unseen ${r.unseenMix.mae.toFixed(3)} on ${r.unseenMix.rows}`)
}
