// Eight different 80/20 splits by mix over ALL rows, written to a file so both models use the same ones.
import { readFileSync, writeFileSync } from 'node:fs'
const d = JSON.parse(readFileSync(process.argv[2] + '/public/model/nn_data.json'))
const g = [...d.train.g, ...d.test.g], groups = [...new Set(g)].sort((a, b) => a - b)
const out = []
for (let s = 1; s <= 8; s++) {
  let a = 1000 + s; const rnd = () => { a = (a * 1664525 + 1013904223) >>> 0; return a / 4294967296 }
  const gs = [...groups]; for (let i = gs.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [gs[i], gs[j]] = [gs[j], gs[i]] }
  const test = new Set(gs.slice(0, Math.round(gs.length * 0.2)))
  out.push(g.map((x) => (test.has(x) ? 1 : 0)))
}
writeFileSync(process.argv[3], JSON.stringify(out)); console.log('8 splits written;', groups.length, 'mixes,', g.length, 'rows')
