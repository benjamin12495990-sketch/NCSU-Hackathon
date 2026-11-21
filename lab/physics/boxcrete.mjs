// Maps lab/external/boxcrete_data.csv (BOxCrete, Baten et al. 2026, arXiv:2603.21525, MIT) to the nine UCI
// features: cement, slag, flyash, water, superplasticizer (HRWR), coarse, fine, age (days), water_binder_ratio
// = water / (cement + slag + flyash), as train/train.py computes it. Strength (Mean) in psi / 145.04 = MPa.
// Kept: concrete rows (coarse aggregate > 0) cured at 22 C with a strength. Its ages stop at 28 days.
import '../guard.mjs'
import path from 'node:path'
import { readFileSync } from 'node:fs'
import { LAB } from '../guard.mjs'

export const BOXCRETE_FILE = path.join(LAB, 'external', 'boxcrete_data.csv')
export function boxcreteRows() {
  const lines = readFileSync(BOXCRETE_FILE, 'utf8').trim().split('\n')
  const head = lines[0].split(','), col = (name) => head.indexOf(name)
  const c = {
    cement: col('Cement (kg/m3)'), fly: col('Fly Ash (kg/m3)'), slag: col('Slag (kg/m3)'), water: col('Water (kg/m3)'), hrwr: col('HRWR (kg/m3)'),
    fine: col('Fine Aggregate (kg/m3)'), coarse: col('Coarse Aggregates (kg/m3)'), temp: col('Temp (C)'), time: col('Time'), mean: col('Strength (Mean)'), mix: col('Mix Name'),
  }
  const X = [], y = [], mix = []
  let total = 0
  for (const line of lines.slice(1)) {
    total++
    const v = line.split(',')
    const num = (k) => Number(v[c[k]])
    if (!(num('coarse') > 0) || num('temp') !== 22 || !Number.isFinite(num('mean')) || v[c.mean] === '') continue
    const binder = num('cement') + num('slag') + num('fly')
    X.push([num('cement'), num('slag'), num('fly'), num('water'), num('hrwr') || 0, num('coarse'), num('fine'), num('time'), num('water') / binder])
    y.push(num('mean') / 145.04)
    mix.push(v[c.mix])
  }
  return { X, y, mix, meta: { source: 'BOxCrete, Baten et al. 2026, arXiv:2603.21525, MIT (facebookresearch/SustainableConcrete)', rowsInFile: total, rowsKept: X.length, mixes: new Set(mix).size, ages: [...new Set(X.map((x) => x[7]))].sort((a, b) => a - b), filter: 'coarse aggregate > 0 and Temp (C) == 22', units: 'psi / 145.04 = MPa' } }
}
