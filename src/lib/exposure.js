// Turns the environment sliders into limits on the concrete mix, using the
// ACI 318-19 exposure classes stored in exposure.json. Where several classes
// apply, ACI takes the lowest water/cement ratio and the highest strength.
export const HUMIDITY = [
  { label: 'Humid air or occasional wetting', classes: ['C1', 'W1'] },
  { label: 'In contact with water, must stay watertight', classes: ['W2'] },
  { label: 'Wet with chlorides: de-icing salt or sea water', classes: ['W2', 'C2'] },
]
export const FREEZE = [
  { label: 'Freezes, rarely wet', classes: ['F1'] },
  { label: 'Freezes, often wet', classes: ['F2'] },
  { label: 'Freezes, often wet, with de-icing salt', classes: ['F3'] },
]
export const WEAR = ['Light foot traffic', 'Vehicles or carts', 'Heavy abrasion']

export function concreteLimits(req, exposure) {
  const applied = []
  const add = (names) => {
    for (const name of names) {
      const c = exposure.classes[name]
      if (c && !applied.some((a) => a.name === name)) applied.push({ name, ...c })
    }
  }
  if (req.humidity.on) add(HUMIDITY[req.humidity.level - 1].classes)
  if (req.freeze.on) add(FREEZE[req.freeze.level - 1].classes)
  let maxWaterBinder = null
  let minStrengthMPa = 0
  for (const c of applied) {
    if (c.max_w_cm != null) maxWaterBinder = maxWaterBinder == null ? c.max_w_cm : Math.min(maxWaterBinder, c.max_w_cm)
    if (c.min_fc_MPa != null) minStrengthMPa = Math.max(minStrengthMPa, c.min_fc_MPa)
  }
  const wear = req.wear.on ? exposure.wear : null
  if (wear) minStrengthMPa = Math.max(minStrengthMPa, wear.min_fc_MPa)
  return { applied, wear, maxWaterBinder, minStrengthMPa }
}
