// Ranks the reference materials in materials.json against the requirements.
// A material is left out if it fails any requirement that is switched on;
// the rest are ordered by the chosen objective.
export const OBJECTIVES = {
  co2: { label: 'Least CO₂ for the strength', unit: 'kg CO₂e per m³ per MPa', lowerIsBetter: true },
  mass: { label: 'Least mass for the strength', unit: 'kg per m³ per MPa', lowerIsBetter: true },
  strength: { label: 'Strongest', unit: 'MPa', lowerIsBetter: false },
}

// A long service life asks for one more step of humidity and wear resistance.
// This is our rule of thumb, not a value from a standard.
const longLife = (req) => (req.life.on && req.life.years >= 50 ? 1 : 0)
export const humidityNeed = (req) => Math.min(5, [3, 4, 5][req.humidity.level - 1] + longLife(req))
export const wearNeed = (req) => Math.min(5, [2, 3, 4][req.wear.level - 1] + longLife(req))

export function rankMaterials(materials, req, strengthMPa, objective) {
  const rows = []
  const excluded = { strength: 0, temperature: 0, humidity: 0, wear: 0 }
  for (const m of materials) {
    const fails = []
    if (m.strength_MPa < strengthMPa) fails.push('strength')
    if (req.serviceTemp.on && !(m.max_service_temp_C >= req.serviceTemp.value)) fails.push('temperature')
    if (req.humidity.on && m.humidity_resistance_1to5 < humidityNeed(req)) fails.push('humidity')
    if (req.wear.on && m.wear_resistance_1to5 < wearNeed(req)) fails.push('wear')
    if (fails.length) {
      excluded[fails[0]]++
      continue
    }
    let index = null
    if (objective === 'co2') index = m.embodied_CO2_kg_per_kg == null ? null : (m.density_kg_m3 * m.embodied_CO2_kg_per_kg) / m.strength_MPa
    else if (objective === 'mass') index = m.density_kg_m3 / m.strength_MPa
    else index = m.strength_MPa
    rows.push({ m, index })
  }
  const dir = OBJECTIVES[objective].lowerIsBetter ? 1 : -1
  rows.sort((a, b) => (a.index == null) - (b.index == null) || dir * (a.index - b.index))
  return { rows, excluded, considered: materials.length }
}
