// Embodied CO2 of one cubic metre of a mix: each ingredient's mass (kg) times
// its emission factor (kg CO2e per kg) from factors.json, added up.
export function co2PerM3(mix, factors, ingredients) {
  let total = 0
  ingredients.forEach((name, i) => (total += mix[i] * factors[name].kgCO2e_per_kg))
  return total
}

// Solid volume of a mix in cubic metres: mass divided by density, added up.
// The designer uses it to keep each new mix the same size as its parent.
export function absoluteVolume(mix, factors, ingredients) {
  let v = 0
  ingredients.forEach((name, i) => (v += mix[i] / factors[name].density_kg_m3))
  return v
}
