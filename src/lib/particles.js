// Turns a concrete mix (kg of each ingredient per cubic metre) into particles
// for the 3D view. Each ingredient's share of the 216 particles follows its
// share of the mix's VOLUME (mass divided by density, the same numbers the
// designer uses). Particle sizes and positions are for looks only, and the
// page says so next to the model.

const LOOK = {
  cement: { color: '#1f3fa8', radius: 0.27 },
  slag: { color: '#12a387', radius: 0.27 },
  flyash: { color: '#e07a2f', radius: 0.27 },
  water: { color: '#3aa6e8', radius: 0.25 },
  superplasticizer: { color: '#e0457b', radius: 0.2 },
  coarse: { color: '#a3a8b8', radius: 0.46 },
  fine: { color: '#e3c88f', radius: 0.3 },
}
const SIDE = 6 // 6 x 6 x 6 sites in the cube

function seeded(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function buildParticles(mix, factors, ingredients) {
  const sites = SIDE ** 3
  const volumes = ingredients.map((name, i) => mix[i] / factors[name].density_kg_m3)
  const total = volumes.reduce((s, v) => s + v, 0)

  // Largest-remainder rounding, so the counts add up to exactly `sites` and
  // any ingredient that is present gets at least one particle.
  const exact = volumes.map((v) => (v / total) * sites)
  const counts = exact.map((e, i) => (volumes[i] > 0 ? Math.max(1, Math.floor(e)) : 0))
  const order = exact.map((e, i) => [e - Math.floor(e), i]).sort((a, b) => b[0] - a[0])
  let spare = sites - counts.reduce((s, c) => s + c, 0)
  for (let k = 0; spare > 0; k = (k + 1) % order.length) {
    if (volumes[order[k][1]] > 0) { counts[order[k][1]]++; spare-- }
  }
  for (let k = 0; spare < 0; k = (k + 1) % counts.length) {
    const biggest = counts.indexOf(Math.max(...counts))
    counts[biggest]--; spare++
  }

  // The same sites are shuffled the same way for every mix, so switching
  // between two mixes changes only the particles whose ingredient changed.
  const rand = seeded(20261120)
  const positions = []
  const half = (SIDE - 1) / 2
  for (let x = 0; x < SIDE; x++)
    for (let y = 0; y < SIDE; y++)
      for (let z = 0; z < SIDE; z++)
        positions.push([x - half + (rand() - 0.5) * 0.3, y - half + (rand() - 0.5) * 0.3, z - half + (rand() - 0.5) * 0.3])
  for (let i = positions.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[positions[i], positions[j]] = [positions[j], positions[i]]
  }

  const particles = []
  ingredients.forEach((name, i) => {
    for (let c = 0; c < counts[i]; c++) {
      particles.push({ name, pos: positions[particles.length], ...LOOK[name] })
    }
  })
  const legend = ingredients.map((name, i) => ({
    name,
    label: factors[name].label,
    color: LOOK[name].color,
    kg: mix[i],
    share: volumes[i] / total,
    co2: mix[i] * factors[name].kgCO2e_per_kg,
  }))
  return { particles, legend, half: SIDE / 2 }
}
