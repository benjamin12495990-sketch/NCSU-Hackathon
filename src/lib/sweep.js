// Sweep test: change one thing at a time and see how a model's prediction
// moves. More water for the same binder should weaken concrete; more curing
// time should strengthen it. A model that learned the data should show both.
export const AGES = [1, 3, 7, 14, 28, 56, 90, 180, 365]

// The same mix with only its water changed, across the water/binder ratios
// that labs actually tested. Returns null if this mix leaves no room to vary.
export function waterSteps(mix, space, count = 21) {
  const binder = mix[0] + mix[1] + mix[2]
  const lo = Math.max(space.ranges.water[0], space.limits.water_binder_ratio[0] * binder)
  const hi = Math.min(space.ranges.water[1], space.limits.water_binder_ratio[1] * binder)
  if (!(hi - lo >= 20)) return null
  return Array.from({ length: count }, (_, i) => {
    const water = lo + ((hi - lo) * i) / (count - 1)
    const changed = mix.slice()
    changed[3] = water
    return { x: water / binder, mix: changed }
  })
}

// How a series of predictions moves: how many steps go up, how many go down.
export function trend(values) {
  let up = 0, down = 0
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[i - 1] + 1e-9) up++
    else if (values[i] < values[i - 1] - 1e-9) down++
  }
  return { steps: values.length - 1, up, down, first: values[0], last: values[values.length - 1] }
}
