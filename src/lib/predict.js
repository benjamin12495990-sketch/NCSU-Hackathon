// Predicts compressive strength in MPa. The model is gradient-boosted trees:
// start from a base value, then add learning_rate x the leaf reached in each
// tree. scikit-learn compares features as 32-bit floats, so we do the same.
export function predict(model, x) {
  const xf = Float32Array.from(x)
  let sum = model.init
  for (const t of model.trees) {
    let n = 0
    while (t.f[n] !== -2) n = xf[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n]
    sum += model.learning_rate * t.v[n]
  }
  return sum
}
