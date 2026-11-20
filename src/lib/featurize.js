// Builds the model's inputs for one mix at one age, in the order listed in
// features.json. The only derived feature is water divided by total binder
// (cement + slag + fly ash); this must match featurize() in train/train.py.
export function featurize(mix, age, spec) {
  const v = { age }
  spec.ingredients.forEach((name, i) => (v[name] = mix[i]))
  v.water_binder_ratio = v.water / (v.cement + v.slag + v.flyash)
  return spec.features.map((name) => v[name])
}
