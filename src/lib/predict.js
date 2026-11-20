// Walks every tree in model.json and averages the leaf values (a random
// forest). scikit-learn compares features as 32-bit floats, so we round the
// feature with Math.fround to take the same branch it would.
export function predict(model, x) {
  let sum = 0
  for (const tree of model.trees) {
    let node = 0
    while (tree.f[node] !== -2) {
      node = Math.fround(x[tree.f[node]]) <= tree.t[node] ? tree.l[node] : tree.r[node]
    }
    sum += tree.v[node]
  }
  return sum / model.trees.length
}
