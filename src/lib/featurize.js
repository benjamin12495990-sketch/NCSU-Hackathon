// Builds the model's feature vector: the fraction of each element in the
// order given by elements.json, followed by the number of distinct elements.
// This must stay identical to the features in train/train.py.
export function featurize(pairs, elements) {
  const index = new Map(elements.map((el, k) => [el, k]))
  const x = new Float64Array(elements.length + 1)
  let total = 0
  for (const [, n] of pairs) total += n
  let distinct = 0
  for (const [el, n] of pairs) {
    x[index.get(el)] += n / total
    distinct++
  }
  x[elements.length] = distinct
  return x
}

// Text key for a composition, used to test "is this already in the dataset".
export function compositionKey(x, nElements) {
  const parts = []
  for (let k = 0; k < nElements; k++) if (x[k] > 0) parts.push(k + ':' + x[k].toFixed(4))
  return parts.join('|')
}
