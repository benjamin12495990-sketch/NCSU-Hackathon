import { prepareKnown } from './nearest.js'

// Downloads the JSON files in public/model/ once, when the page opens.
// Everything after this runs in the browser with no network calls.
export async function loadModel() {
  const base = import.meta.env.BASE_URL + 'model/'
  const get = async (name) => {
    const res = await fetch(base + name)
    if (!res.ok) throw new Error(`Could not load ${name} (HTTP ${res.status}).`)
    return res.json()
  }
  const names = ['features', 'model', 'model_low', 'metrics', 'known', 'space', 'factors', 'test_points', 'materials', 'frontier', 'exposure']
  const [spec, model, modelLow, metrics, knownRaw, space, factorFile, testPoints, materialFile, frontierFile, exposure] = await Promise.all(names.map((n) => get(n + '.json')))
  return {
    spec, model, modelLow, metrics, space, testPoints, factorFile, materialFile, frontierFile, exposure,
    factors: factorFile.ingredients,
    known: prepareKnown(knownRaw, space, spec.ingredients),
  }
}
