import { prepareKnown } from './nearest.js'

// Downloads the JSON files that train/train.py wrote into public/model/.
// Everything after this runs in the browser with no network calls.
export async function loadModel() {
  const base = import.meta.env.BASE_URL + 'model/'
  const get = async (name) => {
    const res = await fetch(base + name)
    if (!res.ok) throw new Error(`Could not load ${name} (HTTP ${res.status}).`)
    return res.json()
  }
  const [elements, model, metrics, knownRaw] = await Promise.all(
    ['elements.json', 'model.json', 'metrics.json', 'known.json'].map(get),
  )
  return { elements, model, metrics, known: prepareKnown(knownRaw) }
}
