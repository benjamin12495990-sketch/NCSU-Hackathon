// Vercel serverless function: the same inverse-design pipeline the browser
// runs (src/lib/design.js), exposed over HTTP so something other than this
// page — a script, a second app, a curl command — can ping it for a fresh
// batch of candidate mixes, generated and ranked by CO2 in one call.
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { design } from '../src/lib/design.js'
import { prepareKnown } from '../src/lib/nearest.js'
import { concreteLimits } from '../src/lib/exposure.js'
import { rankMaterials } from '../src/lib/rank.js'

// Kept in sync with Requirements.jsx's DEFAULT_REQ (that file is JSX and
// can't be imported into a plain Node function). Used when a caller (e.g.
// curl) doesn't send an environment, so no requirement is switched on.
const DEFAULT_REQ = {
  humidity: { on: false, level: 1 },
  freeze: { on: false, level: 1 },
  wear: { on: false, level: 1 },
  serviceTemp: { on: false, value: 200 },
  life: { on: false, years: 50 },
}

const MODEL_DIR = join(process.cwd(), 'public', 'model')
const readJson = (name) => JSON.parse(readFileSync(join(MODEL_DIR, `${name}.json`), 'utf8'))

// Loaded once per warm serverless instance, same JSON files the browser fetches.
let cached = null
function loadData() {
  if (cached) return cached
  const spec = readJson('features')
  const space = readJson('space')
  cached = {
    spec,
    space,
    model: readJson('model'),
    modelLow: readJson('model_low'),
    metrics: readJson('metrics'),
    factors: readJson('factors').ingredients,
    known: prepareKnown(readJson('known'), space, spec.ingredients),
    exposure: readJson('exposure'),
    materialFile: readJson('materials'),
  }
  return cached
}

const toBool = (v, d) => (v === undefined ? d : v === true || v === 'true')

export default function handler(req, res) {
  if (req.method !== 'GET' && req.method !== 'POST') {
    res.setHeader('Allow', 'GET, POST')
    return res.status(405).json({ error: 'Use GET or POST.' })
  }
  const body = req.method === 'POST' ? req.body ?? {} : req.query

  const targetMPa = Number(body.targetMPa)
  if (!Number.isFinite(targetMPa) || targetMPa < 5 || targetMPa > 80) {
    return res.status(400).json({ error: 'targetMPa must be a number between 5 and 80.' })
  }
  const ageDays = Number(body.ageDays ?? 28)
  const limit = Math.min(Math.max(Number(body.limit) || 10, 1), 50)
  // No seed given: a fresh random one each call, so repeated pings surface
  // new candidates instead of the page's fixed demo batch.
  const seed = body.seed !== undefined ? Number(body.seed) : Math.floor(Math.random() * 2 ** 31)
  const cautious = toBool(body.cautious, true)
  const onlySupported = toBool(body.onlySupported, true)
  const envReq = body.req ?? DEFAULT_REQ
  const objective = body.objective ?? 'co2'

  try {
    const data = loadData()
    // Same two steps App.jsx's run() does: the environment can cap the
    // water/binder ratio and raise the strength the concrete needs, and
    // existing reference materials get ranked against the same requirements.
    const limits = concreteLimits(envReq, data.exposure)
    const needed = Math.max(targetMPa, limits.minStrengthMPa)
    const result = design(
      {
        targetMPa: needed,
        ageDays,
        seed,
        cautious,
        onlySupported,
        allowSlag: toBool(body.allowSlag, true),
        allowFlyAsh: toBool(body.allowFlyAsh, true),
        maxWaterBinder: limits.maxWaterBinder,
      },
      data,
      limit,
    )
    res.setHeader('Cache-Control', 'no-store')
    return res.status(200).json({
      ...result,
      cautious,
      onlySupported,
      asked: targetMPa,
      limits,
      req: envReq,
      objective,
      seed,
      shortlist: rankMaterials(data.materialFile.materials, envReq, targetMPa, objective),
    })
  } catch (e) {
    return res.status(500).json({ error: e.message })
  }
}
