import { lazy, Suspense, useEffect, useState } from 'react'
import { loadModel } from './lib/loadModel.js'
import { design } from './lib/design.js'
import { concreteLimits } from './lib/exposure.js'
import { rankMaterials } from './lib/rank.js'
import Requirements, { DEFAULT_REQ } from './components/Requirements.jsx'
import Shortlist from './components/Shortlist.jsx'
import Planned from './components/Planned.jsx'
import NeuralNet from './components/NeuralNet.jsx'
import Formula from './components/Formula.jsx'
import { DOMAINS } from './lib/domains.js'

// Three.js is big, so the 3D stage is downloaded only when a result is shown.
const MixStage = lazy(() => import('./components/MixStage.jsx'))

const PRESETS = [
  { label: 'Light duty', mpa: 15 },
  { label: 'General use', mpa: 25 },
  { label: 'Structural', mpa: 35 },
]
const AGES = [3, 7, 14, 28, 56, 90]
const REPO = 'https://github.com/benjamin12495990-sketch/NCSU-Hackathon'
const kg = (v) => Math.round(v).toLocaleString()
const pct = (v) => {
  const n = Math.round(v * 100)
  return n === 0 ? '0%' : `${n > 0 ? '+' : '−'}${Math.abs(n)}%`
}

// Scatter of the held-out test set: lab-measured strength against the model's
// prediction. Dots on the dashed line are perfect predictions.
function TestChart({ points }) {
  const size = 300, pad = 38, max = 90
  const s = (v) => pad + (v / max) * (size - pad - 8)
  const y = (v) => size - s(v)
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="chart" role="img"
      aria-label="Predicted against measured strength for the held-out test mixes">
      {[0, 30, 60, 90].map((t) => (
        <g key={t}>
          <line x1={s(t)} y1={y(0)} x2={s(t)} y2={y(max)} className="grid" />
          <line x1={s(0)} y1={y(t)} x2={s(max)} y2={y(t)} className="grid" />
          <text x={s(t)} y={size - pad + 14} textAnchor="middle">{t}</text>
          <text x={pad - 6} y={y(t) + 4} textAnchor="end">{t}</text>
        </g>
      ))}
      <line x1={s(0)} y1={y(0)} x2={s(max)} y2={y(max)} className="diag" />
      {points.map(([m, p], i) => <circle key={i} cx={s(m)} cy={y(p)} r="2.4" />)}
      <text x={(size + pad) / 2} y={size - 4} textAnchor="middle">Measured (MPa)</text>
      <text x={12} y={size / 2 - 10} textAnchor="middle" transform={`rotate(-90 12 ${size / 2 - 10})`}>Predicted (MPa)</text>
    </svg>
  )
}

// The single page: loads the model once, then designs mixes in the browser
// each time the user submits a target.
export default function App() {
  const [data, setData] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [target, setTarget] = useState('25')
  const [age, setAge] = useState(28)
  const [allowSlag, setAllowSlag] = useState(true)
  const [allowFlyAsh, setAllowFlyAsh] = useState(true)
  const [cautious, setCautious] = useState(true)
  const [onlySupported, setOnlySupported] = useState(true)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [view, setView] = useState('row0') // which mix the 3D stage shows
  const [req, setReq] = useState(DEFAULT_REQ) // environment switches and sliders
  const [field, setField] = useState('general')
  const [objective, setObjective] = useState('co2')
  const [domainId, setDomainId] = useState('concrete') // which family of materials is shown

  useEffect(() => {
    loadModel().then(setData, (e) => setLoadError(e.message))
  }, [])

  const run = (targetText = target) => {
    setTarget(targetText)
    setView('row0')
    // Accept plain decimals only ("35", "35.5", "35,5", "35 MPa"); reject forms
    // such as 0x23 or 3e1 that JavaScript would quietly turn into a number.
    const text = String(targetText).trim().replace(/\s*mpa$/i, '').replace(',', '.')
    const mpa = /^\d+(\.\d+)?$/.test(text) ? Number(text) : NaN
    if (!Number.isFinite(mpa)) {
      setResult(null)
      return setError('Type the strength you need in MPa as a number, for example 35.')
    }
    if (mpa < 5 || mpa > 80) {
      setResult(null)
      return setError('Choose a strength between 5 and 80 MPa. The lab results behind the model run from about 2 to 83 MPa.')
    }
    try {
      // The environment can cap the water/binder ratio and raise the strength the concrete needs.
      const limits = concreteLimits(req, data.exposure)
      const needed = Math.max(mpa, limits.minStrengthMPa)
      setResult({
        ...design({ targetMPa: needed, ageDays: age, allowSlag, allowFlyAsh, cautious, onlySupported, maxWaterBinder: limits.maxWaterBinder }, data),
        cautious, onlySupported, asked: mpa, limits, req, objective,
        shortlist: rankMaterials(data.materialFile.materials, req, mpa, objective),
      })
      setError('')
    } catch (e) {
      setResult(null)
      setError(`Something went wrong while designing mixes: ${e.message}`)
    }
  }

  const m = data?.metrics
  const best = result?.top[0]
  const bar = result?.cautious ? 'cautious strength estimate' : 'expected strength'

  // The three headline mixes. Each can be put on the 3D stage, and so can any table row.
  const picks = result
    ? [
        {
          id: 'row0', label: 'Best new candidate (not in the dataset)', item: best, title: 'New candidate #1',
          sub: best ? `expected ${best.strength.toFixed(1)} MPa, cautious estimate ${best.low.toFixed(1)} MPa` : 'none found with these settings',
        },
        {
          id: 'tested', label: 'Best mix a lab already tested', item: result.baseline, title: 'Best lab-tested mix',
          sub: result.baseline ? `measured ${result.baseline.strength.toFixed(1)} MPa at ${result.baseline.age} days` : 'none in the dataset reaches this target within your limits',
        },
        {
          id: 'cement', label: 'Best tested mix using cement only', item: result.cementOnly, title: 'Best cement-only tested mix',
          sub: result.cementOnly ? `measured ${result.cementOnly.strength.toFixed(1)} MPa at ${result.cementOnly.age} days` : 'none in the dataset reaches this target within your limits',
        },
      ]
    : []
  const rowIndex = view.startsWith('row') ? Number(view.slice(3)) : -1
  const stage =
    (rowIndex > 0 && result?.top[rowIndex] && { label: 'New candidate (not in the dataset)', title: `New candidate #${rowIndex + 1}`, item: result.top[rowIndex] }) ||
    picks.find((p) => p.id === view && p.item) ||
    picks.find((p) => p.item) ||
    null
  const cementShare = stage ? (stage.item.mix[0] * data.factors.cement.kgCO2e_per_kg) / stage.item.co2 : 0
  const show = (id) => {
    setView(id)
    document.querySelector('.result')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }
  return (
    <main>
      <header>
        <p className="eyebrow">Engineer the Tools of Scientific Discovery</p>
        <h1>Formulation Designer</h1>
        <p className="lede">
          Proposes new formulations for a property you need, and ranks them. Concrete is live. Six
          more families of materials are planned and marked as planned.
        </p>
        <nav className="domains" aria-label="Family of materials">
          {DOMAINS.map((d) => (
            <button type="button" key={d.id} className="chip" aria-pressed={domainId === d.id} onClick={() => setDomainId(d.id)}>
              {d.label} <span className={d.live ? 'tag live' : 'tag'}>{d.live ? 'Live' : 'Planned'}</span>
            </button>
          ))}
        </nav>
      </header>

      {domainId !== 'concrete' ? (
        <Planned domain={DOMAINS.find((d) => d.id === domainId)} onBack={() => setDomainId('concrete')} />
      ) : (
      <>
      <section className="intro">
        <h2>Low-carbon concrete mixes</h2>
        <p className="lede">
          Say how strong the concrete must be. The app proposes new mixes predicted to reach that
          strength and ranks them by embodied CO<sub>2</sub>. For civil and materials engineering
          researchers choosing which trial mixes to cast, when every lab test means waiting 28 days
          for the concrete to cure. Only concrete mixes are generated; every other material on the page is
          an existing one, shown for comparison.
        </p>
      </section>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (data) run()
        }}
      >
        <div className="fields">
          <div>
            <label htmlFor="target">Strength needed (MPa)</label>
            <input id="target" value={target} onChange={(e) => setTarget(e.target.value)}
              inputMode="decimal" autoComplete="off" placeholder="e.g. 35" />
          </div>
          <div>
            <label htmlFor="age">Reached by (days of curing)</label>
            <select id="age" value={age} onChange={(e) => setAge(Number(e.target.value))}>
              {AGES.map((a) => <option key={a} value={a}>{a} days</option>)}
            </select>
          </div>
          <button type="submit" disabled={!data}>{data ? 'Design mixes' : 'Loading model…'}</button>
        </div>
        <div className="examples">
          <span>Try:</span>
          {PRESETS.map((p) => (
            <button type="button" key={p.mpa} className="chip" disabled={!data} onClick={() => run(String(p.mpa))}>
              {p.label} · {p.mpa} MPa
            </button>
          ))}
        </div>
        <Requirements req={req} setReq={setReq} field={field} setField={setField} objective={objective} setObjective={setObjective} />
        <div className="options">
          <label><input type="checkbox" checked={allowSlag} onChange={(e) => setAllowSlag(e.target.checked)} /> Blast furnace slag available</label>
          <label><input type="checkbox" checked={allowFlyAsh} onChange={(e) => setAllowFlyAsh(e.target.checked)} /> Fly ash available</label>
          <label><input type="checkbox" checked={cautious} onChange={(e) => setCautious(e.target.checked)} /> Cautious: the lower strength estimate must reach the target</label>
          <label><input type="checkbox" checked={onlySupported} onChange={(e) => setOnlySupported(e.target.checked)} /> Only mixes backed by nearby lab results</label>
        </div>
        <p className="note small">
          Change an option, then press Design mixes again. The three presets are illustrative
          strength levels, not values from a building code.
        </p>
      </form>

      {loadError && <p className="error" role="alert">The model files did not load: {loadError}</p>}
      {error && <p className="error" role="alert">{error}</p>}

      {result && (
        <>
          <section className={stage ? 'result' : 'result solo'}>
            {stage && (
              <div className="stagewrap">
                <div className="stagehead">
                  <span className="label">{stage.label}</span>
                  <h2>{stage.title}: {kg(stage.item.co2)} kg CO<sub>2</sub>e per m³</h2>
                  <span className="mixname">{stage.item.name}</span>
                  <Formula mix={stage.item.mix} oxides={data.oxides} />
                  <span className="flag hot">Cement pulses: it is {Math.round(cementShare * 100)}% of this mix's CO<sub>2</sub></span>
                </div>
                <Suspense fallback={<div className="stage"><div className="canvas" /></div>}>
                  <MixStage
                    key={stage.item.mix.join('|')}
                    mix={stage.item.mix}
                    factors={data.factors}
                    ingredients={data.spec.ingredients}
                    highlight="cement"
                  />
                </Suspense>
              </div>
            )}
            <div className="readout">
            <h2>{result.targetMPa} MPa at {result.ageDays} days</h2>
            {(result.targetMPa > result.asked || result.limits.maxWaterBinder != null) && (
              <p className="note">
                Your environment changes the concrete brief:
                {result.targetMPa > result.asked && ` strength raised from ${result.asked} to ${result.targetMPa} MPa`}
                {result.targetMPa > result.asked && result.limits.maxWaterBinder != null && ' and'}
                {result.limits.maxWaterBinder != null && ` water / binder capped at ${result.limits.maxWaterBinder}`}. Details below.
              </p>
            )}
            <div className="picks">
              {picks.map((p) => (
                <button
                  type="button"
                  key={p.id}
                  className="pick"
                  disabled={!p.item}
                  aria-pressed={!!stage && stage.item === p.item}
                  onClick={() => show(p.id)}
                >
                  <span className="label">{p.label}</span>
                  {p.item && <span className="mixname">{p.item.name}</span>}
                  <span className="big">{p.item ? kg(p.item.co2) : '–'} <small>kg CO<sub>2</sub>e per m³</small></span>
                  <span className="label">{p.sub}</span>
                </button>
              ))}
            </div>
            {best ? (
              <p className="note">
                {result.baseline && `Against the best tested mix, the top candidate changes embodied CO₂ by ${pct(best.vsBaseline)}. `}
                {result.cementOnly && `Against the best tested cement-only mix, it changes it by ${pct(best.co2 / result.cementOnly.co2 - 1)}. `}
                The tested mixes have measured strengths. The candidate has only a prediction. It is not in this dataset, but it may exist elsewhere in the literature.
              </p>
            ) : (
              <p className="note">
                No new mix has a {bar} of {result.targetMPa} MPa at {result.ageDays} days
                {result.stats.hiddenUnsupported > 0 && ` and enough nearby lab results to back it (${result.stats.hiddenUnsupported} were hidden for that reason)`}.
                Lower the target, allow more curing time, or change the options above.
                {result.targetMPa > result.asked && ' Your environment set this strength, so relaxing a requirement will help more than lowering the target.'}
              </p>
            )}
            {!result.cautious && best && (
              <p className="warn">
                Caution is off. These mixes only reach the target on the model's expected strength, and their
                cautious estimates fall short. Read them as experiments worth running, not as mixes to use.
              </p>
            )}
            </div>
          </section>

          {best && (
            <section>
              <h2>Generated mixes, ranked</h2>
              <p className="note">
                {result.stats.generated.toLocaleString()} mixes were generated in your browser by varying
                the {data.known.entries.length} tested mixes in the dataset. {result.stats.removedKnown.toLocaleString()} already
                exist (within 5 kg of a tested mix on every ingredient) and were removed.
                {result.stats.overWaterLimit > 0 && ` ${result.stats.overWaterLimit.toLocaleString()} broke the water / binder limit from your environment.`}{' '}
                {result.stats.meetTarget.toLocaleString()} of the rest have a {bar} of at least {result.targetMPa} MPa.
                {result.stats.hiddenUnsupported > 0 && ` ${result.stats.hiddenUnsupported} lower-CO₂ mixes are hidden because nearby lab results do not back them.`}{' '}
                Lowest CO<sub>2</sub> first. Amounts are kg per m³ of concrete. Press 3D on a row to see that mix.
              </p>
              <div className="tablewrap">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Formula name and paste formula</th>
                      <th className="num">Cement</th><th className="num">Slag</th><th className="num">Fly ash</th>
                      <th className="num">Water</th><th className="num">Super&shy;plasticizer</th>
                      <th className="num">Coarse agg.</th><th className="num">Fine agg.</th>
                      <th className="num">Water / binder</th>
                      <th className="num">Expected strength</th>
                      <th className="num">Cautious estimate</th>
                      <th className="num">CO<sub>2</sub>e per m³</th>
                      <th className="num">vs best tested</th>
                      <th className="num">Differs from every tested mix by</th>
                      <th>3 nearest tested mixes measured</th>
                      <th>Evidence</th>
                      <th><span className="sr">Show in 3D</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.top.map((c, i) => (
                      <tr key={c.mix.join('|')} className={stage?.item === c ? 'on' : ''}>
                        <td>{i + 1}</td>
                        <td><strong>{c.name}</strong><Formula mix={c.mix} oxides={data.oxides} /></td>
                        {c.mix.map((v, k) => <td key={k} className="num">{v}</td>)}
                        <td className="num">{c.waterBinder.toFixed(2)}</td>
                        <td className="num">{c.strength.toFixed(1)} MPa</td>
                        <td className="num">{c.low.toFixed(1)} MPa</td>
                        <td className="num strong">{kg(c.co2)} kg</td>
                        <td className="num">{c.vsBaseline == null ? '–' : pct(c.vsBaseline)}</td>
                        <td className="num">{Math.round(c.gap)} kg or more</td>
                        <td>{c.neighbours.map((n) => n.strength.toFixed(0)).join(', ')} MPa</td>
                        <td>
                          <span className={c.supported ? 'flag ok' : 'flag low'}>
                            {c.supported ? 'Neighbours passed' : 'Not backed'}
                          </span>
                        </td>
                        <td>
                          <button type="button" className="chip" aria-pressed={stage?.item === c}
                            aria-label={`Show candidate ${i + 1} in 3D`} onClick={() => show(`row${i}`)}>
                            3D
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="note small">
                Concrete is a mixture, not one compound, so the chemical formula under each name is the bulk
                composition of its paste (cement, slag, fly ash and water) per silicon atom, estimated from
                published oxide analyses; aggregates and admixture are left out (sources at the bottom).{' '}
                Formula names are made by the app from the recipe: C, S and F are the shares of cement, slag
                and fly ash in the binder by mass (percent), and W is water ÷ binder × 100. They are labels for
                this page, not standard designations.{' '}
                “Differs by” is the gap to the closest tested mix on the ingredient where they differ most; a
                generated mix must differ by more than 5 kg to count as new. New means not in this dataset: it
                may exist elsewhere in the literature.
                “Neighbours passed” means at least two of the three nearest tested mixes are close to the candidate
                and reached {result.targetMPa} MPa in the lab by {result.ageDays} days. The candidate itself is untested.
              </p>
            </section>
          )}
          <Shortlist data={data} result={{ ...result, objective, shortlist: rankMaterials(data.materialFile.materials, result.req, result.asked, objective) }} />
        </>
      )}

      <NeuralNet data={data} result={result} />

      <section className="caveat">
        <h2>How this can be wrong</h2>
        {m ? (
          <div className="split">
            <ul>
              <li>
                On {m.test_rows} lab results from {m.test_mixes} mixes held out of training, the expected strength is
                off by <strong>{m.test_mae_MPa.toFixed(1)} MPa on average</strong> (R² {m.test_r2.toFixed(2)}).
                A random split that lets the same mix appear in training and testing would have reported{' '}
                {m.random_split_mae_MPa.toFixed(1)} MPa, so we do not use it.
              </li>
              <li>
                The cautious estimate is built to be beaten by 9 lab results in 10. On the held-out mixes it was
                beaten by <strong>{(m.lower_estimate_coverage * 100).toFixed(0)} in 100</strong>, so it is less
                cautious than intended.
              </li>
              <li>
                Candidates are <strong>untested predictions, not proven mixes</strong>. Searching for the lowest
                CO<sub>2</sub> pulls toward mixes where the model is too optimistic, so the best-looking
                candidates are the most likely to disappoint. Cast and test before relying on any of them.
              </li>
              <li>
                The model predicts compressive strength only. It knows nothing about workability, setting
                time, durability, shrinkage or cost, and it cannot tell one source of slag, fly ash or
                aggregate from another.
              </li>
              <li>
                CO<sub>2</sub> totals use one fixed factor per ingredient and typical densities (table below).
                Real values vary by plant, region and transport distance.
              </li>
              <li>
                Environment limits come from ACI 318 as reported by secondary sources, and the materials
                shortlist is a reference table compiled by AI research agents. Most of its values are typical
                figures, not individually checked; the dots show which ones were seen in a paper.
              </li>
            </ul>
            <figure>
              <TestChart points={data.testPoints} />
              <figcaption>Each dot is a held-out lab result. On the dashed line means a perfect prediction.</figcaption>
            </figure>
          </div>
        ) : (
          <p className="note">Loading test metrics…</p>
        )}
      </section>

      {data && (
        <section>
          <h2>Where the CO<sub>2</sub> numbers come from</h2>
          <p className="note">{data.factorFile.note}</p>
          <div className="tablewrap">
            <table className="wrap">
              <thead>
                <tr>
                  <th>Ingredient</th><th className="num">kg CO<sub>2</sub>e per kg</th><th>Source</th>
                  <th className="num">Density (kg/m³)</th><th>Source</th>
                </tr>
              </thead>
              <tbody>
                {data.spec.ingredients.map((name) => {
                  const f = data.factors[name]
                  return (
                    <tr key={name}>
                      <td>{f.label}</td>
                      <td className="num">{f.kgCO2e_per_kg}</td>
                      <td>{f.co2_source}{f.co2_source_url && <> <a href={f.co2_source_url}>Link</a></>}</td>
                      <td className="num">{f.density_kg_m3}</td>
                      <td>{f.density_source}{f.density_source_url && <> <a href={f.density_source_url}>Link</a></>}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {data && (
        <section>
          <h2>Where the chemical formula comes from</h2>
          <p className="note">{data.oxides.note}</p>
          <div className="tablewrap">
            <table className="wrap">
              <thead>
                <tr>
                  <th>Binder</th><th className="num">CaO</th><th className="num">SiO<sub>2</sub></th>
                  <th className="num">Al<sub>2</sub>O<sub>3</sub></th><th className="num">Fe<sub>2</sub>O<sub>3</sub></th>
                  <th className="num">MgO</th><th className="num">SO<sub>3</sub></th><th>Evidence (mass percent)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.oxides.binders).map(([id, b]) => (
                  <tr key={id}>
                    <td>{b.label}</td>
                    {['CaO', 'SiO2', 'Al2O3', 'Fe2O3', 'MgO', 'SO3'].map((ox) => <td key={ox} className="num">{b[ox]}</td>)}
                    <td>{b.evidence} {b.citation}. <a href={b.url}>Link</a></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      </>
      )}

      <footer>
        {m && (
          <p>
            Model: {m.model}, trained {m.trained_at.slice(0, 10)} on {m.train_rows} lab results from{' '}
            {m.train_mixes} mixes. Everything on this page is computed in your browser.
          </p>
        )}
        <p>
          Data: Yeh, I-C. (1998). Concrete Compressive Strength. UCI Machine Learning Repository.{' '}
          <a href="https://doi.org/10.24432/C5PK67">doi.org/10.24432/C5PK67</a> ·{' '}
          <a href={REPO}>Source code on GitHub</a>
        </p>
      </footer>
    </main>
  )
}
