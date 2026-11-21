import { useState } from 'react'
import { OBJECTIVES, humidityNeed, wearNeed } from '../lib/rank.js'

const fmt = (v, digits = 0) => (v == null ? '–' : Number(v).toLocaleString(undefined, { maximumFractionDigits: digits }))

// One number from the reference table. A dot marks a value that was seen in a
// cited paper tonight; values without a dot are typical figures.
function Val({ m, field, digits, unit = '' }) {
  if (m[field] == null) return '–'
  const sourced = m.sourced.includes(field)
  return (
    <span title={sourced ? "Seen tonight in a search tool's summary of a cited paper" : 'Typical value, not individually checked'}>
      {fmt(m[field], digits)}{unit}{sourced && <span className="srcdot" aria-label="sourced"> ●</span>}
    </span>
  )
}

// Everything beyond the concrete design: which exposure limits were applied,
// the ranked shortlist of other engineering materials, and frontier materials.
export default function Shortlist({ result, data }) {
  const { rows, excluded, considered } = result.shortlist
  const [showAll, setShowAll] = useState(false) // the list opens on its ten best rows
  const shown = showAll ? rows : rows.slice(0, 10)
  const objective = OBJECTIVES[result.objective]
  const { req, limits } = result
  const left = considered - rows.length
  const reasons = [
    excluded.strength && `${excluded.strength} weaker than ${result.asked} MPa`,
    excluded.temperature && `${excluded.temperature} below ${req.serviceTemp.value} °C`,
    excluded.humidity && `${excluded.humidity} with a humidity rating under ${humidityNeed(req)}`,
    excluded.wear && `${excluded.wear} with a wear rating under ${wearNeed(req)}`,
  ].filter(Boolean)
  const tags = ['strength', 'low_carbon', req.humidity.on && 'humidity', req.wear.on && 'wear', (req.serviceTemp.on || req.freeze.on) && 'temperature'].filter(Boolean)
  const frontier = [...data.frontierFile.frontier]
    .map((f) => ({ ...f, match: f.helps_with.filter((t) => tags.includes(t)).length }))
    .sort((a, b) => b.match - a.match)

  return (
    <>
      {(limits.applied.length > 0 || limits.wear) && (
        <section>
          <h2>Environment limits applied to the concrete</h2>
          <div className="tablewrap">
            <table className="wrap">
              <thead>
                <tr><th>Rule</th><th>Condition</th><th className="num">Water / binder at most</th><th className="num">Strength at least</th><th>Evidence</th></tr>
              </thead>
              <tbody>
                {limits.applied.map((c) => (
                  <tr key={c.name}>
                    <td>ACI 318 class {c.name}</td>
                    <td>{c.condition}</td>
                    <td className="num">{c.max_w_cm ?? 'no limit'}</td>
                    <td className="num">{c.min_fc_MPa} MPa</td>
                    <td><span className={c.evidence === 'sourced' ? 'flag ok' : 'flag low'}>{c.evidence === 'sourced' ? `Seen in ${c.sources_seen} secondary source${c.sources_seen === 1 ? '' : 's'}` : 'Recalled, not confirmed'}</span></td>
                  </tr>
                ))}
                {limits.wear && (
                  <tr>
                    <td>Wear surface</td>
                    <td>{limits.wear.guidance}</td>
                    <td className="num">no limit</td>
                    <td className="num">{limits.wear.min_fc_MPa} MPa</td>
                    <td><span className="flag low">Recalled, not confirmed</span></td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="note small">{data.exposure.note}</p>
        </section>
      )}

      <section>
        <h2>Existing materials for comparison (not generated)</h2>
        <p className="note">
          These are existing materials, not generated ones: generating a new material needs a model trained
          on real test data, which this project has for concrete only.{' '}
          {rows.length} of {considered} reference materials pass, ranked by “{objective.label}”.
          {left > 0 && ` Left out: ${reasons.join(', ')}.`} Strength kinds differ between materials (yield,
          ultimate, compressive, flexural), so compare within a kind.
        </p>
        {rows.length > 0 && (
          <div className="tablewrap">
            <table className="compact">
              <thead>
                <tr>
                  <th>#</th><th>Material</th><th className="num">Strength (MPa)</th><th className="num">Density (kg/m³)</th>
                  <th className="num">Highest service temperature (°C)</th>
                  <th className="num">Humidity rating, out of 5 (AI judgment)</th>
                  <th className="num">Wear rating, out of 5 (AI judgment)</th>
                  <th className="num">CO<sub>2</sub>e per kg</th>
                  <th className="num">{objective.unit}</th><th>Source</th>
                </tr>
              </thead>
              <tbody>
                {shown.map(({ m, index }, i) => (
                  <tr key={m.name}>
                    <td>{i + 1}</td>
                    <td className="wraps"><strong>{m.name}</strong><span className="label">{m.family}</span></td>
                    <td className="num"><Val m={m} field="strength_MPa" /><span className="label">{m.strength_kind}</span></td>
                    <td className="num"><Val m={m} field="density_kg_m3" /></td>
                    <td className="num"><Val m={m} field="max_service_temp_C" /></td>
                    <td className="num" title={m.humidity_reason}>{m.humidity_resistance_1to5}</td>
                    <td className="num" title={m.wear_reason}>{m.wear_resistance_1to5}</td>
                    <td className="num"><Val m={m} field="embodied_CO2_kg_per_kg" digits={2} /></td>
                    <td className="num strong">{index == null ? 'no CO₂ figure' : fmt(index, index < 10 ? 2 : 0)}</td>
                    <td>{m.sources[0]?.url ? <a href={m.sources[0].url}>Paper</a> : '–'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {rows.length > 10 && (
          <p className="more">
            <button type="button" className="chip" aria-expanded={showAll} onClick={() => setShowAll(!showAll)}>
              {showAll ? 'Show the first 10 only' : `Show all ${rows.length} materials`}
            </button>
          </p>
        )}
        <p className="note small">{data.materialFile.note}</p>
      </section>

      <section>
        <h2>Research frontier: published by others, not generated here</h2>
        <p className="note">{data.frontierFile.note} Listed first: the ones most relevant to what you switched on.</p>
        <div className="cards">
          {frontier.map((f) => (
            <article key={f.name} className="mini">
              <h3>{f.name}</h3>
              <span className="flag plain">{f.readiness}</span>
              <p>{f.best_at}</p>
              <p className="headline">{f.headline_number}</p>
              <p className="fine">Barrier: {f.main_barrier}</p>
              <p className="fine"><a href={f.url}>Source paper</a></p>
              {f.note && (
                <details>
                  <summary>Evidence notes</summary>
                  <p className="fine">{f.note}</p>
                </details>
              )}
            </article>
          ))}
        </div>
      </section>
    </>
  )
}
