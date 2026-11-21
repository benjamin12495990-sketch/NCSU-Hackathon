import { HUMIDITY, FREEZE, WEAR } from '../lib/exposure.js'
import { OBJECTIVES } from '../lib/rank.js'

export const DEFAULT_REQ = {
  humidity: { on: false, level: 1 },
  freeze: { on: false, level: 1 },
  wear: { on: false, level: 1 },
  serviceTemp: { on: false, value: 200 },
  life: { on: false, years: 50 },
}

// Starting points for common kinds of work. They only switch sliders on and
// set first values. They are our choices, not values from a standard.
export const FIELDS = {
  general: { label: 'General (nothing switched on)', objective: 'co2', set: {} },
  civil: { label: 'Building or bridge', objective: 'co2', set: { humidity: { on: true, level: 1 }, life: { on: true, years: 50 } } },
  pavement: { label: 'Pavement or industrial floor', objective: 'co2', set: { humidity: { on: true, level: 1 }, freeze: { on: true, level: 2 }, wear: { on: true, level: 2 }, life: { on: true, years: 30 } } },
  marine: { label: 'Marine or coastal', objective: 'co2', set: { humidity: { on: true, level: 3 }, life: { on: true, years: 50 } } },
  aerospace: { label: 'Aerospace', objective: 'mass', set: { serviceTemp: { on: true, value: 150 }, humidity: { on: true, level: 1 }, life: { on: true, years: 30 } } },
  automotive: { label: 'Automotive or machinery', objective: 'mass', set: { serviceTemp: { on: true, value: 150 }, wear: { on: true, level: 2 }, life: { on: true, years: 20 } } },
  energy: { label: 'Energy or high temperature', objective: 'strength', set: { serviceTemp: { on: true, value: 800 }, life: { on: true, years: 30 } } },
  biomedical: { label: 'Biomedical or wet service', objective: 'mass', set: { humidity: { on: true, level: 2 }, wear: { on: true, level: 1 }, life: { on: true, years: 20 } } },
}

// One requirement: a switch that turns it on and a slider that sets how hard it is.
function Row({ name, on, onToggle, min, max, step, value, onValue, text }) {
  return (
    <div className={on ? 'req' : 'req off'}>
      <label className="switch">
        <input type="checkbox" checked={on} onChange={(e) => onToggle(e.target.checked)} /> {name}
      </label>
      <input type="range" aria-label={name} min={min} max={max} step={step} value={value} disabled={!on}
        onChange={(e) => onValue(Number(e.target.value))} />
      <span className="reqvalue">{on ? text : 'Not a requirement'}</span>
    </div>
  )
}

// The environment panel. Switch on only what the part has to survive; each
// switch enables its slider. Choosing a kind of work presets the switches.
export default function Requirements({ req, setReq, field, setField, objective, setObjective }) {
  const set = (key, patch) => setReq({ ...req, [key]: { ...req[key], ...patch } })
  const pickField = (id) => {
    setField(id)
    const next = Object.fromEntries(Object.entries(DEFAULT_REQ).map(([k, v]) => [k, { ...v, on: false }]))
    for (const [k, v] of Object.entries(FIELDS[id].set)) next[k] = { ...next[k], ...v }
    setReq(next)
    setObjective(FIELDS[id].objective)
  }
  return (
    <fieldset className="reqs">
      <legend>What does it have to survive?</legend>
      <div className="fields two">
        <div>
          <label htmlFor="field">What are you building?</label>
          <select id="field" value={field} onChange={(e) => pickField(e.target.value)}>
            {Object.entries(FIELDS).map(([id, f]) => <option key={id} value={id}>{f.label}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="objective">Rank other materials by</label>
          <select id="objective" value={objective} onChange={(e) => setObjective(e.target.value)}>
            {Object.entries(OBJECTIVES).map(([id, o]) => <option key={id} value={id}>{o.label}</option>)}
          </select>
        </div>
      </div>
      <Row name="Humidity and water" on={req.humidity.on} onToggle={(on) => set('humidity', { on })}
        min={1} max={3} step={1} value={req.humidity.level} onValue={(level) => set('humidity', { level })}
        text={HUMIDITY[req.humidity.level - 1].label} />
      <Row name="Freezing temperatures" on={req.freeze.on} onToggle={(on) => set('freeze', { on })}
        min={1} max={3} step={1} value={req.freeze.level} onValue={(level) => set('freeze', { level })}
        text={FREEZE[req.freeze.level - 1].label} />
      <Row name="Highest service temperature" on={req.serviceTemp.on} onToggle={(on) => set('serviceTemp', { on })}
        min={50} max={1500} step={50} value={req.serviceTemp.value} onValue={(value) => set('serviceTemp', { value })}
        text={`${req.serviceTemp.value} °C in continuous use`} />
      <Row name="Wear" on={req.wear.on} onToggle={(on) => set('wear', { on })}
        min={1} max={3} step={1} value={req.wear.level} onValue={(level) => set('wear', { level })}
        text={WEAR[req.wear.level - 1]} />
      <Row name="Service life" on={req.life.on} onToggle={(on) => set('life', { on })}
        min={10} max={100} step={10} value={req.life.years} onValue={(years) => set('life', { years })}
        text={`${req.life.years} years${req.life.years >= 50 ? ': asks for one more step of humidity and wear resistance' : ''}`} />
      <p className="note small">
        The kinds of work are starting points we chose. Freezing limits apply to the concrete design; the
        service temperature applies to the materials shortlist. The service-life rule is our rule of thumb.
      </p>
    </fieldset>
  )
}
