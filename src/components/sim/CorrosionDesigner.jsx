import { useEffect, useMemo, useRef, useState } from 'react'
import { loadPipeline, fmt } from '../../lib/pipeline.js'
import { validModel, prepare, predictAll, rank, judge, orderRecords, MEASURED_SLACK } from './corrosionInputs.js'
import './CorrosionDesigner.css'

// Corrosion designer (see README.md in this folder). The visitor names what the part will be exposed to (the environment, and
// optionally its concentration, its temperature and how long the exposure lasts) and the highest corrosion rate they accept. The
// page predicts every material of corrosion_model.json in that exposure with the exported trees, ranks them by the cautious upper
// estimate (the prediction plus the 90% half-width, in mm per year), says whether each is within the limit, and puts the measured
// records of that material in that environment beside it. Every number shown comes from that file (inputs are rebuilt by
// corrosionInputs.js exactly as the exporter built them); the constants below are only the limit chips, how many rows a page shows
// and thresholds of wording and rounding.

const PRESETS = ['0.1', '0.5', '1.25'] // limit chips, mm/yr: round numbers, not limits from a standard
const PAGE = 30 // rows of the ranked table shown at a time
const RECORDS_SHOWN = 8 // measured records listed before "show all"
const NAMES_SHOWN = 3 // names spelled out in a table row that folds several materials
const MILS_PER_MM = 1000 / 25.4
const SELDOM = 0.1 // "its cautious estimate is seldom within a tight limit" is said only when at most this share of the list is
const MAX_FIGURES = 6 // most significant figures a rate is printed with when two would put it on the wrong side of the limit

const verb = (n, one, many) => (n === 1 ? one : many)
const count = (n) => n.toLocaleString('en-US')
const reducedMotion = () => typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

// A number to `sig` significant figures in plain decimals: never 2.4e-7, and thousands separated from 1,000 up.
const plain = (v, sig) => Number(v.toPrecision(sig)).toLocaleString('en-US', { maximumSignificantDigits: sig })
// A rate 10^log mm/yr. Predictions get two significant figures (their 90% range is a factor of 40-odd, so more would mislead);
// measured rates get three, which is what the records carry.
const rate = (log, sig = 2) => (Number.isFinite(log) ? plain(10 ** log, sig) : '–')
// A predicted rate printed next to a limit (both log10 mm/yr): two significant figures, or as many more as it takes for the printed
// number to fall on the same side of the limit as the number itself, so "0.1" is never printed beside "not within 0.1".
function rateBy(log, limit) {
  if (!Number.isFinite(log) || limit === null) return rate(log)
  for (let sig = 2; sig < MAX_FIGURES; sig++) {
    if ((Math.log10(Number((10 ** log).toPrecision(sig))) <= limit) === (log <= limit)) return rate(log, sig)
  }
  return rate(log, MAX_FIGURES)
}
const measuredRate = (log) => rate(log, 3)
const hoursText = (h) => (h >= 10 ? Math.round(h).toLocaleString('en-US') : plain(h, 2))
const minus = (text) => String(text).replace('-', '−')
const concText = (c) => plain(c, 3)
const tempText = (t) => minus(plain(t, 4))
const spanText = (s, show) => (s.min === s.max ? show(s.min) : `${show(s.min)} to ${show(s.max)}`)
const ways = (n) => (n === 2 ? 'two' : n === 3 ? 'three' : count(n))
// Whether a number lies outside the span of the records. The export gives each exposure time back as 10^log10(hours), so an edge can
// come back a hair off (4,896 h as 4895.999999999999): a number equal to the edge but for that rounding counts as inside.
const EDGE = 1e-9
const below = (v, s) => v < s.min - EDGE * Math.max(1, Math.abs(s.min))
const above = (v, s) => v > s.max + EDGE * Math.max(1, Math.abs(s.max))
const outside = (v, s) => v !== null && (!s || below(v, s) || above(v, s))

// Plain decimals only ("10", "10.5", "10,5"), as on the concrete page: forms such as 1e3 or 0x10 are rejected, not guessed at.
const PLAIN = /^(\d+(\.\d*)?|\.\d+)$/
const clean = (text, unit) => String(text).trim().replace(unit, '').trim().replace(',', '.')

function parseConc(text, prep) {
  const t = clean(text, /\s*(vol\.?\s*)?%$/i)
  if (t === '') return { value: null, error: '' }
  if (!PLAIN.test(t)) return { value: null, error: 'Type the concentration in vol% as a plain number, for example 10, or leave it blank.' }
  const v = Number(t)
  const max = Math.max(100, prep.all.conc?.max ?? 100)
  if (v > max) return { value: null, error: `Choose a concentration between 0 and ${concText(max)} vol%, the highest in the records.` }
  return { value: v, error: '' }
}

function parseTemp(text, prep) {
  const raw = String(text).trim().toLowerCase()
  if (raw === '') return { value: null, word: null, error: '' }
  if (prep.qual.tempRoom >= 0 && /^(room|room temp(erature)?|ambient)$/.test(raw)) return { value: null, word: 'room', error: '' }
  if (prep.qual.tempBoiling >= 0 && /^(boiling|boil|boiling point|at boiling)$/.test(raw)) return { value: null, word: 'boiling', error: '' }
  const t = clean(raw, /\s*(°\s*c|deg(rees)?\s*c|c)$/i).replace(/^[−–]/, '-')
  if (!/^-?(\d+(\.\d*)?|\.\d+)$/.test(t)) {
    const words = [prep.qual.tempRoom >= 0 && '“room temperature”', prep.qual.tempBoiling >= 0 && '“boiling”'].filter(Boolean).join(' or ')
    return { value: null, word: null, error: `Type the temperature in °C as a plain number, for example 25${words ? `, or ${words}` : ''}, or leave it blank.` }
  }
  const v = Number(t)
  const s = prep.all.temp
  if (s && (below(v, s) || above(v, s))) {
    return { value: null, word: null, error: `Choose a temperature between ${tempText(s.min)} and ${tempText(s.max)} °C, the range of the records.` }
  }
  return { value: v, word: null, error: '' }
}

function parseHours(text, prep) {
  const t = clean(text, /\s*(hours?|hrs?|h)$/i)
  if (t === '') return { value: null, error: '' }
  if (!PLAIN.test(t)) return { value: null, error: 'Type the exposure time in hours as a plain number, for example 720, or leave it blank.' }
  const v = Number(t)
  const s = prep.all.hours
  if (!(v > 0) || (s && (below(v, s) || above(v, s)))) {
    return { value: null, error: s ? `Choose an exposure time between ${hoursText(s.min)} and ${hoursText(s.max)} hours, the range of the records.` : 'The exposure time must be more than 0 hours.' }
  }
  return { value: v, error: '' }
}

function parseLimit(text, prep) {
  const t = clean(text, /\s*mm\s*(\/|per)\s*(yr|year|y|a)$/i)
  if (t === '' || !PLAIN.test(t) || !(Number(t) > 0)) {
    return { log: null, error: 'Type the highest corrosion rate you accept, in mm per year, as a plain number above 0, for example 0.5.' }
  }
  const log = Math.log10(Number(t))
  const s = prep.all.y
  if (s && (log < s.min - MEASURED_SLACK || log > s.max + MEASURED_SLACK)) {
    return { log: null, error: `Choose a limit between ${measuredRate(s.min)} and ${measuredRate(s.max)} mm/yr, the range of the measured rates in the data.` }
  }
  return { log, value: Number(t), error: '' }
}

// "10 vol%, 24 °C, 48 h" for a record; what is not a number in the data is left out.
function recordWhere(r) {
  const parts = []
  if (r.conc !== null) parts.push(`${concText(r.conc)} vol%`)
  if (r.temp !== null) parts.push(`${tempText(r.temp)} °C`)
  if (r.hours !== null) parts.push(`${hoursText(r.hours)} h`)
  return parts.length ? parts.join(', ') : 'conditions not given as numbers'
}

// "17 records: 0.001 to 6 mm/yr, at 20 to 150 °C" for one material in one environment.
function measuredSummary(st) {
  const temps = st.temp ? `, at ${spanText(st.temp, tempText)} °C` : ', temperatures not given as numbers'
  return `${count(st.n)} ${verb(st.n, 'record', 'records')}: ${spanText(st, measuredRate)} mm/yr${temps}`
}

const rowName = (row, lead = row.mats[0]) => (row.mats.length === 1 ? lead.name : `${lead.name} and ${count(row.mats.length - 1)} more`)
// "Stainless steels · Ferritic" for a row. A folded row can hold several rare families, which the model pools into one input.
function rowKind(row) {
  const classes = [...new Set(row.mats.map((m) => m.cls))]
  const families = [...new Set(row.mats.map((m) => m.family))]
  return `${classes.length === 1 ? classes[0] : `${count(classes.length)} rare classes`} · ${families.length === 1 ? families[0] : `${count(families.length)} rare families`}`
}

// CORR-DATA spells some materials more than one way (a doubled space, a capital letter), and each spelling is a material of its own in
// the data. `twins` maps a material's id to the number of spellings of its name; the note says so beside every such row.
function twinNote(row, twins) {
  const hit = row.mats.filter((m) => twins.has(m.id))
  if (!hit.length) return null
  if (row.mats.length === 1) return `spelled ${ways(twins.get(hit[0].id))} ways in CORR-DATA, which holds them as separate materials`
  return `${count(hit.length)} of these names ${verb(hit.length, 'is', 'are')} spelled more than one way in CORR-DATA, which holds each spelling as a separate material`
}

// How the model knows a material: by an input of its own, or only through its class and family.
function knownAs(mat) {
  if (mat.own) return 'the model has an input of its own for this material'
  return mat.ownFamily ? 'no input of its own: predicted from its class and family' : 'no input of its own and a rare family: predicted mostly from its class'
}

// The searchable list of environments (a combobox): type to narrow it, arrow keys and Enter or a click to choose, Escape to leave.
function EnvPicker({ id, envs, value, onChange }) {
  const [open, setOpen] = useState(false)
  const [text, setText] = useState(null) // what the visitor is typing; null shows the chosen environment
  const [at, setAt] = useState(0)
  const listRef = useRef(null)
  const q = (text ?? '').trim().toLowerCase()
  const matches = useMemo(() => (q ? envs.filter((e) => e.name.toLowerCase().includes(q)) : envs), [envs, q])
  const hi = Math.min(at, Math.max(0, matches.length - 1))
  // Keep the highlighted option in view by scrolling the list itself (never the page).
  useEffect(() => {
    const list = listRef.current
    const li = open && list ? list.querySelector('[aria-selected="true"]') : null
    if (!li) return
    if (li.offsetTop < list.scrollTop) list.scrollTop = li.offsetTop
    else if (li.offsetTop + li.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = li.offsetTop + li.offsetHeight - list.clientHeight
  }, [open, hi, q])
  const openList = () => {
    if (open) return
    setOpen(true)
    setAt(Math.max(0, envs.findIndex((e) => e.name === value)))
  }
  const close = () => {
    setOpen(false)
    setText(null)
  }
  const choose = (e) => {
    onChange(e.name)
    close()
  }
  const onKeyDown = (ev) => {
    if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
      ev.preventDefault()
      if (!open) openList()
      else if (matches.length) setAt((hi + (ev.key === 'ArrowDown' ? 1 : matches.length - 1)) % matches.length)
    } else if (ev.key === 'Enter') {
      ev.preventDefault()
      if (open && matches[hi]) choose(matches[hi])
    } else if (ev.key === 'Escape' && open) {
      ev.preventDefault()
      close()
    }
  }
  return (
    <div className="cd-combo">
      <input id={id} role="combobox" aria-expanded={open} aria-controls={`${id}-list`} aria-autocomplete="list"
        aria-activedescendant={open && matches[hi] ? `${id}-opt-${hi}` : undefined}
        value={text ?? value} placeholder="Search the environments" autoComplete="off" spellCheck={false}
        onFocus={(e) => { e.target.select(); openList() }} onClick={(e) => { if (!open) e.target.select(); openList() }} onBlur={close} onKeyDown={onKeyDown}
        onChange={(e) => { setText(e.target.value); setOpen(true); setAt(0) }} />
      {open && (
        <ul id={`${id}-list`} ref={listRef} role="listbox" aria-label="Environments, most records first" className="cd-list"
          onMouseDown={(e) => e.preventDefault()}>
          {matches.length === 0 && <li role="presentation" className="cd-none">No environment in the data matches “{(text ?? '').trim()}”.</li>}
          {matches.map((e, i) => (
            <li key={e.name} id={`${id}-opt-${i}`} role="option" aria-selected={i === hi} className={e.name === value ? 'cd-current' : undefined}
              onMouseMove={() => { if (i !== hi) setAt(i) }} onClick={() => choose(e)}>
              <span>{e.name}</span>
              <span className="cd-count">{count(e.rows)} {verb(e.rows, 'record', 'records')}{e.own ? '' : ' · pooled'}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// A log-scale strip: the 90% range of the prediction, the prediction, the limit, every measured rate in this environment (all
// materials) and the selected material's own measured rates here. Nothing on it moves.
function Strip({ env, row, hw, limit, limitText }) {
  const W = 360
  const lo = Math.floor(Math.min(env.ys.length ? env.ys[0] : Infinity, row.pred - hw, limit))
  const hi = Math.ceil(Math.max(env.ys.length ? env.ys[env.ys.length - 1] : -Infinity, row.pred + hw, limit))
  const PAD = 34
  const x = (v) => PAD + ((v - lo) / (hi - lo)) * (W - 2 * PAD)
  const decade = (v) => (v < 0 ? `0.${'0'.repeat(-v - 1)}1` : (10 ** v).toLocaleString('en-US'))
  // One tick per decade, or every second, third... decade: the first spacing at which the longest label fits at the phone's 13px type.
  let ticks = []
  for (let step = 1; step <= hi - lo; step++) {
    ticks = []
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v)
    if (((W - 2 * PAD) / (hi - lo)) * step >= Math.max(...ticks.map((v) => decade(v).length)) * 7.2 + 10) break
  }
  const rug = useMemo(() => env.ys.map((y) => `M${(PAD + ((y - lo) / (hi - lo)) * (W - 2 * PAD)).toFixed(1)} 56V68`).join(''), [env, lo, hi])
  const own = row.st ? row.st.ys : []
  const flip = x(limit) > W - 56
  const label = `Prediction ${rateBy(row.pred, limit)} mm/yr, 90% range ${rate(row.pred - hw)} to ${rateBy(row.pred + hw, limit)} mm/yr; limit ${limitText}; ` +
    (own.length ? `${count(own.length)} measured ${verb(own.length, 'rate', 'rates')} of this material here, ${spanText(row.st, measuredRate)} mm/yr.` : 'no measured rate of this material here.')
  return (
    <svg className="cd-strip" viewBox={`0 0 ${W} 116`} role="img" aria-label={label}>
      <rect className="cd-band" x={x(row.pred - hw)} y={20} width={x(row.pred + hw) - x(row.pred - hw)} height={16} rx={2} />
      <line className="cd-center" x1={x(row.pred)} x2={x(row.pred)} y1={16} y2={40} />
      <path className="cd-rug" d={rug} />
      {own.map((y, i) => <circle key={i} className="cd-own" cx={x(y)} cy={62} r={3.5} />)}
      <line className="cd-limit" x1={x(limit)} x2={x(limit)} y1={8} y2={74} />
      <text x={flip ? x(limit) - 4 : x(limit) + 4} y={11} textAnchor={flip ? 'end' : 'start'}>limit</text>
      <line className="cd-axis" x1={PAD} x2={W - PAD} y1={78} y2={78} />
      {ticks.map((v) => (
        <g key={v}>
          <line className="cd-axis" x1={x(v)} x2={x(v)} y1={78} y2={82} />
          <text x={x(v)} y={95} textAnchor="middle">{decade(v)}</text>
        </g>
      ))}
      <text x={W / 2} y={112} textAnchor="middle">corrosion rate, mm per year (log scale)</text>
    </svg>
  )
}

// The flag of the "within the limit" column. `extra`: a condition the visitor typed is outside what this environment's records
// cover, where the 90% range was never checked, so the cautious estimate does not get the green flag.
function LimitFlag({ j, extra }) {
  if (j.cautious) return extra ? <span className="flag low">Cautious estimate (extrapolated)</span> : <span className="flag ok">Cautious estimate</span>
  if (j.expected) return <span className="flag low">Prediction only</span>
  return <span className="flag plain">No</span>
}

// The flag of the "measured in this environment" column. With `extra` the measured neighbours were all measured at other conditions
// than the ones asked for, so their agreement is stated without the green flag.
function MeasuredFlag({ j, extra }) {
  if (j.n === 0) return <span className="flag plain">No record here</span>
  if (j.agree) return <span className={extra ? 'flag plain' : 'flag ok'}>Measured neighbours agree</span>
  if (j.disagree) return <span className="flag low">Measured neighbours disagree</span>
  return <span className="flag plain">{j.within === j.n ? 'Measured within the limit' : 'Measured above the limit'}</span>
}

const aboveText = (j) => (j.within === j.n ? '' : `${count(j.n - j.within)} of ${count(j.n)} above the limit · `)

// The panel for the selected row: where the concrete page has its 3D stage.
function Detail({ item, total, env, ask, hw, limit, limitText, extra, twins, open, setOpen }) {
  const { row, j } = item
  const mat = item.lead
  const folded = row.mats.length > 1
  const twin = twinNote(row, twins)
  const ordered = useMemo(() => (row.st ? orderRecords(env.byMat.get(mat.id) ?? [], env, ask) : null), [row, env, mat, ask])
  // What else the data holds when this material has no record here: its own records elsewhere, and its family's records here.
  const context = useMemo(() => {
    if (row.st) return null
    const ids = new Set(row.mats.map((m) => m.id))
    const elsewhere = new Map()
    for (const m of row.mats) for (const [name, n] of m.envs) elsewhere.set(name, (elsewhere.get(name) ?? 0) + n)
    const families = new Set(row.mats.map((m) => `${m.cls}|${m.family}`))
    const kin = env.records.filter((r) => families.has(`${r.mat.cls}|${r.mat.family}`) && !ids.has(r.mat.id))
    const ys = kin.map((r) => r.y).sort((a, b) => a - b)
    return {
      elsewhere: [...elsewhere].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)), total: [...elsewhere.values()].reduce((a, b) => a + b, 0),
      ys, within: ys.filter((y) => y <= limit + MEASURED_SLACK).length, materials: new Set(kin.map((r) => r.mat.id)).size,
      family: families.size === 1 ? `${mat.family} (${mat.cls})` : 'the same families',
    }
  }, [row, env, mat, limit])
  const list = ordered ? (open ? ordered.list : ordered.list.slice(0, RECORDS_SHOWN)) : []
  return (
    <div className="cd-detail">
      <span className="label">Rank {item.rank} of {count(total)} by the cautious upper estimate</span>
      <h2>{rowName(row, mat)}</h2>
      <p className="cd-how">{rowKind(row)} · {folded ? 'no inputs of their own' : knownAs(mat)}</p>
      {twin && <p className="cd-how">{folded ? twin : `This name is ${twin}`}.</p>}
      {folded && (
        <p className="cd-how">
          The model gives these {count(row.mats.length)} materials the same inputs, so it cannot tell them apart, and none of them has a
          record in {env.name}: {row.mats.map((m) => m.name).join('; ')}.
        </p>
      )}
      <div className="stats">
        <div>
          <span className="label">Prediction</span>
          <span className="big">{rateBy(row.pred, limit)} <small>mm/yr</small></span>
          <span className="label">log₁₀ of the rate: {minus(plain(row.pred, 3))}</span>
        </div>
        <div>
          <span className="label">Cautious upper estimate</span>
          <span className="big">{rateBy(row.pred + hw, limit)} <small>mm/yr</small></span>
          <span className="label">90% range from {rate(row.pred - hw)} mm/yr</span>
        </div>
      </div>
      <p className="cd-flags"><LimitFlag j={j} extra={extra} /> <MeasuredFlag j={j} extra={extra} /></p>
      <Strip env={env} row={row} hw={hw} limit={limit} limitText={limitText} />
      <p className="fine">
        Band: the 90% range of the prediction. Accent line: the prediction.{' '}
        {env.ys.length === 0 ? `Grey ticks: none, the data has no measured rate in ${env.name}`
          : env.ys.length === 1 ? `Grey tick: the one measured rate in ${env.name}`
            : `Grey ticks: all ${count(env.ys.length)} measured rates in ${env.name}, every material`}.
        {' '}{!row.st ? 'Circles: none, because this material has no record there'
          : row.st.n === 1 ? 'Circle: the one measured rate of this material there'
            : `Circles: the ${count(row.st.n)} measured rates of this material there`}. Dashed: your limit.
      </p>
      {ordered ? (
        <>
          <h3>Measured in {env.name}: {count(row.st.n)} {verb(row.st.n, 'record', 'records')}</h3>
          <p className="cd-how">
            {j.within === j.n
              ? `${j.n === 1 ? 'The measured rate is' : `All ${count(j.n)} measured rates are`} within your limit.`
              : `${count(j.n - j.within)} of the ${count(j.n)} measured ${verb(j.n, 'rate', 'rates')} ${verb(j.n - j.within, 'is', 'are')} above your limit.`}
            {j.n > 1 && ` They are all its records here, whatever the concentration, temperature and exposure time; ${ordered.closest ? 'closest to your conditions first' : 'highest rate first'}.`}
          </p>
          <div className="sim-scroll">
            <table className="sim-table cd-records">
              <thead>
                <tr>
                  <th className="num">Conc. (vol%)</th><th className="num">Temp. (°C)</th><th className="num">Exposure (h)</th>
                  <th className="num">Measured rate (mm/yr)</th><th>Your limit</th><th className="num">CORR-DATA ref.</th>
                </tr>
              </thead>
              <tbody>
                {list.map((r, i) => (
                  <tr key={i}>
                    <td className="num">{r.conc === null ? '–' : concText(r.conc)}</td>
                    <td className="num">{r.temp === null ? '–' : tempText(r.temp)}</td>
                    <td className="num">{r.hours === null ? '–' : hoursText(r.hours)}</td>
                    <td className="num strong">{measuredRate(r.y)}</td>
                    <td>{r.y <= limit + MEASURED_SLACK ? 'within' : <span className="flag low">above</span>}</td>
                    <td className="num">{r.ref || '–'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {ordered.list.length > RECORDS_SHOWN && (
            <p className="more">
              <button type="button" className="chip" onClick={() => setOpen(!open)}>
                {open ? `Show the first ${RECORDS_SHOWN}` : `Show all ${count(ordered.list.length)} records`}
              </button>
            </p>
          )}
          <p className="fine">
            –: not given as a number in the data. For a temperature that includes records made at room temperature or at boiling; for a
            concentration, ones given in another unit or in words.
          </p>
        </>
      ) : (
        <>
          <h3>No record in {env.name}: pure extrapolation</h3>
          <p className="cd-how">
            {folded ? `None of these ${count(row.mats.length)} materials has a` : `${mat.name} has no`} measured rate in {env.name} in the data, so nothing
            measured stands behind this prediction. The model has seen {folded ? 'them' : 'it'} only in other environments
            {context.total > 0 && ` (${count(context.total)} ${verb(context.total, 'record', 'records')}: ${context.elsewhere.slice(0, 3).map(([name, n]) => `${name} ${count(n)}`).join(', ')}${context.elsewhere.length > 3 ? `, and ${count(context.elsewhere.length - 3)} more ${verb(context.elsewhere.length - 3, 'environment', 'environments')}` : ''})`}
            , and other materials in this one.
          </p>
          <p className="cd-how">
            {context.ys.length > 0
              ? `Other materials of ${context.family} do have records here: ${count(context.ys.length)} measured ${verb(context.ys.length, 'rate', 'rates')} of ${count(context.materials)} ${verb(context.materials, 'material', 'materials')}, ${context.ys.length === 1 ? measuredRate(context.ys[0]) : `${measuredRate(context.ys[0])} to ${measuredRate(context.ys[context.ys.length - 1])}`} mm/yr, ${count(context.within)} of ${count(context.ys.length)} within your limit.`
              : `No other material of ${context.family} has a record here either.`}
          </p>
        </>
      )}
    </div>
  )
}

// A condition the visitor typed that lies outside what the environment's records cover, in words: what was typed against what the
// records hold, then where in the data such values do occur (never in this environment, by construction).
function outText(o, envName) {
  const first = o.range
    ? `Your ${o.what} (${o.value}) is outside every record in ${envName} (${o.range}).`
    : `No record in ${envName} gives ${o.none}, so your ${o.what} (${o.value}) is outside every record there.`
  if (o.n === 0) return first
  const where = o.n === 1 ? `one record, in ${o.top[0][0]}`
    : o.envs === 1 ? `${count(o.n)} records, all in ${o.top[0][0]}`
      : `${count(o.n)} records, ${count(o.top[0][1])} of them in ${o.top[0][0]} and ${count(o.top[1][1])} in ${o.top[1][0]}`
  return `${first} The model has seen ${o.kind} only in other environments (${where}).`
}

export default function CorrosionDesigner({ tasks = [], index }) {
  const [model, setModel] = useState(undefined)
  const [envName, setEnvName] = useState(null) // null: the environment with the most records
  const [concIn, setConcIn] = useState('')
  const [tempIn, setTempIn] = useState('')
  const [hoursIn, setHoursIn] = useState('')
  const [limitIn, setLimitIn] = useState(PRESETS[1])
  const [cls, setCls] = useState('any')
  const [fold, setFold] = useState(true)
  const [onlyWithin, setOnlyWithin] = useState(false)
  const [onlyMeasured, setOnlyMeasured] = useState(false)
  const [sel, setSel] = useState(null) // id of the selected material; null follows the top of the list
  const [more, setMore] = useState({ key: '', n: PAGE })
  const [openRecords, setOpenRecords] = useState('')
  const [kept, setKept] = useState(null) // the last request the form held with every field usable: the one the result is for
  const detailRef = useRef(null)

  useEffect(() => {
    const ctl = new AbortController()
    loadPipeline('corrosion_model.json', ctl.signal).then(setModel, () => {})
    return () => ctl.abort()
  }, [])

  const prep = useMemo(() => {
    if (!model) return null
    try {
      return validModel(model) ? prepare(model) : { broken: true }
    } catch {
      return { broken: true }
    }
  }, [model])
  const ready = !!prep && !prep.broken
  const hw = ready ? prep.hw : null
  const envNow = ready ? prep.envByName.get(envName) ?? prep.environments[0] : null // the environment in the picker
  const pc = ready ? parseConc(concIn, prep) : null
  const pt = ready ? parseTemp(tempIn, prep) : null
  const ph = ready ? parseHours(hoursIn, prep) : null
  const pl = ready ? parseLimit(limitIn, prep) : null
  const clsUsed = ready && cls !== 'any' && prep.classes.some((k) => k.name === cls) ? cls : 'any'

  // What the form holds right now, or null while a field holds a value the page cannot use. The last usable request is kept (state
  // adjusted while rendering, React's way of remembering a value from an earlier render), so a half-typed number such as "-" does not
  // take the result and the table off the screen: they stay, marked as not updated, until the field is usable again.
  const now = ready && !(pc.error || pt.error || ph.error || pl.error)
    ? { environment: envNow.name, conc: pc.value, temp: pt.value, tempWord: pt.word, hours: ph.value, limit: pl.log, limitValue: pl.value }
    : null
  const nowKey = now ? JSON.stringify(now) : ''
  if (now && kept?.key !== nowKey) {
    setKept({
      key: nowKey, env: envNow, limit: now.limit, limitValue: now.limitValue,
      ask: { environment: now.environment, conc: now.conc, temp: now.temp, tempWord: now.tempWord, hours: now.hours },
    })
  }
  const stale = !now && !!kept
  const env = kept?.env ?? null // the environment, the conditions and the limit of the result on screen
  const ask = kept?.ask ?? null
  const limit = kept?.limit ?? null

  // The main interaction: one walk of the trees per group of materials that share all inputs (under a millisecond in total).
  const preds = useMemo(() => (ask ? predictAll(model, prep, ask) : null), [model, prep, ask])
  const rows = useMemo(() => (preds ? rank(prep, preds, env, clsUsed, fold) : null), [prep, preds, env, clsUsed, fold])
  const view = useMemo(() => {
    if (!rows || limit === null) return null
    const judged = rows.map((row, i) => ({ row, rank: i + 1, j: judge(row, limit, hw) }))
    const c = { materials: 0, cautious: 0, cautiousNoRecord: 0, predOnly: 0, measured: 0, agree: 0, disagree: 0, rates: 0, ratesWithin: 0 }
    let lowestMeasured = null
    for (const x of judged) {
      const n = x.row.mats.length
      c.materials += n
      if (x.j.cautious) {
        c.cautious += n
        if (!x.row.st) c.cautiousNoRecord += n
      } else if (x.j.expected) c.predOnly += n
      if (!x.row.st) continue
      c.measured++
      c.rates += x.j.n
      c.ratesWithin += x.j.within
      if (x.j.agree) c.agree++
      if (x.j.disagree) c.disagree++
      const a = x.row.st
      const b = lowestMeasured?.row.st
      if (!b || a.max < b.max || (a.max === b.max && (a.n > b.n || (a.n === b.n && a.median < b.median)))) lowestMeasured = x
    }
    const shown = judged.filter((x) => (!onlyWithin || x.j.expected) && (!onlyMeasured || x.row.st))
    return { judged, shown, counts: c, top: judged[0] ?? null, topAgree: judged.find((x) => x.j.agree) ?? null, lowestMeasured }
  }, [rows, limit, hw, onlyWithin, onlyMeasured])
  const item = useMemo(() => {
    if (!view) return null
    const x = (sel !== null && view.judged.find((k) => k.row.mats.some((m) => m.id === sel))) || view.shown[0] || view.top
    return x ? { ...x, lead: x.row.mats.find((m) => m.id === sel) ?? x.row.mats[0] } : null
  }, [view, sel])
  const viewKey = kept ? JSON.stringify([kept.key, clsUsed, fold, onlyWithin, onlyMeasured]) : ''
  const shownN = more.key === viewKey ? more.n : PAGE
  const hasNumbers = !!ask && (ask.conc !== null || ask.temp !== null || ask.hours !== null)
  // The rows on screen, each with the measured record closest to the visitor's conditions (when they gave any number).
  const page = useMemo(() => {
    if (!view) return []
    const told = (r) => !!r && ((ask.conc !== null && r.conc !== null) || (ask.temp !== null && r.temp !== null) || (ask.hours !== null && r.hours !== null))
    return view.shown.slice(0, shownN).map((x) => {
      const near = x.row.st && x.row.st.n > 1 && hasNumbers ? orderRecords(env.byMat.get(x.row.mats[0].id) ?? [], env, ask).list[0] : null
      return { ...x, near: told(near) ? near : null }
    })
  }, [view, shownN, hasNumbers, env, ask])
  // Materials whose names differ only by spacing or by capitals: id -> how many spellings the data holds.
  const twins = useMemo(() => {
    const out = new Map()
    if (!ready) return out
    const byName = new Map()
    for (const m of prep.materials) {
      const key = m.name.replace(/\s+/g, ' ').trim().toLowerCase()
      if (!byName.has(key)) byName.set(key, [])
      byName.get(key).push(m.id)
    }
    for (const ids of byName.values()) if (ids.length > 1) for (const id of ids) out.set(id, ids.length)
    return out
  }, [ready, prep])
  // The numbers the visitor typed that are outside what this environment's records cover (or that none of them gives), each with
  // where in the data such values do occur. There the model extrapolates and its 90% range was never checked.
  const outs = useMemo(() => {
    if (!ask || !env) return []
    const dims = [
      { v: ask.conc, s: env.conc, get: (r) => r.conc, what: 'concentration', num: concText, unit: 'vol%', above: 'concentrations above', below: 'concentrations below', any: 'concentrations in vol%', none: 'a concentration in vol%' },
      { v: ask.temp, s: env.temp, get: (r) => r.temp, what: 'temperature', num: tempText, unit: '°C', above: 'temperatures above', below: 'temperatures below', any: 'temperatures given as numbers', none: 'a temperature as a number' },
      { v: ask.hours, s: env.hours, get: (r) => r.hours, what: 'exposure time', num: hoursText, unit: 'h', above: 'exposures longer than', below: 'exposures shorter than', any: 'exposure times', none: 'an exposure time' },
    ]
    return dims.filter((d) => outside(d.v, d.s)).map((d) => {
      const over = !!d.s && above(d.v, d.s)
      const beyond = !d.s ? () => true : over ? (x) => above(x, d.s) : (x) => below(x, d.s)
      const by = []
      let n = 0
      for (const e of prep.environments) {
        let k = 0
        for (const r of e.records) {
          const x = d.get(r)
          if (x !== null && beyond(x)) k++
        }
        if (k) by.push([e.name, k])
        n += k
      }
      by.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
      return {
        what: d.what, value: `${d.num(d.v)} ${d.unit}`, range: d.s ? `${spanText(d.s, d.num)} ${d.unit}` : null, none: d.none,
        kind: !d.s ? d.any : over ? `${d.above} ${d.num(d.s.max)} ${d.unit}` : `${d.below} ${d.num(d.s.min)} ${d.unit}`,
        n, envs: by.length, top: by.slice(0, 2),
      }
    })
  }, [prep, env, ask])

  if (model === undefined) return <section className="card fam-section"><p className="note">Loading the corrosion model…</p></section>
  if (!ready) {
    return (
      <section className="card fam-section cd">
        <h2>Rank materials for an exposure</h2>
        <p className="note">
          {model === null ? 'The corrosion model has not been exported to the site yet, so the designer cannot run.' : 'The corrosion model file is incomplete, so the designer cannot run.'}
          {' '}On the training laptop: <code>python -m gpu.export_site --browser-models-only</code>, then commit <code>public/pipeline/corrosion_model.json</code>.
          {tasks.length > 0 && ' The pipeline’s own test results for corrosion rates are below.'}
        </p>
      </section>
    )
  }

  const cv = model.cv
  const med = prep.medians
  const exploratory = tasks.find((t) => t?.exploratory)?.exploratory ?? null
  const times = fmt(10 ** hw, 2)
  // Selecting a material brings its panel into view. On a narrow screen the panel sits under the headline and the three picks.
  const select = (row) => {
    setSel(row.mats[0].id)
    detailRef.current?.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'start' })
  }
  const isSel = (x) => !!item && !!x && item.row.key === x.row.key
  // In the form, a number outside what the environment's records cover (or where none of them gives that number) is flagged at once.
  const hintClass = (v, s) => (outside(v, s) ? 'cd-hint cd-out' : 'cd-hint')
  const rangeHint = (v, s, show, unit, none, records = 'records') => (s ? `${outside(v, s) ? `outside the ${records} here` : `${records} here`}: ${spanText(s, show)} ${unit}` : `${none}${v !== null ? ', so the model is extrapolating' : ''}`)
  const condParts = ask ? [
    ask.conc !== null ? `${concText(ask.conc)} vol%` : null,
    ask.temp !== null ? `${tempText(ask.temp)} °C` : ask.tempWord === 'room' ? 'room temperature' : ask.tempWord === 'boiling' ? 'boiling' : null,
    ask.hours !== null ? `${hoursText(ask.hours)} h` : null,
  ].filter(Boolean) : []
  const blankParts = ask ? [ask.conc === null && 'concentration', ask.temp === null && ask.tempWord === null && 'temperature', ask.hours === null && 'exposure time'].filter(Boolean) : []
  const blankText = blankParts.length > 1 ? `${blankParts.slice(0, -1).join(', ')} and ${blankParts[blankParts.length - 1]}` : blankParts[0]
  const limitShown = kept ? `${plain(kept.limitValue, MAX_FIGURES)} mm/yr` : ''
  const extra = outs.length > 0
  const shownRate = (log) => rateBy(log, limit)
  const c = view?.counts
  const pool = clsUsed === 'any' ? 'materials' : `materials in “${clsUsed}”`
  const onePool = clsUsed === 'any' ? 'material' : `material in “${clsUsed}”`
  // How many of the materials with a cautious estimate within the limit have nothing measured here to stand behind it.
  const noRecordText = !c || c.cautious === 0 ? ''
    : c.cautious === 1 ? (c.cautiousNoRecord ? ' (it has no record here)' : ' (it has a record here)')
      : c.cautiousNoRecord === 0 ? ' (all of them with a record here)'
        : c.cautiousNoRecord === c.cautious ? ' (none of them with a record here)'
          : ` (${count(c.cautiousNoRecord)} of them with no record here)`
  const pooledEnvs = prep.environments.length - prep.nOwnEnvironments
  const detailKey = item ? `${env.name}|${item.row.key}` : ''
  const errors = [pc.error, pt.error, ph.error, pl.error].filter(Boolean)
  // Shown above the result and above the table while a field holds a value the page cannot use (announced once, by the first).
  const notUpdated = (announce) => stale && (
    <p className="cd-how" role={announce ? 'status' : undefined}>
      <span className="flag low">Not updated</span> A field above holds a value the page cannot use. This is still the result for the last
      usable entries.
    </p>
  )
  const pick = (x, title, body, none) => (
    <button type="button" className="pick" disabled={!x} aria-pressed={isSel(x)} onClick={() => x && select(x.row)}>
      <span className="label">{title}</span>
      {x ? body(x) : <span className="label">{none}</span>}
    </button>
  )

  return (
    <>
      <section className="card fam-section cd">
        <h2>Rank materials for an exposure</h2>
        {index?.rehearsal && (
          <p className="note">
            The rehearsal flag above is about the pipeline&apos;s results further down this page, not about this designer. The designer runs
            on its own model file, which the exporter trains on the full data whatever the nightly run was.
          </p>
        )}
        <form onSubmit={(e) => e.preventDefault()}>
          <div className="cd-form">
            <div className="cd-env">
              <label htmlFor="cd-env">Environment</label>
              <EnvPicker id="cd-env" envs={prep.environments} value={envNow.name} onChange={setEnvName} />
              <span className={envNow.own ? 'cd-hint' : 'cd-hint cd-out'}>
                {count(envNow.rows)} measured {verb(envNow.rows, 'rate', 'rates')} of {count(envNow.stats.size)} {verb(envNow.stats.size, 'material', 'materials')} in the data
                {envNow.own ? '' : '. Too few for an input of its own: the model predicts it as “a rare environment”'}
              </span>
            </div>
            <div className="cd-limit">
              <label htmlFor="cd-limit">Highest rate you accept (mm/yr)</label>
              <input id="cd-limit" value={limitIn} onChange={(e) => setLimitIn(e.target.value)} inputMode="decimal" autoComplete="off" placeholder="e.g. 0.5" />
              <span className="cd-hint">{pl.log === null ? 'millimetres of metal lost per year' : `about ${plain(pl.value * MILS_PER_MM, 2)} mils per year`}</span>
            </div>
            <div className="cd-class">
              <label htmlFor="cd-class">Material class</label>
              <select id="cd-class" value={clsUsed} onChange={(e) => setCls(e.target.value)}>
                <option value="any">Any class · {count(prep.materials.length)}</option>
                {prep.classes.map((k) => <option key={k.name} value={k.name}>{k.name} · {count(k.n)}</option>)}
              </select>
              <span className="cd-hint">as classified in CORR-DATA</span>
            </div>
            <div className="cd-third">
              <label htmlFor="cd-conc">Concentration (vol%, optional)</label>
              <input id="cd-conc" value={concIn} onChange={(e) => setConcIn(e.target.value)} inputMode="decimal" autoComplete="off" placeholder="blank: not specified" />
              <span className={hintClass(pc.value, envNow.conc)}>
                {rangeHint(pc.value, envNow.conc, concText, 'vol%', 'no record here gives a concentration in vol%')}
              </span>
            </div>
            <div className="cd-third">
              <label htmlFor="cd-temp">Temperature (°C, optional)</label>
              <input id="cd-temp" value={tempIn} onChange={(e) => setTempIn(e.target.value)} autoComplete="off" spellCheck={false} placeholder="blank: not specified" />
              <span className={hintClass(pt.value, envNow.temp)}>
                {rangeHint(pt.value, envNow.temp, tempText, '°C', 'no record here gives a temperature as a number')}
              </span>
              {(prep.qual.tempRoom >= 0 || prep.qual.tempBoiling >= 0) && (
                <span className="cd-words">
                  <span>or</span>
                  {prep.qual.tempRoom >= 0 && (
                    <button type="button" className="chip" aria-pressed={pt.word === 'room'} onClick={() => setTempIn(pt.word === 'room' ? '' : 'room temperature')}>Room temperature</button>
                  )}
                  {prep.qual.tempBoiling >= 0 && (
                    <button type="button" className="chip" aria-pressed={pt.word === 'boiling'} onClick={() => setTempIn(pt.word === 'boiling' ? '' : 'boiling')}>Boiling</button>
                  )}
                </span>
              )}
            </div>
            <div className="cd-third">
              <label htmlFor="cd-hours">Exposure time (hours, optional)</label>
              <input id="cd-hours" value={hoursIn} onChange={(e) => setHoursIn(e.target.value)} inputMode="decimal" autoComplete="off" placeholder="blank: not specified" />
              <span className={hintClass(ph.value, envNow.hours)}>
                the length of the test that measured a rate, not a service life;{' '}
                {rangeHint(ph.value, envNow.hours, hoursText, 'h', 'no record here gives one', 'tests')}
              </span>
            </div>
          </div>
          <div className="examples">
            <span>Limit:</span>
            {PRESETS.map((p) => (
              <button type="button" key={p} className="chip" aria-pressed={pl.log !== null && pl.value === Number(p)} onClick={() => setLimitIn(p)}>{p} mm/yr</button>
            ))}
          </div>
          <p className="note small">
            The list updates as you type. A field left blank is not “any”: the model is told the number was not given and uses the
            median of the data in its place ({concText(med.conc_vol_pct)} vol%, {tempText(med.temp_C_mid)} °C, {hoursText(10 ** med.log10_hours)} h), as it
            was trained on the records without that number. The three limits are round numbers, not limits from a standard.
          </p>
        </form>
        {errors.map((e) => <p key={e} className="error" role="alert">{e}</p>)}
      </section>

      {view && item && (
        <>
          <section className="result cd-result">
            <div className="stagewrap" ref={detailRef}>
              <Detail item={item} total={view.judged.length} env={env} ask={ask} hw={hw} limit={limit} limitText={limitShown} extra={extra} twins={twins}
                open={openRecords === detailKey} setOpen={(v) => setOpenRecords(v ? detailKey : '')} />
            </div>
            <div className="readout">
              {notUpdated(true)}
              <h2>{env.name}: at most {limitShown}</h2>
              <p className="cd-how">
                {condParts.length ? condParts.join(', ') : 'No conditions given'}
                {blankParts.length > 0 && condParts.length > 0 && `; ${blankText} not specified`}
                {clsUsed !== 'any' && ` · ${clsUsed} only`}
              </p>
              {exploratory && <p className="cd-how"><span className="flag low">Exploratory</span> {exploratory}</p>}
              {extra && (
                <p className="warn cd-extra">
                  {outs.map((o) => outText(o, env.name)).join(' ')} So these predictions are extrapolated, and the 90% range was only checked
                  on conditions like the records: the cautious estimate is not cautious here.
                </p>
              )}
              <div className="picks">
                {pick(view.top, 'Lowest cautious upper estimate (a prediction)', (x) => (
                  <>
                    <span className="mixname">{rowName(x.row)}</span>
                    <span className="big">{shownRate(x.row.pred + hw)} <small>mm/yr, cautious upper estimate{extra ? ' (extrapolated)' : ''}</small></span>
                    <span className="label">prediction {shownRate(x.row.pred)} mm/yr · {x.row.st ? `measured here, ${measuredSummary(x.row.st)}` : 'no record here: pure extrapolation'}</span>
                  </>
                ), 'no material to rank')}
                {pick(view.topAgree, 'Lowest estimate whose measured neighbours agree', (x) => (
                  <>
                    <span className="mixname">{rowName(x.row)}</span>
                    <span className="big">{shownRate(x.row.pred + hw)} <small>mm/yr, cautious upper estimate{extra ? ' (extrapolated)' : ''}</small></span>
                    <span className="label">prediction {shownRate(x.row.pred)} mm/yr · measured here, {measuredSummary(x.row.st)}</span>
                  </>
                ), `none: no ${onePool} has both a prediction within the limit and every measured rate here within it`)}
                {pick(view.lowestMeasured, 'Lowest worst measured rate here (measured, not a prediction)', (x) => (
                  <>
                    <span className="mixname">{rowName(x.row)}</span>
                    <span className="big">{measuredRate(x.row.st.max)} <small>mm/yr, {x.row.st.n === 1 ? 'its one measured rate' : `highest of ${count(x.row.st.n)} measured`}</small></span>
                    <span className="label">the model&apos;s prediction at your conditions: {shownRate(x.row.pred)} mm/yr, cautious upper estimate {shownRate(x.row.pred + hw)} mm/yr</span>
                  </>
                ), `none: no ${onePool} has a record in this environment`)}
              </div>
              <p className="note">
                Of the {count(c.materials)} {c.materials === 1 ? onePool : pool}, {count(c.cautious)} {verb(c.cautious, 'has', 'have')} a cautious upper estimate within the
                limit{noRecordText} and {count(c.predOnly)} more {verb(c.predOnly, 'is', 'are')} within it on the prediction only. {count(c.measured)} {verb(c.measured, 'has', 'have')} a
                measured record in {env.name}
                {c.measured > 0 && ` (${count(c.rates)} measured ${verb(c.rates, 'rate', 'rates')}, ${count(c.ratesWithin)} within the limit); for ${count(c.agree)} of them the measured neighbours agree with a prediction within the limit, and for ${count(c.disagree)} they disagree`}.
              </p>
              {!env.own && (
                <p className="warn">
                  The model has no input of its own for {env.name}: its {env.rows === 1 ? 'one record is' : `${count(env.rows)} records are`} too few. It predicts it as “a rare
                  environment”, with the same numbers for all {count(pooledEnvs)} such environments. Only the measured records are specific to {env.name}.
                </p>
              )}
              <p className="warn">
                The 90% range of a prediction is ±{fmt(hw)} log units: a factor of about {times} either way. The model can point to the
                kinds of material that tend to corrode slowly in an environment, but it cannot rank close
                candidates{c.cautious <= SELDOM * c.materials ? ', and its cautious estimate is seldom within a tight limit' : ''}. Read the list as
                materials worth checking against the measured records, not as results.
              </p>
            </div>
          </section>

          <section className="card fam-section cd">
            <h2>{count(c.materials)} {c.materials === 1 ? onePool : pool}, ranked by the cautious upper estimate</h2>
            {notUpdated(false)}
            <p className="note">
              Every material was predicted in your browser for {env.name}{condParts.length ? ` at ${condParts.join(', ')}` : ''}
              {blankParts.length > 0 && ` (${blankText} not specified)`}. Lowest first. Every prediction gets the same 90% range, so this is
              also the order of the predictions. Materials the model gives the same inputs get the same numbers; a tie goes to the one with
              more measured records here.{fold && view.judged.some((x) => x.row.mats.length > 1) && ' Those with the same inputs and no record here share one row.'}
            </p>
            <div className="options cd-options">
              <label><input type="checkbox" checked={onlyWithin} onChange={(e) => setOnlyWithin(e.target.checked)} /> <span>Only materials predicted within the limit</span></label>
              <label><input type="checkbox" checked={onlyMeasured} onChange={(e) => setOnlyMeasured(e.target.checked)} /> <span>Only materials with a measured record in this environment</span></label>
              <label><input type="checkbox" checked={fold} onChange={(e) => setFold(e.target.checked)} /> <span>One row for materials the model cannot tell apart (same inputs, no record here)</span></label>
            </div>
            {view.shown.length === 0 ? (
              <p className="note">
                No {onePool} is left with these options.
                {onlyWithin && ` None has a prediction within ${limitShown}${onlyMeasured ? ' and a measured record here' : ''}: raise the limit or untick an option.`}
                {!onlyWithin && onlyMeasured && ` None has a measured record in ${env.name}: untick that option to see the extrapolations.`}
              </p>
            ) : (
              <>
                <div className="tablewrap sim-scroll">
                  <table className="ranked cd-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Material</th>
                        <th className="num">Prediction (mm/yr)</th>
                        <th className="num">Cautious upper estimate (mm/yr)</th>
                        <th>Within {limitShown}</th>
                        <th>Measured in this environment</th>
                        <th><span className="sr">Details</span></th>
                      </tr>
                    </thead>
                    <tbody>
                      {page.map((x) => {
                        const { row, j } = x
                        const m = row.mats[0]
                        const twin = twinNote(row, twins)
                        return (
                          <tr key={row.key} className={isSel(x) ? 'on' : ''}>
                            <td>{x.rank}</td>
                            <td className="cd-wrap">
                              <strong>{row.mats.length > NAMES_SHOWN ? `${row.mats.slice(0, NAMES_SHOWN).map((k) => k.name).join('; ')}; and ${count(row.mats.length - NAMES_SHOWN)} more` : row.mats.map((k) => k.name).join('; ')}</strong>
                              <span className="label">{rowKind(row)}{!m.own && (row.mats.length > 1 ? ' · no inputs of their own' : ' · no input of its own')}</span>
                              {twin && <span className="label">{twin}</span>}
                            </td>
                            <td className="num">{shownRate(row.pred)}</td>
                            <td className="num strong">{shownRate(row.pred + hw)}</td>
                            <td><LimitFlag j={j} extra={extra} /></td>
                            <td className="cd-wrap">
                              <MeasuredFlag j={j} extra={extra} />
                              {row.st ? (
                                <>
                                  <span className="label">{aboveText(j)}{measuredSummary(row.st)}</span>
                                  {x.near && <span className="label">closest to your conditions: {measuredRate(x.near.y)} mm/yr ({recordWhere(x.near)})</span>}
                                </>
                              ) : <span className="label">pure extrapolation</span>}
                            </td>
                            <td>
                              <button type="button" className="chip" aria-pressed={isSel(x)} aria-label={`Show details of ${rowName(row)}`} onClick={() => select(row)}>Details</button>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="cd-more">
                  <span className="note small">Showing {count(page.length)} of {count(view.shown.length)} rows{view.shown.length < view.judged.length ? ` (${count(view.judged.length - view.shown.length)} hidden by the options above)` : ''}.</span>
                  {page.length < view.shown.length && (
                    <>
                      <button type="button" className="chip" onClick={() => setMore({ key: viewKey, n: shownN + PAGE })}>Show {Math.min(PAGE, view.shown.length - page.length)} more</button>
                      <button type="button" className="chip" onClick={() => setMore({ key: viewKey, n: view.shown.length })}>Show all {count(view.shown.length)}</button>
                    </>
                  )}
                </div>
              </>
            )}
            <dl className="readme cd-readme">
              <div>
                <dt>Prediction</dt>
                <dd>The trees&apos; prediction of the material&apos;s corrosion rate in this exposure, in millimetres per year. A prediction, not a measurement.</dd>
              </div>
              <div>
                <dt>Cautious upper estimate</dt>
                <dd>
                  The prediction plus {fmt(hw)} log units, which multiplies it by about {times}: the upper edge of its 90% range. The table is
                  ranked by it, lowest first.
                </dd>
              </div>
              <div>
                <dt>Within the limit</dt>
                <dd>
                  “Cautious estimate”: even the upper estimate is within your limit. “Prediction only”: the prediction is within it, the upper
                  estimate is not. “No”: the prediction itself is above it. “(extrapolated)”: a condition you typed is outside this
                  environment&apos;s records, where the 90% range was never checked.
                </dd>
              </div>
              <div>
                <dt>Measured neighbours</dt>
                <dd>
                  The rates CORR-DATA records for that material in this environment, at any concentration, temperature and exposure time.
                  “Agree” only when the prediction is within the limit and so is every one of those measured rates; “disagree” when the
                  prediction is within it and at least one measured rate is above it.
                </dd>
              </div>
              <div>
                <dt>No input of its own</dt>
                <dd>
                  The data has too few records of that material for the model to learn it separately. It is predicted from its class and
                  family, with the same numbers as the other such materials of that family.
                </dd>
              </div>
              <div>
                <dt>No record here</dt>
                <dd>
                  The data holds no measured rate of that material in this environment. Its prediction is a pure extrapolation from other
                  environments and from similar materials.
                </dd>
              </div>
            </dl>
          </section>
        </>
      )}

      <section className="card fam-section cd">
        <h2>How far to trust this designer</h2>
        <ul className="cd-points">
          <li>
            Cross-validated on {count(prep.nRecords)} measured rates, grouped so that the rates from one reference are never split between
            training and testing: the prediction misses the measured rate by <strong>{fmt(cv.mae)} log units on average</strong> (a factor of
            about {fmt(10 ** cv.mae, 2)}).
            {Number.isFinite(cv.mean_baseline_mae) && ` Predicting the average of the data for every row misses by ${fmt(cv.mean_baseline_mae)} on average (a factor of about ${fmt(10 ** cv.mean_baseline_mae, 2)}).`}
            {' '}One cross-validation, not repeated, so these numbers carry no interval.
          </li>
          <li>
            The 90% range is ±{fmt(hw)} log units (a factor of about {times} either way), set so that 90% of the cross-validated misses fall
            within it. The cautious upper estimate is its upper edge. It is the same for every material and environment, so it is wider
            than needed where the data is dense and may be too narrow where it is thin. It was checked only on conditions like the
            records: a concentration, temperature or exposure time outside an environment&apos;s records is marked “extrapolated”.
          </li>
          <li>
            The model has an input of its own for {count(prep.nOwnMaterials)} of the {count(prep.materials.length)} materials and {count(prep.nOwnEnvironments)} of
            the {count(prep.environments.length)} environments. The others have too few records: they are pooled, so a pooled material is predicted
            from its class and family, and all pooled environments get the same prediction.
          </li>
          <li>
            Its other inputs are the concentration, the temperature and the exposure time. It cannot see aeration, flow, impurities, heat
            treatment, welds or surface finish, and a rate in mm per year says nothing about pitting, crevice attack or cracking.
          </li>
          <li>
            This is a compact model trained for this page ({model.kind ?? 'gradient-boosted trees'}
            {Number.isFinite(model.params?.n_estimators) ? `, ${model.params.n_estimators} trees of depth ${model.params.max_depth}` : ''}), separate from the
            pipeline&apos;s tuned trees and network{tasks.length > 0 ? ', whose own test results follow below' : ''}.
          </li>
          {prep.nChecks > 0 && (
            <li>
              {prep.checkWorst < 1e-6
                ? `The page walks the exported trees itself; on the export's ${prep.nChecks} check rows it reproduces the exporter's predictions to within ${plain(prep.checkWorst, 2)} log units.`
                : `Warning: on the export's ${prep.nChecks} check rows the page's predictions differ from the exporter's by up to ${plain(prep.checkWorst, 2)} log units. Treat them with care.`}
            </li>
          )}
        </ul>
        <p className="fine">Data: {model.source ?? '–'}. Target: {model.unit ?? '–'}. Folds: {cv.folds ?? '–'}. Model file generated {model.generated ?? '–'}.</p>
      </section>
    </>
  )
}
