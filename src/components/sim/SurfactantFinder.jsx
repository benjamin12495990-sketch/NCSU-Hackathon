import { useEffect, useMemo, useState } from 'react'
import { loadPipeline, fmt } from '../../lib/pipeline.js'
import './SurfactantFinder.css'

// Surfactant finder (see README.md in this folder). The visitor sets a pCMC the surfactant must reach (at least: a higher pCMC is a
// lower critical micelle concentration) and/or a surface tension at the CMC it must get down to (at most), and optionally one type.
// The page lists the SURFPRO surfactants in surfactants.json that meet every requirement set, using the measured value where there
// is one and the model's prediction (marked "predicted") where there is not, judged at the cautious edge of the 90% range unless the
// visitor turns that off, and sorts them by how far inside the requirements they sit. For measured rows the file carries the
// out-of-fold prediction, so the page shows how close the model came, and tests the predictions against the measured rows for the
// visitor's own requirements. A measured row whose out-of-fold prediction is further off than the 90% range is labelled, and counted
// above the table, because sorting by margin pulls those rows to the top. Every number shown comes from that file; PAGE below is only
// how many rows a "show more" adds and TOP how many leading rows that count looks at.

const PAGE = 50
const TOP = 10
const PROPS = ['pcmc', 'tension']
const two = (v) => (Number.isFinite(v) ? v.toFixed(2) : '–')
const one = (v) => (Number.isFinite(v) ? v.toFixed(1) : '–')
const SHOW = { pcmc: two, tension: one }
const pct = (a, b) => (b ? `${Number(((a / b) * 100).toFixed(1))}%` : '–')
const verb = (n, single, many) => (n === 1 ? single : many)
const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const median = (s) => (s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : null)
// A requirement is met at or above it for pCMC (higher is better) and at or below it for surface tension (lower is better).
const meets = (p, v, req) => (p === 'pcmc' ? v >= req : v <= req)
// The cautious edge of a prediction's 90% range: the side that makes the requirement harder to meet.
const edge = (p, pred, hw) => (p === 'pcmc' ? pred - hw : pred + hw)
// A measured value and its prediction disagree by more than the 90% half-width (the complement of "within" in prepare below).
const beyond = (c, hw) => c.measured !== null && c.predicted !== null && Math.abs(c.predicted - c.measured) > hw
const NAME = { pcmc: 'pCMC', tension: 'surface tension' }

function validData(d) {
  const ok = (p) => !!p && Number.isFinite(p.cv_mae) && Number.isFinite(p.interval90_half_width) && p.interval90_half_width > 0 &&
    Number.isFinite(p.mean_baseline_mae)
  // The surface-tension unit is printed after every tension value, so a file without it is incomplete rather than given a guessed one.
  const unit = d?.properties?.tension?.unit
  return ok(d?.properties?.pcmc) && ok(d?.properties?.tension) && typeof unit === 'string' && unit.trim() !== '' &&
    Array.isArray(d?.surfactants) && d.surfactants.length > 0
}

const cellOf = (c) => ({ measured: num(c?.measured), predicted: num(c?.predicted), oof: c?.out_of_fold === true })

// Everything that depends only on the file: the rows, the types with their counts, and per property the measured range, median and
// how often the out-of-fold prediction fell within the 90% range.
function prepare(d) {
  const P = d.properties
  const hw = { pcmc: P.pcmc.interval90_half_width, tension: P.tension.interval90_half_width }
  const rows = d.surfactants.map((s, i) => {
    const t = String(s?.temp_C ?? '').trim()
    return {
      i, smiles: String(s?.smiles ?? ''), type: String(s?.type ?? '').trim() || 'type not given',
      temp: t === '' ? null : Number.isFinite(Number(t)) ? String(Number(t)) : t,
      pcmc: cellOf(s?.pcmc), tension: cellOf(s?.tension),
    }
  })
  const typeCounts = new Map()
  for (const r of rows) typeCounts.set(r.type, (typeCounts.get(r.type) ?? 0) + 1)
  const types = [...typeCounts].map(([type, n]) => ({ type, n })).sort((a, b) => b.n - a.n || (a.type < b.type ? -1 : 1))
  const stat = {}
  for (const p of PROPS) {
    const ms = rows.map((r) => r[p].measured).filter((v) => v !== null).sort((a, b) => a - b)
    const checked = rows.filter((r) => r[p].measured !== null && r[p].predicted !== null && r[p].oof)
    stat[p] = {
      n: ms.length, min: ms.length ? ms[0] : null, max: ms.length ? ms[ms.length - 1] : null, median: median(ms),
      checked: checked.length, within: checked.filter((r) => Math.abs(r[p].predicted - r[p].measured) <= hw[p]).length,
      predOnly: rows.filter((r) => r[p].measured === null && r[p].predicted !== null).length,
    }
  }
  return { rows, types, hw, stat, noTemp: rows.filter((r) => r.temp === null).length }
}

// One requirement on one row: the value it is judged on (measured, else the prediction, at its cautious edge when asked), whether it
// meets the requirement, and how far inside it is, also divided by the property's 90% half-width so pCMC and tension compare.
function judge(prep, r, p, ask) {
  const req = ask[p]
  if (req === null) return null
  const c = r[p]
  let v
  let used
  if (c.measured !== null) {
    v = c.measured
    used = 'measured'
  } else if (!ask.measuredOnly && c.predicted !== null) {
    v = ask.cautious ? edge(p, c.predicted, prep.hw[p]) : c.predicted
    used = 'predicted'
  } else {
    return { ok: false, used: null }
  }
  const inside = p === 'pcmc' ? v - req : req - v
  return { ok: inside >= 0, used, v, inside, norm: inside / prep.hw[p] }
}

const poolOf = (prep, type) => (type === 'any' ? prep.rows : prep.rows.filter((r) => r.type === type))

// The rows that meet every requirement set, furthest inside first (by the tighter requirement, in 90% half-widths). Ties go to rows
// that rely on no prediction, then to file order. `far` marks a hit with a requirement judged on a measured value whose out-of-fold
// prediction is beyond the 90% range. `measured` is the measured range of each property within the pool (the chosen type), for the
// message shown when nothing matches.
function find(prep, ask) {
  const pool = poolOf(prep, ask.type)
  const hits = []
  const measured = { pcmc: { n: 0, min: null, max: null }, tension: { n: 0, min: null, max: null } }
  for (const r of pool) {
    for (const p of PROPS) {
      const v = r[p].measured
      if (v === null) continue
      const m = measured[p]
      m.n++
      if (m.min === null || v < m.min) m.min = v
      if (m.max === null || v > m.max) m.max = v
    }
    const j = { pcmc: judge(prep, r, 'pcmc', ask), tension: judge(prep, r, 'tension', ask) }
    const set = PROPS.filter((p) => j[p])
    if (!set.every((p) => j[p].ok)) continue
    let tight = set[0]
    for (const p of set) if (j[p].norm < j[tight].norm) tight = p
    hits.push({
      r, j, tight, norm: j[tight].norm, pred: set.some((p) => j[p].used === 'predicted'),
      far: set.some((p) => j[p].used === 'measured' && r[p].oof && beyond(r[p], prep.hw[p])),
    })
  }
  hits.sort((a, b) => b.norm - a.norm || Number(a.pred) - Number(b.pred) || a.r.i - b.r.i)
  return {
    pool: pool.length, hits, measured, nPred: hits.filter((h) => h.pred).length, nFar: hits.filter((h) => h.far).length,
    nFarTop: hits.slice(0, TOP).filter((h) => h.far).length,
  }
}

// The honest test of the predictions for the visitor's own requirements: on the rows measured for every requirement set (whose
// predictions are out of fold), which would the predictions alone have picked, at the cautious edge and on the prediction itself,
// and how many of those really meet the requirements by measurement?
function testPredictions(prep, ask) {
  const set = PROPS.filter((p) => ask[p] !== null)
  const out = { n: 0, real: 0, cautious: { picked: 0, right: 0 }, plain: { picked: 0, right: 0 } }
  for (const r of poolOf(prep, ask.type)) {
    if (!set.every((p) => r[p].measured !== null && r[p].predicted !== null && r[p].oof)) continue
    out.n++
    const byMeasured = set.every((p) => meets(p, r[p].measured, ask[p]))
    if (byMeasured) out.real++
    for (const mode of ['cautious', 'plain']) {
      if (!set.every((p) => meets(p, mode === 'cautious' ? edge(p, r[p].predicted, prep.hw[p]) : r[p].predicted, ask[p]))) continue
      out[mode].picked++
      if (byMeasured) out[mode].right++
    }
  }
  return out
}

// One line of that test, in words.
function testText(t, real) {
  if (t.picked === 0) return 'they pick none of them.'
  return `they pick ${t.picked.toLocaleString()}, and ${t.right.toLocaleString()} of those ${t.picked.toLocaleString()} really ${verb(t.right, 'meets', 'meet')} ` +
    `the requirements by measurement (${pct(t.right, t.picked)}), finding ${t.right.toLocaleString()} of the ${real.toLocaleString()} that do (${pct(t.right, real)}).`
}

// Plain decimals only ("3.1", "3,1", "35 mN/m"); blank means no requirement. A comma before exactly three digits ("1,000", "3,102")
// could be a decimal comma or a thousands separator, so it is refused rather than guessed.
function parseReq(text, p) {
  const raw = String(text).trim().replace(/\s*mn\s*\/\s*m$/i, '')
  if (/^\d+,\d{3}$/.test(raw)) {
    return {
      value: null,
      error: `“${raw}” can be read as ${raw.replace(',', '.')} or as ${raw.replace(',', '')}. Type the ${p === 'pcmc' ? 'pCMC' : 'surface tension'} with a decimal point and no thousands separator.`,
    }
  }
  const t = raw.replace(',', '.')
  if (t === '') return { value: null, error: '' }
  if (!/^(\d+(\.\d*)?|\.\d+)$/.test(t)) {
    return {
      value: null,
      error: p === 'pcmc'
        ? 'Type the pCMC as a plain number, for example 3.5, or leave it blank for no pCMC requirement.'
        : 'Type the surface tension in mN/m as a plain number, for example 35, or leave it blank for no surface-tension requirement.',
    }
  }
  return { value: Number(t), error: '' }
}

// pCMC = -log10(CMC in mol/L), so a pCMC of at least x is a CMC of at most 10^-x mol/L. Past a pCMC of 12 the number is no longer
// worth printing (it would read 1e-93 µmol/L, or 0), so the phrase only says it is below 1e-12 mol/L.
function cmcPhrase(pcmc) {
  if (pcmc > 12) return 'a CMC below 1e-12 mol/L'
  const mol = 10 ** -pcmc
  const [scale, unit] = mol >= 0.1 ? [1, 'mol/L'] : mol >= 1e-4 ? [1e3, 'mmol/L'] : mol >= 1e-7 ? [1e6, 'µmol/L'] : mol >= 1e-10 ? [1e9, 'nmol/L'] : [1e12, 'pmol/L']
  return `a CMC of at most ${fmt(mol * scale, 2)} ${unit}`
}

function Smiles({ text }) {
  return <code className="surf-smiles">{text || 'SMILES not given'}</code>
}

// The two cells of one property in a table row. `title` names the property in the small label that stands in for the column header
// on a phone, where the rows are stacked and the header row is not shown (SurfactantFinder.css).
function PropCells({ p, c, j, hw, cautious, title }) {
  const used = j?.used ?? null
  let prediction = '–'
  if (c.predicted !== null && c.measured !== null) {
    prediction = (
      <>
        {SHOW[p](c.predicted)}
        <span className="label">{c.oof ? 'out of fold' : 'not out of fold'}</span>
        <span className="label">off by {SHOW[p](Math.abs(c.predicted - c.measured))}</span>
        {beyond(c, hw) && <span className="label surf-far">beyond the 90% range</span>}
      </>
    )
  } else if (c.predicted !== null) {
    const judgedAtEdge = used === 'predicted' && cautious
    prediction = (
      <>
        <span className="surf-pred">predicted</span>{' '}
        <span className={used === 'predicted' && !cautious ? 'strong' : ''}>{SHOW[p](c.predicted)}</span>
        <span className="label">cautious edge <span className={judgedAtEdge ? 'strong' : ''}>{SHOW[p](edge(p, c.predicted, hw))}</span></span>
      </>
    )
  }
  return (
    <>
      <td className="num">
        <span className="label surf-narrow">{title}: measured</span>
        {c.measured === null ? <span className="label">not measured</span> : <span className={used === 'measured' ? 'strong' : ''}>{SHOW[p](c.measured)}</span>}
      </td>
      <td className="num">
        <span className="label surf-narrow">{title}: prediction</span>
        {prediction}
      </td>
    </>
  )
}

// A headline card: the surfactant, then only the properties the visitor set a requirement on. The value in bold is the one judged
// against the requirement, as in the table: the measured value, or for a predicted property its cautious edge while Cautious is on.
function Pick({ title, hit, unit, hw, props, cautious }) {
  return (
    <div className="mini surf-pick">
      <h3>{title}</h3>
      {hit ? (
        <>
          <Smiles text={hit.r.smiles} />
          <p className="fine">{hit.r.type} · temperature {hit.r.temp === null ? 'not given' : `${hit.r.temp} °C`}</p>
          {props.map((p) => {
            const c = hit.r[p]
            const used = hit.j[p]?.used ?? null
            if (used === null) return null
            const name = p === 'pcmc' ? 'pCMC' : 'Surface tension'
            const u = p === 'tension' ? ` ${unit}` : ''
            if (used === 'measured') {
              return (
                <p key={p}>
                  {name} <strong>{SHOW[p](c.measured)}{u}</strong> <span className="surf-src">measured</span>
                  {beyond(c, hw[p]) && (
                    <span className="surf-src surf-line">
                      {c.oof ? 'out-of-fold prediction' : 'prediction'} {SHOW[p](c.predicted)}{u}: beyond the 90% range
                    </span>
                  )}
                </p>
              )
            }
            const atEdge = SHOW[p](edge(p, c.predicted, hw[p]))
            return (
              <p key={p}>
                {name} <strong>{cautious ? atEdge : SHOW[p](c.predicted)}{u}</strong> <span className="surf-pred">predicted</span>{' '}
                <span className="surf-src">{cautious ? `at its cautious edge (prediction ${SHOW[p](c.predicted)})` : `(cautious edge ${atEdge})`}</span>
              </p>
            )
          })}
        </>
      ) : <p className="fine">None among the surfactants that meet the requirements.</p>}
    </div>
  )
}

export default function SurfactantFinder({ tasks = [], index }) {
  const [data, setData] = useState(undefined)
  const [xText, setXText] = useState(null) // null until the visitor types: the "Both" example is shown
  const [yText, setYText] = useState(null)
  const [type, setType] = useState('any')
  const [cautious, setCautious] = useState(true)
  const [measuredOnly, setMeasuredOnly] = useState(false)
  const [more, setMore] = useState({ key: '', n: PAGE })
  useEffect(() => {
    const ctl = new AbortController()
    loadPipeline('surfactants.json', ctl.signal).then(setData, () => {})
    return () => ctl.abort()
  }, [])
  const prep = useMemo(() => {
    if (!data) return null
    try {
      return validData(data) ? prepare(data) : { broken: true }
    } catch {
      return { broken: true }
    }
  }, [data])
  const ready = !!prep && !prep.broken
  // The example requirements are the medians, written as the page prints them (four significant figures), so a median that is the
  // mean of two values never shows floating-point noise in the input box.
  const medX = ready && prep.stat.pcmc.median !== null ? fmt(prep.stat.pcmc.median) : ''
  const medY = ready && prep.stat.tension.median !== null ? fmt(prep.stat.tension.median) : ''
  const xs = xText ?? medX
  const ys = yText ?? medY
  const px = parseReq(xs, 'pcmc')
  const py = parseReq(ys, 'tension')
  const xv = px.value
  const yv = py.value
  const bad = !!(px.error || py.error)
  const typeOk = type === 'any' || (ready && prep.types.some((t) => t.type === type))
  const ask = useMemo(
    () => (ready && !bad && (xv !== null || yv !== null) ? { pcmc: xv, tension: yv, type: typeOk ? type : 'any', cautious, measuredOnly } : null),
    [ready, bad, xv, yv, type, typeOk, cautious, measuredOnly],
  )
  const found = useMemo(() => (ask ? find(prep, ask) : null), [prep, ask])
  const test = useMemo(() => (ask ? testPredictions(prep, ask) : null), [prep, ask])

  if (data === undefined) return <section className="card fam-section"><p className="note">Loading the surfactant table…</p></section>
  if (!ready) {
    return (
      <section className="card fam-section surf">
        <h2>Find a surfactant</h2>
        <p className="note">
          {data === null ? 'The surfactant table has not been exported to the site yet, so the finder cannot run.' : 'The surfactant table file is incomplete, so the finder cannot run.'}
          {' '}On the training laptop: <code>python -m gpu.export_site --browser-models-only</code>, then commit <code>public/pipeline/surfactants.json</code>.
          {tasks.length > 0 && ' The pipeline’s own test results for these properties are below.'}
        </p>
      </section>
    )
  }

  const P = data.properties
  const unit = P.tension.unit.trim()
  const hw = prep.hw
  const st = prep.stat
  const typeName = ask && ask.type !== 'any' ? `${ask.type} ` : ''
  const askKey = ask ? JSON.stringify(ask) : ''
  const shownN = more.key === askKey ? more.n : PAGE
  const rows = found ? found.hits.slice(0, shownN) : []
  const bestMeasured = found?.hits.find((h) => !h.pred) ?? null
  const bestPredicted = found?.hits.find((h) => h.pred) ?? null
  const reqParts = []
  if (ask && ask.pcmc !== null) reqParts.push(`pCMC at least ${fmt(ask.pcmc)} (${cmcPhrase(ask.pcmc)})`)
  if (ask && ask.tension !== null) reqParts.push(`surface tension at most ${fmt(ask.tension)} ${unit}`)
  const examples = [
    { label: `Lower CMC than the median measured: pCMC ≥ ${fmt(st.pcmc.median)}`, x: medX, y: '' },
    { label: `Lower surface tension than the median measured: ≤ ${fmt(st.tension.median)} ${unit}`, x: '', y: medY },
    { label: 'Both', x: medX, y: medY },
  ]
  const setReq = (x, y) => {
    setXText(x)
    setYText(y)
  }
  const nHits = found ? found.hits.length : 0
  const nMeasuredHits = found ? nHits - found.nPred : 0
  const setProps = ask ? PROPS.filter((p) => ask[p] !== null) : []
  // How many of the hits lean on a prediction, said in the heading itself so its count is never read as measured results.
  const predNote = !found || found.nPred === 0 ? ''
    : found.nPred < nHits ? ` (${found.nPred.toLocaleString()} of them relying on a prediction)`
      : nHits === 1 ? ' (relying on a prediction)' : ' (all of them relying on a prediction)'
  // How many hits were judged on a measured value that the model's out-of-fold prediction is far from (the rows the sort pulls up).
  const farLead = !found || found.nFar === 0 ? ''
    : nHits === 1 ? 'For this one' : found.nFar === nHits ? `For all ${nHits.toLocaleString()}` : `In ${found.nFar.toLocaleString()} of the ${nHits.toLocaleString()}`
  // Which judgement of a prediction the list is using, for the test of the predictions further down: none when only measured values count.
  const inUse = !ask || ask.measuredOnly ? '' : ask.cautious ? 'cautious' : 'plain'
  const tips = ['Relax a requirement']
  if (ask?.cautious && !ask.measuredOnly) tips.push('untick Cautious')
  if (ask?.measuredOnly) tips.push('allow predictions')
  if (ask && ask.type !== 'any') tips.push('choose another type')
  const tipText = tips.length > 1 ? `${tips.slice(0, -1).join(', ')} or ${tips[tips.length - 1]}` : tips[0]
  // When nothing matches: what the surfactants of the chosen type were measured at, for the properties that have a requirement.
  const poolText = found ? `${found.pool.toLocaleString()} ${typeName}${verb(found.pool, 'surfactant', 'surfactants')}` : ''
  const rangeParts = found ? setProps.map((p) => {
    const m = found.measured[p]
    const u = p === 'tension' ? ` ${unit}` : ''
    if (m.n === 0) return `none has a measured ${NAME[p]}`
    if (m.n === 1) return `only one has a measured ${NAME[p]} (${SHOW[p](m.min)}${u})`
    return `measured ${NAME[p]} runs from ${SHOW[p](m.min)} to ${SHOW[p](m.max)}${u} (${m.n.toLocaleString()} measured)`
  }) : []
  const colTitle = { pcmc: `pCMC${ask && ask.pcmc === null ? ' (no requirement)' : ''}`, tension: `Surface tension, ${unit}${ask && ask.tension === null ? ' (no requirement)' : ''}` }

  return (
    <>
      <section className="card fam-section surf">
        <h2>Find a surfactant</h2>
        {index?.rehearsal && (
          <p className="note small">
            The rehearsal flag above is about the pipeline&apos;s results further down this page. This finder uses its own table, which the
            exporter builds from the full SURFPRO data whatever the nightly run was.
          </p>
        )}
        <form onSubmit={(e) => e.preventDefault()}>
          <div className="surf-form">
            <div>
              <label htmlFor="surf-pcmc">pCMC at least</label>
              <input id="surf-pcmc" value={xs} onChange={(e) => setXText(e.target.value)} inputMode="decimal" autoComplete="off" placeholder="blank: no requirement"
                aria-describedby={px.error ? 'surf-pcmc-err surf-pcmc-hint' : 'surf-pcmc-hint'} aria-invalid={px.error ? 'true' : undefined} />
              <span className="surf-hint" id="surf-pcmc-hint">{xv !== null ? cmcPhrase(xv) : 'blank: no pCMC requirement'}</span>
            </div>
            <div>
              <label htmlFor="surf-tension">Surface tension at most ({unit})</label>
              <input id="surf-tension" value={ys} onChange={(e) => setYText(e.target.value)} inputMode="decimal" autoComplete="off" placeholder="blank: no requirement"
                aria-describedby={py.error ? 'surf-tension-err surf-tension-hint' : 'surf-tension-hint'} aria-invalid={py.error ? 'true' : undefined} />
              <span className="surf-hint" id="surf-tension-hint">{yv !== null ? 'the tension reached at the CMC; lower is stronger' : 'blank: no surface-tension requirement'}</span>
            </div>
            <div>
              <label htmlFor="surf-type">Surfactant type</label>
              <select id="surf-type" value={typeOk ? type : 'any'} onChange={(e) => setType(e.target.value)} aria-describedby="surf-type-hint">
                <option value="any">Any type · {prep.rows.length.toLocaleString()}</option>
                {prep.types.map((t) => <option key={t.type} value={t.type}>{t.type} · {t.n.toLocaleString()}</option>)}
              </select>
              <span className="surf-hint" id="surf-type-hint">as classified in SURFPRO</span>
            </div>
          </div>
          <div className="examples">
            <span>Try:</span>
            {examples.map((ex) => (
              <button type="button" key={ex.label} className="chip" aria-pressed={xs === ex.x && ys === ex.y} onClick={() => setReq(ex.x, ex.y)}>
                {ex.label}
              </button>
            ))}
          </div>
          <div className="options">
            <label>
              <input type="checkbox" checked={cautious} onChange={(e) => setCautious(e.target.checked)} />
              <span>Cautious: a prediction must meet the requirement at the edge of its 90% range (pCMC prediction − {fmt(hw.pcmc)}; surface
                tension prediction + {fmt(hw.tension)} {unit})</span>
            </label>
            <label>
              <input type="checkbox" checked={measuredOnly} onChange={(e) => setMeasuredOnly(e.target.checked)} />
              <span>Measured values only: leave out surfactants that would rely on a prediction</span>
            </label>
          </div>
          <p className="note small">
            {P.pcmc.unit ? `${P.pcmc.unit}: a` : 'A'} higher pCMC means micelles form at a lower concentration. The list updates as you type. The examples use
            the median measured value of each property in the data ({st.pcmc.n.toLocaleString()} measured pCMC values, {st.tension.n.toLocaleString()} measured
            surface tensions), not a requirement from a standard.
          </p>
        </form>
        {px.error && <p className="error" role="alert" id="surf-pcmc-err">{px.error}</p>}
        {py.error && <p className="error" role="alert" id="surf-tension-err">{py.error}</p>}
      </section>

      {!bad && !ask && (
        <section className="card fam-section">
          <p className="note">
            Set at least one requirement above, or pick an example. With neither, every surfactant would qualify and there would be nothing to
            sort them by.
          </p>
        </section>
      )}

      {found && (
        <section className="card fam-section">
          <h2 aria-live="polite">
            {nHits.toLocaleString()} of {found.pool.toLocaleString()} {typeName}surfactants {verb(nHits, 'meets', 'meet')} the requirements{predNote}
          </h2>
          <p className="note">
            Requirements: {reqParts.join(' and ')}{ask.measuredOnly ? ', on measured values only' : ask.cautious ? ', predictions judged at the cautious edge of their 90% range' : ', predictions judged on the prediction itself (Cautious is off)'}.
            {nHits > 0 && (
              <>
                {' '}{nMeasuredHits.toLocaleString()} {verb(nMeasuredHits, 'meets', 'meet')} them on measured values alone;
                {' '}{found.nPred.toLocaleString()} {verb(found.nPred, 'relies', 'rely')} on at least one prediction (marked <span className="surf-pred">predicted</span>).
                {' '}Furthest inside the requirements first.
              </>
            )}
          </p>
          {found.nFar > 0 && (
            <p className="note">
              {farLead}, a measured value you set a requirement on has an out-of-fold prediction further
              from it than the 90% range (labelled &ldquo;beyond the 90% range&rdquo; in the table){nHits > TOP ? `; ${found.nFarTop.toLocaleString()} of the first ${TOP} rows listed ${verb(found.nFarTop, 'is', 'are')} among them` : ''}.
              That is a model miss or a doubtful source value, so check the original measurement before relying on such a row.
            </p>
          )}
          {!ask.cautious && !ask.measuredOnly && found.nPred > 0 && (
            <p className="warn">
              Cautious is off. The rows marked predicted only meet the requirements on the model&apos;s expected value; at the edge of the 90% range
              they may not. Read them as surfactants worth measuring, not as results.
            </p>
          )}
          {nHits === 0 ? (
            <p className="note">
              No {typeName}surfactant meets these requirements. Among the {poolText}, {rangeParts.join(' and ')}. {tipText}.
            </p>
          ) : (
            <>
              <div className="cards surf-picks">
                <Pick title="Furthest inside, on measured values" hit={bestMeasured} unit={unit} hw={hw} props={setProps} cautious={ask.cautious} />
                {!ask.measuredOnly && <Pick title="Furthest inside, relying on a prediction" hit={bestPredicted} unit={unit} hw={hw} props={setProps} cautious={ask.cautious} />}
              </div>
              <div className="tablewrap sim-scroll">
                <table className="ranked surf-table">
                  <thead>
                    <tr>
                      <th rowSpan={2} scope="col" className="surf-n">#</th>
                      <th rowSpan={2} scope="col" className="surf-id">SMILES</th>
                      <th rowSpan={2} scope="col" className="surf-type">Type</th>
                      <th rowSpan={2} scope="col" className="num surf-temp">Temp. (°C)</th>
                      <th colSpan={2} scope="col" className="group">{colTitle.pcmc}</th>
                      <th colSpan={2} scope="col" className="group">{colTitle.tension}</th>
                      <th rowSpan={2} scope="col" className="num">Tightest margin</th>
                    </tr>
                    <tr>
                      <th scope="col" className="num">Measured</th><th scope="col" className="num">Prediction</th>
                      <th scope="col" className="num">Measured</th><th scope="col" className="num">Prediction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((h, k) => (
                      <tr key={h.r.i}>
                        <td className="surf-n">{k + 1}</td>
                        <td className="surf-id">
                          <span className="label surf-idline">
                            #{k + 1} · {h.r.type} · {h.r.temp === null ? 'temperature not given' : `${h.r.temp}\u00a0°C`}
                          </span>
                          <Smiles text={h.r.smiles} />
                        </td>
                        <td className="surf-type">{h.r.type}</td>
                        <td className="num surf-temp">{h.r.temp === null ? <span className="label">not given</span> : h.r.temp}</td>
                        {PROPS.map((p) => <PropCells key={p} p={p} c={h.r[p]} j={h.j[p]} hw={hw[p]} cautious={ask.cautious} title={colTitle[p]} />)}
                        <td className="num surf-margin">
                          <span className="label surf-narrow">Tightest margin</span>
                          <strong>{SHOW[h.tight](h.j[h.tight].inside)} {h.tight === 'tension' ? unit : 'pCMC'}</strong>
                          <span className="label">{fmt(h.norm, 2)}× its 90% range</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="surf-more">
                <span className="note small">Showing {rows.length.toLocaleString()} of {nHits.toLocaleString()}.</span>
                {rows.length < nHits && (
                  <>
                    <button type="button" className="chip" onClick={() => setMore({ key: askKey, n: shownN + PAGE })}>
                      Show {Math.min(PAGE, nHits - rows.length)} more
                    </button>
                    <button type="button" className="chip" onClick={() => setMore({ key: askKey, n: nHits })}>
                      Show all {nHits.toLocaleString()}
                    </button>
                  </>
                )}
              </div>
              <dl className="readme surf-readme">
                <div>
                  <dt>Measured</dt>
                  <dd>The value reported in SURFPRO. When there is one, it is the value judged against your requirement (in bold).</dd>
                </div>
                <div>
                  <dt>Prediction</dt>
                  <dd>
                    For a measured value: the out-of-fold prediction (model never trained on this row&apos;s reference group) and how far it was
                    off, labelled &ldquo;beyond the 90% range&rdquo; when that is more than {fmt(hw.pcmc)} pCMC or {fmt(hw.tension)} {unit}.
                    Marked <span className="surf-pred">predicted</span>: no measured value, so the model trained on all measured rows predicted it,
                    with nothing measured to check it against.
                  </dd>
                </div>
                <div>
                  <dt>Cautious edge</dt>
                  <dd>
                    The prediction moved by its 90% range towards failing: pCMC − {fmt(hw.pcmc)}, surface tension + {fmt(hw.tension)} {unit}.
                    {ask.cautious ? ' While Cautious is ticked it is the value judged for predicted rows (in bold).' : ' Cautious is off, so predicted rows are judged on the prediction itself.'}
                  </dd>
                </div>
                <div>
                  <dt>Tightest margin</dt>
                  <dd>
                    How far the judged value sits inside your requirement (with two, the one with less room), in its own unit and as a multiple
                    of that property&apos;s 90% range, so pCMC and surface tension compare. Rows are sorted by it, most room first.
                  </dd>
                </div>
                <div>
                  <dt>Temp.</dt>
                  <dd>
                    The CMC temperature recorded in SURFPRO, an input to the model.
                    {prep.noTemp > 0 && (
                      <>
                        {' '}Not given for {prep.noTemp.toLocaleString()} {verb(prep.noTemp, 'surfactant', 'surfactants')}; for those the model was given the
                        median temperature of the rows it trained on instead, so their predictions assume a typical temperature.
                      </>
                    )}
                  </dd>
                </div>
              </dl>
            </>
          )}
        </section>
      )}

      <section className="card fam-section">
        <h2>How close the model comes</h2>
        <ul className="surf-list">
          <li>
            Typical miss (mean absolute error) on rows it never trained on (cross-validated, out of fold): pCMC <strong>{fmt(P.pcmc.cv_mae)}</strong>, against{' '}
            {fmt(P.pcmc.mean_baseline_mae)} for always predicting the average; surface tension <strong>{fmt(P.tension.cv_mae)} {unit}</strong>, against{' '}
            {fmt(P.tension.mean_baseline_mae)} {unit}. From one cross-validation, not repeated, with no interval on these numbers.
          </li>
          <li>
            90% range: ±{fmt(hw.pcmc)} pCMC and ±{fmt(hw.tension)} {unit}. The out-of-fold prediction was within it for{' '}
            {st.pcmc.within.toLocaleString()} of {st.pcmc.checked.toLocaleString()} measured pCMC values ({pct(st.pcmc.within, st.pcmc.checked)}) and{' '}
            {st.tension.within.toLocaleString()} of {st.tension.checked.toLocaleString()} measured surface tensions ({pct(st.tension.within, st.tension.checked)}).
            The range was set from these same misses, so this shows how it was built, not an independent test.
          </li>
          {test && (
            <li>
              {test.n === 0 ? (
                <>No {typeName}surfactant has a measured value for every requirement you set, so the predictions cannot be tested against your requirements.</>
              ) : (
                <>
                  Test of the predictions for your requirements on the {test.n.toLocaleString()} {typeName}surfactants measured for{' '}
                  {reqParts.length > 1 ? 'both properties' : 'that property'}, using only their out-of-fold predictions. {test.real.toLocaleString()} of them really{' '}
                  {verb(test.real, 'meets', 'meet')} the requirements by measurement.{' '}
                  At the cautious edge{inUse === 'cautious' ? ' (in use)' : ''}, {testText(test.cautious, test.real)}{' '}
                  On the prediction itself{inUse === 'plain' ? ' (in use)' : ''}, {testText(test.plain, test.real)}{' '}
                  {inUse === ''
                    ? 'Neither is in use: Measured values only is ticked, so nothing in the list relies on a prediction.'
                    : 'It is the closest check there is for the rows marked predicted.'}
                </>
              )}
            </li>
          )}
          <li>
            The rows marked predicted ({st.pcmc.predOnly.toLocaleString()} surfactants without a measured pCMC, {st.tension.predOnly.toLocaleString()} without
            a measured surface tension) were predicted by the model trained on all measured rows. Surfactants nobody measured may be unlike the
            measured ones, so the model can miss them by more than the numbers above.
          </li>
        </ul>
        <p className="note small surf-foot">
          Model: {data.kind ?? 'not stated'}
          {Number.isFinite(data.params?.n_estimators) ? `, ${data.params.n_estimators} trees of depth ${data.params.max_depth}` : ''}. A compact model
          for this finder, separate from the pipeline&apos;s network and tree models{tasks.length > 0 ? ', whose results are below' : ''}. Folds: pCMC{' '}
          {P.pcmc.folds ?? 'not stated'}; surface tension {P.tension.folds ?? 'not stated'}.{data.generated ? ` Exported ${data.generated}.` : ''}{' '}
          Data: {data.source ?? 'not stated'}.
        </p>
      </section>
    </>
  )
}
