import { memo, useCallback, useDeferredValue, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { loadPipeline, fmt, signed, factor, MODELS } from '../../lib/pipeline.js'
import { MODEL_KEYS, prepare, parseBound, readQuery, screen, show, toShown } from './crystalScreen.js'
import CrystalViewer from './CrystalViewer.jsx'
import './CrystalFinder.css'

// Crystal finder (see README.md in this folder). The visitor sets limits on the DFT-calculated properties the pipeline trains (bulk
// and shear modulus, typed in GPa although the files hold log10 GPa; formation energy; band gap), picks the model that screens
// (network average, strongest trees, their blend, or the one the registry recommends for each property) and whether a prediction
// must clear the limits with its whole 90% range (cautious). The page then lists the held-out crystals predicted to meet every limit,
// most room first, beside their own DFT-calculated values, and tests the screen against those values. Selecting a crystal shows its
// unit cell in 3D (CrystalViewer.jsx) and its DFT-calculated values against all three predictions with their ranges.
// Every number shown comes from index.json, the crystal_<target>.json rows files and structures.json. The example limits are the
// median and quartiles of the DFT-calculated values in those files. PAGE is only how many rows a "show more" adds.

const PAGE = 25
const SHORT = { network: 'network average', trees: 'strongest trees', blend: 'blend' }
const HEAD = { network: 'Network average', trees: 'Strongest trees', blend: 'Blend, half each' }
const pct = (a, b) => (b ? `${Number(((a / b) * 100).toFixed(1))}%` : '–')
const share = (c) => (typeof c === 'number' && Number.isFinite(c) ? `${fmt(c * 100, 3)}%` : '–')
const count = (n) => n.toLocaleString()
const verb = (n, one, many) => (n === 1 ? one : many)
const list = (items, last = 'and') => (items.length > 1 ? `${items.slice(0, -1).join(', ')} ${last} ${items[items.length - 1]}` : items[0] ?? '')
const lower = (s) => s.charAt(0).toLowerCase() + s.slice(1)
const reducedMotion = () => typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
const touchScreen = () => typeof window !== 'undefined' && !!window.matchMedia?.('(pointer: coarse)').matches
const withUnit = (p, text) => (p.unit ? `${text} ${p.unit}` : text)
// The screening model of a property when "recommended" is chosen, in words. The blend stands in where the export recommends none.
const recWords = (p) => `${lower(p.label)}: ${SHORT[p.rec]}${p.recKnown ? '' : ', no recommendation in the export'}`

// A typed limit, as short text for a table header ("≥ 67.1") and as words for a sentence ("at least 67.1 GPa"). The number is
// echoed exactly as it is used (123.45 stays 123.45), not rounded like the values from the files.
const typed = (n) => String(n)
function limitShort(req) {
  if (!req || (req.min === null && req.max === null)) return ''
  if (req.min !== null && req.max !== null) return `${typed(req.minShown)} to ${typed(req.maxShown)}`
  return req.min !== null ? `≥ ${typed(req.minShown)}` : `≤ ${typed(req.maxShown)}`
}
function limitWords(p, req) {
  const name = lower(p.label)
  if (req.min !== null && req.max !== null) return `${name} between ${typed(req.minShown)} and ${withUnit(p, typed(req.maxShown))}`
  return req.min !== null ? `${name} at least ${withUnit(p, typed(req.minShown))}` : `${name} at most ${withUnit(p, typed(req.maxShown))}`
}

// How far a prediction was from the DFT-calculated value: a factor for the moduli (their files are log10), a difference otherwise.
function missText(p, pred, y) {
  if (p.convert) return `×${fmt(10 ** Math.abs(pred - y), 3)} ${pred >= y ? 'high' : 'low'}`
  return `off by ${signed(pred - y)}`
}
const rangeText = (p, pred, hw) => `${show(p, pred - hw)} to ${show(p, pred + hw)}`
// Said wherever such a range is shown: a range in the property's own units (band gap) is not stopped at zero.
const belowZero = (names) => `The range is the same width on both sides of the prediction, so for ${list(names)} it can run below zero, which no ${list(names, 'or')} can be.`

// The example limits: the median and quartiles of the DFT-calculated values of the held-out crystals in this export.
function examples(prep) {
  const by = Object.fromEntries(prep.props.map((p) => [p.key, p]))
  const val = (p, v) => fmt(toShown(p, v), 3)
  const B = by.bulk_modulus_kv, G = by.shear_modulus_gv, F = by.formation_energy_peratom, E = by.optb88vdw_bandgap
  const out = []
  if (B?.stats) out.push({ label: `Bulk modulus above the median: at least ${withUnit(B, val(B, B.stats.median))}`, set: { [B.key]: { min: val(B, B.stats.median) } } })
  if (B?.stats && G?.stats) {
    out.push({
      label: `Stiff both ways: bulk at least ${withUnit(B, val(B, B.stats.median))}, shear at least ${withUnit(G, val(G, G.stats.median))} (the medians)`,
      set: { [B.key]: { min: val(B, B.stats.median) }, [G.key]: { min: val(G, G.stats.median) } },
    })
  }
  if (F?.stats) out.push({ label: `Formation energy in the lowest quarter: at most ${withUnit(F, val(F, F.stats.q1))}`, set: { [F.key]: { max: val(F, F.stats.q1) } } })
  if (E?.stats && E.stats.q3 > 0) out.push({ label: `Band gap in the top quarter: at least ${withUnit(E, val(E, E.stats.q3))}`, set: { [E.key]: { min: val(E, E.stats.q3) } } })
  if (!out.length) {
    const p = prep.props.find((x) => x.stats)
    if (p) out.push({ label: `${p.label} above the median: at least ${withUnit(p, val(p, p.stats.median))}`, set: { [p.key]: { min: val(p, p.stats.median) } } })
  }
  return out
}

// The keyboard a phone shows for a limit. The decimal keypad of an iPhone has no minus key, so it is asked for only where a negative
// number makes no sense (the moduli, band gap); formation energy, which is mostly negative, gets the full keyboard.
const keypad = (p) => (p.signed ? 'text' : 'decimal')

const blank = (prep) => Object.fromEntries(prep.props.map((p) => [p.key, { min: '', max: '' }]))
const filled = (prep, set) => {
  const t = blank(prep)
  for (const [key, v] of Object.entries(set)) t[key] = { min: v.min ?? '', max: v.max ?? '' }
  return t
}
const sameTexts = (prep, a, b) => prep.props.every((p) => a[p.key].min.trim() === b[p.key].min.trim() && a[p.key].max.trim() === b[p.key].max.trim())

// A wide table scrolls sideways inside its frame. When it really is wider than the frame, at any screen width, a line under it says
// so (`see` = what is out of view); nothing is said while every column is in view. Measured again whenever the frame or the table
// changes size: the list gains columns' worth of text when a second limit is set.
function TableScroll({ see, children }) {
  const ref = useRef(null)
  const [cut, setCut] = useState(false)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return undefined
    const measure = () => setCut(el.scrollWidth > el.clientWidth + 4) // a few pixels of cell padding out of view hide nothing
    measure()
    if (typeof ResizeObserver === 'undefined') return undefined
    const watch = new ResizeObserver(measure)
    watch.observe(el)
    if (el.firstElementChild) watch.observe(el.firstElementChild)
    return () => watch.disconnect()
  }, [])
  return (
    <>
      <div className="tablewrap sim-scroll" ref={ref}>{children}</div>
      {cut && <p className="fine cf-swipe">{touchScreen() ? 'Swipe' : 'Scroll'} the table sideways to see {see}.</p>}
    </>
  )
}

// "C11H18O4Zn" with its counts as subscripts.
function Chem({ text }) {
  if (!text) return <span className="cf-chem">(no formula)</span>
  return <span className="cf-chem">{String(text).split(/(\d+(?:\.\d+)?)/).map((s, i) => (i % 2 ? <sub key={i}>{s}</sub> : s))}</span>
}

// The selected crystal: its DFT-calculated values against the three predictions, each with its 90% range.
function Detail({ row, prep, found, reqs, runsText, position, total, onStep }) {
  const c = row.c
  const floored = prep.props.filter((p) => p.floorZero).map((p) => lower(p.label))
  return (
    <>
      <h3>DFT-calculated values against the three predictions</h3>
      <TableScroll see="all three models">
        <table className="compact cf-detail">
          <thead>
            <tr>
              <th>Property</th>
              <th className="num">DFT-calculated</th>
              {MODEL_KEYS.map((m) => <th key={m} className="num">{HEAD[m]}<span className="label">prediction</span></th>)}
            </tr>
          </thead>
          <tbody>
            {prep.props.map((p) => {
              const cell = c.v[p.key]
              const j = row.j[p.key]
              const used = found.modelOf[p.key]
              const lim = limitShort(reqs[p.key])
              return (
                <tr key={p.key}>
                  <th scope="row">
                    {p.label}
                    {p.unit && <span className="label">{p.unit}</span>}
                    {lim && <span className="label">your limit: {lim}</span>}
                  </th>
                  {cell ? (
                    <>
                      <td className="num">
                        <strong>{show(p, cell.y)}</strong>
                        {j && <span className="label">{j.truth ? 'meets your limit' : 'outside your limit'}</span>}
                      </td>
                      {MODEL_KEYS.map((m) => (
                        <td key={m} className="num">
                          <span className={m === used ? 'strong' : undefined}>{show(p, cell[m])}</span>
                          <span className="label">{rangeText(p, cell[m], p.hw[m])}</span>
                          <span className="label">{Math.abs(cell[m] - cell.y) <= p.hw[m] ? 'DFT value inside' : 'DFT value outside'}</span>
                          <span className="label">{missText(p, cell[m], cell.y)}</span>
                        </td>
                      ))}
                    </>
                  ) : (
                    <td colSpan={4} className="cf-none">This crystal is not among the test crystals of this property in the export.</td>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>
      </TableScroll>
      <p className="fine cf-key">
        Under each prediction: its 90% range (prediction ± that model&apos;s half-width; for the moduli a factor either way), whether the
        DFT-calculated value fell inside it, and how far off the prediction was. Bold: the model that screens that property.
        {floored.length > 0 && ` ${belowZero(floored)}`}
        {runsText && ` The network figure is ${runsText}.`} The DFT-calculated values are calculations, not measurements.
      </p>
      <div className="cf-step">
        <button type="button" className="chip" disabled={position <= 0} onClick={() => onStep(-1)}>Previous</button>
        <button type="button" className="chip" disabled={position >= total - 1} onClick={() => onStep(1)}>Next</button>
        <span className="note small">{count(position + 1)} of {count(total)} in the list below</span>
      </div>
    </>
  )
}

// One property of one crystal in the list: the screening model's prediction with its 90% range, then the DFT-calculated value.
function propCells(h, p, modelOf) {
  const v = h.c.v[p.key]
  if (!v) return <td key={p.key} colSpan={2} className="cf-none">not in the export</td>
  const m = modelOf[p.key]
  const j = h.j[p.key]
  return [
    <td key={`${p.key}-p`} className="num">
      <strong>{show(p, v[m])}</strong>
      <span className="label">{rangeText(p, v[m], p.hw[m])}</span>
    </td>,
    <td key={`${p.key}-y`} className="num">
      {show(p, v.y)}
      {j && !j.truth && <span className="label">outside your limit</span>}
    </td>,
  ]
}

// Memoised: with thousands of rows shown, picking a crystal redraws only the row that gains the highlight and the one that loses it.
const Row = memo(function Row({ h, rank, on, limited, others, modelOf, tightLabel, onPick }) {
  return (
    <tr className={on ? 'on' : undefined}>
      <td>{rank}</td>
      <td className="cf-name">
        <div className="cf-namebox">
          <span>
            <strong><Chem text={h.c.formula} /></strong>
            <span className="label">{h.c.jid}</span>
          </span>
          <button type="button" className="chip" aria-pressed={on} onClick={() => onPick(h.c.jid)}
            aria-label={`Show ${h.c.formula || h.c.jid}: unit cell and all three predictions`}>Show</button>
        </div>
      </td>
      {limited.map((p) => propCells(h, p, modelOf))}
      {limited.length > 0 && (
        <td className="num">
          <strong>{fmt(h.norm, 2)}</strong>
          <span className="label">half-widths{limited.length > 1 ? `, on ${tightLabel}` : ''}</span>
        </td>
      )}
      {limited.length > 0 && (
        <td><span className={h.truth ? 'flag ok' : 'flag low'}>{h.truth ? 'meet the limits' : 'do not meet them'}</span></td>
      )}
      {others.map((p) => propCells(h, p, modelOf))}
    </tr>
  )
})

// The list. Columns: the properties with a limit first, then the two verdict columns, then the other properties.
function RankTable({ rows, shownN, selected, found, reqs, props, onPick }) {
  const { limited, others, labels } = useMemo(() => ({
    limited: props.filter((p) => found.set.includes(p)),
    others: props.filter((p) => !found.set.includes(p)),
    labels: Object.fromEntries(props.map((p) => [p.key, lower(p.label)])),
  }), [props, found])
  const groupHead = (p) => (
    <th key={p.key} colSpan={2} className="group">
      {p.label}{p.unit ? `, ${p.unit}` : ''}
      <span className="label">{SHORT[found.modelOf[p.key]]}{limitShort(reqs[p.key]) ? ` · limit ${limitShort(reqs[p.key])}` : ''}</span>
    </th>
  )
  return (
    <TableScroll see="every column">
      <table className="ranked cf-table">
        <thead>
          <tr>
            <th rowSpan={2}>#</th>
            <th rowSpan={2}>Crystal</th>
            {limited.map(groupHead)}
            {limited.length > 0 && <th rowSpan={2} className="num">Room to the nearest limit</th>}
            {limited.length > 0 && <th rowSpan={2}>Its DFT-calculated values</th>}
            {others.map(groupHead)}
          </tr>
          <tr>
            {[...limited, ...others].map((p) => [
              <th key={`${p.key}-p`} className="num">Predicted</th>,
              <th key={`${p.key}-y`} className="num">DFT-calculated</th>,
            ])}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, shownN).map((h, k) => (
            <Row key={h.c.jid} h={h} rank={k + 1} on={h.c.jid === selected} limited={limited} others={others} modelOf={found.modelOf}
              tightLabel={h.tight ? labels[h.tight] : ''} onPick={onPick} />
          ))}
        </tbody>
      </table>
    </TableScroll>
  )
}

export default function CrystalFinder({ index, tasks = [] }) {
  const [loaded, setLoaded] = useState(undefined) // undefined: loading; then { tasks, files: { [task name]: rows file or null } }
  const [structures, setStructures] = useState(undefined) // undefined: loading; null: missing; object: structures.json
  const [texts, setTexts] = useState(null) // null until the visitor types or picks an example: the first example is shown
  const [queryText, setQueryText] = useState('')
  const [model, setModel] = useState('rec')
  const [cautious, setCautious] = useState(true)
  const [view, setView] = useState('hits')
  const [sel, setSel] = useState(null)
  const [more, setMore] = useState({ key: '', n: PAGE })
  const resultRef = useRef(null)

  const okTasks = useMemo(() => tasks.filter((t) => t.kind === 'crystal' && t.status === 'ok' && t.rows_file), [tasks])
  const fileKey = okTasks.map((t) => `${t.name}=${t.rows_file}`).join('|')
  const structuresFile = index?.structures_file ?? 'structures.json'
  useEffect(() => {
    if (!fileKey) return undefined
    let live = true
    // No abort signal: PipelineResults further down the page shares these cached fetches.
    Promise.all(okTasks.map((t) => loadPipeline(t.rows_file).catch(() => null))).then((all) => {
      if (live) setLoaded({ tasks: okTasks, files: Object.fromEntries(okTasks.map((t, i) => [t.name, all[i]])) })
    })
    loadPipeline(structuresFile).then((d) => { if (live) setStructures(d && d.cells ? d : null) }, () => { if (live) setStructures(null) })
    return () => { live = false }
    // okTasks is a new array on every render of the page; fileKey says when the files to fetch really change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileKey, structuresFile])

  const prep = useMemo(() => {
    if (!loaded) return null
    try {
      return prepare(loaded.tasks, loaded.files)
    } catch {
      return { props: [], dropped: [], crystals: [], elements: new Set(), complete: 0 }
    }
  }, [loaded])
  const ready = !!prep && prep.props.length > 0 && prep.crystals.length > 0
  const presets = useMemo(() => (ready ? examples(prep) : []), [ready, prep])
  const shownTexts = useMemo(() => (ready ? texts ?? (presets[0] ? filled(prep, presets[0].set) : blank(prep)) : null), [ready, prep, presets, texts])

  // The limits as typed, turned into the units of the files. `pending`: a number is still being typed; `errors`: one cannot be read.
  const parsed = useMemo(() => {
    if (!ready) return null
    const reqs = {}
    const errors = []
    let pending = false
    for (const p of prep.props) {
      const lo = parseBound(shownTexts[p.key]?.min, p, 'min')
      const hi = parseBound(shownTexts[p.key]?.max, p, 'max')
      if (lo.error) errors.push(lo.error)
      if (hi.error) errors.push(hi.error)
      if (lo.pending || hi.pending) pending = true
      if (!lo.error && !hi.error && lo.value !== null && hi.value !== null && lo.value > hi.value) {
        errors.push(`${p.label}: the lowest value (${typed(lo.shown)}) is above the highest (${typed(hi.shown)}).`)
      }
      reqs[p.key] = { min: lo.value, max: hi.value, minShown: lo.shown, maxShown: hi.shown }
    }
    return { reqs, errors, pending }
  }, [ready, prep, shownTexts])
  // While a limit is half typed or mistyped, the results keep the last limits that could be read.
  const readable = parsed && parsed.errors.length === 0 && !parsed.pending ? parsed.reqs : null
  const [heldReqs, setHeldReqs] = useState(null)
  if (readable && heldReqs !== readable) setHeldReqs(readable)
  const reqs = readable ?? heldReqs
  const ask = useMemo(
    () => (ready && reqs ? { reqs, model, cautious, query: readQuery(prep, queryText) } : null),
    [ready, prep, reqs, model, cautious, queryText],
  )
  const shownAsk = useDeferredValue(ask)
  const found = useMemo(() => (shownAsk && ready ? screen(prep, shownAsk) : null), [prep, ready, shownAsk])
  const pick = useCallback((jid) => {
    setSel(jid)
    resultRef.current?.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'start' })
  }, [])
  const withCell = useMemo(() => (prep && structures?.cells ? prep.crystals.filter((c) => structures.cells[c.jid]).length : 0), [prep, structures])

  if (index === undefined) return <section className="card fam-section"><p className="note">Loading the crystal results…</p></section>
  const notOk = tasks.filter((t) => t.kind === 'crystal' && !(t.status === 'ok' && t.rows_file))
  if (!fileKey || (loaded && !ready)) {
    return (
      <section className="card fam-section cf">
        <h2>Find a crystal</h2>
        <p className="note">
          {index === null
            ? 'The crystal results have not been exported to the site yet, so the finder cannot run.'
            : !fileKey
              ? 'This export holds no crystal task with rows, so the finder cannot run.'
              : 'The crystal rows files are missing from this export or cannot be read, so the finder cannot run.'}
          {' '}On the training laptop, after a nightly run: <code>python -m gpu.export_site</code>, then commit <code>public/pipeline/</code>.
        </p>
        {notOk.length > 0 && (
          <ul className="cf-notes">
            {notOk.map((t) => <li key={t.id}>{t.title ?? t.name}: not in this export ({t.status ?? 'unknown status'}){t.reason ? `: ${t.reason}` : ''}.</li>)}
          </ul>
        )}
      </section>
    )
  }
  if (!loaded) return <section className="card fam-section"><p className="note">Loading the held-out crystals…</p></section>

  const rehearsal = !!index?.rehearsal || Object.values(loaded.files).some((d) => d?.rehearsal === true)
  const props = prep.props
  const exploratory = [...new Set(props.map((p) => p.task.exploratory).filter(Boolean))]
  const runs = [...new Set(props.map((p) => p.task.n_runs).filter((n) => Number.isFinite(n)))]
  const runsText = runs.length === 1 ? `the average of ${runs[0]} training ${verb(runs[0], 'run', 'runs')}` : runs.length > 1 ? 'an average of several training runs' : ''
  const query = shownAsk?.query ?? { mode: 'all' }
  const set = found?.set ?? []
  const lacked = found ? found.pool - found.judged : 0
  const canMiss = set.length > 0
  const activeView = canMiss && view === 'missed' ? 'missed' : 'hits'
  const rows = found ? (activeView === 'missed' ? found.missed : found.hits) : []
  const position = rows.length ? Math.max(0, rows.findIndex((r) => r.c.jid === sel)) : -1
  const cur = position >= 0 ? rows[position] : null
  const askKey = shownAsk ? JSON.stringify([shownAsk.reqs, shownAsk.model, shownAsk.cautious, shownAsk.query.text, activeView]) : ''
  const wanted = more.key === askKey ? more.n : PAGE
  const shownN = Math.min(rows.length, Math.max(wanted, Math.ceil((position + 1) / PAGE) * PAGE))
  const cells = structures?.cells ?? null
  const floored = props.filter((p) => p.floorZero).map((p) => lower(p.label))
  const cell = cur && cells ? cells[cur.c.jid] ?? null : null

  const setBound = (key, side, value) => setTexts({ ...shownTexts, [key]: { ...shownTexts[key], [side]: value } })
  // An example (or "No limits") starts over: its ranked hits are shown, not the "not picked" list left open from the limits before.
  const choose = (next) => {
    setTexts(next)
    setView('hits')
  }
  const step = (d) => {
    const next = rows[position + d]
    if (next) setSel(next.c.jid)
  }

  const modelWords = shownAsk?.model !== 'rec' ? `the ${SHORT[shownAsk?.model] ?? ''}`
    : set.every((p) => !p.recKnown) ? `the blend, because the export recommends no model for ${list(set.map((p) => lower(p.label)))}`
      : `the model the registry recommends for each property${set.some((p) => !p.recKnown) ? ', or the blend where it recommends none' : ''} (${set.map(recWords).join('; ')})`
  const searchWords = query.mode === 'elements'
    ? `containing ${list(query.elements)}`
    : query.mode === 'text' ? `with “${query.text}” in the formula, chemical system or JARVIS id` : ''
  // The search matches no crystal at all: said once, plainly, instead of "0 of 0".
  const nothing = !!found && found.pool === 0
  const absent = query.mode === 'elements' ? query.missing ?? [] : []
  const nothingWords = !nothing ? ''
    : absent.length ? `No held-out crystal contains ${list(absent, 'or')}`
      : query.mode === 'elements' ? `No held-out crystal contains ${list(query.elements)} together`
        : query.mode === 'text' ? `No held-out crystal has “${query.text}” in its formula, chemical system or JARVIS id`
          : 'No held-out crystal matches the search'
  // Crystals were found, but none has every property with a limit, so none can be screened.
  const unjudged = !!found && !nothing && set.length > 0 && found.judged === 0
  const lackProps = set.filter((p) => found.lacking[p.key])
  const lackWords = lackProps.length === 1
    ? `no ${lower(lackProps[0].label)} for ${verb(lacked, 'it', 'them')}`
    : `${list(lackProps.map((p) => `no ${lower(p.label)} for ${count(found.lacking[p.key])}`))} (a crystal can lack more than one)`
  const tips = ['Relax a limit']
  if (shownAsk?.cautious) tips.push('untick Cautious')
  tips.push('choose another model')
  if (query.mode !== 'all') tips.push('clear the search')
  const tipText = `${tips.slice(0, -1).join(', ')} or ${tips[tips.length - 1]}`

  let headline = ''
  if (found) {
    if (nothing) headline = nothingWords
    else if (unjudged) headline = found.pool === 1 ? `The held-out crystal${searchWords ? ` ${searchWords}` : ''} cannot be judged` : `None of the ${count(found.pool)} held-out crystals${searchWords ? ` ${searchWords}` : ''} can be judged`
    else if (!set.length) headline = `${count(found.pool)} held-out ${verb(found.pool, 'crystal', 'crystals')}${searchWords ? ` ${searchWords}` : ''}`
    else if (activeView === 'missed') headline = `${count(found.missed.length)} held-out ${verb(found.missed.length, 'crystal meets', 'crystals meet')} your limits by DFT-calculated values, but not by prediction`
    else headline = `${count(found.hits.length)} of ${count(found.judged)} held-out ${verb(found.judged, 'crystal', 'crystals')} ${found.hits.length === 1 || found.judged === 1 ? 'is' : 'are'} predicted to meet your limits`
  }
  const rankWords = !cur ? '' : !set.length ? `Crystal ${count(position + 1)} of ${count(rows.length)}`
    : activeView === 'missed' ? `Not picked by the screen · ${count(position + 1)} of ${count(rows.length)}`
      : `Rank ${count(position + 1)} of ${count(rows.length)} predicted to meet your limits`

  return (
    <>
      <section className="card fam-section cf">
        <h2>Find a crystal</h2>
        <p className="note">
          {count(prep.crystals.length)} {verb(prep.crystals.length, 'crystal', 'crystals')} from chemical systems the models never trained on, each with DFT-calculated values (calculations,
          not measurements) and three predictions. Set limits on any property; a blank box is no limit. The list updates as you type.
        </p>
        <form onSubmit={(e) => e.preventDefault()}>
          <div className="cf-reqs">
            {props.map((p) => (
              <fieldset className="cf-req" key={p.key}>
                <legend>{p.label}{p.unit ? ` (${p.unit})` : ''}</legend>
                <div className="cf-pair">
                  <div>
                    <label htmlFor={`cf-${p.key}-min`}>At least</label>
                    <input id={`cf-${p.key}-min`} value={shownTexts[p.key].min} onChange={(e) => setBound(p.key, 'min', e.target.value)}
                      inputMode={keypad(p)} autoComplete="off" autoCapitalize="off" autoCorrect="off" spellCheck={false} placeholder="no limit" />
                  </div>
                  <div>
                    <label htmlFor={`cf-${p.key}-max`}>At most</label>
                    <input id={`cf-${p.key}-max`} value={shownTexts[p.key].max} onChange={(e) => setBound(p.key, 'max', e.target.value)}
                      inputMode={keypad(p)} autoComplete="off" autoCapitalize="off" autoCorrect="off" spellCheck={false} placeholder="no limit" />
                  </div>
                </div>
                <span className="cf-hint">
                  {p.stats
                    ? `DFT-calculated in this export: ${show(p, p.stats.min)} to ${withUnit(p, show(p, p.stats.max))}, median ${show(p, p.stats.median)} (${count(p.n)} ${verb(p.n, 'crystal', 'crystals')})`
                    : 'No crystal with this property in the export'}
                </span>
              </fieldset>
            ))}
          </div>
          <div className="cf-row2">
            <div>
              <label htmlFor="cf-search">Formula, elements or JARVIS id (optional)</label>
              <input id="cf-search" value={queryText} onChange={(e) => setQueryText(e.target.value)} autoComplete="off" spellCheck={false} placeholder="e.g. Li O" />
              <span className="cf-hint">
                {query.mode === 'elements' && found
                  ? nothing
                    ? `Read as elements: ${lower(nothingWords)}.`
                    : `Read as elements: ${count(found.pool)} ${verb(found.pool, 'crystal contains', 'crystals contain')} ${list(query.elements)}.`
                  : query.mode === 'text' && found
                    ? `Read as text: ${nothing ? lower(nothingWords) : `${count(found.pool)} ${verb(found.pool, 'crystal has', 'crystals have')} “${query.text}” in the formula, chemical system or JARVIS id`}. For elements, type their symbols with capitals (Li O).`
                    : 'Element symbols with capitals (Li O, or LiFePO4) list the crystals containing all of them. Anything else is looked for inside the formula, chemical system or JARVIS id.'}
              </span>
            </div>
            <div>
              <label htmlFor="cf-model">Model that screens</label>
              <select id="cf-model" value={model} onChange={(e) => setModel(e.target.value)}>
                <option value="rec">Recommended for each property</option>
                {MODELS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
              </select>
              <span className="cf-hint">
                {model !== 'rec'
                  ? `The ${SHORT[model]} screens every property.`
                  : props.some((p) => p.recKnown)
                    ? `From the pipeline's registry: ${props.map(recWords).join('; ')}.${props.some((p) => !p.recKnown) ? ' Where it recommends none, the blend is used.' : ''}`
                    : `This export recommends no model for ${verb(props.length, 'this property', 'these properties')}, so the blend screens ${verb(props.length, 'it', 'each')}.`}
              </span>
            </div>
          </div>
          <div className="examples">
            <span>Try:</span>
            {presets.map((ex) => (
              <button type="button" key={ex.label} className="chip" aria-pressed={sameTexts(prep, shownTexts, filled(prep, ex.set))} onClick={() => choose(filled(prep, ex.set))}>
                {ex.label}
              </button>
            ))}
            <button type="button" className="chip" aria-pressed={sameTexts(prep, shownTexts, blank(prep))} onClick={() => choose(blank(prep))}>No limits: list every crystal</button>
          </div>
          <div className="options">
            <label>
              <input type="checkbox" checked={cautious} onChange={(e) => setCautious(e.target.checked)} />
              <span>Cautious: the whole 90% range of a prediction (prediction ± the model&apos;s half-width) must sit inside your limits</span>
            </label>
          </div>
          <p className="note small">
            The examples use the median and quartiles of the DFT-calculated values of these crystals, not requirements from a standard. Moduli
            are typed in GPa; the models were trained on their log10, so their ranges are a factor either way.
          </p>
        </form>
        {parsed?.errors.map((e, i) => <p className="error" role="alert" key={`${i} ${e}`}>{e}</p>)}
        {parsed?.errors.length > 0 && found && <p className="note small">Until that is corrected, the results below keep the last limits that could be read.</p>}
        {prep.dropped.map((d) => <p className="note small" key={d.key}>{d.label} is left out of the finder: {d.why}.</p>)}
        {notOk.map((t) => <p className="note small" key={t.id}>{t.title ?? t.name}: not in this export ({t.status ?? 'unknown status'}){t.reason ? `: ${t.reason}` : ''}.</p>)}
      </section>

      {found && (
        <section className="cf-result" ref={resultRef} aria-label="Screening result">
          <div className="cf-headline">
            <h2>{headline}</h2>
            {rehearsal && <p className="flag low">Rehearsal: the code ran on a tiny test run; the numbers on this page mean nothing.</p>}
            {exploratory.length > 0 && <p className="cf-flagline"><span className="flag low">Exploratory</span> {exploratory.join('; ')}.</p>}
            {nothing ? null : set.length > 0 ? (
              <p className="note">
                Limits: {set.map((p) => limitWords(p, shownAsk.reqs[p.key])).join('; ')}
                {searchWords ? `, among the ${count(found.pool)} ${verb(found.pool, 'crystal', 'crystals')} ${searchWords}` : ''}. Screened with {modelWords}
                {shownAsk.cautious ? ', at the cautious edge: the whole 90% range of each prediction must sit inside the limits.' : ', on the prediction itself (Cautious is off).'}
                {lacked > 0 && ` ${lacked === found.pool
                  ? verb(lacked, 'That crystal is', `All ${count(lacked)} crystals are`)
                  : `${count(lacked)} of the ${count(found.pool)} crystals ${verb(lacked, 'is', 'are')}`} left out because the export has ${lackWords}.`}
              </p>
            ) : (
              <p className="note">
                No limit is set, so nothing is screened or ranked: the crystals are listed by formula. Set a limit above, or pick an example, to
                screen them.
              </p>
            )}
            {canMiss && !nothing && !unjudged && (
              <div className="cf-views">
                <button type="button" className="chip" aria-pressed={activeView === 'hits'} onClick={() => setView('hits')}>
                  Predicted to meet the limits · {count(found.hits.length)}
                </button>
                <button type="button" className="chip" aria-pressed={activeView === 'missed'} onClick={() => setView('missed')}>
                  DFT-calculated values meet them, the prediction does not · {count(found.missed.length)}
                </button>
              </div>
            )}
            {canMiss && !shownAsk.cautious && activeView === 'hits' && found.hits.length > 0 && (
              <p className="warn">
                Cautious is off. These crystals meet the limits on the model&apos;s expected value only; at the edge of the 90% range they may not.
              </p>
            )}
          </div>
          {cur ? (
            <div className="cf-split">
              <div className="stagewrap">
                <div className="stagehead">
                  <span className="label">{rankWords}</span>
                  <h2><Chem text={cur.c.formula} /></h2>
                  <span className="fine">JARVIS id {cur.c.jid}{cur.c.system ? ` · chemical system ${cur.c.system}` : ''} · held out of training</span>
                </div>
                {cell ? (
                  <CrystalViewer cell={cell} name={cur.c.formula} />
                ) : (
                  <div className="cf-nocell">
                    <p className="note">
                      {structures === undefined ? 'Loading the unit cells…'
                        : structures === null ? 'The unit cells are not in this export, so there is no 3D view. The numbers beside this box do not depend on it.'
                          : `This crystal has no unit cell in the export, so there is no 3D view of it.${Number.isFinite(structures.max_atoms) ? ` Cells with more than ${structures.max_atoms} atoms are left out${Number.isFinite(structures.skipped_larger_cells) ? ` (${count(structures.skipped_larger_cells)} in this export)` : ''}.` : ''}`}
                    </p>
                  </div>
                )}
              </div>
              <div className="readout cf-readout">
                {set.length > 0 && (
                  <p className="cf-status">
                    {activeView === 'missed'
                      ? <><span className="flag low">Not picked</span> {shownAsk.cautious ? 'The cautious edge of its prediction' : 'Its prediction'} is outside your limits by {fmt(-cur.norm, 2)} half-widths, yet its DFT-calculated values meet them.</>
                      : <><span className="flag hot">Predicted to meet your limits</span> Its DFT-calculated values {cur.truth ? 'meet them too.' : 'do not meet them.'}</>}
                  </p>
                )}
                <Detail row={cur} prep={prep} found={found} reqs={shownAsk.reqs} runsText={runsText} position={position} total={rows.length} onStep={step} />
              </div>
            </div>
          ) : (
            <div className="cf-empty">
              <p className="note">
                {nothing
                  ? absent.length
                    ? `None of the ${count(prep.crystals.length)} held-out crystals in this export contains ${list(absent, 'or')}. Try other elements, or clear the search.`
                    : query.mode === 'elements'
                      ? 'Each of these elements is in some held-out crystal, but no crystal has them all. Try fewer elements, or clear the search.'
                      : 'Type element symbols with capitals (Li O), part of a formula, or a JARVIS id; or clear the search.'
                  : found.judged === 0
                    ? `${verb(found.pool, 'This crystal lacks', `Each of these ${count(found.pool)} crystals lacks`)} a property you set a limit on, so the screen has nothing to judge. Remove that limit${query.mode !== 'all' ? ' or clear the search' : ''}.`
                    : activeView === 'missed'
                      ? 'No crystal that meets your limits by its DFT-calculated values was left out by the prediction.'
                      : `No held-out crystal is predicted to meet these limits${shownAsk.cautious ? ' at the cautious edge' : ''}.${shownAsk.cautious && found.test.plain.picked > 0 ? ` ${count(found.test.plain.picked)} would on the prediction itself.` : ''} ${tipText}.`}
              </p>
            </div>
          )}
        </section>
      )}

      {found && rows.length > 0 && (
        <section className="card fam-section cf">
          <h2>{!set.length ? 'The held-out crystals' : activeView === 'missed' ? 'Crystals the screen did not pick' : 'Ranked list'}</h2>
          <p className="note">
            {!set.length
              ? 'By formula. Each property shows the prediction of the screening model beside the DFT-calculated value.'
              : activeView === 'missed'
                ? 'Their DFT-calculated values meet your limits, the prediction does not. Nearest to being picked first.'
                : 'Most room inside your limits first. The DFT-calculated value beside each prediction is the check on it.'}
            {' '}Press Show to see a crystal&apos;s unit cell and all three predictions.
          </p>
          <RankTable rows={rows} shownN={shownN} selected={cur ? cur.c.jid : null} found={found} reqs={shownAsk.reqs} props={props} onPick={pick} />
          <div className="cf-more">
            <span className="note small">Showing {count(shownN)} of {count(rows.length)}.</span>
            {shownN < rows.length && (
              <>
                <button type="button" className="chip" onClick={() => setMore({ key: askKey, n: shownN + PAGE })}>Show {Math.min(PAGE, rows.length - shownN)} more</button>
                <button type="button" className="chip" onClick={() => setMore({ key: askKey, n: rows.length })}>Show all {count(rows.length)}</button>
              </>
            )}
          </div>
          <dl className="readme">
            <div>
              <dt>Predicted</dt>
              <dd>
                The screening model&apos;s prediction for this crystal (the model is named under each property), with its 90% range underneath.
                {floored.length > 0 && ` ${belowZero(floored)}`}
                {' '}The model never trained on any crystal of this chemical system.
              </dd>
            </div>
            <div>
              <dt>DFT-calculated</dt>
              <dd>The value calculated with density functional theory in JARVIS-DFT: a calculation, not a measurement. The models are scored against it.</dd>
            </div>
            {set.length > 0 && (
              <div>
                <dt>Room to the nearest limit</dt>
                <dd>
                  How far the judged value sits inside your limits, counted in 90% half-widths of that property so different units compare.
                  {shownAsk.cautious ? ' Cautious is on, so the judged value is the end of the 90% range nearest the limit.' : ' Cautious is off, so the judged value is the prediction itself.'}
                  {' '}Negative: outside the limits.
                </dd>
              </div>
            )}
            {set.length > 0 && (
              <div>
                <dt>Its DFT-calculated values</dt>
                <dd>Whether the crystal&apos;s own DFT-calculated values meet your limits. This is the check on the prediction, shown because these crystals were held out.</dd>
              </div>
            )}
            <div>
              <dt>Not in the export</dt>
              <dd>
                Each property has its own set of test crystals, so a crystal can lack some. {count(prep.complete)} of the {count(prep.crystals.length)} have
                all {props.length}. A crystal without a property you set a limit on cannot be judged and is left out.
              </dd>
            </div>
          </dl>
        </section>
      )}

      <section className="card fam-section cf">
        <h2>How far to trust this screen</h2>
        {rehearsal ? (
          <p className="cf-flagline">
            <span className="flag low">Rehearsal</span> The export code ran on a tiny test run and wrote {count(props.length)} crystal {verb(props.length, 'task', 'tasks')}.
            No result is stated here: the list above and the figures below show the format only.
          </p>
        ) : found && set.length > 0 && found.test.n > 0 ? (
          <>
            <p className="cf-test">
              Checked against the DFT-calculated values of the {count(found.test.n)} held-out {verb(found.test.n, 'crystal', 'crystals')}
              {searchWords ? ` ${searchWords}` : ''} that {verb(found.test.n, 'has', 'have')} every property you set a limit on:{' '}
              <strong>{count(found.test.real)}</strong> {verb(found.test.real, 'meets', 'meet')} your limits by those values.
            </p>
            <ul className="cf-notes">
              {[['cautious', 'At the cautious edge'], ['plain', 'On the prediction itself']].map(([mode, words]) => {
                const t = found.test[mode]
                const inUse = (mode === 'cautious') === shownAsk.cautious
                return (
                  <li key={mode}>
                    {words}{inUse ? ' (in use)' : ''}, the predictions pick {count(t.picked)}
                    {t.picked > 0 && (
                      <>
                        ; <strong>{count(t.right)} of those {count(t.picked)}</strong> {verb(t.right, 'meets', 'meet')} the limits by {verb(t.right, 'its', 'their')} DFT-calculated
                        values ({pct(t.right, t.picked)}), which finds {count(t.right)} of the {count(found.test.real)} that do{found.test.real > 0 ? ` (${pct(t.right, found.test.real)})` : ''}
                      </>
                    )}.
                  </li>
                )
              })}
            </ul>
            <p className="cf-flagline">
              {exploratory.length > 0 && <><span className="flag low">Exploratory</span> {exploratory.join('; ')}. </>}
              One fixed set of test crystals, not repeated: these shares have no interval.
            </p>
          </>
        ) : found && set.length > 0 ? (
          <p className="note">
            {nothing ? `${nothingWords}, so` : 'None of the crystals found has every property you set a limit on, so'} there is nothing to check the screen against.
            Clear the search{nothing ? '' : ' or remove a limit'} to test it.
          </p>
        ) : (
          <p className="note">Set a limit above to test the screen: the page then counts how many of the crystals it picks meet your limits by their DFT-calculated values.</p>
        )}

        <h3>The models behind the screen</h3>
        <TableScroll see="the numbers">
          <table className="compact cf-models">
            <thead>
              <tr>
                <th>Property, as trained</th>
                <th>Model that screens</th>
                <th className="num">Typical miss</th>
                <th className="num">Miss of always predicting the average</th>
                <th className="num">90% range</th>
                <th className="num">How often it held</th>
              </tr>
            </thead>
            <tbody>
              {props.map((p) => {
                const m = model === 'rec' ? p.rec : model
                const mae = p.task.mae ?? {}
                const logUnit = p.convert ? 'log10 units' : p.unit
                return (
                  <tr key={p.key}>
                    <td className="cf-wrap">{p.label}<span className="label">{p.unitText}</span></td>
                    <td className="cf-wrap">
                      {m === 'network' && p.task.n_runs ? `Network (average of ${p.task.n_runs} runs)` : MODELS.find((x) => x.key === m)?.label}
                      {p.recKnown
                        ? m === p.rec && <span className="label">recommended in the registry</span>
                        : model === 'rec' && <span className="label">no recommendation in the export; the blend is used</span>}
                    </td>
                    <td className="num">{fmt(mae[m])}<span className="label">{p.convert ? `${logUnit}, ${factor(mae[m])}` : logUnit}</span></td>
                    <td className="num">{fmt(mae.mean)}<span className="label">{p.convert ? `${logUnit}, ${factor(mae.mean)}` : logUnit}</span></td>
                    <td className="num">±{fmt(p.hw[m])}<span className="label">{p.convert ? `${factor(p.hw[m])} either way` : logUnit}</span></td>
                    <td className="num">{share(p.coverage[m])}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </TableScroll>
        <p className="note small">
          Typical miss: the mean absolute error on the test crystals, in the units the model was trained on. The 90% range is calibrated on
          validation crystals; &ldquo;how often it held&rdquo; is the share of test crystals whose DFT-calculated value fell inside it.
          {rehearsal ? ' Rehearsal figures: they mean nothing.' : exploratory.length > 0 ? ` Exploratory: ${exploratory.join('; ')}.` : ''}
          {' '}Whether one model is clearly better than another is stated for each property in the pipeline&apos;s test below.
        </p>
        <ul className="cf-notes">
          <li>
            Every crystal here comes from a chemical system held out of training, so each prediction is for a system the model has not seen
            {props[0]?.split ? ` (the export names the split “${props[0].split}”)` : ''}. The properties are DFT-calculated, never measured; the table
            above gives the method with each one.
          </li>
          <li>
            The export shows {props.map((p) => `${count(p.rowsShown)}${p.rowsTotal !== null ? ` of ${count(p.rowsTotal)}` : ''} test crystals for ${lower(p.label)}`).join(', ')}
            {props[0]?.sampleRule ? ` (${props[0].sampleRule})` : ''}. That is {count(prep.crystals.length)} different crystals.
          </li>
          {runsText && <li>&ldquo;Network average&rdquo; is {runsText}; how one network at a time does against the trees is in the pipeline&apos;s test below.</li>}
          <li>
            {structures === undefined ? 'The unit cells are still loading.'
              : structures === null ? 'The unit cells are not in this export, so no crystal has a 3D view.'
                : `Unit cells: ${count(withCell)} of the ${count(prep.crystals.length)} crystals have one in the export${Number.isFinite(structures.max_atoms) ? ` (cells with more than ${structures.max_atoms} atoms are left out)` : ''}. Source: ${structures.source ?? 'not stated'}.`}
          </li>
        </ul>
      </section>
    </>
  )
}
