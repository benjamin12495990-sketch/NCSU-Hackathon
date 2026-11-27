import { useDeferredValue, useEffect, useId, useMemo, useRef, useState } from 'react'
import { loadPipeline, fmt, signed, interval, comparisonText, factor, MODELS, matchRows } from '../../lib/pipeline.js'
import './PipelineResults.css'

// One nightly task of the GPU pipeline, as the export reports it: how the network average, the strongest tree model and their blend
// did on rows they never trained on (mean absolute error, the paired 95% comparisons, the 90% ranges and how often they held), then
// the held-out rows themselves: a parity plot (truth against prediction, with the diagonal and the 90% band of the chosen model) and a
// searchable table. Mounted once per task, so the rows file is fetched only when this part scrolls into view or on a click.
// Every number comes from public/pipeline/ (gpu/export_site.py); nothing is typed here.
// Wording (docs/AGENTS.md): a rehearsal export gets its numbers but no verdict; anything the export marks exploratory says so beside
// the number; for cross-validated tasks the 90% half-width is an AVERAGE of per-fold widths and is called one.

const PAGE = 25
const TOPBAR = 56 // the sticky top bar is 52px tall
const SHORT = { network: 'network average', trees: 'strongest trees', blend: 'blend' }
const BETTER = { network: 'Network average', trees: 'Strongest trees', blend: 'Blend' }
const NOUN = { network: 'the network average', trees: 'the strongest tree model', blend: 'the blend' }
// Label fields: the first one present names the row; the rest are shown small beside it.
const PRIMARY = ['material', 'name', 'composition', 'smiles', 'formula', 'group', 'id', 'jid']
const MONO = new Set(['smiles', 'composition'])
// Words for the export's field names (the small print beside a row and the search box's hint), so no raw field name reaches the page.
// A qualifier is how the source stated the value beside it (a range, "boiling", an upper bound), not a second value.
const KEY_WORDS = {
  material_class: 'class', material_family: 'family', reference: 'ref.', jid: 'JARVIS id', system: 'system', type: 'type', smiles: 'SMILES',
  conc_vol_pct: 'concentration', temp_C: 'temperature', hours: 'exposure',
  conc_qualifier: 'concentration given as', temp_qualifier: 'temperature given as',
}
const SUFFIX = { conc_vol_pct: ' vol%', temp_C: ' °C', hours: ' h' }
const word = (k) => KEY_WORDS[k] ?? k.replace(/_/g, ' ')
// The rule by which the networks' crystal graphs cut each atom's neighbour list (gpu/README.md section 9), in plain words.
const RULES = {
  all: 'neighbours tied at the edge of the list are all kept',
  first: 'the original rule: ties at the edge of the list are cut by the order the atoms are listed in',
}
const ruleText = (r) => (RULES[r] ? `“${r}” (${RULES[r]})` : `“${r}”`)

const isNum = (v) => typeof v === 'number' && Number.isFinite(v)
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s)
const pct = (c) => (isNum(c) ? `${fmt(c * 100, 3)}%` : '–')
const count = (n) => (isNum(n) ? n.toLocaleString() : '–')
const safeId = (id) => id.replace(/[^a-zA-Z0-9_-]/g, '')
const modelKey = (rec) => (!rec ? null : rec.startsWith('network') ? 'network' : rec === 'trees' || rec === 'blend' ? rec : null)
const labelOf = (r) => r.label ?? { formula: r.formula, system: r.system, jid: r.jid }
const reducedMotion = () => typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

// The export's comparison in the project's wording (comparisonText), naming the network as an average. A rehearsal gets the
// difference and its interval but no verdict; `run` is 'real', 'rehearsal' or 'unknown' (index.json not read yet).
function compare(c, run, ex) {
  if (!c) return 'Not in this export.'
  if (run === 'rehearsal') return `Rehearsal: no verdict. Difference ${signed(c.diff)} ${interval(c.ci95)}, which means nothing.`
  if (run !== 'real') return '…'
  return `${cap(comparisonText({ ...c, better: c.better ? BETTER[c.better] ?? c.better : null }))}${ex}`
}

// Said beside the numbers of a task the export marks exploratory, and beside every number of a rehearsal.
const Ex = ({ task, what }) => (task.exploratory ? <> <strong>Exploratory {what}</strong>: {task.exploratory}.</> : null)
const Reh = ({ on }) => (on ? <> <strong>Rehearsal</strong>: these numbers mean nothing.</> : null)

function describe(label) {
  // A qualifier that only says the number beside it is exact adds nothing; any other qualifier is shown.
  const keys = Object.keys(label).filter((k) => label[k] !== null && label[k] !== undefined && label[k] !== ''
    && !(k.endsWith('_qualifier') && label[k] === 'exact'))
  const main = PRIMARY.find((k) => keys.includes(k)) ?? keys[0]
  const rest = keys.filter((k) => k !== main).map((k) => {
    const v = label[k]
    // A qualifier is a token of the parser (upper_bound, room_to_value): written as words. Names and formulas are left as they are.
    const text = isNum(v) ? fmt(v) : k.endsWith('_qualifier') ? String(v).replace(/_/g, ' ') : String(v)
    return SUFFIX[k] ? `${text}${SUFFIX[k]}` : `${word(k)}: ${text}`
  })
  const title = main === undefined ? '' : main === 'group' ? `Group ${label[main]}` : String(label[main])
  return { main, title, rest, keys }
}

// What the page writes for a row: its name and the small print under it. The table, the selected-row panel and the search all read
// this one description, so the search finds exactly the words on the screen.
function shownLabel(row) {
  const lab = labelOf(row)
  const d = describe(lab)
  const rest = d.rest.length ? d.rest
    : row.group !== undefined && !('group' in lab) && String(row.group) !== String(lab[d.main]) ? [`group: ${row.group}`] : []
  return { ...d, rest }
}

function RowLabel({ d }) {
  return (
    <>
      <strong className={MONO.has(d.main) ? 'pr-mono' : undefined}>{d.title || '(no label)'}</strong>
      {d.rest.length > 0 && <span className="pr-rest">{d.rest.join(' · ')}</span>}
    </>
  )
}

// Round axis ticks across [lo, hi].
function ticks(lo, hi, n = 5) {
  const raw = (hi - lo) / n
  const mag = 10 ** Math.floor(Math.log10(raw))
  const e = raw / mag
  const step = mag * (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1)
  const out = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)))
  return out
}

const SIZE = 400, LEFT = 48, BOTTOM = 40, TOP = 10, RIGHT = 12

// Truth (x) against the chosen model's prediction (y). The dashed diagonal is a perfect prediction; the shaded band is ± the model's
// 90% half-width around it (for cross-validated tasks that half-width is the average over the folds, and the caption says so).
// A click or tap selects the nearest point.
function Parity({ rows, model, label, half, truthWord, hits, selected, onSelect }) {
  const clip = `prclip${safeId(useId())}`
  const svgRef = useRef(null)
  const dom = useMemo(() => {
    let lo = Infinity, hi = -Infinity
    for (const r of rows) for (const v of [r.y, r.network, r.trees, r.blend]) if (isNum(v)) { if (v < lo) lo = v; if (v > hi) hi = v }
    if (!Number.isFinite(lo)) return null
    const span = hi - lo || Math.abs(hi) || 1
    return [lo - span * 0.04, hi + span * 0.04]
  }, [rows])
  const scale = useMemo(() => {
    if (!dom) return null
    const w = SIZE - LEFT - RIGHT, h = SIZE - TOP - BOTTOM
    const sx = (v) => LEFT + ((v - dom[0]) / (dom[1] - dom[0])) * w
    const sy = (v) => TOP + h - ((v - dom[0]) / (dom[1] - dom[0])) * h
    return { sx, sy, t: ticks(dom[0], dom[1]) }
  }, [dom])
  // The dots are drawn once per model / search; selecting a point redraws only its marker.
  const dots = useMemo(() => {
    if (!scale) return null
    const r = rows.length > 800 ? 1.9 : 2.6
    return rows.map((row, i) => {
      const p = row[model]
      if (!isNum(row.y) || !isNum(p)) return null
      const cls = hits ? (hits.has(i) ? 'pr-dot pr-hit' : 'pr-dot pr-dim') : 'pr-dot'
      return <circle key={i} cx={scale.sx(row.y)} cy={scale.sy(p)} r={r} className={cls} />
    })
  }, [rows, model, scale, hits])
  if (!scale) return <p className="note">No rows with numbers to plot.</p>
  const { sx, sy, t } = scale
  const [a, b] = dom
  const w = isNum(half) ? half : null
  const pick = (e) => {
    const m = svgRef.current?.getScreenCTM()
    if (!m) return
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse())
    let best = -1, bestD = 16 * 16
    rows.forEach((row, i) => {
      if ((hits && !hits.has(i)) || !isNum(row.y) || !isNum(row[model])) return
      const dx = sx(row.y) - pt.x, dy = sy(row[model]) - pt.y
      const d = dx * dx + dy * dy
      if (d < bestD) { bestD = d; best = i }
    })
    if (best >= 0) onSelect(best)
  }
  const sel = selected !== null ? rows[selected] : null
  return (
    <svg ref={svgRef} viewBox={`0 0 ${SIZE} ${SIZE}`} className="pr-plot" role="img" onClick={pick}
      aria-label={`${cap(truthWord)} value against the prediction of the ${SHORT[model]} for ${rows.length} held-out rows. Select a row in the table below to mark it here.`}>
      <defs><clipPath id={clip}><rect x={LEFT} y={TOP} width={SIZE - LEFT - RIGHT} height={SIZE - TOP - BOTTOM} /></clipPath></defs>
      <rect x={LEFT} y={TOP} width={SIZE - LEFT - RIGHT} height={SIZE - TOP - BOTTOM} className="pr-frame" />
      {t.map((v) => (
        <g key={v}>
          <line x1={sx(v)} y1={TOP} x2={sx(v)} y2={SIZE - BOTTOM} className="pr-grid" />
          <line x1={LEFT} y1={sy(v)} x2={SIZE - RIGHT} y2={sy(v)} className="pr-grid" />
          <text x={sx(v)} y={SIZE - BOTTOM + 14} textAnchor="middle">{fmt(v, 3)}</text>
          <text x={LEFT - 5} y={sy(v) + 3.5} textAnchor="end">{fmt(v, 3)}</text>
        </g>
      ))}
      <g clipPath={`url(#${clip})`}>
        {w !== null && (
          <>
            <polygon className="pr-band" points={`${sx(a)},${sy(a - w)} ${sx(b)},${sy(b - w)} ${sx(b)},${sy(b + w)} ${sx(a)},${sy(a + w)}`} />
            <line x1={sx(a)} y1={sy(a + w)} x2={sx(b)} y2={sy(b + w)} className="pr-edge" />
            <line x1={sx(a)} y1={sy(a - w)} x2={sx(b)} y2={sy(b - w)} className="pr-edge" />
          </>
        )}
        <line x1={sx(a)} y1={sy(a)} x2={sx(b)} y2={sy(b)} className="pr-diag" />
        {dots}
        {sel && isNum(sel.y) && isNum(sel[model]) && <circle cx={sx(sel.y)} cy={sy(sel[model])} r={5.5} className="pr-sel" />}
      </g>
      <text x={(LEFT + SIZE - RIGHT) / 2} y={SIZE - 6} textAnchor="middle" className="pr-axis">{cap(truthWord)} value</text>
      <text x={12} y={(TOP + SIZE - BOTTOM) / 2} textAnchor="middle" className="pr-axis"
        transform={`rotate(-90 12 ${(TOP + SIZE - BOTTOM) / 2})`}>Predicted: {label}</text>
    </svg>
  )
}

function Picked({ row, label, iv, truthWord, unit, cv, task, rehearsal }) {
  if (!row) return <p className="note small">Click or tap a point, or press Show on a table row, to see that row here.</p>
  // Cross-validated tasks: the band drawn is the average of the per-fold half-widths, so it is named an average.
  const band = cv ? 'the average 90% band' : 'its 90% range'
  return (
    <div className="pr-picked">
      <span className="label">Selected held-out row</span>
      <div className="pr-pickname"><RowLabel d={label} /></div>
      <dl className="pr-dl">
        <div><dt>{cap(truthWord)}</dt><dd className="num strong">{fmt(row.y)}</dd></div>
        {MODELS.map((m) => {
          const p = row[m.key], hw = iv?.[m.key]?.half_width
          const miss = isNum(p) && isNum(row.y) ? p - row.y : null
          return (
            <div key={m.key}>
              <dt>Predicted, {SHORT[m.key]}</dt>
              <dd className="num">{fmt(p)} <span className="pr-miss">miss {signed(miss)}</span>
                {miss !== null && isNum(hw) && (
                  <span className={Math.abs(miss) <= hw ? 'flag ok' : 'flag low'}>{Math.abs(miss) <= hw ? 'inside' : 'outside'} {band}</span>
                )}
              </dd>
            </div>
          )
        })}
      </dl>
      <p className="fine">
        Values in the target&apos;s units{unit ? ` (${unit})` : ''}. Miss = prediction minus the {truthWord} value (positive: predicted too
        high). This row was predicted by models that never trained on it.
        {cv && ' The average band is the one drawn on the plot; the export counts each row against its own fold\'s width.'}
        <Ex task={task} what="row" /><Reh on={rehearsal} />
      </p>
    </div>
  )
}

// How many plotted rows fall inside the drawn band, beside the export's own coverage. Two percentages are printed only when they
// describe different sets of rows (a sample against all rows); for the same rows the count stands beside the export's figure.
function BandCount({ inside, coverage, sampled, total, cv, rehearsal }) {
  if (!inside) return null
  const all = isNum(total) ? `all ${count(total)} held-out rows` : 'all the held-out rows'
  if (sampled) {
    return (
      <p className="note small">
        Inside the drawn band on the {count(inside.n)} sampled rows plotted: <strong>{count(inside.k)}</strong> ({pct(inside.k / inside.n)}).
        {isNum(coverage) && ` On ${all} the export reports ${pct(coverage)}${cv ? ', counted with each fold\'s own width' : ''}.`}
        <Reh on={rehearsal} />
      </p>
    )
  }
  return (
    <p className="note small">
      <strong>{count(inside.k)}</strong> of the {count(inside.n)} rows plotted are inside the drawn band{cv ? ', which uses the average half-width' : ''}.
      {isNum(coverage) && (cv
        ? ` The export's figure, ${pct(coverage)} of ${all}, counts each row against its own fold's width, so rows near the edge can be counted differently.`
        : ` The export reports ${pct(coverage)} for these rows.`)}
      <Reh on={rehearsal} />
    </p>
  )
}

function Rows({ data, task, labels, truthWord, isLog, rehearsal }) {
  const [model, setModel] = useState(modelKey(task.recommended) ?? 'blend')
  const [query, setQuery] = useState('')
  const [order, setOrder] = useState('sample')
  const [page, setPage] = useState(PAGE)
  const [selected, setSelected] = useState(null)
  const uid = safeId(useId())
  const figRef = useRef(null)
  const rows = data.rows
  const cv = task.mode === 'cv'
  const crystal = task.kind === 'crystal'
  const iv = data.interval90 ?? task.interval90
  const half = iv?.[model]?.half_width
  const dq = useDeferredValue(query)
  // Each row's name and small print, and the same in lower case for the search, worked out once per rows file.
  const shown = useMemo(() => rows.map(shownLabel), [rows])
  const text = useMemo(() => rows.map((r, i) => {
    const lab = labelOf(r)
    return {
      title: shown[i].title.toLowerCase().replace(/\s+/g, ' '),
      shown: `${shown[i].title} ${shown[i].rest.join(' · ')}`.toLowerCase().replace(/\s+/g, ' '),
      values: shown[i].keys.map((k) => String(lab[k]).replace(/\s+/g, ' ')).join('\n').toLowerCase(),
    }
  }), [rows, shown])
  // matchRows (the shared helper) searches the raw label JSON: it also hits field names, quotes and values the page hides, and it
  // misses the words the page writes ("Group 12", "70 vol%", "class: stainless"). So a row matches when the query is in the text it
  // shows, or when matchRows finds it and it is in the raw value of a field the row shows (a number before rounding, say).
  const q = dq.trim().toLowerCase().replace(/\s+/g, ' ')
  const matches = useMemo(() => {
    if (!q) return rows.map((_, i) => i)
    const raw = new Set(matchRows(rows, q))
    const out = []
    rows.forEach((r, i) => { if (text[i].shown.includes(q) || (raw.has(r) && text[i].values.includes(q))) out.push(i) })
    return out
  }, [rows, q, text])
  const hits = useMemo(() => (q ? new Set(matches) : null), [q, matches])
  // In the exported order a search lists the rows named exactly as typed first ("Group 12" before "Group 120"): `exact` holds them.
  const exact = useMemo(() => (q ? matches.filter((i) => text[i].title === q) : []), [q, matches, text])
  const sorted = useMemo(() => {
    if (order === 'sample') return exact.length && exact.length < matches.length ? [...exact, ...matches.filter((i) => text[i].title !== q)] : matches
    const miss = (i) => (isNum(rows[i][model]) && isNum(rows[i].y) ? Math.abs(rows[i][model] - rows[i].y) : -1)
    const y = (i) => (isNum(rows[i].y) ? rows[i].y : -Infinity)
    const out = [...matches]
    if (order === 'miss') out.sort((a, b) => miss(b) - miss(a))
    else if (order === 'low') out.sort((a, b) => y(a) - y(b))
    else out.sort((a, b) => y(b) - y(a))
    return out
  }, [matches, exact, q, text, order, rows, model])
  // Rows with both a truth value and this model's prediction (the dots), and how many of them fall inside the drawn band.
  const plotted = useMemo(() => {
    let n = 0, k = 0
    for (const r of rows) if (isNum(r.y) && isNum(r[model])) { n += 1; if (isNum(half) && Math.abs(r[model] - r.y) <= half) k += 1 }
    return { n, k }
  }, [rows, model, half])
  const keys = useMemo(() => {
    const seen = new Set()
    for (const r of rows.slice(0, 50)) for (const k of Object.keys(labelOf(r))) seen.add(k)
    return [...seen].map(word).slice(0, 4)
  }, [rows])
  // Alloys, insulators and polymers: the export names a row by its group and nothing else, so several rows share a name.
  const groupNames = useMemo(() => {
    const names = new Set()
    for (const r of rows) {
      const lab = labelOf(r), k = Object.keys(lab)
      if (k.length !== 1 || k[0] !== 'group') return null
      names.add(String(lab.group))
    }
    return names.size || null
  }, [rows])
  const show = (i) => {
    setSelected(i)
    const fig = figRef.current
    if (!fig) return
    // Leave the page where it is when the whole plot is already on screen below the sticky top bar; otherwise bring the plot
    // itself (not the taller block around it) to the top, so the marked point is visible on a phone too.
    const box = (fig.querySelector('svg') ?? fig).getBoundingClientRect()
    if (box.top >= TOPBAR && box.bottom <= window.innerHeight) return
    fig.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'start' })
  }
  const unit = data.unit ?? task.unit
  const total = data.rows_total ?? task.rows
  const have = data.rows_shown ?? rows.length
  const sampled = isNum(total) && have < total
  const outside = cv ? 'outside the average band' : 'outside its 90% range'
  return (
    <>
      <p className="note">
        {sampled
          ? `${count(have)} of the ${count(total)} held-out rows${data.sample_rule ? ` (${data.sample_rule})` : ''}.`
          : `All ${count(have)} held-out rows.`}
        {' '}Each row was predicted by models that never trained on it{cv ? ' (cross-validated by group)' : crystal ? ' (its chemical system was kept out of training)' : ''}.
        {groupNames !== null && ` This family's export names a row only by the group it belongs to${cv ? ' in that cross-validation' : ''}${groupNames < rows.length
          ? `, so rows of one group share a name: the ${count(rows.length)} rows here carry ${count(groupNames)} group names` : ''}.`}
        <Ex task={task} what="rows" /><Reh on={rehearsal} />
      </p>
      {data.split && <p className="fine pr-split">Split as the export records it: {data.split}.</p>}
      <div className="pr-toggle" role="group" aria-label="Model shown in the plot">
        {MODELS.map((m) => (
          <button key={m.key} type="button" className="chip" aria-pressed={model === m.key} onClick={() => setModel(m.key)}>
            {labels[m.key]}{modelKey(task.recommended) === m.key ? ' · recommended' : ''}
          </button>
        ))}
      </div>
      <div className="pr-plotwrap">
        <figure className="pr-figure" ref={figRef}>
          {plotted.n > 0 ? (
            <>
              <Parity rows={rows} model={model} label={model === 'network' ? labels.network : MODELS.find((m) => m.key === model)?.label ?? model}
                half={half} truthWord={truthWord} hits={hits} selected={selected} onSelect={setSelected} />
              <figcaption>
                Each dot is a held-out row: its {truthWord} value across, the prediction of the {SHORT[model]} up. The dashed line is a
                perfect prediction{isNum(half) ? `; the shaded band is ±${fmt(half)}${isLog ? ` (${factor(half)})` : ''}, ${cv
                  ? 'the average half-width of this model\'s 90% ranges (each fold has its own width)' : 'this model\'s 90% range'}` : ''}.
                {hits && ` Solid dots match the search (${count(matches.length)}); faint dots do not.`} Axes in the target&apos;s units{unit ? ` (${unit})` : ''}.
              </figcaption>
            </>
          ) : (
            <p className="note">No row has both a {truthWord} value and a prediction of the {SHORT[model]}, so there is nothing to plot.</p>
          )}
        </figure>
        <div className="pr-side">
          {plotted.n > 0 && isNum(half) && <BandCount inside={plotted} coverage={iv?.[model]?.coverage} sampled={sampled} total={total} cv={cv} rehearsal={rehearsal} />}
          <Picked row={selected !== null ? rows[selected] : null} label={selected !== null ? shown[selected] : null}
            iv={iv} truthWord={truthWord} unit={unit} cv={cv} task={task} rehearsal={rehearsal} />
        </div>
      </div>

      <div className="pr-search">
        <div>
          <label htmlFor={`${uid}q`}>Search the rows</label>
          <input id={`${uid}q`} type="search" value={query} autoComplete="off" spellCheck={false}
            placeholder={keys.length ? `${keys.join(', ')}…` : 'Type to filter'}
            onChange={(e) => { setQuery(e.target.value); setPage(PAGE) }} />
        </div>
        <div>
          <label htmlFor={`${uid}o`}>Order</label>
          <select id={`${uid}o`} value={order} onChange={(e) => setOrder(e.target.value)}>
            <option value="sample">As exported</option>
            <option value="miss">Largest miss first</option>
            <option value="low">{cap(truthWord)} value, low to high</option>
            <option value="high">{cap(truthWord)} value, high to low</option>
          </select>
        </div>
      </div>
      <p className="note small">
        {/* Only the counts are announced when the search changes; the reading note beside them stays quiet. */}
        <span aria-live="polite">
          {q ? `${count(matches.length)} of ${count(rows.length)} rows match.` : `${count(rows.length)} rows.`}
          {sorted.length > 0 && ` Showing ${count(Math.min(page, sorted.length))}.`}
          {order === 'sample' && exact.length > 0 && exact.length < matches.length
            && ` The ${count(exact.length)} named exactly “${shown[exact[0]].title}” ${exact.length === 1 ? 'comes' : 'come'} first.`}
          {order === 'miss' && ` Ordered by the miss of the ${SHORT[model]}.`}
        </span>
        {sorted.length > 0 && ` Miss = prediction minus the ${truthWord} value (positive: predicted too high). Values in the target's units${unit ? ` (${unit})` : ''}.`}
      </p>
      {sorted.length > 0 ? (
        <div className="tablewrap sim-scroll">
          <table className="compact pr-table">
            <thead>
              <tr>
                <th>Held-out row</th>
                <th className="num">{cap(truthWord)}</th>
                {MODELS.map((m) => <th key={m.key} className={m.key === model ? 'num pr-on' : 'num'}>Predicted, {SHORT[m.key]}</th>)}
                <th className="num">Miss of the {SHORT[model]}</th>
                <th><span className="sr">Show on the plot</span></th>
              </tr>
            </thead>
            <tbody>
              {sorted.slice(0, page).map((i) => {
                const r = rows[i]
                const miss = isNum(r[model]) && isNum(r.y) ? r[model] - r.y : null
                return (
                  <tr key={i} className={selected === i ? 'on' : undefined}>
                    {/* The width lives on a block inside the cell: a family page resets min-width on the cell itself at phone width. */}
                    <td className="pr-labelcell"><div className="pr-labelbox"><RowLabel d={shown[i]} /></div></td>
                    <td className="num strong">{fmt(r.y)}</td>
                    {MODELS.map((m) => <td key={m.key} className="num">{fmt(r[m.key])}</td>)}
                    <td className="num">
                      {signed(miss)}
                      {miss !== null && isNum(half) && Math.abs(miss) > half && <span className="label">{outside}</span>}
                    </td>
                    <td>
                      <button type="button" className="chip" aria-pressed={selected === i} onClick={() => show(i)}
                        aria-label={`Show ${shown[i].title || 'this row'} on the plot`}>Show</button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="note">{q ? <>No row matches &ldquo;{dq}&rdquo;. Try part of a name, or of the small print under it.</> : 'No rows to list.'}</p>
      )}
      {sorted.length > page && (
        <p className="more"><button type="button" className="chip" onClick={() => setPage((n) => n + PAGE)}>Show {Math.min(PAGE, sorted.length - page)} more</button></p>
      )}
    </>
  )
}

export default function PipelineResults({ task }) {
  const [rehearsal, setRehearsal] = useState(undefined) // undefined: index.json not read yet
  const [want, setWant] = useState(false)
  const [data, setData] = useState(undefined) // undefined: not fetched yet; null: missing; object: the rows file
  const lazyRef = useRef(null)
  const file = task?.status === 'ok' ? task.rows_file : null

  // index.json is cached by loadPipeline (SimPage already asked for it), so this costs no second download. A request cancelled by
  // another caller is asked for once more, so a verdict is never printed before the page knows whether this is a rehearsal.
  useEffect(() => {
    let live = true
    const set = (ix) => { if (live) setRehearsal(!!ix?.rehearsal) }
    loadPipeline('index.json').then(set, () => loadPipeline('index.json').then(set, () => set(null)))
    return () => { live = false }
  }, [])
  // Fetch the rows only once this part of the page is near the screen (or the visitor asks).
  useEffect(() => {
    const el = lazyRef.current
    if (want || !file || !el || typeof IntersectionObserver === 'undefined') return undefined
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) setWant(true) }, { rootMargin: '250px 0px' })
    io.observe(el)
    return () => io.disconnect()
  }, [want, file])
  useEffect(() => {
    if (!want || !file) return undefined
    let live = true
    loadPipeline(file).then((d) => { if (live) setData(d && Array.isArray(d.rows) ? d : null) }, () => { if (live) setData(null) })
    return () => { live = false }
  }, [want, file])

  if (!task) return null
  const title = task.title ?? task.name ?? task.id
  if (task.status !== 'ok') {
    return (
      <section className="card fam-section pr">
        <h2>{title}: the pipeline&apos;s test</h2>
        <p className="note">Not in this export ({task.status ?? 'unknown status'}){task.reason ? `: ${task.reason}` : ''}.</p>
      </section>
    )
  }
  const crystal = task.kind === 'crystal'
  const cv = task.mode === 'cv'
  const unit = task.unit ?? ''
  const isLog = /log10/i.test(unit)
  const truthWord = crystal || /DFT/i.test(unit) ? 'DFT-calculated' : 'measured'
  // A rehearsal (the index or the rows file says so) gets its numbers but no result sentence.
  const rehearsalRun = rehearsal === true || !!data?.rehearsal
  const run = rehearsalRun ? 'rehearsal' : rehearsal === false ? 'real' : 'unknown'
  const ex = task.exploratory ? ' (exploratory)' : ''
  const exTag = task.exploratory ? <> <span className="pr-extag">· exploratory</span></> : null
  const rec = modelKey(task.recommended)
  const mae = task.mae ?? {}
  const iv = task.interval90 ?? {}
  const sn = task.single_network
  const nRuns = task.n_runs ?? sn?.n_runs
  const labels = {
    network: isNum(nRuns) ? `Network (average of ${nRuns} runs)` : MODELS[0].label,
    trees: crystal && task.strongest_trees ? `${MODELS[1].label} (feature set: ${task.strongest_trees})` : MODELS[1].label,
    blend: MODELS[2].label,
  }
  const mode = cv ? 'cross-validated by group (each row predicted by models trained without its group)'
    : crystal ? 'test crystals from chemical systems kept out of training' : task.mode
  // Models this run's own comparisons call clearly better than the recommended one (the recipe is not changed by one run).
  const beats = rec && run === 'real'
    ? [...new Set([task.network_vs_trees, task.blend_vs_trees, task.blend_vs_network]
      .filter((c) => c?.better && c.better !== rec && NOUN[c.better] && (c.first === rec || c.second === rec)).map((c) => c.better))]
    : []
  // docs/AGENTS.md: a claim that the average of the networks is clearly better comes with the count for one network at a time. The
  // export counts single networks against the trees only, so the count names the trees when the recommended model is another one.
  const single = beats.includes('network') && sn && isNum(sn.clearly_better) && isNum(sn.n_runs)
    ? `; one network at a time is clearly better than ${rec === 'trees' ? 'it' : NOUN.trees} in ${sn.clearly_better} of ${sn.n_runs} runs${rec === 'trees' ? '' : ' (the export counts single networks against the trees only)'}`
    : ''
  return (
    <section className="card fam-section pr" aria-label={`${title}: the pipeline's test`}>
      <h2>{title}: the pipeline&apos;s test on rows it never trained on</h2>
      <p className="note">
        {run === 'rehearsal' ? 'Rehearsal run' : run === 'real' ? 'Nightly run' : 'Run'} of {task.trained ?? 'an unknown date'}.{' '}
        {count(task.rows)} {crystal ? 'crystals' : 'rows'} in {count(task.groups)}{' '}
        {crystal ? 'chemical systems' : 'groups'}{mode ? `, ${mode}` : ''}. Target: {unit || 'not stated'}.
        {crystal && task.neighbor_rule?.length > 0
          && ` How the networks' crystal graphs cut each atom's neighbour list: rule ${task.neighbor_rule.map(ruleText).join(' and ')}.`}
      </p>
      {rehearsalRun && <p className="flag low">Rehearsal: the code ran on a tiny test run; these numbers mean nothing.</p>}
      {task.exploratory && <p className="pr-flagline"><span className="flag low">Exploratory</span> {task.exploratory}.</p>}

      <h3>Typical miss (mean absolute error, lower is better){exTag}</h3>
      <div className="pr-mae">
        {MODELS.map((m) => (
          <div key={m.key} className={rec === m.key ? 'pr-cell pr-rec' : 'pr-cell'}>
            <span className="label">{labels[m.key]}</span>
            <span className="big">{fmt(mae[m.key])}</span>
            {rec === m.key && <span className="flag plain">Recommended by the recipe</span>}
          </div>
        ))}
        <div className="pr-cell">
          <span className="label">Predicting the average every time (no model)</span>
          <span className="big">{fmt(mae.mean)}</span>
        </div>
      </div>
      <p className="note small">
        In the target&apos;s units{crystal ? ', on the test crystals' : ', out of fold'}.{' '}
        {truthWord === 'DFT-calculated' ? 'The values the models are scored against are DFT-calculated, not measured.' : 'Scored against the measured values.'}
        <Ex task={task} what="errors" /><Reh on={rehearsalRun} />
      </p>
      {rec && (
        <p className="note small">
          &ldquo;Recommended&rdquo; is the model the project&apos;s reviewed recipe (gpu/recipes.json) names for this task. It was set from
          earlier evidence by a reviewed commit, not by this run{run === 'real' ? '; this run\'s comparisons are below' : ''}.
          {beats.length > 0 && ` In them, ${beats.map((k) => (k === 'network' && isNum(nRuns) ? `the average of ${nRuns} networks` : NOUN[k])).join(' and ')} ${beats.length > 1 ? 'are each' : 'is'} clearly better than ${NOUN[rec]}, the recommended model${ex}${single}.`}
        </p>
      )}

      <h3>Is one model really better?{exTag}</h3>
      <ul className="pr-comp">
        <li><span className="pr-pair">Network average against strongest trees</span> {compare(task.network_vs_trees, run, ex)}</li>
        <li><span className="pr-pair">Blend against strongest trees</span> {compare(task.blend_vs_trees, run, ex)}</li>
        <li><span className="pr-pair">Blend against network average</span> {compare(task.blend_vs_network, run, ex)}</li>
        {sn && (
          <li>
            <span className="pr-pair">One network at a time against the trees</span>
            {' '}{run === 'rehearsal' ? 'Rehearsal: no count stated.'
              : run === 'real' ? `Clearly better in ${sn.clearly_better} of ${sn.n_runs}, clearly worse in ${sn.clearly_worse} of ${sn.n_runs}${ex}.` : '…'}
            {' '}Average error of one network at a time over the {sn.n_runs} runs: {fmt(sn.mean_mae)}{run === 'rehearsal' ? ', which means nothing' : ''}.
          </li>
        )}
      </ul>
      <p className="note small">
        Each figure is the error of the first minus the error of the second (negative: the first misses less), with its 95% interval from
        resampling whole groups.{' '}
        {run === 'rehearsal' ? 'This is a rehearsal, so no verdict is drawn: the figures only show that the code ran.'
          : <>&ldquo;Clearly better&rdquo; only when the whole interval is on one side of zero.</>}
      </p>

      <h3>90% ranges{exTag}</h3>
      <div className="tablewrap sim-scroll">
        <table className="compact pr-iv">
          <thead>
            <tr>
              <th>Model</th>
              <th className="num">{cv ? 'Average ± over the folds' : 'Prediction ±'}{isLog ? ' (and as a factor)' : ''}</th>
              <th className="num">How often it held</th>
            </tr>
          </thead>
          <tbody>
            {MODELS.map((m) => (
              <tr key={m.key}>
                <td>{labels[m.key]}{rec === m.key ? ' (recommended)' : ''}</td>
                <td className="num">
                  {isNum(iv[m.key]?.half_width) ? `±${fmt(iv[m.key].half_width)}` : '–'}
                  {isLog && isNum(iv[m.key]?.half_width) && <span className="label">{factor(iv[m.key].half_width)}</span>}
                </td>
                <td className="num">{pct(iv[m.key]?.coverage)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note small">
        {cv ? (
          <>A prediction ± its fold&apos;s half-width is built to hold the true value 9 times in 10. Each fold&apos;s half-width is set from the
            misses on the other folds; the ± shown is the average of those widths. &ldquo;How often it held&rdquo; is the share of all the
            held-out rows that fell inside their own fold&apos;s range.</>
        ) : (
          <>A prediction ± the half-width is built to hold the true value 9 times in 10. &ldquo;How often it held&rdquo; is the share of the held-out
            {crystal ? ' test crystals where it did (the range is calibrated on the validation crystals)' : ' rows where it did'}.</>
        )}
        {isLog && ' The target is a log10, so ±w is a factor of 10^w either way.'}
        <Ex task={task} what="ranges" /><Reh on={rehearsalRun} />
      </p>

      <h3>Every held-out row: truth against prediction{exTag}</h3>
      <div ref={lazyRef} className="pr-rows">
        {!file ? (
          <p className="note">This task&apos;s rows are not published on the site; only the numbers above are.</p>
        ) : data === undefined ? (
          <>
            <p className="note">
              The export holds this task&apos;s held-out rows ({count(task.rows)} in all; a fixed sample of them when there are many), each with
              its {truthWord} value and all three predictions.
            </p>
            <button type="button" className="chip" onClick={() => setWant(true)} disabled={want}>{want ? 'Loading the rows…' : 'Show the rows'}</button>
          </>
        ) : data === null ? (
          <p className="note">The rows file ({file}) is not in this export.</p>
        ) : data.rows.length === 0 ? (
          <p className="note">The rows file ({file}) holds no rows, so there is nothing to plot or list.</p>
        ) : (
          <Rows data={data} task={task} labels={labels} truthWord={truthWord} isLog={isLog} rehearsal={rehearsalRun} />
        )}
      </div>
    </section>
  )
}
