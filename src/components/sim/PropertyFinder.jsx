import { useDeferredValue, useEffect, useId, useMemo, useRef, useState } from 'react'
import { loadPipeline, fmt, signed, factor, MODELS } from '../../lib/pipeline.js'
import './PropertyFinder.css'

// Property finder (see README.md in this folder), used for alloys, insulators, polymers and general chemistry. The visitor sets a
// target (the value must be at least, or at most, a number in the task's unit), chooses one of the pipeline's three models and
// whether a prediction must meet the target at the cautious edge of its 90% range. The page then picks, from the task's rows file,
// the rows predicted to meet the target and checks every pick against the value in the data. Those rows are out of fold: each was
// predicted by models that never trained on it, so the check shows how the finder does on unseen rows. It reports how many picks really meet
// the target (precision), how many of the rows that meet it were found, and what picking at random would give.
// Every number shown comes from index.json and the task's rows file; PAGE is only how many rows a "show more" adds.

// index.json is asked for as soon as this module loads, with no signal. loadPipeline keeps one shared request per file and ties it
// to its first caller's AbortSignal; the pages that mount this finder (SimPage, FamilyPipeline) pass one and cancel it when they
// unmount. Under React's development double mount, or when a page is left while the file is on its way, every later caller is then
// handed the cancelled request and the finder never appears. A first request that nobody can cancel avoids that for index.json.
// It is a workaround kept inside this file: the cure is for loadPipeline not to tie the shared request to one caller's signal.
if (typeof window !== 'undefined' && typeof fetch === 'function') loadPipeline('index.json').catch(() => {})

const PAGE = 25
const SHORT = { network: 'network average', trees: 'strongest trees', blend: 'blend' }
const LISTS = ['hits', 'wrong', 'missed']

// Words for the four tasks this finder serves; any other task falls back to GENERIC. `unit` is the short unit written after a
// number, `plain` names the value behind a log10 target so the visitor can type it directly. Both are used only while the export's
// own unit text still says the same thing (shortUnit, and the `when` test), so a changed export is never given the wrong unit.
const WORDS = {
  alloys: { things: 'alloys', unit: 'HV', rowHead: 'Alloy (atomic fractions)', filter: 'an element, for example Ti' },
  polymers: { things: 'polymer rows', unit: 'K', rowHead: 'Polymer row', filter: 'a group id' },
  insulators: { things: 'compounds', unit: '', rowHead: 'Compound', filter: 'a group id', plain: { label: 'dielectric constant', unit: '', when: /dielectric constant/i } },
  chemistry: { things: 'compounds', unit: '', rowHead: 'Compound', filter: 'an AqSolDB id', plain: { label: 'solubility', unit: 'mol/L', when: /solubility.*log10\s*\(?\s*mol\s*\/\s*L/i } },
}
const GENERIC = { things: 'rows', unit: '', rowHead: 'Held-out row', filter: 'part of a label' }
const PLAIN = { label: 'value before the log10', unit: '' }

const isNum = (v) => typeof v === 'number' && Number.isFinite(v)
const num = (v) => (isNum(v) ? v : null)
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s)
const count = (n) => (isNum(n) ? n.toLocaleString() : '–')
const share = (a, b) => (b ? `${fmt((a / b) * 100, 3)}%` : '–')
const verb = (n, one, many) => (n === 1 ? one : many)
const safeId = (id) => id.replace(/[^a-zA-Z0-9_-]/g, '')

// The short unit is written after a number only when the export's unit text ends with it ("Vickers hardness, HV").
function shortUnit(unit, short) {
  const u = String(unit ?? '').trim()
  if (!short || !u.endsWith(short)) return ''
  return u.length === short.length || /[^A-Za-z0-9]/.test(u[u.length - short.length - 1]) ? short : ''
}

// A number in running text, never in e-notation: plain decimals from 0.001 up to a billion, "2.51 × 10⁻¹²" outside that. With
// `sig` it is rounded as fmt rounds; without, every digit is kept (a target is echoed exactly as it is judged).
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' }
function plainNum(x, sig) {
  if (!isNum(x)) return '–'
  if (x === 0) return '0'
  const a = Math.abs(x)
  if (a >= 1e-3 && a < 1e9) return sig ? fmt(x, sig) : String(x)
  const [m, e] = (sig ? x.toExponential(sig - 1) : x.toExponential()).split('e')
  return `${Number(m)} × 10${[...String(Number(e))].map((c) => SUP[c]).join('')}`
}
// 10 to the power of a log10 value, in the same form. A power too large or too small to hold as a number is written out in words.
function tenTo(v, sig = 3) {
  if (!isNum(v)) return '–'
  const x = 10 ** v
  return isNum(x) && x > 0 ? plainNum(x, sig) : `10 to the power of ${plainNum(v, 4)}`
}
const reducedMotion = () => typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

// A target is met at or above it ("at least") or at or below it ("at most").
const meets = (dir, v, t) => (dir === 'min' ? v >= t : v <= t)
// How far inside the target a value sits (negative: outside).
const room = (dir, v, t) => (dir === 'min' ? v - t : t - v)
// The cautious edge of a prediction's 90% range: the side that makes the target harder to meet.
const edgeOf = (dir, p, hw) => (dir === 'min' ? p - hw : p + hw)
const dirWord = (d) => (d === 'min' ? 'at least' : 'at most')
// The sign of the cautious edge; a no-break space keeps it on one line with its number.
const sign = (d) => (d === 'min' ? '−' : '+')

// A typed number: "35", "-2.5", "35,5" (decimal comma), "1e-3", or with grouped thousands where the grouping is certain
// ("1,000,000", "1,000.5", "1.000,5"). `error` is '' when the text is usable or not finished:
//   blank, or a number still being typed ("-", "1e-"): no value yet, and no error (`pending` when something is typed);
//   'ambiguous': one comma before exactly three digits ("1,000") is a thousand to some and a decimal to others, so it is refused
//                with both readings instead of guessed;
//   'big': too large to hold as a number (and to draw); 'form': anything else, such as 0x23 or words.
const BIGGEST = 1e300
function parseNumber(text) {
  const t = String(text).trim().replace(/[−–]/g, '-')
  if (t === '') return { value: null, error: '' }
  if (/^[-+]?[.,]?$/.test(t) || /^[-+]?(\d+[.,]?\d*|[.,]\d+)e[-+]?$/i.test(t)) return { value: null, error: '', pending: true }
  const m = /^([-+]?)([\d.,]+)(e[-+]?\d+)?$/i.exec(t)
  if (!m) return { value: null, error: 'form' }
  const minus = m[1] === '-' ? '-' : ''
  let body = m[2]
  if (/^\d{1,3}(,\d{3})+\.\d*$/.test(body) || /^\d{1,3}(,\d{3}){2,}$/.test(body)) body = body.replace(/,/g, '')
  else if (/^\d{1,3}(\.\d{3})+,\d*$/.test(body) || /^\d{1,3}(\.\d{3}){2,}$/.test(body)) body = body.replace(/\./g, '').replace(',', '.')
  else if (/^[1-9]\d{0,2},\d{3}$/.test(body)) {
    const tail = m[3] ?? ''
    return { value: null, error: 'ambiguous', typed: `${minus}${body}${tail}`, readings: [`${minus}${body.replace(',', '')}${tail}`, `${minus}${body.replace(',', '.')}${tail}`] }
  } else body = body.replace(',', '.')
  if (!/^(\d+(\.\d*)?|\.\d+)$/.test(body)) return { value: null, error: 'form' }
  const v = Number(`${minus}${body}${m[3] ?? ''}`)
  if (!Number.isFinite(v) || Math.abs(v) > BIGGEST) return { value: null, error: 'big' }
  return { value: v, error: '' }
}

// Everything that depends only on the rows file: the usable rows, the quartiles of the values in the data (for the examples) and
// the rows of each group (held out together).
function prepare(data) {
  const rows = []
  let skipped = 0
  data.rows.forEach((raw, i) => {
    if (!raw || !isNum(raw.y)) { skipped += 1; return }
    rows.push({ i, raw, y: raw.y, network: num(raw.network), trees: num(raw.trees), blend: num(raw.blend), group: raw.group ?? raw.label?.group ?? null })
  })
  const ys = rows.map((r) => r.y).sort((a, b) => a - b)
  const at = (p) => (ys.length ? ys[Math.round(p * (ys.length - 1))] : null)
  const groups = new Map()
  for (const r of rows) {
    if (r.group === null) continue
    const k = String(r.group)
    if (!groups.has(k)) groups.set(k, [])
    groups.get(k).push(r)
  }
  // When the site carries only a sample of the held-out rows, other rows of a group can sit outside it.
  const sampled = isNum(data.rows_total) && (isNum(data.rows_shown) ? data.rows_shown : data.rows.length) < data.rows_total
  return { rows, skipped, min: ys[0] ?? null, max: ys[ys.length - 1] ?? null, q1: at(0.25), median: at(0.5), q3: at(0.75), groups, sampled }
}

// The finder for one model and one setting: every row with its judged value, the picks (furthest inside the target first), the
// picks that do not meet it when checked (furthest outside first) and the rows that meet the target but were left out (best first).
function find(prep, ask, half) {
  const useEdge = ask.cautious && isNum(half)
  const items = []
  const byRow = new Map()
  let lo = ask.target
  let hi = ask.target
  let real = 0
  for (const r of prep.rows) {
    const p = r[ask.model]
    if (p === null) continue
    const edge = isNum(half) ? edgeOf(ask.dir, p, half) : null
    const judged = useEdge ? edge : p
    const it = { r, p, edge, judged, margin: room(ask.dir, judged, ask.target), really: meets(ask.dir, r.y, ask.target), trueRoom: room(ask.dir, r.y, ask.target) }
    it.picked = it.margin >= 0
    if (it.really) real += 1
    items.push(it)
    byRow.set(r.i, it)
    lo = Math.min(lo, r.y, judged)
    hi = Math.max(hi, r.y, judged)
  }
  const hits = items.filter((it) => it.picked).sort((a, b) => b.margin - a.margin || a.r.i - b.r.i)
  const wrong = hits.filter((it) => !it.really).sort((a, b) => a.trueRoom - b.trueRoom || a.r.i - b.r.i)
  const missed = items.filter((it) => !it.picked && it.really).sort((a, b) => b.trueRoom - a.trueRoom || a.r.i - b.r.i)
  const span = hi - lo || Math.abs(hi) || 1
  return { items, byRow, hits, wrong, missed, n: items.length, real, right: hits.length - wrong.length, useEdge, dom: [lo - span * 0.04, hi + span * 0.04] }
}

// The same count for any model and setting, without the lists: how many rows it picks and how many of those really meet the target.
function tally(prep, ask, model, cautious, half) {
  const useEdge = cautious && isNum(half)
  const out = { model, cautious, usable: !cautious || isNum(half), n: 0, picked: 0, right: 0, real: 0 }
  for (const r of prep.rows) {
    const p = r[model]
    if (p === null) continue
    out.n += 1
    const really = meets(ask.dir, r.y, ask.target)
    if (really) out.real += 1
    if (meets(ask.dir, useEdge ? edgeOf(ask.dir, p, half) : p, ask.target)) {
      out.picked += 1
      if (really) out.right += 1
    }
  }
  return out
}

// How a row is named. The export carries labels, not names: alloys by composition, AqSolDB compounds by id, the rest by group id.
// `title` and `rest` are what the page prints; `words` are the printed parts that name the row, which a search must also find.
function describeRow(r, taskName, prep) {
  const lab = r.raw.label ?? {}
  const has = (k) => lab[k] !== undefined && lab[k] !== null && lab[k] !== ''
  if (taskName === 'alloys' && has('group')) {
    // Several rows can share one composition; say so, or two lines of the table look like one row listed twice.
    const same = prep?.groups.get(String(r.group))?.length ?? 1
    const where = prep?.sampled ? ' on the site' : ''
    return { title: String(lab.group), comp: true, words: [], rest: same > 1 ? [`one of ${same} held-out rows${where} with this composition (row ${r.i + 1} of the export)`] : [] }
  }
  if (taskName === 'chemistry' && has('id')) {
    const rest = [has('name') ? `AqSolDB group ${lab.name}` : null, r.group !== null ? `InChIKey block ${r.group}` : null].filter(Boolean)
    return { title: `AqSolDB ${lab.id}`, rest, words: rest }
  }
  const keys = Object.keys(lab).filter(has)
  const main = ['name', 'material', 'composition', 'formula', 'smiles', 'id', 'group'].find((k) => keys.includes(k)) ?? keys[0]
  if (main === undefined) {
    const rest = r.group !== null ? [`group ${r.group}`] : []
    return { title: `Row ${r.i + 1} of the export`, rest, words: rest }
  }
  const words = keys.filter((k) => k !== main).map((k) => `${k.replace(/_/g, ' ')}: ${lab[k]}`)
  return { title: main === 'group' ? `Group ${lab.group}` : String(lab[main]), words, rest: main === 'group' ? [...words, `row ${r.i + 1} of the export`] : words }
}

// A search finds a row by what the table prints for it (its title and the label parts under it) and by the label's own values and
// the group, never by the field names. Two forms are exact: "row 494" is that row of the export, and a capitalised element symbol
// typed for an alloy is that element ("C" is carbon, not also Co, Cr and Cu).
function matchItems(list, query, taskName) {
  const q = query.trim().replace(/\s+/g, ' ')
  if (!q) return list
  const row = /^row ?(\d+)\b/i.exec(q)
  if (row) return list.filter((it) => it.r.i + 1 === Number(row[1]))
  if (taskName === 'alloys' && /^[A-Z][a-z]?$/.test(q)) {
    return list.filter((it) => String(it.r.raw.label?.group ?? '').split(/\s+/).some((tok) => tok.replace(/[\d.]+$/, '') === q))
  }
  const low = q.toLowerCase()
  return list.filter((it) => {
    const d = describeRow(it.r, taskName)
    return [d.title, ...d.words, ...Object.values(it.r.raw.label ?? {}), it.r.group ?? ''].join(' ').toLowerCase().includes(low)
  })
}

// "Al0.200 Co0.200 Cr0.200" with the fractions as subscripts; any other text is shown as it is.
function Composition({ text }) {
  const parts = String(text).trim().split(/\s+/).map((t) => /^([A-Z][a-z]?)(\d*\.?\d+)$/.exec(t))
  if (!parts.length || parts.some((m) => !m)) return <>{text}</>
  return <span className="pf-comp">{parts.map((m, k) => <span key={k}>{m[1]}<sub>{m[2]}</sub>{k < parts.length - 1 ? ' ' : ''}</span>)}</span>
}

function RowName({ r, taskName, prep, bare }) {
  const d = describeRow(r, taskName, prep)
  return (
    <>
      <strong>{d.comp ? <Composition text={d.title} /> : d.title}</strong>
      {!bare && d.rest.length > 0 && <span className="pf-rest">{d.rest.join(' · ')}</span>}
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
  for (let k = Math.ceil(lo / step); k * step <= hi + step * 1e-9; k += 1) out.push(Number((k * step).toPrecision(12)))
  return out
}

const SIZE = 400, LEFT = 50, BOTTOM = 40, TOP = 10, RIGHT = 12

// The test as a picture: the value in the data (across) against the value the finder judged (up). The two dashed lines are the
// target, so the plot falls into four parts: picked and right, picked and wrong, left out though it meets the target, rightly left
// out. A click or tap selects the nearest row.
function Quadrants({ found, ask, truthWord, judgedWord, modelWord, selected, onSelect }) {
  const clip = `pfclip${safeId(useId())}`
  const svgRef = useRef(null)
  const { items, dom } = found
  const scale = useMemo(() => {
    const w = SIZE - LEFT - RIGHT, h = SIZE - TOP - BOTTOM
    const sx = (v) => LEFT + ((v - dom[0]) / (dom[1] - dom[0])) * w
    const sy = (v) => TOP + h - ((v - dom[0]) / (dom[1] - dom[0])) * h
    return { sx, sy, t: ticks(dom[0], dom[1]) }
  }, [dom])
  // Left-out rows are drawn first so the picks sit on top. Selecting a row redraws only its marker.
  const dots = useMemo(() => {
    const rad = items.length > 800 ? 1.9 : 2.6
    const out = [], picks = []
    for (const it of items) {
      const c = <circle key={it.r.i} cx={scale.sx(it.r.y)} cy={scale.sy(it.judged)} r={rad} className={it.picked ? (it.really ? 'pf-dot pf-tp' : 'pf-dot pf-fp') : 'pf-dot pf-out'} />
      if (it.picked) picks.push(c)
      else out.push(c)
    }
    return [out, picks]
  }, [items, scale])
  const { sx, sy, t } = scale
  const [a, b] = dom
  const tx = sx(ask.target), ty = sy(ask.target)
  const pick = (e) => {
    const m = svgRef.current?.getScreenCTM()
    if (!m) return
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse())
    let best = null, bestD = 16 * 16
    for (const it of items) {
      const dx = sx(it.r.y) - pt.x, dy = sy(it.judged) - pt.y
      const d = dx * dx + dy * dy
      if (d < bestD) { bestD = d; best = it }
    }
    if (best) onSelect(best.r.i)
  }
  return (
    <svg ref={svgRef} viewBox={`0 0 ${SIZE} ${SIZE}`} className="pf-plot" role="img" onClick={pick}
      aria-label={`${cap(truthWord)} value against the ${judgedWord} of the ${modelWord} for ${items.length} held-out rows, with the target drawn on both axes. The counts of the four parts are listed below the plot.`}>
      <defs><clipPath id={clip}><rect x={LEFT} y={TOP} width={SIZE - LEFT - RIGHT} height={SIZE - TOP - BOTTOM} /></clipPath></defs>
      <rect x={LEFT} y={ask.dir === 'min' ? TOP : ty} width={SIZE - LEFT - RIGHT} height={Math.max(0, ask.dir === 'min' ? ty - TOP : SIZE - BOTTOM - ty)} className="pf-zone" />
      <rect x={LEFT} y={TOP} width={SIZE - LEFT - RIGHT} height={SIZE - TOP - BOTTOM} className="pf-frame" />
      {t.map((v) => (
        <g key={v}>
          <line x1={sx(v)} y1={TOP} x2={sx(v)} y2={SIZE - BOTTOM} className="pf-gridline" />
          <line x1={LEFT} y1={sy(v)} x2={SIZE - RIGHT} y2={sy(v)} className="pf-gridline" />
          <text x={sx(v)} y={SIZE - BOTTOM + 14} textAnchor="middle">{plainNum(v, 3)}</text>
          <text x={LEFT - 5} y={sy(v) + 3.5} textAnchor="end">{plainNum(v, 3)}</text>
        </g>
      ))}
      <g clipPath={`url(#${clip})`}>
        <line x1={sx(a)} y1={sy(a)} x2={sx(b)} y2={sy(b)} className="pf-diag" />
        {dots[0]}
        <line x1={LEFT} y1={ty} x2={SIZE - RIGHT} y2={ty} className="pf-target" />
        <line x1={tx} y1={TOP} x2={tx} y2={SIZE - BOTTOM} className="pf-target" />
        {dots[1]}
        {selected && <circle cx={sx(selected.r.y)} cy={sy(selected.judged)} r={5.5} className="pf-sel" />}
      </g>
      <text x={(LEFT + SIZE - RIGHT) / 2} y={SIZE - 6} textAnchor="middle" className="pf-axis">{cap(truthWord)} value</text>
      <text x={12} y={(TOP + SIZE - BOTTOM) / 2} textAnchor="middle" className="pf-axis"
        transform={`rotate(-90 12 ${(TOP + SIZE - BOTTOM) / 2})`}>{cap(judgedWord)}, {modelWord}</text>
    </svg>
  )
}

// One row in full: what the data says, what each of the three models predicted and whether each would have picked it, and the
// other rows of its group (the nearest things to it in the data, held out with it).
function Detail({ item, heading, prep, ask, halves, taskName, truthWord, val, plainOf, showPlain }) {
  if (!item) return <p className="note small">Click or tap a dot, or press Show on a table row, to see that row here.</p>
  const r = item.r
  const mates = r.group === null ? [] : (prep.groups.get(String(r.group)) ?? []).filter((m) => m !== r)
  const verdict = item.picked
    ? (item.really ? { cls: 'flag ok', text: `Picked, and its ${truthWord} value meets the target` } : { cls: 'flag low', text: `Picked, but its ${truthWord} value does not meet the target` })
    : (item.really ? { cls: 'flag low', text: `Left out, though its ${truthWord} value meets the target` } : { cls: 'flag plain', text: `Left out, and its ${truthWord} value does not meet the target` })
  return (
    <div className="pf-detail">
      <span className="label">{heading}</span>
      <div className="pf-name"><RowName r={r} taskName={taskName} prep={prep} /></div>
      <p className="pf-verdict"><span className={verdict.cls}>{verdict.text}</span></p>
      <dl className="pf-dl">
        <div>
          <dt>{cap(truthWord)}</dt>
          <dd>
            <span className="strong">{val(r.y)}</span>
            {showPlain && <span className="pf-sub">{plainOf(r.y)}</span>}
            <span className="pf-sub">{plainNum(Math.abs(item.trueRoom), 4)} {item.trueRoom >= 0 ? 'inside' : 'outside'} the target</span>
          </dd>
        </div>
        {MODELS.map((m) => {
          const p = r[m.key]
          if (p === null) return <div key={m.key}><dt>Predicted, {SHORT[m.key]}</dt><dd>–</dd></div>
          const hw = halves[m.key]
          const e = isNum(hw) ? edgeOf(ask.dir, p, hw) : null
          const useEdge = ask.cautious && e !== null
          const picks = meets(ask.dir, useEdge ? e : p, ask.target)
          return (
            <div key={m.key}>
              <dt>Predicted, {SHORT[m.key]}{m.key === ask.model ? ' (in use)' : ''}</dt>
              <dd>
                <span className={useEdge ? undefined : 'strong'}>{val(p)}</span>
                {e !== null && <span className="pf-sub">cautious edge <span className={useEdge ? 'strong' : undefined}>{fmt(e)}</span></span>}
                <span className="pf-sub">{picks ? 'picks it' : 'leaves it out'}: {picks === item.really ? 'right' : 'wrong'}</span>
              </dd>
            </div>
          )
        })}
      </dl>
      {mates.length > 0 && (
        <p className="fine">
          {mates.length === 1 ? 'The other held-out row' : `The other ${count(mates.length)} held-out rows`} of its group
          {prep.sampled ? ' among the rows on the site' : ''} ({truthWord} {verb(mates.length, 'value', 'values')}):{' '}
          {mates.slice(0, 8).map((m) => fmt(m.y)).join(', ')}{mates.length > 8 ? ` and ${count(mates.length - 8)} more` : ''}.
          {prep.sampled ? ' The site carries a sample of the held-out rows, so the group can have more rows than these.' : ''} A group is held
          out together, so none of its rows trained the models that predicted this row.
        </p>
      )}
      <p className="fine">Values in the task&apos;s unit. The value judged against the target is in bold.</p>
    </div>
  )
}

function Finder({ task, data, prep, rehearsal }) {
  const uid = safeId(useId())
  const recommended = MODELS.some((m) => m.key === task.recommended) ? task.recommended : null
  const [entry, setEntry] = useState(null) // null until the visitor types or picks an example: the top-quarter example is shown
  const [dir, setDir] = useState('min')
  const [model, setModel] = useState(recommended ?? 'blend')
  const [cautious, setCautious] = useState(true)
  const [sel, setSel] = useState(null) // index of the selected row in the rows file; null shows the top pick
  const [list, setList] = useState('hits')
  const [query, setQuery] = useState('')
  const [more, setMore] = useState({ key: '', n: PAGE })
  const plotRef = useRef(null)

  const unit = data.unit ?? task.unit ?? ''
  const isLog = /log10/i.test(unit)
  const truthWord = /DFT/i.test(unit) ? 'DFT-calculated' : 'measured'
  const words = WORDS[task.name] ?? GENERIC
  const short = shortUnit(unit, words.unit)
  const plain = isLog ? (words.plain?.when.test(unit) ? words.plain : PLAIN) : null
  const withUnit = (text) => `${text}${short ? ` ${short}` : ''}`
  // A value from the data, rounded like every number on the site; and the value behind a log10 one.
  const val = (v) => (isNum(v) ? withUnit(fmt(v)) : '–')
  const plainOf = (v, about = '') => (plain && isNum(v) ? `${plain.label} ${about}${tenTo(v)}${plain.unit ? ` ${plain.unit}` : ''}` : '')
  // A target is written with every digit, so the page never states a different target from the one it judges by. One typed as the
  // plain value is a log10 with endless digits: it is written to 12 significant figures, beside the plain value exactly as typed.
  const targetText = (a) => withUnit(plainNum(a.plainValue === null ? a.target : Number(a.target.toPrecision(12))))
  const targetPlain = (a) => (a.plainValue === null ? plainOf(a.target, 'about ') : `${plain.label} ${plainNum(a.plainValue)}${plain.unit ? ` ${plain.unit}` : ''}`)
  const ranges = data.interval90 ?? task.interval90
  const halves = useMemo(() => ({ network: num(ranges?.network?.half_width), trees: num(ranges?.trees?.half_width), blend: num(ranges?.blend?.half_width) }), [ranges])
  const half = halves[model]
  const canCaution = isNum(half)

  // The target: typed in the task's unit, or (for a log10 target) as the value behind the log.
  const base = entry ?? { mode: 'unit', text: prep.q3 === null ? '' : String(prep.q3) }
  const parsed = parseNumber(base.text)
  let target = null
  let error = ''
  if (parsed.error === 'ambiguous') {
    error = `“${parsed.typed}” can be read as ${parsed.readings[0]} or as ${parsed.readings[1]}. Type it as ${parsed.readings[0]} or as ${parsed.readings[1]}.`
  } else if (base.mode === 'plain') {
    const log = parsed.value !== null && parsed.value > 0 ? Math.log10(parsed.value) : null
    if (parsed.error || (parsed.value !== null && !isNum(log))) {
      error = `${parsed.error === 'big' ? 'That number is too large to use. ' : ''}Type the ${plain?.label ?? 'value'} as a plain number above zero${isNum(prep.median) ? `, for example ${fmt(10 ** prep.median, 3)}` : ''}, or leave it blank.`
    } else target = log
  } else if (parsed.error) {
    error = `${parsed.error === 'big' ? 'That number is too large to use. ' : ''}Type the target as a plain number${short ? ` in ${short}` : ' in the task’s unit'}${isNum(prep.median) ? `, for example ${fmt(prep.median)}` : ''}, or leave it blank.`
  } else target = parsed.value
  // The field that was not typed in mirrors the target in a form that can be typed back.
  const plainValue = base.mode === 'plain' && target !== null ? parsed.value : null
  const mirrored = target === null ? null : 10 ** target
  const unitText = base.mode === 'unit' ? base.text : target === null ? '' : String(Number(target.toPrecision(12)))
  const plainText = base.mode === 'plain' ? base.text : isNum(mirrored) && mirrored > 0 ? fmt(mirrored, 4) : ''

  const ask = useMemo(
    () => (target === null ? null : { target, plainValue, dir, model, cautious: cautious && canCaution }),
    [target, plainValue, dir, model, cautious, canCaution],
  )
  const dAsk = useDeferredValue(ask)
  const found = useMemo(() => (dAsk ? find(prep, dAsk, halves[dAsk.model]) : null), [prep, dAsk, halves])
  const grid = useMemo(
    () => (dAsk ? MODELS.flatMap((m) => [true, false].map((c) => tally(prep, dAsk, m.key, c, halves[m.key]))) : null),
    [prep, dAsk, halves],
  )
  const dq = useDeferredValue(query)
  const listed = useMemo(() => (found ? matchItems(found[list], dq, task.name) : []), [found, list, dq, task.name])
  // A row's place in the whole list, which a search does not change.
  const rank = useMemo(() => new Map((found ? found[list] : []).map((it, k) => [it.r.i, k + 1])), [found, list])

  const setTarget = (text, mode = 'unit') => setEntry({ mode, text })
  const examples = [
    { label: `Top quarter of the ${truthWord} values: at least`, dir: 'min', v: prep.q3 },
    { label: 'Upper half: at least', dir: 'min', v: prep.median },
    { label: 'Bottom quarter: at most', dir: 'max', v: prep.q1 },
  ].filter((ex) => isNum(ex.v)).map((ex) => ({ ...ex, label: `${ex.label} ${withUnit(plainNum(ex.v))}${plain ? ` (${plainOf(ex.v, 'about ')})` : ''}` }))
  const show = (i) => {
    setSel(i)
    plotRef.current?.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'nearest' })
  }

  const things = words.things
  const title = task.title ?? task.name ?? task.id
  const total = data.rows_total ?? task.rows
  const shownOf = `${count(data.rows_shown ?? data.rows.length)} of the ${count(total)} held-out rows`

  // Everything below describes the deferred request, so the words and the lists always agree.
  const A = dAsk
  const judgedWord = A?.cautious ? 'cautious edge' : 'prediction'
  const coverage = A ? ranges?.[A.model]?.coverage : null
  const selItem = found ? (sel !== null && found.byRow.get(sel)) || found.hits[0] || null : null
  const selHeading = !selItem ? '' : sel !== null && found.byRow.has(sel) ? 'Selected held-out row' : 'Top pick (select another row to see it here)'
  const listKey = A ? `${A.target}|${A.dir}|${A.model}|${A.cautious}|${list}|${dq}` : ''
  const shownN = more.key === listKey ? more.n : PAGE
  const rows = listed.slice(0, shownN)
  const N = found?.hits.length ?? 0
  const K = found?.right ?? 0
  const R = found?.real ?? 0
  const M = found?.n ?? 0
  const quads = {
    tp: { n: K, text: `picked, ${truthWord} value meets the target` },
    fp: { n: N - K, text: `picked, ${truthWord} value does not meet the target` },
    fn: { n: R - K, text: 'left out, though the value meets the target' },
    tn: { n: M - N - (R - K), text: 'left out, value does not meet the target' },
  }
  const quadOrder = A?.dir === 'max' ? ['fn', 'tn', 'tp', 'fp'] : ['fp', 'tp', 'tn', 'fn']
  const corner = ['Upper left', 'Upper right', 'Lower left', 'Lower right']
  const picks = found ? [
    { id: 'top', label: `Top pick: furthest inside the target on the ${judgedWord}`, it: found.hits[0] ?? null, none: 'None: no row is picked.' },
    { id: 'wrong', label: `Pick whose ${truthWord} value is furthest outside the target`, it: found.wrong[0] ?? null, none: 'None among the picks.' },
    { id: 'missed', label: `Best row the finder left out, by its ${truthWord} value`, it: found.missed[0] ?? null, none: 'None among the rows left out.' },
  ] : []
  const listWords = {
    hits: { chip: `Picked (${count(N)})`, head: `Picked rows, furthest inside the target on the ${judgedWord} first` },
    wrong: { chip: `Picked, but do not meet it (${count(N - K)})`, head: `Picks whose ${truthWord} value does not meet the target, furthest outside first` },
    missed: { chip: `Meet it, but left out (${count(R - K)})`, head: 'Rows that meet the target but were left out, furthest inside first' },
  }
  const tips = ['Relax the target']
  if (A?.cautious) tips.push('untick Cautious')
  tips.push('choose another model')
  const tipText = `${tips.slice(0, -1).join(', ')} or ${tips[tips.length - 1]}`
  const rangeText = `${plainNum(prep.min)} to ${withUnit(plainNum(prep.max))}`

  return (
    <>
      <section className="card fam-section pf">
        <h2>{title}: find rows that meet a target</h2>
        <p className="note">
          Say what the {truthWord} value must be. The page picks the held-out {things} that the chosen model predicts will meet it, then
          checks every pick against its {truthWord} value. Each of these rows was predicted by models that never trained on it, so the check
          shows how the finder does on rows it has not seen. Target: {unit || 'unit not stated'}.
        </p>
        <form onSubmit={(e) => e.preventDefault()}>
          <div className={plain ? 'pf-form pf-form4' : 'pf-form'}>
            <div>
              <label htmlFor={`${uid}dir`}>The value must be</label>
              <select id={`${uid}dir`} value={dir} onChange={(e) => setDir(e.target.value)}>
                <option value="min">at least the target</option>
                <option value="max">at most the target</option>
              </select>
              <span className="pf-hint">{dir === 'min' ? 'higher values qualify' : 'lower values qualify'}</span>
            </div>
            <div>
              <label htmlFor={`${uid}target`}>Target{short ? ` (${short})` : isLog ? ' (log10, the task’s unit)' : ''}</label>
              <input id={`${uid}target`} value={unitText} onChange={(e) => setTarget(e.target.value)} inputMode="text" autoComplete="off" spellCheck={false}
                placeholder={isNum(prep.median) ? `e.g. ${fmt(prep.median)}` : 'a number'} />
              <span className="pf-hint">
                {isNum(prep.min) ? `${truthWord} values in these rows run from ${plainNum(prep.min)} to ${plainNum(prep.max)}` : 'in the task’s unit'}
              </span>
            </div>
            {plain && (
              <div>
                <label htmlFor={`${uid}plain`}>Or as the {plain.label}{plain.unit ? ` (${plain.unit})` : ''}</label>
                <input id={`${uid}plain`} value={plainText} onChange={(e) => setTarget(e.target.value, 'plain')} inputMode="text" autoComplete="off" spellCheck={false}
                  placeholder={isNum(prep.median) ? `e.g. ${fmt(10 ** prep.median, 3)}` : 'a number above zero'} />
                <span className="pf-hint">10 to the power of the target, rounded unless typed here; 1e-3 is accepted</span>
              </div>
            )}
            <div>
              <label htmlFor={`${uid}model`}>Model</label>
              <select id={`${uid}model`} value={model} onChange={(e) => setModel(e.target.value)}>
                {MODELS.map((m) => <option key={m.key} value={m.key}>{m.label}{m.key === recommended ? ' · recommended' : ''}</option>)}
              </select>
              <span className="pf-hint">
                {!recommended ? 'the export names no recommended model here; the blend is the default'
                  : rehearsal ? `this rehearsal export marks the ${SHORT[recommended]} as recommended, which means nothing`
                    : `the nightly run recommends the ${SHORT[recommended]} for this task`}
              </span>
            </div>
          </div>
          {examples.length > 0 && (
            <div className="examples">
              <span>Try:</span>
              {examples.map((ex) => (
                <button type="button" key={ex.label} className="chip" aria-pressed={dir === ex.dir && base.mode === 'unit' && base.text === String(ex.v)}
                  onClick={() => { setDir(ex.dir); setTarget(String(ex.v)) }}>
                  {ex.label}
                </button>
              ))}
            </div>
          )}
          <div className="options">
            <label>
              <input type="checkbox" checked={cautious && canCaution} disabled={!canCaution} onChange={(e) => setCautious(e.target.checked)} />
              <span>
                Cautious: the edge of the prediction&apos;s 90% range must meet the target
                {canCaution
                  ? ` (prediction ${sign(dir)}\u00a0${fmt(half)} for the ${SHORT[model]}${isLog ? `; in plain units that range is ${factor(half)} either way` : ''})`
                  : ' (this export has no 90% range for this model, so the prediction itself is judged)'}
              </span>
            </label>
          </div>
          <p className="note small">
            The list updates as you type. The examples are quartiles of the {truthWord} values in these held-out rows, not requirements from a
            standard. {cap(shownOf)} are on the site{data.sample_rule ? ` (${data.sample_rule})` : ''}.{data.split ? ` Split: ${data.split}.` : ''}
            {prep.skipped > 0 && ` ${count(prep.skipped)} ${verb(prep.skipped, 'row has', 'rows have')} no value in the data and ${verb(prep.skipped, 'is', 'are')} left out.`}
          </p>
        </form>
        {error && <p className="error" role="alert">{error}</p>}
      </section>

      {!error && !ask && (
        <section className="card fam-section pf">
          <p className="note">
            {parsed.pending ? 'The target is not a complete number yet. The rows appear once it is.'
              : 'Set a target above, or pick an example. Without one there is nothing to pick the rows by.'}
          </p>
        </section>
      )}

      {found && A && (
        <section className="card fam-section pf" aria-label={`${title}: rows predicted to meet the target`}>
          <h2 aria-live="polite">
            {rehearsal ? `Held-out ${things} predicted to meet the target` : `${count(N)} of ${count(M)} held-out ${things} are predicted to meet the target`}
          </h2>
          <p className="note">
            Target: {truthWord} value {dirWord(A.dir)} {targetText(A)}{plain ? ` (${targetPlain(A)})` : ''}, judged on the{' '}
            {A.cautious ? `cautious edge (prediction ${sign(A.dir)}\u00a0${fmt(halves[A.model])})` : 'prediction'} of the {SHORT[A.model]}.
            {!rehearsal && isNum(coverage) && isNum(halves[A.model])
              && ` The export reports that this model's 90% range of ±${fmt(halves[A.model])} held for ${fmt(coverage * 100, 3)}% of all ${count(total)} held-out rows.`}
          </p>
          {rehearsal && (
            <p className="pf-flagline">
              <span className="flag low">Rehearsal</span> The code ran on a tiny test run. No result is stated here: the figures below show the
              format only and mean nothing.
            </p>
          )}
          {task.exploratory && <p className="pf-flagline"><span className="flag low">Exploratory</span> {task.exploratory}.</p>}

          <div className="pf-tiles">
            <div className="pf-tile">
              <span className="label">Predicted to meet the target (the picks)</span>
              <span className="big">{count(N)}</span>
              <span className="label">of {count(M)} held-out rows</span>
            </div>
            <div className="pf-tile">
              <span className="label">Picks whose {truthWord} value really meets it</span>
              <span className="big">{count(K)}</span>
              <span className="label">{N ? `${share(K, N)} of the picks (precision)` : 'no picks to check'}</span>
            </div>
            <div className="pf-tile">
              <span className="label">Rows that really meet it, found by the picks</span>
              <span className="big">{count(K)} <small>of {count(R)}</small></span>
              <span className="label">{R ? `${share(K, R)} found` : 'no row really meets it'}</span>
            </div>
            <div className="pf-tile">
              <span className="label">A row taken at random really meets it</span>
              <span className="big">{share(R, M)}</span>
              <span className="label">{count(R)} of {count(M)} rows, with no model</span>
            </div>
          </div>

          {!rehearsal && N > 0 && (
            <p className="note">
              Of the {count(N)} {verb(N, 'row', 'rows')} picked, <strong>{count(K)}</strong> really {verb(K, 'meets', 'meet')} the target by{' '}
              {verb(K, 'its', 'their')} {truthWord} value ({share(K, N)}); a row taken at random from these {count(M)} does so {share(R, M)} of the
              time. The picks contain {count(K)} of the {count(R)} rows that really meet it ({share(K, R)}). These are counts on one fixed sample of
              held-out rows for one target; no interval is computed for them.{task.exploratory ? ' Exploratory.' : ''}
            </p>
          )}
          {N === 0 && (rehearsal ? (
            <p className="note">No row is picked for this target. {tipText}.</p>
          ) : (
            <p className="note">
              No held-out row is predicted to meet this target on the {judgedWord} of the {SHORT[A.model]}. The {truthWord} values in these rows run
              from {rangeText}, and {count(R)} of them really {verb(R, 'meets', 'meet')} it. {R === 0 ? 'Relax the target' : tipText}.
            </p>
          ))}
          {!A.cautious && N > 0 && (
            <p className="warn">
              {isNum(halves[A.model]) ? 'Cautious is off. These rows meet the target on the model’s expected value only; at the edge of the 90% range they may not.'
                : 'This export has no 90% range for this model, so these rows are judged on the model’s expected value only.'}
              {' '}Read the list as rows worth checking, not as rows that meet the target.
            </p>
          )}

          <div className="pf-split" ref={plotRef}>
            <figure className="pf-figure">
              <Quadrants found={found} ask={A} truthWord={truthWord} judgedWord={judgedWord} modelWord={SHORT[A.model]} selected={selItem} onSelect={setSel} />
              <ul className="pf-quads">
                {quadOrder.map((k, n) => (
                  <li key={k}><span className="label">{corner[n]}</span><strong>{count(quads[k].n)}</strong> {quads[k].text}</li>
                ))}
              </ul>
              <figcaption>
                Each dot is a held-out row: its {truthWord} value across, the {judgedWord} of the {SHORT[A.model]} up. The dashed lines mark the
                target. Rows {A.dir === 'min' ? 'above' : 'below'} the horizontal line are picked (shaded); rows {A.dir === 'min' ? 'right' : 'left'} of
                the vertical line really meet the target. Solid dots are picks that meet it, rings are picks that do not, faint dots were left
                out. On the dotted diagonal the {judgedWord} equals the {truthWord} value. Axes in the task&apos;s unit{unit ? ` (${unit})` : ''}.
              </figcaption>
            </figure>
            <div className="pf-side">
              <div className="picks pf-picks">
                {picks.map((p) => (
                  <button type="button" key={p.id} className="pick" disabled={!p.it} aria-pressed={!!p.it && selItem === p.it} onClick={() => p.it && setSel(p.it.r.i)}>
                    <span className="label">{p.label}</span>
                    {p.it ? (
                      <>
                        <span className="mixname"><RowName r={p.it.r} taskName={task.name} bare /></span>
                        <span className="big">{val(p.it.r.y)} <small>{truthWord}</small></span>
                        <span className="label">predicted {val(p.it.p)}{p.it.edge !== null ? `, cautious edge ${fmt(p.it.edge)}` : ''}</span>
                      </>
                    ) : <span className="label">{p.none}</span>}
                  </button>
                ))}
              </div>
              <Detail item={selItem} heading={selHeading} prep={prep} ask={A} halves={halves} taskName={task.name} truthWord={truthWord}
                val={val} plainOf={plainOf} showPlain={!!plain} />
            </div>
          </div>

          <h3>{listWords[list].head}</h3>
          <div className="pf-toggle" role="group" aria-label="Rows listed in the table">
            {LISTS.map((k) => (
              <button type="button" key={k} className="chip" aria-pressed={list === k} onClick={() => setList(k)}>{listWords[k].chip}</button>
            ))}
          </div>
          <div className="pf-search">
            <label htmlFor={`${uid}q`}>Search these rows by label</label>
            <input id={`${uid}q`} type="search" value={query} autoComplete="off" spellCheck={false} placeholder={words.filter}
              onChange={(e) => setQuery(e.target.value)} />
          </div>
          {listed.length === 0 ? (
            <p className="note">
              {found[list].length === 0
                ? (list === 'hits' ? 'No row is picked for this target.' : list === 'wrong' ? 'No picked row is in this list.' : 'No row that meets the target was left out.')
                : `No row in this list matches “${dq.trim()}”.`}
            </p>
          ) : (
            <>
              <p className="note small pf-swipe">Swipe the table sideways for the {truthWord} value and the check against the target.</p>
              <div className="tablewrap sim-scroll">
                <table className="ranked pf-table">
                  <thead>
                    <tr>
                      <th className="num">#</th>
                      <th>{words.rowHead}</th>
                      <th className="num">Predicted{short ? ` (${short})` : ''}</th>
                      <th className="num">Cautious edge</th>
                      <th className="num">{cap(truthWord)}{short ? ` (${short})` : ''}</th>
                      <th>Checked against the target</th>
                      <th className="num">Miss (predicted − {truthWord})</th>
                      <th><span className="sr">Show on the plot</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((it) => {
                      const miss = it.p - it.r.y
                      const hw = halves[A.model]
                      return (
                        <tr key={it.r.i} className={selItem === it ? 'on' : undefined}>
                          <td className="num">{rank.get(it.r.i)}</td>
                          <td className="pf-labelcell"><RowName r={it.r} taskName={task.name} prep={prep} /></td>
                          <td className={found.useEdge ? 'num' : 'num strong'}>{fmt(it.p)}</td>
                          <td className={found.useEdge ? 'num strong' : 'num'}>{it.edge === null ? '–' : fmt(it.edge)}</td>
                          <td className="num">
                            <span className="strong">{fmt(it.r.y)}</span>
                            {plain && <span className="label">{plainOf(it.r.y)}</span>}
                          </td>
                          <td>
                            <span className={it.really ? 'flag ok' : 'flag low'}>{it.really ? 'Meets it' : 'Does not meet it'}</span>
                            <span className="label">{plainNum(Math.abs(it.trueRoom), 4)} {it.trueRoom >= 0 ? 'inside' : 'outside'}</span>
                          </td>
                          <td className="num">
                            {signed(miss)}
                            {isNum(hw) && <span className="label">{Math.abs(miss) <= hw ? 'inside' : 'outside'} its 90% range</span>}
                          </td>
                          <td>
                            <button type="button" className="chip" aria-pressed={selItem === it} onClick={() => show(it.r.i)}
                              aria-label={`Show ${describeRow(it.r, task.name).title} on the plot`}>Show</button>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
              <div className="pf-more">
                <span className="note small">
                  Showing {count(rows.length)} of {count(listed.length)}{dq.trim() ? ` that match “${dq.trim()}”. The # column is each row’s place in the whole list` : ''}.
                </span>
                {rows.length < listed.length && (
                  <>
                    <button type="button" className="chip" onClick={() => setMore({ key: listKey, n: shownN + PAGE })}>
                      Show {Math.min(PAGE, listed.length - rows.length)} more
                    </button>
                    {listed.length <= 500 && (
                      <button type="button" className="chip" onClick={() => setMore({ key: listKey, n: listed.length })}>Show all {count(listed.length)}</button>
                    )}
                  </>
                )}
              </div>
            </>
          )}
          <dl className="readme">
            <div>
              <dt>Predicted</dt>
              <dd>
                The {SHORT[A.model]}&apos;s prediction for the row, made by models that never trained on it (out of fold).
                {A.model === 'network' && ' It is an average of several networks, not one network.'}
                {A.model === 'blend' && ' It is half the network average and half the strongest tree model.'}
              </dd>
            </div>
            <div>
              <dt>Cautious edge</dt>
              <dd>
                {isNum(halves[A.model])
                  ? `The prediction moved by its 90% range towards failing the target: prediction ${sign(A.dir)}\u00a0${fmt(halves[A.model])}${isLog ? ` (in plain units, ${factor(halves[A.model])} either way)` : ''}.`
                  : 'Not available: this export has no 90% range for this model.'}
                {A.cautious ? ' While Cautious is ticked it is the value judged (in bold).' : ' Cautious is off, so the prediction itself is judged (in bold).'}
              </dd>
            </div>
            <div>
              <dt>{cap(truthWord)}</dt>
              <dd>
                {truthWord === 'DFT-calculated'
                  ? 'The value in the data, calculated with DFT; it is not a laboratory measurement. The check is against that calculation.'
                  : 'The measured value in the data. The check is against it.'}
                {plain && ` Small text gives the ${plain.label} (10 to the power of the value).`}
              </dd>
            </div>
            <div>
              <dt>Checked against the target</dt>
              <dd>
                Whether the {truthWord} value itself is {dirWord(A.dir)} the target, and by how much it is inside or outside it. A pick whose value
                does not meet the target is a wrong pick.
              </dd>
            </div>
            <div>
              <dt>Row labels</dt>
              <dd>
                {task.name === 'alloys' ? 'An alloy is named by its composition in atomic fractions; several rows can share one composition.'
                  : task.name === 'chemistry' ? 'The export names a compound by its AqSolDB id and the value of AqSolDB’s Group column; it does not carry the compound’s name or structure.'
                    : 'The export carries each row’s group id, not a name or a structure. Rows of one group were held out together.'}
              </dd>
            </div>
          </dl>
        </section>
      )}

      {grid && A && (
        <section className="card fam-section pf" aria-label={`${title}: the same target with every model and setting`}>
          <h2>The same target with every model and setting</h2>
          <p className="note">
            How many rows each of the three models would pick for {dirWord(A.dir)} {targetText(A)}, judged at the cautious edge and on the prediction
            itself, and how many of those picks really meet the target by their {truthWord} value.
          </p>
          {rehearsal && (
            <p className="pf-flagline"><span className="flag low">Rehearsal</span> The figures in this table show the format only and mean nothing.</p>
          )}
          {task.exploratory && <p className="pf-flagline"><span className="flag low">Exploratory</span> {task.exploratory}.</p>}
          <p className="note small pf-swipe">Swipe the table sideways for the counts.</p>
          <div className="tablewrap sim-scroll">
            <table className="compact pf-grid">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Judged on</th>
                  <th className="num">Picked</th>
                  <th className="num">Picks that really meet it</th>
                  <th className="num">Share of picks right (precision)</th>
                  <th className="num">Found, of the {count(R)} that meet it</th>
                  <th><span className="sr">Use this model and setting</span></th>
                </tr>
              </thead>
              <tbody>
                {grid.map((g) => {
                  const on = g.model === A.model && g.cautious === A.cautious
                  return (
                    <tr key={`${g.model}${g.cautious}`} className={on ? 'on' : undefined}>
                      <td className="pf-modelcell">{MODELS.find((m) => m.key === g.model)?.label}{g.model === recommended ? ' (recommended)' : ''}</td>
                      <td>{g.cautious ? (g.usable ? `cautious edge, ${sign(A.dir)}\u00a0${fmt(halves[g.model])}` : 'cautious edge: no 90% range') : 'prediction'}</td>
                      <td className="num">{g.usable ? count(g.picked) : '–'}</td>
                      <td className="num">{g.usable ? count(g.right) : '–'}</td>
                      <td className="num strong">{g.usable && g.picked ? share(g.right, g.picked) : '–'}</td>
                      <td className="num">{g.usable && g.real ? share(g.right, g.real) : '–'}</td>
                      <td>
                        {g.usable && (
                          <button type="button" className="chip" aria-pressed={on} onClick={() => { setModel(g.model); setCautious(g.cautious) }}
                            aria-label={`Use the ${SHORT[g.model]}, judged on the ${g.cautious ? 'cautious edge' : 'prediction'}`}>{on ? 'In use' : 'Use'}</button>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <ul className="pf-list">
            <li>
              These are counts on {shownOf} for one target, with no interval behind them. A small gap between two lines of the table is not evidence
              that one model finds rows better; the pipeline&apos;s own comparison of the models, with intervals, is in the section below.
            </li>
            <li>
              Judging at the cautious edge picks fewer rows and leaves more good ones out; judging on the prediction picks more rows, and more of
              them turn out not to meet it. Which trade is right depends on what a wasted experiment costs against a missed material.
            </li>
            <li>
              Every row here already has a {truthWord} value: the finder proposes nothing new. The table shows how well each model would have
              picked among rows like these; a material unlike the data can do worse.
            </li>
            {truthWord === 'DFT-calculated' && (
              <li>The values checked against are DFT-calculated, not measured, so a right pick agrees with the calculation, not with an experiment.</li>
            )}
            {isLog && isNum(halves[A.model]) && (
              <li>The target is a log10, so the 90% range of ±{fmt(halves[A.model])} is {factor(halves[A.model])} either way in plain units.</li>
            )}
          </ul>
        </section>
      )}
    </>
  )
}

// Fetches the task's rows file once this part of the page is near the screen (PipelineResults below asks for the same file, and
// loadPipeline keeps one copy), then hands the prepared rows to the finder.
function Loader({ task, rehearsal }) {
  const [want, setWant] = useState(typeof IntersectionObserver === 'undefined')
  const [data, setData] = useState(undefined) // undefined: not fetched yet; null: missing or unreadable; object: the rows file
  const lazyRef = useRef(null)
  const file = task.rows_file
  useEffect(() => {
    const el = lazyRef.current
    if (want || !el || typeof IntersectionObserver === 'undefined') return undefined
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) setWant(true) }, { rootMargin: '400px 0px' })
    io.observe(el)
    return () => io.disconnect()
  }, [want])
  useEffect(() => {
    if (!want) return undefined
    let live = true
    loadPipeline(file).then((d) => { if (live) setData(d && Array.isArray(d.rows) ? d : null) }, () => { if (live) setData(null) })
    return () => { live = false }
  }, [want, file])
  const prep = useMemo(() => {
    if (!data) return null
    try {
      const p = prepare(data)
      return p.rows.length ? p : null
    } catch {
      return null
    }
  }, [data])
  const title = task.title ?? task.name ?? task.id
  if (data && prep) return <Finder task={task} data={data} prep={prep} rehearsal={rehearsal || !!data.rehearsal} />
  return (
    <section className="card fam-section pf" ref={lazyRef}>
      <h2>{title}: find rows that meet a target</h2>
      {data === undefined ? (
        <p className="note">Loading the held-out rows…</p>
      ) : (
        <p className="note">
          {data === null ? `The rows file (${file}) is not in this export or cannot be read` : `The rows file (${file}) holds no row with a value`}, so
          the finder cannot run. On the training laptop, after a nightly run: <code>python -m gpu.export_site</code>, then commit{' '}
          <code>public/pipeline/</code>.
        </p>
      )}
    </section>
  )
}

export default function PropertyFinder({ family, index: given, tasks: givenTasks = [] }) {
  const uid = safeId(useId())
  // The page hands over index.json once it has it. Its own request can be cancelled before it finishes (React's development double
  // mount, or leaving a page while the file is on its way), and then nothing would ever arrive. So the finder also asks for the
  // file, without a signal and once more if that request was cancelled by another caller. loadPipeline keeps one copy.
  const [own, setOwn] = useState(undefined)
  useEffect(() => {
    let live = true
    const set = (ix) => { if (live) setOwn(ix ?? null) }
    loadPipeline('index.json').then(set, () => loadPipeline('index.json').then(set, () => set(null)))
    return () => { live = false }
  }, [])
  const index = given !== undefined ? given : own
  const tasks = given !== undefined || !family ? givenTasks : (own?.tasks ?? []).filter((t) => t?.site === family)
  const usable = tasks.filter((t) => t && t.status === 'ok' && t.kind !== 'crystal' && t.rows_file)
  const [taskId, setTaskId] = useState(null)
  const task = usable.find((t) => t.id === taskId) ?? usable[0] ?? null
  if (!task) {
    // While index.json is loading there are no tasks yet.
    if (index === undefined && tasks.length === 0) return <section className="card fam-section pf"><p className="note">Loading the pipeline&apos;s results…</p></section>
    return (
      <section className="card fam-section pf">
        <h2>Find rows that meet a target</h2>
        {tasks.length === 0 ? (
          <p className="note">
            {index === null ? 'The pipeline’s results have not been exported to the site yet' : 'This export holds no task for this family'}, so the
            finder has no held-out rows to search.
          </p>
        ) : (
          <>
            <p className="note">The export lists this family&apos;s {verb(tasks.length, 'task', 'tasks')} without rows the finder can search:</p>
            <ul className="pf-list">
              {tasks.map((t, k) => (
                <li key={t?.id ?? k}>
                  {t?.title ?? t?.name ?? t?.id ?? 'Unnamed task'}: {t?.status !== 'ok' ? `not in this export (${t?.status ?? 'unknown status'}${t?.reason ? `: ${t.reason}` : ''})`
                    : t.kind === 'crystal' ? 'a crystal task, shown on the crystals page' : 'its rows are not published on the site'}.
                </li>
              ))}
            </ul>
          </>
        )}
      </section>
    )
  }
  return (
    <>
      {usable.length > 1 && (
        <section className="card fam-section pf">
          <div className="pf-task">
            <label htmlFor={`${uid}task`}>Property to search by</label>
            <select id={`${uid}task`} value={task.id} onChange={(e) => setTaskId(e.target.value)}>
              {usable.map((t) => <option key={t.id} value={t.id}>{t.title ?? t.name ?? t.id}</option>)}
            </select>
          </div>
        </section>
      )}
      <Loader key={task.id} task={task} rehearsal={!!index?.rehearsal} />
    </>
  )
}
