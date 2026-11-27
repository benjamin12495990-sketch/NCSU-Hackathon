import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { loadPipeline, fmt, signed, interval, comparisonText, factor, MODELS } from '../../lib/pipeline.js'
import '../family.css'
import './sim.css'
import './SimDashboard.css'

// "All simulations: nightly results" (see README.md in this folder). One row for every task in public/pipeline/index.json: how the
// network average, the strongest tree model and their blend did on rows they never trained on, the 90% range of the model judged
// (the recommended one unless the visitor picks another), the network-against-trees verdict in the project's wording, when it was
// trained and whether it is exploratory. It works like the concrete designer: the visitor says how good a model must be (its miss
// as a share of the miss of predicting the average every time), narrows by family or verdict, and gets the tasks that meet it in a
// chosen order; selecting one shows its errors and its three paired comparisons drawn against zero. The family link opens that page.
// Every number comes from index.json (gpu/export_site.py); the share is the ratio of two of its errors and the page says so. The
// presets below are only starting points for the visitor's own requirement.
// Wording (docs/AGENTS.md): a rehearsal export gets its numbers but no result sentence (no verdict, no one-network count, no verdict
// filter); an exploratory task says so with the export's own reason; a cross-validated 90% half-width is an AVERAGE of per-fold widths.

const NAME = { network: 'Network average', trees: 'Strongest trees', blend: 'Blend' }
const SHORT = { network: 'network average', trees: 'strongest trees', blend: 'blend' }
const PAIRS = [
  ['network_vs_trees', 'Network average against strongest trees'],
  ['blend_vs_trees', 'Blend against strongest trees'],
  ['blend_vs_network', 'Blend against network average'],
]
const JUDGES = [
  { key: 'recommended', label: 'The recommended model', word: 'the recommended model' },
  { key: 'network', label: NAME.network, word: 'the network average' },
  { key: 'trees', label: NAME.trees, word: 'the strongest trees' },
  { key: 'blend', label: NAME.blend, word: 'the blend' },
]
const SHARE_PRESETS = [75, 50, 25]
const ORDERS = [
  { key: 'family', label: 'By family' },
  { key: 'best', label: 'Smallest share of the no-model miss first' },
  { key: 'worst', label: 'Largest share of the no-model miss first' },
  { key: 'trained', label: 'Most recently trained first' },
]
const VERDICTS = [
  { key: 'any', label: 'Any verdict' },
  { key: 'trees', label: 'Strongest trees clearly better' },
  { key: 'network', label: 'Network average clearly better' },
  { key: 'none', label: 'No clear difference' },
]

const isNum = (v) => typeof v === 'number' && Number.isFinite(v)
const text = (v) => (typeof v === 'string' ? v.trim() : isNum(v) ? String(v) : '')
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s)
const pct = (c) => (isNum(c) ? `${fmt(c * 100, 3)}%` : '–')
const count = (n) => (isNum(n) ? n.toLocaleString() : '–')
const plural = (n, one, many) => (n === 1 ? one : many)
const noDot = (s) => s.replace(/[.\s]+$/, '') // the page adds the full stop after an export's reason
const modelKey = (rec) => (typeof rec !== 'string' ? null : rec.startsWith('network') ? 'network' : rec === 'trees' || rec === 'blend' ? rec : null)
const reducedMotion = () => typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

// The export's comparison in the project's wording (comparisonText), naming the network as an average. Null when it is absent.
// A rehearsal gets the difference and its interval but no verdict, in the words PipelineResults uses for the same task.
// The spaces inside the bracket are made non-breaking so an interval is never split across two lines.
function compare(c, rehearsal) {
  if (!c) return null
  const s = rehearsal ? `Rehearsal: no verdict. Difference ${signed(c.diff)} ${interval(c.ci95)}, which means nothing.`
    : cap(comparisonText({ ...c, better: c.better ? NAME[c.better] ?? c.better : null }))
  return s.replace(/\[[^\]]*\]/, (m) => m.replace(/ /g, '\u00a0'))
}

// "2026-11-27 07:01:23" as a date and a time; anything else is shown as it is.
function stamp(s) {
  const m = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})/.exec(s ?? '')
  return m ? { date: m[1], time: m[2] } : { date: s || 'not stated', time: '' }
}

const splitText = (t) => (t.mode === 'cv' ? 'cross-validated by group (each row predicted by models trained without its group)'
  : t.kind === 'crystal' ? 'test crystals from chemical systems kept out of training'
    : t.mode === 'fixed' ? 'one fixed train and test split (not cross-validated)' : text(t.mode))

// Everything that depends only on the file: one record per task, the families present, and the counts the page states.
function prepare(index, domains) {
  const doms = Array.isArray(domains) ? domains : []
  const list = Array.isArray(index?.tasks) ? index.tasks.filter((t) => t && typeof t === 'object') : []
  const newest = list.reduce((m, t) => (text(t.trained) > m ? text(t.trained) : m), '')
  const rows = list.map((t, i) => {
    const di = doms.findIndex((d) => d.id === t.site)
    const dom = doms[di]
    const title = text(t.title) || text(t.name) || text(t.id) || `Task ${i + 1}`
    const family = dom?.label ?? (text(t.site) || 'no family given')
    const unit = text(t.unit)
    const crystal = t.kind === 'crystal'
    const trained = text(t.trained)
    const c = t.network_vs_trees
    return {
      t, i, id: text(t.id) || `task-${i}`, ok: t.status === 'ok', title, site: text(t.site), family, unit, crystal, trained,
      canOpen: !!dom?.live && !dom.overview, famOrder: di < 0 ? doms.length : di, rec: modelKey(t.recommended),
      isLog: /log10/i.test(unit), dft: crystal || /DFT/i.test(unit), earlier: !!trained && trained < newest,
      exploratory: text(t.exploratory),
      verdict: !c ? 'na' : !c.better ? 'none' : c.better === 'network' || c.better === 'trees' ? c.better : 'other',
      hay: [title, family, unit, t.name, t.id, t.site, t.recommended].map(text).join(' ').toLowerCase(),
    }
  })
  const ok = rows.filter((r) => r.ok)
  const fams = new Map()
  for (const r of ok) {
    const f = fams.get(r.site) ?? { site: r.site, label: r.family, order: r.famOrder, n: 0 }
    f.n += 1
    fams.set(r.site, f)
  }
  const tally = {}
  for (const v of ['trees', 'network', 'none', 'na', 'other']) tally[v] = ok.filter((r) => r.verdict === v)
  // The export's own reason for each exploratory mark, with the tasks that carry it.
  const why = new Map()
  for (const r of ok) if (r.exploratory) why.set(r.exploratory, [...(why.get(r.exploratory) ?? []), r.title])
  return {
    rows, ok, bad: rows.filter((r) => !r.ok), families: [...fams.values()].sort((a, b) => a.order - b.order), tally, newest,
    why: [...why].map(([reason, titles]) => ({ reason, titles })),
  }
}

// The model a task is judged by, its miss as a share of the no-model miss, and its 90% range.
function judgeTask(r, judge) {
  const m = judge === 'recommended' ? r.rec : judge
  const mae = r.t.mae ?? {}
  const share = m && isNum(mae[m]) && isNum(mae.mean) && mae.mean > 0 ? mae[m] / mae.mean : null
  return { r, m, share, iv: (m && r.t.interval90?.[m]) || null }
}

// Plain percentages only ("50", "50%", "62,5"), above 0 and no more than MAX_SHARE; blank means no requirement.
const MAX_SHARE = 1000
function parseShare(raw) {
  const s = String(raw).trim().replace(/\s*%$/, '').replace(',', '.')
  if (s === '') return { value: null, error: '' }
  if (!/^(\d+(\.\d*)?|\.\d+)$/.test(s) || !(Number(s) > 0) || Number(s) > MAX_SHARE) {
    return { value: null, error: `Type the share as a plain number above 0 and no more than ${MAX_SHARE}, for example 50, or leave it blank for no requirement.` }
  }
  return { value: Number(s), error: '' }
}

function FamilyLink({ r, onOpen }) {
  if (!r.canOpen || !onOpen) return <span>{r.family}</span>
  return <button type="button" className="dash-link" onClick={() => onOpen(r.site)}>{r.family}</button>
}

// One task's figures, shared by the table (wide screens) and the cards (phones).
function Mae({ r, model }) {
  return (
    <>
      {fmt(r.t.mae?.[model])}
      {r.rec === model && <span className="label">recommended</span>}
    </>
  )
}

function Share({ j }) {
  const { r, m, share } = j
  return (
    <>
      {share === null ? '–' : pct(share)}
      {share !== null && <span className="dash-meter" aria-hidden="true"><i style={{ width: `${Math.min(share, 1) * 100}%` }} /></span>}
      <span className={m ? 'label' : 'label dash-wraplabel'}>{m ? `no model: ${fmt(r.t.mae?.mean)}` : 'the export names no recommended model'}</span>
    </>
  )
}

function Range({ j }) {
  const { r, m, iv } = j
  if (!isNum(iv?.half_width)) return <>–<span className="label">not in this export</span></>
  return (
    <>
      ±{fmt(iv.half_width)}
      {r.isLog && <span className="label">{factor(iv.half_width)}</span>}
      <span className="label">held {pct(iv.coverage)}</span>
      <span className="label">{SHORT[m]}</span>
    </>
  )
}

// The network-against-trees comparison of one task. A rehearsal gets the difference with no verdict and no one-network count.
function Verdict({ r, rehearsal }) {
  const sn = r.t.single_network
  return (
    <>
      {compare(r.t.network_vs_trees, rehearsal) ?? <span className="label">not in this export</span>}
      {sn && !rehearsal && (
        <span className="label">
          One network at a time: clearly better in {sn.clearly_better} of {sn.n_runs}, clearly worse in {sn.clearly_worse} of {sn.n_runs}
        </span>
      )}
    </>
  )
}

function Trained({ r }) {
  const when = stamp(r.trained)
  return (
    <>
      {when.date}
      {when.time && <span className="label">{when.time}</span>}
      {r.earlier && <span className="label">an earlier night</span>}
    </>
  )
}

// The three models' misses and the no-model miss as bars on one scale (the largest of the four fills the track).
function Bars({ r, judged }) {
  const mae = r.t.mae ?? {}
  const labels = {
    network: r.crystal && r.t.n_runs ? `Network (average of ${r.t.n_runs} runs)` : MODELS[0].label,
    trees: r.crystal && r.t.strongest_trees ? `${MODELS[1].label} (${r.t.strongest_trees})` : MODELS[1].label,
    blend: MODELS[2].label,
  }
  const items = [...MODELS.map((m) => ({ key: m.key, label: labels[m.key], v: mae[m.key] })),
    { key: 'mean', label: 'Predicting the average every time (no model)', v: mae.mean }]
  const max = Math.max(0, ...items.map((x) => (isNum(x.v) ? x.v : 0)))
  return (
    <ul className="dash-bars">
      {items.map((x) => (
        <li key={x.key} className={x.key === judged ? 'dash-barrow dash-judged' : 'dash-barrow'}>
          <span className="dash-barlabel">
            {x.label}
            {r.rec === x.key && <span className="flag plain">Recommended</span>}
          </span>
          <span className="dash-bartrack" aria-hidden="true"><i style={{ width: isNum(x.v) && max > 0 ? `${(x.v / max) * 100}%` : 0 }} /></span>
          <span className="dash-barvalue">{fmt(x.v)}</span>
        </li>
      ))}
    </ul>
  )
}

// The three paired comparisons on one scale: a dot at (miss of the first minus miss of the second), a bar across its 95% interval,
// and a dashed line at zero. A bar that crosses the line is "no clear difference" and is drawn faint. A rehearsal gets the figures
// with no verdict: no verdict words, no one-network count, and no faint bars.
function Comparisons({ t, rehearsal }) {
  const pairs = PAIRS.map(([key, label]) => ({ key, label, c: t[key] ?? null }))
  const vals = pairs.flatMap((p) => (p.c ? [p.c.diff, ...(Array.isArray(p.c.ci95) ? p.c.ci95 : [])] : [])).filter(isNum)
  const lo0 = Math.min(0, ...vals), hi0 = Math.max(0, ...vals)
  const pad = (hi0 - lo0) * 0.05
  const lo = lo0 - pad, hi = hi0 + pad
  const pos = (v) => (hi > lo ? ((v - lo) / (hi - lo)) * 100 : 50)
  const sn = t.single_network
  return (
    <>
      <ul className="dash-ci">
        {pairs.map(({ key, label, c }) => {
          const ci = Array.isArray(c?.ci95) && c.ci95.every(isNum) ? c.ci95 : null
          return (
            <li key={key} className={c && !c.better && !rehearsal ? 'dash-cross' : undefined}>
              <span className="dash-pair">{label}</span>
              {compare(c, rehearsal) ?? 'Not in this export.'}
              {c && isNum(c.diff) && (
                <span className="dash-track" aria-hidden="true">
                  <i className="dash-zero" style={{ left: `${pos(0)}%` }} />
                  {ci && <i className="dash-cibar" style={{ left: `${pos(ci[0])}%`, width: `${pos(ci[1]) - pos(ci[0])}%` }} />}
                  <i className="dash-cidot" style={{ left: `${pos(c.diff)}%` }} />
                </span>
              )}
            </li>
          )
        })}
        {sn && (
          <li>
            <span className="dash-pair">One network at a time against the strongest trees</span>
            {rehearsal ? 'Rehearsal: no count stated.'
              : `Clearly better in ${sn.clearly_better} of ${sn.n_runs}, clearly worse in ${sn.clearly_worse} of ${sn.n_runs}.`}
            {' '}Average miss of one network at a time over the {sn.n_runs} runs: {fmt(sn.mean_mae)}{rehearsal ? ', which means nothing' : ''}.
          </li>
        )}
      </ul>
      <p className="note small">
        Each figure is the miss of the first minus the miss of the second (negative: the first misses less), then its 95% interval from
        resampling whole groups.{vals.length > 0 && ` The bars share one scale, ${signed(lo0)} to ${signed(hi0)} in the target's unit; the dashed line is zero.`}{' '}
        {rehearsal ? 'This is a rehearsal, so no verdict is drawn: the figures only show that the code ran.'
          : <>&ldquo;Clearly better&rdquo; only when the whole interval is on one side of zero.</>}
      </p>
    </>
  )
}

function Detail({ item, rehearsal, onOpen, position, total, onStep }) {
  const { r, m } = item
  const t = r.t
  const iv = t.interval90 ?? null
  const when = stamp(r.trained)
  const what = r.crystal ? ['crystal', 'crystals', 'chemical system', 'chemical systems'] : ['row', 'rows', 'group', 'groups']
  const cv = t.mode === 'cv'
  const sized = `${count(t.rows)} ${plural(t.rows, what[0], what[1])} in ${count(t.groups)} ${plural(t.groups, what[2], what[3])}`
  // The export's rows and groups are the test rows only for cross-validated tasks (every row is held out once) and for crystals (the
  // test crystals). For a fixed split, gpu/pipeline/analyze.py counts the rows the models were trained on; the test set is separate.
  const tested = cv || r.crystal ? `Tested on ${sized}${splitText(t) ? `: ${splitText(t)}` : ''}.`
    : t.mode === 'fixed' ? `One fixed train and test split (not cross-validated). The export's count for this task, ${sized}, is of the rows the models were trained on; it does not carry the size of the separate test set.`
      : `The export counts ${sized} for this task${splitText(t) ? ` (split: ${splitText(t)})` : ''} and does not say how many of them were test rows.`
  return (
    <>
      <div className="dash-detailhead">
        <div>
          <h2>{r.title}</h2>
          <p className="note">
            {r.family}. Target: {r.unit || 'not stated'}. Trained {when.date}{when.time ? ` at ${when.time}` : ''}
            {text(t.folder) ? ` (run ${text(t.folder)})` : ''}{r.earlier ? ', in an earlier night than the newest task' : ''}.
          </p>
        </div>
        {total > 1 && (
          <div className="dash-step">
            <button type="button" className="chip" onClick={() => onStep(-1)} disabled={position <= 0}>Previous task</button>
            <span className="note small">{position + 1} of {total} shown</span>
            <button type="button" className="chip" onClick={() => onStep(1)} disabled={position >= total - 1}>Next task</button>
          </div>
        )}
      </div>
      {rehearsal && <p className="flag low">Rehearsal: the code ran on a tiny test run; these numbers mean nothing.</p>}
      {r.exploratory && <p className="dash-flagline"><span className="flag low">Exploratory</span> {noDot(r.exploratory)}.</p>}
      <p className="note">
        {tested}{' '}
        {r.dft ? 'The values the models are scored against are DFT-calculated, not measured.' : 'Scored against the measured values.'}
        {r.crystal && Array.isArray(t.neighbor_rule) && t.neighbor_rule.length > 0 && ` Networks' neighbour rule: ${t.neighbor_rule.join(', ')}.`}
      </p>
      <div className="dash-two">
        <div>
          <h3>Typical miss (mean absolute error, lower is better)</h3>
          <Bars r={r} judged={m} />
          <p className="note small">
            In the target&apos;s unit{r.isLog ? ' (a log10: a miss of w is a factor of 10^w either way)' : ''}. The solid bar is the model judged; the others are faint.
          </p>
        </div>
        <div>
          <h3>Is one model really better?</h3>
          <Comparisons t={t} rehearsal={rehearsal} />
        </div>
      </div>
      <h3>90% ranges</h3>
      {iv ? (
        <div className="tablewrap sim-scroll">
          <table className="compact dash-iv">
            <thead>
              <tr>
                <th>Model</th>
                <th className="num">{cv ? 'Average ± over the folds' : 'Prediction ±'}{r.isLog ? ' (and as a factor)' : ''}</th>
                <th className="num">How often it held</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map((mm) => (
                <tr key={mm.key} className={mm.key === m ? 'on' : undefined}>
                  <td>{NAME[mm.key]}{r.rec === mm.key ? ' (recommended)' : ''}</td>
                  <td className="num">
                    {isNum(iv[mm.key]?.half_width) ? `±${fmt(iv[mm.key].half_width)}` : '–'}
                    {r.isLog && isNum(iv[mm.key]?.half_width) && <span className="label">{factor(iv[mm.key].half_width)}</span>}
                  </td>
                  <td className="num">{pct(iv[mm.key]?.coverage)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="note">This export carries no 90% range for this task.</p>
      )}
      {iv && (
        <p className="note small">
          {cv ? (
            <>A prediction ± its fold&apos;s half-width is built to hold the true value 9 times in 10. Each fold&apos;s half-width is set from the
              misses on the other folds; the ± shown is the average of those widths. &ldquo;How often it held&rdquo; is the share of all the
              held-out rows that fell inside their own fold&apos;s range.</>
          ) : (
            <>A prediction ± the half-width is built to hold the true value 9 times in 10. &ldquo;How often it held&rdquo; is the share of the held-out
              {r.crystal ? ' test crystals where it did (the range is calibrated on the validation crystals)' : ` ${what[1]} where it did`}.</>
          )}
          {m && ' The highlighted row is the model judged.'}
        </p>
      )}
      <div className="dash-actions">
        {r.canOpen && onOpen && <button type="button" className="dash-open" onClick={() => onOpen(r.site)}>Open the page: {r.family}</button>}
        <span className="note small">
          {r.site === 'concrete' ? 'Its page is the live designer; this task’s rows are not published on the site.'
            : t.rows_file ? 'That page has this task’s held-out rows (a plot of truth against prediction, and a search) and its interactive part.'
              : 'This task’s rows are not published on the site; only the numbers above are.'}
        </span>
      </div>
    </>
  )
}

// "How this can be wrong": on the page in every state (README.md). The bullets that depend on the export appear only with one.
function Caveats({ prep, rehearsal }) {
  const dftTasks = prep ? prep.ok.filter((r) => r.dft) : []
  const why = prep?.why ?? []
  return (
    <section className="card fam-section warn-box dash">
      <h2>How this can be wrong</h2>
      <ul>
        <li>Every figure is a test on data that already exists. It narrows the list of experiments; it does not replace one.</li>
        {rehearsal && <li>This export is a rehearsal on tiny runs: none of its numbers is a result.</li>}
        <li>
          The misses are in different units, on different data sets. Only the share of the no-model miss compares across tasks, and it
          also depends on how spread out each data set is.
        </li>
        <li>The 90% ranges are calibrated on rows like the ones tested. A material unlike anything in the data can fall outside them more often.</li>
        <li>&ldquo;Network average&rdquo; is an average of several trained networks set against one tree model; one network alone can do worse than the average.</li>
        {dftTasks.length > 0 && (
          <li>
            {count(dftTasks.length)} {plural(dftTasks.length, 'task is', 'tasks are')} scored against DFT-calculated values, not measurements
            ({dftTasks.map((r) => r.title).join(', ')}). Agreeing with a calculation is not agreeing with an experiment.
          </li>
        )}
        {why.length > 0 && (
          <li>
            Tasks marked exploratory, each with the reason the export gives:
            <ul className="dash-why">
              {why.map((w) => <li key={w.reason}><strong>{w.titles.join(', ')}</strong>: {noDot(w.reason)}.</li>)}
            </ul>
          </li>
        )}
      </ul>
    </section>
  )
}

export default function SimDashboard({ domain, domains, onOpen, onBack }) {
  const [index, setIndex] = useState(undefined) // undefined: loading; null: no export; object: index.json
  const [query, setQuery] = useState('')
  const [family, setFamily] = useState('all')
  const [verdict, setVerdict] = useState('any')
  const [order, setOrder] = useState('family')
  const [judge, setJudge] = useState('recommended')
  const [shareText, setShareText] = useState('')
  const [hideExploratory, setHideExploratory] = useState(false)
  const [picked, setPicked] = useState(null)
  const uid = useId()
  const detailRef = useRef(null)
  const badRef = useRef(null)

  // No abort signal: an aborted fetch would sit in loadPipeline's cache for the second mount of a development double render. If a
  // request another page started and then aborted is still cached, the one retry asks again.
  useEffect(() => {
    let live = true
    const get = (retry) => loadPipeline('index.json').then(
      (ix) => { if (live) setIndex(ix && typeof ix === 'object' ? ix : null) },
      () => { if (!live) return; if (retry) get(false); else setIndex(null) },
    )
    get(true)
    return () => { live = false }
  }, [])

  const prep = useMemo(() => (index ? prepare(index, domains) : null), [index, domains])
  const rehearsal = !!index?.rehearsal
  const verdictOn = rehearsal ? 'any' : verdict // a rehearsal has no verdicts to filter by
  const ask = parseShare(shareText)
  const maxShare = ask.value
  const found = useMemo(() => {
    if (!prep) return { shown: [], unjudged: 0 }
    const q = query.trim().toLowerCase()
    let unjudged = 0
    const shown = []
    for (const r of prep.ok) {
      if (q && !r.hay.includes(q)) continue
      if (family !== 'all' && r.site !== family) continue
      if (verdictOn !== 'any' && r.verdict !== verdictOn) continue
      if (hideExploratory && r.exploratory) continue
      const j = judgeTask(r, judge)
      if (maxShare !== null) {
        if (j.share === null) { unjudged += 1; continue }
        if (j.share * 100 > maxShare) continue
      }
      shown.push(j)
    }
    const byShare = (a, b, dir) => (a.share === null) - (b.share === null) || dir * ((a.share ?? 0) - (b.share ?? 0)) || a.r.i - b.r.i
    if (order === 'best') shown.sort((a, b) => byShare(a, b, 1))
    else if (order === 'worst') shown.sort((a, b) => byShare(a, b, -1))
    else if (order === 'trained') shown.sort((a, b) => (a.r.trained < b.r.trained ? 1 : a.r.trained > b.r.trained ? -1 : a.r.i - b.r.i))
    else shown.sort((a, b) => a.r.famOrder - b.r.famOrder || a.r.i - b.r.i)
    return { shown, unjudged }
  }, [prep, query, family, verdictOn, hideExploratory, judge, maxShare, order])

  const title = domain?.label ?? 'All simulations: nightly results'
  const head = (
    <section className="card fam-head">
      <h1>{title}</h1>
      <p className="lede">
        One row for every task the nightly pipeline trains. For each, three models were tested on rows they never trained on: the average
        of the networks, the strongest tree model, and a half-and-half blend of the two. Say how good a model must be, narrow by
        family{rehearsal ? '' : ' or verdict'}, and select a task to see its comparisons. The family name opens that simulation.
      </p>
      {rehearsal && <p className="flag low">Rehearsal export: these numbers come from a tiny test run and mean nothing.</p>}
      {prep && prep.rows.length > 0 && (
        <div className="dash-stats">
          <div><span className="label">Tasks in this export</span><span className="big">{count(prep.rows.length)}</span></div>
          <div><span className="label">With results</span><span className="big">{count(prep.ok.length)}</span></div>
          <div>
            <span className="label">Listed without results</span>
            <span className="big">{count(prep.bad.length)}</span>
            {prep.bad.length > 0 && (
              <button type="button" className="dash-link dash-jump"
                onClick={() => badRef.current?.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'start' })}>
                See why
              </button>
            )}
          </div>
          <div>
            <span className="label">Newest training</span>
            <span className="big dash-date">{stamp(prep.newest).date}</span>
            {stamp(prep.newest).time && <span className="label">at {stamp(prep.newest).time}</span>}
          </div>
        </div>
      )}
      {index && text(index.generated) && <p className="note small">Export written {text(index.generated)}{text(index.registry) ? ` from ${text(index.registry)}` : ''}.</p>}
    </section>
  )
  const back = onBack && <p className="dash-back"><button type="button" className="chip" onClick={onBack}>Open the live concrete designer</button></p>

  if (!prep || prep.rows.length === 0) {
    return (
      <>
        {head}
        <section className="card fam-section">
          {index === undefined ? <p className="note" role="status">Loading the nightly results…</p> : index === null ? (
            <p className="note">The pipeline&apos;s results have not been exported to the site yet. On the training laptop, after a nightly run:
              {' '}<code>python -m gpu.export_site</code>, then commit <code>public/pipeline/</code>.</p>
          ) : <p className="note">This export lists no tasks.</p>}
        </section>
        <Caveats prep={null} rehearsal={rehearsal} />
        {back}
      </>
    )
  }

  const { shown, unjudged } = found
  const selAt = Math.max(0, shown.findIndex((j) => j.r.id === picked))
  const sel = shown[selAt] ?? null
  const judgeWord = JUDGES.find((j) => j.key === judge)?.word ?? 'the recommended model'
  const filtered = query.trim() !== '' || family !== 'all' || verdictOn !== 'any' || hideExploratory || shareText.trim() !== '' || judge !== 'recommended' || order !== 'family'
  const reset = () => { setQuery(''); setFamily('all'); setVerdict('any'); setHideExploratory(false); setShareText(''); setJudge('recommended'); setOrder('family') }
  const open = (site) => { onOpen?.(site); window.scrollTo?.({ top: 0, behavior: 'auto' }) }
  const select = (id) => {
    setPicked(id)
    detailRef.current?.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'nearest' })
  }
  const step = (d) => { const next = shown[selAt + d]; if (next) setPicked(next.r.id) }
  const nOk = prep.ok.length
  const expl = (list) => { const n = list.filter((r) => r.exploratory).length; return n ? ` (${n} of them exploratory)` : '' }

  return (
    <>
      {head}

      <section className="card fam-section dash" aria-label="What to show">
        <h2>What do you need from a model?</h2>
        <div className={rehearsal ? 'dash-form dash-form3' : 'dash-form'}>
          <div>
            <label htmlFor={`${uid}q`}>Search the tasks</label>
            <input id={`${uid}q`} type="search" value={query} autoComplete="off" spellCheck={false} placeholder="task, family or unit…"
              onChange={(e) => setQuery(e.target.value)} />
          </div>
          <div>
            <label htmlFor={`${uid}f`}>Family</label>
            <select id={`${uid}f`} value={family} onChange={(e) => setFamily(e.target.value)}>
              <option value="all">All families ({count(nOk)} {plural(nOk, 'task', 'tasks')})</option>
              {prep.families.map((f) => <option key={f.site} value={f.site}>{f.label} ({f.n})</option>)}
            </select>
          </div>
          {!rehearsal && (
            <div>
              <label htmlFor={`${uid}v`}>Network average against strongest trees</label>
              <select id={`${uid}v`} value={verdict} onChange={(e) => setVerdict(e.target.value)}>
                {VERDICTS.map((v) => <option key={v.key} value={v.key}>{v.label}</option>)}
              </select>
            </div>
          )}
          <div>
            <label htmlFor={`${uid}o`}>Order</label>
            <select id={`${uid}o`} value={order} onChange={(e) => setOrder(e.target.value)}>
              {ORDERS.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
            </select>
          </div>
        </div>
        <fieldset className="dash-req">
          <legend>How good must it be?</legend>
          <div className="dash-reqrow">
            <label htmlFor={`${uid}s`}>Typical miss at most this share of the no-model miss (%)</label>
            <div>
              <div className="dash-sharefield">
                <input id={`${uid}s`} value={shareText} inputMode="decimal" autoComplete="off" placeholder="any"
                  aria-describedby={`${uid}sh`} aria-invalid={ask.error ? 'true' : undefined} onChange={(e) => setShareText(e.target.value)} />
                <div className="examples dash-presets">
                  <span>Try:</span>
                  <button type="button" className="chip" aria-pressed={shareText.trim() === ''} onClick={() => setShareText('')}>Any</button>
                  {SHARE_PRESETS.map((p) => (
                    <button type="button" key={p} className="chip" aria-pressed={maxShare === p} onClick={() => setShareText(String(p))}>{p}%</button>
                  ))}
                </div>
              </div>
              <span className="dash-hint" id={`${uid}sh`}>
                The no-model miss is the miss of predicting the average every time; 100% would be no better than that. Blank: no requirement.
              </span>
              {ask.error && <p className="error" role="alert">{ask.error}</p>}
            </div>
          </div>
          <div className="dash-reqrow">
            <span className="dash-reqlabel" id={`${uid}j`}>Judge each task by</span>
            <div className="dash-toggle" role="group" aria-labelledby={`${uid}j`}>
              {JUDGES.map((j) => (
                <button key={j.key} type="button" className="chip" aria-pressed={judge === j.key} onClick={() => setJudge(j.key)}>{j.label}</button>
              ))}
            </div>
          </div>
        </fieldset>
        <div className="options">
          <label><input type="checkbox" checked={hideExploratory} onChange={(e) => setHideExploratory(e.target.checked)} /> Leave out tasks marked exploratory</label>
        </div>
      </section>

      <section className="card fam-section dash" aria-label="Nightly results by task">
        <h2>Nightly results by task</h2>
        {rehearsal ? (
          <p className="dash-flagline">
            <span className="flag low">Rehearsal</span> The export code ran and wrote {count(prep.rows.length)} {plural(prep.rows.length, 'task', 'tasks')}.
            No result is stated here: the figures below show the format only.
          </p>
        ) : nOk > 0 && (
          <ul className="dash-tally">
            <li>
              <span>Strongest trees clearly better than the network average: <strong>{prep.tally.trees.length} of {nOk}</strong> {plural(nOk, 'task', 'tasks')}{expl(prep.tally.trees)}.</span>
              {prep.tally.trees.length > 0 && <button type="button" className="chip" aria-pressed={verdict === 'trees'} onClick={() => setVerdict(verdict === 'trees' ? 'any' : 'trees')}>Show these</button>}
            </li>
            <li>
              <span>
                Network average clearly better than the strongest trees: <strong>{prep.tally.network.length} of {nOk}</strong> {plural(nOk, 'task', 'tasks')}{expl(prep.tally.network)}.
                {prep.tally.network.length > 0 && ' That is an average of several networks against one tree model; each row says how one network at a time did.'}
              </span>
              {prep.tally.network.length > 0 && <button type="button" className="chip" aria-pressed={verdict === 'network'} onClick={() => setVerdict(verdict === 'network' ? 'any' : 'network')}>Show these</button>}
            </li>
            <li>
              <span>
                No clear difference: <strong>{prep.tally.none.length} of {nOk}</strong> {plural(nOk, 'task', 'tasks')}{expl(prep.tally.none)}
                {prep.tally.none.length > 0 ? ': ' : '.'}
                {prep.tally.none.map((r, k) => (
                  <span key={r.id}>{k > 0 && '; '}{r.title} <span className="dash-nums">{signed(r.t.network_vs_trees.diff)} {interval(r.t.network_vs_trees.ci95)}</span></span>
                ))}
                {prep.tally.none.length > 0 && ' (difference in miss and its 95% interval, in each target’s unit).'}
              </span>
              {prep.tally.none.length > 0 && <button type="button" className="chip" aria-pressed={verdict === 'none'} onClick={() => setVerdict(verdict === 'none' ? 'any' : 'none')}>Show these</button>}
            </li>
            {prep.tally.na.length + prep.tally.other.length > 0 && (
              <li><span>No such comparison in this export: {prep.tally.na.length + prep.tally.other.length} of {nOk}.</span></li>
            )}
          </ul>
        )}
        <div className="dash-status">
          <p className="note small" aria-live="polite">
            Showing {count(shown.length)} of {count(nOk)} {plural(nOk, 'task', 'tasks')} with results
            {maxShare !== null ? `: those where the miss of ${judgeWord} is at most ${fmt(maxShare)}% of the no-model miss` : ''}.
            {unjudged > 0 && ` ${count(unjudged)} left out because the export has no miss for that model or for no model.`}
          </p>
          {filtered && <button type="button" className="chip" onClick={reset}>Clear the choices</button>}
        </div>
        {shown.length === 0 ? (
          <p className="note">
            {nOk === 0 ? 'No task in this export has results. Each one is listed further down with the reason the export gives.'
              : `No task matches these choices. Raise the share, pick another family${rehearsal ? '' : ' or verdict'}, or clear the choices.`}
          </p>
        ) : (
          <>
            <div className="tablewrap sim-scroll dash-wide">
              <table className="ranked dash-table">
                <thead>
                  <tr>
                    <th rowSpan={2}>Task and target</th>
                    <th rowSpan={2}>Family (opens its page)</th>
                    <th colSpan={3} className="group">Typical miss on rows never trained on (MAE, target&apos;s unit)</th>
                    <th rowSpan={2} className="num">Share of the no-model miss</th>
                    <th rowSpan={2} className="num">90% range of {judgeWord}</th>
                    <th rowSpan={2} className="dash-verdicthead">Network average against strongest trees (difference in miss, 95% interval)</th>
                    <th rowSpan={2}>Trained</th>
                  </tr>
                  <tr>
                    {MODELS.map((m) => <th key={m.key} className="num">{NAME[m.key]}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {shown.map((j) => {
                    const { r, m } = j
                    const on = sel?.r.id === r.id
                    return (
                      <tr key={r.id} className={on ? 'on' : undefined}>
                        <td className="dash-taskcell">
                          <strong>{r.title}</strong>
                          <span className="dash-unit">{r.unit || 'unit not stated'}</span>
                          <span className="dash-taskfoot">
                            <button type="button" className="chip" aria-pressed={on} onClick={() => select(r.id)} aria-label={`Details of ${r.title}`}>Details</button>
                            {r.exploratory && <span className="flag low">Exploratory</span>}
                          </span>
                        </td>
                        <td className="dash-famcell"><FamilyLink r={r} onOpen={open} /></td>
                        {MODELS.map((mm) => <td key={mm.key} className={mm.key === m ? 'num strong' : 'num'}><Mae r={r} model={mm.key} /></td>)}
                        <td className="num"><Share j={j} /></td>
                        <td className="num"><Range j={j} /></td>
                        <td className="dash-verdictcell"><Verdict r={r} rehearsal={rehearsal} /></td>
                        <td><Trained r={r} /></td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            {/* The same rows as cards: shown instead of the table where the table would not fit (SimDashboard.css), so nothing needs a sideways scroll. */}
            <ul className="dash-cards">
              {shown.map((j) => {
                const { r, m } = j
                const on = sel?.r.id === r.id
                return (
                  <li key={r.id} className={on ? 'dash-card dash-on' : 'dash-card'}>
                    <strong className="dash-cardtitle">{r.title}</strong>
                    <span className="dash-unit">{r.unit || 'unit not stated'}</span>
                    <p className="dash-cardfam">
                      <FamilyLink r={r} onOpen={open} />
                      {r.exploratory && <span className="flag low">Exploratory</span>}
                    </p>
                    <span className="dash-cardcap">Typical miss on rows never trained on (MAE)</span>
                    <dl className="dash-cardmae">
                      {MODELS.map((mm) => (
                        <div key={mm.key} className={mm.key === m ? 'strong' : undefined}><dt>{NAME[mm.key]}</dt><dd><Mae r={r} model={mm.key} /></dd></div>
                      ))}
                    </dl>
                    <dl className="dash-cardfacts">
                      <div className="dash-inline"><dt>Share of the no-model miss</dt><dd><Share j={j} /></dd></div>
                      <div className="dash-inline"><dt>90% range of {judgeWord}</dt><dd><Range j={j} /></dd></div>
                      <div><dt>Network average against strongest trees</dt><dd><Verdict r={r} rehearsal={rehearsal} /></dd></div>
                    </dl>
                    <div className="dash-cardfoot">
                      <span className="dash-inline"><span className="dash-k">Trained</span> <Trained r={r} /></span>
                      <button type="button" className="chip" aria-pressed={on} onClick={() => select(r.id)} aria-label={`Details of ${r.title}`}>Details</button>
                    </div>
                  </li>
                )
              })}
            </ul>
          </>
        )}
        <dl className="readme">
          <div>
            <dt>Typical miss (MAE)</dt>
            <dd>
              The mean absolute error on rows the model never trained on, in the target&apos;s own unit (under the task&apos;s name), so it cannot be
              compared between tasks. Lower is better. The figure in bold is the model judged; &ldquo;recommended&rdquo; marks the one the project&apos;s
              recipe names for the task.
            </dd>
          </div>
          <div>
            <dt>Share of the no-model miss</dt>
            <dd>
              The miss of the model judged (named under its 90% range) divided by the miss of predicting the average every time, worked out
              here from the two exported errors. It is the one figure that compares across tasks. It has no interval of its own, so tasks a
              few points apart are not clearly different.
            </dd>
          </div>
          <div>
            <dt>90% range</dt>
            <dd>
              A prediction ± this half-width is built to hold the true value 9 times in 10; &ldquo;held&rdquo; is how often it did on the held-out
              rows. For cross-validated tasks the ± is the average of the per-fold half-widths. For a log10 target the factor beside it is
              10 to that power.
            </dd>
          </div>
          <div>
            <dt>Network average against strongest trees</dt>
            <dd>
              The miss of the network average minus the miss of the strongest tree model (negative: the networks miss less), with its 95%
              interval from resampling whole groups.{' '}
              {rehearsal ? 'This is a rehearsal, so no verdict is drawn from it: the figures only show that the code ran.' : (
                <>&ldquo;Clearly better&rdquo; only when the whole interval is on one side of zero. It sets an average of several networks against
                  one tree model, so the line under it says how one network at a time did.</>
              )}
            </dd>
          </div>
        </dl>
        <p className="note small dash-after">
          Exploratory: the export&apos;s reason is in the task&apos;s details. Trained: when the task&apos;s models were last trained, as the
          export records it.
        </p>
      </section>

      <section className="card fam-section dash dash-detail" ref={detailRef} aria-label="Details of the selected task">
        {sel ? <Detail item={sel} rehearsal={rehearsal} onOpen={open} position={selAt} total={shown.length} onStep={step} />
          : <p className="note">No task is shown, so there are no details.{nOk > 0 && ' Clear the choices above to see every task.'}</p>}
      </section>

      {prep.bad.length > 0 && (
        <section className="card fam-section dash dash-detail" ref={badRef} aria-label="Tasks without results in this export">
          <h2>Listed without results ({count(prep.bad.length)})</h2>
          <p className="note">The pipeline knows these tasks, but this export has no numbers for them. The reason is the export&apos;s own.</p>
          <ul className="dash-badlist">
            {prep.bad.map((r) => (
              <li key={r.id}>
                <div className="dash-badhead">
                  <strong>{r.title}</strong>
                  <span className="flag low">{text(r.t.status) || 'status unknown'}</span>
                  {r.exploratory && <span className="flag low">Exploratory</span>}
                </div>
                <p className="dash-badreason"><span className="dash-k">Reason given by the export</span> {text(r.t.reason) || 'none given'}</p>
                <p className="note small">
                  <FamilyLink r={r} onOpen={open} /> · Last trained {stamp(r.trained).date}{stamp(r.trained).time ? ` at ${stamp(r.trained).time}` : ''} ·
                  Recommended model: {r.rec ? SHORT[r.rec] : text(r.t.recommended) || 'not stated'}
                </p>
              </li>
            ))}
          </ul>
        </section>
      )}

      <Caveats prep={prep} rehearsal={rehearsal} />
      {back}
    </>
  )
}
