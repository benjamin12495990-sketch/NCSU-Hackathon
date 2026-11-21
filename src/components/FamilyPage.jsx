import { useEffect, useRef, useState } from 'react'
import { createFamilyTrainer, familyMetrics } from '../lib/nnFamily.js'
import AlloyGenerator from './AlloyGenerator.jsx'
import './neuralnet.css'
import './family.css'

// The page for one trained family (alloys, insulators, polymers). Every number comes from
// public/model/<family>/family.json (written by scripts/build_family_files.mjs from the family
// agent's result files) or is measured live in this browser; none is typed here.
// The files are fetched only when this page opens.

const FOLDS = 3 // extra runs used to measure the 90% interval live
const LEVEL = 0.9
const TICK_MS = 30 // train in slices of about this long so the page stays responsive

// Runs a trainer in slices of about TICK_MS, yielding to the browser between slices.
// onDone gets the wall seconds the whole run took.
function runSliced(t, timer, onTick, onDone) {
  const started = performance.now()
  const tick = () => {
    const t0 = performance.now()
    do t.runEpochs(1); while (!t.done && performance.now() - t0 < TICK_MS)
    onTick?.()
    if (t.done) onDone((performance.now() - started) / 1000)
    else timer.current = setTimeout(tick, 0)
  }
  timer.current = setTimeout(tick, 0)
}

// Spearman rank correlation, for the band-gap trend check.
function spearman(a, b) {
  const rank = (v) => {
    const idx = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0])
    const r = new Array(v.length)
    for (let i = 0; i < idx.length;) {
      let j = i
      while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++
      for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2
      i = j + 1
    }
    return r
  }
  const ra = rank(a), rb = rank(b), n = a.length
  const ma = ra.reduce((s, v) => s + v, 0) / n, mb = rb.reduce((s, v) => s + v, 0) / n
  let num = 0, da = 0, db = 0
  for (let i = 0; i < n; i++) { num += (ra[i] - ma) * (rb[i] - mb); da += (ra[i] - ma) ** 2; db += (rb[i] - mb) ** 2 }
  return num / Math.sqrt(da * db)
}

// Four round ticks from 0 to just above max.
function niceTicks(max) {
  const raw = max / 3
  const p = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw)
  return [0, step, 2 * step, 3 * step]
}

// How each family shows its numbers: as is, or (insulators) 10^y for a log10 target.
function displayFor(fam) {
  const pow = fam.display.kind === 'pow10'
  const d = fam.display.digits
  return {
    pow,
    value: (v) => (pow ? (10 ** v).toFixed(d) : `${v.toFixed(d)} ${fam.unit}`),
    error: (mae) => (pow ? `${mae.toFixed(3)} log10 (a factor of ${(10 ** mae).toFixed(2)})` : `${mae.toFixed(1)} ${fam.unit}`),
    short: (mae) => (pow ? mae.toFixed(3) : mae.toFixed(1)),
    width: (q) => (pow ? `×/÷ ${(10 ** q).toFixed(2)}` : `± ${q.toFixed(1)} ${fam.unit}`),
    errUnit: pow ? 'log10' : fam.unit,
  }
}

// Learning curve: error on the held-out rows after each pass, with dashed lines for the trees
// (first settings and tuned by cross-validation) and for guessing the training mean, all from the file.
function FamilyCurve({ history, total, trees, treesTuned, mean, unit, digits }) {
  const w = 520, h = 240, left = 48, bottom = 34, top = 12, right = 12
  const ticks = niceTicks(mean * 1.25)
  const maxY = ticks[3]
  const x = (e) => left + (Math.log(1 + e) / Math.log(1 + total)) * (w - left - right)
  const y = (v) => top + (1 - Math.min(v, maxY) / maxY) * (h - top - bottom)
  const pts = history.filter((p) => p.mae != null)
  const points = pts.map((p) => `${x(p.epoch).toFixed(1)},${y(p.mae).toFixed(1)}`).join(' ')
  const last = pts[pts.length - 1]
  const xt = [0, 1, 3, 10, 30, 100, 300].filter((e) => e <= total)
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="nncurve" role="img" aria-label="Error on held-out rows falling as the network trains">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={left} y1={y(t)} x2={w - right} y2={y(t)} className="grid" />
          <text x={left - 6} y={y(t) + 4} textAnchor="end">{+t.toFixed(digits + 1)}</text>
        </g>
      ))}
      {xt.map((e) => <text key={e} x={x(e)} y={h - bottom + 16} textAnchor="middle">{e}</text>)}
      <line x1={left} y1={y(trees)} x2={w - right} y2={y(trees)} className="tree" />
      <text x={left + 6} y={y(trees) + (trees >= treesTuned ? -6 : 14)}>trees, first settings: {trees.toFixed(digits)}</text>
      <line x1={left} y1={y(treesTuned)} x2={w - right} y2={y(treesTuned)} className="tree" />
      <text x={w - right - 6} y={y(treesTuned) + (treesTuned > trees ? -6 : 14)} textAnchor="end">trees, tuned: {treesTuned.toFixed(digits)}</text>
      <line x1={left} y1={y(mean)} x2={w - right} y2={y(mean)} className="tree" />
      <text x={left + 6} y={y(mean) - 6}>guess the mean: {mean.toFixed(digits)}</text>
      {pts.length > 1 && <polyline points={points} className="line" />}
      {last && <circle cx={x(last.epoch)} cy={y(last.mae)} r="4" className="dot" />}
      <text x={(w + left) / 2} y={h - 2} textAnchor="middle">Passes over the training data (log scale)</text>
      <text x={12} y={(h - bottom) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(h - bottom) / 2})`}>Error ({unit})</text>
    </svg>
  )
}

// A plain x-y chart: lines (solid = live network, dashed = from the file), an optional band and dots.
function XYChart({ lines, dots = [], band = 0, xLabel, yLabel, logY = false }) {
  const w = 400, h = 240, left = 48, bottom = 34, top = 10, right = 10
  const all = [...lines.flatMap((l) => l.points), ...dots]
  const xs = all.map((p) => p.x), ys = all.map((p) => p.y)
  const x0 = Math.min(...xs), x1 = Math.max(...xs)
  const y0 = logY ? Math.min(...ys) : 0
  const y1 = Math.max(...ys) + band
  const ticks = logY ? [y0, (y0 + y1) / 2, y1] : niceTicks(y1)
  const yMax = logY ? y1 : ticks[3]
  const x = (v) => left + ((v - x0) / (x1 - x0 || 1)) * (w - left - right)
  const y = (v) => top + (1 - (Math.max(y0, Math.min(v, yMax)) - y0) / (yMax - y0 || 1)) * (h - top - bottom)
  const path = (p) => p.map((q) => `${x(q.x).toFixed(1)},${y(q.y).toFixed(1)}`).join(' ')
  const live = lines.find((l) => l.live)
  const area = live && band > 0 && [...live.points.map((p) => `${x(p.x).toFixed(1)},${y(p.y + band).toFixed(1)}`),
    ...[...live.points].reverse().map((p) => `${x(p.x).toFixed(1)},${y(p.y - band).toFixed(1)}`)].join(' ')
  const xt = [x0, (x0 + x1) / 2, x1]
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="nncurve" role="img" aria-label={`${yLabel} against ${xLabel}`}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={left} y1={y(t)} x2={w - right} y2={y(t)} className="grid" />
          <text x={left - 6} y={y(t) + 4} textAnchor="end">{logY ? (10 ** t).toFixed(0) : +t.toFixed(1)}</text>
        </g>
      ))}
      {xt.map((t) => <text key={t} x={x(t)} y={h - bottom + 16} textAnchor="middle">{+t.toFixed(1)}</text>)}
      {area && <polygon points={area} className="band" />}
      {lines.map((l) => <polyline key={l.key} points={path(l.points)} className={l.live ? 'line' : 'tree'} fill="none" />)}
      {dots.map((d, i) => <circle key={i} cx={x(d.x)} cy={y(d.y)} r="2.6" className="fam-dot" />)}
      <text x={(w + left) / 2} y={h - 2} textAnchor="middle">{xLabel}</text>
      <text x={12} y={(h - bottom) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(h - bottom) / 2})`}>{yLabel}</text>
    </svg>
  )
}

// Rank the held-out compounds (insulators): EXISTING compounds the network never trained on.
function HeldoutRanking({ fam, heldout, predictions }) {
  const [atLeast, setAtLeast] = useState(5)
  const [revealed, setRevealed] = useState(false)
  const q = fam.interval.q
  const cap = Math.floor(fam.highTail.largestPrediction / 10 ** q) // the largest lower edge the file's predictions reach
  const rows = heldout.map((h, i) => ({ ...h, pred: predictions[i], lower: 10 ** (predictions[i] - q) }))
    .filter((r) => r.lower >= atLeast)
    .sort((a, b) => b.pred - a.pred)
  const shown = rows.slice(0, 15)
  const hits = rows.filter((r) => 10 ** r.y >= atLeast).length
  return (
    <section className="card fam-section">
      <h2>Rank compounds the network never trained on</h2>
      <span className="flag low">EXISTING compounds, held out from training (not new materials)</span>
      <p className="note">
        This model cannot invent new compounds tonight. Instead it ranks the {heldout.length} real compounds that were
        held out of training, by the lower edge of its 90% range ({`prediction ÷ ${(10 ** q).toFixed(2)}`}). Choose
        the threshold first; the calculated values are shown only after you press Reveal.
      </p>
      <div className="reqs fam-reqs">
        <label className="req">
          <span>Dielectric constant at least</span>
          <input type="range" min="2" max={cap} step="1" value={atLeast}
            onChange={(e) => { setAtLeast(Number(e.target.value)); setRevealed(false) }} />
          <span className="reqvalue">{atLeast}</span>
        </label>
      </div>
      <p className="note small">
        The slider stops at {cap}: the largest held-out prediction in the results file is {fam.highTail.largestPrediction.toFixed(1)}, so
        the lower edge never goes much above {cap}, while the largest calculated value is {fam.highTail.largestTrue.toFixed(0)}.
        On held-out compounds above 30 the network predicts about {Math.round(100 * fam.highTail.medianPredOverTrue)}% of
        the calculated value (median): it under-predicts the high tail by about {(1 / fam.highTail.medianPredOverTrue).toFixed(1)}×, so it cannot find them.
      </p>
      {shown.length === 0 ? (
        <p className="note">No held-out compound has a lower edge at or above {atLeast}.</p>
      ) : (
        <div className="tablewrap">
          <table>
            <thead>
              <tr><th className="num">Rank</th><th>Compound</th><th className="num">Predicted</th><th className="num">Lower edge (results-file 90% range, ÷ {(10 ** q).toFixed(2)})</th>{revealed && <th className="num">Calculated (check)</th>}</tr>
            </thead>
            <tbody>
              {shown.map((r, i) => (
                <tr key={r.id}>
                  <td className="num">{i + 1}</td><td>{r.formula} <span className="label">{r.id}</span></td>
                  <td className="num strong">{(10 ** r.pred).toFixed(1)}</td><td className="num">{r.lower.toFixed(1)}</td>
                  {revealed && <td className="num">{(10 ** r.y).toFixed(1)}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="nnrow">
        <button type="button" className="chip" onClick={() => setRevealed(true)} disabled={revealed || shown.length === 0}>Reveal the calculated values</button>
        {revealed && <span className="note">{hits} of the {rows.length} ranked compounds have a calculated value of at least {atLeast}{rows.length > shown.length ? ` (top ${shown.length} shown)` : ''}.</span>}
      </div>
    </section>
  )
}

export default function FamilyPage({ family, onBack }) {
  const [fam, setFam] = useState(null)
  const [nnData, setNnData] = useState(null)
  const [heldout, setHeldout] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [history, setHistory] = useState([])
  const [status, setStatus] = useState('idle') // idle, training, measuring, done
  const [foldsDone, setFoldsDone] = useState(0)
  const [band, setBand] = useState(null)
  const [preds, setPreds] = useState(null) // live test predictions
  const [sweepLive, setSweepLive] = useState(null)
  const [info, setInfo] = useState(null) // { weights, layers, seconds }
  const [net, setNet] = useState(null) // the network trained on this page (the alloy generator uses it)
  const timer = useRef(0)
  const afterTrain = useRef(null) // called once with (network, live band or null) when a train run finishes

  useEffect(() => {
    const ctl = new AbortController()
    const get = (f) => fetch(`${import.meta.env.BASE_URL}model/${family}/${f}`, { signal: ctl.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${f}: HTTP ${r.status}`))))
    Promise.all([get('family.json'), get('nn_data.json')])
      .then(([f, d]) => {
        setFam(f); setNnData(d)
        if (family === 'insulators') return get('heldout.json').then((h) => setHeldout(h.rows))
        return null
      })
      .catch((e) => { if (e.name !== 'AbortError') setLoadError(e.message) })
    const pending = timer // the ref object itself; its .current is the running timeout
    return () => { ctl.abort(); clearTimeout(pending.current) }
  }, [family])

  if (loadError) return <section className="card"><p className="error">The family files did not load: {loadError}</p></section>
  if (!fam || !nnData) return <section className="card"><p className="note">Loading the {family} data…</p></section>

  const show = displayFor(fam)
  const opts = { ...fam.options }

  // 3-fold grouped cross-validation on the training rows only, as the concrete panel does.
  const measure = (main) => {
    const groups = [...new Set(nnData.train.g)].sort((a, b) => a - b)
    const foldOf = new Map(groups.map((id, i) => [id, i % FOLDS]))
    const errors = []
    let k = 0
    const next = () => {
      const tr = { X: [], y: [] }, left = { X: [], y: [] }
      nnData.train.X.forEach((row, i) => {
        const part = foldOf.get(nnData.train.g[i]) === k ? left : tr
        part.X.push(row); part.y.push(nnData.train.y[i])
      })
      const fold = createFamilyTrainer({ features: nnData.features, train: tr, test: left }, opts)
      runSliced(fold, timer, null, () => {
        left.X.forEach((row, i) => errors.push(Math.abs(fold.predict(row) - left.y[i])))
        k++
        setFoldsDone(k)
        if (k < FOLDS) return next()
        errors.sort((a, b) => a - b)
        const q = errors[Math.min(errors.length - 1, Math.ceil(LEVEL * (errors.length + 1)) - 1)]
        const inside = nnData.test.X.filter((row, i) => Math.abs(main.predict(row) - nnData.test.y[i]) <= q).length
        const b = { q, coverage: inside / nnData.test.X.length, errors: errors.length }
        setBand(b)
        finish(main, b)
      })
    }
    next()
  }

  const finish = (t, b) => {
    setStatus('done')
    const after = afterTrain.current
    afterTrain.current = null
    after?.(t, b)
  }

  const train = () => {
    clearTimeout(timer.current)
    const t = createFamilyTrainer(nnData, { ...opts, trackTest: true })
    setHistory([...t.history]); setBand(null); setPreds(null); setSweepLive(null); setFoldsDone(0); setNet(null)
    setStatus('training')
    runSliced(t, timer, () => setHistory([...t.history]), (seconds) => {
      setInfo({ weights: t.weights, layers: t.layers, seconds })
      setNet(t)
      setPreds(nnData.test.X.map((x) => t.predict(x)))
      if (fam.sweep) setSweepLive(fam.sweep.rows.map((row) => t.predict(row)))
      if (fam.interval.liveMeasure) { setStatus('measuring'); measure(t) } else finish(t, null)
    })
  }

  // For the alloy generator: run cb(network, band) once the whole train flow (with the live
  // interval) has finished, starting the flow if nothing is running.
  const whenTrained = (cb) => {
    afterTrain.current = cb
    if (status !== 'training' && status !== 'measuring') train()
  }

  const last = history.filter((p) => p.mae != null).at(-1)
  const trained = preds != null
  const live = trained ? familyMetrics(preds, nnData.test.y) : null
  const busy = status === 'training' || status === 'measuring'
  const kindClass = fam.kind === 'CALCULATED' ? 'flag low' : 'flag ok'
  const caveats = [...fam.caveats, 'Settings were chosen by cross-validation on training rows; the test split was scored once.']

  return (
    <div className="fam">
      <section className="card fam-head">
        <span className={kindClass}>{fam.kindText}</span>
        <h1>{fam.label}</h1>
        <p className="lede">{fam.task}.</p>
        {fam.important && <p className="note"><strong>{fam.important}</strong></p>}
        <dl className="plan">
          <div><dt>Source</dt><dd>{fam.data.citation}{fam.data.source.startsWith('http') ? <> · <a href={fam.data.source}>data</a></> : <> · {fam.data.source}</>}</dd></div>
          <div><dt>Licence</dt><dd>{fam.data.licence}</dd></div>
          <div><dt>Rows</dt><dd>{fam.data.rows} rows, {fam.data.groups} {fam.data.groupWord}</dd></div>
          <div><dt>Train / test</dt><dd>{fam.data.train} / {fam.data.test}, {fam.data.splitRule}</dd></div>
          <div><dt>Inputs</dt><dd>{fam.data.inputs} numbers: {fam.data.inputsText}</dd></div>
          <div><dt>Target</dt><dd>{fam.target}{fam.unit ? ` (${fam.unit})` : ''}</dd></div>
        </dl>
      </section>

      <section className="nn" id={`train-${family}`}>
        <h2>Train the network yourself</h2>
        <p className="note">
          Press the button and a small neural network starts from random numbers and learns from the {fam.data.train} training
          rows, here in your browser (seed {opts.seed}, layers {[nnData.features.length, ...opts.hidden, 1].join(', ')}, {opts.epochs} passes).
          Nothing pre-trained is downloaded. The curve is its error on the {fam.data.test} held-out rows, which it never trains on.
        </p>
        <div className="nnrow">
          <button type="button" className="chip nnbtn" onClick={train} disabled={busy}>
            {status === 'idle' ? 'Train the network' : status === 'training' ? 'Training…'
              : status === 'measuring' ? `Measuring its uncertainty: run ${Math.min(FOLDS, foldsDone + 1)} of ${FOLDS}…` : 'Train again from scratch'}
          </button>
          {last && (
            <div className="nnstats">
              <div><span className="label">Pass</span><span className="big">{last.epoch} <small>of {opts.epochs}</small></span></div>
              <div><span className="label">Error on held-out rows</span><span className="big">{show.short(last.mae)} <small>{show.errUnit}</small></span></div>
              {band && <div><span className="label">90% interval (live)</span><span className="big">{show.width(band.q)}</span></div>}
            </div>
          )}
        </div>
        {history.length > 0 && (
          <div className="nnbody">
            <figure>
              <FamilyCurve history={history} total={opts.epochs} trees={fam.test.trees} treesTuned={fam.test.treesTuned} mean={fam.test.mean} unit={show.errUnit} digits={show.pow ? 3 : 1} />
              <figcaption>Lower is better. Dashed lines, from the results file: gradient-boosted trees (first settings, and tuned by cross-validation) and always guessing the training mean.</figcaption>
            </figure>
          </div>
        )}
        {info && <p className="note small">Trained in {info.seconds.toFixed(1)} s in this browser, {info.weights.toLocaleString()} weights.</p>}
      </section>

      <section className="card fam-section">
        <h2>How good is it?</h2>
        <div className="tablewrap">
          <table>
            <thead><tr><th>Model, on the {fam.data.test} held-out rows</th><th className="num">Average miss ({show.errUnit})</th><th className="num">R²</th></tr></thead>
            <tbody>
              <tr className={trained ? 'on' : undefined}><td>This browser's network (one network, seed {opts.seed})</td><td className="num strong">{live ? show.short(live.mae) : 'press Train'}</td><td className="num">{live ? live.r2.toFixed(2) : ''}</td></tr>
              <tr><td>Five networks averaged (trained in Node with the same code)</td><td className="num">{show.short(fam.test.network)}</td><td className="num">{fam.test.networkR2.toFixed(2)}</td></tr>
              <tr><td>Gradient-boosted trees, first settings</td><td className="num">{show.short(fam.test.trees)}</td><td className="num">{fam.test.treesR2.toFixed(2)}</td></tr>
              <tr><td>Gradient-boosted trees, tuned fairly by cross-validation</td><td className="num">{show.short(fam.test.treesTuned)}</td><td className="num"></td></tr>
              <tr><td>Always guess the training mean</td><td className="num">{show.short(fam.test.mean)}</td><td className="num"></td></tr>
            </tbody>
          </table>
        </div>
        <p className="note"><strong>Verdict.</strong> {fam.verdict}</p>
        <p className="note small">{fam.eightSplits.wording}</p>
        {fam.subsets && (
          <>
            <h3>Where the error comes from</h3>
            <div className="tablewrap">
              <table>
                <thead><tr><th>Held-out rows</th><th className="num">Rows</th><th className="num">Network ({fam.unit})</th><th className="num">Trees ({fam.unit})</th><th className="num">Inside the 90% range</th></tr></thead>
                <tbody>
                  {fam.subsets.map((s) => (
                    <tr key={s.name}><td>{s.name}</td><td className="num">{s.rows}</td><td className="num strong">{s.network.toFixed(1)}</td><td className="num">{s.trees.toFixed(1)}</td><td className="num">{Math.round(s.coverage * 100)}%</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="note small">Networks averaged in Node, from the results file. On a split where whole monomers are unseen: network {fam.unseenMonomer.network.toFixed(1)} K, trees {fam.unseenMonomer.trees.toFixed(1)} K.</p>
          </>
        )}
      </section>

      <section className="card fam-section">
        <h2>90% interval</h2>
        <p className="big">{show.width(fam.interval.q)}</p>
        <p className="note">
          From the results file: {fam.interval.rule}. On the held-out rows it held for {Math.round(fam.interval.coverage * 100)} in 100.
          {band && <> Estimated live in this browser by training {FOLDS} more networks on two thirds of the training rows each and scoring
          the {band.errors} rows left out: <strong>{show.width(band.q)}</strong>, which held for <strong>{Math.round(band.coverage * 100)} in 100</strong> held-out rows.</>}
          {!fam.interval.liveMeasure && <> {fam.interval.noLiveReason}</>}
        </p>
        <p className="note small">The width is one constant everywhere, which is crude. It fails on {fam.interval.failsOn}.</p>
      </section>

      {fam.sweep && (
        <section className="card fam-section">
          <h2>Sweep test: {fam.sweep.title}</h2>
          <XYChart xLabel={fam.sweep.xLabel} yLabel={`Predicted ${fam.unit}`} band={band ? band.q : 0}
            lines={[
              { key: 'file', points: fam.sweep.x.map((x, i) => ({ x, y: fam.sweep.network[i] })) },
              ...(sweepLive ? [{ key: 'live', live: true, points: fam.sweep.x.map((x, i) => ({ x, y: sweepLive[i] })) }] : []),
            ]}
            dots={fam.sweep.measured} />
          <p className="note small">
            Dashed: the five-network average from the results file. {sweepLive ? 'Solid: the network you just trained' : 'Press Train to add the network you train'}
            {band ? ', with its live 90% interval as the band' : ''}. Dots: as-cast measurements in the dataset.
            {sweepLive && ` Live: ${sweepLive[0].toFixed(0)} ${fam.unit} at x = ${fam.sweep.x[0]} to ${sweepLive.at(-1).toFixed(0)} ${fam.unit} at x = ${fam.sweep.x.at(-1)}, rising in ${sweepLive.slice(1).filter((v, i) => v > sweepLive[i]).length} of ${sweepLive.length - 1} steps.`}
            {' '}{fam.sweep.note}
          </p>
        </section>
      )}

      {fam.compositionSweep && (
        <section className="card fam-section">
          <h2>Sweep test: composition</h2>
          <p className="note">
            From the results file: across {fam.compositionSweep.systems} held-out copolymer systems, moving the composition from one
            monomer to the other in {fam.compositionSweep.points} steps, the network's Tg curve was monotonic in{' '}
            {Math.round(fam.compositionSweep.network * 100)}% of systems, the trees' in {Math.round(fam.compositionSweep.trees * 100)}%, and the
            measured data in {Math.round(fam.compositionSweep.measured * 100)}%. This is not re-run live: the per-composition input rows
            were not exported.
          </p>
        </section>
      )}

      {fam.bandGap && (
        <section className="card fam-section">
          <h2>Trend check: band gap</h2>
          <p className="note">
            Predictions fall as calculated band gap rises (Spearman {fam.bandGap.networkSpearman.toFixed(2)}; the calculated values
            themselves: {fam.bandGap.calculatedSpearman.toFixed(2)}). Band gap is not an input, so this is a consistency check, not
            evidence of physics.
            {trained && heldout && <> The network you just trained: Spearman <strong>{spearman(preds, heldout.map((h) => h.bandGap)).toFixed(2)}</strong>.</>}
          </p>
          {trained && heldout && (
            <XYChart xLabel="Calculated band gap (eV)" yLabel="Predicted dielectric constant" logY lines={[]}
              dots={heldout.map((h, i) => ({ x: h.bandGap, y: preds[i] }))} />
          )}
        </section>
      )}

      {family === 'insulators' && heldout && (trained
        ? <HeldoutRanking fam={fam} heldout={heldout} predictions={preds} />
        : <section className="card fam-section"><h2>Rank compounds the network never trained on</h2><p className="note">Train the network first; the ranking uses the network you train.</p></section>)}

      {family === 'materials' && <AlloyGenerator fam={fam} net={net} band={band} status={status} whenTrained={whenTrained} folds={FOLDS} liveMae={live?.mae} />}

      <section className="caveat">
        <h2>How this can be wrong</h2>
        <div className="split">
          <ul>{caveats.map((c) => <li key={c}>{c}</li>)}</ul>
          {trained && (
            <figure>
              <XYChart xLabel={`Value in the data${fam.unit ? ` (${fam.unit})` : ', log10'}`} yLabel="Predicted" lines={[]}
                dots={nnData.test.y.map((v, i) => ({ x: v, y: preds[i] }))} />
              <figcaption>The {fam.data.test} held-out rows: value in the data against the network you just trained.</figcaption>
            </figure>
          )}
        </div>
      </section>

      <footer>
        <p>Data: {fam.data.citation}; {fam.data.source}; licence: {fam.data.licence}. Files: public/model/{family}/ (family.json, nn_data.json{family === 'insulators' ? ', heldout.json' : ''}), built by scripts/build_family_files.mjs.</p>
        {onBack && <p><button type="button" className="chip" onClick={onBack}>Open the live concrete designer</button></p>}
      </footer>
    </div>
  )
}
