import { useEffect, useRef, useState } from 'react'
import { createTrainer } from '../lib/nn.js'
import { featurize } from '../lib/featurize.js'
import { predict } from '../lib/predict.js'
import { mixName } from '../lib/name.js'
import { busy } from '../lib/busy.js'
import { AGES, waterSteps, trend } from '../lib/sweep.js'
import './neuralnet.css'

// One pass per step at first, with a short pause so the early learning is
// visible, then four passes per step with no pause. A timer drives it, not
// the screen's frame rate, so it takes about three seconds on any device.
const passesThisStep = (epoch) => (epoch < 24 ? 1 : 4)
const pauseAfter = (epoch) => (epoch < 24 ? 30 : 0)
const FOLDS = 3 // extra training runs used to measure the interval
const LEVEL = 0.9 // the interval is meant to hold nine times in ten

// The learning curve: error on the held-out mixes after each pass over the
// training data. The dashed line is the tree model the designer uses.
function Curve({ history, total, treeMae }) {
  const w = 520, h = 240, left = 44, bottom = 34, top = 12, right = 12
  const maxY = Math.max(18, Math.ceil(history[0]?.mae ?? 18))
  // Most of the learning happens in the first passes, so the pass axis is
  // logarithmic: equal steps for 1, 3, 10, 30, 100 and 300 passes.
  const x = (e) => left + (Math.log(1 + e) / Math.log(1 + total)) * (w - left - right)
  const y = (v) => top + (1 - Math.min(v, maxY) / maxY) * (h - top - bottom)
  const points = history.map((p) => `${x(p.epoch).toFixed(1)},${y(p.mae).toFixed(1)}`).join(' ')
  const last = history[history.length - 1]
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="nncurve" role="img"
      aria-label="Error on held-out mixes falling as the network trains">
      {[0, 6, 12, 18].filter((t) => t <= maxY).map((t) => (
        <g key={t}>
          <line x1={left} y1={y(t)} x2={w - right} y2={y(t)} className="grid" />
          <text x={left - 6} y={y(t) + 4} textAnchor="end">{t}</text>
        </g>
      ))}
      {[0, 1, 3, 10, 30, 100, 300].filter((e) => e <= total).map((e) => (
        <text key={e} x={x(e)} y={h - bottom + 16} textAnchor="middle">{e}</text>
      ))}
      <line x1={left} y1={y(treeMae)} x2={w - right} y2={y(treeMae)} className="tree" />
      <text x={left + 6} y={y(treeMae) + 14}>tree model: {treeMae.toFixed(1)} MPa</text>
      {history.length > 1 && <polyline points={points} className="line" />}
      {last && <circle cx={x(last.epoch)} cy={y(last.mae)} r="4" className="dot" />}
      <text x={(w + left) / 2} y={h - 2} textAnchor="middle">Passes over the training data (log scale)</text>
      <text x={12} y={(h - bottom) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(h - bottom) / 2})`}>Error on held-out mixes (MPa)</text>
    </svg>
  )
}

// One sweep: the network's prediction (solid line, with its interval as a
// band) and the tree model's (dashed) as a single input changes.
function SweepChart({ points, band, xLabel, logX, ticks }) {
  const w = 400, h = 230, left = 44, bottom = 34, top = 10, right = 10
  const xs = points.map((p) => p.x)
  const fx = (v) => (logX ? Math.log(v) : v)
  const x0 = fx(Math.min(...xs)), x1 = fx(Math.max(...xs))
  const maxY = Math.ceil(Math.max(...points.map((p) => Math.max(p.net + band, p.tree))) / 10) * 10
  const x = (v) => left + ((fx(v) - x0) / (x1 - x0)) * (w - left - right)
  const y = (v) => top + (1 - Math.max(0, Math.min(v, maxY)) / maxY) * (h - top - bottom)
  const path = (key) => points.map((p) => `${x(p.x).toFixed(1)},${y(p[key]).toFixed(1)}`).join(' ')
  const area = [...points.map((p) => `${x(p.x).toFixed(1)},${y(p.net + band).toFixed(1)}`),
    ...[...points].reverse().map((p) => `${x(p.x).toFixed(1)},${y(p.net - band).toFixed(1)}`)].join(' ')
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="nncurve" role="img" aria-label={`Predicted strength against ${xLabel}`}>
      {[0, maxY / 2, maxY].map((t) => (
        <g key={t}>
          <line x1={left} y1={y(t)} x2={w - right} y2={y(t)} className="grid" />
          <text x={left - 6} y={y(t) + 4} textAnchor="end">{t}</text>
        </g>
      ))}
      {ticks.map((t) => <text key={t} x={x(t)} y={h - bottom + 16} textAnchor="middle">{t}</text>)}
      {band > 0 && <polygon points={area} className="band" />}
      <polyline points={path('tree')} className="tree" fill="none" />
      <polyline points={path('net')} className="line" />
      <text x={(w + left) / 2} y={h - 2} textAnchor="middle">{xLabel}</text>
      <text x={12} y={(h - bottom) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(h - bottom) / 2})`}>Predicted strength (MPa)</text>
    </svg>
  )
}

// Trains a neural network from random weights in the browser when the button
// is pressed, measures how far off it tends to be, and tests whether it has
// learned two trends every concrete engineer knows.
export default function NeuralNet({ data, result }) {
  const [nnData, setNnData] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [history, setHistory] = useState([])
  const [status, setStatus] = useState('idle') // idle, training, measuring or done
  const [foldsDone, setFoldsDone] = useState(0)
  const [band, setBand] = useState(null) // { halfWidth, coverage, errors }
  const trainer = useRef(null)
  const frame = useRef(0)

  useEffect(() => {
    fetch(import.meta.env.BASE_URL + 'model/nn_data.json')
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then(setNnData, (e) => setLoadError(e.message))
    return () => {
      clearTimeout(frame.current)
      busy.training = false
    }
  }, [])

  // How wide is "probably within"? Train the same design three more times,
  // each time leaving out a third of the training mixes and scoring on them.
  // The interval is the size of miss that 90% of those unseen results stayed
  // inside (cross-validation on training rows; the held-out mixes are not used).
  const measure = (main) => {
    const groups = [...new Set(nnData.train.g)].sort((a, b) => a - b)
    const foldOf = new Map(groups.map((id, i) => [id, i % FOLDS]))
    const errors = []
    let k = 0, fold = null, left = null
    const start = () => {
      const tr = { X: [], y: [] }
      left = { X: [], y: [] }
      nnData.train.X.forEach((row, i) => {
        const part = foldOf.get(nnData.train.g[i]) === k ? left : tr
        part.X.push(row); part.y.push(nnData.train.y[i])
      })
      fold = createTrainer({ features: nnData.features, train: tr, test: left })
    }
    const tick = () => {
      fold.runEpochs(30)
      if (!fold.done) return void (frame.current = setTimeout(tick, 0))
      left.X.forEach((row, i) => errors.push(Math.abs(fold.predict(row) - left.y[i])))
      k++
      setFoldsDone(k)
      if (k < FOLDS) {
        start()
        return void (frame.current = setTimeout(tick, 0))
      }
      errors.sort((a, b) => a - b)
      const halfWidth = errors[Math.min(errors.length - 1, Math.ceil(LEVEL * (errors.length + 1)) - 1)]
      const inside = nnData.test.X.filter((row, i) => Math.abs(main.predict(row) - nnData.test.y[i]) <= halfWidth).length
      setBand({ halfWidth, coverage: inside / nnData.test.X.length, errors: errors.length })
      busy.training = false
      setStatus('done')
    }
    start()
    frame.current = setTimeout(tick, 0)
  }

  const train = () => {
    clearTimeout(frame.current)
    const t = createTrainer(nnData)
    trainer.current = t
    setHistory([...t.history])
    setBand(null)
    setFoldsDone(0)
    setStatus('training')
    busy.training = true // the 3D view pauses its redraws until training ends
    const tick = () => {
      t.runEpochs(passesThisStep(t.epoch))
      setHistory([...t.history])
      if (t.done) {
        setStatus('measuring')
        measure(t)
      } else frame.current = setTimeout(tick, pauseAfter(t.epoch))
    }
    frame.current = setTimeout(tick, 0)
  }

  // The "Train the neural network" button beside "Design mixes" asks for a run.
  useEffect(() => {
    if (!nnData) return undefined
    window.addEventListener('train-network', train)
    return () => window.removeEventListener('train-network', train)
  }) // re-bound every render so it always starts a run with the current data

  if (!data) return null
  const m = data.metrics
  const t = trainer.current
  const first = history[0], last = history[history.length - 1]
  const trained = (status === 'measuring' || status === 'done') && t && nnData
  const ageIndex = nnData ? nnData.features.indexOf('age') : -1
  const q = band ? band.halfWidth : 0
  // Eight held-out lab results, spread through the test set, to show measured against predicted.
  const samples = trained
    ? Array.from({ length: 8 }, (_, k) => Math.floor((k * nnData.test.X.length) / 8)).map((i) => {
        const row = nnData.test.X[i]
        return { name: mixName(row.slice(0, 7)), age: row[ageIndex], measured: nnData.test.y[i], predicted: t.predict(row) }
      })
    : []
  const sameFeatures = nnData && nnData.features.join() === data.spec.features.join()
  const ranked = trained && result?.top?.length && sameFeatures
    ? result.top.slice(0, 5).map((c) => ({ name: c.name, trees: c.strength, net: t.predict(featurize(c.mix, result.ageDays, data.spec)) }))
    : []

  // Sweep test. The mix is the top generated one if there is a result, or a
  // tested mix from the middle of the dataset otherwise.
  let sweep = null
  if (trained && sameFeatures) {
    const refMix = result?.top?.[0]?.mix ?? data.known.entries[Math.floor(data.known.entries.length / 2)].mix
    const refAge = result?.ageDays ?? 28
    const both = (mix, age) => {
      const x = featurize(mix, age, data.spec)
      return { net: t.predict(x), tree: predict(data.model, x) }
    }
    const steps = waterSteps(refMix, data.space)
    const water = steps ? steps.map((s) => ({ x: s.x, ...both(s.mix, refAge) })) : null
    const age = AGES.map((a) => ({ x: a, ...both(refMix, a) }))
    // The same two sweeps on about 26 tested mixes spread through the dataset
    // (the mixes npm run sweep uses), to see how often each model gets the
    // direction right, and whether the network ever predicts a negative strength.
    const all = { mixes: 0, netWater: 0, netAge: 0, treeWater: 0, treeAge: 0, negative: 0 }
    const stride = Math.floor(data.known.entries.length / 24)
    for (let i = 0; i < data.known.entries.length; i += stride) {
      const mix = data.known.entries[i].mix, ws = waterSteps(mix, data.space)
      if (!ws) continue
      const w = ws.map((s) => both(s.mix, 28)), a = AGES.map((d) => both(mix, d))
      all.mixes++
      if (w[w.length - 1].net < w[0].net) all.netWater++
      if (a[a.length - 1].net > a[0].net) all.netAge++
      if (w[w.length - 1].tree < w[0].tree) all.treeWater++
      if (a[a.length - 1].tree > a[0].tree) all.treeAge++
      if ([...w, ...a].some((p) => p.net < 0)) all.negative++
    }
    sweep = {
      name: mixName(refMix), refAge, water, age, all,
      waterTrend: water && trend(water.map((p) => p.net)),
      ageTrend: trend(age.map((p) => p.net)),
    }
  }

  return (
    <section className="nn" id="train">
      <h2>Train a neural network yourself</h2>
      <p className="note">
        Press the button and a neural network starts from random numbers and learns to predict strength
        from the {m.train_rows} training lab results, here in your browser. Nothing pre-trained is
        downloaded. The curve is its error on the {m.test_rows} lab results from {m.test_mixes} mixes that
        are held out, which it never trains on.
      </p>
      <div className="nnrow">
        <button type="button" className="chip nnbtn" onClick={train}
          disabled={!nnData || status === 'training' || status === 'measuring'}>
          {status === 'idle' ? 'Train the network' : status === 'training' ? 'Training…'
            : status === 'measuring' ? `Measuring its uncertainty: run ${Math.min(FOLDS, foldsDone + 1)} of ${FOLDS}…` : 'Train again from scratch'}
        </button>
        {loadError && <span className="note">The training data did not load: {loadError}</span>}
        {last && (
          <div className="nnstats">
            <div><span className="label">Pass</span><span className="big">{last.epoch} <small>of {t.totalEpochs}</small></span></div>
            <div><span className="label">Error on held-out mixes</span><span className="big">{last.mae.toFixed(1)} <small>MPa</small></span></div>
            <div><span className="label">Before any training</span><span className="big">{first.mae.toFixed(1)} <small>MPa</small></span></div>
            <div><span className="label">Tree model, same mixes</span><span className="big">{m.test_mae_MPa.toFixed(1)} <small>MPa</small></span></div>
            {band && <div><span className="label">90% interval</span><span className="big">± {q.toFixed(1)} <small>MPa</small></span></div>}
          </div>
        )}
      </div>

      {history.length > 0 && (
        <div className="nnbody">
          <figure>
            <Curve history={history} total={t.totalEpochs} treeMae={m.test_mae_MPa} />
            <figcaption>
              Lower is better. Each point is the average miss on the held-out lab results after one more pass.
            </figcaption>
          </figure>
          {trained && (
            <div className="nntables">
              <p className="note">
                It went from {first.mae.toFixed(1)} MPa off to <strong>{last.mae.toFixed(1)} MPa off</strong> (R²{' '}
                {last.r2.toFixed(2)}) in {last.epoch} passes, with {t.weights.toLocaleString()} weights in layers
                of {t.layers.join(', ')}. The tree model is still more accurate on the same mixes
                ({m.test_mae_MPa.toFixed(1)} MPa), so the designer keeps using the trees and the network is a
                second opinion.
                {band && (
                  <> Its <strong>90% interval is ± {q.toFixed(1)} MPa</strong>: the size of miss that nine in ten
                  results stayed inside when the same design was trained {FOLDS} more times and scored on{' '}
                  {band.errors} training results it had not seen. On the held-out results the interval held
                  for <strong>{Math.round(band.coverage * 100)} in 100</strong>.</>
                )}
              </p>
              <div className="tablewrap">
                <table>
                  <thead>
                    <tr>
                      <th>Held-out lab result (never trained on)</th><th className="num">Age</th>
                      <th className="num">Measured</th><th className="num">Network predicts</th>
                      <th className="num">Miss</th>{band && <th>Within ± {q.toFixed(1)}</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {samples.map((s, i) => (
                      <tr key={i}>
                        <td>{s.name}</td>
                        <td className="num">{s.age} d</td>
                        <td className="num">{s.measured.toFixed(1)} MPa</td>
                        <td className="num strong">{s.predicted.toFixed(1)} MPa</td>
                        <td className="num">{Math.abs(s.predicted - s.measured).toFixed(1)}</td>
                        {band && <td>{Math.abs(s.predicted - s.measured) <= q ? 'Yes' : 'No'}</td>}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {ranked.length > 0 && (
                <div className="tablewrap">
                  <table>
                    <thead>
                      <tr><th>Generated mix (top of the ranking)</th><th className="num">Trees expect</th><th className="num">Network predicts</th><th className="num">Difference</th></tr>
                    </thead>
                    <tbody>
                      {ranked.map((r) => (
                        <tr key={r.name}>
                          <td>{r.name}</td>
                          <td className="num">{r.trees.toFixed(1)} MPa</td>
                          <td className="num strong">{r.net.toFixed(1)}{band && ` ± ${q.toFixed(1)}`} MPa</td>
                          <td className="num">{(r.net - r.trees > 0 ? '+' : '−') + Math.abs(r.net - r.trees).toFixed(1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {sweep && (
        <div className="nnsweep">
          <h3>Sweep test: did it learn two things every concrete engineer knows?</h3>
          <p className="note">
            One mix, <strong>{sweep.name}</strong>, with one thing changed at a time. Solid line: the network you
            just trained{band && ', with its 90% interval as the shaded band'}. Dashed line: the tree model.
            Nobody told either model these rules; they can only have come from the lab results.
          </p>
          <div className="nnsweeps">
            {sweep.water && (
              <figure>
                <SweepChart points={sweep.water} band={q} xLabel={`Water ÷ binder, at ${sweep.refAge} days`}
                  ticks={[sweep.water[0].x, sweep.water[10].x, sweep.water[20].x].map((v) => +v.toFixed(2))} />
                <figcaption>
                  <strong>More water should mean weaker concrete.</strong> The network's prediction goes from{' '}
                  {sweep.waterTrend.first.toFixed(1)} to {sweep.waterTrend.last.toFixed(1)} MPa and falls in{' '}
                  {sweep.waterTrend.down} of {sweep.waterTrend.steps} steps
                  {sweep.waterTrend.down === sweep.waterTrend.steps ? ': it learned the trend.'
                    : sweep.waterTrend.last < sweep.waterTrend.first ? ': the trend is right overall, with some steps the wrong way.'
                    : ': it did NOT learn this trend for this mix.'}
                </figcaption>
              </figure>
            )}
            <figure>
              <SweepChart points={sweep.age} band={q} xLabel="Curing age in days (log scale)" logX ticks={[1, 7, 28, 90, 365]} />
              <figcaption>
                <strong>Older concrete should be stronger.</strong> From 1 to 365 days the network's prediction goes
                from {sweep.ageTrend.first.toFixed(1)} to {sweep.ageTrend.last.toFixed(1)} MPa and rises in{' '}
                {sweep.ageTrend.up} of {sweep.ageTrend.steps} steps
                {sweep.ageTrend.up === sweep.ageTrend.steps ? ': it learned the trend.'
                  : sweep.ageTrend.last > sweep.ageTrend.first ? ': the trend is right overall, with some steps the wrong way.'
                  : ': it did NOT learn this trend for this mix.'}
              </figcaption>
            </figure>
          </div>
          <p className="note">
            Across {sweep.all.mixes} tested mixes spread through the dataset, more water lowered the network's
            prediction on <strong>{sweep.all.netWater}</strong> and more age raised it on{' '}
            <strong>{sweep.all.netAge}</strong> (tree model: {sweep.all.treeWater} and {sweep.all.treeAge}).
            {sweep.all.negative > 0 && ` On ${sweep.all.negative} of them the network predicted a strength below zero somewhere, usually at one day old. That is physically impossible, and nothing in the network stops it.`}
          </p>
          <p className="note small">
            These sentences are computed from the run you just watched, not written in advance. Changing only
            the water is a thought experiment: a real mix would also rebalance its aggregates. Following a
            trend is not the same as understanding it: the network has no idea what water or cement is.
          </p>
        </div>
      )}

      <p className="note small">
        The network is written from scratch in JavaScript (src/lib/nn.js): two hidden layers, ReLU, the Adam
        optimiser, squared-error loss. It gets the tree model's nine inputs plus the logarithm of age. The
        starting weights use a fixed seed, so every press gives the same run. Its settings were chosen
        after a few trial runs scored on these same held-out mixes, so its final error is slightly
        flattering. The eight lab results in the table are spread evenly through the held-out set, not
        picked for being good. The interval is the same width everywhere, which is crude: misses are
        larger for young concrete. One held-out split is a thin basis for ranking two models, so we also
        ran both on eight other splits by mix: an ensemble of five of these networks averaged 4.2 MPa and
        the trees 4.3 MPa (difference −0.1 MPa, 95% confidence interval −0.5 to +0.3), with the ensemble
        ahead on four splits, behind on three and level on one. That is a tie, not a win; the numbers are
        in research/nn_vs_trees.json.
      </p>
    </section>
  )
}
