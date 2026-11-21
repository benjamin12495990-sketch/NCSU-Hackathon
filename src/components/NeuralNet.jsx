import { useEffect, useRef, useState } from 'react'
import { createTrainer } from '../lib/nn.js'
import { featurize } from '../lib/featurize.js'
import { mixName } from '../lib/name.js'
import { busy } from '../lib/busy.js'
import './neuralnet.css'

// One pass per step at first, with a short pause so the early learning is
// visible, then four passes per step with no pause. A timer drives it, not
// the screen's frame rate, so it takes about three seconds on any device.
const passesThisStep = (epoch) => (epoch < 24 ? 1 : 4)
const pauseAfter = (epoch) => (epoch < 24 ? 30 : 0)

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

// Trains a neural network from random weights in the browser when the button
// is pressed, and shows what it has learned on lab results it never saw.
export default function NeuralNet({ data, result }) {
  const [nnData, setNnData] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [history, setHistory] = useState([])
  const [status, setStatus] = useState('idle') // idle, training or done
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

  const train = () => {
    clearTimeout(frame.current)
    const t = createTrainer(nnData)
    trainer.current = t
    setHistory([...t.history])
    setStatus('training')
    busy.training = true // the 3D view pauses its redraws until training ends
    const tick = () => {
      t.runEpochs(passesThisStep(t.epoch))
      setHistory([...t.history])
      if (t.done) {
        busy.training = false
        setStatus('done')
      }
      else frame.current = setTimeout(tick, pauseAfter(t.epoch))
    }
    frame.current = setTimeout(tick, 0)
  }

  if (!data) return null
  const m = data.metrics
  const t = trainer.current
  const first = history[0], last = history[history.length - 1]
  const ageIndex = nnData ? nnData.features.indexOf('age') : -1
  // Eight held-out lab results, spread through the test set, to show measured against predicted.
  const samples = status === 'done' && nnData
    ? Array.from({ length: 8 }, (_, k) => Math.floor((k * nnData.test.X.length) / 8)).map((i) => {
        const row = nnData.test.X[i]
        return { name: mixName(row.slice(0, 7)), age: row[ageIndex], measured: nnData.test.y[i], predicted: t.predict(row) }
      })
    : []
  const sameFeatures = nnData && nnData.features.join() === data.spec.features.join()
  const ranked = status === 'done' && result?.top?.length && sameFeatures
    ? result.top.slice(0, 5).map((c) => ({ name: c.name, trees: c.strength, net: t.predict(featurize(c.mix, result.ageDays, data.spec)) }))
    : []

  return (
    <section className="nn">
      <h2>Train a neural network yourself</h2>
      <p className="note">
        Press the button and a neural network starts from random numbers and learns to predict strength
        from the {m.train_rows} training lab results, here in your browser. Nothing pre-trained is
        downloaded. The curve is its error on the {m.test_rows} lab results from {m.test_mixes} mixes that
        are held out, which it never trains on.
      </p>
      <div className="nnrow">
        <button type="button" className="chip nnbtn" onClick={train} disabled={!nnData || status === 'training'}>
          {status === 'idle' ? 'Train the network' : status === 'training' ? 'Training…' : 'Train again from scratch'}
        </button>
        {loadError && <span className="note">The training data did not load: {loadError}</span>}
        {last && (
          <div className="nnstats">
            <div><span className="label">Pass</span><span className="big">{last.epoch} <small>of {t.totalEpochs}</small></span></div>
            <div><span className="label">Error on held-out mixes</span><span className="big">{last.mae.toFixed(1)} <small>MPa</small></span></div>
            <div><span className="label">Before any training</span><span className="big">{first.mae.toFixed(1)} <small>MPa</small></span></div>
            <div><span className="label">Tree model, same mixes</span><span className="big">{m.test_mae_MPa.toFixed(1)} <small>MPa</small></span></div>
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
          {status === 'done' && (
            <div className="nntables">
              <p className="note">
                It went from {first.mae.toFixed(1)} MPa off to <strong>{last.mae.toFixed(1)} MPa off</strong> (R²{' '}
                {last.r2.toFixed(2)}) in {last.epoch} passes, with {t.weights.toLocaleString()} weights in layers
                of {t.layers.join(', ')}. The tree model is still more accurate on the same mixes
                ({m.test_mae_MPa.toFixed(1)} MPa), so the designer keeps using the trees and the network is a
                second opinion.
              </p>
              <div className="tablewrap">
                <table>
                  <thead>
                    <tr><th>Held-out lab result (never trained on)</th><th className="num">Age</th><th className="num">Measured</th><th className="num">Network predicts</th><th className="num">Miss</th></tr>
                  </thead>
                  <tbody>
                    {samples.map((s, i) => (
                      <tr key={i}>
                        <td>{s.name}</td>
                        <td className="num">{s.age} d</td>
                        <td className="num">{s.measured.toFixed(1)} MPa</td>
                        <td className="num strong">{s.predicted.toFixed(1)} MPa</td>
                        <td className="num">{Math.abs(s.predicted - s.measured).toFixed(1)}</td>
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
                          <td className="num strong">{r.net.toFixed(1)} MPa</td>
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
      <p className="note small">
        The network is written from scratch in JavaScript (src/lib/nn.js): two hidden layers, ReLU, the Adam
        optimiser, squared-error loss. It gets the tree model's nine inputs plus the logarithm of age. The
        starting weights use a fixed seed, so every press gives the same run. Its settings were chosen
        after a few trial runs scored on these same held-out mixes, so its final error is slightly
        flattering. The eight lab results in the table are spread evenly through the held-out set, not
        picked for being good.
      </p>
    </section>
  )
}
