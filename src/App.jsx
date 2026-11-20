import { useEffect, useState } from 'react'
import { loadModel } from './lib/loadModel.js'
import { analyze } from './lib/analyze.js'

const EXAMPLES = ['YBa2Cu3O7', 'MgB2', 'La1.85Sr0.15CuO4']
const REPO = 'https://github.com/benjamin12495990-sketch/NCSU-Hackathon'
const K = (v) => `${v.toFixed(1)} K`

// The single page: loads the model once, then runs every prediction in the
// browser each time the user submits a formula.
export default function App() {
  const [data, setData] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [text, setText] = useState('YBa2Cu3O7')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    loadModel().then(setData, (e) => setLoadError(e.message))
  }, [])

  const run = (formula) => {
    setText(formula)
    try {
      setResult(analyze(formula, data))
      setError('')
    } catch (e) {
      setResult(null)
      setError(e.message)
    }
  }

  const m = data?.metrics
  return (
    <main>
      <header>
        <p className="eyebrow">Engineer the Tools of Scientific Discovery</p>
        <h1>Superconductor Candidate Generator</h1>
        <p className="lede">
          Type a formula to predict its superconducting critical temperature (T<sub>c</sub>) and get
          new, untested compositions ranked by predicted T<sub>c</sub>. For materials science grad
          students deciding which compositions deserve weeks of synthesis or simulation time.
        </p>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (data) run(text)
        }}
      >
        <label htmlFor="formula">Chemical formula</label>
        <div className="row">
          <input
            id="formula"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. YBa2Cu3O7"
            autoComplete="off"
            spellCheck="false"
          />
          <button type="submit" disabled={!data}>
            {data ? 'Predict' : 'Loading model…'}
          </button>
        </div>
        <div className="examples">
          <span>Try:</span>
          {EXAMPLES.map((ex) => (
            <button type="button" key={ex} className="chip" disabled={!data} onClick={() => run(ex)}>
              {ex}
            </button>
          ))}
        </div>
      </form>

      {loadError && <p className="error" role="alert">The model files did not load: {loadError}</p>}
      {error && <p className="error" role="alert">{error}</p>}

      {result && (
        <>
          <section className="card">
            <h2>{result.formula}</h2>
            <div className="stats">
              <div>
                <span className="label">Predicted T<sub>c</sub></span>
                <span className="big">{K(result.tc)}</span>
              </div>
              <div>
                <span className="label">Measured T<sub>c</sub> in dataset</span>
                <span className="big">{result.known ? K(result.known.tc) : 'not in dataset'}</span>
              </div>
            </div>
            <p className="note">
              {result.known
                ? `This composition is in the dataset as ${result.known.formula}, so the model may have seen it in training.`
                : `Not in the dataset. Nearest known material: ${result.nearestFormula} (measured ${K(result.nearestTc)}).`}
              {result.lowConfidence && ' It is far from the training data, so treat the prediction as low confidence.'}
            </p>
          </section>

          <section>
            <h2>New candidates</h2>
            <p className="note">
              {result.generated} one-element swaps generated; {result.removedKnown} removed because
              they are already in the dataset. Top {result.top.length} of the rest by predicted T
              <sub>c</sub>:
            </p>
            {result.top.length === 0 ? (
              <p className="note">No new candidates: every swap is already in the dataset or no similar element exists.</p>
            ) : (
              <div className="tablewrap">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Candidate</th>
                      <th>Swap</th>
                      <th className="num">Predicted T<sub>c</sub></th>
                      <th>Nearest known material</th>
                      <th className="num">Its measured T<sub>c</sub></th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.top.map((c, i) => (
                      <tr key={c.formula}>
                        <td>{i + 1}</td>
                        <td className="formula">{c.formula}</td>
                        <td>{c.swap}</td>
                        <td className="num">{K(c.tc)}</td>
                        <td className="formula">{c.nearestFormula}</td>
                        <td className="num">{K(c.nearestTc)}</td>
                        <td>
                          <span className={c.lowConfidence ? 'flag low' : 'flag ok'}>
                            {c.lowConfidence ? 'Low confidence' : 'Near known data'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}

      <section className="caveat">
        <h2>How this can be wrong</h2>
        {m ? (
          <ul>
            <li>
              On {m.test_rows.toLocaleString()} held-out materials the model is off by{' '}
              <strong>{m.test_mae_K.toFixed(1)} K on average</strong> (mean absolute error; R²{' '}
              {m.test_r2.toFixed(2)}). The split is random, so close relatives of test materials
              were in training and the real error on new chemistry is larger.
            </li>
            <li>
              Candidates are <strong>untested predictions, not discoveries</strong>. Nothing here
              checks whether a candidate can be made, is stable, or superconducts at all.
            </li>
            <li>
              The dataset contains only superconductors, so the model cannot predict “not a
              superconductor”. It is weakest on compositions unlike its training data; those rows
              are flagged low confidence.
            </li>
            <li>
              The model sees only element fractions. It knows nothing about crystal structure,
              pressure or doping order.
            </li>
          </ul>
        ) : (
          <p className="note">Loading test metrics…</p>
        )}
      </section>

      <footer>
        {m && (
          <p>
            Model: {m.model}, trained {m.trained_at.slice(0, 10)} on {m.train_rows.toLocaleString()} rows.
            Runs entirely in your browser.
          </p>
        )}
        <p>
          Data: Hamidieh, K. (2018). Superconductivty Data. UCI Machine Learning Repository.{' '}
          <a href="https://doi.org/10.24432/C53P47">doi.org/10.24432/C53P47</a> ·{' '}
          <a href={REPO}>Source code on GitHub</a>
        </p>
      </footer>
    </main>
  )
}
