import { useEffect, useRef, useState } from 'react'
import { generateAsync, NEIGHBOURS, SUPPORT_NEEDED, NOT_NEW, TOO_FAR } from '../lib/alloyGen.js'

// "Generate untested alloys" on the alloy page. It proposes compositions inside tested element
// sets (src/lib/alloyGen.js), predicts them with the network trained on this page, and shows the
// nearest tested alloys as the evidence. Every number comes from alloys.json, family.json, the
// network trained here, or the generator's own counts; none is typed here.

const ROWS = 10 // table rows shown
const SLICE_MS = 40 // give the browser a turn between element sets after this long

const pct = (x) => `${Math.round(x * 100)} at.%`
const hv = (v) => `${Math.round(v)} HV`
const signed = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(Math.round(v))} HV`

// One switch-plus-slider row, the same markup as the concrete form (src/components/Requirements.jsx).
function Row({ name, on, onToggle, min, max, step, value, onValue, text, offText, disabled = false }) {
  return (
    <div className={on ? 'req' : 'req off'}>
      <label className="switch">
        <input type="checkbox" checked={on} disabled={disabled} onChange={(e) => onToggle(e.target.checked)} /> {name}
      </label>
      {onValue ? (
        <input type="range" aria-label={name} min={min} max={max} step={step} value={value} disabled={!on || disabled}
          onChange={(e) => onValue(Number(e.target.value))} />
      ) : <span />}
      <span className="reqvalue">{on ? text : offText}</span>
    </div>
  )
}

// Why nothing was kept, in plain words, from the generator's own counts.
function nothingKept(stats, target) {
  const ex = stats.excluded
  if (stats.sets.used === 0) {
    return `No tested element set could be varied with these settings (${stats.sets.notAllowed} sets use an element that is switched off, ` +
      `${stats.sets.routeNotTested} were never tested by this route, ${stats.sets.cappedOut} are ruled out by the maximum fractions, ` +
      `${stats.sets.singleComposition} have only one tested composition), so there was nothing to generate.`
  }
  if (stats.bestCautious == null) {
    return 'Every composition drawn was either too close to a tested alloy, too far from all of them, a copy of a tested alloy with an element left out, ' +
      'or predicted at the edge of the training range, so the network made no usable prediction.'
  }
  if (target != null && stats.bestCautious < target) {
    return `No candidate reaches ${hv(target)} by the lower edge of its 90% range. The highest lower edge any candidate reached is ` +
      `${hv(stats.bestCautious)} (the network's cap allows at most ${hv(stats.ceilingCautious)}). The network does not support a target this high ` +
      'inside the tested element sets; lower the target, allow other elements, or switch off the route rule.'
  }
  if (ex.unsupported > 0) {
    return `${ex.unsupported} candidates reach ${hv(target)} on the network's prediction, but for each of them fewer than ${SUPPORT_NEEDED} of its ` +
      `${NEIGHBOURS} nearest tested alloys measured at least ${hv(target)}. The measured evidence does not back them, so none is shown. ` +
      'Switch off "Only candidates whose nearest tested alloys also reached the target" to see them anyway.'
  }
  return 'No candidate passed every rule.'
}

export default function AlloyGenerator({ fam, net, band, status, whenTrained, folds }) {
  const [data, setData] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [target, setTarget] = useState({ on: true, value: 400 })
  const [proc, setProc] = useState('CAST')
  const [routeRule, setRouteRule] = useState(true)
  const [supported, setSupported] = useState(true)
  const [allowed, setAllowed] = useState(null) // { El: { on, max } }
  const [waiting, setWaiting] = useState(false) // pressed Generate; runs when the network is ready
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null) // { net, res, settings }
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    const ctl = new AbortController()
    fetch(`${import.meta.env.BASE_URL}model/materials/alloys.json`, { signal: ctl.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`alloys.json: HTTP ${r.status}`))))
      .then((d) => {
        setData(d)
        // Elements that occur in tested element sets the generator can use (own column, not pooled).
        const own = new Set(d.elements)
        const used = new Set(d.rows.flatMap((r) => Object.keys(r.frac)).filter((el) => own.has(el)))
        setAllowed(Object.fromEntries(d.elements.filter((el) => used.has(el)).map((el) => [el, { on: true, max: 1 }])))
      })
      .catch((e) => { if (e.name !== 'AbortError') setLoadError(e.message) })
    return () => { alive.current = false; ctl.abort() }
  }, [])

  const ready = net != null && status === 'done'
  const intervalFor = (b) => (b
    ? { q: b.q, qSource: `measured live in this browser (${folds} extra networks on parts of the training rows)` }
    : { q: fam.interval.q, qSource: 'from the results file (the live measure has not run)' })
  const { q, qSource } = intervalFor(band)

  const settingsNow = () => ({
    targetHV: target.on ? target.value : null,
    proc,
    routeMustBeTested: routeRule,
    onlySupported: target.on && supported,
    allowed: Object.fromEntries(Object.entries(allowed).filter(([, a]) => a.on).map(([el, a]) => [el, a.max])),
    limit: ROWS,
    seed: 1,
  })

  const run = async (settings, model, b) => {
    const iv = intervalFor(b)
    setRunning(true)
    const res = await generateAsync(settings, data, (row) => model.predict(row), iv.q, { sliceMs: SLICE_MS })
    if (!alive.current) return
    setResult({ net: model, res, settings, ...iv })
    setRunning(false)
  }

  // Pressed before the network was trained: the page's own train flow runs first (including the
  // live interval), then the generator runs with the settings chosen at the press.
  const press = () => {
    const settings = settingsNow()
    if (ready) return run(settings, net, band)
    setWaiting(true)
    whenTrained((model, b) => { setWaiting(false); run(settings, model, b) })
  }

  if (loadError) return <section className="card fam-section"><h2>Generate untested alloys</h2><p className="error">The generator file did not load: {loadError}</p></section>
  if (!data || !allowed) return <section className="card fam-section"><h2>Generate untested alloys</h2><p className="note">Loading the tested alloys…</p></section>

  const shown = result && result.net === net ? result : null
  const stats = shown?.res.stats
  const ex = stats?.excluded
  const list = shown?.res.candidates ?? []
  const set = (el, patch) => setAllowed({ ...allowed, [el]: { ...allowed[el], ...patch } })
  const onCount = Object.values(allowed).filter((a) => a.on).length
  const capped = Object.values(allowed).filter((a) => a.on && a.max < 1).length
  const familiar = Math.round(fam.test.network), unseen = Math.round(fam.wholeSystems.network)
  const busyText = status === 'training' ? 'Training the network first…' : status === 'measuring' ? 'Measuring its uncertainty first…' : 'Generating…'

  return (
    <section className="card fam-section fam-gen" id="generate-alloys">
      <h2>Generate untested alloys</h2>
      <span className="flag low">Predictions only: nothing here has been made or measured</span>
      <p className="note">
        The generator mixes new proportions of element sets that labs have tested, keeps each element inside the range those
        tested alloys used, and asks the network you trained here to predict the hardness. It drops anything within {NOT_NEW.toFixed(2)} of
        a tested alloy (not new), anything further than {TOO_FAR.toFixed(2)} from every tested alloy (too far from the data), and anything the
        network predicts at the edge of its training range. Distance is the sum of the differences in element fractions.
      </p>

      <div className="reqs fam-reqs">
        <Row name="Target hardness" on={target.on} onToggle={(on) => setTarget({ ...target, on })}
          min={200} max={900} step={10} value={target.value} onValue={(value) => setTarget({ ...target, value })}
          text={`Lower edge of the 90% range at least ${target.value} HV`} offText="Off: as hard as possible" />
        <div className="req">
          <label className="switch" htmlFor="alloy-route">Processing route</label>
          <select id="alloy-route" value={proc} onChange={(e) => setProc(e.target.value)}>
            {data.procs.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
          <span className="reqvalue">The route the network is asked to predict for</span>
        </div>
        <Row name="Only element sets that were tested by this route" on={routeRule} onToggle={setRouteRule}
          text={`Only sets that labs made by ${proc}`} offText={`Off: also sets never made by ${proc} (each row says so)`} />
        <Row name="Only candidates whose nearest tested alloys also reached the target" on={target.on && supported}
          onToggle={setSupported} disabled={!target.on}
          text={`At least ${SUPPORT_NEEDED} of the ${NEIGHBOURS} nearest tested alloys measured ${target.value} HV or more`}
          offText={target.on ? 'Off: the prediction alone decides' : 'Needs a target'} />
        <details>
          <summary>Which elements may it use? ({onCount} of {Object.keys(allowed).length} allowed{capped ? `, ${capped} with a maximum` : ''})</summary>
          {Object.entries(allowed).map(([el, a]) => (
            <Row key={el} name={el} on={a.on} onToggle={(on) => set(el, { on })}
              min={0.05} max={1} step={0.05} value={a.max} onValue={(max) => set(el, { max })}
              text={a.max >= 1 ? 'Allowed, up to what the tested alloys used' : `Allowed, at most ${pct(a.max)}`} offText="Not allowed" />
          ))}
        </details>
      </div>

      <div className="nnrow">
        <button type="button" className="chip nnbtn" onClick={press} disabled={running || waiting}>
          {running || waiting ? busyText : 'Generate untested alloys'}
        </button>
        <span className="note small">
          {ready ? `Uses the network trained on this page and the 90% half-width ±${q.toFixed(0)} HV ${qSource}.`
            : 'The network on this page is not trained yet: the button trains it first (a few seconds), then generates.'}
        </span>
      </div>

      {shown && (
        <>
          <p className="note">
            Request: {shown.settings.targetHV == null ? 'as hard as possible' : `lower edge at least ${shown.settings.targetHV} HV`}, route {shown.settings.proc},{' '}
            {shown.settings.routeMustBeTested ? 'only element sets tested by this route' : 'any tested element set'}
            {shown.settings.onlySupported ? ', only candidates whose nearest tested alloys reached the target' : ''}.{' '}
            From {stats.sets.used} tested element sets the generator drew {stats.draws} compositions. Excluded: {ex.duplicate} repeat draws,{' '}
            {ex.notNew} not new (within {NOT_NEW.toFixed(2)} of a tested alloy), {ex.tooFar} too far (over {TOO_FAR.toFixed(2)} from every tested alloy),{' '}
            {ex.subsetCopy} subset copies (a tested alloy with an element left out), {ex.clamped} at the clamp (predicted at the edge of the training range),{' '}
            {ex.belowTarget} below the target, {ex.unsupported} unsupported (fewer than {SUPPORT_NEEDED} of {NEIGHBOURS} nearest tested alloys reached the target).{' '}
            <strong>Kept {stats.kept}</strong>{stats.kept > list.length ? `; the top ${list.length} are shown` : ''}.
            {!shown.settings.routeMustBeTested && ` ${stats.keptRouteNotTested} of the kept candidates come from element sets never made by ${shown.settings.proc}.`}
            {' '}Interval: ±{shown.q.toFixed(0)} HV, {shown.qSource}. Took {Math.round(stats.timing.ms)} ms
            {stats.timing.yields ? `, giving the browser ${stats.timing.yields} turns so the page did not freeze` : ''}.
          </p>
          {list.length === 0 ? (
            <p className="note"><strong>Nothing to show.</strong> {nothingKept(stats, shown.settings.targetHV)}</p>
          ) : (
            <>
              <p className="note">
                These are predictions from a network whose typical miss is about {familiar} HV on alloys like the tested ones and
                about {unseen} HV on alloy systems it never saw. The ranking follows the prediction, because the 90% range is one
                width (±{shown.q.toFixed(0)} HV) everywhere; its lower edge is valid for alloys like the tested ones and wider in reality
                for new element sets. The nearest tested alloy is the best evidence on the page: it was made and measured, the
                candidates were not. A large positive gap means the network rates the candidate far above anything measured near it.
              </p>
              <div className="tablewrap">
                <table className="compact">
                  <thead>
                    <tr>
                      <th className="num">Rank</th>
                      <th>Composition (at.%)</th>
                      <th className="num">Predicted HV</th>
                      <th className="num">Lower edge of the 90% range</th>
                      <th>Nearest tested alloy (route, measured) and gap</th>
                      <th>Distance and tier</th>
                      <th>Nearest {NEIGHBOURS} tested alloys that reached the target</th>
                    </tr>
                  </thead>
                  <tbody>
                    {list.map((c, i) => (
                      <tr key={c.composition}>
                        <td className="num">{i + 1}</td>
                        <td className="wraps">
                          <strong>{c.composition}</strong>
                          <span className="label">Untested composition: predicted only, never made or measured</span>
                          {!c.routeTestedInSet && <span className="label"> This element set was never made by {c.proc} (tested only as {c.setRoutes.join(', ')}).</span>}
                        </td>
                        <td className="num strong">{Math.round(c.predicted)}</td>
                        <td className="num">{Math.round(c.cautious)}</td>
                        <td className="wraps">
                          {c.nearest.formula}, {c.nearest.proc}, {hv(c.nearest.hv)}
                          <span className="label"> Gap: {signed(c.gapHV)}</span>
                        </td>
                        <td>{c.distance.toFixed(3)}<span className="label"> {c.tier}</span></td>
                        <td>
                          {c.neighboursReached == null ? 'no target set' : `${c.neighboursReached} of ${NEIGHBOURS}`}
                          <span className="label"> {c.neighbours.map((nb) => Math.round(nb.hv)).join(', ')} HV</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      )}
    </section>
  )
}
