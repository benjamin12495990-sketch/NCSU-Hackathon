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
function nothingKept(stats, target, offCount) {
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
      `${hv(stats.bestCautious)} (even a prediction at the training maximum, ${hv(stats.rules.clamp[1])}, would have a lower edge of ` +
      `${hv(stats.ceilingCautious)}). ` +
      (stats.bestMeasured ? `Tested alloys in these element sets measured up to ${hv(stats.bestMeasured.hv)} (${stats.bestMeasured.formula}, ` +
        `${stats.bestMeasured.proc}); those were made and measured, so this is a limit of the generator and its interval, not of alloys. ` : '') +
      `To see candidates, lower the target to ${Math.floor(stats.bestCautious / 10) * 10} HV or less` +
      (offCount ? ', or allow the elements you switched off.' : '.')
  }
  if (ex.unsupported > 0) {
    return `${ex.unsupported} candidates reach ${hv(target)} on the network's prediction, but for each of them fewer than ${SUPPORT_NEEDED} of its ` +
      `${NEIGHBOURS} nearest tested compositions measured at least ${hv(target)}. The measured evidence does not back them, so none is shown. ` +
      'Switch off "Only candidates whose nearest tested alloys also reached the target" to see them anyway.'
  }
  return 'No candidate passed every rule.'
}

export default function AlloyGenerator({ fam, net, band, status, whenTrained, folds, liveMae }) {
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
    // Candidates the network rates more than one familiar-system error above their nearest tested
    // alloy are ranked after the others (the number is family.json test.network).
    gapLimit: fam.test.network,
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
  const familiar = Math.round(fam.test.network), unseen = Math.round(fam.wholeSystems.network), unseenTrees = Math.round(fam.wholeSystems.trees)
  const offCount = Object.values(allowed).filter((a) => !a.on).length
  const hasTarget = shown?.settings.targetHV != null
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
            {(data.routes_selectable ?? data.procs).map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
          <span className="reqvalue">The route the network is asked to predict for</span>
        </div>
        <Row name="Only element sets that were tested by this route" on={routeRule} onToggle={setRouteRule}
          text={`Only sets that labs made by ${proc}`} offText={`Off: also sets never made by ${proc} (each row says so)`} />
        <Row name="Only candidates whose nearest tested alloys also reached the target" on={target.on && supported}
          onToggle={setSupported} disabled={!target.on}
          text={`At least ${SUPPORT_NEEDED} of the ${NEIGHBOURS} nearest distinct tested compositions measured ${target.value} HV or more`}
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
            {hasTarget ? `${ex.belowTarget} below the target, ${ex.unsupported} unsupported (fewer than ${SUPPORT_NEEDED} of the ${NEIGHBOURS} nearest distinct tested compositions reached the target).` : 'no target, so nothing was excluded for missing it.'}{' '}
            <strong>Kept {stats.kept}</strong>{stats.kept > list.length ? `; the top ${list.length} are shown` : ''}.
            {!shown.settings.routeMustBeTested && ` ${stats.keptRouteNotTested} of the kept candidates come from element sets never made by ${shown.settings.proc}.`}
            {' '}Interval: ±{shown.q.toFixed(0)} HV, {shown.qSource}. Took {Math.round(stats.timing.ms)} ms
            {stats.timing.yields ? `, giving the browser ${stats.timing.yields} ${stats.timing.yields === 1 ? 'turn' : 'turns'} so the page did not freeze` : ''}.
          </p>
          {list.length === 0 ? (
            <p className="note"><strong>Nothing to show.</strong> {nothingKept(stats, shown.settings.targetHV, offCount)}</p>
          ) : (
            <>
              <p className="note">
                These are predictions. The five-network ensemble trained in Node with the same code missed by about {familiar} HV on
                held-out alloys from familiar systems and about {unseen} HV on alloy systems it never saw (trees {unseenTrees})
                {liveMae != null ? `; the one network trained on this page missed by ${Math.round(liveMae)} HV on the held-out alloys` : ''}.
                Ranking: candidates the network rates more than {familiar} HV above their nearest tested alloy go last
                ({stats.keptFarAbove} of {stats.kept} kept), the rest by the lower edge of the 90% range, which is one width
                (±{shown.q.toFixed(0)} HV) everywhere, so inside each group it follows the prediction. The lower edge is valid for
                alloys like the tested ones and wider in reality for new element sets. The nearest tested alloy is the best evidence
                on the page: it was made and measured, the candidates were not. Repeat measurements of one composition count as one
                neighbour, shown as their mean.
              </p>
              <div className="tablewrap">
                <table className="compact">
                  <thead>
                    <tr>
                      <th className="num">Rank</th>
                      <th>Composition (at.%)</th>
                      <th>Distance and tier</th>
                      <th>Nearest tested alloy (route, measured) and gap</th>
                      <th className="num">Lower edge of the 90% range (valid for alloys like the tested ones; wider in reality for new element sets)</th>
                      <th className="num">Predicted HV</th>
                      <th>Nearest {NEIGHBOURS} distinct tested compositions: how many reached the target</th>
                    </tr>
                  </thead>
                  <tbody>
                    {list.map((c, i) => (
                      <tr key={c.composition}>
                        <td className="num">{i + 1}</td>
                        <td className="wraps">
                          <strong>{c.composition}</strong>
                          <span className="label">Untested composition: predicted only, never made or measured.
                            {' '}Distance {c.distance.toFixed(3)} to the nearest tested alloy ({c.nearest.proc}, {hv(c.nearest.hv)}); gap {signed(c.gapHV)}.</span>
                          {c.farAbove && <span className="label"> The network rates this far above its measured neighbours (more than {familiar} HV above the nearest), so it is ranked last.</span>}
                          {!c.routeTestedInSet && <span className="label"> This element set was never made by {c.proc} (tested only as {c.setRoutes.join(', ')}).</span>}
                        </td>
                        <td>{c.distance.toFixed(3)}<span className="label"> {c.tier}</span></td>
                        <td className="wraps">
                          {c.nearest.formula}, {c.nearest.proc}, {hv(c.nearest.hv)}
                          {c.nearest.proc !== c.proc && <span className="label"> (made by {c.nearest.proc}, not {c.proc})</span>}
                          <span className="label"> Gap: {signed(c.gapHV)}</span>
                        </td>
                        <td className="num">{Math.round(c.cautious)}</td>
                        <td className="num">{Math.round(c.predicted)}</td>
                        <td className="wraps">
                          {c.neighboursReached == null ? '— (no target)' : `${c.neighboursReached} of ${NEIGHBOURS}`}
                          <span className="label"> measured: {c.neighbours.map((nb) => `${Math.round(nb.hv)} HV ${nb.routes.join('/')}${nb.n > 1 ? ` (mean of ${nb.n})` : ''}`).join('; ')}</span>
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
