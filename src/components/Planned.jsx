import { useEffect, useState } from 'react'

// Shown for a family of materials that has no page of its own. It states the plan from docs/SCOPE.md and says plainly
// that nothing is generated here.
// A family trained offline but not on the site (domain.trainedOffline) gets one line with its
// verified numbers, read from public/model/<id>/family.json (no training data is shipped).
// A family trained and scored in the GPU pipeline (domain.pipeline) shows its report numbers from public/model/<id>/pipeline.json;
// domain.noPlan marks one that docs/SCOPE.md has no plan for (its footer says so instead of describing the concrete method).
export default function Planned({ domain, onBack }) {
  const [offline, setOffline] = useState(null)
  useEffect(() => {
    if (!domain.trainedOffline) return undefined
    const ctl = new AbortController()
    fetch(`${import.meta.env.BASE_URL}model/${domain.id}/family.json`, { signal: ctl.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then(setOffline, () => {})
    return () => ctl.abort()
  }, [domain])
  const [pipe, setPipe] = useState(null)
  useEffect(() => {
    if (!domain.pipeline) return undefined
    const ctl = new AbortController()
    fetch(`${import.meta.env.BASE_URL}model/${domain.id}/pipeline.json`, { signal: ctl.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then(setPipe, () => {})
    return () => ctl.abort()
  }, [domain])
  return (
    <section className="card planned">
      {domain.pipeline ? <span className="flag plain">Trained offline, not on this page</span> : <span className="flag low">Planned, not built</span>}
      <h1>{domain.label}</h1>
      <p className="note">
        {domain.pipeline
          ? (domain.noPlan
            ? 'No design page exists for this family, and docs/SCOPE.md has no plan for one yet.'
            : 'No design page exists for this family yet. The plan is to reuse the method that already runs for concrete.')
          : <>Nothing is generated for this family yet.{domain.trainedOffline ? '' : ' It has no lab-tested dataset, no trained model and no references.'} The
        plan is to reuse the method that already runs for concrete.</>}
      </p>
      {domain.pipeline && (
        <div className="note">
          <p>
            Models for this family were trained and scored offline in the GPU pipeline. They do not run on this
            page, and nothing here is generated from them.
          </p>
          {pipe && (
            <>
              <p>{pipe.intro} {pipe.exploratoryNote}</p>
              <ul>
                {pipe.tasks.map((t) => (
                  <li key={t.name}>
                    <strong>{t.name}</strong> ({t.unit}{t.size ? `; ${t.size}` : ''}): network ({t.networkLabel}) {t.network},
                    one tuned tree model (chosen without the test rows) {t.trees}, guess the average {t.mean}. Network average minus
                    that tree model: {t.diff} [{t.ci[0]}, {t.ci[1]}],
                    {' '}{t.verdict}. {t.single} {t.extra}
                  </li>
                ))}
              </ul>
              <p className="small">Source: <code>{pipe.source}</code>. Errors are mean absolute error; brackets are 95% intervals.</p>
            </>
          )}
        </div>
      )}
      {offline && (
        <p className="note">
          A water-solubility (LogS) model was trained and checked offline; its scripts and results files are
          in <code>{offline.folder}</code>, but its data file is not in the repo or on the site. On held-out compounds its
          average miss is {offline.test.network.toFixed(2)} {offline.test.unit}, against {offline.test.treesTuned.toFixed(2)} for
          fairly tuned trees (a tie) and {offline.test.mean.toFixed(2)} for guessing the average. {offline.reason}
        </p>
      )}
      <dl className="plan">
        <div><dt>Would predict</dt><dd>{domain.predict}</dd></div>
        <div><dt>Would vary</dt><dd>{domain.vary}</dd></div>
        <div><dt>Would rank by</dt><dd>{domain.rankBy}</dd></div>
        <div><dt>Needs first</dt><dd>{domain.noPlan ? 'A plan in docs/SCOPE.md, then a model small enough to run in the browser and a design page' : domain.pipeline ? 'A model small enough to run in the browser, and a design page' : domain.trainedOffline ? 'A training file small enough to load on a phone' : 'A dataset of lab results to learn from'}</dd></div>
      </dl>
      {domain.noPlan ? (
        <p className="note small">
          docs/SCOPE.md has no plan for this family yet. The method that runs for concrete learns from lab data; every label
          here is a DFT calculation, so whether or how that method would apply is not decided.
        </p>
      ) : (
        <p className="note small">
          The method, as it runs for concrete: learn from lab data, estimate cautiously, generate and rank
          new candidates, check them against nearby lab results, and cite or label every assumption. The
          full plan is in docs/SCOPE.md in the repository.
        </p>
      )}
      <button type="button" className="chip" onClick={onBack}>Open the live concrete designer</button>
    </section>
  )
}
