import { useEffect, useState } from 'react'

// Shown for a family of materials that is planned but not built. It states
// the plan from docs/SCOPE.md and says plainly that nothing is generated yet.
// A family trained offline but not on the site (domain.trainedOffline) gets one line with its
// verified numbers, read from public/model/<id>/family.json (no training data is shipped).
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
  return (
    <section className="card planned">
      <span className="flag low">Planned, not built</span>
      <h1>{domain.label}</h1>
      <p className="note">
        Nothing is generated for this family yet.{domain.trainedOffline ? '' : ' It has no lab-tested dataset, no trained model and no references.'} The
        plan is to reuse the method that already runs for concrete.
      </p>
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
        <div><dt>Needs first</dt><dd>{domain.trainedOffline ? 'A training file small enough to load on a phone' : 'A dataset of lab results to learn from'}</dd></div>
      </dl>
      <p className="note small">
        The method, as it runs for concrete: learn from lab data, estimate cautiously, generate and rank
        new candidates, check them against nearby lab results, and cite or label every assumption. The
        full plan is in docs/SCOPE.md in the repository.
      </p>
      <button type="button" className="chip" onClick={onBack}>Open the live concrete designer</button>
    </section>
  )
}
