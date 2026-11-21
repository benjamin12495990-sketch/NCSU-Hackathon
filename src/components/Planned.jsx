// Shown for a family of materials that is planned but not built. It states
// the plan from docs/SCOPE.md and says plainly that nothing is generated yet.
export default function Planned({ domain, onBack }) {
  return (
    <section className="card planned">
      <span className="flag low">Planned, not built</span>
      <h2>{domain.label}</h2>
      <p className="note">
        Nothing is generated for this family yet. It has no lab-tested dataset, no trained model and no
        references. The plan is to reuse the method that already runs for concrete.
      </p>
      <dl className="plan">
        <div><dt>Would predict</dt><dd>{domain.predict}</dd></div>
        <div><dt>Would vary</dt><dd>{domain.vary}</dd></div>
        <div><dt>Would rank by</dt><dd>{domain.rankBy}</dd></div>
        <div><dt>Needs first</dt><dd>A dataset of lab results to learn from</dd></div>
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
