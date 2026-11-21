# Scope: from concrete to a tool for scientific discovery

## The grand challenge

> Engineer the Tools of Scientific Discovery.

The concrete app (`src/`, `train/`, `public/model/`) is the **first worked example**, not the
whole project. It demonstrates a pattern that generalises far past concrete.

## The pattern

Every physical experiment is slow and expensive. A concrete mix takes up to 28 days to cure; a
new alloy, a candidate molecule, or a protein variant each carry their own long, costly loop. So
only a handful of the possible candidates are ever tried.

The pattern this project applies to that problem:

1. **Learn a surrogate.** Train a model on a lab-tested dataset so it predicts the expensive
   outcome (strength, conductivity, binding affinity, …) from cheap-to-compute inputs.
2. **Quantify caution.** Pair the expected-value model with a conservative lower estimate, so the
   tool proposes candidates that are *likely* to clear the bar, not just those that might.
3. **Generate and rank candidates.** Perturb known-good examples to propose new, unseen ones;
   drop duplicates; keep those the cautious model clears; rank by a cost objective (CO2, price,
   toxicity, synthesis difficulty).
4. **Check against the real data.** Flag a candidate as "lab-supported" only when its nearest
   measured neighbours agree. Never pretend a prediction is a measurement.
5. **Ground every assumption in the literature.** Any number the model does not learn from data
   (CO2 factors, densities, physical constants) is sourced to a published paper, retrieved and
   cross-checked through the research MCPs in `research/connectors.md`.

## Domains in scope

Selected directions (see `domains/`):

| Domain | Expensive outcome | Candidate space | Cost objective |
| --- | --- | --- | --- |
| **Materials** (live: concrete) | compressive strength, cure time | mix / composition | embodied CO2 |
| **Materials** (next) | alloy / polymer / battery properties | composition, processing | cost, CO2, scarcity |
| **Chemistry** | molecular property, reaction yield | molecular structure | synthesis difficulty, toxicity |
| **Biology** | binding, immunogenicity, MIC | sequence / variant | off-target risk, developability |

Each domain reuses steps 1–5. What changes per domain is the dataset, the featurisation, and the
literature sources — not the method.

## What stays true across every domain

- **Nothing is mocked.** Predictions are labelled as predictions; measurements as measurements.
- **Every non-learned number is cited** to a retrievable source.
- **The tool narrows the experiment list; it does not replace the experiment.**
