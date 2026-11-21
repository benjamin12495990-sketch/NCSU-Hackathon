# Scope: a general engine for proposing physical formulations

> **Status: only concrete is built.** Every other domain in this repository is a plan. None has a
> dataset, a trained model or any references yet.

## The grand challenge

> Engineer the Tools of Scientific Discovery.

The concrete app (`src/`, `train/`, `public/model/`) is the **first worked example**. The ambition
is much larger: a single engine that proposes new, unseen **formulations and material
compositions** for a target property and ranks them by cost — across chemistry and materials
science, one domain after another.

## Out of scope (deliberately)

**Biology.** No proteins, sequences, assays, drugs, or living systems. It is a complete subject of
its own with its own data, safety and regulatory world, and folding it in would dilute everything
else. Everything here is **non-living chemistry and materials**: molecules, mixtures, solids, and
solutions.

## The pattern (unchanged across every domain)

Every physical experiment is slow and expensive, so only a handful of the possible candidates ever
get tried. The engine:

1. **Learns a surrogate** on a lab-tested dataset: predict the expensive outcome from cheap inputs.
2. **Quantifies caution**: pair the expected value with a calibrated lower (or upper) bound, so it
   proposes candidates *likely* to clear the bar.
3. **Generates and ranks candidates**: perturb known-good examples into new ones, drop duplicates,
   keep those the cautious model clears, rank by a cost objective.
4. **Checks against real data**: flag "neighbours passed" only when nearest measured neighbours agree.
5. **Grounds every assumption in the literature** via the research MCPs in `research/connectors.md`.

Two model families exist for steps 1 and 2, both written for concrete: gradient-boosted trees
(`train/train.py`, used by the live page) and a neural network (`pipeline/train_nn.py`, not used by
the live page and less accurate). Either could be pointed at another tabular dataset. That has not
been done for any domain below.

## The domain map (plans, apart from concrete)

Each row reuses steps 1–5. What changes is the dataset, the featurisation, and the literature
sources — **not the method**.

| Domain | Expensive outcome to predict | Candidate space | Cost / objective |
| --- | --- | --- | --- |
| **Concrete** (live) | compressive strength, cure time | mix proportions | embodied CO2 |
| **Carbon-based polymers** | glass transition, tensile strength, thermal stability | monomer choice, chain/composition | synthesis cost, CO2, recyclability |
| **Corrosive solutions** | corrosion rate / material compatibility | acid·base·salt·inhibitor + concentration + temperature | material loss, hazard, cost |
| **Cleaning products** | cleaning efficacy, critical micelle concentration, surface tension | surfactant blend + builders + solvent | cost, biodegradability, irritancy |
| **Insulators** | thermal conductivity, dielectric breakdown strength | composition, porosity, processing | cost, weight, CO2 |
| **General materials** | any measured property (hardness, conductivity, density) | composition + processing | cost, scarcity, CO2 |
| **General chemistry** | solubility, boiling point, reaction yield, viscosity | molecular structure, mixture | synthesis difficulty, cost |

### Reach domains (same engine, later)
coatings · adhesives · lubricants · electrolytes & battery chemistries · refrigerants · dyes &
pigments · catalysts · propellants · sealants · composites · glasses & ceramics.

## What stays true across every domain

- **Nothing is mocked.** Predictions are labelled as predictions; measurements as measurements.
- **Every non-learned number is cited** to a retrievable source (`research/references/<domain>.md`).
- **The engine narrows the experiment list; it does not replace the experiment.**
- **Chemistry and materials only — never biology.**
