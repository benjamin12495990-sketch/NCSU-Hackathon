# Insulators (thermal & electrical / dielectrics)

> **Status: plan only.** No dataset, model or references yet.

Propose materials or composites that insulate best for the lowest cost and weight.

## 1. The expensive outcome
Measuring thermal conductivity or dielectric breakdown needs fabricated samples and specialised
rigs. Predictable targets:
- **Thermal conductivity** (low = good thermal insulator).
- **Dielectric constant** and **breakdown strength** (electrical insulation).

## 2. Dataset (confirm current access via the research MCPs)
Strong public sources exist here:
- **Materials Project** — dielectric tensor data for many compounds.
- **JARVIS-DFT** (NIST) — dielectric / thermal properties.
- **Matbench** tasks include dielectric and phonon targets.
Verify licence and exact fields via OpenAlex / the dataset docs; cite each.
Features: composition, porosity, density, processing.

## 3. Candidate space
Vary composition and porosity within the dataset's range; for composites, vary the component
fractions.

## 4. Models
Expected property + calibrated cautious bound (caution = don't over-promise insulation). Reuse
either model family. Split by material family.

## 5. Cost objective
Cost, **weight / density**, and embodied CO2. Factors cited in
`../../research/references/insulators.md`.

## 6. Neighbour check
Flag only when nearest measured materials agree.

---
**Ground rules:** predictions as predictions; every non-learned number cited; never biology.
See `../../docs/AGENTS.md`.
