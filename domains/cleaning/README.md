# Cleaning products & surfactants

Propose detergent / cleaner formulations that clean well at the lowest cost and environmental load.

## 1. The expensive outcome
Formulating and bench-testing a cleaner (soil-removal assays, foam, stability) takes days.
Predictable targets:
- **Critical micelle concentration (CMC)** and **surface tension** — the physical-chemistry
  handles on cleaning power, well studied for ML.
- **Cleaning efficacy** score where a dataset provides one.

## 2. Dataset (confirm current access via the research MCPs)
- Published **CMC / surface-tension datasets** for surfactants (find via OpenAlex; several ML-CMC
  papers release their tables).
- Surfactant property compilations (cite the source).
Features: surfactant structure (head/tail descriptors), blend ratios, builder & solvent content,
temperature, water hardness.

## 3. Candidate space
Blend known surfactants at new ratios, add/scale builders and solvents, stay within tested ranges.

## 4. Models
Expected CMC / efficacy + calibrated cautious bound. Reuse either model family. Split by surfactant
family to avoid leakage.

## 5. Cost objective
Cost per wash, **biodegradability** (physico-chemical proxies, cite them), and irritancy class —
treated as sourced reference data, **not** modelled biology. Factors cited in
`../../research/references/cleaning.md`.

## 6. Lab-supported check
Flag only when nearest measured surfactant systems agree.

---
**Ground rules:** predictions as predictions; every non-learned number cited; chemistry & materials
only — biodegradability/irritancy enter only as cited reference classes, never as biological
models. See `../../docs/AGENTS.md`.
