# Carbon-based polymers

Propose new polymer formulations that hit a target property, ranked by cost and recyclability.

## 1. The expensive outcome
Synthesising and characterising a new polymer takes days to weeks (polymerisation, purification,
then DSC / tensile / TGA testing). Predictable targets:
- **Glass transition temperature (Tg)** — the classic polymer-informatics regression task.
- **Tensile strength / modulus**, **thermal stability** (decomposition onset).

## 2. Dataset (confirm current access via the research MCPs)
Candidate public sources — verify licence and fields before use:
- **PoLyInfo** (NIMS) — measured polymer properties.
- **Polymer Genome** / Ramprasad-group datasets — Tg and other properties with structure.
- Property datasets collected in polymer-informatics review papers (find via OpenAlex).
Featurise from monomer structure (fingerprints / descriptors) plus processing where available.

## 3. Candidate space
Perturb known polymers: swap monomers, vary comonomer ratios and chain length, stay inside the
descriptor ranges the dataset actually covers.

## 4. Models
Reuse `train/train.py` (trees) or `pipeline/train_nn.py` (neural net): an expected-value model plus
a calibrated cautious bound. **Split by polymer family**, not by row, to avoid leakage.

## 5. Cost objective
Synthesis cost, embodied CO2 of the monomers, and **recyclability** (prefer thermoplastics /
depolymerisable backbones). Every non-learned factor cited in
`../../research/references/polymers.md`.

## 6. Lab-supported check
Flag a candidate only when its nearest measured polymers agree with the prediction.

---
**Ground rules:** predictions labelled as predictions; every non-learned number cited; chemistry &
materials only, never biology. See `../../docs/AGENTS.md`.
