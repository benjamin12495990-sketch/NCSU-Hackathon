# General materials

> **Status: one property is live: alloy hardness (Vickers HV), measured values collected from published
> papers (Borg et al., Scientific Data 7, 430 (2020)).** `prepare.py` and `nn_generic.js` here are the
> agent's files, copied byte-identical; the numbers are in `public/model/materials/family.json`. Sources
> were not checked again through MCPs. The rest of this page is the original plan.

The umbrella domain: any measured material property as the surrogate target. Concrete (repo root)
is the first fully worked example; polymers and insulators are specialised siblings.

- **Outcome:** hardness, conductivity, density, strength — whatever a dataset measures.
- **Datasets (confirm via MCPs):** Materials Project, Matbench, JARVIS-DFT.
- **Candidate space:** composition + processing within tested ranges.
- **Models:** `train/train.py` (trees) or `pipeline/train_nn.py` (neural net), expected + cautious.
- **Cost:** cost, scarcity, embodied CO2 — cited in `../../research/references/materials.md`.

Use this folder for cross-cutting materials work; spin out a dedicated folder (like `polymers/`)
when a property family deserves its own featurisation and sources.
