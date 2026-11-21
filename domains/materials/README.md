# General materials

> **Status: plan only.** No dataset, model or references yet.

The umbrella domain: any measured material property as the surrogate target. Concrete (repo root)
is the first fully worked example; polymers and insulators are specialised siblings.

- **Outcome:** hardness, conductivity, density, strength — whatever a dataset measures.
- **Datasets (confirm via MCPs):** Materials Project, Matbench, JARVIS-DFT.
- **Candidate space:** composition + processing within tested ranges.
- **Models:** `train/train.py` (trees) or `pipeline/train_nn.py` (neural net), expected + cautious.
- **Cost:** cost, scarcity, embodied CO2 — cited in `../../research/references/materials.md`.

Use this folder for cross-cutting materials work; spin out a dedicated folder (like `polymers/`)
when a property family deserves its own featurisation and sources.
