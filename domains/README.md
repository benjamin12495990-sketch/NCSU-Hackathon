# domains/

Scope beyond concrete. Each subfolder is one domain that reuses the project's method (see
`../docs/SCOPE.md`): a cautious surrogate model, candidate generation by perturbing known-good
examples, ranking by a cost objective, and "lab-supported" flagging from nearest measured
neighbours.

- `_TEMPLATE/` — copy this to start a new domain.

The live concrete app currently lives in the repo root (`src/`, `train/`, `public/model/`). As
domains mature they can move or link here; for now this folder holds the expansion scaffolding so
it never collides with the app directories other agents are editing.

## Domains in scope
- **materials** — alloys, polymers, batteries (concrete is the first worked example)
- **chemistry** — molecular property / reaction prediction
- **biology** — protein / antimicrobial / drug-target design

Literature grounding for every domain lives in `../research/references/<domain>.md`.
