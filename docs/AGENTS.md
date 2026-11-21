# Notes for AI agents working in this repo

Several agents work this repo in parallel. Read this before editing so we don't collide.

## Who owns what

| Path | Purpose | Touch if you are… |
| --- | --- | --- |
| `src/`, `index.html`, `vite.config.js` | the live concrete web app | working on the app UI / inference |
| `train/` | model training + dataset | working on models |
| `public/model/` | exported models + factors the app loads | changing models or factors |
| `scripts/` | parity / build checks | working on tooling |
| `research/` | literature grounding, MCP usage, citations | adding or checking sources |
| `domains/` | scope beyond concrete (materials, chem, bio) | expanding scope |
| `docs/` | project context, scope, these notes | documenting |

To avoid merge pain: keep new work in `research/`, `domains/`, and `docs/` (added by the
scope-expansion effort) separate from the app directories above. Prefer a branch per task.

## Ground rules (apply to every agent and every domain)

1. **Predictions are labelled as predictions, measurements as measurements.** Never present a
   model output as a lab result.
2. **Every number the model does not learn from data is cited** to a retrievable source. Put the
   citation in the matching `research/references/<domain>.md` file.
3. **Retrieve sources through the research MCPs** listed in `research/connectors.md`, and
   cross-check each factor with a second, independent search — the same method the concrete
   `factors.json` already follows.
4. **The tool narrows the experiment list; it does not replace the experiment.**

## The method, in one line

Train a cautious surrogate on lab data → generate candidates by perturbing known-good ones → keep
what the cautious model clears → rank by a cost objective → flag "lab-supported" only when nearest
measured neighbours agree. See `docs/SCOPE.md` for the full version.
