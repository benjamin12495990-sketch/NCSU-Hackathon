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
| `domains/` | scope beyond concrete (materials and chemistry; no biology). Plans only | expanding scope |
| `docs/` | project context, scope, these notes | documenting |

To avoid merge pain: keep new work in `research/`, `domains/`, and `docs/` (added by the
scope-expansion effort) separate from the app directories above. Prefer a branch per task.

## Ground rules (apply to every agent and every domain)

1. **Predictions are labelled as predictions, measurements as measurements.** Never present a
   model output as a lab result.
2. **Every number the model does not learn from data gets a citation, or a label saying it has
   none.** Put citations in the matching `research/references/<domain>.md` file. On the live page,
   most of the materials table is uncited typical values, and the page says so.
3. **Retrieve sources through the research MCPs** listed in `research/connectors.md`, run a second,
   independent search for each factor, and record whether it confirmed the value. That is the
   method the concrete `factors.json` follows; several of its values were not confirmed and are
   labelled.
4. **The tool narrows the experiment list; it does not replace the experiment.**

## The method, in one line

Train a cautious surrogate on lab data → generate candidates by perturbing known-good ones → keep
what the cautious model clears → rank by a cost objective → flag "neighbours passed" only when nearest
measured neighbours agree. See `docs/SCOPE.md` for the full version.
