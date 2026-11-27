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

## Before you say "done" (five questions)

Before any message that says a result is done, verified, ready or better, and before editing a file that states a result
(README, a document in `docs/`, a `REPORT.md`, the web page in `src/`), work through these five questions and put the
answers to 2, 3 and 4 in the message, in a few lines.

1. **Asked versus meant.** What was literally asked, and what does the person actually want? If they differ, say so.
2. **Cheap wins.** Name at least two ways this result could look good without being good. Caught in this project so far
   (Lessons in `docs/DECISIONS.md`): an average of several networks against one tree model (LES-010); a split by exact
   formula, or one with near-copies on both sides, behind a claim about unseen material (LES-001, LES-011); an untuned
   or under-informed baseline (LES-003, LES-011); a choice made after looking at the rows it is scored on (LES-005,
   LES-006); two models compared on different rows or with different training length (LES-002, LES-013); code never run
   on the target machine described as working (LES-007, LES-009). Also watch for: a gain on one seed; a rehearsal's
   numbers quoted as a result.
3. **Strongest objection.** Write the first thing a careful reviewer would say against this result. Fix it, or state it
   next to the claim.
4. **Not checked.** Name every claim not backed by a file you opened or a check you ran, every test the request implies
   that did not run (for example: never run on the GPU or on Windows), and every number quoted from memory (tag it
   `[UNVERIFIED - from memory]`). "Roughly" and "as expected" name nothing.
5. **Revise, or say what cannot be done.** Saying which part cannot be met, and what it would take, is a correct answer.
   Reporting the result as done while one of the checks above failed is not.

Before training or scoring a model, the nine-question checklist in `docs/ISOLATED_MODEL_TRAINING_PROTOCOL.md` section 8
applies as well.

Match the strength of a sentence to the test behind it. The interval is the 95% paired-bootstrap interval of (error of A
minus error of B), resampling whole groups, against the strongest baseline chosen without test rows. Use the report's own
three labels. If two rows apply, the weaker wording wins.

| What the test showed | Wording |
| --- | --- |
| Whole interval on one side of zero for the average and for every single seed or run | Direct statement naming the winner, the opponent, the split and the interval: "A clearly better than B on held-out systems: -0.083 [-0.091, -0.076] eV/atom" |
| Whole interval on one side of zero for the average of N runs, but not for every single run | "The average of N networks is clearly better than one tree model; one network at a time is clearly better in K of N" (give K, and the result against the average of the 3 best tree models) |
| Interval includes zero | "No clear difference", with the interval (older files say "not distinguishable" or "a tie"; never "comparable", "matches" or "as good as") |
| A number with no interval behind it (one seed, one unrepeated split) | Qualified, naming the gap: "on one seed; not repeated" |
| Rows that already informed a choice: every concrete score from `gpu/`, a fixed test split scored before, anything on the opened vault rows | Add "exploratory" (vault rows: "post-hoc") to whichever wording above applies |
| A rehearsal (`-Rehearsal`) or a smoke run | No result sentence. Say only that the code ran; its numbers mean nothing |

Terms: a seed is one training of a network from one random start, and a run is one such training of a crystal network; a
rehearsal is a `-Rehearsal` or smoke run (`gpu/README.md` section 5); the person is the human you are working for.

Never, and asking does not change it: read `lab/vault/` or the five blocked concrete files (listed in `CLAUDE.md`); run the
vault opening again; let code or a search choose a model, a setting or a stopping point using test rows. (A person may
use a finished, reported test comparison as evidence for changing `gpu/recipes.json`: DEC-015, DEC-018.)

Stop and ask the person before: changing a split, a baseline or a pass condition (the rule that turns an interval into a
verdict) after results have been seen, and if the answer is yes, label the new result exploratory or post-hoc and add a
row to `docs/DECISIONS.md`; any change to `gpu/recipes.json` (it changes only by a reviewed commit); scoring again on a
fixed test split that was already scored (the new number is exploratory); deleting or overwriting any file; anything
that costs money (the team decided not to spend any: `research/connectors.md`); installing into an environment someone
else also uses (use `.venv-gpu`).

A result far from what was expected is reported as it is, not explained away.

(Adapted from get-physics-done (GPD), https://github.com/psi-oss/get-physics-done, version 1.2.2, commit 0f417695 of its
main branch; Copyright 2026 Physical Superintelligence PBC; licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). The text here was changed from the original. The five questions adapt "The
five-item gate", and the wording table adapts "S4. Confidence-to-language mapping", both in
`src/gpd/specs/references/shared/reward-hacking-self-check.md`. The `[UNVERIFIED - ...]` tag adapts GPD's
`[UNVERIFIED - training data]` in `src/gpd/agents/gpd-executor.md` and
`src/gpd/specs/references/execution/executor-numerical-protocol.md`. The examples and the table rows are this project's.
What else GPD offers: `research/get_physics_done.md`.)

## Decision log and lessons

`docs/DECISIONS.md` holds two append-only tables: decisions already made and mistakes already caught, each with its
evidence. Read it before proposing a change of method or evaluation. Add a row when you make such a change, or when you
withdraw or weaken a result that was already reported. Never edit or remove a row: a reversed decision, or a row found to
be wrong, gets a new row that names the old one.

