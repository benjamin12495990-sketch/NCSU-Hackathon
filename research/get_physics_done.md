# Get Physics Done (GPD): what it is and what to borrow

Written 2026-11-27. Question: can this project use the open-source get-physics-done project, and how?

How this was found: one reviewer read a clone of the GPD repository on 2026-11-27, and a second reviewer checked every
finding against the same clone and this repository the same day. Where the check corrected the first read (README line
numbers, the count of error-class checks, which analysis commands run outside a GPD project, the test tooling here, the
site's existing distance checks), the corrected version is used below. Both reports are kept outside the repository.
Every GPD path in this file was checked again against the clone for this document. GPD was not installed or run.

GPD paths below are relative to its repository at commit `0f417695ccf1f987af44bb4f4b3cbbdeb5f2c48b`. To open one, put it
after `https://github.com/psi-oss/get-physics-done/blob/0f417695ccf1f987af44bb4f4b3cbbdeb5f2c48b/`. Line numbers are at that
commit.

## In short

- [Certain] GPD is a large set of prompts, workflows and checklists for an AI coding agent doing physics research, plus a
  Python command-line tool and MCP servers that keep its project state. Apache-2.0, Copyright 2026 Physical
  Superintelligence PBC.
- [Certain] Its code does bookkeeping, not physics. Its verification tool says so: "This tool is non-authoritative: it
  does not perform physics verification" (`src/gpd/mcp/servers/verification_server.py` line 2151). What is worth taking is
  the written protocols, not the software.
- [Certain] This project already uses four GPD pieces: the five questions, the wording table, the `[UNVERIFIED - ...]`
  tag and the decision-log columns. The old credit named one wrong source file and left out the copyright holder, the URL,
  the version and the licence link. This change fixes that in `docs/AGENTS.md` and `docs/DECISIONS.md`.
- [Likely] The most useful next borrowings are: a written registration with "forbidden proxies" before the next cycle;
  invariance and physical-bounds counts every night; a check that one number per task is the same in the analysis, the
  report and the website files; and input sweeps on every live simulation.
- [Likely] Installing GPD into this repository is not worth it. It changes shared agent settings and adds a second
  project process, and its tools would not check this project's numbers.

## What GPD is

**Who makes it.** Physical Superintelligence PBC (PSI). `README.md` line 20: "an open-source agentic AI system for
physics research from Physical Superintelligence PBC (PSI), released as a community contribution". `CITATION.cff`,
`pyproject.toml` and `package.json` name the same company as author. No people are named here: the company's website did
not load from the review session.

**Licence.** Apache License 2.0. `LICENSE` ends with "Copyright 2026 Physical Superintelligence PBC" and links the licence
text at http://www.apache.org/licenses/LICENSE-2.0. `CITATION.cff`, `pyproject.toml` and `package.json` also say
Apache-2.0. There is no NOTICE file, so section 4(d) of the licence does not apply. `CITATION.cff` asks to be cited "If
GPD contributes to published research"; that is a request, not a licence term.

**Version and commit read.** The main branch at commit `0f417695` (committed 2026-05-28), read on 2026-11-27.
`CITATION.cff` and `pyproject.toml` give version 1.2.2; `CITATION.cff` gives the release date 2026-05-07, 21 days before
this commit. `CHANGELOG.md` has no v1.2 section: it opens with "vNEXT", 21 unreleased entries, then v1.1.0, the "Public
open-source release". So this commit is probably later than the v1.2.2 release, and "version 1.2.2" in the credits means
the version this commit's files give, not the tagged release. The second reviewer saw, with `git ls-remote`, that the
remote main was still at this commit on 2026-11-27 and that tags v1.1.0, v1.2.0, v1.2.1 and v1.2.2 exist; that was not
repeated here. Which commit the v1.2.2 tag points to was not checked (the clone is shallow and has no tags).

**What it contains** (counted in the clone):

| Part | Count | Where |
| --- | --- | --- |
| Slash commands | 71 | `src/gpd/commands/*.md` |
| Specialist agents (planner, executor, verifier, consistency checker, bibliographer, referee and others) | 24 | `src/gpd/agents/*.md` |
| Method protocols (statistics, reproducibility, machine learning for physics, DFT and others) | 56 | `src/gpd/specs/references/protocols/*.md` |
| Domain verification files / subfield files | 13 / 16 | `src/gpd/specs/references/verification/domains/`, `src/gpd/specs/references/subfields/` |
| Checks in the verification registry | 14 universal (5.1 to 5.14), 10 contract checks (5.15 to 5.24) and 25 error-class entries (IDs from 1 to 104) | `src/gpd/core/verification_checks.py`; the 104 error classes are described in `src/gpd/specs/references/verification/errors/` |
| MCP servers | 9 entry points; the installer wires 8 (the Wolfram bridge is not built in) | `pyproject.toml`, `src/gpd/mcp/builtin_servers.py` |
| Files under `src/` / Python files under `tests/` | 844 / 537 (478 named `test*.py`) | |

The 71 and 24 come from the folders; the README does not state them.

**What it claims about itself.** "GPD is built to favor scientific rigor and critical thinking over agreeability"
(`README.md` line 95). "GPD is a scalpel, not an autopilot. Treat each agent turn like a graduate student's work"
(line 97). Two figures have no source in GPD and should not be quoted as measured: the universal checks "catch ~60% of
all physics errors" (`src/gpd/specs/references/verification/core/verification-core.md` line 16), and dimensional analysis
"catches ~30% of errors" (`src/gpd/commands/dimensional-analysis.md` line 46).

**What its code does.** The checking is done by the AI agent writing and running its own code. The MCP verification
tools record and route; they compute no physics. `run_check` is "non-authoritative" (verification_server.py line 2151).
`dimensional_check` (line 4908) compares bracket annotations such as `[M][L]^2[T]^-2`. `limiting_case_check` returns
`"status": "documented"` and says to use SymPy (line 5024).

**How it is used.**

- Install: `npx -y get-physics-done` (`README.md` line 33), or for Claude Code in one folder,
  `npx -y get-physics-done --claude --local` (`docs/claude-code.md` line 45). It needs Node.js 20+, Python 3.11+ with
  `venv`, and a supported agent runtime (README line 120). GPD "does not install your runtime or provide model access,
  billing, or API credits" (lines 76-77).
- Inside Claude Code: `/gpd:help`, `/gpd:start`, `/gpd:tour`, then `/gpd:new-project --minimal` or `/gpd:map-research`,
  then `/gpd:resume-work` (README line 245).
- Research loop: new-project, discuss-phase, plan-phase, execute-phase, verify-work, complete-milestone (README line 350).
  Its four stages are Formulate, Plan, Execute and Verify (lines 258-261).
- Analysis commands that can run on one explicit target without a full GPD project (README line 375): derive-equation,
  dimensional-analysis, limiting-cases, numerical-convergence and sensitivity-analysis (outputs under `GPD/analysis/`),
  and parameter-sweep (outputs under `GPD/sweeps/`). The README says error-propagation and graph are not in this group.
  compare-experiment, compare-results and regression-check are not in it either.
- Before an unattended run: `gpd validate unattended-readiness --runtime claude-code --autonomy supervised`
  (`docs/claude-code.md` line 75).

## What this project already borrows

| Here | From GPD | What changed |
| --- | --- | --- |
| The five questions, `docs/AGENTS.md` "Before you say done" (lines 39-59) | "The five-item gate" in `src/gpd/specs/references/shared/reward-hacking-self-check.md` (line 22): literal vs. spirit, cheap wins, adversarial self-review, uncertainty disclosure, revise or refuse | Reworded and shortened; the examples are this project's Lessons rows. Added in commit 48ee231 (2026-11-26). |
| The wording table, `docs/AGENTS.md` lines 64-75 | "S4. Confidence-to-language mapping", same file (line 127): four levels, from INDEPENDENTLY CONFIRMED (direct statement) to UNRELIABLE (do not present) | Same idea; every row is this project's (grouped paired bootstrap, every-seed agreement, exploratory, rehearsal). |
| The `[UNVERIFIED - from memory]` tag, `docs/AGENTS.md` question 4 | `[UNVERIFIED - training data]` in `src/gpd/agents/gpd-executor.md` (line 443) and `src/gpd/specs/references/execution/executor-numerical-protocol.md` (line 69) | Renamed. The old credit said it came from the self-check file, which does not contain it. |
| The columns of the decision table, `docs/DECISIONS.md` | `src/gpd/specs/templates/decisions.md`: ID, Decision, Rationale, Alternatives Considered, Phase, Date, Impact; "Append-only" | Date moved second, Rationale renamed Why, Phase dropped, Impact written as prose, an Evidence column (commit and file) added. |

Not from GPD: the Lessons table (LES rows) is this project's own. GPD's nearest equivalent is its backtrack ledger
(`src/gpd/specs/workflows/record-backtrack.md` line 43). DEC-002 (predictions, measurements and DFT calculations named
apart) is close to S3 in the self-check file (line 111), but was decided here on 2026-11-20, six days before the
GPD-based text was added (48ee231).

**Attribution.** Before this change, `docs/AGENTS.md` and the `docs/DECISIONS.md` header named only "get-physics-done" and
"Apache-2.0". Section 4 of the licence asks anyone who redistributes the Work or a Derivative Work to (a) give recipients
a copy of the licence, (b) "cause any modified files to carry prominent notices stating that You changed the files" and
(c) keep the copyright and attribution notices (`LICENSE` lines 89-104). Both credits now name GPD, the URL, version 1.2.2
and commit `0f417695`, the copyright holder, the licence with a link to its text, the exact GPD files, and say the text
here was changed. [Likely] commit `0f417695` is the one the earlier text was adapted from: that text was committed on
2026-11-26 (48ee231), and main had not moved from `0f417695` between 2026-05-28 and 2026-11-27. Still open: this
repository has no LICENSE file and no copy of the Apache text, only a link. [Speculative] a link may not meet 4(a) if the
repository is redistributed; whether reworded checklist text is a Derivative Work at all is a legal question. This is
not legal advice.

## Worth borrowing next

Each item copies a written protocol into this repository's own files; none needs GPD installed. Effort figures are
judgement, not measured.

| # | What | Effort | Risk | Priority |
| --- | --- | --- | --- | --- |
| B1 | Written registration with forbidden proxies, before the next cycle | about 1 h | low | High |
| B2 | Invariance and physical-bounds counts in every nightly | 2-3 h | low | High |
| B3 | One test value checked from the analysis to the website files | about 2 h | low | High |
| B4 | Input sweeps on every live simulation | half a day | low | High |
| B5 | A replay job each night and a provenance file per dataset | about 2 h | low | High-Medium |
| B6 | Flag inputs outside the training data on family and simulation pages | half a day | medium | Medium-High |
| B7 | Count the comparisons made each night | 1-2 h | process | Medium |
| B8 | A known-answer gate at the start of each night | about 2 h | low | Medium |
| B9 | Citation rules for references and Perplexity leads | minutes | none | Medium |
| B10 | Two rules left out of the borrowed files | minutes | process | Medium |
| B11 | Residuals and an uncertainty budget per family | half a day | low | Medium-Low |

### B1. Written registration with forbidden proxies

- **From:** `src/gpd/specs/templates/project-contract-schema.md`: claims, acceptance tests with a pass condition, and
  `forbidden_proxies` entries `{id, subject, proxy, reason}` (lines 133-140 and 202). GPD can validate a file with
  `gpd validate project-contract <file.json>` (README line 523).
- **Change here:** DEC-019 says the cycle-3 rules lived only in committed code and that a written registration with named
  claims was "Not done for this cycle; noted here so the next cycle can do it before starting." Before the next cycle,
  commit a hand-written file (for example `docs/registrations/<cycle>.json`; the folder does not exist yet) with one claim
  per task, the pass condition already in the analysis code and the wording table (the 95% interval of the paired
  bootstrap over whole groups excludes zero; for a direct statement, for the average and for every seed), and forbidden
  proxies taken from the Lessons table: an average of several networks against one tree model (LES-010), a split by
  exact formula, or one with near-copies on both sides, behind a claim about unseen material (LES-001, LES-011), an
  untuned baseline (LES-003), models trained for different lengths (LES-013), and a DFT-calculated value presented as a
  measurement (DEC-002).
- **Risk:** low. The schema is closed ("Do not invent extra keys", line 189) and written for theory claims, so copy its
  shape; do not say it validates. Never fill it in after results are seen.
- [Likely] the cheapest way to close the gap DEC-019 names.

### B2. Invariance and physical-bounds counts every night

- **From:** `src/gpd/specs/references/protocols/machine-learning-physics.md`, Step 6 item 3: "Apply symmetry
  transformations to the input and verify the output transforms correctly" (line 85), and "Report sigma_rot as a model
  quality metric" (line 405). Checks 5.8 (physical plausibility) and 5.12 (positivity constraints) in
  `src/gpd/core/verification_checks.py` (lines 153 and 193). "Never silently replace `NaN` or `Inf` ... or proceed with
  placeholder values" (`executor-numerical-protocol.md` line 35).
- **Change here:** `gpu/structure/invariance.py` already measures four rewrites: atoms relisted, origin shifted, crystal
  rotated, and a 2x2x2 supercell for cells of up to 10 atoms. Its jobs are planned only in `tiefix_jobs`
  (`gpu/pipeline/plan.py` lines 315-356, estimated at 3 or 6 CPU minutes per property); `gpu/pipeline/nightly.py` has
  none. Add one invariance job per retrained crystal task to the nightly plan. Add a count, not a clip, of predictions
  outside physical bounds per task: band gap below 0 eV (band gap and formation energy are trained untransformed,
  `gpu/structure/train_structure.py` lines 39-44, so a negative band gap is possible), dielectric constant below 1, and
  similar. The moduli, dielectric constant, corrosion rate and conductivity are predicted as log10 and cannot go
  negative (`gpu/structure/train_structure.py` for the moduli; the report's family table for the others).
- **Risk:** low. Fix the failure threshold in code before results are seen. Using the count to change
  `gpu/recipes.json` needs a reviewed commit (DEC-018).
- [Likely] it turns the hand check that found LES-017 into a standing guard. Under the fixed neighbour rule the check
  passes today, on one seed per property and 300 test crystals each (LES-019: no rewrite moved a prediction by more
  than 1% of the spread).

### B3. One test value from the analysis to the website

- **From:** `src/gpd/agents/gpd-consistency-checker.md` line 23: "For every cross-phase transfer, verify physical
  meaning, units or dimensions, one concrete test value, and convention alignment."
  `src/gpd/specs/references/verification/meta/verification-independence.md`: the verifier is not given the summaries,
  because "'claimed convergence' is not evidence of convergence" (line 45), and must re-derive (line 54), not
  pattern-match (line 79).
- **Change here:** numbers travel from the analysis output to the report, through `gpu/export_site.py` to
  `public/pipeline/index.json` (not in this checkout until the next export), and on to the pages in
  `src/components/sim/`. `npm run pipeline-check` (`scripts/check_pipeline_models.mjs`) already checks that the browser
  walks the exported trees as scikit-learn predicted them. `gpu/tests/test_export_site.py` has three tests (which side
  is better, the row sample, rounding). None compares a reported number with the exported one. Add a test that takes
  one number per task (test MAE, interval half-width or coverage) from the analysis output and asserts the same value
  and unit in `index.json` and in the report table. After each night, a fresh agent recomputes two or three headline
  numbers from the job outputs without reading the report text.
- **Risk:** low. The agent pass uses Claude usage, not money.
- [Likely] the cheapest guard against the site showing a number that has drifted from the analysis.

### B4. Input sweeps on every live simulation

- **From:** `src/gpd/specs/references/analysis/physics-validation-recipes.md`, "Parameter Sweep Checks" (line 61):
  "Feature detection checks extrema, rapid-change regions, monotonicity, asymptotics, NaN/Inf values, non-physical
  values, and identical-output bugs" (lines 69-70). "Sensitivity Checks" (line 75): compute the elasticity
  `S_i = (partial f / partial p_i) * (p_i / f)` and flag `|S| > 100`, sign changes and null directions (lines 77-84).
- **Change here:** `scripts/sweep.mjs` (`npm run sweep`) already checks the concrete network and trees for two trends
  (more water at the same binder gives weaker concrete; older concrete is stronger) and fails only if the network gets
  the direction wrong on most mixes. It reads `public/model/nn_data.json` and `known.json`, two of the blocked files
  (`CLAUDE.md`), so build on its approach but put the page sweeps in a new script beside it that reads only
  `public/pipeline/` and the page code (the pages take every number from `public/pipeline/`:
  `src/components/sim/README.md` line 5). `package.json` has no test runner (no vitest); checks are plain node scripts run through npm scripts. Sweep
  every input of every page in `src/components/sim/` over its range on the page, with the other inputs at their
  defaults. Fail on NaN or Inf, on values outside physical bounds, and on an input that changes nothing across its whole
  range. Report the elasticities, and count direction violations row by row instead of asking for "most".
- **Evidence that sweeps find things.** Probe scripts kept outside the repository were re-run for this document on
  2026-11-27. They read only `public/pipeline/corrosion_model.json`, `public/pipeline/electrolyte_model.json`,
  `public/model/model.json`, `public/model/model_low.json` and the mix amounts in `lab/data/train.json`.
  - Exported electrolyte trees: with lithium removed, the predicted ionic conductivity goes up for 23% of 478
    compositions, all of which contain lithium (down for 76%).
  - Live concrete trees (`train/train.py`): predicted strength falls with age somewhere for 8 of 181 mixes (by up to
    1.13 MPa). Adding 0.05 to water/binder at the same binder raises predicted strength in 12.3% of 724 pairs (by up to
    6.73 MPa). The cautious 10th-percentile model sits above the expected-value model at 75 of 1,629 sweep points (4.6%,
    by up to 5.65 MPa). Exploratory, like every concrete number in this project: this tree model was trained before the
    sealed rows were split off.
  - Exported corrosion trees: the predicted rate falls somewhere between 20 and 100 °C for 283 of 311
    material/environment pairs (91%), but most falls are small. The median largest fall between 10 °C steps is 0.012
    log10; 17 pairs (5%) fall by more than 0.1 log10 (a factor of 1.26), the largest by 0.31 log10. The apparent
    activation energy from 25 to 75 °C has a median of 5.8 kJ/mol (10th to 90th percentile 1.2 to 12.5), so the
    predicted response to temperature is weak: at the median the rate rises by a factor of about 1.4 from 25 to 75 °C
    (computed here from that median). This is a flag, not proof of an error: the review notes, without a source, that in
    aerated open systems the rate can peak or fall with temperature.
- **Risk:** low. Keep direction checks as warnings until each expected trend has a cited source.
- [Certain] the exported models give the outputs above (the second reviewer and this re-run reproduced them).
  [Likely] a sweep per page is how to keep such cases visible as the models change.

### B5. Replay noise and data provenance

- **From:** `src/gpd/specs/references/protocols/reproducibility.md`: re-run with the same seed and expect identical
  output, then with a different seed and expect agreement within error bars; "If step 3 fails, there is a
  non-determinism bug (... GPU non-determinism ...)" (lines 62-75). Reproducibility levels, and the minimum standard of
  "at least 2 independent seed sets" (lines 259-268). Every input dataset gets a source, URL, access date, SHA-256,
  licence and the transformations applied (lines 191-205).
- **Change here:** LES-019 says "Nothing yet measures replay noise ... index_add_ is not deterministic on the GPU".
  DEC-021 records one repeat of seed 11 that gave the same test MAE to four decimals (0.1756). Add one small replay job
  per night (one seed of one retrained task, rotating through the tasks) and print its difference next to the
  every-seed verdicts. Add a provenance file per `domains/<family>/` with the six fields; new datasets from the Perplexity
  leads would start with one. `gpu/common.py` (`hardware()`) already records the Python, platform and torch versions.
- **Risk:** low. It costs GPU minutes inside the nightly budget. On the GPU expect agreement within noise, not identical
  output.
- [Likely] per-seed differences cannot be read fully until replay noise is measured on more than one seed.

### B6. Flag inputs outside the training data

- **From:** `machine-learning-physics.md` Step 5 item 3: "The model must flag when it is extrapolating. Use: ensemble
  disagreement, latent space distance to training data, or input feature monitoring." (line 79). Step 6 item 4: map where
  the model breaks down (line 86). Pitfall: "ML models do not know they are extrapolating" (line 320).
- **State here:** part of this exists. `src/lib/alloyGen.js` uses L1 distance on element fractions to the nearest tested
  alloy, with tiers at 0.10, 0.20 and 0.30, and drops candidates beyond 0.40. `src/components/sim/ElectrolyteDesigner.jsx`
  shows the nearest measured compositions by L1 distance. 90% conformal intervals with measured coverage exist
  (`gpu/pipeline/analyze.py`, `conformal_split` and `conformal_cv`, lines 480-584). The gap: `clamped()` in
  `src/lib/nnFamily.js` (line 183) flags only a prediction that sits on the output clamp, and only when `clampOutput` is
  on (alloys). `inputClip` caps standardised inputs without telling the visitor: 3 on the alloy family page
  (`public/model/materials/family.json`), and in the GPU networks 3 for alloys and electrolytes and 5 for chemistry, both
  surfactant tasks and corrosion (`gpu/train_tabular.py` lines 36-45).
- **Change here:** for each family and simulation page, export a distance-to-training score (nearest-neighbour distance
  in scaled input space, or the spread across seeds for networks), set its threshold on validation rows only, and show
  "outside the training data" next to the prediction.
- **Risk:** medium. A threshold chosen on test rows breaks the `docs/AGENTS.md` rules. Seed spread exists only for
  networks, and the corrosion and electrolyte pages run trees.
- [Likely] useful; less urgent than B3 and B4, because the two designer pages already show the nearest measured
  neighbours.

### B7. Count the comparisons made each night

- **From:** `src/gpd/specs/references/protocols/statistical-inference.md` line 145, "The look-elsewhere effect (trials
  factor)", with `p_global ≈ p_local × N_trials` (Bonferroni, line 149), and "Ignoring look-elsewhere" in its table of
  common errors (line 256).
- **Change here:** the "At a glance" section of `docs/results/20261122T014943Z_overnight_REPORT_20261126T2001.md` alone
  gives 40 interval verdicts on averages, each at 95% (counted by hand: 9 table families, 4 crystal properties on each of
  4 splits, 4 generation searches, 1 search round, 6 fine-tuning and 4 neighbour-rule comparisons). If no difference were
  real, about 2 of 40 would still exclude zero by chance. The comparisons share data, so that is a scale, not a count of
  false results. Print the number of comparisons next to the verdict table, and optionally a family-wise column
  (Bonferroni or Holm).
- **Risk:** a column is only reporting. Using it to decide a verdict changes a pass condition, which needs the person's
  approval and a new `docs/DECISIONS.md` row before the next cycle (`docs/AGENTS.md`, "Stop and ask").
- [Likely] the every-seed rule already blocks most chance wins; the count makes the remaining risk visible.

### B8. A known-answer gate at the start of each night

- **From:** `src/gpd/agents/gpd-experiment-designer.md` line 157: "Pre-production sanity gate using less than 5% of the
  budget: dimensional analysis, known limit or benchmark, symmetry/conservation check when applicable, and a scaling
  smoke test"; line 227: "Pre-register the design before production runs".
  `src/gpd/specs/references/execution/executor-deviation-rules.md`, "Escalation 3": a forced stop after 3 failed
  adjustments (lines 122-124).
- **Change here:** rehearsal and smoke runs only show that the code runs; their numbers mean nothing (`docs/AGENTS.md`).
  Add a 2 to 5 minute gate before training: each trainer fits a small synthetic target with a known function to a fixed
  error, and the invariance check runs on a few crystals. A family whose gate fails is skipped and reported, not trained
  for hours.
- **Risk:** low; the thresholds are fixed in code beforehand.
- [Likely] cheap insurance. How often it would fire is not known.

### B9. Citation rules for references and Perplexity leads

- **From:** `reward-hacking-self-check.md` S2 (lines 98-109): cite a source only if it is real and supports the specific
  claim; otherwise drop it or insert a `MISSING:` placeholder (line 105). `src/gpd/agents/gpd-bibliographer.md`:
  "Re-check cache hits for claim-content match, not just existence" (line 75), "Do not add unverified citations"
  (line 125), "Do not guess arXiv IDs, DOIs, or texkeys" (line 127).
- **State here:** `research/README.md` already says a factor without an entry does not ship, and
  `research/perplexity/ROUND1_DIGEST.md` calls its contents "leads, not findings". New: each row in
  `research/references/*.md`, and each dataset taken from the Perplexity answers, names the specific claim its source
  supports, and says `MISSING:` or "not checked" where it has none.
- [Likely] little downside: it extends the existing `research/README.md` rule to name the claim each source supports.

### B10. Two rules left out of the borrowed files

- **From:** `executor-numerical-protocol.md` line 72: "Reduce confidence by one level for any result that depends on an
  unverified benchmark." `reward-hacking-self-check.md`, item 4: if no uncertainty is found, "enumerate at least three
  failure modes" (line 66).
- **Change here:** under the wording table, a result that rests on an `[UNVERIFIED]` value takes the next weaker row. In
  question 4, "nothing found" is answered with three named ways the result could be wrong. Both change rules in
  `docs/AGENTS.md`, so the person decides.
- [Likely] cheap, and both fit the rules already there.

### B11. Residuals and an uncertainty budget per family

- **From:** `src/gpd/specs/workflows/compare-experiment.md`, "3d. Residual analysis" (lines 180-198): trending residuals
  suggest a wrong functional form, a constant offset a normalisation error. `src/gpd/specs/templates/uncertainty-budget.md`,
  "Dominant Uncertainty Analysis" (line 63). `machine-learning-physics.md`: a model trained on DFT data inherits the
  functional's systematic errors (line 318). `src/gpd/specs/references/protocols/density-functional-theory.md` line 62:
  "The KS gap systematically underestimates the true gap."
- **Change here:** in the report, per family, plot validation-row residuals against the main physical input (age,
  temperature, tail length, a composition fraction), and give a short budget: model error against the scatter of repeat
  measurements where repeats exist, and, for DFT-calculated targets, against the functional's own error. The band-gap
  label already says "OptB88vdW (a known underestimate of real gaps)" (`gpu/structure/train_structure.py` line 44).
- **Risk:** low; validation rows only. A choice taken from test rows would be exploratory.
- [Likely] shows how far model error sits above the data's own noise.

**Small and optional.** A per-task process status kept apart from the verdict, as in
`src/gpd/specs/references/verification/verification-status-authority.md` ("Do not use top-level `failed` for a
scientific or evidence gap", line 17). LES-015 was fixed in `update_registry`, and the `index.json` that
`gpu/export_site.py` writes already carries `status` ("ok", "missing", "error"), so only the words need defining once in
`docs/AGENTS.md`. Also, the GPD decision template's High/Medium/Low impact tag and the backtrack ledger's `category` and
`reverted_commit` fields could be used in new rows; adding them to old rows would break "never edit a row".
[Speculative] small gains.

## Not recommended

**Installing GPD into this repository or onto a shared machine.** [Likely] the cost is higher than the gain:

- It writes agent settings: `.claude/settings.json` hooks, including a SessionStart update check that calls the network
  (`ensure_update_hook` in `src/gpd/adapters/install_utils.py` line 3200; `urllib.request.urlopen` in
  `src/gpd/hooks/check_update.py` line 172), a status line, and `.mcp.json` with 8 MCP servers, one of them an arXiv
  bridge (`src/gpd/adapters/claude_code.py` lines 320-322).
- It installs a Python environment from PyPI at `${GPD_HOME:-~/.gpd}/venv` (README line 222) and creates a `GPD/`
  project tree (PROJECT.md, ROADMAP.md, STATE.md, state.json). Uninstalling leaves `GPD/` and the GPD home in place
  (README line 597).
- Several agents work this repository in parallel, and `docs/AGENTS.md` asks for the person's approval before installing
  into an environment someone else also uses.
- GPD's process (phases, ROADMAP, STATE) would compete with this repository's (`docs/DECISIONS.md`, `gpu/recipes.json`
  changed only by a reviewed commit, the nightly pipeline).
- Its tools would not check this project's numbers: they are "non-authoritative" (see above).
- Its convention lock has 18 keys, almost all field-theory notation (metric signature, gauge, gamma matrices and so on;
  `src/gpd/core/conventions.py` lines 68-87). Only `natural_units` touches this project.
- Its agents are written for derivations and manuscripts, not for benchmarking table and crystal-graph models.

If someone wants to try it, ask the person first, then use a throwaway copy outside the shared repository:
`npx -y get-physics-done --claude --local`, run `/gpd:peer-review` (a "staged six-pass peer review" that accepts a `.md`
file: `src/gpd/commands/peer-review.md` lines 3 and 23-32) on one finished report, then
`npx -y get-physics-done --uninstall --claude --local` (the pattern in README line 595) and delete the copy. GPD itself is
free; the review uses Claude usage.

Also not recommended: quoting GPD's "~60%" or "~30%" figures, or citing a passed `gpd validate` as evidence that a result
is right.

## Not checked

- GPD was not installed or run; no command or validator was tested.
- Line numbers are at commit `0f417695` and will move on later commits.
- The GitHub file pages were not opened (they returned 403 through the session's proxy). Each per-file URL is built from
  the commit, and each path was confirmed to exist in the clone. The raw `LICENSE` file at that commit did load (HTTP 200)
  for the second reviewer.
- The remote main commit and the tags come from the second reviewer's `git ls-remote`; not repeated here. Which commit
  the v1.2.2 tag points to was not checked.
- A search snippet lists GPD as MIT on a mirror site; it was not opened. GPD's own `LICENSE` says Apache-2.0 and governs.
- About 30 GPD files and every top-level file except `uv.lock` were read, not all 844 files or the tests. Subfield
  material outside this project's scope was skipped.
- Effort figures and priorities are judgement. The licence notes are not legal advice.

## Next actions, ranked

The order weighs effort as well as the priority column: B9 takes minutes and must come before any Perplexity lead is
trained on, so it comes before B5; B6 waits because the two designer pages already show the nearest measured neighbours.

1. Done in this change: the credits in `docs/AGENTS.md` and `docs/DECISIONS.md` now name GPD, its URL, version 1.2.2 and
   commit `0f417695`, the copyright holder, the licence with a link, and the right GPD files. [Certain]
2. Before the next cycle starts, write and commit its registration with forbidden proxies (B1). [Likely]
3. Add the one-test-value check from the analysis to `public/pipeline/` (B3). [Likely]
4. Put the invariance job and the physical-bounds count in the nightly plan (B2). [Likely]
5. Add input sweeps of every simulation page in a new script beside `scripts/sweep.mjs`, starting with the electrolyte
   lithium-removal case (B4). [Likely]
6. Apply the citation rules to the Perplexity leads before any of them is trained on (B9). [Likely]
7. Add the nightly replay job and a provenance file per family (B5). [Likely]
8. Ask the person about the step-down rule and the comparison count (B10, B7); record any change in `docs/DECISIONS.md`
   before the next cycle. [Likely]
9. Later: the input-distance flag (B6), the known-answer gate (B8), residuals and the uncertainty budget (B11).
   [Speculative] on order.
10. Do not install GPD in this repository (see "Not recommended"). [Likely]
