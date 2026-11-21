# The lab: networks that learn from young concrete and are tested on old concrete

Small networks learn concrete strength from the youngest lab results only (up to 7 days). They are
rewarded for predicting what comes next (14 to 28 days). They have children in proportion to how good
they are, and none is ever deleted. A sealed set of older results (over 28 days) is opened exactly
once, after the models to be scored have been named in advance. So far this folder is only the
machinery and a smoke test. The long evolution is meant to run after physics pre-training is added.

## The three data tiers

| Tier | Ages | Rows | Where | Used for |
| --- | --- | --- | --- | --- |
| TRAIN | 1, 3, 7 d | 253 | `lab/data/train.json` | the only rows any network trains on |
| SELECT | 14, 28 d | 481 | `lab/data/select.json` | the reward, so it is also what selection is tuned to |
| FUTURE | 56 to 365 d | 271 | `lab/vault/future.json` (sealed) | one final score, once |

All 1,005 rows come from `public/model/nn_data.json` (the page's train and test parts together). Of
the 271 FUTURE rows, 263 come from a mix (the same seven ingredient amounts) that was also tested in
TRAIN or SELECT, and **8 do not**. The row indices of each group were fixed when the vault was sealed.
With 8 rows, the "never seen" score will be very noisy. On SELECT, 48.6% of rows come from a mix that
has TRAIN rows.

## The reward

`reward = -(MAE on SELECT, MPa) - lambda * violations`, with `lambda = 2` MPa per unit share.

- `violations` = the share of TRAIN mixes on which predicted strength falls anywhere along the ages
  1, 3, 7, 14, 28, 56, 90, 180, 365 days (a drop of more than 1e-6 MPa counts), **plus** the share
  of those sweep predictions (mixes × 9 ages) that are below zero.
- Lambda = 2 means that breaking the age rule on every training mix costs as much as being 2 MPa less
  accurate, and breaking it on 10% of mixes costs 0.2 MPa. The archive stores the MAE, the R², both
  violation shares and the total, so the reward can be recomputed with another lambda.

## The selection rule and the no-deletion rule

- Generation 0 is N random genomes. Each later generation makes K children.
- Each parent is drawn from the **whole archive** (every individual ever trained, as it stood when the
  generation began) with probability proportional to `exp(reward / T)`, with `T = 0.5` MPa. A network
  0.5 MPa better is e ≈ 2.7 times as likely to be picked, and one 2 MPa better about 55 times. Nobody
  is removed, so a weak line can still have children.
- A child is its parent's genome after `mutate()`. If its layer shapes still match the parent's (same
  hidden sizes and same input set), it starts from the parent's trained weights (Lamarckian
  inheritance, with a new Adam run and a new seed) and then trains on TRAIN. Otherwise it starts fresh
  from its own seed.
- **Nothing is deleted or overwritten.** `lineage.jsonl` is only appended to. Weights files are
  written with flag `wx`, or accepted only if an identical file is already there, which covers a
  resume after a crash. `guard.mjs` refuses `unlink`, `rm`, `rename`, `truncate` and any overwrite
  under `lab/archive*/` at runtime. `check_isolation.mjs` also scans every lab script for those calls.
- The plan for generation g depends only on the run seed and on generations before g. So a run is
  deterministic, and rerunning the same command resumes it. `npm run lab:smoke` checks this against
  the committed `archive_smoke/lineage.jsonl`.

## The vault protocol: seal, evolve, pre-register, open once

1. **Seal.** `make_splits.mjs` writes the three tiers, the vault, `vault/SEALED.json` (SHA-256, row
   count, creation time, seen and unseen row indices) and a copy of that seal outside the vault,
   `seal_receipt.json`. A second run only checks the hash.
2. **Evolve.** Every lab script except `open_vault.mjs` imports `guard.mjs` first. The guard makes
   every `node:fs` read of `lab/vault/` throw, and also every read of the full dataset files that
   contain FUTURE rows (only `net_parity.mjs` may read `nn_data.json`, to prove parity).
3. **Pre-register.** `preregister.mjs` writes `lab/preregistration.json` once. It names the champion by
   reward, the best individual without physics inputs, the best individual that did not inherit
   weights, and every baseline. Each model's SHA-256 is recorded, together with the vault's SHA-256,
   the lineage hash and the git commit.
4. **Open once.** `open_vault.mjs` refuses to start unless the pre-registration exists and every hash
   matches. It writes the marker `results/VAULT_OPENED` before scoring, then scores each listed model
   once on FUTURE: overall, seen and unseen mixes, mix groups, and age bands. The result goes to
   `results/vault_opening.json`. Any later run only prints that stored result. **It has not been run.**

## Files

| File | What it does |
| --- | --- |
| `make_splits.mjs` | Splits the 1,005 rows into TRAIN, SELECT and FUTURE by age and seals FUTURE with its SHA-256. A second run writes nothing and only checks the seal. |
| `guard.mjs` | Wraps the `node:fs` read functions so that any path in `lab/vault/` (and the full dataset files) throws, and blocks deleting or overwriting archive files. Every script except `open_vault.mjs` and `make_splits.mjs` imports it first. |
| `check_isolation.mjs` | A runtime test that the guard throws on every read route into the vault and on every destructive write into an archive. A static scan that no lab code file mentions the vault, reads the full data, deletes, or writes outside `io.mjs`. |
| `io.mjs` | `sha256`, `readJson`, `writeNew` (fails if the file exists) and `writeSame` (accepts an existing file only if it is identical). It is the only way lab scripts write files. |
| `net.mjs` | The trainer: `seeded()` and `Mlp` copied from `src/lib/nnFamily.js`, and the input sets `base`, `rich` and `physics` copied from `src/lib/nn.js`, with squared or Huber loss. It adds `getWeights`, `setWeights` (warm start), `fromWeights`, and a `pretrain(rows)` seam that does nothing yet. |
| `net_parity.mjs`, `allow_full_data.mjs` | Trains `nn.js` and `net.mjs` side by side on the page's split and requires identical predictions on all 1,005 rows. The second file is the only permission to read `nn_data.json`. |
| `genome.mjs` | A genome (`hidden`, `lr`, `decay`, `loss`, `epochs`, `features`, `pretrain`, `seed`), `randomGenome`, and `mutate`, whose step sizes are documented in the file. `genomeId` is a 12-character SHA-256 of the genome. |
| `reward.mjs` | The reward: SELECT MAE plus `lambda` times the physics violations, with both parts returned separately. `LAMBDA = 2`. |
| `archive.mjs` | Opens an archive folder: the append-only `lineage.jsonl` plus write-once `weights/<id>.json.gz`. A weights hash is the SHA-256 of the uncompressed JSON, checked on every load. |
| `evolve.mjs` | The evolution loop: selection over the whole archive, Lamarckian warm starts, deterministic and resumable. Options: `--archive --seed --N --K --generations --T --lambda`. |
| `baselines.mjs`, `trees.py`, `curve.mjs` | Train the plain network, the site's physics switch, gradient-boosted trees and an age curve with no learning on TRAIN, and score them on SELECT. Each model is saved once in `results/models/`, and `results/baselines_select.json` records its hash. |
| `preregister.mjs` | Names the models that will be scored on FUTURE, with their hashes and the vault hash, and writes `preregistration.json` once. `--dry-run` prints instead, and `--out` writes elsewhere. |
| `scoring.mjs` | Re-hashes each pre-registered model file and scores it on a set of rows, overall, by seen and unseen mixes, by mix group and by age band. `open_vault.mjs` uses it. |
| `rehearse_scoring.mjs` | Runs `scoring.mjs` on SELECT for any pre-registration and checks the MAE equals the registered SELECT MAE. It tests the opening's code path without the vault. |
| `open_vault.mjs` | The single opening described above. It is written but has never been run. |
| `smoke.mjs` | `npm run lab:smoke`: splits, baselines, a 14-network evolution in a fresh temporary archive, the isolation check, and a determinism check against `archive_smoke/`. |

### The baselines exactly

- **plain:** `createNet(TRAIN, {})`, the page's default network (64-32, base inputs, squared loss,
  300 epochs, seed 11).
- **physics:** `PANEL_DESIGN` from `src/lib/nn.js` (four textbook laws as inputs, Huber, decay 0.003).
- **trees:** `GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
  subsample=0.8, random_state=42)` on the nine features, run with `python3 -I`, pickled once.
- **curve:** no learning. Strength at age t is `s28 · exp(0.25 · (1 − √(28/t)))`, the fib/CEB-FIP
  curve for ordinary cement. `s28` is one number per mix, the least-squares fit to that mix's own
  TRAIN rows. A mix with no TRAIN rows gets the `s28` fitted to all TRAIN rows. **curve_global**
  gives every mix that all-TRAIN `s28`. The literal "mean TRAIN strength at the same age" cannot be
  computed, because TRAIN has no rows at 14 or 28 days.

## Smoke-test numbers (`npm run lab:smoke`, run seed 1)

| Model on SELECT | MAE, MPa | Violations | Reward |
| --- | --- | --- | --- |
| Plain network | 12.467 | 0.179 | −12.825 |
| Physics switch (PANEL_DESIGN) | 7.552 | 0.162 | −7.876 |
| Gradient-boosted trees | 13.035 | 0.006 | −13.046 |
| Age curve, per mix | 9.480 | 0.000 | −9.480 |
| Age curve, one for all mixes | 10.794 | 0.000 | −10.794 |
| Best of 14 evolved (N=6, K=4, 2 generations) | 6.513 | 0.026 | −6.566 |

- The best of the 14 is a random genome from generation 0 (base inputs, Huber loss, 69-53, 280
  epochs). Its two warm-started children scored −6.895 and −6.897, so no child beat it in 2
  generations.
- Trees cannot extrapolate in age: past 7 days they predict a flat line. Many random networks
  predict strength *falling* after 7 days (falling share 1.0), and the reward punishes that.
- These numbers come from 14 networks. They show that the machinery works, not what evolution can do.

## Archive size

Weights are stored as gzipped JSON of float64 values, about 9.6 bytes per weight. A random genome
has about 2,600 weights on average (about 25 kB), and the smoke run averaged 3,591 (35 kB). **500
individuals ≈ 12 to 18 MB**, plus about 0.35 MB of lineage. The ceiling, if every network drifts to
128-128 with rich inputs, is about 180 kB each, or 90 MB. For long runs, consider keeping
`archive/weights/` out of git: the lineage holds every hash.

## Run it

```sh
npm run lab:smoke        # splits (if needed), baselines, 14-network evolution, isolation check
npm run lab:parity       # net.mjs vs src/lib/nn.js, must be identical
npm run lab:check        # isolation check alone
npm run lab:evolve -- --archive lab/archive --seed 1 --N 24 --K 12 --generations 10
npm run lab:preregister  # after evolution; writes lab/preregistration.json once
npm run lab:rehearse     # scores the pre-registered models on SELECT (no vault)
node lab/open_vault.mjs  # once, at the very end
```

## Phase 1 limits, and what phase 2 changed

- Phase 1 left `pretrain(rows)` as a no-op seam. Phase 2 implements it (below). Phase-1 scripts still
  call it with no rows, so the smoke test is unchanged.
- The guard covers `node:fs` inside Node only. The Python scripts are kept out of the vault by the
  static scan, and receive FUTURE rows only through stdin from `open_vault.mjs`.
- `preregister.mjs` and `open_vault.mjs` were rewritten for phase 2. `scoring.mjs` and
  `rehearse_scoring.mjs` belong to phase 1 and are superseded by `open_vault.mjs --dry-run`.

# Phase 2: physics pre-training, research-backed evolution, the full baseline set

The rules follow the lab's methods research ("Recommended design"). A table at the end of this
section lists the departures.

## Data

- **SELECT is split by mix group** (`split_select.mjs`, seed 2027):
  - **REWARD** (321 rows, 199 groups) is the only set the reward sees.
  - **MONITOR** (160 rows, 100 groups) is never used for selection. It is read only for the
    monitor-gap stopping rule and for reporting.
- **Pre-training rows:**
  - `physics`: 1,800 synthetic rows from `physics/synthetic.mjs`. These are 300 synthetic mixes (convex
    combinations of two TRAIN mixes with the same ingredients present), 6 log-uniform ages each from
    1 to 365 days, per-mix random tau, beta and n, and 5% noise.
  - `boxcrete`: the 395 BOxCrete concrete rows at 22 C (79 mixes, ages 1 to 28 days, MIT licence,
    `external/LICENCE.md`).
  - `both`: the two sets together.
- **Generators:** Powers' gel-space law with Mills' a_u, a generic tau of 13 h and beta of 0.8, and
  efficiency factors k_s(t) and k_f(t). The fib curve is the second, simpler generator. Each has one
  scale, fitted on TRAIN only: Powers 0.589 (TRAIN RMSE 5.99 MPa), fib 0.622 (TRAIN RMSE 6.60 MPa).
  Every parameter's confidence label and citation is in `PARAMS` and in `results/baselines2.json`
  under `physics`.
- **Pre-training settings:** 40 epochs, lr 0.004, decay 1e-4, squared loss. One input and target
  scaling, computed from TRAIN plus the pre-training rows, serves both stages. Pre-trained weights are
  cached in `lab/cache/` per (layers, features, kind, pre-training seed). The cache is gitignored and
  is rebuilt deterministically when missing.

## Evolution rules

| Rule | Implementation |
| --- | --- |
| Seeds | 2 per individual. Fitness pools both seeds' squared errors. |
| Fitness (lower is better) | REWARD RMSE + bootstrap SE (200 fixed mix-group resamples) + 1 × (V_age + V_wb) in MPa. |
| Physics filter | More than 5% of query pairs violated means no children (the individual stays in the archive). |
| Parents, 16 of 20 | p ∝ exp(−(rank−1)/τ) / √(1 + children so far), τ = max(3, 0.1 N), over the whole archive. |
| Parents, 4 of 20 | Uniform over the MAP-Elites cell elites (size bucket × pretrain kind × feature set). |
| Mutation | One per child: continuous 0.5 (×0.8 or ×1.2, or a prior resample with p 0.2), layer 0.2 (×0.8 or ×1.25), discrete 0.3. |
| Inheritance | Only when layers, features and pretrain kind are unchanged, and then with p 0.5: start from the parent's weights and train 30% of the epochs. |
| Champion | Ladder rule: a challenger needs loss < champion's − the paired bootstrap SE. |
| Stopping | 25 generations, or 5 generations without a new champion (Ladder), or the monitor gap. |
| Random-search control | `archive_random/`: as many individuals as evolution, all drawn from the prior, scored and picked the same way. |
| Parallelism | 3 worker processes. The same lineage results with 2 or 3 workers (checked). |

## Results so far (run seed 1; the vault is still sealed)

- **Evolution stopped after generation 13** on the Ladder rule: no new champion for 5 generations.
  It trained 280 individuals (560 networks) in 46 s of generation time; the random search took 75 s.
- **The champion** is `g008-000-428625cae26f`, with genome: hidden 51-18, lr 7.4e-4, decay 1.2e-3,
  Huber loss, 177 epochs, rich inputs, physics pre-training. It inherited weights from its parent.
  It scored REWARD RMSE 7.12 (SE 0.38) and MONITOR RMSE 7.43.
- **The random-search champion** is `g006-001-0eb67f2ec90e`: 18-24 units, rich inputs, both kinds of
  pre-training. It scored REWARD 7.29 and MONITOR 7.91.

All baselines below were trained on TRAIN only. REWARD scores of evolved models are optimistic,
because evolution selected on REWARD.

| Model | REWARD RMSE | MONITOR RMSE |
| --- | --- | --- |
| Physics equation (Powers, calibrated on TRAIN) | 10.31 | 9.66 |
| Physics equation (fib) | 9.93 | 9.08 |
| Gradient-boosted trees | 14.70 | 15.83 |
| Random forest | 15.80 | 16.75 |
| Log-age ridge | 8.90 | 8.63 |
| Plain network, median of 10 seeds | 14.12 | 14.24 |
| PANEL_DESIGN, median of 10 | 10.40 | 12.14 |
| Plain network + physics pre-training, median of 10 | 7.72 | 8.57 |
| Champion retrained from scratch, median of 10 | 7.92 | 8.30 |
| Champion without pre-training, median of 10 | 21.65 | 21.00 |
| Champion as archived (2 seeds) | 6.87 | 7.21 |
| Random-search champion | 7.09 | 7.74 |
| Top-10 archive ensemble | 6.89 | 7.33 |

- **Physics pre-training does most of the work.** It takes the plain network from 14.2 to 8.6 MPa on
  MONITOR. Without it the champion's genome is unstable: its median is 21 MPa.
- **Evolution added little beyond that.** The champion retrained from scratch is 8.30 MPa on
  MONITOR, against 8.57 for the plain network with pre-training and 7.74 for the random-search
  champion.
- **The champion does not retrain to its archived score** (7.92 against 7.12 on REWARD). That is the
  gap Real et al.'s criterion warns about.

## Pre-registration and the dry run

- **Pre-registration.** `lab/preregistration.json` has SHA-256
  `4b67533ec706c4ecf3f70d70d5f855e388404a6cbf7f7b98c969344e9b1ae6b8`. It lists all 13 models with
  their hashes, the FUTURE hash, both lineage hashes, the champion rule, the metrics, and the
  success criterion. The criterion is that champion_retrained beats the best of: physics_powers,
  physics_fib, gbt, rf, ridge, plain and random_champion, with a paired mix-grouped bootstrap 95%
  interval below zero.
- **Dry run.** `npm run lab:dry-run` runs the whole opening on a decoy vault of MONITOR rows in a
  temporary folder, with the guard loaded. It checks the hashes, scores the models, writes the
  result and then the marker, shows that a second run only prints the stored result, and confirms
  that `lab/vault/future.json` throws. The output is in `results/vault_dry_run.txt`.
  - On the decoy, the primary difference is +0.07 MPa with a 95% interval of [−0.49, 0.69] against
    the random-search champion. That is not a success.
  - That is a MONITOR number, not a FUTURE one.
- **The real opening** (`node lab/open_vault.mjs`) has not been run. It now writes the marker only
  after the result is saved.

## Departures from the research note

| Note | What was done |
| --- | --- |
| Loss MAE, activation toggle | MAE loss added. The activation toggle is not implemented: Mlp is copied unchanged and is ReLU only. |
| "Toggle one input feature" | Switch between the three input sets (base, rich, physics). |
| Champion must retrain well (5 seeds) | Not used to choose the champion. Pre-registered instead as champion_retrained over 10 seeds, which is the primary model. |
| Ladder compares RMSE | It compares RMSE + 1 × (V_age + V_wb), with the SE of the RMSE difference. |
| Fit A and n on UCI 28-day rows | Only one scale per generator, fitted on TRAIN (1 to 7 days), as instructed for this lab. |
