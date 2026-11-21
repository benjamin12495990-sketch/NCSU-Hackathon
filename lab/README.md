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

## Not done yet

- `pretrain(rows)` is a seam that does nothing: physics pre-training on synthetic rows still has to
  be written. The `pretrain` flag in a genome is always false.
- No long evolution has been run, and `lab/archive/` does not exist yet.
- `open_vault.mjs` has never been executed, by design. Its scoring code is exercised by
  `rehearse_scoring.mjs`, but its own hash checks and vault reads are not.
- The guard covers `node:fs` inside Node only. `trees.py` is kept out of the vault by the static scan,
  and receives FUTURE rows only through stdin from `open_vault.mjs`.
