> Reconstructed from the documentation snapshot of commit eff7c37 and checked against this repository on 2026-11-21.

# Running experiments locally

The lab experiments use small tabular networks. The current trainer is JavaScript and CPU-only; the
RTX 5080 does not accelerate `lab:evolve2`. Those runs are already light enough to run locally for
hours without model-token or API charges. The 5080 is useful if you later move the network
training loop to PyTorch, but that port should be benchmarked against the existing deterministic
JavaScript reference before becoming the default. GPU work is being set up in the `gpu/` folder;
see `../gpu/README.md`.

The evolution scripts use only Node built-ins. `npm install` is needed for the web app, not for
them. The baseline scripts (`lab:baselines`, `lab:baselines2`) also call `python3` for the tree
models.

## Check the setup first

Run from the repository root:

```sh
npm run lab:smoke   # splits (if needed), baselines, a tiny evolution, isolation check
npm run lab:check   # isolation check alone
```

`lab:smoke` runs its tiny evolution into a temporary archive outside the repository and checks
that the new lineage equals the committed `lab/archive_smoke/lineage.jsonl`. The guard
(`lab/guard.mjs`) blocks reads of `lab/vault/` and of the full data files that contain the opened
future rows.

## Long evolution runs

The evolution code supports deterministic resume and has its own archive guard. Run modestly
first:

```sh
npm run lab:evolve2 -- --archive lab/archive_local --seed 17 --children 20 --generations 25 --workers 3
```

Rerun the same command to resume. The archive's first record stores the run settings and the data
hashes; a rerun with different settings is refused. Use a new archive directory and seed for each
run. Archives are append-only; do not reuse or delete an archive to change its configuration. The
guard refuses to overwrite, delete or rename files under `lab/archive*/`. GPU acceleration is not
wired into evolution.

The worker count controls concurrent CPU training jobs for one population. The worker count is not
part of the stored settings, so it can change between resumes. `--workers 6` is accepted:

```sh
npm run lab:evolve2 -- --archive lab/archive_laptop_6w --seed 17 --children 20 --generations 25 --workers 6
```

This repository has checked only that 2 and 3 workers give the same lineage (`lab/README.md`).
Check that 6 workers give the same lineage on the laptop before relying on it.

On the laptop, time the same small run at 2, 4, then 6 workers before leaving a long run. These
are CPU processes; the RTX 5080 will not speed them up. More workers can shorten wall time if the
CPU has spare cores, but they do not improve model quality by themselves. Keep the generation and
stopping rules fixed when comparing worker counts. Do not run six independent searches against the
same REWARD/MONITOR split and treat their best score as independent evidence: that multiplies
opportunities to overfit the same small data. The current study found evolution and equal-budget
random search close on this data, so spend compute on controlled comparisons and new data before
making the population larger.

### The random-search control

`npm run lab:random` runs `lab/evolve2.mjs --random`. By default it reads its budget (`--total`)
from `lab/archive`, the committed evolution run, and writes to `lab/archive_random`. For a local
evolution run, give it its own archive and the evolution run's actual number of individuals:

```sh
npm run lab:random -- --archive lab/archive_random_local --seed 17 --total <individuals in lab/archive_local>
```

## Choosing generations and children

There is no evidence yet that a larger population or more generations improve generalization here.
The committed run (`lab/archive/`) evaluated 280 evolved individuals, and an equal-budget random
search (`lab/archive_random/`, 280 individuals) was run beside it. Evolution stopped after generation 13 under the
five-generation no-new-champion patience rule. On the MONITOR decoy, the retrained evolution
champion was worse than the retrained random-search champion by +0.60 MPa (95% interval [0.12,
1.08]). On the opened FUTURE set, the difference was +0.79 MPa (95% interval [-0.94, 2.38];
`lab/results/vault_opening.json`). That makes random search the simpler current control.

The `--children` value sets both the generation-zero starting population and the number added in
each later generation, so the uncapped total is children × (generations + 1). The loop may stop
earlier under its existing patience rules (the patience of 5 generations is fixed in the code). To
tune the loop, keep the planned total near 280 and compare, for example:

| `--children` | Max `--generations` after generation zero | Planned individuals |
| --- | --- | --- |
| 10 | 27 | 280 |
| 20 | 13 | 280 |
| 40 | 6 | 280 |

Run every setting against an equal-budget random search, on several pre-fixed group-aware
development splits. Match each random-search control to the evolution run's actual
evaluated-individual count, since evolution can stop early. Keep one monitor split out of
selection. Report champion and ensemble results, paired mix-group confidence intervals, training
time, and the count of distinct candidates. These are development comparisons; settle the winner
using new measurements or a genuinely new dataset. Do not spend hours increasing generations after
the monitor score stops improving.

Since these data and the opened far-future result have already informed project decisions, every
new score on them is exploratory. Do not use it as a fresh confirmation or compare it to the opened
future result as if that test were still sealed.

## Datasets and GPU

There are now seven property datasets at different maturity levels: concrete, alloys, dielectric
materials, polymers, AqSolDB, SURFPRO, and NIST CORR-DATA. The OBELiX solid-electrolyte data is
also in `domains/electrolytes/work/data/`. The first four have trained families. The chemistry
grouped CPU reproduction and the OBELiX baseline are being regenerated in each family's
`work/reproductions/` folder. SURFPRO needs a feature/split/model pipeline, and CORR-DATA needs
unit normalization and study-held-out splits. See `../docs/DATA_ROADMAP.md` for source limits and
priorities.

The current workspace has no RTX device. For the laptop GPU, read `GPU_LAPTOP.md` for the reasoning
and `../gpu/README.md` for the commands. The GPU throughput profile is a throughput check, not a
scientific model result.

## Bringing results back for review

Share the archive's `lineage.jsonl` (its `generation` and `stop` records are a compact summary),
the exact command, git commit hash, Node version, and hardware/OS. For performance comparisons,
record elapsed time as well. Keep the `weights/` folders local unless someone needs to audit a
specific model; the lineage is much smaller.

## Not yet in this repository

The snapshot of eff7c37 also documented two things whose code was only in the unpushed commits
d670587 and eff7c37 (branch `experiment/physics-capped-age`):

- a capped-age exploratory model, run with `npm run lab:capped-age` and writing
  `lab/results/capped_age_exploratory.json`;
- an in-page "Train / continue locally" browser control that saved network weights in the
  browser's local storage.

Neither exists here: there is no `lab:capped-age` script in `package.json` and no such control in
`src/`. Do not try to run them from this repository. If you have the workspace with those commits,
please push the branch so they can be reviewed and this page updated.
