> Reconstructed from the documentation snapshot of commit eff7c37 and checked against this repository on 2026-11-21.

# Protocol for isolated, evidence-based model training

Use this file as the operating instructions for any coding agent that trains, tunes, or evaluates a
model in this repository. The goal is to let the model make predictions on data it has not seen,
then measure those predictions against the withheld outcomes. Preserve that separation so a good
score means something.

## 1. Start by mapping the data and the task

Before changing code, identify:

- The prediction target, units, intended use, and what counts as a successful prediction.
- The independent experimental unit. Examples: a concrete mix, a patient, a machine, a site, or a
  time series. Repeated rows from one unit must stay together when splitting.
- All data sources, provenance, licenses, labels, missing values, and known duplicates.
- Which files contain labels or rows that must be held out. Record paths and hashes without
  printing or reading sealed labels.
- The current baselines, model code, evaluation scripts, repository rules, and hardware.

Read `docs/AGENTS.md`, the relevant experiment README, and any `AGENTS.md` files before editing.
Do not assume a file called "test" is genuinely untouched; inspect the history and ask whether
prior model decisions used its results.

## 2. Define the data boundary before training

Create fixed tiers before any model search:

| Tier | Purpose | Permitted use |
| --- | --- | --- |
| TRAIN | Fit weights and preprocessing | Training, feature scaling, and fitting learned parameters |
| SELECT / REWARD | Choose features, hyperparameters, stopping rules, and model variants | Repeated comparison is allowed; scores are development results |
| MONITOR | Check whether repeated selection is overfitting | Report and predeclared stopping checks only; do not optimize directly against it |
| SEALED TEST | Final estimate of performance on the target population | Open once, only after all models and the analysis are fixed |

Use group-aware splits so all rows for the same independent unit stay in one tier. When predicting
future outcomes, use a temporal split that matches deployment, and also group repeated measurements
when possible. If the same formulation appears under different rounded values, define and document
a tolerance before splitting. Freeze random seeds and row/group indices.

Avoid leakage through preprocessing: calculate means, standard deviations, imputation values,
feature selection, encoders, and calibration parameters from TRAIN only. Any pre-training dataset
must be declared. It must not contain the sealed outcomes or be derived from them. If outside data
overlaps with a held-out group, disclose that overlap and adjust the claim.

## 3. Make the data boundary enforceable

Do not rely on an agent remembering not to inspect withheld data. Implement access controls:

- Put sealed labels in a separate file or store with restricted access. Keep the seal receipt, row
  count, and SHA-256 outside the sealed location.
- Add a guard that refuses reads of sealed paths and any duplicate full-data file that contains
  sealed rows. Fail closed on aliases, relative paths, symlinks, streams, and alternate read APIs
  where practical.
- Keep ordinary training and evaluation scripts from importing the one-time opener or reading raw
  full datasets.
- Allow only a small, separately reviewed opening script to access sealed labels. Check model
  hashes and the registered code/data hashes before scoring.
- Mark the opening as started before reading labels. Never retry an interrupted or completed
  opening. Record the result and marker immediately.
- Keep archives and experiment outputs append-only or write-once. A changed run gets a new output
  path and explicit run ID; never silently replace prior results.

Repository guards are an important defense, not a security boundary against a malicious user with
full filesystem access. For high-value or sponsor-facing tests, put sealed labels in a separate
account, machine, or custodian-controlled store and give the training agent only the permitted
data.

## 4. Pre-register before the final evaluation

Before opening SEALED TEST, write a versioned registration containing:

- The task, population, units, success criterion, primary metric, and any secondary metrics.
- Exact split procedure, group definitions, seeds, row counts, and data hashes.
- Baselines, model variants, feature sets, hyperparameters, and how an ensemble will be formed.
- All selection and stopping rules, including how MONITOR may be used.
- Training code, dependency versions, source commit, model artifact hashes, and hardware/software
  details.
- The statistical comparison, resampling unit, confidence interval method, and handling of missing
  or failed predictions.
- Planned subgroup analyses and known limitations.

Do not add a model to the comparison after seeing sealed results. If a code or model change is
needed, it requires a new untouched test set or a clearly labeled exploratory analysis.

## 5. Train and compare honestly

- Include simple, relevant baselines. For age extrapolation, include a log(age) regression and a
  documented design-code or physical curve where applicable.
- Train every candidate using the same permitted TRAIN data and comparable tuning budgets. Record
  failed runs and seed-to-seed variation.
- Keep model outputs labeled as predictions. Distinguish measured values from generated or
  simulated values.
- Report overall metrics and performance by meaningful groups, time bands, and seen/unseen cases.
  Include sample counts.
- Quantify uncertainty by resampling the independent unit, not individual repeated rows. Report
  paired differences against the strongest baseline with a confidence interval.
- Treat synthetic physics data as assumptions and pre-training, not as measurements. Cite external
  constants and label uncertain assumptions.
- Prefer the simplest model that meets the predeclared goal. Evolutionary search, large networks,
  and GPU acceleration need evidence that their added complexity improves the target metric.

## 6. Interpret and preserve the result

After final scoring, report what was registered, whether the protocol was followed, all model
scores, paired differences and intervals, subgroup results, deviations, and the limitations. A
failed hypothesis is a valid result. Do not tune against the opened set and then call that set a
confirmation. Keep the opened set available for historical analysis but mark every later analysis
of it as post-hoc.

For the current concrete lab, the existing implementation is in `lab/`:

- `guard.mjs` blocks reads of `lab/vault/` and of the full data files that hold future rows.
- `make_splits.mjs` seals the age split.
- `split_select.mjs` creates the group-aware REWARD/MONITOR subsets.
- `preregister.mjs` and `open_vault.mjs` implement registration and one-time scoring.

The future vault in this repository has already been opened. The result and the markers are in
`lab/results/` (`OPENING_STARTED.json`, `vault_opening.json`, `VAULT_OPENED`). It cannot be a fresh
test for another model.

## 7. Move training onto a local device

The user owns the machine and runs the long job. A language model can help write and review code,
but should not need to generate training examples or supervise every optimizer step. Keep the
training loop offline after dependencies and data are installed; local inference and training then
incur no per-token API cost.

### First make the existing run reproducible

1. Clone the repository and create a dedicated branch for the experiment.
2. Install pinned dependencies from the project's lock files or requirements files.
3. Record the commit, OS, CPU, RAM, GPU model and VRAM, driver/toolkit versions, runtime versions,
   and exact command.
4. Run the smallest smoke configuration, verify guards and deterministic behavior, then run one full
   CPU reference job.
5. Save checkpoints, logs, metrics, and a compact JSON summary under a unique run ID. Keep
   write-once outputs; resume only when the run configuration hash matches.

### Decide whether a GPU port is worth it

Measure wall time, examples/second, peak RAM/VRAM, and energy or power if available. The current
JavaScript MLP (`lab/net.mjs`) is CPU-only; an RTX 5080 will not accelerate it automatically. Its
tabular networks are small, so CPU may already be faster than GPU once data-transfer and launch
overhead are included. Benchmark before porting.

If GPU is the bottleneck for larger workloads, add an optional PyTorch implementation that:

- Uses the same split files, features, normalization, seeds, loss, and evaluation code as the
  reference implementation.
- Checks CUDA availability and falls back cleanly to CPU; never silently changes datasets or
  metrics.
- Uses mini-batches, pinned data only when it helps, and mixed precision only after checking
  numerical parity and stability. For small tabular models, full precision may be simpler and just
  as fast.
- Saves model state, optimizer state, epoch, random-number-generator state, run configuration, and
  data hashes in resumable checkpoints.
- Compares predictions against the reference within a declared tolerance before results are
  accepted.
- Writes to a new output location and never replaces registered artifacts.

GPU work in this repository is being set up in the `gpu/` folder. See `gpu/README.md` for its
commands and `lab/GPU_LAPTOP.md` for the reasoning.

For an NVIDIA laptop GPU, verify the vendor-supported driver and the PyTorch CUDA build on the
actual machine. Do not install a CUDA toolkit by guesswork; first check the installed driver and the
current PyTorch installation instructions. Keep the laptop plugged in, use a stable cooling surface,
and monitor temperature and VRAM during a short benchmark. GPU availability is not a reason to
enlarge the model or search budget without a measured need.

### Running for hours without paid model calls

- Keep the training script, datasets, package cache, and checkpoints local. Do not call a hosted
  LLM inside training or validation loops.
- Run a bounded job with a clear seed, epoch/generation limit, checkpoint interval, and stop
  condition. Use the OS process manager or terminal multiplexer so a closed chat does not stop the
  job.
- Log progress to a local file and write periodic checkpoints atomically. On restart, verify the
  configuration and data hashes before resuming.
- At completion, produce a compact report containing run ID, commit, config, hardware, elapsed
  time, training/validation metrics, baseline deltas, uncertainty intervals, and checkpoint/model
  hashes.
- Bring the summary and selected logs back to an LLM for interpretation. Keep the LLM outside the
  sealed-data boundary: share only permitted development summaries until the registered final
  opener is run.

## 8. Agent checklist

Before training, the coding agent should be able to answer "yes" to each:

- Have I identified the independent group and prevented group leakage?
- Are all sealed labels inaccessible to training, tuning, and model-selection code?
- Are transformations and learned preprocessing fitted on TRAIN only?
- Are baselines and the metric fixed before final scoring?
- Can I reproduce the run from its command, commit, versions, seeds, and data hashes?
- Are checkpoints and results saved without overwriting prior runs?
- Does the local job run without hosted model/API calls?
- Is GPU use measured and optional, with CPU fallback and a parity check?
- Is every claim labeled as measured, predicted, simulated, or post-hoc as appropriate?

If any answer is "no," stop before opening sealed data. Fix the protocol or explicitly downgrade the
result to exploratory.
