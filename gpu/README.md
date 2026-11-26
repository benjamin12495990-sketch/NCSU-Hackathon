# gpu/: local GPU training kit (RTX 50-series laptop GPU)

> First measured on an NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, driver 592.19), not the RTX 5080 named in the
> earlier project snapshot. Results and hardware are in `reference_gpu/`. Everything here works on any
> CUDA GPU that your PyTorch build supports.

This folder is a PyTorch training path for the project. It does not replace anything: the JavaScript
trainers, splits and results stay the reference. It was built and tested **on CPU only** (no GPU on the
build machine). CUDA is a device switch (`--device cuda`). The first CUDA run on the laptop is also the
first time this code meets a GPU, so run the checks below in order.

| File | What it does |
| --- | --- |
| `check_env.py` | Python, PyTorch, CUDA and cuDNN versions, the GPU name, compute capability and memory, the driver from `nvidia-smi`, whether the build has the GPU's `sm_` code, and a timed matmul. Writes one JSON and refuses to overwrite it. |
| `profile_batched_mlp.py` | Throughput of banks of K independent MLPs trained together. Uses synthetic data shaped like the chemistry family (8,000 x 17, 17-96-48-1). Reports model-rows/second, wall time and peak memory, and refuses to overwrite `--output`. It measures speed only and is not a model result. |
| `bank.py` | `MlpBank`: K networks trained in one set of batched matmuls (`torch.baddbmm`). Each member has its own weights, Adam state, lr, decay, loss, seed and row order. |
| `data.py` | Family loaders that read only repo files, through a guard (below). Each returns X, y, groups, a fixed split and/or grouped CV folds, the SHA-256 of the arrays and the SHA-256 of each source file. |
| `train_tabular.py` | Mirror of `src/lib/nnFamily.js` (hidden, epochs, batch, lr with linear decay to 10%, L2 decay, squared or Huber loss, inputClip, clampOutput, logTarget, standardisation on training rows only). Trains seed replicas and/or a grid of configurations at the same time. |
| `structure/` | The first structure-aware workload: JARVIS-DFT download (`fetch_jarvis.py`), periodic neighbour graphs (`graph.py`), a CGCNN in plain PyTorch (`model.py`), and the bulk-modulus trainer (`train_structure.py`). |
| `grids/` | Example configuration grids for `train_tabular.py --grid`. |
| `laptop_suite.py` | Runs the whole first-day sequence below and gathers the JSONs to share, with a comparison against `reference_cpu/`. |
| `run_pipeline.py`, `run_overnight.ps1` | **The unattended pipeline (section 5).** One command trains every model family, runs strong CPU baselines beside it, survives crashes and reboots, and writes `REPORT.md` + `share.zip`. Start here if you do not want to run the steps by hand. |
| `structure/features.py`, `structure/element_table.json` | Hand-built features for the strongest tree baseline: element-property statistics plus neighbour-geometry statistics (checked against rocksalt NaCl in `tests/test_features.py`). The element table was generated once from the `mendeleev` library (`make_element_table.py`) and is committed. |
| `pipeline/evolve.py` | Generations of crystal networks (sections 6, 7). |
| `pipeline/nightly.py`, `recipes.json`, `schedule_nightly.ps1` | The nightly run (section 8): retrains what changed, keeps a registry of current models. `recipes.json` is the committed list of what to train and which model is recommended. |
| `structure/element_split.json` | The fixed leave-out-elements split (made by `make_element_split.py`). |
| `pipeline/` | Its parts: `orchestrator.py` (job runner, resume, thermal guard), `plan.py` (what runs), `analyze.py` (cross-fitted comparisons), `report.py`. `baselines_tabular.py` (next to this file) is the tuned tree/linear baseline for each table-based family. |
| `tests/smoke.py` | CPU end-to-end test of every piece on tiny settings: `python -m gpu.tests.smoke`. `tests/test_guard.py` tests the sealed-data guard on its own. |
| `reference_cpu/` | Compact CPU results from the build machine, to compare the laptop against. |

Local outputs go to `gpu/runs/` and downloads to `gpu/data_cache/`. Both are gitignored. Each run writes
to a new folder and never overwrites an earlier one.

## Rules this kit keeps

- **Sealed data.** `data.py` refuses `lab/vault/`. It also refuses the full concrete files that hold the
  opened far-future rows: `public/model/nn_data.json`, `known.json`, `test_points.json`, `parity.json`
  and `train/data/concrete.csv`. Symlinks and `..` paths are refused too. Concrete is read only from
  `lab/data/train.json`, `reward.json` and `monitor.json`. The smoke test checks all of this. Because those
  rows already informed project decisions, every concrete score is exploratory (`lab/README.md`).
- **Grouped evaluation.** Each family is scored by grouped K-fold CV on its training rows. Alloys and
  insulators use the agents' own fold files; the other families use the same group ids. The electrolyte
  loader reads only the OBELiX training partition (`train_idx.csv`). Its groups are connected components of
  rows that share a Composition string or a DOI token. This is the rule of `domains/electrolytes/work/baseline.py`,
  and it gives the same 124 groups and 5 folds. The fixed test splits (`--mode fixed`) were already scored once by
  the family agents, so new numbers on them are exploratory.
- **No exact parity with JavaScript.** The PyTorch trainer uses different random streams and float32 by
  default (`--dtype float64` is available), and it sums in a different order. Compare metrics, not
  predictions. Within PyTorch, an interrupted run that resumes gives the same predictions bit for bit as an
  uninterrupted run on the same device (the smoke test checks this). CPU and CUDA will differ slightly.
- **Computed labels stay labelled.** Insulator and JARVIS targets are DFT-calculated, not measured.

## 1. Set up the laptop

PyTorch is not in `requirements.txt`. Install it from the official selector,
<https://pytorch.org/get-started/locally/>: choose Stable, your OS, Pip, Python, and the newest CUDA
build, then run the command it prints.

> **From the project snapshot (verify on the selector):** RTX 50-series GPUs (5070, 5080 and so on) are
> Blackwell, compute capability 12.0. It needs a PyTorch build that includes `sm_120`. Use CUDA 13.0+ wheels and, on Windows, NVIDIA
> driver 580.88 or later. The snapshot says PyTorch 2.15 ships CUDA 13.2 wheels by default and 13.4
> alongside them. `check_env.py` prints `sm in build: True/False` for the actual GPU.

Use Python 3.10 to 3.13. The kit was tested here on Python 3.13 with `torch 2.15.0+cu132` from PyPI,
running on CPU. The CPU-only index (`--index-url https://download.pytorch.org/whl/cpu`) was blocked by this
machine's network proxy. On Windows, peak CPU memory is not reported (`null`), because Python has no
`resource` module there. CUDA memory is always reported.

### Windows (PowerShell), from the repository root

```powershell
nvidia-smi                                   # driver version; the GPU should be listed
py -3.12 -m venv .venv-gpu
# if activation is blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
# paste the selector's command here; it looks like (cu130 is an example; use what the selector shows):
#   pip3 install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r gpu\requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA device')"
```

### Linux, from the repository root

```bash
nvidia-smi
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
# paste the selector's command; on Linux the default PyPI wheel (pip install torch) is already a CUDA build
pip install -r gpu/requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Write down the exact install command, the driver version and the torch version. `check_env` records them.

## 2. Check, profile, then train

**One command for the first day:** `python -m gpu.laptop_suite --device cuda --label laptop` runs every step
below in order: smoke test, environment checks, profiles, all six tabular families, a 32-seed bank, the
JARVIS download and the 3,000-record structure run, on CUDA and, for comparison, on CPU. Each step's log
goes to `gpu/runs/<UTC time>_suite_laptop/`. The compact JSONs and a `summary.json` that compares them with
`reference_cpu/` go to its `share/` subfolder. Add `--full-structure` for the full JARVIS run. The steps one
by one:

Commands use `/`. PowerShell accepts that in these arguments. `--threads N` caps PyTorch's CPU threads. Every `--output` and run folder must be new;
use a new name for each run. Keep the laptop plugged in. During CUDA runs, watch temperature and power
in a second terminal with `nvidia-smi -l 2`.

```bash
python -m gpu.tests.smoke                                        # CPU, about 15 s; must end "10/10 passed"
python -m gpu.check_env --device cuda --output gpu/runs/env/laptop_cuda.json
python -m gpu.check_env --device cpu  --output gpu/runs/env/laptop_cpu.json

python -m gpu.profile_batched_mlp --device cpu  --counts 1,5,32,128 --epochs 10 --output gpu/runs/profile/laptop_cpu.json
python -m gpu.profile_batched_mlp --device cuda --counts 1,5,32,128 --epochs 10 --output gpu/runs/profile/laptop_cuda.json
python -m gpu.profile_batched_mlp --device cuda --counts 1,5,32,128,512 --epochs 10 --output gpu/runs/profile/laptop_cuda_warm.json
```

Repeat each profile once the laptop is warm. Compare **model-rows/second** at each K, not the wall time of
one network: one small tabular network underuses a GPU, and a bank of 32 to 512 networks shows whether
training candidates together is worth it. Here on 2 CPU threads, a bank of 128 gave 9.1x the throughput
of one network (`reference_cpu/profile_cpu.json`).

### Tabular families

```bash
# the family agent's settings, 5 seeds, grouped CV; prints the JavaScript reference numbers beside the new ones
python -m gpu.train_tabular --family alloys       --mode cv --device cuda --compare-reference
python -m gpu.train_tabular --family electrolytes --mode cv --device cuda --compare-reference
python -m gpu.train_tabular --family chemistry    --mode cv --device cuda --compare-reference   # runs prepare.py once into gpu/data_cache/
python -m gpu.train_tabular --family insulators   --mode cv --device cuda --compare-reference
python -m gpu.train_tabular --family polymers     --mode cv --device cuda --compare-reference
python -m gpu.train_tabular --family concrete     --mode fixed --device cuda --compare-reference # TRAIN -> REWARD, MONITOR

# the same alloy run on CPU, to compare like with like
python -m gpu.train_tabular --family alloys --mode cv --device cpu --tag laptop_cpu

# a bank: 32 seed replicas, or a grid of configurations x seeds, trained together
python -m gpu.train_tabular --family alloys --members 32 --device cuda --tag seeds32
python -m gpu.train_tabular --family alloys --members 5 --device cuda --tag grid --grid gpu/grids/alloys_example.json
```

A grid is a JSON list of setting overrides; each entry is trained with every seed. Put it in a `.json` file,
as in `gpu/grids/alloys_example.json`. On Linux an inline `--grid '[{"decay":0.001},{"hidden":[64,32]}]'`
also works. Windows PowerShell 5.1 strips the inner double quotes from arguments passed to native programs,
so use a file there. Override any setting with `--set key=value`, for example `--set epochs=50`,
`--set hidden=[64,32]` or `--set loss=huber`. Members with the same layer sizes, epochs, batch and transforms
share one bank. Different layer sizes make separate banks, which still run in the same process.

**Checkpoints and resume.** Checkpoints are written every 25 epochs (`--checkpoint-every`) to
`<run>/ckpt/`. Each holds the weights, the Adam state, the epoch, every RNG state, the configuration and the
data hash. To continue a stopped run, repeat the exact command and add `--resume gpu/runs/<family>/<run>`.
The run is refused if the configuration or the data hash differs, and a finished run is never reopened.

### Structure model (JARVIS-DFT bulk modulus)

```bash
python -m gpu.structure.fetch_jarvis                  # prints the URL and Content-Length (46.2 MiB) and stops
python -m gpu.structure.fetch_jarvis --download       # gpu/data_cache/jarvis/ + provenance JSON (URL, size, SHA-256, date)
python -m gpu.structure.train_structure --device cpu  --limit 3000 --epochs 15 --neighbor-ties first --tag laptop_cpu_3000   # same as reference_cpu (original neighbour rule, section 9)
python -m gpu.structure.train_structure --device cuda --limit 3000 --epochs 15 --tag cuda_3000
python -m gpu.structure.train_structure --device cuda --epochs 100 --tag cuda_full                      # all 31,897 records
python -m gpu.structure.train_structure --device cuda --epochs 100 --batch 128 --amp --tag cuda_full_bf16 # optional
```

The expected SHA-256 of `jdft_3d-9-24-2025.json.zip` is
`f9e0a3309f0000d5de1ec9e49c93963109ea45e63f451103f8f6595d2eabf7f5` (48,447,610 bytes, downloaded
2026-11-21). If figshare has replaced the file, the hash will differ. Record the new hash, and do not
compare its numbers directly with `reference_cpu/`.

- Target: `log10(bulk_modulus_kv / GPa)`, the computed (OptB88vdW) Voigt bulk modulus. Of 93,902
  records, 61,469 have no value (`na`) and 536 are not positive. Both are dropped, leaving 31,897.
- Split: each **reduced formula** goes to train, val or test (80/10/10) by SHA-256, so no formula is on both
  sides. The split does not depend on file order or `--limit`. `split.json` in the run folder holds the list
  and its hash. This split is the kit's own, not the JARVIS leaderboard split.
- Baselines on the same split: the training mean, and ridge regression on element fractions (alpha chosen
  on val). The val set picks the best epoch. The test set is scored once at the end.
- Graphs (8 A cutoff, 12 neighbours; from 2026-11-26 every neighbour tied with the 12th is kept too, see section 9) are built once per subset and cached in `gpu/data_cache/jarvis/graphs/`.
  All 31,897 took 20.6 s here. Loading the 254 MB JSON takes about 6 s and 1.2 GB of RAM.
- Checkpoints are written after every epoch (`<run>/ckpt/last.pt`). Resume with `--resume <run>` on the
  same command. On CPU a resumed run matches an uninterrupted one exactly. On CUDA, `index_add_` uses atomic
  adds, so the two runs agree only to within rounding.

## 3. Record and share results

For every run, keep:

1. **The JSON outputs.** `check_env` and profiler JSONs, and `result.json` plus `run_config.json` from each
   `gpu/runs/<family>/<run>/` (and `split.json` for structure runs). They include the argv, the device,
   the dtype, the data SHA-256, the source-file hashes, timings and peak memory.
2. **The commit:** `git rev-parse HEAD` (also recorded in every JSON, with a dirty flag).
3. **The hardware:** laptop model, OS build, CPU, RAM, GPU, driver (`nvidia-smi`), power mode (plugged in,
   performance profile), and the exact PyTorch install command.

Copy the compact JSONs, not checkpoints or `predictions.npz`, into a new folder such as
`gpu/reference_gpu/`. Commit it on a branch together with a short note of the hardware and commands.
Do not commit `gpu/runs/` or `gpu/data_cache/`. Compare the laptop's numbers with `gpu/reference_cpu/`:

- Accuracy on the same command should agree within seed noise. CPU and CUDA are not bit-identical. The structure numbers in
  `reference_cpu/` were made with the original neighbour rule, so compare them with runs made with `--neighbor-ties first`
  (`gpu.laptop_suite` does this). Without the flag the run uses the fixed rule and different graphs (on CPU: test MAE 0.144 instead of 0.135).
- Speed: compare `train_seconds` and model-rows/s (tabular) or structures/s (structure).

## CPU reference numbers (build machine)

These are the files in `gpu/reference_cpu/`. They ran on 2 threads of a shared 4-CPU Linux container,
with Python 3.13 and torch 2.15.0+cu132 running on CPU, in float32.

| Run (command after `python -m gpu.`) | Result | Time |
| --- | --- | --- |
| `check_env --device cpu` | float32 matmul 2048: 0.40 TFLOP/s; float64 1024: 0.24 TFLOP/s | |
| `profile_batched_mlp --device cpu --counts 1,5,32,128 --epochs 3 --threads 2` (synthetic) | K=1: 72,910; K=5: 187,965; K=32: 408,336; K=128: 660,008 model-rows/s (9.1x K=1) | 0.33 / 0.64 / 1.88 / 4.65 s |
| `train_tabular --family alloys --mode cv` (agent's 3 grouped folds, 5 seeds) | ensemble MAE 77.71 HV, RMSE 107.6, R2 0.742; members 79.2 +- 2.9; train mean 173.8. JavaScript reference: ensemble 75.34 / 105.1 / 0.753 | 5.0 s |
| same with `--members 32` (one bank of 32 seeds) | ensemble MAE 76.78 HV, R2 0.747 | 10.2 s (6.4x the networks in 2.0x the time) |
| `train_tabular --family electrolytes --mode cv` (5 grouped folds, 5 seeds) | ensemble MAE 1.934 log10(S/cm), RMSE 2.815, R2 0.316; train mean 2.554. That is the same value, fold for fold, as the mean baseline in `domains/electrolytes/work/reproductions/obelix_group_cv.json` (on main), so the groups and folds match. Its ExtraTrees reaches 1.520, so this MLP is weaker than the trees | 19.4 s |
| `train_tabular --family insulators --mode cv` (agent's 3 folds, 5 seeds) | ensemble MAE 0.1168 log10, R2 0.604; train mean 0.207. JavaScript: single network 0.1176, ensemble 0.1175 | 7.2 s |
| `train_tabular --family polymers --mode cv` (GroupKFold 3, 5 seeds) | ensemble MAE 15.18 K, R2 0.926; train mean 65.7. JavaScript single network 15.1 (its folds recompute the Fox features, which these do not) | 11.4 s |
| `train_tabular --family chemistry --mode cv` (prepare.py folds, 5 seeds) | ensemble MAE 0.765 LogS, R2 0.778; members 0.793; train mean 1.884. JavaScript single seed 0.8025 | 13.7 s |
| `train_tabular --family concrete --mode fixed` (TRAIN -> REWARD; MONITOR) | REWARD ensemble MAE 13.33 MPa, MONITOR 12.35 MPa; train mean 14.99 / 15.80. Exploratory; the lab README's plain nn.js network scored 12.47 on SELECT, with other inputs | 1.8 s |
| `structure.train_structure --limit 3000 --epochs 15` (JARVIS bulk modulus, split 2,375 / 286 / 339; original neighbour rule, now `--neighbor-ties first`) | test MAE in log10(GPa): mean 0.320, ridge on element fractions 0.175 (R2 0.590), CGCNN 0.135 (R2 0.686; best val epoch 7) | 74.1 s training, 481 structures/s; graphs 2.0 s; peak RSS 1.9 GB (includes the 254 MB JSON) |

| `structure.train_structure --epochs 2` (all 31,897 records, split 25,432 / 3,174 / 3,291; original neighbour rule) | test MAE in log10(GPa): mean 0.324, ridge 0.171 (R2 0.520), CGCNN after 2 epochs 0.126 (R2 0.678) | 55 s per epoch, 459 structures/s; graphs 20.6 s; JSON load 6.5 s; peak RSS 1.9 GB |

A test MAE of 0.135 in log10 means a typical error of a factor of 10^0.135, about 1.36, against the DFT
value. On 3,000 records the CGCNN beats both baselines. On the full set, 2 epochs already beat both. The
full 100-epoch run (about 1.5 hours on this CPU) is the laptop's job.

## Not done here

- There was no GPU, so no CUDA run, no `sm_120` check and no GPU memory figures. The laptop run is the first.
- The PyTorch trainer has not been checked for parity against the JavaScript trainer beyond comparing
  metrics. Alloy CV MAE is 77.7 here and 75.3 in JavaScript.
- The structure model is a plain CGCNN with fixed hyperparameters (64 features, 3 convolutions, AdamW
  2e-3, 60 epochs by default). It has not been tuned, and it has no uncertainty estimate.


## 4. Is the crystal network better than trees?

The ridge baseline in `train_structure.py` is weak, so beating it proves little. `baseline_trees.py` trains tree models on
the same records, target and split (element fractions; and fractions plus volume per atom and atoms per cell), choosing
settings on validation rows only. `compare_structure.py` then does a paired bootstrap over whole formulas.

```
python -m gpu.structure.baseline_trees --threads 4 --output gpu/runs/structure/trees_laptop.json
python -m gpu.structure.predict_run --run gpu/runs/structure/<a finished run without predictions.npz> --device cuda
python -m gpu.structure.train_structure --device cuda --epochs 100 --seed 12 --tag seed12      # new runs save predictions.npz
python -m gpu.structure.train_structure --device cuda --epochs 100 --seed 13 --tag seed13
python -m gpu.structure.compare_structure --runs gpu/runs/structure/<run1> gpu/runs/structure/<run2> gpu/runs/structure/<run3> --trees gpu/runs/structure/trees_laptop_predictions.npz --output gpu/runs/structure/compare_laptop.json
```
(Each command is one line. On Windows use backslashes in paths.) A negative difference favours the network; the interval is
the 95% bootstrap range. "Not distinguishable" means the intervals include zero. Do not claim the network is better than
trees unless the interval lies below zero against the cell-feature tree baseline too.


## 5. Unattended pipeline (start it, leave, read the report)

Everything above can be run by hand. This section runs all of it, in order, and tells you at the end what happened.

**Before you leave the laptop** (Windows):
1. Plugged in, on a hard surface with free air under it. A long GPU run heats a laptop; the pipeline watches the GPU temperature
   (stops a job at 92 C, waits for 75 C, retries; waits between jobs if the card is at 82 C or more) but cannot cool the room.
2. Lid open, or set *Settings > System > Power > Lid, power and sleep > When I close the lid > Plugged in: Do nothing*. The pipeline
   asks Windows not to sleep while it runs, but a **closed lid sleeps the laptop whatever a program asks**. Optional command for the
   same setting (not tested by the author): `powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0` then `powercfg /setactive SCHEME_CURRENT`.
3. Pause Windows Update for the night (Settings > Windows Update > Pause), or it may restart the machine. If that happens anyway, nothing is lost beyond the running job's last checkpoint: run the resume command below.
4. At least 10 GB of free disk (the preflight stops with a message below 3 GB).

```
git pull
.\gpu\run_overnight.ps1 -Rehearsal        # 3-10 minutes on the GPU: proves the whole chain works; its numbers mean nothing
.\gpu\run_overnight.ps1                   # the real run, default time budget 10 hours
python -m gpu.run_pipeline --profile standard --dry-run     # just list the jobs and guessed times
```
(Same thing without the launcher: `python -m gpu.run_pipeline --profile standard --device cuda --tag overnight`.)

**What the standard profile runs** (GPU jobs one at a time; CPU baselines beside them, 2 threads each):
- Crystal-graph networks on JARVIS-DFT for bulk modulus and shear modulus (3 seeds each), formation energy per atom and band gap (2 seeds each; capped at 40,000 crystals if the computer reports under 24 GB of RAM), each with tree baselines on the same split.
- For each table-based family (alloys, polymers, solid electrolytes, insulators/dielectric, concrete, chemistry): a 288-setting network sweep, 5 seeds each, plus tuned tree and linear baselines on the same folds.
- Order puts the cheapest, most informative GPU work first, so a run cut short by the time budget still leaves a useful report.

**While it runs**: `gpu\runs\pipeline\<folder>\STATUS.txt` is refreshed every 30 s (jobs done, current job, GPU temperature, ETA guess). To stop politely, create an empty file named `STOP` in that folder: the running job finishes, nothing new starts. `STOP_NOW` (or Ctrl+C) interrupts the running jobs, which can be resumed from their last checkpoint.

**If it is interrupted** (reboot, power cut, closed lid, Ctrl+C, a crash): `.\gpu\run_overnight.ps1 -Resume`. Finished jobs are skipped, interrupted ones continue from their checkpoint, failed jobs are not retried unless you add `--retry-failed`. A job that fails is retried once before the pipeline moves on, so one bad job cannot stop the rest.

**What you get** in `gpu\runs\pipeline\<folder>\`: `REPORT.md` (plain-language results and what they do not show), `summary.json`, `telemetry.csv` (GPU temperature/power/utilisation every 10 seconds), `logs\` (one per job), and **`share.zip`** (the report, summary, log, plan and every result file; send this one file back). Rebuild the report at any time from what finished: `python -m gpu.pipeline.report --pipeline gpu\runs\pipeline\<folder>`.

**How the comparisons are made** (so the report is not fooled by its own sweep): for each cross-validation fold, the network setting (and each baseline) is picked using only the *other* folds' errors, then scored on the fold it did not see ("cross-fitted"). The difference between network and tuned trees is a paired bootstrap over whole groups (mixes, alloy systems, papers; 2,000 resamples). Concrete has no grouped folds, so it is selected on REWARD and scored on MONITOR and labelled exploratory.

**What it cannot tell you**: every label is a historical measurement or a DFT calculation, so a win here is not a new material; the crystal split is by exact formula (not by chemical system); time estimates in the plan are the minutes measured on one RTX 5070 Laptop GPU (the first full run: 26 jobs in 2.8 h; the report records the real ones for your machine); the pipeline has been exercised end to end on CPU (rehearsal profile, with injected failures and interrupts) but not yet on the RTX GPU or on Windows.
The pipeline reads data only through `gpu/data.py`, which refuses the sealed concrete folder; it never writes to `lab/`, `src/` or `public/`, and it checks at the end that `lab/vault/` is byte-for-byte unchanged in size and modification time.


## 6. Second cycle: held-out chemical systems, stronger trees, generations

What the first overnight run could not say: the crystal split was by exact formula, the trees saw only composition plus two cell numbers, and the
headline compared an *average* of networks with one tree model. This cycle adds the missing tests and then spends the spare GPU time searching.

```
git pull
.\gpu\run_overnight.ps1 -Rehearsal            # (if you have not already) a few minutes; creates a throw-away rehearsal folder
.\gpu\run_overnight.ps1 -Cycle2 -Rehearsal    # ~10 minutes: the whole second cycle in miniature on the GPU
.\gpu\run_overnight.ps1 -Cycle2               # the real thing: ADDS jobs to your newest real run folder, default budget 12 h
```
It prints a forecast first (from how long the same jobs took in your earlier run) and then the folder it is extending. Nothing already in that folder is changed
(the old plan is saved next to it as `plan.before_extend_<time>.json`, an earlier `REPORT.md` is renamed, not overwritten). A rehearsal can never extend a real folder or the reverse:
their job names are the same, so the program refuses.

**What it adds**
1. **Held-out chemical systems.** Crystal networks (bulk, shear, formation energy, band gap; 3, 3, 2, 2 seeds) trained and scored with whole element combinations (Fe-O, Li-Fe-P-O, ...) kept out of training.
   The matching tree baselines run on the same split.
2. **Stronger trees.** A third feature set (`rich`: element-property statistics + neighbour-geometry statistics), on both splits. The opponent in every verdict is the tree feature set with the lowest *validation* error, chosen without the test rows.
3. **Generations.** Per property: generation 0 = the default settings plus 5 random ones; then 3 more generations of 4 children (copies of the best two so far with two settings nudged). Each individual is a normal run
   (40 epochs, held-out-systems split); **the reward is validation MAE on held-out systems**, never test. The winner is retrained for 100 epochs on **fresh seeds** and scored on the test systems against the default settings trained the
   same way earlier, and against the strongest trees. If the defaults are still the best, nothing is retrained and the report says so.
4. **Two GPU jobs at once** if the computer reports at least 28 GB of memory (one card was only ~20% busy with one job); otherwise one. Override with `--gpu-slots`.

Knobs: `--generations 4 --population 6 --children 4 --search-epochs 40 --final-epochs 100 --budget-hours 12 --gpu-slots 2`. The search is deterministic (same results, same children), so `-Resume` after any interruption continues the same population.
If the time budget would be exceeded the program skips the remaining generations and goes straight to the finalists, so the cycle always ends with the fair test.

**Every individual is kept.** Each is a run folder under `jobs\evo_<property>_g<generation>_c<n>\` (config, split, predictions, weights); `lineage.json` lists every individual with its parent, settings and validation error.

**How to read the result.** The search curve ("best validation MAE by generation") is optimistic by construction: it is the best of ~18 candidates on the same rows. Only the finalist line counts: evolved vs default, on fresh seeds and
unseen test systems, paired bootstrap over whole chemical systems. My expectation before running it [likely, not proven]: a few percent on validation, little or nothing on the finalist test; if the finalists beat the defaults clearly, that is a real finding.
Even this split keeps every element and most sub-systems in training, so it is not a "new element" test.


## 7. Third cycle: held-out elements, fine-tuning, a wider formation-energy search

```
git pull
.\gpu\run_overnight.ps1 -Cycle3 -Rehearsal    # a few minutes (extends the newest rehearsal folder)
.\gpu\run_overnight.ps1 -Cycle3               # the real thing: ADDS to your newest real run folder; default budget 12 h
```
0. **Every simulation gets a network.** Cycle 3 also adds the network sweep + tree/linear baselines for the table families the folder had never trained:
   `cleaning_pcmc` and `cleaning_tension` (SURFPRO surfactants: critical micelle concentration and surface tension; 1,395 and 972 rows, paper-held-out folds) and
   `corrosion` (NIST CORR-DATA point-value corrosion rates; 5,761 rows, reference-held-out folds; exploratory, not safety advice). With the six families from the first run
   and the four crystal properties, one folder and one report then cover all 13 simulations. The surfactant features (RDKit descriptors + fingerprint counts) were exported once
   to `domains/cleaning/work/cleaned/surfpro_features.npz` by `domains/cleaning/work/export_features.py`, so the training machine needs no extra package.
1. **Held-out elements** (`--split element`, `structure/element_split.json`): every crystal containing Pu, Pa, Hf, I, He, Ce or Cr is test (10% of crystals),
   every crystal containing Tb, Tm, Yb, Pr, Ag or Mo (and none of those) is validation; neither set appears in training. Two network variants:
   the default (one learned vector per element: an unseen element keeps its random start, so it is expected to fail) and `--atom-input both`
   (element properties, as in the original CGCNN paper, so an unseen element is described by what it is). Trees on all three feature sets on the same split.
2. **Fine-tuning** (`--init-from <run>`): bulk and shear modulus networks start from the formation-energy network (94k labels) trained on the same split,
   output layer reset, lr 1e-3. The trainer refuses a source that trained on, or chose its epoch with, any of the new run's validation or test crystals.
   Compared with the same network from scratch on formula, held-out-systems and held-out-elements splits.
3. **Wider formation-energy search** (round r2): starts from the cycle-2 winner, allows up to 8 convolutions, 256 atom features, 384 hidden, lr 8e-3;
   finalists trained on fresh seeds for the same 100 epochs as the cycle-2 winner they are compared with (so longer training cannot pass for better settings).
4. **Prediction intervals** for every task in every report (also old folders: rebuild the report), see the "Prediction intervals" section.

## 8. Nightly

Run these when no other pipeline is running (a nightly started beside a cycle refuses to start, without pulling new code):
```
.\gpu\run_overnight.ps1 -Nightly -Rehearsal   # a few minutes: proves it works
.\gpu\schedule_nightly.ps1                    # then: a Windows scheduled task runs -Nightly every day at 01:17 (-Time to change, -Remove to stop)
```
Each night (own folder `gpu\runs\pipeline\<time>_nightly`): `git pull --ff-only`, then for each of the 10 tasks a key from (data hash, `recipes.json` entry, code version);
a task is retrained only if its key changed since its last nightly (`gpu\runs\pipeline\registry.json`), `--force` retrains all. Table families: the standard sweep + baselines
(the report shows the recommended trees or blend). Crystal properties: the recipe's settings and seeds on the held-out-systems split + tree baselines. The report starts with a table
of the current model, test error and 90% interval for every task. The first night trains everything (about 2-3 h on the RTX 5070 Laptop); a night with nothing new trains nothing.
A pipeline refuses to start while another is running (`RUNNING.lock`), so the nightly task never collides with a long cycle. `-Resume` always continues the main run (`LAST_PIPELINE.txt`); the nightly and the rehearsals keep their own pointers (`-Resume -Nightly`, `-Resume -Rehearsal`), so a nightly can never hide an unfinished cycle.
**Recipes change only by a reviewed commit**: a cycle that finds a better model is evidence for editing `gpu/recipes.json`, never an automatic promotion,
and no decision is ever made on test rows.

## 9. The neighbour rule (fixed 2026-11-26) and how to measure what the fix changed

Each atom's neighbour list is cut at 12. Until this fix, when several neighbours sat at the same distance as the 12th (equal to within rounding), the
ones kept were whichever came first in the file, so **the same crystal written two ways (atoms in another order, cell shifted, rotated or doubled) could
get two predictions**. How many crystals this touches depends on what is counted (3,000 crystals each for bulk modulus and formation energy):
- About three in four (76% and 72%) have another neighbour within 0.001 angstrom of the 12th, so the fix gives them a slightly different graph.
- About four in ten (49% and 40%) have neighbours equally far to within rounding, the only case where the original rule depended on the order.
- The order matters for the prediction mostly when the tied neighbours are different elements: under 1 in 10 crystals (8.5% and 7.4%). In a review
  check with random rewrites, the original-rule prediction moved (by more than 0.0001) for about 3-4% of crystals.

- **The fix** (`gpu/structure/graph.py`, `--neighbor-ties all`): keep every neighbour within 0.001 angstrom of the 12th. A whole shell of equally distant
  neighbours is kept together, so the graph depends on the geometry only. Graphs get 6-9% more edges (13.1 per atom instead of 12.0 for bulk modulus).
- **New runs use the fixed rule** (networks and the structure-feature trees). Runs made before the fix are untouched; resuming one, or fine-tuning from
  one (`--init-from`), keeps the rule it was made with, and asking for the other rule is refused. `--neighbor-ties first` reproduces an old run.
  The two rules never share a graph or feature cache.
- **Continuing a folder made before the fix keeps its rule.** `-Resume`, `-TieFix` and a new cycle all start a folder's unfinished jobs again
  (never started, skipped, retried from scratch, new generations of an unfinished search). The runner first adds `--neighbor-ties first` to
  every crystal job of the folder that states no rule and records the search round's rule, so those jobs train like their finished siblings
  (a copy of the old plan.json is kept; finished runs are unchanged). Every crystal job planned from now on states its rule.
- **Nothing measured earlier is deleted or rewritten.** Reports say which rule the networks and trees of each crystal section used (a line
  under each network-versus-trees section, a column in the fine-tuning table, a line per search round, the split label in the intervals
  table) and flag comparisons that mix the two.
- `python -m gpu.tests.test_invariance` proves the property on four test crystals (worst change of a prediction: 0.00001 under the fixed rule, 0.35 under
  the original one, untrained networks). `python -m gpu.structure.invariance --run <run folder> --output inv.json` measures it for any finished run.

What the fix does to **accuracy** is not known until it is measured, because three in four crystals now give the network a slightly different graph:
```
git pull
.\gpu\run_overnight.ps1 -TieFix -Rehearsal   # a few minutes (extends the newest rehearsal folder; its second cycle must have run BEFORE the fix)
.\gpu\run_overnight.ps1 -TieFix              # the real thing: ADDS to your newest real run folder
```
It trains the held-out-systems networks again (same seeds, epochs, settings and crystals; 10 network runs) and the trees on the same split, all under
the fixed rule. Any unfinished jobs of the folder run in the same session, under the rule they were planned with. It adds two sections to the report:
the fixed-rule numbers, and "Neighbour-rule fix", which compares fixed against original seed by seed on the same test crystals and shows how much one
network of each kind changes its predictions when a crystal is only rewritten. The comparison uses only runs whose recorded rule matches their side, so a
folder whose held-out-systems runs were made after the fix (a rehearsal made after the pull, say) has nothing to compare and the run says so instead
of starting. Expect about 2-3 hours with two GPU jobs at a time (a guess from the cycle-2 times plus the larger graphs; the report records the real
times). What it re-measures for `gpu/recipes.json`: the default-settings networks, the trees and their comparison on held-out systems, which covers the
bulk-modulus, shear-modulus and band-gap entries. It does not retrain the formation-energy entry's evolved settings (seeds 21/22), so the evidence that
they beat the defaults stays an original-rule measurement until a new search or finalist run. The nightly retrains every crystal task once under the
fixed rule by itself.

The fix is not a smooth cutoff: a neighbour just outside the 0.001 angstrom tolerance is still dropped whole.
