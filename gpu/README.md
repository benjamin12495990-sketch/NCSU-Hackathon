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
python -m gpu.structure.train_structure --device cpu  --limit 3000 --epochs 15 --tag laptop_cpu_3000   # same as reference_cpu
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
- Graphs (8 A cutoff, 12 neighbours) are built once per subset and cached in `gpu/data_cache/jarvis/graphs/`.
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

- Accuracy on the same command should agree within seed noise. CPU and CUDA are not bit-identical.
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
| `structure.train_structure --limit 3000 --epochs 15` (JARVIS bulk modulus, split 2,375 / 286 / 339) | test MAE in log10(GPa): mean 0.320, ridge on element fractions 0.175 (R2 0.590), CGCNN 0.135 (R2 0.686; best val epoch 7) | 74.1 s training, 481 structures/s; graphs 2.0 s; peak RSS 1.9 GB (includes the 254 MB JSON) |

| `structure.train_structure --epochs 2` (all 31,897 records, split 25,432 / 3,174 / 3,291) | test MAE in log10(GPa): mean 0.324, ridge 0.171 (R2 0.520), CGCNN after 2 epochs 0.126 (R2 0.678) | 55 s per epoch, 459 structures/s; graphs 20.6 s; JSON load 6.5 s; peak RSS 1.9 GB |

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
python -m gpu.structure.compare_structure --runs gpu/runs/structure/<run1> gpu/runs/structure/<run2> gpu/runs/structure/<run3> ^
    --trees gpu/runs/structure/trees_laptop_predictions.npz --output gpu/runs/structure/compare_laptop.json
```
(`^` continues a line in PowerShell; use `\` on Linux.) A negative difference favours the network; the interval is
the 95% bootstrap range. "Not distinguishable" means the intervals include zero. Do not claim the network is better than
trees unless the interval lies below zero against the cell-feature tree baseline too.
