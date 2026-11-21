> Reconstructed from the documentation snapshot of commit eff7c37 and checked against this repository on 2026-11-21.

# RTX 5080 local training path

This page gives the reasoning. The commands for GPU work are in `../gpu/README.md`.

The laptop can run all current training without hosted model calls. The current JavaScript trainers
(`lab/net.mjs` and the family `nn_generic.js` files) are CPU-only. This workspace has no
`nvidia-smi` and no PyTorch installation, so it cannot see or benchmark the RTX 5080.

## Blackwell, the CUDA wheel and the driver

From the project snapshot, verify on the PyTorch selector:

- NVIDIA lists RTX 5000-series GPUs as Blackwell, compute capability 12.0.
- PyTorch 2.15 currently ships CUDA 13.2 wheels by default and CUDA 13.4 wheels alongside them.
- PyTorch's 2.12 release notes say Blackwell users need CUDA 13.0+ wheels and driver 580.88 or later
  on Windows.

Use the live installer selector for the laptop's OS and current stable option, then record the
exact command and driver version.

- PyTorch installer: https://pytorch.org/get-started/locally/
- PyTorch release notes: https://pytorch.org/blog/
- NVIDIA's architecture table: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/prerequisites.html

## Why the small tabular jobs gain little from CUDA

The existing tabular family trainers, the concrete lab evolution and the OBELiX composition baseline
are small CPU workloads. Putting them on CUDA may make little or no practical difference. Many small
tabular networks can underuse a GPU one at a time, and data-transfer and launch overhead can make
the CPU faster. OBELiX has only 321 randomized CIF files, and in the snapshot its formula-only model
was too imprecise for candidate ranking.

The first benchmark is a throughput profile. The snapshot's profiler, `lab/gpu/profile_batched_mlp.py`,
is being rebuilt as `gpu/profile_batched_mlp.py`. It trains a bank of independent MLPs on synthetic
data with the size and layer shape of the chemistry dataset. It reports throughput and memory only;
it makes no predictions and saves no model. Compare model-rows/second at several candidate counts,
not one network's wall time. Keep the laptop plugged in and watch temperature and power with
`nvidia-smi` during the CUDA run.

Do not replace the JavaScript reference based on the synthetic profile alone.

## Why JARVIS, and later OC20/OC22, are the meaningful GPU workloads

The stronger GPU demonstration is a structure-aware model on a selected NIST JARVIS-DFT task, such
as elastic tensors or Seebeck coefficients. JARVIS has tens of thousands of computed crystal-property
records and includes atomistic structures. Keep each target labeled as computed, and use a benchmark
split grouped by structure or composition.

Do not download the full roughly 1.42 GB JARVIS collection just to start. Choose one property first,
then fetch the matching named dataset using the official JARVIS-Tools API or NIST Figshare record.
Record its version and checksum, run a CPU reference on a small subset, and then benchmark the same
training configuration on CUDA. The RTX 5080 value is demonstrated by measured structure-model
throughput and reproducible candidate screening, not by a synthetic MLP profile alone.

- JARVIS dataset API: https://jarvis-tools.readthedocs.io/en/master/databases.html
- JARVIS data portal: https://jarvis.nist.gov/

Later, Open Catalyst OC20/OC22 structure-level DFT tasks are a natural graph-neural-network workload
for the 5080. Their full datasets are large. Start with an official small task/split after the
JARVIS pipeline works. See `../docs/DATA_ROADMAP.md` for sizes, licences and links.

## Order of work

1. Set up a clean Python environment on the laptop, install PyTorch from the official selector, and
   check that it sees the device (`../gpu/README.md`).
2. Run the batched MLP throughput profile on CPU and on CUDA with the same workload. Each output path
   must be new. Repeat each run once after the laptop is warm.
3. If the 5080 shows a useful gain at realistic candidate counts, port one training family as an
   optional PyTorch backend:
   1. Keep the current CPU trainer, split files, seeds, feature transforms, loss, and metric as the
      reference.
   2. Batch independent candidate networks and seed replicas on the GPU; avoid starting six isolated
      processes that all compete for one GPU.
   3. Compare CPU and CUDA predictions on identical fixed splits, then compare grouped
      cross-validation scores and uncertainty. Checkpoint weights, optimizer state, epoch, RNG
      state, configuration, and data hashes; resume only when the checkpoint configuration matches.
   4. Keep the GPU path optional until it passes parity and group-split checks. It must not change
      the sealed-data protocol or overwrite earlier runs.
4. Move to a narrow JARVIS structure-aware task, CPU reference first, then CUDA.
5. Only after that, an official small OC20/OC22 task.

Long searches belong in the local offline scripts, where checkpoints and run summaries can be
audited. The browser trainer the snapshot mentions is not in this repository (see `LOCAL_RUN.md`).
