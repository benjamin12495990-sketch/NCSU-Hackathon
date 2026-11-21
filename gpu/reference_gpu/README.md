# First measurements on the laptop GPU (2026-11-21)

Transcribed by hand from the console output and the suite's `summary.json` that the user pasted into the
project chat. The original files (`gpu/runs/...`) stayed on the laptop and are not in this repository. To add
them, copy the compact JSONs here and commit them (see `../README.md`).

## Hardware and software

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, 8,151 MiB, maximum power 90 W (`nvidia-smi`) |
| Driver | 592.19 (reports CUDA 13.1) on Windows (WDDM) |
| PyTorch | 2.15.0+cu132 |
| Not the card named in the old snapshot | The project documents said RTX 5080 (16 GB class). This machine is a 5070 Laptop GPU. |

## Crystal-structure model: bulk modulus, full JARVIS-DFT 3D run

Command: `python -m gpu.structure.train_structure --device cuda --epochs 100 --tag full100`
Run folder on the laptop: `gpu\runs\structure\20261121T222614Z_full100\`.

Target: log10 of the **DFT-calculated** Voigt bulk modulus (GPa), 31,897 usable records. The split is by reduced
formula (a fixed SHA-256 rule); it is not a held-out chemical-system split. Best epoch is chosen on validation.

| log10(K / GPa) | Validation MAE | Test MAE | Test R2 |
| --- | --- | --- | --- |
| Mean baseline | 0.3201 | 0.3238 | -0.000 |
| Ridge on element fractions | 0.1631 | 0.1707 | 0.520 |
| Crystal-graph network (best epoch 83 of 100) | 0.0953 | 0.0980 | 0.704 |

- Training time 615.9 s for 100 epochs (about 6.2 s per epoch), 4,129 structures per second on CUDA.
- A test MAE of 0.098 log10 units is a typical error of about a factor of 1.25. The ridge baseline is 1.48, the mean 2.1.
- The 2-epoch CPU reference (`../reference_cpu/`) had test MAE 0.126, so 100 epochs lowered the error by about 22%.

## Suite run `20261121T220153Z_suite_rtx5080` (before the Windows path fix)

The CPU reference is the cloud machine's 2 threads, not this laptop's CPU. "Speedup" below 1 means slower than that reference.

| Task | Ensemble MAE, CPU reference | MAE, this GPU | Seconds, CPU reference | Seconds, this GPU | Speedup |
| --- | --- | --- | --- | --- | --- |
| Alloys, grouped CV (HV) | 77.71 | 77.72 | 5.0 | 46.5 | 0.11 |
| Alloys, 32 seeds in one bank (HV) | 76.78 | 76.75 | 10.2 | 24.8 | 0.41 |
| Electrolytes, grouped CV (log10 S/cm) | 1.934 | 1.932 | 19.4 | 90.5 | 0.21 |
| Insulators, grouped CV (log10) | 0.1168 | 0.1168 | 7.2 | 29.7 | 0.24 |
| Polymers, grouped CV (K) | 15.18 | 15.17 | 11.4 | 24.2 | 0.47 |
| Chemistry, grouped CV (LogS) | 0.7648 | 0.7646 | 13.7 | 57.6 | 0.24 |
| Concrete, fixed split (MPa) | 13.33 | 13.30 | 1.8 | 7.6 | 0.24 |
| Crystal model, 3,000 records, 15 epochs (log10 GPa) | 0.13507 | 0.13509 | 481 structures/s | 4,164 structures/s | 8.7x faster |

Throughput of banks of independent small networks, model-rows per second (synthetic data, 8,000 rows, 17-96-48-1):

| Networks trained together | Laptop CPU | This GPU |
| --- | --- | --- |
| 1 | 35,851 | 5,502 |
| 5 | 119,819 | 33,830 |
| 32 | 367,291 | 186,221 |
| 128 | 946,820 | 764,453 |
| 512 | not measured | 2,913,568 |

What this shows: accuracy on the GPU equals the CPU's to the third or fourth digit; the small table models do not
benefit from the GPU; the crystal-structure model and large banks of networks do. Still to measure: the laptop's own
CPU on the crystal model, and CPU against GPU at 512 networks and above.
