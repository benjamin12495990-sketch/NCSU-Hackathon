"""Environment check: Python, PyTorch, CUDA/cuDNN, device, driver, and a timed matmul.

  python -m gpu.check_env --device cuda --output gpu/runs/env/laptop_cuda.json
Refuses to overwrite --output. --device auto picks CUDA when available.
"""
import argparse
import shutil
import subprocess
import sys
import time

from gpu.common import git_info, hardware, now_iso, pick_device, refuse_existing, sync, write_json_new


def nvidia_smi():
    exe = shutil.which('nvidia-smi')
    if not exe:
        return None
    try:
        out = subprocess.run([exe, '--query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=20).stdout.strip()
        return {'path': exe, 'query': out}
    except Exception as e:  # noqa: BLE001
        return {'path': exe, 'error': str(e)}


def timed_matmul(device, n, dtype, reps):
    import torch
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    (a @ b).sum().item()  # warm-up (kernel selection, lazy init)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        c = a @ b
    sync(device)
    dt = (time.perf_counter() - t0) / reps
    return {'n': n, 'dtype': str(dtype).replace('torch.', ''), 'reps': reps, 'seconds_per_matmul': dt,
            'tflops': 2 * n ** 3 / dt / 1e12, 'checksum_finite': bool(torch.isfinite(c).all().item())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='auto', help='auto | cpu | cuda | cuda:N')
    ap.add_argument('--size', type=int, default=2048, help='matmul size n (n x n)')
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--output', required=True, help='new JSON path (refused if it exists)')
    a = ap.parse_args(argv)
    refuse_existing(a.output)
    import torch
    device = pick_device(a.device)
    rep = {'created': now_iso(), 'git': git_info(), 'device': str(device), 'hardware': hardware(device),
           'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
           'cuda_arch_list': torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
           'nvidia_smi': nvidia_smi(), 'matmul': []}
    if device.type == 'cuda':
        cc = torch.cuda.get_device_capability(device)
        rep['sm_in_build'] = f'sm_{cc[0]}{cc[1]}' in rep['cuda_arch_list']
        if not rep['sm_in_build']:
            print(f'WARNING: this PyTorch build lacks sm_{cc[0]}{cc[1]}; install a newer CUDA build (README).', file=sys.stderr)
    for dt in ([torch.float32, torch.float64] if device.type == 'cpu' else [torch.float32, torch.float16, torch.float64]):
        size = a.size if dt != torch.float64 else min(a.size, 1024)
        rep['matmul'].append(timed_matmul(device, size, dt, a.reps))
    write_json_new(a.output, rep)
    h = rep['hardware']
    print(f"python {h['python']} | torch {h['torch']} (CUDA build {h['torch_cuda_build']}) | cuDNN {rep['cudnn']} | device {device}")
    if device.type == 'cuda':
        print(f"GPU {h['gpu']} | compute capability {h['compute_capability']} | {h['gpu_memory_gb']} GB | sm in build: {rep['sm_in_build']}")
    print('nvidia-smi:', (rep['nvidia_smi'] or {}).get('query', 'not found'))
    for m in rep['matmul']:
        print(f"matmul {m['n']}x{m['n']} {m['dtype']}: {m['seconds_per_matmul'] * 1e3:.2f} ms, {m['tflops']:.3f} TFLOP/s")
    print('wrote', a.output)
    return rep


if __name__ == '__main__':
    main()
