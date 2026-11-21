"""Shared helpers: repository paths, hashes, write-once files, run folders, device choice."""
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

GPU = Path(__file__).resolve().parent
REPO = GPU.parent
RUNS = GPU / 'runs'
CACHE = GPU / 'data_cache'


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def git_info():
    def run(*a):
        try:
            return subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ''
    return {'commit': run('rev-parse', 'HEAD') or None, 'branch': run('rev-parse', '--abbrev-ref', 'HEAD') or None,
            'dirty': bool(run('status', '--porcelain', '--untracked-files=no'))}


def refuse_existing(path):
    path = Path(path)
    if path.exists():
        sys.exit(f'refusing to overwrite {path}: choose a new output path')
    return path


def write_json_new(path, obj):
    """Writes JSON to a path that must not exist yet (O_EXCL), creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'w') as f:
        json.dump(obj, f, indent=1, default=_jsonable)
        f.write('\n')
    return path


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f'not JSON serialisable: {type(o)}')


def new_run_dir(root, tag=''):
    """A fresh folder root/<UTC time>[_tag][-n]; never reuses an existing one."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    base = stamp + (f'_{tag}' if tag else '')
    for i in range(1000):
        d = root / (base if i == 0 else f'{base}-{i}')
        try:
            d.mkdir()
            return d
        except FileExistsError:
            continue
    raise RuntimeError('could not create a new run folder')


def pick_device(name):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        sys.exit('--device cuda requested but torch.cuda.is_available() is False; see gpu/README.md (install a CUDA build of PyTorch)')
    return torch.device(name)


def hardware(device=None):
    import torch
    info = {'python': platform.python_version(), 'platform': platform.platform(), 'machine': platform.machine(),
            'processor': platform.processor() or None, 'cpu_count': os.cpu_count(), 'torch': torch.__version__,
            'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}
    if device is not None and device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        info.update({'gpu': p.name, 'compute_capability': f'{p.major}.{p.minor}', 'gpu_memory_gb': round(p.total_memory / 2**30, 2)})
    return info


def peak_memory_mb(device):
    import torch
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 2**20
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024 if sys.platform != 'darwin' else rss / 2**20  # Linux reports kB
    except ImportError:  # Windows
        return None


def sync(device):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
