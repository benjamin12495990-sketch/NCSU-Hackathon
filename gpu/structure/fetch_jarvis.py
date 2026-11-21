"""Download the JARVIS-DFT 3D dataset (the file jarvis-tools' `data('dft_3d')` uses) into gpu/data_cache/jarvis/.

The URL map below is copied from jarvis-tools 2026.6.12, jarvis/db/figshare.py, get_db_info(); with
--from-jarvis-tools it is read from an installed jarvis-tools instead. Figshare record (CC BY 4.0):
https://doi.org/10.6084/m9.figshare.6815699

  python -m gpu.structure.fetch_jarvis                 # prints the size (Content-Length) and stops
  python -m gpu.structure.fetch_jarvis --download      # downloads, verifies the size, records SHA-256

The download goes to gpu/data_cache/jarvis/<file>.zip (gitignored) and a provenance JSON beside it
(URL, final URL host, size, SHA-256, date). Nothing is overwritten: an existing zip is only re-hashed.
"""
import argparse
import json
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from gpu.common import CACHE, now_iso, sha256_file, write_json_new

DB = {  # name: (url, json name inside the zip, description) from jarvis-tools 2026.6.12
    'dft_3d': ('https://ndownloader.figshare.com/files/64391379', 'jdft_3d-9-24-2025.json', '3D dataset 94k (9-24-2025)'),
    'dft_3d_2022': ('https://ndownloader.figshare.com/files/38521619', 'jdft_3d-12-12-2022.json', '3D dataset 76k (12-12-2022)'),
    'dft_3d_2021': ('https://ndownloader.figshare.com/files/29204826', 'jdft_3d-8-18-2021.json', '3D dataset 55k (8-18-2021)'),
}
OUT = CACHE / 'jarvis'


def lookup(name, from_tools):
    if from_tools:
        from jarvis.db.figshare import get_db_info  # pip install jarvis-tools
        info = get_db_info()[name]
        return info[0], info[1], info[2]
    return DB[name]


def open_get(url):
    """GET (not HEAD: the figshare redirect is a presigned GET URL); headers arrive before any body is read."""
    req = urllib.request.Request(url, headers={'User-Agent': 'gpu-kit-fetch/1'})
    return urllib.request.urlopen(req, timeout=120)  # follows the redirect to figshare's storage host


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='dft_3d', choices=sorted(DB))
    ap.add_argument('--from-jarvis-tools', action='store_true', help='read the URL map from an installed jarvis-tools')
    ap.add_argument('--download', action='store_true', help='download after printing the size')
    ap.add_argument('--max-mb', type=float, default=3000, help='refuse files larger than this')
    a = ap.parse_args(argv)
    url, js_name, desc = lookup(a.dataset, a.from_jarvis_tools)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / (js_name + '.zip')
    prov = OUT / (js_name + '.zip.provenance.json')
    print(f'{a.dataset}: {desc}\n  url  {url}\n  file {dest}')
    if dest.exists():
        rec = {'dataset': a.dataset, 'url': url, 'json_name': js_name, 'path': str(dest), 'bytes': dest.stat().st_size,
               'sha256': sha256_file(dest), 'checked': now_iso()}
        print(json.dumps(rec, indent=1))
        if not prov.exists():
            write_json_new(prov, {**rec, 'note': 'file was already present; download date unknown'})
        return rec
    try:
        r = open_get(url)
    except Exception as e:  # noqa: BLE001
        sys.exit(f'GET {url} failed: {e}\n(If a proxy blocks the figshare storage host, run this on the laptop instead.)')
    size = int(r.headers.get('Content-Length') or 0)
    print(f'  Content-Length {size:,} bytes ({size / 2**20:.1f} MiB); storage host {urllib.parse.urlparse(r.geturl()).netloc}')
    if not a.download:
        r.close()
        print('  not downloaded (add --download)')
        return {'url': url, 'bytes': size}
    if size and size > a.max_mb * 2**20:
        r.close()
        sys.exit(f'refusing: {size / 2**20:.0f} MiB > --max-mb {a.max_mb}')
    free = shutil.disk_usage(OUT).free
    if size and free < 4 * size:
        r.close()
        sys.exit(f'refusing: only {free / 2**30:.1f} GiB free; need about {4 * size / 2**30:.1f} GiB (zip + unpacked JSON)')
    part = dest.with_suffix('.zip.part')
    if part.exists():
        sys.exit(f'{part} exists (an earlier download stopped); inspect or move it, then retry')
    started = now_iso()
    got = 0
    with r, open(part, 'xb') as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            got += len(block)
            if got % (50 << 20) < (1 << 20):
                print(f'  {got / 2**20:.0f} MiB', flush=True)
    if size and got != size:
        sys.exit(f'size mismatch: got {got} bytes, expected {size}; partial file left at {part}')
    part.rename(dest)
    rec = {'dataset': a.dataset, 'description': desc, 'url': url, 'json_name': js_name, 'path': str(dest),
           'bytes': got, 'content_length': size, 'sha256': sha256_file(dest), 'download_started': started,
           'download_finished': now_iso(), 'source_map': 'jarvis-tools 2026.6.12 jarvis/db/figshare.py get_db_info()',
           'licence': 'CC BY 4.0 (figshare 10.6084/m9.figshare.6815699)'}
    write_json_new(prov, rec)
    print(json.dumps(rec, indent=1))
    return rec


if __name__ == '__main__':
    main()
