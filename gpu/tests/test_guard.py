"""Tests of the sealed-data guard in gpu/data.py.  python -m gpu.tests.test_guard  (or pytest gpu/tests)"""
import os
import tempfile
from pathlib import Path

from gpu.common import REPO
from gpu.data import BLOCKED_FILES, SealedDataError, guard_path, load_concrete, read_bytes, read_json

REFUSED = [*BLOCKED_FILES, REPO / 'lab' / 'vault', REPO / 'lab' / 'vault' / 'future.json', REPO / 'lab' / 'vault' / 'SEALED.json',
           REPO / 'lab' / 'data' / '..' / 'vault' / 'future.json', REPO / 'public' / 'model' / '..' / 'model' / 'nn_data.json',
           REPO / 'lab' / 'vault' / 'not_yet_there.json']
ALLOWED = [REPO / 'lab' / 'data' / 'train.json', REPO / 'lab' / 'data' / 'reward.json', REPO / 'lab' / 'data' / 'monitor.json',
           REPO / 'public' / 'model' / 'materials' / 'nn_data.json', REPO / 'lab' / 'seal_receipt.json']


def _refused(p):
    try:
        guard_path(p)
    except SealedDataError:
        return True
    return False


def test_refuses_sealed_and_full_data():
    for p in REFUSED:
        assert _refused(p), f'guard let {p} through'


def test_relative_paths_are_refused_too():
    cwd = os.getcwd()
    try:
        os.chdir(REPO)
        for rel in ('lab/vault/future.json', 'public/model/nn_data.json', './train/data/concrete.csv', 'lab/./vault'):
            assert _refused(rel), rel
    finally:
        os.chdir(cwd)


def test_read_helpers_go_through_the_guard():
    for fn in (read_bytes, read_json):
        try:
            fn(REPO / 'public' / 'model' / 'nn_data.json')
        except SealedDataError:
            continue
        raise AssertionError(f'{fn.__name__} read a blocked file')


def test_symlink_to_blocked_file_is_refused():
    with tempfile.TemporaryDirectory() as d:
        link = Path(d) / 'innocent.json'
        try:
            os.symlink(REPO / 'public' / 'model' / 'nn_data.json', link)
        except OSError:
            return  # no symlink permission (Windows without Developer Mode)
        assert _refused(link)


def test_recorded_sources_use_forward_slashes():
    from gpu.data import _rel
    assert '\\' not in _rel(REPO / 'lab' / 'data' / 'train.json')
    assert _rel(REPO / 'lab' / 'data' / 'train.json') == 'lab/data/train.json'


def test_windows_slash_and_case_variants_are_refused():
    if os.name != 'nt':
        return  # only meaningful where paths are case-insensitive and accept both slashes
    root = str(REPO)
    variants = [root.upper() + '\\LAB\\VAULT\\FUTURE.JSON', root.lower() + '/Lab/Vault/future.json',
                root + '\\lab\\vault', root + '\\public\\model\\NN_DATA.JSON', root + '/PUBLIC/model/nn_data.json']
    for v in variants:
        assert _refused(v), f'guard let {v} through'
    cwd = os.getcwd()
    try:
        os.chdir(REPO)
        for rel in ('lab\\vault\\future.json', 'LAB\\Vault', 'public\\model\\nn_data.json'):
            assert _refused(rel), rel
    finally:
        os.chdir(cwd)


def test_allowed_files_pass_and_concrete_reads_only_lab_data():
    for p in ALLOWED:
        guard_path(p)
    ds = load_concrete()
    assert set(ds.sources) == {'lab/data/train.json', 'lab/data/reward.json', 'lab/data/monitor.json'}


if __name__ == '__main__':
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            n += 1
            print('ok', name)
    print(f'{n} guard tests passed')
