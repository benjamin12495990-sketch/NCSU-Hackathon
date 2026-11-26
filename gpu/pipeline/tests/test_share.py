"""share.zip and REPORT.md do not carry this computer's paths.  python -m gpu.pipeline.tests.test_share  (no data, no training)"""
import json
import tempfile
import zipfile
from pathlib import Path

from gpu.pipeline import report as rp


def test_windows_paths_in_every_spelling_become_repo_and_home_markers():
    repo, home = 'C:\\Users\\someone\\NCSU-Hackathon', 'C:\\Users\\someone'
    pats = rp._local_path_patterns([(repo, '<repo>'), (home, '~')])
    text = '\n'.join([
        'data C:\\Users\\someone\\NCSU-Hackathon\\gpu\\data_cache\\x.zip',                 # a log line
        json.dumps({'run': 'C:\\Users\\someone\\NCSU-Hackathon\\gpu\\runs\\r1'}),          # JSON escapes the backslashes
        'cwd c:/users/SOMEONE/NCSU-Hackathon/gpu',                                         # forward slashes, other case
        'venv C:\\Users\\someone\\.venv-gpu\\python.exe',                                  # under home, outside the repo
        'other C:\\Users\\someone2\\file',                                                  # a different user: left alone
    ])
    for rx, label in pats:
        text = rx.sub(lambda m: label, text)
    lines = text.split('\n')
    assert lines[0] == 'data <repo>\\gpu\\data_cache\\x.zip', lines[0]
    assert json.loads(lines[1]) == {'run': '<repo>\\gpu\\runs\\r1'}, lines[1]
    assert lines[2] == 'cwd <repo>/gpu', lines[2]
    assert lines[3] == 'venv ~\\.venv-gpu\\python.exe', lines[3]
    assert lines[4] == 'other C:\\Users\\someone2\\file', lines[4]
    assert rp._local_path_patterns([('/', '~'), ('C:\\', '~')]) == []   # a bare root would mangle every path


def test_share_zip_holds_only_this_calls_files_and_no_local_paths():
    with tempfile.TemporaryDirectory() as d:
        pipe = Path(d) / 'pipe'
        (pipe / 'logs').mkdir(parents=True)
        (pipe / 'jobs' / 'trees_x').mkdir(parents=True)
        here = str(rp.REPO.resolve())
        (pipe / 'REPORT.md').write_text(f'# report\nFolder: `{rp.shown_folder(pipe)}`\n', encoding='utf-8')
        (pipe / 'logs' / 'a.log').write_text(f'python {here}/gpu/structure/train_structure.py\r\nnext line\r\n', encoding='utf-8', newline='')
        (pipe / 'jobs' / 'trees_x' / 'trees.json').write_text(json.dumps({'data': f'{here}/gpu/data_cache/j.zip'}), encoding='utf-8')
        (pipe / 'share').mkdir()
        (pipe / 'share' / 'stale.txt').write_text(f'old copy {here}', encoding='utf-8')   # left from an earlier run
        z = Path(rp.share_bundle(pipe))
        with zipfile.ZipFile(z) as zf:
            names = sorted(zf.namelist())
            body = {n: zf.read(n).decode('utf-8') for n in names}
        assert names == ['REPORT.md', 'logs/a.log', 'results/trees_x/trees.json'], names
        assert all(here not in t for t in body.values()), body
        assert body['logs/a.log'] == 'python <repo>/gpu/structure/train_structure.py\r\nnext line\r\n'   # line endings kept
        assert json.loads(body['results/trees_x/trees.json']) == {'data': '<repo>/gpu/data_cache/j.zip'}
        assert (pipe / 'share' / 'stale.txt').exists()            # nothing is deleted, it is only left out of the zip
        assert 'Folder: `pipe`' in body['REPORT.md']               # outside the repo: the folder name only
        assert rp.shown_folder(rp.REPO / 'gpu' / 'runs' / 'pipeline' / 'x') == str(Path('gpu/runs/pipeline/x'))


if __name__ == '__main__':
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
