"""Tests of gpu/pipeline/orchestrator.py with fake jobs (no PyTorch needed).  python -m gpu.pipeline.tests.test_orchestrator"""
import json
import tempfile
import threading
import time
from pathlib import Path

from gpu.pipeline.orchestrator import Orchestrator

D = "import pathlib, sys; d = pathlib.Path(r'{JOB_DIR}'); d.mkdir(parents=True, exist_ok=True); "
WRITE = "(d / 'out.txt').write_text('x')"


def job(jid, code, queue='cpu', **kw):
    return {'id': jid, 'queue': queue, 'argv': ['-c', D + code], 'expect': kw.pop('expect', ['out.txt']), 'timeout_s': kw.pop('timeout_s', 30), **kw}


def run(jobs, tmp, **kw):
    kw.setdefault('poll', 0.1)
    kw.setdefault('status_every', 0.5)
    o = Orchestrator(tmp, jobs, **kw)
    return o, o.run()


def status(o, jid):
    return o.state['jobs'][jid]['status']


def test_success_and_files():
    with tempfile.TemporaryDirectory() as t:
        o, rc = run([job('a', WRITE)], t)
        assert rc == 0 and status(o, 'a') == 'done'
        for name in ('state.json', 'STATUS.txt', 'HEARTBEAT', 'pipeline.log'):
            assert (Path(t) / name).exists(), name
        assert json.loads((Path(t) / 'state.json').read_text())['jobs']['a']['status'] == 'done'


def test_flaky_job_is_retried_and_failing_job_does_not_stop_the_rest():
    flaky = "m = d / 'seen'\nif not m.exists():\n    m.write_text('1'); sys.exit(3)\n" + WRITE
    with tempfile.TemporaryDirectory() as t:
        o, rc = run([job('flaky', flaky), job('bad', 'sys.exit(5)'), job('after', WRITE)], t)
        assert status(o, 'flaky') == 'done' and len(o.state['jobs']['flaky']['attempts']) == 2
        assert status(o, 'bad') == 'failed' and len(o.state['jobs']['bad']['attempts']) == 2
        assert status(o, 'after') == 'done'
        assert rc == 1  # something failed


def test_missing_output_counts_as_failure():
    with tempfile.TemporaryDirectory() as t:
        o, rc = run([job('quiet', 'pass')], t)
        assert status(o, 'quiet') == 'failed'


def test_timeout_is_not_retried():
    with tempfile.TemporaryDirectory() as t:
        o, rc = run([job('slow', 'import time; time.sleep(60)', timeout_s=1.5)], t)
        assert status(o, 'slow') == 'timeout' and len(o.state['jobs']['slow']['attempts']) == 1


def test_resume_skips_finished_jobs_and_keeps_failed_ones_unless_asked():
    with tempfile.TemporaryDirectory() as t:
        jobs = [job('ok', WRITE), job('bad', 'sys.exit(2)')]
        run(jobs, t)
        o2, rc = run(jobs, t)
        assert len(o2.state['jobs']['ok']['attempts']) == 1  # not run again
        assert len(o2.state['jobs']['bad']['attempts']) == 2  # failed job left alone
        o3, rc = run(jobs, t, retry_failed=True)
        assert len(o3.state['jobs']['bad']['attempts']) == 4  # --retry-failed runs it again


def test_interrupted_run_is_resumed_with_its_run_folder():
    code = ("run = d / 'family' / 'stamp_sweep'; run.mkdir(parents=True, exist_ok=True)\n"
            "if '--resume' in sys.argv:\n    assert sys.argv[sys.argv.index('--resume') + 1] == str(run), sys.argv\n"
            "    (run / 'result.json').write_text('{}')\nelse:\n    (run / 'run_config.json').write_text('{}'); sys.exit(1)")
    with tempfile.TemporaryDirectory() as t:
        o, rc = run([job('r', code, resumable=True, resume_glob='*/*/run_config.json', expect=['*/*/result.json'])], t)
        att = o.state['jobs']['r']['attempts']
        assert status(o, 'r') == 'done' and att[1]['resumed'] is True and att[0]['resumed'] is False, att


def test_failed_resume_falls_back_to_a_fresh_run():
    code = ("run = d / 'f' / 'stamp'; run.mkdir(parents=True, exist_ok=True)\n"
            "if '--resume' in sys.argv:\n    sys.exit(4)\n"  # resume always refused
            "n = len(list((d / 'f').iterdir()))\n"
            "(run / 'run_config.json').write_text('{}')\n"
            "if (d / 'once').exists():\n    (run / 'result.json').write_text('{}')\nelse:\n    (d / 'once').write_text('1'); sys.exit(1)")
    with tempfile.TemporaryDirectory() as t:
        o, rc = run([job('r', code, resumable=True, resume_glob='*/*/run_config.json', expect=['*/*/result.json'], )], t, max_attempts=3)
        assert status(o, 'r') == 'done', o.state['jobs']['r']


def test_stop_file_finishes_current_job_and_resume_runs_the_rest():
    with tempfile.TemporaryDirectory() as t:
        jobs = [job('first', 'import time; time.sleep(1.5); ' + WRITE), job('second', WRITE)]
        threading.Timer(0.6, lambda: (Path(t) / 'STOP').write_text('x')).start()
        o, rc = run(jobs, t)
        assert status(o, 'first') == 'done' and status(o, 'second') == 'skipped_stop'
        o2, rc = run(jobs, t)  # the old STOP file is renamed, not obeyed
        assert status(o2, 'second') == 'done' and len(o2.state['jobs']['first']['attempts']) == 1
        assert list(Path(t).glob('STOP.*.old'))


def test_stop_now_terminates_and_resume_finishes():
    with tempfile.TemporaryDirectory() as t:
        jobs = [job('long', "import time\nif not (d / 'second').exists():\n    (d / 'second').write_text('1'); time.sleep(60)\n" + WRITE)]
        threading.Timer(1.0, lambda: (Path(t) / 'STOP_NOW').write_text('x')).start()
        o, rc = run(jobs, t)
        assert status(o, 'long') == 'interrupted' or status(o, 'long') == 'pending', status(o, 'long')
        o2, rc = run(jobs, t)
        assert status(o2, 'long') == 'done'


def test_time_budget_skips_later_jobs():
    with tempfile.TemporaryDirectory() as t:
        o, rc = run([job('a', 'import time; time.sleep(1.2); ' + WRITE), job('b', WRITE)], t, budget_hours=0.0003)
        assert status(o, 'a') == 'done' and status(o, 'b') == 'skipped_budget'
        o2, rc = run([job('a', WRITE), job('b', WRITE)], t)  # a new session with no budget continues
        assert status(o2, 'b') == 'done'


class HotThenCool:
    """Fake telemetry: hot between 0.8 s and 2.0 s after creation, cool otherwise."""

    def __init__(self):
        self.t0 = time.time()

    def temps(self, n=3):
        dt = time.time() - self.t0
        return [95.0] * n if 0.8 <= dt < 2.0 else [60.0] * n


def test_thermal_stop_cools_down_and_retries_without_counting_a_failure():
    code = "m = d / 'ran'\nif not m.exists():\n    m.write_text('1'); import time; time.sleep(30)\n" + WRITE
    with tempfile.TemporaryDirectory() as t:
        o, rc = run([job('hot', code, queue='gpu')], t, telemetry=HotThenCool(), hot_c=92, warm_c=82, cool_c=75)
        att = o.state['jobs']['hot']['attempts']
        assert status(o, 'hot') == 'done' and att[0]['note'] == 'hot', att
        assert rc == 0


def test_cpu_jobs_run_beside_the_gpu_queue():
    sleepy = 'import time; time.sleep(1.5); ' + WRITE
    with tempfile.TemporaryDirectory() as t:
        t0 = time.time()
        o, rc = run([job('g', sleepy, queue='gpu'), job('c1', sleepy), job('c2', sleepy)], t, cpu_workers=2)
        assert rc == 0 and time.time() - t0 < 3.6, time.time() - t0  # three 1.5 s jobs in parallel, not 4.5 s in a row
        t1 = time.time()
        with tempfile.TemporaryDirectory() as t2:
            run([job('g', sleepy, queue='gpu'), job('c1', sleepy)], t2, cpu_workers=0)
            assert time.time() - t1 >= 2.9  # serial when cpu_workers = 0


def test_two_gpu_slots_run_two_gpu_jobs_at_once():
    sleepy = 'import time; time.sleep(1.5); ' + WRITE
    with tempfile.TemporaryDirectory() as t:
        t0 = time.time()
        o, rc = run([job('g1', sleepy, queue='gpu'), job('g2', sleepy, queue='gpu'), job('g3', sleepy, queue='gpu')], t, gpu_slots=2)
        dt = time.time() - t0
        assert rc == 0 and 2.9 <= dt < 4.4, dt   # three 1.5 s jobs on two slots: two rounds (~3 s), not one (1.5 s) and not three (4.5 s)


def test_truncated_output_is_not_accepted():
    code = "(d / 'out.json').write_text('{\"half\": ')"
    with tempfile.TemporaryDirectory() as t:
        o, rc = run([job('trunc', code, expect=['out.json'])], t)
        assert status(o, 'trunc') == 'failed'


def test_job_that_finished_just_before_the_stop_is_done_not_rerun():
    code = WRITE + "; import time; time.sleep(30)"  # writes everything, then lingers
    with tempfile.TemporaryDirectory() as t:
        threading.Timer(1.0, lambda: (Path(t) / 'STOP_NOW').write_text('x')).start()
        o, rc = run([job('late', code)], t)
        assert status(o, 'late') == 'done' and len(o.state['jobs']['late']['attempts']) == 1


if __name__ == '__main__':
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            t0 = time.time()
            fn()
            n += 1
            print(f'ok {name} ({time.time() - t0:.1f} s)')
    print(f'{n} orchestrator tests passed')
