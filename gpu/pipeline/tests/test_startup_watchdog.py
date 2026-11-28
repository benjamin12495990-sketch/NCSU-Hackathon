"""The start-up watchdog: a start-up that hangs must stop itself and say where (2026-11-27/28: the scheduled nightly sat silent for 10 hours).
python -m pytest gpu/pipeline/tests/test_startup_watchdog.py"""
import subprocess
import sys
import tempfile
from pathlib import Path

from gpu import run_pipeline as R

REPO = Path(R.__file__).resolve().parents[1]


def test_a_hanging_startup_stops_itself_and_names_the_frame():
    with tempfile.TemporaryDirectory() as d:
        code = ("from gpu import run_pipeline as R; import time\n"
                "R.arm_startup_watchdog(0.01, r'%s')\n"
                "def stuck_in_the_gpu_check():\n    time.sleep(30)\n"
                "stuck_in_the_gpu_check()\n" % d)
        r = subprocess.run([sys.executable, '-c', code], cwd=str(REPO), capture_output=True, text=True, timeout=25)
        assert r.returncode == 1                                   # stopped by the watchdog, not after the 30 s sleep
        txt = (Path(d) / 'startup_hang.txt').read_text()
        assert 'Start-up took longer than 0.01 minutes' in txt
        assert 'stuck_in_the_gpu_check' in txt                      # the frame where the program was stuck


def test_a_finished_startup_leaves_no_file_and_cancels_the_timer():
    with tempfile.TemporaryDirectory() as d:
        code = ("from gpu import run_pipeline as R; import time, sys\n"
                "R.arm_startup_watchdog(0.01, r'%s')\n"
                "R.disarm_startup_watchdog(); R.disarm_startup_watchdog()\n"      # twice: harmless
                "time.sleep(1.5); print('alive')\n" % d)
        r = subprocess.run([sys.executable, '-c', code], cwd=str(REPO), capture_output=True, text=True, timeout=25)
        assert r.returncode == 0 and 'alive' in r.stdout            # the cancelled timer did not fire
        assert not (Path(d) / 'startup_hang.txt').exists()


def test_the_launcher_reads_the_hang_file():
    ps1 = (REPO / 'gpu' / 'run_overnight.ps1').read_text()
    assert 'startup_hang.txt' in ps1
