"""Unattended job runner: a GPU queue and a CPU queue, state on disk, resume, time budget, thermal guard.

Knows nothing about machine learning. A job is a dict:

  id          unique name, used for folders and logs
  queue       'gpu' (one at a time) or 'cpu' (up to cpu_workers at a time)
  argv        arguments after the Python executable, e.g. ['-m', 'gpu.train_tabular', ...]; the text {JOB_DIR} and
              {PIPE_DIR} are replaced by the job's own folder and the pipeline folder
  expect      list of glob patterns (relative to the job folder); the job counts as done only if each matches a file
  timeout_s   a job running longer than this is killed and marked 'timeout' (not retried)
  resumable   if True, a retry continues an interrupted run found by resume_glob instead of starting again
  resume_glob pattern (relative to the job folder) of run_config.json files; a run folder without result.json is resumable
  env         extra environment variables
  est_minutes a rough guess, shown in the plan and in STATUS.txt only

Files in the pipeline folder: plan.json, state.json (atomic), STATUS.txt and HEARTBEAT (refreshed every 30 s),
telemetry.csv, logs/<job>.log, jobs/<job>/ (everything a job writes). Nothing here ever deletes a file.

Stopping: create a file named STOP (finish the running job, start no more) or STOP_NOW (terminate the running jobs; they
can be resumed). Ctrl+C acts like STOP_NOW. Resume later with the same folder: finished jobs are skipped.
"""
import csv
import ctypes
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from gpu.common import REPO, now_iso

DONE, FAILED, TIMEOUT, PENDING, RUNNING = 'done', 'failed', 'timeout', 'pending', 'running'
FINAL = {DONE, FAILED, TIMEOUT, 'skipped_budget', 'skipped_stop'}


def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    for i in range(10):  # Windows: antivirus or an editor can hold the file for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.2 * (i + 1))
    os.replace(tmp, path)


class KeepAwake:
    """Stops Windows from sleeping while the pipeline runs (it cannot stop a closed lid from sleeping the laptop)."""

    def __enter__(self):
        self.on = False
        if os.name == 'nt':
            try:
                ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
                self.on = bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
            except Exception:
                pass
        return self

    def __exit__(self, *a):
        if self.on:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


class Telemetry(threading.Thread):
    """Samples nvidia-smi every `interval` seconds into telemetry.csv. Silent if nvidia-smi is missing."""
    QUERY = 'utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm'

    def __init__(self, path, interval=10.0):
        super().__init__(daemon=True)
        self.path, self.interval, self.rows = Path(path), interval, []
        self.stop_event = threading.Event()
        self.available = shutil.which('nvidia-smi') is not None
        self.lock = threading.Lock()

    @staticmethod
    def _num(s):
        try:
            return float(s)
        except ValueError:
            return None

    def sample(self):
        if not self.available:
            return None
        try:
            out = subprocess.run(['nvidia-smi', f'--query-gpu={self.QUERY}', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
            vals = [self._num(x.strip()) for x in out.stdout.strip().splitlines()[0].split(',')][:5]
            vals += [None] * (5 - len(vals))
        except Exception:
            return None
        row = [now_iso(), *vals]
        with self.lock:
            self.rows.append(row)
        new = not self.path.exists()
        with open(self.path, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['time', 'util_pct', 'mem_mib', 'power_w', 'temp_c', 'sm_clock_mhz'])
            w.writerow(row)
        return row

    def run(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def temps(self, n=3):
        with self.lock:
            return [r[4] for r in self.rows[-n:] if r[4] is not None]

    def stop(self):
        self.stop_event.set()


def telemetry_summary(rows):
    """rows: [time, util, mem, power, temp, clock]; ignores missing values."""
    def col(i):
        return [r[i] for r in rows if r[i] is not None]
    util, mem, pw, tp = col(1), col(2), col(3), col(4)
    return {'samples': len(rows), 'peak_temp_c': max(tp) if tp else None, 'mean_util_pct': sum(util) / len(util) if util else None,
            'peak_mem_mib': max(mem) if mem else None, 'peak_power_w': max(pw) if pw else None}


class Orchestrator:
    def __init__(self, pipe_dir, jobs, *, budget_hours=None, cpu_workers=1, telemetry=None, hot_c=92.0, warm_c=82.0, cool_c=75.0,
                 poll=2.0, max_attempts=2, retry_failed=False, log=None, status_every=30.0):
        self.pipe = Path(pipe_dir)
        self.jobs = {j['id']: j for j in jobs}
        self.order = [j['id'] for j in jobs]
        self.budget_hours, self.cpu_workers = budget_hours, cpu_workers
        self.tel, self.hot_c, self.warm_c, self.cool_c = telemetry, hot_c, warm_c, cool_c
        self.poll, self.max_attempts, self.retry_failed, self.status_every = poll, max_attempts, retry_failed, status_every
        self.lock = threading.Lock()
        self.procs = {}
        self.t0 = time.time()
        self.logf = open(self.pipe / 'pipeline.log', 'a', encoding='utf-8', buffering=1)
        self._log = log or (lambda m: None)
        for sub in ('logs', 'jobs'):
            (self.pipe / sub).mkdir(exist_ok=True)
        for name in ('STOP', 'STOP_NOW'):  # a stop request from an earlier session must not stop this one (renamed, not deleted)
            if (self.pipe / name).exists():
                (self.pipe / name).rename(self.pipe / f'{name}.{time.strftime("%Y%m%dT%H%M%S")}.old')
        self.state_path = self.pipe / 'state.json'
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding='utf-8'))
            self.say('resuming: ' + ', '.join(f'{k}={v}' for k, v in self._counts().items()))
        else:
            self.state = {'created': now_iso(), 'jobs': {}, 'events': []}
        for jid in self.order:
            st = self.state['jobs'].setdefault(jid, {'status': PENDING, 'attempts': []})
            if st['status'] in (RUNNING, 'interrupted', 'skipped_budget', 'skipped_stop'):
                st['status'] = PENDING  # an interrupted run is resumed from its checkpoint by run_job
            elif st['status'] == DONE and not self.outputs_ok(self.jobs[jid]):
                st['status'] = PENDING  # its output went missing
            elif st['status'] in (FAILED, TIMEOUT) and retry_failed:
                st['status'] = PENDING
        self.save()

    # ---- bookkeeping -------------------------------------------------------------------------
    def say(self, msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {msg}'
        self.logf.write(line + '\n')
        self._log(line)

    def save(self):
        with self.lock:
            for i in range(10):  # another thread may be adding a key while we serialise
                try:
                    text = json.dumps(self.state, indent=1, default=str)
                    break
                except RuntimeError:
                    time.sleep(0.05)
            else:
                return
            atomic_write(self.state_path, text)

    def _counts(self):
        c = {}
        for st in self.state['jobs'].values():
            c[st['status']] = c.get(st['status'], 0) + 1
        return c

    def event(self, jid, what):
        with self.lock:
            self.state['events'].append({'time': now_iso(), 'job': jid, 'what': what})
        self.say(f'[{jid}] {what}')

    def job_dir(self, jid):
        return self.pipe / 'jobs' / jid

    def _argv(self, job, extra=()):
        sub = lambda s: s.replace('{JOB_DIR}', str(self.job_dir(job['id']))).replace('{PIPE_DIR}', str(self.pipe))  # noqa: E731
        self.job_dir(job['id']).mkdir(parents=True, exist_ok=True)  # jobs may assume their folder exists
        return [sys.executable, *[sub(a) for a in job['argv']], *extra]

    @staticmethod
    def _file_ok(path):
        """A file that exists is not enough: a job killed mid-write leaves a truncated .json or .npz."""
        try:
            if path.endswith('.json'):
                json.loads(Path(path).read_text())
            elif path.endswith('.npz'):
                with zipfile.ZipFile(path) as z:
                    return z.testzip() is None
            return os.path.getsize(path) > 0
        except Exception:
            return False

    def outputs_ok(self, job):
        d = self.job_dir(job['id'])
        for pat in job.get('expect', []):
            hits = glob.glob(str(d / pat), recursive=True)
            if not hits or not any(self._file_ok(h) for h in hits):
                return False
        return True

    def _resume_dir(self, job):
        pat = job.get('resume_glob')
        if not (job.get('resumable') and pat):
            return None
        found = sorted(glob.glob(str(self.job_dir(job['id']) / pat), recursive=True))
        for cfg in reversed(found):
            run = Path(cfg).parent
            if not (run / 'result.json').exists():
                return run
        return None

    # ---- stop signals ------------------------------------------------------------------------
    def stop_requested(self):
        return (self.pipe / 'STOP').exists() or (self.pipe / 'STOP_NOW').exists()

    def stop_now(self):
        return (self.pipe / 'STOP_NOW').exists()

    def over_budget(self):
        return self.budget_hours is not None and (time.time() - self.t0) / 3600.0 > self.budget_hours

    # ---- thermal -----------------------------------------------------------------------------
    def too_hot(self):
        t = self.tel.temps(3) if self.tel else []
        return len(t) >= 3 and min(t) >= self.hot_c

    def cool_down(self, jid, limit_s=900):
        """Between GPU jobs: if the card is hot, wait until it is below cool_c (at most limit_s)."""
        t = self.tel.temps(1) if self.tel else []
        if not t or t[0] < self.warm_c:
            return
        self.event(jid, f'GPU at {t[0]:.0f} C before starting; waiting for it to cool below {self.cool_c:.0f} C')
        end = time.time() + limit_s
        while time.time() < end and not self.stop_now():
            t = self.tel.temps(1)
            if t and t[0] <= self.cool_c:
                break
            time.sleep(self.poll)

    # ---- running one job ---------------------------------------------------------------------
    def _spawn(self, job, argv, attempt):
        """Runs argv; returns (return code or None, note). note: ok, exit, timeout, stopped, hot."""
        logp = self.pipe / 'logs' / f'{job["id"]}.log'
        env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8', **job.get('env', {})}
        with open(logp, 'a', encoding='utf-8', errors='replace') as lf:
            lf.write(f'\n===== attempt {attempt} at {now_iso()}: {" ".join(argv[1:])}\n')
            lf.flush()
            proc = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, cwd=str(REPO), env=env)
            self.procs[job['id']] = proc
            t_start, note = time.time(), 'exit'
            while True:
                rc = proc.poll()
                if rc is not None:
                    note = 'ok' if rc == 0 else 'exit'
                    break
                if self.stop_now():
                    note = 'stopped'
                elif time.time() - t_start > job.get('timeout_s', 6 * 3600):
                    note = 'timeout'
                elif job['queue'] == 'gpu' and self.too_hot():
                    note = 'hot'
                if note in ('stopped', 'timeout', 'hot'):
                    proc.terminate()
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    rc = proc.returncode
                    break
                time.sleep(min(self.poll, 0.5))
            lf.write(f'===== ended: {note}, return code {rc}, {time.time() - t_start:.0f} s\n')
        self.procs.pop(job['id'], None)
        return rc, note, time.time() - t_start

    def run_job(self, jid):
        job, st = self.jobs[jid], self.state['jobs'][jid]
        if st['status'] == DONE and self.outputs_ok(job):
            return
        failures, thermal, attempt, fresh_only = 0, 0, len(st['attempts']), False
        while True:
            if self.stop_requested():
                st['status'] = 'skipped_stop'
                self.save()
                return
            if self.over_budget():
                st['status'] = 'skipped_budget'
                self.event(jid, 'not started: the time budget is used up')
                self.save()
                return
            if job['queue'] == 'gpu':
                self.cool_down(jid)
            resume = None if fresh_only else self._resume_dir(job)
            extra = ['--resume', str(resume)] if resume else []
            attempt += 1
            st['status'] = RUNNING
            self.event(jid, f'attempt {attempt}' + (f' (resuming {resume.name})' if resume else ''))
            self.save()
            rc, note, secs = self._spawn(job, self._argv(job, extra), attempt)
            rec = {'attempt': attempt, 'seconds': round(secs, 1), 'return_code': rc, 'note': note, 'resumed': bool(resume), 'ended': now_iso()}
            st['attempts'].append(rec)
            if note in ('ok', 'stopped', 'hot') and self.outputs_ok(job):  # a job stopped just after finishing is finished
                st['status'] = DONE
                st['seconds'] = sum(a['seconds'] for a in st['attempts'])
                self.event(jid, f'done in {secs / 60:.1f} min' + ('' if note == 'ok' else f' (it had already written all its results when it was {"stopped" if note == "stopped" else "cooled down"})'))
                self.save()
                return
            if note == 'stopped':
                st['status'] = 'interrupted'
                self.event(jid, 'stopped (STOP_NOW or Ctrl+C); it can be resumed')
                self.save()
                return
            if note == 'timeout':
                st['status'] = TIMEOUT
                self.event(jid, f'killed: longer than {job.get("timeout_s", 21600) / 3600:.1f} h')
                self.save()
                return
            if note == 'hot':
                thermal += 1
                self.event(jid, f'GPU at or above {self.hot_c:.0f} C: job stopped to cool down (thermal stop {thermal})')
                if thermal > 3:
                    st['status'] = FAILED
                    self.save()
                    return
                end = time.time() + 1200
                while time.time() < end and not self.stop_now():
                    t = self.tel.temps(1) if self.tel else []
                    if not t or t[0] <= self.cool_c:
                        break
                    time.sleep(self.poll)
                continue
            failures += 1
            reason = 'finished but its expected output is missing' if note == 'ok' else f'exit code {rc}'
            self.event(jid, f'failed ({reason}); see logs/{jid}.log')
            if resume:
                fresh_only = True  # a resume that fails (for example, the saved config differs) is retried from scratch
            if failures >= self.max_attempts:
                st['status'] = FAILED
                self.save()
                return

    # ---- queues ------------------------------------------------------------------------------
    def _worker(self, ids):
        for jid in ids:
            try:
                self.run_job(jid)
            except Exception as e:  # a bug in the runner must not kill the night
                self.state['jobs'][jid]['status'] = FAILED
                self.state['jobs'][jid]['error'] = repr(e)
                self.event(jid, f'runner error: {e!r}')
                self.save()

    def _queues(self):
        todo = [j for j in self.order if self.state['jobs'][j]['status'] == PENDING]
        if self.cpu_workers <= 0:
            return [todo]
        gpu = [j for j in todo if self.jobs[j]['queue'] == 'gpu']
        cpu = [j for j in todo if self.jobs[j]['queue'] != 'gpu']
        qs = [gpu]
        for w in range(self.cpu_workers):  # CPU jobs are dealt out round-robin
            qs.append(cpu[w::self.cpu_workers])
        return [q for q in qs if q]

    def write_status(self):
        c = self._counts()
        running = [j for j, s in self.state['jobs'].items() if s['status'] == RUNNING]
        left = sum(self.jobs[j].get('est_minutes', 0) for j, s in self.state['jobs'].items() if s['status'] in (PENDING, RUNNING))
        temp = self.tel.temps(1) if self.tel else []
        txt = (f'{time.strftime("%Y-%m-%d %H:%M:%S")}  elapsed {(time.time() - self.t0) / 3600:.2f} h\n'
               f'jobs: {", ".join(f"{k} {v}" for k, v in sorted(c.items()))}\nrunning: {", ".join(running) or "-"}\n'
               f'rough time left in the plan (sum of guesses, both queues): {left / 60:.1f} h\n'
               f'GPU temperature: {temp[0] if temp else "unknown"}\n'
               f'to stop after the current job: create a file named STOP in this folder; to stop now: STOP_NOW\n')
        atomic_write(self.pipe / 'STATUS.txt', txt)
        atomic_write(self.pipe / 'HEARTBEAT', now_iso())

    def run(self):
        queues = self._queues()
        self.say(f'starting {sum(len(q) for q in queues)} job(s) in {len(queues)} queue(s)')
        threads = [threading.Thread(target=self._worker, args=(q,), daemon=True) for q in queues]
        interrupted = False
        with KeepAwake():
            for t in threads:
                t.start()
            last = 0.0
            try:
                while any(t.is_alive() for t in threads):
                    if time.time() - last >= self.status_every:
                        self.write_status()
                        last = time.time()
                    time.sleep(self.poll)
            except KeyboardInterrupt:
                interrupted = True
                self.say('Ctrl+C: stopping running jobs (they can be resumed)')
                atomic_write(self.pipe / 'STOP_NOW', now_iso())
                for t in threads:
                    t.join(timeout=60)
        self.write_status()
        c = self._counts()
        self.say('finished: ' + ', '.join(f'{k}={v}' for k, v in sorted(c.items())))
        bad = c.get(FAILED, 0) + c.get(TIMEOUT, 0)
        if interrupted or any(c.get(k) for k in ('interrupted', 'skipped_stop', 'skipped_budget')):
            return 2  # not everything ran: stopped, interrupted or out of time (resume to continue)
        return 1 if bad else 0
