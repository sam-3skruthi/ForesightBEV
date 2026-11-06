"""Background jobs: one at a time, fixed commands, persisted state and logs.

Jobs are subprocesses, so a crash cannot take the API down and the API can restart
without losing track of what ran. Only commands built here can run. No request
field reaches a shell. Heavy jobs refuse to start while another evaluation or
training run is active on this machine, including ones started outside the API.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from foresightbev.errors import JobConflict

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_PATTERN = r"foresightbev\.(cli eval|forecast\.learned)"


def build_command(kind: str, p: dict) -> list[str]:
    base = [sys.executable, "-m", "foresightbev.cli"]
    if kind == "sample":
        return base + ["sample"]
    if kind == "episode":
        return base + ["run", "--scenario", p["scenario"], "--seed", str(p["seed"]), "--mode", p["mode"]]
    if kind == "prepare_cam2bev":
        return base + ["prepare", "cam2bev"]
    if kind == "prepare_nuscenes":
        extra = [x for s in p.get("scenes", []) for x in ("--scene", s)]
        return base + ["prepare", "nuscenes", "--scale", str(p.get("scale", 0.5)), *extra]
    if kind == "pipeline":
        command = base + ["pipeline", "--sequence", p["sequence"]]
        command += ["--name", p["name"]] if p.get("name") else []
        command += ["--pipeline-config", p["pipeline_config"]] if p.get("pipeline_config") else []
        return command + (["--force"] if p.get("force") else [])
    if kind == "evaluation":
        command = base + ["eval", "--name", p["name"], "--config", p.get("config", "configs/experiment_default.yaml"), "--workers", str(p.get("workers", 8))]
        return command + (["--smoke"] if p.get("smoke") else []) + (["--final"] if p.get("final") else [])
    if kind == "showcase":
        return base + ["showcase", "--scene", p["scene"]] + (["--fault", p["fault"]] if p.get("fault") else [])
    if kind == "evaluate_nuscenes":
        return base + ["eval-nuscenes", "--name", p["name"]]
    if kind == "train_forecaster":
        return [sys.executable, "-m", "foresightbev.forecast.learned", "--name", p["name"], "--epochs", str(p.get("epochs", 25))]
    raise ValueError(f"unknown job kind {kind}")


def external_activity() -> list[str]:
    try:
        out = subprocess.run(["pgrep", "-fl", EXTERNAL_PATTERN], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        logger.warning("could not inspect running processes: %s", error)
        return []
    return [line for line in out.splitlines() if line.strip()]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class JobManager:
    HEAVY = {"evaluation", "train_forecaster", "prepare_nuscenes", "evaluate_nuscenes"}

    def __init__(self, directory: Path, activity=external_activity):
        self.directory = directory
        self.activity = activity
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.active: str | None = None
        self._recover()

    def _path(self, job_id: str) -> Path:
        return self.directory / job_id / "job.json"

    def _save(self, job: dict) -> None:
        path = self._path(job["id"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=1))
        temporary.replace(path)

    def _recover(self) -> None:
        """A job marked running from a previous server is lost unless its process lives."""
        for path in self.directory.glob("*/job.json"):
            job = json.loads(path.read_text())
            if job["status"] in ("queued", "running") and not (job.get("pid") and _alive(job["pid"])):
                job.update(status="lost", finished_unix=time.time(), error="server restarted while the job was running")
                self._save(job)
                logger.warning("job %s marked lost after restart", job["id"])

    def list(self) -> list[dict]:
        jobs = [json.loads(p.read_text()) for p in self.directory.glob("*/job.json")]
        return sorted(jobs, key=lambda j: j["created_unix"], reverse=True)

    def get(self, job_id: str) -> dict | None:
        path = self._path(job_id)
        return json.loads(path.read_text()) if path.is_file() else None

    def log(self, job_id: str, lines: int) -> list[str]:
        path = self.directory / job_id / "output.log"
        return path.read_text(errors="replace").splitlines()[-lines:] if path.is_file() else []

    def submit(self, kind: str, params: dict) -> dict:
        command = build_command(kind, params)
        with self.lock:
            if self.active is not None and self.process is not None and self.process.poll() is None:
                raise JobConflict(f"job {self.active} is still running. Wait for it or cancel it.")
            if kind in self.HEAVY:
                running = self.activity()
                if running:
                    raise JobConflict(f"another evaluation or training process is active: {running[:2]}")
            job_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
            folder = self.directory / job_id
            folder.mkdir()
            job = dict(id=job_id, kind=kind, params=params, status="running", command=command, created_unix=time.time(), started_unix=time.time(), finished_unix=None, returncode=None, error=None)
            output = (folder / "output.log").open("w")
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT / "src"))
            self.process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            output.close()
            job["pid"] = self.process.pid
            self.active = job_id
            self._save(job)
        threading.Thread(target=self._watch, args=(job, self.process), daemon=True).start()
        logger.info("job %s started: %s", job_id, kind, extra=dict(job_id=job_id))
        return job

    def _watch(self, job: dict, process: subprocess.Popen) -> None:
        code = process.wait()
        current = self.get(job["id"]) or job
        if current["status"] == "running":
            current["status"] = "succeeded" if code == 0 else "failed"
        current.update(returncode=code, finished_unix=time.time())
        if code != 0 and current["status"] == "failed":
            current["error"] = (self.log(job["id"], 1) or ["no output"])[0]
            logger.error("job %s failed with code %s", job["id"], code, extra=dict(job_id=job["id"]))
        self._save(current)
        with self.lock:
            if self.active == job["id"]:
                self.active = None

    def cancel(self, job_id: str) -> dict | None:
        job = self.get(job_id)
        if job is None:
            return None
        if job["status"] != "running":
            raise JobConflict(f"job {job_id} is {job['status']}, not running")
        job["status"] = "cancelled"
        self._save(job)
        try:
            os.killpg(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            logger.warning("job %s process had already exited", job_id)
        return job
