"""Minimal persistent job store with a single-worker executor.

Renders are heavy (Demucs + Whisper + optional MusicGen), so jobs run
serially in one background thread. State persists to disk so preview
links survive server restarts.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

JOBS_ROOT = Path("output/campaigns")


@dataclass
class Job:
    id: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"  # queued | running | done | error
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    _store: "JobStore | None" = None

    def log(self, message: str, pct: int | None = None) -> None:
        self.events.append({"t": round(time.time() - self.created_at, 1), "msg": message, "pct": pct})
        self.updated_at = time.time()
        print(f"[job {self.id[:8]}] {message}")
        if self._store:
            self._store.persist(self)

    @property
    def dir(self) -> Path:
        return JOBS_ROOT / self.id

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "events": self.events,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "params": {k: v for k, v in self.params.items() if not str(k).startswith("_")},
        }


class JobStore:
    _instance: "JobStore | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sonic-job")
        JOBS_ROOT.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    @classmethod
    def get(cls) -> "JobStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = JobStore()
            return cls._instance

    def submit(self, kind: str, params: dict[str, Any], runner: Callable[[Job], dict[str, Any]]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
        job._store = self
        self.jobs[job.id] = job
        job.dir.mkdir(parents=True, exist_ok=True)
        self.persist(job)

        def _run() -> None:
            job.status = "running"
            job.log("Job started.")
            try:
                job.result = runner(job)
                job.status = "done"
                job.log("Job complete.", pct=100)
            except Exception as exc:
                job.status = "error"
                job.error = str(exc)
                traceback.print_exc()
                job.log(f"Job failed: {exc}")
            self.persist(job)

        self.executor.submit(_run)
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def persist(self, job: Job) -> None:
        try:
            (job.dir / "job.json").write_text(json.dumps(job.as_dict(), indent=2, default=str))
        except Exception as exc:
            print(f"[jobs] persist failed for {job.id}: {exc}")

    def _load_existing(self) -> None:
        for job_file in JOBS_ROOT.glob("*/job.json"):
            try:
                data = json.loads(job_file.read_text())
                job = Job(
                    id=data["id"],
                    kind=data.get("kind", "campaign"),
                    params=data.get("params", {}),
                    status=data.get("status", "done"),
                    events=data.get("events", []),
                    result=data.get("result", {}),
                    error=data.get("error"),
                    created_at=data.get("created_at", 0),
                    updated_at=data.get("updated_at", 0),
                )
                if job.status in ("queued", "running"):  # orphaned by a restart
                    job.status = "error"
                    job.error = "Interrupted by server restart."
                job._store = self
                self.jobs[job.id] = job
            except Exception as exc:
                print(f"[jobs] could not load {job_file}: {exc}")
