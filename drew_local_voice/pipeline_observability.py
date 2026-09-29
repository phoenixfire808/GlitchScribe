"""Privacy-safe, content-blind dictation pipeline observability."""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable, Optional


_STAGES = {
    "capture",
    "transcribe",
    "cleanup",
    "command",
    "preview",
    "insertion",
    "complete",
}
_STATUSES = {"started", "completed", "skipped", "blocked", "failed"}


@dataclass(frozen=True)
class PipelineStageRecord:
    session_id: str
    sequence: int
    stage: str
    status: str
    duration_ms: float
    input_count: int
    output_count: int
    reason: str

    def as_event_fields(self) -> dict:
        return asdict(self)


class PipelineObserver:
    """Emits bounded metadata and intentionally has no text/content argument."""

    def __init__(
        self,
        emitter: Callable[..., None],
        *,
        clock: Callable[[], float] = time.monotonic,
        history_limit: int = 64,
    ) -> None:
        self.emitter = emitter
        self.clock = clock
        self._lock = threading.Lock()
        self._records: deque[PipelineStageRecord] = deque(
            maxlen=max(8, min(int(history_limit), 256))
        )
        self._session_id = ""
        self._session_started_at = 0.0
        self._sequence = 0

    @property
    def session_id(self) -> str:
        with self._lock:
            return self._session_id

    def begin_session(self, *, id_factory: Callable[[], str] = lambda: uuid.uuid4().hex) -> str:
        with self._lock:
            self._session_id = str(id_factory())
            self._session_started_at = self.clock()
            self._sequence = 0
            return self._session_id

    def record(
        self,
        stage: str,
        status: str,
        *,
        duration_seconds: float = 0.0,
        input_count: int = 0,
        output_count: int = 0,
        reason: str = "ok",
    ) -> PipelineStageRecord:
        normalized_stage = str(stage).strip().lower()
        normalized_status = str(status).strip().lower()
        if normalized_stage not in _STAGES:
            raise ValueError(f"Unsupported pipeline stage: {normalized_stage}")
        if normalized_status not in _STATUSES:
            raise ValueError(f"Unsupported pipeline status: {normalized_status}")
        safe_reason = re.sub(r"[^a-z0-9_]+", "_", str(reason).strip().lower()).strip("_")
        safe_reason = (safe_reason or "unspecified")[:64]
        with self._lock:
            if not self._session_id:
                raise RuntimeError("Pipeline session has not started")
            self._sequence += 1
            record = PipelineStageRecord(
                session_id=self._session_id,
                sequence=self._sequence,
                stage=normalized_stage,
                status=normalized_status,
                duration_ms=round(max(0.0, float(duration_seconds)) * 1000.0, 3),
                input_count=max(0, int(input_count)),
                output_count=max(0, int(output_count)),
                reason=safe_reason,
            )
            self._records.append(record)
        self.emitter("pipeline_stage", **record.as_event_fields())
        return record

    def snapshot(self) -> tuple[PipelineStageRecord, ...]:
        with self._lock:
            return tuple(
                record
                for record in self._records
                if record.session_id == self._session_id
            )

    def summary(self) -> dict:
        records = self.snapshot()
        with self._lock:
            elapsed_ms = (
                max(0.0, self.clock() - self._session_started_at) * 1000.0
                if self._session_started_at
                else 0.0
            )
            session_id = self._session_id
        return {
            "session_id": session_id,
            "stage_count": len(records),
            "elapsed_ms": round(elapsed_ms, 3),
            "blocked_count": sum(item.status == "blocked" for item in records),
            "failed_count": sum(item.status == "failed" for item in records),
        }
