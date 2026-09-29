"""Voice-to-Crew bridge: GlitchScribe transcription → AgentSandbox job → Review pane.

Orchestrates the full round-trip: capture objective, submit to AgentSandbox,
poll status, surface artifacts in the PreviewGate."""
from __future__ import annotations
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from drew_local_voice.agentsandbox_client import (
    AgentSandboxClient,
    ASBJobStatus,
    ASBJobError,
)


@dataclass
class CrewJobResult:
    """Final result of a voice-to-crew round-trip."""

    job_id: str
    objective: str
    status: str
    artifacts: list[dict] = field(default_factory=list)
    output_text: str = ""
    errors: tuple[str, ...] = field(default_factory=tuple)
    duration_seconds: float = 0.0

    @property
    def is_success(self) -> bool:
        return self.status == "completed"

    def to_preview_card(self) -> dict:
        """Build a PreviewCard-compatible dict from the result."""
        import json
        return {
            "title": f"Crew result: {self.objective[:60]}",
            "body": self.output_text or self._summarize_artifacts(),
            "artifacts": self.artifacts,
            "errors": list(self.errors),
            "job_id": self.job_id,
            "duration_s": round(self.duration_seconds, 1),
        }

    def _summarize_artifacts(self) -> str:
        if not self.artifacts:
            return "(no artifacts returned)"
        lines = []
        for a in self.artifacts[:5]:
            name = a.get("name", a.get("id", "artifact"))
            lines.append(f"• {name}")
        return "\n".join(lines)


class VoiceCrewBridge:
    """Submit voice objectives to AgentSandbox and track them to completion.

    Thread-safe: all API calls run on a background thread.
    Status callbacks fire on the background thread; dispatch to your UI thread as needed.

    Usage:
        bridge = VoiceCrewBridge()
        bridge.submit(
            objective="Summarize my project notes",
            context="See the attached notes file",
            on_status=lambda s: show_bubble(s.status),
            on_complete=lambda r: show_preview(r.to_preview_card()),
        )
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090",
        poll_interval: float = 5.0,
        max_wait: float = 600.0,
    ) -> None:
        self._client = AgentSandboxClient(
            base_url=base_url,
            default_poll_interval=poll_interval,
        )
        self._poll_interval = poll_interval
        self._max_wait = max_wait
        self._active_jobs: dict[str, ASBJobStatus] = {}
        self._lock = threading.Lock()

    @property
    def client(self) -> AgentSandboxClient:
        return self._client

    def is_available(self) -> bool:
        """Return True if AgentSandbox is reachable."""
        health = self._client.health_check()
        return health.get("reachable", False)

    def submit(
        self,
        objective: str,
        *,
        agents: list[dict] | None = None,
        crew_id: str | None = None,
        context: str = "",
        on_status: Callable[[ASBJobStatus], None] | None = None,
        on_complete: Callable[[CrewJobResult], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> str:
        """Submit a voice objective to AgentSandbox asynchronously.

        Returns immediately with a job_id. Calls on_status zero or more times,
        then exactly one of on_complete or on_error on the background thread.

        Args:
            objective: natural-language task from voice transcription
            agents: optional per-job agent configs
            crew_id: optional crew definition to use
            context: optional additional context text
            on_status: called each poll cycle with current job status
            on_complete: called when job reaches terminal state with final result
            on_error: called if submission or polling fails

        Returns:
            job_id for tracking
        """
        job_id = uuid.uuid4().hex[:12]

        def run() -> None:
            import time

            start_time = time.monotonic()
            try:
                job = self._client.submit_job(
                    objective=objective,
                    agents=agents,
                    crew_id=crew_id,
                    context=context,
                )
                job_id_from_server = job.id

                with self._lock:
                    self._active_jobs[job_id] = job

                if on_status:
                    on_status(job)

                # Poll to completion
                final = self._client.poll_job(
                    job_id_from_server,
                    poll_interval=self._poll_interval,
                    max_wait=self._max_wait,
                )

                with self._lock:
                    self._active_jobs.pop(job_id, None)

                duration = time.monotonic() - start_time
                result = CrewJobResult(
                    job_id=job_id_from_server,
                    objective=objective,
                    status=final.status,
                    artifacts=list(final.artifacts),
                    output_text=final.agent_output,
                    errors=final.errors,
                    duration_seconds=duration,
                )
                if on_complete:
                    on_complete(result)

            except ASBJobError as exc:
                with self._lock:
                    self._active_jobs.pop(job_id, None)
                if on_error:
                    on_error(exc)
            except Exception as exc:
                with self._lock:
                    self._active_jobs.pop(job_id, None)
                if on_error:
                    on_error(exc)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return job_id

    def cancel(self, job_id: str) -> bool:
        """Attempt to cancel an active job."""
        with self._lock:
            job = self._active_jobs.get(job_id)
        if job is None:
            return False
        return self._client.cancel_job(job.id)

    def active_job_ids(self) -> list[str]:
        """Return ids of jobs currently being tracked."""
        with self._lock:
            return list(self._active_jobs.keys())

    def get_status(self, job_id: str) -> ASBJobStatus | None:
        """Return current status for a tracked job."""
        with self._lock:
            return self._active_jobs.get(job_id)


# ---------------------------------------------------------------------------
# GlitchScribe Action Wheel integration helpers
# ---------------------------------------------------------------------------

CREW_ACTION_KIND = "crew"
"""WheelAction.kind value that routes to AgentSandbox dispatch."""


def is_crew_action(instruction: str, kind: str) -> bool:
    """Return True if this WheelAction should be dispatched via AgentSandbox."""
    if kind == CREW_ACTION_KIND:
        return True
    # Also support ai-kind actions with the !crew prefix in instruction
    return instruction.strip().startswith("!crew ") if kind == "ai" else False


def extract_crew_objective(instruction: str, selected_text: str) -> tuple[str, str]:
    """Parse a crew WheelAction instruction into (objective, context).

    Supports two forms:
      !crew Summarize this document
      (direct objective text in instruction, selected_text as context)
    """
    stripped = instruction.strip()
    if stripped.startswith("!crew "):
        objective = stripped[6:].strip()
        context = selected_text
    else:
        objective = stripped
        context = selected_text
    return objective, context
