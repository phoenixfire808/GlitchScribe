"""AgentSandbox API client — dispatches CrewAI workflows from GlitchScribe.

Base URL configurable, default http://127.0.0.1:8090.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


class ASBError(Exception):
    """Base AgentSandbox API error."""
    pass


class ASBConnectionError(ASBError):
    """Cannot reach AgentSandbox server."""
    pass


class ASBJobError(ASBError):
    """Job execution failed."""
    pass


@dataclass
class ASBJobStatus:
    """Status of an AgentSandbox job."""

    id: str
    status: str  # pending | running | completed | failed | cancelled
    agent_id: str = ""
    agent_output: str = ""
    result: Optional[dict] = None
    artifacts: list[dict] = field(default_factory=list)
    errors: tuple[str, ...] = field(default_factory=tuple)
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    @property
    def is_running(self) -> bool:
        return self.status in ("pending", "running")


class AgentSandboxClient:
    """HTTP client for the AgentSandbox API.

    Usage:
        client = AgentSandboxClient()
        job = client.submit_job("Summarize my project notes", agents=[...])
        status = client.poll_job(job.id)
        while not status.is_terminal:
            time.sleep(5)
            status = client.poll_job(job.id)
        print(status.artifacts)
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090",
        timeout: float = 30.0,
        default_poll_interval: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_poll_interval = default_poll_interval

    def _request(
        self,
        method: str,
        path: str,
        data: dict | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the AgentSandbox API."""
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode("utf-8") if data is not None else None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        try:
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))
        except urllib.error.URLError as exc:
            if hasattr(exc, "code"):
                if exc.code == 404:
                    raise ASBError(f"endpoint not found: {path}") from exc
                if exc.code >= 500:
                    raise ASBConnectionError(f"AgentSandbox server error: {exc}") from exc
            raise ASBConnectionError(f"cannot reach AgentSandbox at {self.base_url}: {exc}") from exc
        except TimeoutError as exc:
            raise ASBConnectionError(f"timeout connecting to {self.base_url}") from exc

    def health_check(self) -> dict[str, Any]:
        """Check if AgentSandbox is reachable."""
        try:
            return self._request("GET", "/health")
        except ASBError:
            return {"status": "error", "reachable": False}
        return {"status": "ok", "reachable": True}

    def submit_job(
        self,
        objective: str,
        agents: list[dict] | None = None,
        crew_id: str | None = None,
        context: str = "",
    ) -> ASBJobStatus:
        """Submit a new job to AgentSandbox.

        Args:
            objective: natural-language task description
            agents: optional list of agent configs (role, backstory, tools, etc.)
            crew_id: optional crew definition id
            context: optional context text

        Returns ASBJobStatus with id and initial status.
        """
        payload: dict[str, Any] = {
            "objective": objective,
            "context": context,
        }
        if agents:
            payload["agents"] = agents
        if crew_id:
            payload["crew_id"] = crew_id

        try:
            result = self._request("POST", "/jobs", data=payload)
            return self._parse_job(result)
        except ASBError:
            raise
        except Exception as exc:
            raise ASBError(f"failed to submit job: {exc}") from exc

    def get_job(self, job_id: str) -> ASBJobStatus:
        """Get the current status of a job."""
        result = self._request("GET", f"/jobs/{job_id}")
        return self._parse_job(result)

    def poll_job(
        self,
        job_id: str,
        poll_interval: float | None = None,
        max_wait: float = 600.0,
    ) -> ASBJobStatus:
        """Poll until the job reaches a terminal state.

        Args:
            job_id: the job id to poll
            poll_interval: seconds between polls (default: client's default)
            max_wait: maximum total seconds to wait (default: 600s = 10min)

        Returns the final ASBJobStatus.
        Raises ASBJobError if the job fails.
        """
        interval = poll_interval or self.default_poll_interval
        deadline = time.monotonic() + max_wait

        status = self.get_job(job_id)
        while not status.is_terminal:
            if time.monotonic() >= deadline:
                raise ASBJobError(f"job {job_id} did not complete within {max_wait}s")
            time.sleep(interval)
            status = self.get_job(job_id)

        if status.status == "failed":
            error_msg = "; ".join(status.errors) if status.errors else "unknown"
            raise ASBJobError(f"job {job_id} failed: {error_msg}")

        return status

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job. Returns True on success."""
        try:
            self._request("POST", f"/jobs/{job_id}/cancel")
            return True
        except ASBError:
            return False

    def list_jobs(
        self,
        status_filter: str | None = None,
        limit: int = 20,
    ) -> list[ASBJobStatus]:
        """List recent jobs."""
        params = f"?limit={limit}"
        if status_filter:
            params += f"&status={urllib.parse.quote(status_filter)}"
        result = self._request("GET", f"/jobs{params}")
        jobs = result if isinstance(result, list) else result.get("jobs", [])
        return [self._parse_job(j) for j in jobs]

    def _parse_job(self, data: dict[str, Any]) -> ASBJobStatus:
        """Parse raw API response dict into ASBJobStatus."""
        return ASBJobStatus(
            id=data.get("id", ""),
            status=data.get("status", "unknown"),
            agent_id=data.get("agent_id", ""),
            agent_output=data.get("agent_output", ""),
            result=data.get("result"),
            artifacts=data.get("artifacts", []),
            errors=tuple(data.get("errors", [])),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
        )


# ---------------------------------------------------------------------------
# Job status polling helper
# ---------------------------------------------------------------------------


def poll_with_callback(
    client: AgentSandboxClient,
    job_id: str,
    on_status: callable,  # (ASBJobStatus) -> None
    poll_interval: float = 5.0,
    max_wait: float = 600.0,
) -> ASBJobStatus:
    """Poll a job, calling on_status after each status update.

    Useful for showing "Thinking..." bubbles in the GlitchScribe overlay.
    """
    deadline = time.monotonic() + max_wait
    status = client.get_job(job_id)
    on_status(status)

    while not status.is_terminal:
        if time.monotonic() >= deadline:
            raise ASBJobError(f"job {job_id} timed out after {max_wait}s")
        time.sleep(poll_interval)
        status = client.get_job(job_id)
        on_status(status)

    return status
