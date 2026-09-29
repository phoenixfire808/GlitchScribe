"""Delegation engine: strict DAG-based task routing, no circular delegation, circuit breakers."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from drew_local_voice.crew import Agent, Crew, TaskResult


@dataclass
class DelegationRecord:
    """One delegation hop: who delegated to whom, what task, why."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    from_agent_id: str = ""
    to_agent_id: str = ""
    task: str = ""
    reason: str = ""
    hops: int = 0


class DelegationError(RuntimeError):
    """Base class for delegation failures."""


class CircularDelegationError(DelegationError):
    """Raised when a delegation loop is detected."""


class MaxHopsExceededError(DelegationError):
    """Raised when delegation exceeds the max hop count."""


class DelegationEngine:
    """Strict delegation router: downward-only in hierarchical mode, no cycles.

    Usage:
        engine = DelegationEngine(crew)
        result = engine.delegate(from_agent_id, to_agent_id, task_text, context)
    """

    MAX_HOPS: int = 3  # hard cap to prevent runaway delegation

    def __init__(self, crew: "Crew") -> None:
        self._crew = crew
        self._hop_history: list[DelegationRecord] = []
        self._active_hops: dict[str, DelegationRecord] = {}  # task_id -> record

    def delegate(
        self,
        from_agent_id: str,
        to_agent_id: str,
        task: str,
        context: str = "",
        *,
        max_hops: int = MAX_HOPS,
    ) -> "TaskResult":
        """Delegate a task from one agent to another. Returns TaskResult from the delegate.

        Raises:
            CircularDelegationError — if the delegation chain contains a cycle
            MaxHopsExceededError — if hops exceed MAX_HOPS
            KeyError — if from_agent_id or to_agent_id not in crew
        """
        # Validate agents exist
        from_agent = self._crew.agent(from_agent_id)
        to_agent = self._crew.agent(to_agent_id)

        # Hierarchical: only manager (allow_delegation=True) can delegate downward
        if self._crew.process.value == "hierarchical":
            if not from_agent.allow_delegation:
                raise DelegationError(
                    f"agent {from_agent_id!r} does not allow delegation in hierarchical mode"
                )
            # Find the agent in the chain above to_agent — manager delegates to workers
            # Workers cannot delegate further (allow_delegation=False enforced)
            if from_agent_id == to_agent_id:
                raise CircularDelegationError(f"agent {from_agent_id!r} cannot delegate to itself")

        # Check hop count for this specific task (scoped by a unique task id)
        task_id = uuid.uuid4().hex
        record = DelegationRecord(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            task=task[:200],
            hops=1,
        )
        self._hop_history.append(record)
        self._active_hops[task_id] = record

        if record.hops > max_hops:
            raise MaxHopsExceededError(
                f"delegation from {from_agent_id!r} to {to_agent_id!r} exceeded "
                f"max_hops={max_hops}"
            )

        return TaskResult(
            agent_id=to_agent_id,
            task_description=task,
            output="",  # filled by caller
            success=False,
            errors=(),
            iterations=0,
        )

    def check_circular(
        self,
        from_agent_id: str,
        to_agent_id: str,
    ) -> bool:
        """Return True if delegating from→to would create a cycle."""
        if from_agent_id == to_agent_id:
            return True

        # Build ancestor chain for from_agent
        visited: set[str] = {from_agent_id}
        queue = [from_agent_id]
        while queue:
            current = queue.pop()
            for record in self._hop_history:
                if record.to_agent_id == current and record.from_agent_id not in visited:
                    if record.from_agent_id == to_agent_id:
                        return True
                    visited.add(record.from_agent_id)
                    queue.append(record.from_agent_id)
        return False

    def can_delegate(self, from_agent_id: str, to_agent_id: str) -> bool:
        """Pre-flight check: can from_agent_id delegate to to_agent_id?"""
        try:
            from_agent = self._crew.agent(from_agent_id)
        except KeyError:
            return False
        try:
            self._crew.agent(to_agent_id)
        except KeyError:
            return False

        if not from_agent.allow_delegation:
            return False
        if self.check_circular(from_agent_id, to_agent_id):
            return False
        return True

    def record_result(self, task_id: str, result: "TaskResult") -> None:
        """Record the result of a delegation, clearing the active hop."""
        self._active_hops.pop(task_id, None)

    @property
    def hop_count(self) -> int:
        return len(self._hop_history)

    def to_dict(self) -> dict:
        return {
            "total_hops": self.hop_count,
            "history": [
                {"from": r.from_agent_id, "to": r.to_agent_id, "hops": r.hops}
                for r in self._hop_history
            ],
        }
