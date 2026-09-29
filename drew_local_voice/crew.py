"""Crew definition: a team of agents with a defined process and shared LLM."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from drew_local_voice.ai.providers import AIProvider


class ProcessType(str, Enum):
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"


@dataclass(frozen=True)
class CrewConfig:
    """Frozen configuration for a crew of agents."""

    id: str = ""
    process: ProcessType = ProcessType.SEQUENTIAL
    manager_role: str = "Manager"
    verbose: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", uuid.uuid4().hex[:12])


@dataclass
class Crew:
    """Runtime crew: holds crew config + live agent instances.

    Created once per kickoff, NOT reused across runs.
    """

    config: CrewConfig
    agents: tuple["Agent", ...] = field(default_factory=tuple)
    _provider: Optional["AIProvider"] = field(default=None, init=False, repr=False)
    _default_model: str = field(default="", init=False)
    _default_temperature: float = field(default=0.3, init=False)
    _default_max_tokens: int = field(default=2048, init=False)

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("crew must have at least one agent")
        if self.config.process == ProcessType.HIERARCHICAL:
            delegator = next((a for a in self.agents if a.allow_delegation), None)
            if delegator is None:
                raise ValueError(
                    "hierarchical process requires at least one agent with allow_delegation=True "
                    "(the manager)"
                )

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def process(self) -> ProcessType:
        return self.config.process

    @property
    def agent_map(self) -> dict[str, "Agent"]:
        return {a.id: a for a in self.agents}

    def set_provider(
        self,
        provider: "AIProvider",
        *,
        default_model: str = "",
        default_temperature: float = 0.3,
        default_max_tokens: int = 2048,
    ) -> None:
        self._provider = provider
        self._default_model = default_model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

    @property
    def provider(self) -> "AIProvider":
        if self._provider is None:
            raise RuntimeError("crew provider not set — call crew.set_provider() before kickoff")
        return self._provider

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def default_temperature(self) -> float:
        return self._default_temperature

    @property
    def default_max_tokens(self) -> int:
        return self._default_max_tokens

    def agent(self, agent_id: str) -> "Agent":
        """Look up an agent by id."""
        agent = self.agent_map.get(agent_id)
        if agent is None:
            raise KeyError(f"agent not found in crew: {agent_id!r}")
        return agent

    def coworker_ids(self, for_agent_id: str) -> tuple[str, ...]:
        """Return the ids of all agents except the requesting agent."""
        return tuple(a.id for a in self.agents if a.id != for_agent_id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "process": self.process.value,
            "agents": [a.to_dict() for a in self.agents],
        }


@dataclass
class TaskResult:
    """Result of a single task executed by an agent."""

    agent_id: str
    task_description: str
    output: str
    success: bool
    errors: tuple[str, ...] = ()
    iterations: int = 0
