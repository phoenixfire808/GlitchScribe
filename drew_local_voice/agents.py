"""Multi-agent primitives for GlitchScribe — no CrewAI dependency, no LiteLLM."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, ClassVar

if TYPE_CHECKING:
    from drew_local_voice.ai.providers import AIProvider


@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for one agent in a crew."""

    id: str
    role: str
    backstory: str = ""
    goal: str = ""
    allow_delegation: bool = False
    max_iter: int = 5
    temperature: float = 0.3
    max_tokens: int = 2048
    tools: tuple["AgentTool", ...] = ()
    model: str = ""  # empty = inherit from crew default

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("agent id cannot be empty")
        if not self.role.strip():
            raise ValueError("agent role cannot be empty")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")


@dataclass
class Agent:
    """Runtime agent: holds config + mutable execution state for one agent in a crew.

    Agents are created from AgentConfig by the Crew at kickoff time.
    They are NOT frozen — execution state (iter_count, last_output, errors) mutates per run.
    """

    config: AgentConfig
    crew_id: str
    _iter_count: int = field(default=0, init=False)
    _last_output: str = field(default="", init=False)
    _errors: list[str] = field(default_factory=list, init=False)

    # Callable tools this agent can use. Each must be a callable that accepts
    # (text: str, context: str) and returns str.
    tools: tuple[Callable[[str, str], str], ...] = field(default_factory=tuple, init=False)

    def __post_init__(self) -> None:
        # Resolve tools from config (lazy import to avoid circular)
        from drew_local_voice.agent_tools import resolve_tools

        self.tools = resolve_tools(self.config.tools)

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def role(self) -> str:
        return self.config.role

    @property
    def backstory(self) -> str:
        return self.config.backstory

    @property
    def goal(self) -> str:
        return self.config.goal

    @property
    def allow_delegation(self) -> bool:
        return self.config.allow_delegation

    @property
    def max_iter(self) -> int:
        return self.config.max_iter

    @property
    def iter_count(self) -> int:
        return self._iter_count

    @property
    def last_output(self) -> str:
        return self._last_output

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(self._errors)

    def increment_iter(self) -> int:
        self._iter_count += 1
        return self._iter_count

    def set_output(self, text: str) -> None:
        self._last_output = text

    def add_error(self, error: str) -> None:
        self._errors.append(str(error)[:500])

    def clear_errors(self) -> None:
        self._errors.clear()

    def system_prompt(self, task: str, context: str, coworkers: tuple[str, ...] = ()) -> str:
        """Build the system prompt for this agent given the current task and context."""
        parts = [
            f"You are a {self.role}.",
        ]
        if self.backstory:
            parts.append(f"Backstory: {self.backstory}")
        if self.goal:
            parts.append(f"Your goal: {self.goal}")
        parts.append(f"\nCurrent task: {task}")
        if context:
            parts.append(f"\nContext:\n{context}")
        if self.tools:
            tool_names = ", ".join(t.__name__ for t in self.tools)
            parts.append(f"\nYou have access to these tools: {tool_names}")
            parts.append(
                "If you need to use a tool, invoke it and the result will be returned to you. "
                "If a tool fails after 3 attempts, halt immediately and state 'Information Unavailable'."
            )
        if self.allow_delegation and coworkers:
            parts.append(f"\nYou may delegate to these coworkers: {', '.join(coworkers)}")
        if not self.allow_delegation:
            parts.append("\nDo not delegate tasks. Handle everything yourself.")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "backstory": self.backstory,
            "goal": self.goal,
            "allow_delegation": self.allow_delegation,
            "max_iter": self.max_iter,
            "iter_count": self._iter_count,
            "errors": self._errors,
        }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentTool:
    """Declaration of one tool available to an agent (name + description only).

    Actual callable resolution happens at runtime via resolve_tools().
    """

    name: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name cannot be empty")


def resolve_tools(specs: tuple[AgentTool, ...]) -> tuple[Callable[[str, str], str], ...]:
    """Resolve AgentTool specs into actual callable tool functions.

    Tools are registered via register_tool(). The registry is module-level
    so agents can enumerate available tools at startup.
    """
    from drew_local_voice.agent_tools import TOOL_REGISTRY

    resolved = []
    for spec in specs:
        func = TOOL_REGISTRY.get(spec.name)
        if func is None:
            raise ValueError(f"agent tool not registered: {spec.name!r}")
        resolved.append(func)
    return tuple(resolved)


def register_tool(name: str, func: Callable[[str, str], str]) -> None:
    """Register a tool function so agents can reference it by name."""
    from drew_local_voice.agent_tools import TOOL_REGISTRY

    TOOL_REGISTRY[name] = func


# ---------------------------------------------------------------------------
# Built-in read-only tools (safe defaults)
# ---------------------------------------------------------------------------


def _read_file_tool(text: str, context: str) -> str:
    """Tool: read a local file and return its contents. Path given in text arg."""
    import os
    path = text.strip()
    if not path:
        return "Error: no file path provided"
    try:
        full_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(full_path):
            return f"Error: file not found: {full_path}"
        with open(full_path, encoding="utf-8") as fh:
            return fh.read()[:8000]  # cap output
    except Exception as exc:
        return f"Error reading file: {exc}"


def _list_dir_tool(text: str, context: str) -> str:
    """Tool: list a directory. Path given in text arg."""
    import os
    path = text.strip() or "."
    try:
        full_path = os.path.abspath(os.path.expanduser(path))
        entries = os.listdir(full_path)
        return "\n".join(sorted(entries))
    except Exception as exc:
        return f"Error listing directory: {exc}"


def _grep_tool(text: str, context: str) -> str:
    """Tool: grep pattern in text (pattern in text arg, content in context arg)."""
    import re
    parts = text.strip().split(maxsplit=1)
    if not parts:
        return "Error: no pattern provided"
    pattern = parts[0]
    content = parts[1] if len(parts) > 1 else context
    try:
        matches = re.findall(pattern, content)
        if not matches:
            return "No matches found"
        return "\n".join(matches[:50])
    except re.error as exc:
        return f"Regex error: {exc}"


# Pre-register built-in tools
TOOL_REGISTRY: dict[str, Callable[[str, str], str]] = {
    "read_file": _read_file_tool,
    "list_directory": _list_dir_tool,
    "grep": _grep_tool,
}
