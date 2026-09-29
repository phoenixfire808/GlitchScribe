"""Static tool filtering — enforce least-privilege tool access per agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


class AccessViolation(Exception):
    """Raised when an agent attempts to use a blocked tool."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"tool access denied: {tool_name!r}")


@dataclass
class ToolFilter:
    """Per-agent tool access filter.

    Created via create_static_tool_filter(). Default-deny: any tool not explicitly
    in the allow list is blocked.
    """

    allowed: frozenset[str]
    on_violation: Callable[[str], str] = field(
        default=lambda name: f"Tool {name!r} is not available to this agent."
    )

    def check(self, tool_name: str) -> str:
        """Check if a tool is allowed. Returns the tool name if allowed.

        Raises AccessViolation if blocked — callers should catch and return the
        on_violation message to the LLM.
        """
        if tool_name not in self.allowed:
            raise AccessViolation(tool_name)
        return tool_name

    def filter_tools(self, tool_names: list[str]) -> list[str]:
        """Return only the allowed tools from a list."""
        return [t for t in tool_names if t in self.allowed]

    def is_allowed(self, tool_name: str) -> bool:
        """True if the tool is in the allow list."""
        return tool_name in self.allowed


def create_static_tool_filter(
    allowed_tool_names: list[str],
    *,
    on_violation: Callable[[str], str] | None = None,
) -> ToolFilter:
    """Create a static tool filter with an explicit allow list.

    Args:
        allowed_tool_names: List of tool names this agent is permitted to use.
            All others are blocked by default.
        on_violation: Optional callback. If provided, instead of raising AccessViolation,
            the filter calls this with the tool name and returns the result to the LLM.
            Signature: (tool_name: str) -> str (error message)

    Example:
        filter = create_static_tool_filter(
            ["read_file", "list_directory"],
            on_violation=lambda name: f"Tool {name!r} access denied"
        )
        result = filter.check("read_file")  # returns "read_file"
        result = filter.check("write_file")  # raises AccessViolation or returns error msg
    """
    return ToolFilter(
        allowed=frozenset(allowed_tool_names),
        on_violation=on_violation if on_violation else AccessViolation,
    )


# ---------------------------------------------------------------------------
# Default filter for read-only local access
# ---------------------------------------------------------------------------

READ_ONLY_FILTER = create_static_tool_filter([
    "read_file",
    "list_directory",
    "grep",
])


# ---------------------------------------------------------------------------
# Filter that blocks all destructive operations
# ---------------------------------------------------------------------------

SAFE_FILTER = create_static_tool_filter([
    "read_file",
    "list_directory",
    "grep",
    "submit_job",
    "get_job_status",
])
