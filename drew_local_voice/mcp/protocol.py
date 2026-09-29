"""MCP message protocol types — request, response, tool, and connection errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import json


class MCPError(Exception):
    """Base error for all MCP operations."""
    pass


class MCPConnectionError(MCPError):
    """Failed to connect to the MCP server."""
    pass


class MCPTimeoutError(MCPError):
    """MCP request timed out."""
    pass


class MCPTransportError(MCPError):
    """Underlying transport failure."""
    pass


class MCPAccessViolation(MCPError):
    """Tool access denied by filter."""

    def __init__(self, tool_name: str, message: str = "access denied") -> None:
        self.tool_name = tool_name
        super().__init__(f"tool {tool_name!r}: {message}")


@dataclass(frozen=True)
class MCPTool:
    """One tool exposed by an MCP server."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPTool":
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            input_schema=dict(data.get("inputSchema", {})),
        )


@dataclass
class MCPRequest:
    """One MCP JSON-RPC request."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    def to_json(self) -> bytes:
        payload = {"jsonrpc": "2.0", "method": self.method, "params": self.params}
        if self.id is not None:
            payload["id"] = self.id
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    @classmethod
    def list_tools(cls) -> "MCPRequest":
        return cls(method="tools/list", params={}, id=1)

    @classmethod
    def call_tool(cls, name: str, arguments: dict[str, Any]) -> "MCPRequest":
        return cls(method="tools/call", params={"name": name, "arguments": arguments}, id=2)


@dataclass
class MCPResponse:
    """One MCP JSON-RPC response."""

    id: Optional[int] = None
    result: Any = None
    error: Optional[dict[str, Any]] = None

    @classmethod
    def from_json(cls, raw: bytes) -> "MCPResponse":
        data = json.loads(raw.decode("utf-8"))
        error = None
        if "error" in data:
            error = dict(data["error"])
        return cls(
            id=data.get("id"),
            result=data.get("result"),
            error=error,
        )

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def tools(self) -> list[MCPTool]:
        """Extract tool list from result."""
        if self.error or not self.result:
            return []
        tools = self.result.get("tools", []) if isinstance(self.result, dict) else []
        return [MCPTool.from_dict(t) for t in tools]

    def content(self) -> str:
        """Extract text content from a call_tool result."""
        if self.error or not self.result:
            return ""
        content = self.result.get("content", []) if isinstance(self.result, dict) else []
        texts = [c.get("text", "") for c in content if isinstance(c, dict)]
        return "\n".join(texts)


@dataclass
class MCPMessage:
    """Generic MCP message wrapper for logging/debugging."""

    direction: str  # "send" | "recv"
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    raw: bytes = field(default_factory=b"", repr=False)

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "method": self.method,
            "params": self.params,
        }
