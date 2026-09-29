"""MCP server adapter: translates MCP protocol into agent Tool protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .protocol import MCPResponse, MCPAccessViolation
from .transport import MCPTransport


@dataclass
class MCPAdapterConfig:
    """Configuration for an MCP server adapter."""

    transport: MCPTransport
    server_name: str = "mcp"
    tool_filter: Optional[Any] = None  # ToolFilter or None
    connect_on_init: bool = True
    timeout: float = 30.0


class MCPServerAdapter:
    """Bridges MCP protocol to the agent Tool interface.

    Acts as a context manager. Use `tools()` to enumerate available tools
    and `call(name, args)` to invoke one.
    """

    def __init__(self, config: MCPAdapterConfig) -> None:
        self._config = config
        self._transport = config.transport
        self._server_name = config.server_name
        self._tool_filter = config.tool_filter
        self._connected = False
        self._cached_tools: list[str] = []

    def __enter__(self) -> "MCPServerAdapter":
        if self._config.connect_on_init:
            self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def connect(self) -> None:
        """Establish connection to the MCP server."""
        if self._connected:
            return
        try:
            self._transport.connect()
            self._connected = True
        except Exception:
            self._connected = False
            raise

    def close(self) -> None:
        """Close the transport."""
        self._transport.close()
        self._connected = False

    def tools(self) -> list[str]:
        """List available tool names from the MCP server."""
        if not self._connected:
            return []

        try:
            resp = self._transport.send(
                __import__("drew_local_voice.mcp.protocol", fromlist=["MCPRequest"]).MCPRequest.list_tools()
            )
            if resp.is_error:
                return []
            self._cached_tools = [t.name for t in resp.tools()]
            return self._cached_tools
        except Exception:
            return []

    def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server.

        Raises:
            MCPAccessViolation — if the tool is blocked by the filter
            MCPConnectionError — if the transport is not connected
        """
        if not self._connected:
            raise MCPAccessViolation(tool_name, "MCP server not connected")

        # Apply filter if set
        if self._tool_filter is not None:
            try:
                self._tool_filter.check(tool_name)
            except Exception as exc:
                # Let the filter's on_violation return the error to the LLM
                if hasattr(self._tool_filter, "on_violation"):
                    result = self._tool_filter.on_violation(tool_name)
                    return f"Access Denied: {result}"
                raise

        try:
            resp = self._transport.call_tool(tool_name, arguments)
            if resp.is_error:
                error_msg = resp.error.get("message", "unknown error") if resp.error else "unknown error"
                return f"Tool error: {error_msg}"
            return resp.content()
        except Exception as exc:
            return f"Tool execution failed: {exc}"

    def is_connected(self) -> bool:
        return self._connected
