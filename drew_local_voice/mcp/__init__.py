"""Model Context Protocol (MCP) integration layer for GlitchScribe.

Supports three transport mechanisms: Stdio (local subprocess), HTTP (REST webhooks),
and SSE (streaming events). Every tool access goes through a static filter that
enforces the principle of least privilege.
"""

from __future__ import annotations

from .adapter import MCPServerAdapter, MCPAdapterConfig
from .filter import ToolFilter, create_static_tool_filter
from .protocol import MCPMessage, MCPRequest, MCPResponse, MCPTool
from .transport import MCPServerHTTP, MCPServerSSE, MCPServerStdio, MCPTransport

__all__ = [
    "MCPTransport",
    "MCPServerStdio",
    "MCPServerHTTP",
    "MCPServerSSE",
    "MCPMessage",
    "MCPRequest",
    "MCPResponse",
    "MCPTool",
    "MCPServerAdapter",
    "MCPAdapterConfig",
    "ToolFilter",
    "create_static_tool_filter",
]
