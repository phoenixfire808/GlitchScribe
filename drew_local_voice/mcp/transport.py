"""MCP transport implementations: Stdio, HTTP, and SSE."""

from __future__ import annotations

import subprocess
import json
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .protocol import (
    MCPRequest,
    MCPResponse,
    MCPConnectionError,
    MCPTimeoutError,
    MCPTransportError,
    MCPMessage,
)


@dataclass
class MCPTransport(ABC):
    """Abstract base for MCP server transports."""

    timeout: float = 30.0

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the MCP server."""
        ...

    @abstractmethod
    def send(self, request: MCPRequest) -> MCPResponse:
        """Send a request and wait for response."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the transport."""
        ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPResponse:
        """Convenience: send a tools/call request."""
        return self.send(MCPRequest.call_tool(name, arguments))


# ---------------------------------------------------------------------------
# Stdio transport
# ---------------------------------------------------------------------------


@dataclass
class MCPServerStdio(MCPTransport):
    """Spawns a local subprocess and communicates via stdin/stdout.

    Ideal for fully localized execution with no network traversal.
    """

    command: str = "python"
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    _process: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _request_id: int = field(default=0, init=False)

    def connect(self) -> None:
        import os

        if self._process is not None:
            return

        full_env = {**os.environ, **self.env}
        try:
            self._process = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=full_env,
                text=False,
            )
        except FileNotFoundError as exc:
            raise MCPConnectionError(f"command not found: {self.command!r}") from exc

    def send(self, request: MCPRequest) -> MCPResponse:
        if self._process is None:
            raise MCPTransportError("not connected — call connect() first")

        with self._lock:
            req_id = self._request_id
            self._request_id += 1
            request.id = req_id

            try:
                raw = request.to_json() + b"\n"
                self._process.stdin.write(raw)  # type: ignore[union-attr]
                self._process.stdin.flush()  # type: ignore[union-attr]

                # Read until we get a JSON-RPC response with matching id
                start = time.monotonic()
                while True:
                    if time.monotonic() - start > self.timeout:
                        raise MCPTimeoutError(f"request {req_id} timed out after {self.timeout}s")
                    line = self._process.stdout.readline()  # type: ignore[union-attr]
                    if not line:
                        raise MCPTransportError("MCP server process terminated")
                    try:
                        resp = MCPResponse.from_json(line)
                        if resp.id == req_id:
                            return resp
                    except (json.JSONDecodeError, KeyError):
                        continue

            except (BrokenPipeError, OSError) as exc:
                raise MCPTransportError(f"stdio communication error: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._process is not None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                self._process = None


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------


@dataclass
class MCPServerHTTP(MCPTransport):
    """Connects to a REST endpoint for MCP tool execution."""

    base_url: str = "http://127.0.0.1:8080"
    headers: dict[str, str] = field(default_factory=dict)

    def connect(self) -> None:
        # HTTP is connectionless; no persistent connection needed
        pass

    def send(self, request: MCPRequest) -> MCPResponse:
        import urllib.request

        url = f"{self.base_url.rstrip('/')}/mcp"
        payload = request.to_json()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", **self.headers},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return MCPResponse.from_json(resp.read())
        except urllib.error.URLError as exc:
            raise MCPConnectionError(f"HTTP MCP request failed: {exc}") from exc
        except TimeoutError as exc:
            raise MCPTimeoutError(f"HTTP MCP request timed out after {self.timeout}s") from exc

    def close(self) -> None:
        pass  # connectionless


# ---------------------------------------------------------------------------
# SSE transport (Server-Sent Events)
# ---------------------------------------------------------------------------


@dataclass
class MCPServerSSE(MCPTransport):
    """Unidirectional real-time streaming from a remote server via SSE.

    Used for monitoring external logs, live data streams, or long-running async tasks.
    """

    base_url: str = "http://127.0.0.1:8080"
    headers: dict[str, str] = field(default_factory=dict)
    _buffer: str = field(default="", init=False)
    _listeners: list[Callable[[MCPMessage], None]] = field(
        default_factory=list, init=False
    )

    def connect(self) -> None:
        pass

    def add_listener(self, handler: Callable[[MCPMessage], None]) -> None:
        self._listeners.append(handler)

    def _process_sse_line(self, line: str) -> None:
        """Parse one SSE line and dispatch to listeners."""
        if line.startswith("data: "):
            data = line[len("data: "):]
            try:
                msg = json.loads(data)
                if isinstance(msg, dict) and "method" in msg:
                    mcp_msg = MCPMessage(
                        direction="recv",
                        method=msg.get("method", ""),
                        params=msg.get("params", {}),
                    )
                    for listener in self._listeners:
                        try:
                            listener(mcp_msg)
                        except Exception:
                            pass
            except json.JSONDecodeError:
                pass

    def send(self, request: MCPRequest) -> MCPResponse:
        # SSE is primarily for server→client streaming; requests go via HTTP POST
        raise NotImplementedError("use HTTP transport for request/response on SSE-backed servers")

    def close(self) -> None:
        self._listeners.clear()
