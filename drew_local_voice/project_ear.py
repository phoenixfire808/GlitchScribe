"""Authenticated, metadata-only Project Ear bridge contracts."""

from __future__ import annotations

import re
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePosixPath
from pathlib import Path
from typing import Callable, Optional


class InventoryRejected(ValueError):
    """The editor payload violated the metadata-only inventory contract."""


@dataclass(frozen=True)
class ActiveFileMetadata:
    relative_path: str
    language_id: str


@dataclass(frozen=True)
class DiagnosticMetadata:
    relative_path: str
    severity: int
    code: str
    source: str
    start_line: int
    start_character: int
    end_line: int
    end_character: int


@dataclass(frozen=True)
class WorkspaceInventory:
    client_id: str
    workspace_names: tuple[str, ...]
    relative_paths: tuple[str, ...]
    received_file_count: int
    truncated: bool
    observed_at: float
    active_file: Optional[ActiveFileMetadata] = None
    diagnostics: tuple[DiagnosticMetadata, ...] = ()
    active_file_approved: bool = False
    diagnostics_approved: bool = False
    protocol: int = 1


class ProjectEarRegistry:
    def __init__(self, *, max_files: int = 5_000) -> None:
        self.max_files = max(1, min(int(max_files), 20_000))
        self._lock = threading.Lock()
        self._snapshot: Optional[WorkspaceInventory] = None

    @property
    def snapshot(self) -> Optional[WorkspaceInventory]:
        with self._lock:
            return self._snapshot

    def ingest(
        self,
        payload: object,
        *,
        clock: Callable[[], float] = time.time,
    ) -> WorkspaceInventory:
        if not isinstance(payload, dict) or payload.get("protocol") not in {1, 2}:
            raise InventoryRejected("unsupported Project Ear protocol")
        protocol = int(payload["protocol"])
        client_id = _bounded_text(payload.get("client_id"), "client_id", 128)
        raw_names = payload.get("workspace_names")
        raw_paths = payload.get("relative_paths")
        if not isinstance(raw_names, list) or len(raw_names) > 32:
            raise InventoryRejected("workspace_names must be a bounded list")
        if not isinstance(raw_paths, list) or len(raw_paths) > 20_000:
            raise InventoryRejected("relative_paths must be a bounded list")
        names = tuple(_bounded_text(value, "workspace name", 128) for value in raw_names)
        paths = tuple(_safe_relative_path(value) for value in raw_paths)
        active_file = None
        diagnostics: tuple[DiagnosticMetadata, ...] = ()
        active_file_approved = False
        diagnostics_approved = False
        if protocol == 2:
            approvals = payload.get("metadata_approvals")
            if not isinstance(approvals, dict) or set(approvals) != {
                "active_file",
                "diagnostics",
            }:
                raise InventoryRejected("metadata approvals must be explicit")
            active_file_approved = approvals.get("active_file")
            diagnostics_approved = approvals.get("diagnostics")
            if not isinstance(active_file_approved, bool) or not isinstance(
                diagnostics_approved, bool
            ):
                raise InventoryRejected("metadata approvals must be boolean")
            raw_active = payload.get("active_file")
            if raw_active is not None:
                if not active_file_approved:
                    raise InventoryRejected("active file metadata was not approved")
                active_file = _active_file_metadata(raw_active)
            raw_diagnostics = payload.get("diagnostics", [])
            if not isinstance(raw_diagnostics, list) or len(raw_diagnostics) > 200:
                raise InventoryRejected("diagnostics must be a bounded list")
            if raw_diagnostics and not diagnostics_approved:
                raise InventoryRejected("diagnostic metadata was not approved")
            diagnostics = tuple(
                _diagnostic_metadata(value) for value in raw_diagnostics
            )
        snapshot = WorkspaceInventory(
            client_id=client_id,
            workspace_names=names,
            relative_paths=paths[: self.max_files],
            received_file_count=len(paths),
            truncated=len(paths) > self.max_files,
            observed_at=float(clock()),
            active_file=active_file,
            diagnostics=diagnostics,
            active_file_approved=active_file_approved,
            diagnostics_approved=diagnostics_approved,
            protocol=protocol,
        )
        with self._lock:
            self._snapshot = snapshot
        return snapshot


def _bounded_text(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise InventoryRejected(f"{label} must be text")
    rendered = value.strip()
    if not rendered or len(rendered) > maximum or "\x00" in rendered:
        raise InventoryRejected(f"{label} is empty or exceeds its bound")
    return rendered


def _safe_relative_path(value: object) -> str:
    rendered = _bounded_text(value, "relative path", 512).replace("\\", "/")
    if rendered.startswith("/") or re.match(r"^[A-Za-z]:/", rendered):
        raise InventoryRejected("absolute paths are not allowed")
    path = PurePosixPath(rendered)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise InventoryRejected("relative path traversal is not allowed")
    return path.as_posix()


def _active_file_metadata(value: object) -> ActiveFileMetadata:
    if not isinstance(value, dict) or set(value) != {"relative_path", "language_id"}:
        raise InventoryRejected("active file metadata has unsupported fields")
    language_id = _bounded_text(value.get("language_id"), "language ID", 64)
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", language_id):
        raise InventoryRejected("language ID contains unsupported characters")
    return ActiveFileMetadata(
        relative_path=_safe_relative_path(value.get("relative_path")),
        language_id=language_id,
    )


def _bounded_integer(value: object, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise InventoryRejected(f"{label} is outside its bound")
    return value


def _diagnostic_metadata(value: object) -> DiagnosticMetadata:
    fields = {
        "relative_path",
        "severity",
        "code",
        "source",
        "start_line",
        "start_character",
        "end_line",
        "end_character",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise InventoryRejected("diagnostic metadata has unsupported fields")
    start_line = _bounded_integer(value.get("start_line"), "start line", 10_000_000)
    start_character = _bounded_integer(
        value.get("start_character"), "start character", 1_000_000
    )
    end_line = _bounded_integer(value.get("end_line"), "end line", 10_000_000)
    end_character = _bounded_integer(
        value.get("end_character"), "end character", 1_000_000
    )
    if (end_line, end_character) < (start_line, start_character):
        raise InventoryRejected("diagnostic range is reversed")
    return DiagnosticMetadata(
        relative_path=_safe_relative_path(value.get("relative_path")),
        severity=_bounded_integer(value.get("severity"), "severity", 3),
        code=_bounded_text(value.get("code"), "diagnostic code", 128),
        source=_bounded_text(value.get("source"), "diagnostic source", 64),
        start_line=start_line,
        start_character=start_character,
        end_line=end_line,
        end_character=end_character,
    )


class ProjectEarServer:
    def __init__(
        self,
        registry: ProjectEarRegistry,
        rendezvous_path: Path,
        *,
        token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
    ) -> None:
        self.registry = registry
        self.rendezvous_path = Path(rendezvous_path)
        self.token = str(token_factory())
        if len(self.token) < 16:
            raise ValueError("Project Ear token is too short")
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._server is not None:
            return
        registry = self.registry
        token = self.token

        class Handler(BaseHTTPRequestHandler):
            server_version = "GlitchScribeProjectEar/1"
            sys_version = ""

            def log_message(self, _format: str, *_args) -> None:
                return

            def do_PUT(self) -> None:
                if self.path != "/v1/workspace-inventory":
                    self._json(404, {"error": "not_found"})
                    return
                supplied = self.headers.get("Authorization", "")
                expected = f"Bearer {token}"
                if not hmac.compare_digest(supplied, expected):
                    self._json(401, {"error": "unauthorized"})
                    return
                if self.headers.get("Origin"):
                    self._json(403, {"error": "browser_origin_rejected"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    length = 0
                if not 0 < length <= 262_144:
                    self._json(413, {"error": "payload_too_large"})
                    return
                if not self.headers.get("Content-Type", "").startswith(
                    "application/json"
                ):
                    self._json(415, {"error": "json_required"})
                    return
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    snapshot = registry.ingest(payload)
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    InventoryRejected,
                ) as exc:
                    self._json(400, {"error": type(exc).__name__})
                    return
                self._json(
                    202,
                    {
                        "accepted": True,
                        "file_count": len(snapshot.relative_paths),
                        "received_file_count": snapshot.received_file_count,
                        "truncated": snapshot.truncated,
                    },
                )

            def _json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        port = int(self._server.server_address[1])
        self._publish_rendezvous(port)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="GlitchScribeProjectEar",
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        self._remove_owned_rendezvous()

    def _publish_rendezvous(self, port: int) -> None:
        self.rendezvous_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = {
            "protocol": 1,
            "host": "127.0.0.1",
            "port": port,
            "token": self.token,
            "pid": os.getpid(),
            "created_at": time.time(),
        }
        temporary = self.rendezvous_path.with_name(
            f".{self.rendezvous_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, self.rendezvous_path)

    def _remove_owned_rendezvous(self) -> None:
        try:
            descriptor = json.loads(self.rendezvous_path.read_text(encoding="utf-8"))
            if hmac.compare_digest(str(descriptor.get("token", "")), self.token):
                self.rendezvous_path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            return
