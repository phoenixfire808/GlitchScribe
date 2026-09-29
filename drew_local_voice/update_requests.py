"""Durable local update-request queue for the GlitchScribe UI integration lane.

This module provides a small, isolated API the in-app text box can use to push
user-written update intentions into a local JSON-backed queue. The queue is:

* Local: persists under a caller-supplied path. Never reaches the network.
* Bounded: empty / oversized text is rejected. No scripts, secrets, or
  diagnostic context are captured or persisted.
* Atomic: every write goes through a temp file + ``os.replace`` so a crash or
  power loss cannot leave the store in a torn state.
* Reviewable: a Markdown export clearly labels itself as a non-executable
  review handoff so a human (or a later review pass) decides what runs.
* Non-executable: this module never runs, executes, evaluates, or sends the
  stored text anywhere. It is inert storage.

The module has no import-time side effects and depends only on the standard
library.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


SCHEMA_VERSION = 1
STATUS_PENDING = "pending"
STATUS_READY = "ready"
ALLOWED_STATUSES: tuple[str, ...] = (STATUS_PENDING, STATUS_READY)

# Hard upper bound for a single update request body. Keeps the on-disk file
# small and predictable; large inputs are rejected instead of silently
# truncated so the UI lane can surface the problem.
MAX_TEXT_CHARS = 4_000
MAX_REQUESTS = 1_000

# Disallow C0/C1 control characters (other than common whitespace) so the
# queue stays reviewable and cannot smuggle ANSI escapes, BOM mid-stream, or
# other hidden payloads.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class UpdateRequestError(ValueError):
    """Raised when an update request is malformed, missing, or rejected."""


@dataclass(frozen=True)
class UpdateRequest:
    """A single user-written update intention queued for review.

    Only safe review fields are stored. No transcript text, no clipboard
    contents, no credentials, no diagnostic context, and no environment data
    are persisted alongside the request.
    """

    request_id: str
    created_at: str  # ISO-8601 UTC, e.g. "2026-08-05T12:34:56.789012+00:00"
    text: str
    status: str  # one of ALLOWED_STATUSES

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise UpdateRequestError("request_id must be a non-empty string")
        if self.status not in ALLOWED_STATUSES:
            raise UpdateRequestError(
                f"status must be one of {ALLOWED_STATUSES!r}, got {self.status!r}"
            )
        if not isinstance(self.text, str):
            raise UpdateRequestError("text must be a string")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise UpdateRequestError("created_at must be a non-empty ISO-8601 string")

    @property
    def is_pending(self) -> bool:
        return self.status == STATUS_PENDING

    @property
    def is_ready(self) -> bool:
        return self.status == STATUS_READY


def _now_iso() -> str:
    """Return a deterministic, UTC, ISO-8601 timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(text: object) -> str:
    """Validate, trim, and normalize an update request body.

    Empty input, whitespace-only input, or over-long input is rejected so the
    UI lane can show a clear error instead of silently queueing garbage.
    """
    if not isinstance(text, str):
        raise UpdateRequestError("update text must be a string")
    # Normalize line endings so the on-disk file is platform-stable.
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL_CHARS.sub("", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        raise UpdateRequestError("update text cannot be empty or whitespace-only")
    if len(cleaned) > MAX_TEXT_CHARS:
        raise UpdateRequestError(
            f"update text exceeds the local limit of {MAX_TEXT_CHARS} characters"
        )
    return cleaned


def _normalize_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise UpdateRequestError("request_id must be a non-empty string")
    return value


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` as JSON to ``path`` via temp file + os.replace.

    On Windows, ``os.replace`` is atomic for files on the same volume, so a
    crash mid-write leaves the previous file intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                # fsync can fail on some non-file-backed handles; safe to skip.
                pass
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise UpdateRequestError(f"cannot read update-request store: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UpdateRequestError(
            f"update-request store is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise UpdateRequestError("update-request store must be a JSON object")
    if payload.get("schema") != SCHEMA_VERSION:
        raise UpdateRequestError(
            f"unsupported update-request schema: {payload.get('schema')!r}"
        )
    records = payload.get("requests")
    if not isinstance(records, list):
        raise UpdateRequestError("update-request records must be a list")
    return records


def _dump_records(path: Path, records: list[dict]) -> None:
    payload = {
        "schema": SCHEMA_VERSION,
        "requests": records,
    }
    _atomic_write_json(path, payload)


def _record_to_request(record: object) -> Optional[UpdateRequest]:
    if not isinstance(record, dict):
        return None
    try:
        return UpdateRequest(
            request_id=_normalize_id(record.get("request_id")),
            created_at=str(record.get("created_at", "")),
            text=str(record.get("text", "")),
            status=str(record.get("status", "")),
        )
    except UpdateRequestError:
        return None


def _request_to_record(request: UpdateRequest) -> dict:
    return {
        "request_id": request.request_id,
        "created_at": request.created_at,
        "text": request.text,
        "status": request.status,
    }


class UpdateRequestStore:
    """Local, durable, atomic JSON store of user-written update intentions.

    The store is intentionally minimal: append-only ``submit``, a ``pending``
    view for the review list, ``mark_ready`` to flip status, ``clear`` to
    remove a request, and ``export_markdown`` for a human-review handoff.

    Construction does not touch the filesystem until a write method is called.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # ----- internal helpers -----

    def _load(self) -> list[UpdateRequest]:
        records = _load_records(self.path)
        requests: list[UpdateRequest] = []
        for record in records:
            request = _record_to_request(record)
            if request is not None:
                requests.append(request)
        return requests

    def _save(self, requests: list[UpdateRequest]) -> None:
        _dump_records(self.path, [_request_to_record(r) for r in requests])

    # ----- public API -----

    def submit(self, text: str) -> UpdateRequest:
        """Validate ``text`` and append a new pending ``UpdateRequest``.

        Returns the freshly created request. Raises ``UpdateRequestError``
        on empty/over-long input or when the store is at ``MAX_REQUESTS``.
        """
        normalized = _normalize_text(text)
        current = self._load()
        pending_count = sum(1 for r in current if r.status == STATUS_PENDING)
        if pending_count >= MAX_REQUESTS:
            raise UpdateRequestError(
                f"pending queue is full ({MAX_REQUESTS}); clear or mark_ready first"
            )
        request = UpdateRequest(
            request_id=uuid.uuid4().hex,
            created_at=_now_iso(),
            text=normalized,
            status=STATUS_PENDING,
        )
        current.append(request)
        self._save(current)
        return request

    def pending(self) -> tuple[UpdateRequest, ...]:
        """Return all requests whose status is ``pending``, oldest first."""
        return tuple(
            request
            for request in self._load()
            if request.status == STATUS_PENDING
        )

    def all(self) -> tuple[UpdateRequest, ...]:
        """Return every stored request in insertion order."""
        return tuple(self._load())

    def mark_ready(self, request_id: str) -> UpdateRequest:
        """Flip a pending request to ``ready`` and return the updated record.

        Raises ``UpdateRequestError`` if the id is unknown or already ready.
        ``ready`` is a human-review marker only; nothing is executed.
        """
        _normalize_id(request_id)
        current = self._load()
        for index, request in enumerate(current):
            if request.request_id != request_id:
                continue
            if request.status == STATUS_READY:
                raise UpdateRequestError(
                    f"request {request_id!r} is already marked ready"
                )
            updated = UpdateRequest(
                request_id=request.request_id,
                created_at=request.created_at,
                text=request.text,
                status=STATUS_READY,
            )
            current[index] = updated
            self._save(current)
            return updated
        raise UpdateRequestError(f"unknown request_id: {request_id!r}")

    def clear(self, request_id: str) -> None:
        """Remove a request from the store. Unknown ids are an error."""
        _normalize_id(request_id)
        current = self._load()
        kept = [r for r in current if r.request_id != request_id]
        if len(kept) == len(current):
            raise UpdateRequestError(f"unknown request_id: {request_id!r}")
        self._save(kept)

    def export_markdown(self, path: Path | str) -> Path:
        """Write a clearly-labeled, non-executable Markdown review handoff.

        The output is plain prose suitable for a reviewer to triage. It
        contains code-fenced text so a human can copy/paste intent into
        another tool; nothing in this module runs that text. The output
        explicitly states it is a review handoff and is not executable.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        requests = self._load()
        pending = [r for r in requests if r.status == STATUS_PENDING]
        ready = [r for r in requests if r.status == STATUS_READY]

        lines: list[str] = []
        lines.append("# Update Request Review Handoff")
        lines.append("")
        lines.append(
            "> **REVIEW HANDOFF — NOT EXECUTABLE.** "
            "This document is a static Markdown export of queued user-written "
            "update intentions for GlitchScribe. Nothing in this file is run, "
            "compiled, evaluated, or sent anywhere. A human reviewer decides "
            "what (if anything) to act on."
        )
        lines.append("")
        lines.append(f"- Exported at (UTC): `{_now_iso()}`")
        lines.append(f"- Pending requests: **{len(pending)}**")
        lines.append(f"- Ready requests: **{len(ready)}**")
        lines.append(f"- Source store: `{self.path}`")
        lines.append("")
        lines.append("## Pending")
        lines.append("")
        if not pending:
            lines.append("_No pending requests._")
            lines.append("")
        else:
            for index, request in enumerate(pending, start=1):
                lines.append(f"### {index}. `{request.request_id}`")
                lines.append("")
                lines.append(f"- Created (UTC): `{request.created_at}`")
                lines.append(f"- Status: `{request.status}`")
                lines.append("- Text (review only — do not execute):")
                lines.append("")
                lines.append("```text")
                lines.append(request.text)
                lines.append("```")
                lines.append("")
        lines.append("## Ready for follow-up")
        lines.append("")
        if not ready:
            lines.append("_No requests marked ready yet._")
            lines.append("")
        else:
            for index, request in enumerate(ready, start=1):
                lines.append(f"### {index}. `{request.request_id}`")
                lines.append("")
                lines.append(f"- Created (UTC): `{request.created_at}`")
                lines.append(f"- Status: `{request.status}`")
                lines.append("- Text (review only — do not execute):")
                lines.append("")
                lines.append("```text")
                lines.append(request.text)
                lines.append("```")
                lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(
            "End of handoff. This file is a review snapshot generated by "
            "`drew_local_voice.update_requests.UpdateRequestStore.export_markdown`."
        )
        lines.append("")

        # Atomic write so a half-written export cannot be mistaken for a
        # complete review snapshot. We use the same temp-file + os.replace
        # pattern as _atomic_write_json, but for Markdown text rather than
        # JSON so the export is plain, human-readable prose.
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{target.stem}-",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as handle:
                handle.write("\n".join(lines))
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
                temporary_path = Path(handle.name)
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return target
