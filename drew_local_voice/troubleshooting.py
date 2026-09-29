"""Privacy-safe, rotation-aware troubleshooting log for GlitchScribe.

The troubleshooting log is intentionally separate from the existing
``runtime-events.jsonl`` so that structured, incident-focused diagnostics
remain readable even when the broader event stream is busy. It is designed
to answer Drew's recurring questions without the noise of the full pipeline:

* "Did my hotkey press reach the app?"
* "Was the keyboard hook installed?"
* "Why did the engine reject my recording?"
* "Is the model stuck in a reload loop?"

Design rules (mirroring ``AGENTS.md`` product boundaries):

* Never log transcript text, selected text, captured audio, clipboard
  contents, credentials, or target-field contents. Only bounded metadata,
  action IDs, modes, timing, and failure classes.
* The on-disk log is a single JSONL file that is rotated when it exceeds
  a configurable byte budget. Two historical generations are kept on disk
  so fresh questions can be answered without losing the recent trail.
* Hot-path emit helpers never raise. Failures are bounded to the three
  obvious modes (path missing, disk full, JSON encoding) and surfaced only
  through a co-operating ``append_runtime_event`` call so the app never
  crashes because the logger could not write.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional


# Reserved field names that may never leak transcript / capture content.
# These are intentionally narrow; the existing runtime events already
# forbid a few useful keys, but the troubleshooting layer is stricter.
_FORBIDDEN_FIELDS = frozenset(
    {
        "text",
        "transcript",
        "partial",
        "final_text",
        "audio",
        "audio_bytes",
        "clipboard",
        "clipboard_contents",
        "selection",
        "selected_text",
        "password",
        "captured",
        "target_field_text",
    }
)

# Maximum payload size for any single field. We do not want a single
# error path to dump a stack trace or a long device name into the log.
_MAX_FIELD_BYTES = 2048
# Maximum troubleshooting log size before rotation (default ~5 MB).
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
# In addition to the active log, keep ``MAX_ARCHIVE`` rotated copies.
DEFAULT_MAX_ARCHIVE = 2


@dataclass(frozen=True)
class TroubleshootingConfig:
    """Configurable knobs for the troubleshooting sink.

    ``cooperating_sink`` lets the project wire the troubleshooting layer
    into the existing ``runtime-events.jsonl`` without creating a circular
    import. When unset, the troubleshooting log writes to its own file.
    """

    path: Path
    max_bytes: int = DEFAULT_MAX_BYTES
    max_archive: int = DEFAULT_MAX_ARCHIVE
    cooperating_sink: Optional[Callable[[str, Mapping[str, Any]], None]] = None


def _safe_field(value: Any) -> Any:
    """Coerce a value to a JSON-safe, bounded representation."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        text = value
        if len(text) > _MAX_FIELD_BYTES:
            return text[:_MAX_FIELD_BYTES] + f"...<trimmed:{len(text) - _MAX_FIELD_BYTES}>"
        return text
    if isinstance(value, (list, tuple)):
        return [_safe_field(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _safe_field(item) for key, item in value.items()}
    return str(value)[:_MAX_FIELD_BYTES]


def _sanitize_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Drop forbidden fields and coerce the rest to JSON-safe primitives."""
    sanitized: dict[str, Any] = {}
    for key, value in fields.items():
        key_str = str(key)
        key_norm = key_str.lower().strip()
        if key_norm in _FORBIDDEN_FIELDS:
            continue
        sanitized[key_str] = _safe_field(value)
    return sanitized


class TroubleshootingLog:
    """Append-only, rotating, privacy-safe troubleshooting log.

    The class is intentionally tiny and thread-safe. It is safe to share a
    single instance across the main thread, the keyboard-hook thread, and
    the engine worker thread. All writes are guarded by a single lock so
    JSONL ordering is preserved across threads.
    """

    def __init__(
        self,
        config: TroubleshootingConfig,
        *,
        clock: Callable[[], float] = time.time,
        pid_provider: Callable[[], int] = os.getpid,
    ) -> None:
        self._config = config
        self._clock = clock
        self._pid_provider = pid_provider
        self._lock = threading.Lock()
        try:
            self._config.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            # The directory is best-effort; subsequent writes will retry.
            pass

    @property
    def path(self) -> Path:
        return self._config.path

    def emit(self, event: str, **fields: Any) -> None:
        """Append a single troubleshooting event.

        This call is non-blocking and never raises; failures are silently
        swallowed so that the diagnostics layer can never crash the app.
        """
        try:
            payload = _sanitize_fields(fields)
            payload.update(
                {
                    "timestamp": round(self._clock(), 3),
                    "event": str(event),
                    "pid": int(self._pid_provider()),
                    "source": "troubleshooting",
                }
            )
            line = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        except (TypeError, ValueError):
            # Last-resort: never crash the app because the logger
            # rejected an exotic value. Drop the payload entirely.
            return
        with self._lock:
            try:
                self._maybe_rotate_locked(len(line) + 1)
                with self._config.path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            except OSError:
                # Disk full, file locked, or path missing. The cooperating
                # sink (when present) will still carry the event.
                pass
        sink = self._config.cooperating_sink
        if sink is not None:
            try:
                sink(event, payload)
            except Exception:
                pass

    # Convenience helpers --------------------------------------------------

    def hotkey_pressed(self, **fields: Any) -> None:
        self.emit("hotkey_pressed", **fields)

    def hotkey_match(self, **fields: Any) -> None:
        self.emit("hotkey_match", **fields)

    def hotkey_miss(self, **fields: Any) -> None:
        self.emit("hotkey_miss", **fields)

    def hotkey_hook_installed(self, **fields: Any) -> None:
        self.emit("hotkey_hook_installed", **fields)

    def hotkey_hook_install_failed(self, **fields: Any) -> None:
        self.emit("hotkey_hook_install_failed", **fields)

    def hotkey_callback_error(self, **fields: Any) -> None:
        self.emit("hotkey_callback_error", **fields)

    def hotkey_session_reset(self, **fields: Any) -> None:
        self.emit("hotkey_session_reset", **fields)

    def engine_rejected(self, **fields: Any) -> None:
        self.emit("engine_rejected", **fields)

    def engine_reload_alarm(self, **fields: Any) -> None:
        self.emit("engine_reload_alarm", **fields)

    def view_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent ``limit`` lines as parsed events.

        This is the single most useful read path for Drew when he asks
        "did anything happen?". The caller passes the bare limit; this
        function loads the active log plus the rotated archives in
        chronological order so the most recent events are first.
        """
        limit = max(1, int(limit))
        collected: list[dict[str, Any]] = []
        paths = [self._config.path, *self._archive_paths()]
        for path in paths:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as stream:
                    lines = stream.readlines()
            except OSError:
                continue
            for raw in lines:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(parsed, dict):
                    collected.append(parsed)
        collected.sort(key=lambda item: float(item.get("timestamp", 0.0)))
        if len(collected) > limit:
            return collected[-limit:]
        return collected

    def events_filtered(
        self,
        *,
        event: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return recent events matching the optional ``event`` filter."""
        recent = self.view_recent(limit=max(limit * 4, limit))
        if event is None:
            return recent[-limit:]
        wanted = event.strip().lower()
        matched = [item for item in recent if str(item.get("event", "")).lower() == wanted]
        return matched[-limit:]

    def summarize_hotkey(self, limit: int = 200) -> dict[str, Any]:
        """Count categories of hotkey-related events for fast triage."""
        summary = {
            "pressed": 0,
            "matches": 0,
            "misses": 0,
            "hook_installed": 0,
            "hook_install_failed": 0,
            "callback_errors": 0,
            "session_resets": 0,
            "last_event": None,
        }
        for record in self.view_recent(limit=limit):
            name = str(record.get("event", ""))
            summary["last_event"] = record
            if name == "hotkey_pressed":
                summary["pressed"] += 1
            elif name == "hotkey_match":
                summary["matches"] += 1
            elif name == "hotkey_miss":
                summary["misses"] += 1
            elif name == "hotkey_hook_installed":
                summary["hook_installed"] += 1
            elif name == "hotkey_hook_install_failed":
                summary["hook_install_failed"] += 1
            elif name == "hotkey_callback_error":
                summary["callback_errors"] += 1
            elif name == "hotkey_session_reset":
                summary["session_resets"] += 1
        return summary

    # Internal rotation logic ---------------------------------------------

    def _archive_paths(self) -> Iterable[Path]:
        max_archive = max(0, int(self._config.max_archive))
        for index in range(1, max_archive + 1):
            yield self._config.path.with_suffix(self._config.path.suffix + f".{index}")

    def _maybe_rotate_locked(self, incoming_bytes: int) -> None:
        if incoming_bytes <= 0:
            return
        if int(self._config.max_bytes) <= 0:
            return
        try:
            current_size = self._config.path.stat().st_size
        except OSError:
            return
        if current_size + incoming_bytes <= int(self._config.max_bytes):
            return
        # Rotate oldest -> newest. archive.1 is the most recent prior file.
        archives = list(self._archive_paths())
        for index in range(len(archives) - 1, 0, -1):
            destination = archives[index]
            source = archives[index - 1]
            try:
                if destination.exists():
                    destination.unlink()
            except OSError:
                pass
            try:
                source.replace(destination)
            except OSError:
                pass
        try:
            self._config.path.replace(archives[0])
        except OSError:
            # If rename fails, just truncate so the next write can land.
            try:
                with self._config.path.open("w", encoding="utf-8"):
                    pass
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Module-level default singleton. The GlitchScribe runtime installs a
# configured instance during process startup; tests and modules that are
# imported before startup can still call ``record_troubleshooting`` and
# the helper will skip silently when no instance is installed.
# ---------------------------------------------------------------------------

_DEFAULT_LOCK = threading.Lock()
_DEFAULT: Optional[TroubleshootingLog] = None


def configure_default(log: TroubleshootingLog) -> None:
    """Install the process-wide troubleshooting log."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        _DEFAULT = log


def default_log() -> Optional[TroubleshootingLog]:
    with _DEFAULT_LOCK:
        return _DEFAULT


def record_troubleshooting(event: str, **fields: Any) -> None:
    """Emit a troubleshooting event if a default log is configured."""
    log = default_log()
    if log is None:
        return
    log.emit(event, **fields)


def resolve_troubleshooting_path(app_dir: Path) -> Path:
    """Return the default troubleshooting log path for the given app dir."""
    return Path(app_dir) / "troubleshooting.jsonl"
