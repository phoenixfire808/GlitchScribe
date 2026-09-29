"""Window Shield core.

A pure, testable module that monitors top-level visible windows, identifies
terminal/shell windows by executable name (and optional title keywords), and
hides them with ``ShowWindow(SW_HIDE)`` without terminating the underlying
processes.

Design goals
------------

* Platform code is isolated behind a small :class:`PlatformAdapter` protocol.
  A real :class:`WindowsAdapter` uses ``ctypes``/``user32``; on non-Windows
  or when ``user32`` is unavailable, a :class:`NoOpAdapter` activates that
  lets the rest of the system run for tests and cross-platform imports.
* The monitor loop is intentionally tiny -- ``tick()`` performs a single
  scan, returns the *new* block events it produced, and never blocks.
  Threading/scheduling lives in the UI / launch layer.
* Config persists as JSON under ``%LOCALAPPDATA%\\DrewWindowShield\\config.json``
  by default; the path can be overridden (used by tests).

This module deliberately does no process termination, no focus stealing, no
global hook installation, and no rewriting of the system policy.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

LOGGER = logging.getLogger("drew_local_voice.window_shield")

DEFAULT_TERMINAL_EXECUTABLES: tuple[str, ...] = (
    "windowsterminal.exe",
    "wt.exe",
    "cmd.exe",
    "powershell.exe",
    "pwsh.exe",
    "conhost.exe",
    "mintty.exe",
    "bash.exe",
    "ubuntu.exe",
    "wsl.exe",
)

DEFAULT_TITLE_KEYWORDS: tuple[str, ...] = (
    "terminal",
    "shell",
    "command prompt",
)

# System / UI processes that must never be hidden even if their names happen to
# match a heuristic. Match is by lowercased executable basename.
DEFAULT_NEVER_HIDE_EXECUTABLES: tuple[str, ...] = (
    "explorer.exe",
    "dwm.exe",
    "winlogon.exe",
    "csrss.exe",
    "lsass.exe",
    "services.exe",
    "svchost.exe",
    "taskhostw.exe",
    "winsystemtrayicons.exe",
    "windowshield.exe",
    "window_shield.exe",
    "drewwindowshield.exe",
)

DEFAULT_POLL_SECONDS: float = 0.5
MIN_POLL_SECONDS: float = 0.05
MAX_POLL_SECONDS: float = 10.0
DEFAULT_CONFIG_DIRNAME = "DrewWindowShield"
DEFAULT_CONFIG_FILENAME = "config.json"


# ---------------------------------------------------------------------------
# Platform adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowInfo:
    """Snapshot of a single top-level window at a point in time."""

    hwnd: int
    pid: int
    process_name: str  # lowercased basename, no directory
    title: str
    visible: bool


class PlatformAdapter(Protocol):
    """Minimal platform surface the shield depends on."""

    name: str

    def enum_visible_windows(self) -> list[WindowInfo]: ...

    def hide_window(self, hwnd: int) -> bool: ...

    def show_window(self, hwnd: int) -> bool: ...

    def is_window_visible(self, hwnd: int) -> bool: ...

    def current_pid(self) -> int: ...


class NoOpAdapter:
    """Cross-platform fallback used when ``user32`` is not importable.

    Reports no windows and no-ops every action. The shield still loads,
    parses config, and is fully unit-testable on Linux/macOS.
    """

    name = "noop"

    def __init__(self) -> None:
        self._hidden: set[int] = set()
        self._visible_windows: dict[int, WindowInfo] = {}
        self._pid = os.getpid()

    # Test-friendly hooks --------------------------------------------------

    def set_visible_windows(self, windows: Iterable[WindowInfo]) -> None:
        self._visible_windows = {int(w.hwnd): w for w in windows}

    def hide_window(self, hwnd: int) -> bool:
        hwnd_i = int(hwnd)
        if hwnd_i not in self._visible_windows:
            return False
        self._hidden.add(hwnd_i)
        self._visible_windows.pop(hwnd_i, None)
        return True

    def show_window(self, hwnd: int) -> bool:
        hwnd_i = int(hwnd)
        if hwnd_i not in self._hidden:
            return False
        self._hidden.discard(hwnd_i)
        self._visible_windows.setdefault(
            hwnd_i,
            WindowInfo(
                hwnd=hwnd_i, pid=0, process_name="", title="", visible=True
            ),
        )
        return True

    def is_window_visible(self, hwnd: int) -> bool:
        return int(hwnd) in self._visible_windows

    def enum_visible_windows(self) -> list[WindowInfo]:
        return sorted(
            self._visible_windows.values(),
            key=lambda w: int(w.hwnd),
        )

    def current_pid(self) -> int:
        return self._pid


# ---------------------------------------------------------------------------
# Windows adapter (ctypes user32)
# ---------------------------------------------------------------------------

SW_HIDE = 0
SW_SHOW = 5
SW_RESTORE = 9

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000

# IsWindowVisible / GetWindowLongPtrW require 64-bit-safe calls on x64 Python.
_user32 = None
try:  # pragma: no cover - exercised only on Windows
    _user32 = ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]
except (OSError, AttributeError):
    _user32 = None


def _have_user32() -> bool:
    return bool(_user32 and sys.platform == "win32")


# psapi is needed to translate a process handle -> module name -> basename.
_psapi = None
try:  # pragma: no cover - exercised only on Windows
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)  # type: ignore[attr-defined]
except (OSError, AttributeError):
    _psapi = None


if _user32 is not None:  # pragma: no cover - Windows-only branch
    _EnumWindowsProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

    _user32.EnumWindows.argtypes = [_EnumWindowsProc, wt.LPARAM]
    _user32.EnumWindows.restype = wt.BOOL

    _user32.IsWindowVisible.argtypes = [wt.HWND]
    _user32.IsWindowVisible.restype = wt.BOOL

    _user32.GetWindowTextLengthW.argtypes = [wt.HWND]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int

    _user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
    _user32.GetWindowTextW.restype = ctypes.c_int

    _user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
    _user32.ShowWindow.restype = wt.BOOL

    _user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wt.DWORD

    _user32.GetAncestor.argtypes = [wt.HWND, ctypes.c_uint]
    _user32.GetAncestor.restype = wt.HWND

    GA_ROOT = 2


def _trim_to_basename(value: str) -> str:
    value = value or ""
    value = value.replace("\\", "/")
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    return value.strip().lower()


class WindowsAdapter:
    """Real ``user32``-backed adapter."""

    name = "windows"

    def __init__(self) -> None:
        if not _have_user32():
            raise RuntimeError(
                "WindowsAdapter requires user32 (Windows-only)."
            )

    # Process query ---------------------------------------------------------

    @staticmethod
    def _open_process(pid: int) -> Optional[int]:
        if pid <= 0:
            return None
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        return int(handle) if handle else None

    @staticmethod
    def _close_handle(handle: int) -> None:
        try:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - best-effort cleanup
            pass

    def _process_name(self, pid: int) -> str:
        handle = self._open_process(pid)
        if not handle:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(512)
            # GetModuleBaseNameW is available on Vista+ and is the cheapest
            # way to get just the executable name (no path).
            try:
                ok = _psapi.GetModuleBaseNameW(handle, None, buf, len(buf))  # type: ignore[attr-defined]
            except AttributeError:
                ok = 0
            if ok:
                return _trim_to_basename(buf.value)

            # QueryFullProcessImageNameW is more reliable for some processes
            # and from callbacks in restricted desktop environments.
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.QueryFullProcessImageNameW.argtypes = [
                ctypes.c_void_p,
                wt.DWORD,
                wt.LPWSTR,
                ctypes.POINTER(wt.DWORD),
            ]
            kernel32.QueryFullProcessImageNameW.restype = wt.BOOL
            path_buf = ctypes.create_unicode_buffer(32768)
            path_len = wt.DWORD(len(path_buf))
            if kernel32.QueryFullProcessImageNameW(
                handle, 0, path_buf, ctypes.byref(path_len)
            ):
                return _trim_to_basename(path_buf.value)
            return ""
        finally:
            self._close_handle(handle)

    # Window enumeration ----------------------------------------------------

    def enum_visible_windows(self) -> list[WindowInfo]:
        assert _user32 is not None  # noqa: S101 - guarded by _have_user32
        results: list[WindowInfo] = []
        seen: set[int] = set()

        pid_buf = wt.DWORD(0)

        def _cb(hwnd: int, _lparam: int) -> bool:
            try:
                if not _user32.IsWindowVisible(hwnd):
                    return True
                root = _user32.GetAncestor(hwnd, GA_ROOT)
                if root and root != hwnd:
                    # Only top-level windows.
                    return True
                if int(hwnd) in seen:
                    return True
                seen.add(int(hwnd))

                pid_buf.value = 0
                _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_buf))
                pid = int(pid_buf.value)
                name = self._process_name(pid)
                if not name:
                    # Skip windows whose owning process we can't query -- if
                    # we can't read the name we can't make a safe decision.
                    return True

                length = _user32.GetWindowTextLengthW(hwnd) or 1
                buf = ctypes.create_unicode_buffer(max(2, length + 1))
                _user32.GetWindowTextW(hwnd, buf, len(buf))
                title = (buf.value or "").strip()

                results.append(
                    WindowInfo(
                        hwnd=int(hwnd),
                        pid=pid,
                        process_name=name,
                        title=title,
                        visible=True,
                    )
                )
            except Exception:  # pragma: no cover - never let an enum crash
                LOGGER.debug("enum callback failure", exc_info=True)
            return True

        try:
            _user32.EnumWindows(_EnumWindowsProc(_cb), 0)
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug("EnumWindows failure", exc_info=True)
        return results

    # Show / hide / check ----------------------------------------------------

    @staticmethod
    def hide_window(hwnd: int) -> bool:
        if not _have_user32():
            return False
        try:
            return bool(_user32.ShowWindow(int(hwnd), SW_HIDE))
        except Exception:  # pragma: no cover
            LOGGER.debug("ShowWindow(SW_HIDE) failed for hwnd=%s", hwnd,
                          exc_info=True)
            return False

    @staticmethod
    def show_window(hwnd: int) -> bool:
        if not _have_user32():
            return False
        try:
            return bool(_user32.ShowWindow(int(hwnd), SW_RESTORE))
        except Exception:  # pragma: no cover
            LOGGER.debug("ShowWindow(SW_RESTORE) failed for hwnd=%s", hwnd,
                          exc_info=True)
            return False

    @staticmethod
    def is_window_visible(hwnd: int) -> bool:
        if not _have_user32():
            return False
        try:
            return bool(_user32.IsWindowVisible(int(hwnd)))
        except Exception:  # pragma: no cover
            return False

    @staticmethod
    def current_pid() -> int:
        return int(os.getpid())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ShieldConfig:
    enabled: bool = True
    poll_interval_seconds: float = DEFAULT_POLL_SECONDS
    blocked_executables: list[str] = field(
        default_factory=lambda: list(DEFAULT_TERMINAL_EXECUTABLES)
    )
    title_keywords: list[str] = field(
        default_factory=lambda: list(DEFAULT_TITLE_KEYWORDS)
    )
    allowlist_executables: list[str] = field(default_factory=list)
    never_hide_executables: list[str] = field(
        default_factory=lambda: list(DEFAULT_NEVER_HIDE_EXECUTABLES)
    )
    use_title_keyword_match: bool = True
    log_max_events: int = 500

    # --- derived -----------------------------------------------------------

    @classmethod
    def defaults(cls) -> "ShieldConfig":
        return cls()

    def cloned(self) -> "ShieldConfig":
        return ShieldConfig(
            enabled=self.enabled,
            poll_interval_seconds=self.poll_interval_seconds,
            blocked_executables=list(self.blocked_executables),
            title_keywords=list(self.title_keywords),
            allowlist_executables=list(self.allowlist_executables),
            never_hide_executables=list(self.never_hide_executables),
            use_title_keyword_match=self.use_title_keyword_match,
            log_max_events=self.log_max_events,
        )

    # --- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["poll_interval_seconds"] = float(self.poll_interval_seconds)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ShieldConfig":
        cfg = cls.defaults()
        if not isinstance(data, dict):
            return cfg
        if "enabled" in data:
            cfg.enabled = bool(data["enabled"])
        if "poll_interval_seconds" in data:
            try:
                cfg.poll_interval_seconds = _clamp_poll(
                    float(data["poll_interval_seconds"])  # type: ignore[arg-type]
                )
            except (TypeError, ValueError):
                pass
        if "blocked_executables" in data:
            cfg.blocked_executables = _clean_string_list(
                data["blocked_executables"]  # type: ignore[arg-type]
            ) or list(DEFAULT_TERMINAL_EXECUTABLES)
        if "title_keywords" in data:
            cfg.title_keywords = _clean_string_list(
                data["title_keywords"]  # type: ignore[arg-type]
            ) or list(DEFAULT_TITLE_KEYWORDS)
        if "allowlist_executables" in data:
            cfg.allowlist_executables = _clean_string_list(
                data["allowlist_executables"]  # type: ignore[arg-type]
            )
        if "never_hide_executables" in data:
            cfg.never_hide_executables = _clean_string_list(
                data["never_hide_executables"]  # type: ignore[arg-type]
            ) or list(DEFAULT_NEVER_HIDE_EXECUTABLES)
        if "use_title_keyword_match" in data:
            cfg.use_title_keyword_match = bool(data["use_title_keyword_match"])
        if "log_max_events" in data:
            try:
                cfg.log_max_events = max(50, int(data["log_max_events"]))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
        return cfg


def _clean_string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in values:
        if not isinstance(raw, str):
            continue
        text = _trim_to_basename(raw)
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _clamp_poll(value: float) -> float:
    return max(MIN_POLL_SECONDS, min(MAX_POLL_SECONDS, float(value)))


def default_config_path() -> Path:
    """Return the default on-disk config path.

    Honours ``%LOCALAPPDATA%`` on Windows and ``$XDG_CONFIG_HOME`` /
    ``~/.config`` elsewhere. Tests pass an explicit override.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / DEFAULT_CONFIG_DIRNAME / DEFAULT_CONFIG_FILENAME
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / DEFAULT_CONFIG_DIRNAME / DEFAULT_CONFIG_FILENAME


def load_config(
    *,
    path: Optional[Path] = None,
    adapter: Optional[PlatformAdapter] = None,
) -> ShieldConfig:
    """Load config; return defaults if the file is missing or corrupt.

    ``adapter`` is only used to enrich the never-hide list with the
    process name of the current host (so a self-named ``window_shield.exe``
    is excluded automatically).
    """
    cfg = ShieldConfig.defaults()
    target = Path(path) if path is not None else default_config_path()
    if target.is_file():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            cfg = ShieldConfig.from_dict(data)  # type: ignore[arg-type]
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "Could not read Window Shield config %s: %s -- using defaults",
                target,
                exc,
            )
            cfg = ShieldConfig.defaults()
    # Always ensure current process name is in the never-hide list.
    current = ""
    try:
        current = _trim_to_basename(sys.argv[0] or "")
    except Exception:  # pragma: no cover
        current = ""
    if not current and adapter is not None:
        try:
            windows = adapter.enum_visible_windows()
            if windows:
                current = windows[0].process_name
        except Exception:  # pragma: no cover
            current = ""
    if current and current not in cfg.never_hide_executables:
        cfg.never_hide_executables = list(cfg.never_hide_executables) + [current]
    return cfg


def save_config(cfg: ShieldConfig, *, path: Optional[Path] = None) -> Path:
    target = Path(path) if path is not None else default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(target)
    return target


# ---------------------------------------------------------------------------
# Block event log + monitor
# ---------------------------------------------------------------------------


@dataclass
class BlockEvent:
    timestamp: str
    hwnd: int
    pid: int
    process_name: str
    title: str
    matched_by: str  # "executable" | "title" | "both"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class EventLog:
    """Bounded, thread-safe ring buffer of :class:`BlockEvent` records."""

    def __init__(self, *, max_events: int = 500) -> None:
        self._events: list[BlockEvent] = []
        self._lock = threading.Lock()
        self._max = max(50, int(max_events))

    @property
    def max_events(self) -> int:
        return self._max

    def configure_max(self, value: int) -> None:
        with self._lock:
            self._max = max(50, int(value))
            if len(self._events) > self._max:
                # Drop oldest first; the list is small (max ~5000).
                del self._events[: len(self._events) - self._max]

    def append(self, event: BlockEvent) -> None:
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max:
                del self._events[: len(self._events) - self._max]

    def extend(self, events: Iterable[BlockEvent]) -> None:
        for event in events:
            self.append(event)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def snapshot(self) -> list[BlockEvent]:
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def __iter__(self) -> Iterable[BlockEvent]:
        return iter(self.snapshot())


class WindowShield:
    """Coordinator: applies policy to platform windows and records events."""

    def __init__(
        self,
        *,
        config: ShieldConfig,
        adapter: PlatformAdapter,
        log: Optional[EventLog] = None,
        clock=time.monotonic,
    ) -> None:
        self._config = config.cloned()
        self._adapter = adapter
        self._log = log or EventLog(max_events=config.log_max_events)
        self._clock = clock
        self._known_hidden: dict[int, float] = {}
        self._session_unblocked: set[int] = set()
        self._last_status: str = "starting"

    # --- accessors ---------------------------------------------------------

    @property
    def config(self) -> ShieldConfig:
        return self._config.cloned()

    @property
    def adapter(self) -> PlatformAdapter:
        return self._adapter

    @property
    def events(self) -> EventLog:
        return self._log

    @property
    def status(self) -> str:
        return self._last_status

    def update_config(self, *, config: Optional[ShieldConfig] = None,
                      **changes: object) -> ShieldConfig:
        if config is not None:
            self._config = config.cloned()
        else:
            for key, value in changes.items():
                if not hasattr(self._config, key):
                    raise AttributeError(f"Unknown config key: {key}")
                setattr(self._config, key, value)
        self._config.poll_interval_seconds = _clamp_poll(
            float(self._config.poll_interval_seconds)
        )
        self._log.configure_max(int(self._config.log_max_events))
        return self._config.cloned()

    # --- policy ------------------------------------------------------------

    def is_terminal(self, info: WindowInfo) -> Optional[str]:
        """Return the match reason (``"executable"`` / ``"title"`` / ``"both"``)
        if ``info`` should be hidden, otherwise ``None``.
        """
        name = (info.process_name or "").lower()
        title = (info.title or "").lower()

        blocked_by_exe = bool(name) and name in self._config.blocked_executables
        if blocked_by_exe and name in self._config.allowlist_executables:
            blocked_by_exe = False
        if name and name in self._config.never_hide_executables:
            return None
        if info.pid == self._adapter.current_pid():
            return None
        if int(info.hwnd) in self._session_unblocked:
            return None
        matched_by_title = False
        if (
            self._config.use_title_keyword_match
            and title
            and any(kw and kw in title for kw in self._config.title_keywords)
        ):
            matched_by_title = True
            if name in self._config.allowlist_executables:
                matched_by_title = False
        if blocked_by_exe and matched_by_title:
            return "both"
        if blocked_by_exe:
            return "executable"
        if matched_by_title:
            return "title"
        return None

    # --- scanning ----------------------------------------------------------

    def tick(self) -> list[BlockEvent]:
        """Run a single scan.

        Returns the *new* block events produced during this call.
        """
        events: list[BlockEvent] = []
        if not self._config.enabled:
            self._last_status = "paused"
            return events
        try:
            windows = self._adapter.enum_visible_windows()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("enum_visible_windows failed: %s", exc)
            self._last_status = "error:enum"
            return events

        for window in windows:
            reason = self.is_terminal(window)
            if not reason:
                continue
            if self._adapter.hide_window(int(window.hwnd)):
                self._record_event(window, reason)
                events.append(self._last_event(window, reason))
                self._known_hidden[int(window.hwnd)] = float(self._clock())
            else:
                LOGGER.debug(
                    "hide_window returned False for hwnd=%s pid=%s",
                    window.hwnd,
                    window.pid,
                )

        self._last_status = (
            f"active · {len(self._known_hidden)} hidden"
            if self._config.enabled
            else "paused"
        )
        return events

    # --- controls ----------------------------------------------------------

    def pause(self) -> None:
        self._config.enabled = False
        self._last_status = "paused"

    def resume(self) -> None:
        self._config.enabled = True
        self._last_status = "active"

    @property
    def is_paused(self) -> bool:
        return not self._config.enabled

    def unblock_hwnd(self, hwnd: int) -> bool:
        """Restore a window that the shield previously hid."""
        hwnd_i = int(hwnd)
        self._session_unblocked.add(hwnd_i)
        if not self._adapter.show_window(hwnd_i):
            return False
        self._known_hidden.pop(hwnd_i, None)
        return True

    def restore_all(self) -> int:
        """Restore every window hidden by this shield instance."""
        restored = 0
        for hwnd in list(self._known_hidden):
            if self._adapter.show_window(hwnd):
                restored += 1
            self._known_hidden.pop(hwnd, None)
        return restored

    def clear_log(self) -> int:
        cleared = len(self._log)
        self._log.clear()
        return cleared

    # --- helpers -----------------------------------------------------------

    def _record_event(self, info: WindowInfo, reason: str) -> None:
        self._log.append(
            BlockEvent(
                timestamp=_utcnow_iso(),
                hwnd=int(info.hwnd),
                pid=int(info.pid),
                process_name=info.process_name,
                title=info.title,
                matched_by=reason,
            )
        )

    def _last_event(self, info: WindowInfo, reason: str) -> BlockEvent:
        return BlockEvent(
            timestamp=_utcnow_iso(),
            hwnd=int(info.hwnd),
            pid=int(info.pid),
            process_name=info.process_name,
            title=info.title,
            matched_by=reason,
        )


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_default_adapter() -> PlatformAdapter:
    """Return a :class:`WindowsAdapter` if available, else :class:`NoOpAdapter`."""
    if _have_user32():
        try:
            return WindowsAdapter()
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug("WindowsAdapter construction failed", exc_info=True)
    return NoOpAdapter()


def default_config_path_or_default_path(path: Optional[Path]) -> Path:
    """Small helper for tests/UI: returns ``path`` or :func:`default_config_path`."""
    return Path(path) if path is not None else default_config_path()


__all__ = [
    "DEFAULT_TERMINAL_EXECUTABLES",
    "DEFAULT_TITLE_KEYWORDS",
    "DEFAULT_NEVER_HIDE_EXECUTABLES",
    "DEFAULT_POLL_SECONDS",
    "MIN_POLL_SECONDS",
    "MAX_POLL_SECONDS",
    "BlockEvent",
    "EventLog",
    "NoOpAdapter",
    "PlatformAdapter",
    "ShieldConfig",
    "WindowInfo",
    "WindowShield",
    "WindowsAdapter",
    "build_default_adapter",
    "default_config_path",
    "load_config",
    "save_config",
]
