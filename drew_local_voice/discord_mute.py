"""Prior-state-aware Discord microphone isolation for GlitchScribe.

The default integration controls only Discord's Windows capture-audio session.
It does not press Discord's UI mute button, use account tokens, send messages,
change the physical microphone endpoint, or steal foreground focus.
"""

from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .discord_audio import WindowsDiscordCaptureSessions


EventSink = Callable[..., None]


class DiscordMuteBackend(Protocol):
    """Small backend seam so lifecycle behavior can be tested without Discord."""

    def muted_state(self) -> bool | None:
        """Return True/False for Discord self-mute, or None when unavailable."""

    def set_muted(self, muted: bool) -> bool:
        """Request and verify the desired Discord self-mute state."""

    def reconcile_muted(self) -> tuple[int, int, bool]:
        """Own and mute current sessions; return count, new count, verification."""

    def close(self) -> None:
        """Release thread-affine backend resources."""


@dataclass(frozen=True)
class DiscordMuteReceipt:
    operation: str
    status: str
    changed: bool = False
    prior_muted: bool | None = None
    current_muted: bool | None = None
    error_type: str = ""


class WindowsDiscordUIAutomation:
    """Thread-affine UIA adapter for the native Discord desktop client."""

    _CLSID_CUIAUTOMATION = "{FF48DBA4-60EF-4201-AA87-54103EEF594E}"
    _TREE_SCOPE_DESCENDANTS = 4
    _UIA_CONTROL_TYPE_PROPERTY_ID = 30003
    _UIA_BUTTON_CONTROL_TYPE_ID = 50000
    _UIA_INVOKE_PATTERN_ID = 10000
    _UIA_TOGGLE_PATTERN_ID = 10015

    def __init__(self, *, settle_seconds: float = 0.06) -> None:
        if os.name != "nt":
            raise OSError("Discord UI Automation is available only on Windows")
        import comtypes
        import comtypes.client
        from comtypes import GUID
        from comtypes.automation import VARIANT

        comtypes.CoInitialize()
        self._comtypes = comtypes
        self._variant = VARIANT
        comtypes.client.GetModule("UIAutomationCore.dll")
        import comtypes.gen.UIAutomationClient as uia

        self._uia_module = uia
        self._automation = comtypes.client.CreateObject(
            GUID(self._CLSID_CUIAUTOMATION),
            interface=uia.IUIAutomation,
        )
        self._settle_seconds = max(0.0, float(settle_seconds))
        self._closed = False

    @staticmethod
    def _discord_windows() -> tuple[int, ...]:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        process_query_limited_information = 0x1000
        handles: list[int] = []
        callback_type = ctypes.WINFUNCTYPE(
            ctypes.c_bool, wintypes.HWND, wintypes.LPARAM
        )

        def callback(hwnd: int, _lparam: int) -> bool:
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process = kernel32.OpenProcess(
                process_query_limited_information, False, pid.value
            )
            if not process:
                return True
            try:
                size = wintypes.DWORD(32768)
                path = ctypes.create_unicode_buffer(size.value)
                if not kernel32.QueryFullProcessImageNameW(
                    process, 0, path, ctypes.byref(size)
                ):
                    return True
                if path.value.casefold().endswith("\\discord.exe"):
                    handles.append(int(hwnd))
            finally:
                kernel32.CloseHandle(process)
            return True

        user32.EnumWindows(callback_type(callback), 0)
        return tuple(handles)

    def _self_mute_button(self) -> Any | None:
        condition = self._automation.CreatePropertyCondition(
            self._UIA_CONTROL_TYPE_PROPERTY_ID,
            self._variant(self._UIA_BUTTON_CONTROL_TYPE_ID),
        )
        for hwnd in self._discord_windows():
            try:
                root = self._automation.ElementFromHandle(hwnd)
                buttons = root.FindAll(self._TREE_SCOPE_DESCENDANTS, condition)
            except Exception:
                continue
            for index in range(buttons.Length):
                try:
                    element = buttons.GetElement(index)
                    name = str(element.CurrentName or "").strip().casefold()
                    class_name = str(element.CurrentClassName or "").casefold()
                except Exception:
                    continue
                if name in {"mute", "unmute"} and "audiobuttonwithmenu" in class_name:
                    return element
        return None

    def muted_state(self) -> bool | None:
        button = self._self_mute_button()
        if button is None:
            return None
        name = str(button.CurrentName or "").strip().casefold()
        if name == "unmute":
            return True
        if name == "mute":
            return False
        return None

    def set_muted(self, muted: bool) -> bool:
        desired = bool(muted)
        for action_name in ("toggle", "legacy", "invoke"):
            button = self._self_mute_button()
            if button is None:
                return False
            name = str(button.CurrentName or "").strip().casefold()
            current = True if name == "unmute" else False if name == "mute" else None
            if current is None:
                return False
            if current == desired:
                return True
            try:
                if action_name == "toggle":
                    if not bool(button.GetCurrentPropertyValue(30041)):
                        continue
                    pattern = button.GetCurrentPattern(self._UIA_TOGGLE_PATTERN_ID)
                    toggle = pattern.QueryInterface(
                        self._uia_module.IUIAutomationTogglePattern
                    )
                    toggle.Toggle()
                elif action_name == "legacy":
                    if not bool(button.GetCurrentPropertyValue(30090)):
                        continue
                    pattern = button.GetCurrentPattern(10018)
                    legacy = pattern.QueryInterface(
                        self._uia_module.IUIAutomationLegacyIAccessiblePattern
                    )
                    legacy.DoDefaultAction()
                else:
                    if not bool(button.GetCurrentPropertyValue(30031)):
                        continue
                    pattern = button.GetCurrentPattern(self._UIA_INVOKE_PATTERN_ID)
                    invoke = pattern.QueryInterface(
                        self._uia_module.IUIAutomationInvokePattern
                    )
                    invoke.Invoke()
            except Exception:
                continue
            for _attempt in range(6):
                if self._settle_seconds:
                    time.sleep(self._settle_seconds)
                if self.muted_state() == desired:
                    return True
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._automation = None
        self._comtypes.CoUninitialize()


@dataclass
class _Command:
    name: str
    done: threading.Event
    receipt: DiscordMuteReceipt | None = None


class DiscordMuteCoordinator:
    """Serialize Discord mute acquisition/restoration on a COM worker thread."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        event_sink: EventSink | None = None,
        backend_factory: Callable[[], DiscordMuteBackend] = WindowsDiscordCaptureSessions,
        monitor_interval_seconds: float = 0.25,
    ) -> None:
        self.enabled = bool(enabled)
        self._event_sink = event_sink
        self._backend_factory = backend_factory
        self._monitor_interval_seconds = max(
            0.05, float(monitor_interval_seconds)
        )
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker,
            name="GlitchScribeDiscordMute",
            daemon=True,
        )
        self._thread.start()

    def _emit(self, event: str, **fields: Any) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink(event, **fields)
        except Exception:
            pass

    def _request(self, name: str, *, wait_timeout: float) -> DiscordMuteReceipt:
        command = _Command(name=name, done=threading.Event())
        self._commands.put(command)
        if not command.done.wait(max(0.0, float(wait_timeout))):
            return DiscordMuteReceipt(name, "queued")
        return command.receipt or DiscordMuteReceipt(name, "no_receipt")

    def begin_recording(self, *, wait_timeout: float = 0.75) -> DiscordMuteReceipt:
        if not self.enabled:
            receipt = DiscordMuteReceipt("begin", "disabled")
            self._emit("discord_mute_skipped", operation="begin", reason="disabled")
            return receipt
        return self._request("begin", wait_timeout=wait_timeout)

    def end_recording(self, *, wait_timeout: float = 0.75) -> DiscordMuteReceipt:
        return self._request("end", wait_timeout=wait_timeout)

    def close(self, *, wait_timeout: float = 1.5) -> DiscordMuteReceipt:
        receipt = self._request("close", wait_timeout=wait_timeout)
        self._thread.join(timeout=max(0.0, float(wait_timeout)))
        return receipt

    def _worker(self) -> None:
        backend: DiscordMuteBackend | None = None
        session_active = False
        changed_by_us = False
        owns_sessions = False
        try:
            try:
                backend = self._backend_factory()
                self._emit(
                    "discord_mute_backend_ready",
                    backend="windows_capture_session",
                )
            except Exception as exc:
                self._emit(
                    "discord_mute_backend_unavailable",
                    error_type=type(exc).__name__,
                )
            while True:
                try:
                    command = self._commands.get(
                        timeout=(
                            self._monitor_interval_seconds if session_active else None
                        )
                    )
                except queue.Empty:
                    if backend is None or not session_active:
                        continue
                    try:
                        reconcile = getattr(backend, "reconcile_muted", None)
                        if callable(reconcile):
                            session_count, newly_owned, verified = reconcile()
                        else:
                            verified = bool(backend.set_muted(True))
                            session_count = 1 if verified else 0
                            newly_owned = 0
                        if session_count:
                            owns_sessions = True
                        if newly_owned:
                            self._emit(
                                "discord_isolation_new_session_muted",
                                session_count=int(session_count),
                                newly_owned_count=int(newly_owned),
                                verified=bool(verified),
                            )
                        if session_count and not verified:
                            self._emit(
                                "discord_isolation_failed",
                                operation="monitor",
                                reason="verification_failed",
                                session_count=int(session_count),
                            )
                    except Exception as exc:
                        self._emit(
                            "discord_isolation_failed",
                            operation="monitor",
                            error_type=type(exc).__name__,
                        )
                    continue
                try:
                    if command.name == "begin":
                        if session_active:
                            command.receipt = DiscordMuteReceipt(
                                "begin", "already_active", changed_by_us
                            )
                        elif backend is None:
                            session_active = True
                            changed_by_us = False
                            command.receipt = DiscordMuteReceipt(
                                "begin", "discord_unavailable"
                            )
                        else:
                            prior = backend.muted_state()
                            session_active = True
                            if prior is None:
                                changed_by_us = False
                                owns_sessions = False
                                command.receipt = DiscordMuteReceipt(
                                    "begin", "control_unavailable"
                                )
                            else:
                                reconcile = getattr(backend, "reconcile_muted", None)
                                if callable(reconcile):
                                    session_count, _newly_owned, verified = reconcile()
                                    owns_sessions = bool(session_count)
                                else:
                                    verified = bool(backend.set_muted(True))
                                    owns_sessions = verified
                                changed_by_us = bool(not prior and verified)
                                command.receipt = DiscordMuteReceipt(
                                    "begin",
                                    (
                                        "already_muted"
                                        if prior and verified
                                        else "muted"
                                        if verified
                                        else "mute_failed"
                                    ),
                                    changed=changed_by_us,
                                    prior_muted=prior,
                                    current_muted=(True if verified else None),
                                )
                        self._emit(
                            "discord_mute_begin_completed",
                            status=command.receipt.status,
                            changed=command.receipt.changed,
                            prior_muted=command.receipt.prior_muted,
                            current_muted=command.receipt.current_muted,
                        )
                    elif command.name in {"end", "close"}:
                        prior_changed = changed_by_us
                        prior_owned = owns_sessions
                        current: bool | None = None
                        restored = False
                        if backend is not None and owns_sessions:
                            current = backend.muted_state()
                            restored = bool(backend.set_muted(False))
                            if restored:
                                current = backend.muted_state()
                        status = (
                            "restored"
                            if prior_owned and restored
                            else "restore_failed"
                            if prior_owned
                            else "unchanged"
                        )
                        session_active = False
                        changed_by_us = False
                        owns_sessions = bool(prior_owned and not restored)
                        command.receipt = DiscordMuteReceipt(
                            command.name,
                            status,
                            changed=bool(prior_changed and restored),
                            prior_muted=current,
                            current_muted=current,
                        )
                        self._emit(
                            "discord_mute_end_completed",
                            operation=command.name,
                            status=status,
                            changed_by_integration=prior_changed,
                            owned_sessions=prior_owned,
                            restored=restored,
                            observed_muted=current,
                        )
                        if command.name == "close":
                            return
                    else:
                        command.receipt = DiscordMuteReceipt(
                            command.name, "unknown_command"
                        )
                except Exception as exc:
                    command.receipt = DiscordMuteReceipt(
                        command.name,
                        "error",
                        error_type=type(exc).__name__,
                    )
                    self._emit(
                        "discord_mute_operation_failed",
                        operation=command.name,
                        error_type=type(exc).__name__,
                    )
                finally:
                    command.done.set()
        finally:
            if backend is not None:
                try:
                    if owns_sessions:
                        backend.set_muted(False)
                except Exception:
                    pass
                try:
                    backend.close()
                except Exception:
                    pass
