"""Physical Windows keyboard events for hold-to-talk shortcut sessions."""

from __future__ import annotations

import ctypes
import os
import threading
from ctypes import wintypes
from typing import Callable, Optional

from .insertion import SecondaryHotkeyTracker
from .shortcut_sessions import ShortcutAction, ShortcutSessionController
from .troubleshooting import TroubleshootingLog, default_log, record_troubleshooting


def _troubleshoot(event: str, **fields: object) -> None:
    """Emit a troubleshooting event without breaking the hotkey path."""
    log = default_log()
    if log is None:
        return
    pid = fields.get("pid")
    if pid is None:
        fields = {**fields, "pid": os.getpid()}
    log.emit(event, **fields)


def _troubleshoot_hook(
    event: str,
    *,
    log: Optional[TroubleshootingLog],
    **fields: object,
) -> None:
    """Emit on a caller-supplied log when available, otherwise the default."""
    pid = fields.get("pid")
    if pid is None:
        fields = {**fields, "pid": os.getpid()}
    if log is not None:
        log.emit(event, **fields)
        return
    _troubleshoot(event, **fields)


LLKHF_LOWER_IL_INJECTED = 0x02
LLKHF_INJECTED = 0x10
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_QUIT = 0x0012
VK_ESCAPE = 0x1B

_KEY_NAMES = {
    0x10: "SHIFT",
    0x11: "CTRL",
    0x12: "ALT",
    0x20: "SPACE",
    0x2D: "INSERT",
    0x5B: "WIN",
    0x5C: "WIN",
    0x77: "F8",
    0x78: "F9",
    0x79: "F10",
    0x7A: "F11",
    0x7B: "F12",
    0xA0: "SHIFT",
    0xA1: "SHIFT",
    0xA2: "CTRL",
    0xA3: "CTRL",
    0xA4: "ALT",
    0xA5: "ALT",
}

# A-Z / 0-9 for secondary chords such as paste-last (Ctrl+Shift+V).
for _code in range(0x41, 0x5B):
    _KEY_NAMES[_code] = chr(_code)
for _code in range(0x30, 0x3A):
    _KEY_NAMES[_code] = chr(_code)


class _KeyboardHookData(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class Win32KeyboardHookApi:
    def __init__(self) -> None:
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._hook_proc = None

    def current_thread_id(self) -> int:
        return int(self.kernel32.GetCurrentThreadId())

    def install(self, callback: Callable[[int, bool, int], ShortcutAction]) -> int:
        hook_proc_type = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )

        def hook_proc(code, message, data_address):
            if code >= 0 and message in {
                WM_KEYDOWN,
                WM_KEYUP,
                WM_SYSKEYDOWN,
                WM_SYSKEYUP,
            }:
                data = ctypes.cast(
                    data_address,
                    ctypes.POINTER(_KeyboardHookData),
                ).contents
                try:
                    action = callback(
                        int(data.vkCode),
                        message in {WM_KEYDOWN, WM_SYSKEYDOWN},
                        int(data.flags),
                    )
                    if action is ShortcutAction.CANCEL:
                        return 1
                except Exception:
                    pass
            return self.user32.CallNextHookEx(None, code, message, data_address)

        self._hook_proc = hook_proc_type(hook_proc)
        self.user32.SetWindowsHookExW.argtypes = (
            ctypes.c_int,
            hook_proc_type,
            wintypes.HINSTANCE,
            wintypes.DWORD,
        )
        self.user32.SetWindowsHookExW.restype = ctypes.c_void_p
        self.user32.CallNextHookEx.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        self.user32.CallNextHookEx.restype = ctypes.c_ssize_t
        self.kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
        self.kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        ctypes.set_last_error(0)
        handle = self.user32.SetWindowsHookExW(
            WH_KEYBOARD_LL,
            self._hook_proc,
            self.kernel32.GetModuleHandleW(None),
            0,
        )
        if not handle:
            raise OSError(ctypes.get_last_error(), "Could not install keyboard hook")
        return int(handle)

    def message_loop(self) -> None:
        message = wintypes.MSG()
        while self.user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            self.user32.TranslateMessage(ctypes.byref(message))
            self.user32.DispatchMessageW(ctypes.byref(message))

    def post_quit(self, thread_id: int) -> None:
        self.user32.PostThreadMessageW(int(thread_id), WM_QUIT, 0, 0)

    def uninstall(self, handle: int) -> None:
        self.user32.UnhookWindowsHookEx(ctypes.c_void_p(int(handle)))
        self._hook_proc = None


class PhysicalShortcutHook:
    def __init__(
        self,
        *,
        callback: Callable[[ShortcutAction, int], None],
        hold_key: str = "F8",
        latch_modifier: str = "SHIFT",
        foreground_reader: Callable[[], int],
        escape_callback: Callable[[], bool] | None = None,
        secondary_callback: Callable[[str, int], None] | None = None,
        paste_last_hotkey: str = "",
        hold_enabled: bool = True,
        api_factory: Callable[[], Win32KeyboardHookApi] = Win32KeyboardHookApi,
        troubleshooting_log: Optional[TroubleshootingLog] = None,
    ) -> None:
        self.callback = callback
        self.foreground_reader = foreground_reader
        self.escape_callback = escape_callback
        self.secondary_callback = secondary_callback
        self.hold_enabled = bool(hold_enabled)
        self.controller = ShortcutSessionController(
            hold_chord={hold_key},
            latch_chord={hold_key, latch_modifier},
        )
        self.paste_last_tracker = SecondaryHotkeyTracker(paste_last_hotkey or "")
        self.api_factory = api_factory
        self.troubleshooting_log = troubleshooting_log
        self.registered = threading.Event()
        self.error: str | None = None
        self.thread: threading.Thread | None = None
        self.thread_id = 0
        self.hook_handle: int | None = None
        self._api = None
        self._escape_down = False
        self._escape_consumed = False
        self._press_count = 0
        self._last_match_kind: str = ""
        self._hold_key = str(hold_key)
        self._latch_modifier = str(latch_modifier)

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.error = None
        self.registered.clear()
        self._press_count = 0
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="voice-hold-shortcut",
        )
        self.thread.start()
        self.registered.wait(timeout=2.0)
        if not self.registered.is_set():
            _troubleshoot_hook(
                "hotkey_hook_install_failed",
                log=self.troubleshooting_log,
                reason="register_timeout",
                wait_seconds=2.0,
            )
            self.error = "Hook installation timed out"
        elif self.error:
            _troubleshoot_hook(
                "hotkey_hook_install_failed",
                log=self.troubleshooting_log,
                reason=self.error,
            )

    def stop(self) -> None:
        api = self._api
        if api is not None and self.thread_id:
            api.post_quit(self.thread_id)
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        _troubleshoot_hook(
            "hotkey_hook_stopped",
            log=self.troubleshooting_log,
            press_count=self._press_count,
        )

    def _run(self) -> None:
        api = self.api_factory()
        self._api = api
        self.thread_id = int(api.current_thread_id())
        try:
            self.hook_handle = int(api.install(self.handle_key_event))
            self.registered.set()
            _troubleshoot_hook(
                "hotkey_hook_installed",
                log=self.troubleshooting_log,
                thread_id=self.thread_id,
            )
            api.message_loop()
        except OSError as exc:
            self.error = str(exc)
            self.registered.set()
        finally:
            if self.hook_handle is not None:
                api.uninstall(self.hook_handle)
                self.hook_handle = None

    def handle_key_event(
        self,
        virtual_key: int,
        is_down: bool,
        flags: int = 0,
    ) -> ShortcutAction:
        if flags & (LLKHF_INJECTED | LLKHF_LOWER_IL_INJECTED):
            return ShortcutAction.NONE
        if int(virtual_key) == VK_ESCAPE:
            if is_down:
                if self._escape_down:
                    return (
                        ShortcutAction.CANCEL
                        if self._escape_consumed
                        else ShortcutAction.NONE
                    )
                self._escape_down = True
                self._escape_consumed = bool(
                    self.escape_callback is not None and self.escape_callback()
                )
                return (
                    ShortcutAction.CANCEL
                    if self._escape_consumed
                    else ShortcutAction.NONE
                )
            consumed = self._escape_consumed
            self._escape_down = False
            self._escape_consumed = False
            return ShortcutAction.CANCEL if consumed else ShortcutAction.NONE
        key = _KEY_NAMES.get(int(virtual_key))
        if key is None and not self.hold_enabled and not self.paste_last_tracker.enabled:
            return ShortcutAction.NONE
        if (
            key is not None
            and self.secondary_callback is not None
            and self.paste_last_tracker.enabled
            and self.paste_last_tracker.handle(key, is_down=is_down)
        ):
            target_window = int(self.foreground_reader())
            try:
                self.secondary_callback("paste_last", target_window)
            except Exception as exc:
                _troubleshoot_hook(
                    "hotkey_callback_error",
                    log=self.troubleshooting_log,
                    action="paste_last",
                    error_type=type(exc).__name__,
                    error=str(exc)[:256],
                )
            return ShortcutAction.CANCEL
        if not self.hold_enabled:
            return ShortcutAction.NONE
        if key is None:
            return ShortcutAction.NONE
        # Record that the hook received a recognised key. This is the
        # one event that answers Drew's "did the press even reach the
        # app?" question without spamming the log on every typed character.
        if is_down and key in {self._hold_key, self._latch_modifier}:
            self._press_count += 1
            _troubleshoot_hook(
                "hotkey_pressed",
                log=self.troubleshooting_log,
                key=key,
                hold_key=self._hold_key,
                latch_modifier=self._latch_modifier,
                press_count=self._press_count,
            )
        action = self.controller.handle(
            key,
            is_down=is_down,
            injected=False,
        )
        if action is not ShortcutAction.NONE:
            self._last_match_kind = action.name
            target_window = (
                int(self.foreground_reader())
                if action in {ShortcutAction.START_HOLD, ShortcutAction.START_LATCHED}
                else 0
            )
            _troubleshoot_hook(
                "hotkey_match",
                log=self.troubleshooting_log,
                key=key,
                action=action.name,
                target_window=target_window,
            )
            try:
                self.callback(action, target_window)
            except Exception as exc:
                _troubleshoot_hook(
                    "hotkey_callback_error",
                    log=self.troubleshooting_log,
                    action=action.name,
                    error_type=type(exc).__name__,
                    error=str(exc)[:256],
                )
        return action

    def reset_session(self) -> None:
        self.controller.reset()
        _troubleshoot_hook(
            "hotkey_session_reset",
            log=self.troubleshooting_log,
        )
