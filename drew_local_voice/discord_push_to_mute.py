"""Hold Discord's configured F7 push-to-mute key during Speak Now.

This uses Windows SendInput without focusing Discord. A detached watchdog releases
F7 if GlitchScribe exits unexpectedly, preventing a stranded synthetic key-down.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import uuid
from ctypes import wintypes
from pathlib import Path
from typing import Callable

_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008
_MAPVK_VK_TO_VSC = 0
_VK_F7 = 0x76

_ULONG_PTR = wintypes.WPARAM


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]


def _send_f7(*, key_up: bool) -> bool:
    user32 = ctypes.windll.user32
    scan_code = int(user32.MapVirtualKeyW(_VK_F7, _MAPVK_VK_TO_VSC))
    if scan_code <= 0:
        return False
    flags = _KEYEVENTF_SCANCODE | (_KEYEVENTF_KEYUP if key_up else 0)
    payload = _INPUT(
        type=_INPUT_KEYBOARD,
        union=_INPUTUNION(
            ki=_KEYBDINPUT(
                wVk=0,
                wScan=scan_code,
                dwFlags=flags,
                time=0,
                dwExtraInfo=0,
            )
        ),
    )
    return int(user32.SendInput(1, ctypes.byref(payload), ctypes.sizeof(_INPUT))) == 1


class DiscordPushToMuteKey:
    """Idempotent, observable F7 key-down/key-up ownership."""

    def __init__(self, *, event_sink: Callable[..., None] | None = None) -> None:
        self._event_sink = event_sink
        self._held = False
        self._watchdog_lock: Path | None = None
        self._watchdog: subprocess.Popen | None = None
        atexit.register(self.release)

    @property
    def held(self) -> bool:
        return self._held

    def _emit(self, event: str, **fields: object) -> None:
        if self._event_sink is not None:
            self._event_sink(event, pid=os.getpid(), **fields)

    def _start_watchdog(self) -> Path:
        lock = Path(tempfile.gettempdir()) / (
            f"glitchscribe-f7-{os.getpid()}-{uuid.uuid4().hex}.lock"
        )
        lock.write_text(
            json.dumps({"owner_pid": os.getpid(), "key": "F7"}),
            encoding="utf-8",
        )
        script = Path(__file__).with_name("f7_release_watchdog.py")
        executable = Path(sys.executable)
        pythonw = executable.with_name("pythonw.exe")
        if pythonw.exists():
            executable = pythonw
        flags = (
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )
        self._watchdog = subprocess.Popen(
            [str(executable), str(script), str(os.getpid()), str(lock)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=flags,
        )
        return lock

    def hold(self) -> bool:
        if self._held:
            return True
        try:
            lock = self._start_watchdog()
        except Exception as exc:
            self._emit("discord_push_to_mute_failed", operation="watchdog", error_type=type(exc).__name__)
            return False
        if not _send_f7(key_up=False):
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                pass
            self._emit("discord_push_to_mute_failed", operation="key_down")
            return False
        self._watchdog_lock = lock
        self._held = True
        self._emit("discord_push_to_mute_held", key="F7")
        return True

    def release(self) -> bool:
        if not self._held and self._watchdog_lock is None:
            return True
        released = _send_f7(key_up=True)
        lock = self._watchdog_lock
        self._watchdog_lock = None
        self._held = False
        if lock is not None:
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                pass
        self._emit(
            "discord_push_to_mute_released" if released else "discord_push_to_mute_failed",
            key="F7",
            operation="key_up",
        )
        return released
