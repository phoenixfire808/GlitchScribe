"""Detached failsafe that releases an injected F7 if GlitchScribe dies."""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path

_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008
_MAPVK_VK_TO_VSC = 0
_VK_F7 = 0x76
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259

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


def _owner_alive(pid: int) -> bool:
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return int(exit_code.value) == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _release_f7() -> None:
    user32 = ctypes.windll.user32
    scan_code = int(user32.MapVirtualKeyW(_VK_F7, _MAPVK_VK_TO_VSC))
    if scan_code <= 0:
        return
    payload = _INPUT(
        type=_INPUT_KEYBOARD,
        union=_INPUTUNION(
            ki=_KEYBDINPUT(
                wVk=0,
                wScan=scan_code,
                dwFlags=_KEYEVENTF_SCANCODE | _KEYEVENTF_KEYUP,
                time=0,
                dwExtraInfo=0,
            )
        ),
    )
    user32.SendInput(1, ctypes.byref(payload), ctypes.sizeof(_INPUT))


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    owner_pid = int(sys.argv[1])
    lock = Path(sys.argv[2])
    while lock.exists():
        if not _owner_alive(owner_pid):
            _release_f7()
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                pass
            return 0
        time.sleep(0.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
