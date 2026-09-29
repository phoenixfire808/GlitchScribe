"""Small Win32 geometry helpers shared by anchored Tk surfaces."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Optional


class _MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


def monitor_work_area_for_point(
    x: int,
    y: int,
) -> Optional[tuple[int, int, int, int]]:
    """Return the taskbar-excluded work area containing a screen point."""
    if os.name != "nt":
        return None
    try:
        user32 = ctypes.windll.user32
        point = wintypes.POINT(int(x), int(y))
        monitor = user32.MonitorFromPoint(point, 2)  # MONITOR_DEFAULTTONEAREST
        if not monitor:
            return None
        info = _MonitorInfo()
        info.cbSize = ctypes.sizeof(_MonitorInfo)
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return None
        rect = info.rcWork
        left, top, right, bottom = (
            int(rect.left),
            int(rect.top),
            int(rect.right),
            int(rect.bottom),
        )
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom
    except (AttributeError, OSError, TypeError, ValueError):
        return None
