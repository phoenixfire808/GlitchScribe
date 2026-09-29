from __future__ import annotations

import os
import subprocess
import time
from typing import Callable

import win32con
import win32gui

TASK_NAME = "Drew Local Voice Always Running"
TRAY_CLASS_PREFIX = "DrewLocalVoiceTray."
EXTERNAL_OPEN_SETTINGS_MESSAGE = win32con.WM_APP + 27
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def find_tray_window(
    *,
    enum_windows: Callable = win32gui.EnumWindows,
    get_class_name: Callable[[int], str] = win32gui.GetClassName,
) -> int:
    matches: list[int] = []

    def inspect(hwnd: int, _extra: object) -> bool:
        try:
            if get_class_name(hwnd).startswith(TRAY_CLASS_PREFIX):
                matches.append(int(hwnd))
        except win32gui.error:
            pass
        return True

    enum_windows(inspect, None)
    return matches[0] if matches else 0


def request_open_settings(
    *,
    find_window: Callable[[], int] = find_tray_window,
    post_message: Callable = win32gui.PostMessage,
) -> bool:
    hwnd = int(find_window() or 0)
    if not hwnd:
        return False
    try:
        post_message(hwnd, EXTERNAL_OPEN_SETTINGS_MESSAGE, 0, 0)
    except win32gui.error:
        return False
    return True


def start_managed_task(*, runner: Callable = subprocess.run) -> bool:
    result = runner(
        ["schtasks.exe", "/Run", "/TN", TASK_NAME],
        capture_output=True,
        creationflags=CREATE_NO_WINDOW,
        timeout=20,
        check=False,
    )
    return int(result.returncode) == 0


def main() -> int:
    if request_open_settings():
        return 0
    if not start_managed_task():
        return 1
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if request_open_settings():
            return 0
        time.sleep(0.25)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
