from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable, Optional

if os.name == "nt":
    import win32api
    import win32con
    import win32gui


class SystemTrayIcon:
    """Native Windows notification icon with Tk-safe callbacks."""

    CALLBACK_MESSAGE = (win32con.WM_USER + 27) if os.name == "nt" else 0
    EXTERNAL_OPEN_SETTINGS_MESSAGE = (
        (win32con.WM_APP + 27) if os.name == "nt" else 0
    )
    OPEN_SETTINGS = 1023
    TOGGLE_FLYOUT = 1024
    TOGGLE_DICTATION = 1025

    def __init__(
        self,
        root,
        icon_path: Path,
        *,
        on_toggle_dictation: Callable[[int], None],
        on_open_settings: Callable[[], None],
        on_toggle_flyout: Callable[[], None],
        post_to_ui: Optional[Callable[[Callable[[], None]], bool]] = None,
        tooltip: str = "GlitchScribe — starting",
    ) -> None:
        self.root = root
        self.icon_path = Path(icon_path)
        self.on_toggle_dictation = on_toggle_dictation
        self.on_open_settings = on_open_settings
        self.on_toggle_flyout = on_toggle_flyout
        self.post_to_ui = post_to_ui
        self.tooltip = str(tooltip)[:127]
        self.hwnd = 0
        self.hicon = 0
        self.error: Optional[str] = None
        self.started = threading.Event()
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._class_name = f"DrewLocalVoiceTray.{os.getpid()}"
        self._taskbar_created = 0

    def start(self) -> None:
        if os.name != "nt" or (self._thread and self._thread.is_alive()):
            return
        self._thread = threading.Thread(
            target=self._run,
            name="drew-local-voice-tray",
            daemon=False,
        )
        self._thread.start()
        self.started.wait(timeout=3.0)

    def _run(self) -> None:
        try:
            self._taskbar_created = win32gui.RegisterWindowMessage("TaskbarCreated")
            message_map = {
                win32con.WM_CLOSE: self._on_close,
                win32con.WM_DESTROY: self._on_destroy,
                self.CALLBACK_MESSAGE: self._on_notify,
                self.EXTERNAL_OPEN_SETTINGS_MESSAGE: self._on_external_open_settings,
                self._taskbar_created: self._on_taskbar_created,
            }
            window_class = win32gui.WNDCLASS()
            window_class.hInstance = win32api.GetModuleHandle(None)
            window_class.lpszClassName = self._class_name
            window_class.lpfnWndProc = message_map
            atom = win32gui.RegisterClass(window_class)
            self.hwnd = win32gui.CreateWindow(
                atom,
                self._class_name,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                window_class.hInstance,
                None,
            )
            self._load_icon()
            self._add_icon()
            self.started.set()
            win32gui.PumpMessages()
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.started.set()

    def _load_icon(self) -> None:
        if self.icon_path.exists():
            self.hicon = win32gui.LoadImage(
                0,
                str(self.icon_path),
                win32con.IMAGE_ICON,
                0,
                0,
                win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE,
            )
        if not self.hicon:
            self.hicon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)

    def _notification_data(self):
        return (
            self.hwnd,
            0,
            win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP,
            self.CALLBACK_MESSAGE,
            self.hicon,
            self.tooltip,
        )

    def _add_icon(self) -> None:
        win32gui.Shell_NotifyIcon(win32gui.NIM_ADD, self._notification_data())

    def update_tooltip(self, text: str) -> None:
        self.tooltip = str(text)[:127]
        if self.hwnd and not self._stopping.is_set():
            try:
                win32gui.Shell_NotifyIcon(win32gui.NIM_MODIFY, self._notification_data())
            except win32gui.error:
                pass

    def _post_to_tk(self, callback: Callable[[], None]) -> None:
        if self.post_to_ui is not None:
            self.post_to_ui(callback)
            return
        try:
            self.root.after(0, callback)
        except Exception:
            pass

    def _on_external_open_settings(self, hwnd, message, wparam, lparam):
        del hwnd, message, wparam, lparam
        self._post_to_tk(self.on_open_settings)
        return 0

    def _on_notify(self, hwnd, message, wparam, lparam):
        del hwnd, message, wparam
        if lparam == win32con.WM_LBUTTONDBLCLK:
            self._post_to_tk(self.on_open_settings)
        elif lparam == win32con.WM_RBUTTONUP:
            self._show_menu()
        return 0

    def _show_menu(self) -> None:
        menu = win32gui.CreatePopupMenu()
        try:
            previous_foreground = int(win32gui.GetForegroundWindow())
            win32gui.AppendMenu(menu, win32con.MF_STRING | win32con.MF_DISABLED, 0, self.tooltip)
            win32gui.AppendMenu(menu, win32con.MF_SEPARATOR, 0, "")
            win32gui.AppendMenu(menu, win32con.MF_STRING, self.TOGGLE_DICTATION, "Start / stop dictation")
            win32gui.AppendMenu(menu, win32con.MF_STRING, self.OPEN_SETTINGS, "Open settings")
            win32gui.AppendMenu(menu, win32con.MF_STRING, self.TOGGLE_FLYOUT, "Show / hide flyout")
            win32gui.AppendMenu(menu, win32con.MF_SEPARATOR, 0, "")
            win32gui.AppendMenu(menu, win32con.MF_STRING | win32con.MF_DISABLED, 0, "Ctrl + Alt + Space to dictate")
            position = win32gui.GetCursorPos()
            win32gui.SetForegroundWindow(self.hwnd)
            command = win32gui.TrackPopupMenu(
                menu,
                win32con.TPM_LEFTALIGN | win32con.TPM_BOTTOMALIGN | win32con.TPM_RETURNCMD,
                position[0],
                position[1],
                0,
                self.hwnd,
                None,
            )
            if command == self.TOGGLE_DICTATION:
                self._restore_foreground(previous_foreground)
                self._post_to_tk(
                    lambda: self.on_toggle_dictation(previous_foreground)
                )
            elif command == self.OPEN_SETTINGS:
                self._post_to_tk(self.on_open_settings)
            elif command == self.TOGGLE_FLYOUT:
                self._restore_foreground(previous_foreground)
                self._post_to_tk(self.on_toggle_flyout)
            else:
                self._restore_foreground(previous_foreground)
            win32gui.PostMessage(self.hwnd, win32con.WM_NULL, 0, 0)
        finally:
            win32gui.DestroyMenu(menu)

    @staticmethod
    def _restore_foreground(hwnd: int) -> None:
        if not hwnd:
            return
        try:
            win32gui.SetForegroundWindow(int(hwnd))
        except win32gui.error:
            pass

    def _on_taskbar_created(self, hwnd, message, wparam, lparam):
        del hwnd, message, wparam, lparam
        if not self._stopping.is_set():
            self._add_icon()
        return 0

    def _on_close(self, hwnd, message, wparam, lparam):
        del message, wparam, lparam
        win32gui.DestroyWindow(hwnd)
        return 0

    def _on_destroy(self, hwnd, message, wparam, lparam):
        del hwnd, message, wparam, lparam
        try:
            win32gui.Shell_NotifyIcon(win32gui.NIM_DELETE, self._notification_data())
        except win32gui.error:
            pass
        win32gui.PostQuitMessage(0)
        return 0

    def stop(self, timeout: float = 3.0) -> None:
        self._stopping.set()
        if self.hwnd:
            try:
                win32gui.PostMessage(self.hwnd, win32con.WM_CLOSE, 0, 0)
            except win32gui.error:
                pass
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._thread = None
