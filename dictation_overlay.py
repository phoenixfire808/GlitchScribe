from __future__ import annotations

import ctypes
import os
import tkinter as tk
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, Optional

from drew_local_voice.theme import ThemePalette, build_palette


def overlay_position(
    screen_width: int,
    screen_height: int,
    width: int,
    height: int,
    margin: int = 48,
) -> tuple[int, int]:
    x = max(margin, screen_width - width - margin)
    y = max(margin, screen_height - height - margin)
    return x, y


def overlay_position_in_work_area(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
    margin: int = 48,
) -> tuple[int, int]:
    x = max(left + margin, right - width - margin)
    y = max(top + margin, bottom - height - margin)
    return x, y


def clamp_overlay_position(
    x: int,
    y: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
    margin: int = 8,
) -> tuple[int, int]:
    max_x = max(left + margin, right - width - margin)
    max_y = max(top + margin, bottom - height - margin)
    return (
        min(max(int(x), left + margin), max_x),
        min(max(int(y), top + margin), max_y),
    )


def caret_overlay_position(
    caret_x: int,
    caret_y: int,
    work_left: int,
    work_top: int,
    work_right: int,
    work_bottom: int,
    overlay_width: int,
    overlay_height: int,
    *,
    gap: int = 16,
) -> tuple[int, int]:
    """Place a preview near a caret without covering the current text line."""
    x = int(caret_x) - 40
    y = int(caret_y) - int(overlay_height) - int(gap)
    if y < int(work_top) + 8:
        y = int(caret_y) + int(gap)
    return clamp_overlay_position(
        x,
        y,
        work_left,
        work_top,
        work_right,
        work_bottom,
        overlay_width,
        overlay_height,
    )


def dragged_overlay_position(
    origin_x: int,
    origin_y: int,
    pointer_start_x: int,
    pointer_start_y: int,
    pointer_current_x: int,
    pointer_current_y: int,
) -> tuple[int, int]:
    return (
        int(origin_x) + int(pointer_current_x) - int(pointer_start_x),
        int(origin_y) + int(pointer_current_y) - int(pointer_start_y),
    )


def anchored_bubble_position(
    anchor_x: int,
    anchor_y: int,
    work_left: int,
    work_top: int,
    work_right: int,
    work_bottom: int,
    bubble_width: int,
    bubble_height: int,
    *,
    offset_x: int = 18,
    offset_y: int = -88,
) -> tuple[int, int]:
    """Place a bubble at a persisted offset from one session anchor."""
    return clamp_overlay_position(
        int(anchor_x) + int(offset_x),
        int(anchor_y) + int(offset_y),
        work_left,
        work_top,
        work_right,
        work_bottom,
        bubble_width,
        bubble_height,
    )


def bubble_offset_for_position(
    position_x: int,
    position_y: int,
    *,
    anchor_x: int,
    anchor_y: int,
) -> tuple[int, int]:
    return int(position_x) - int(anchor_x), int(position_y) - int(anchor_y)


def clamp_overlay_size(
    width: int,
    height: int,
    *,
    minimum_width: int = 260,
    minimum_height: int = 92,
    maximum_width: int = 720,
    maximum_height: int = 240,
) -> tuple[int, int]:
    return (
        min(max(int(width), int(minimum_width)), int(maximum_width)),
        min(max(int(height), int(minimum_height)), int(maximum_height)),
    )


def resized_overlay_size(
    origin_width: int,
    origin_height: int,
    pointer_start_x: int,
    pointer_start_y: int,
    pointer_current_x: int,
    pointer_current_y: int,
) -> tuple[int, int]:
    return clamp_overlay_size(
        int(origin_width) + int(pointer_current_x) - int(pointer_start_x),
        int(origin_height) + int(pointer_current_y) - int(pointer_start_y),
    )


def compact_overlay_text(text: str, max_chars: int = 26) -> str:
    value = text.strip() or "Listening…"
    limit = max(1, int(max_chars))
    if len(value) <= limit:
        return value
    return "…" + value[-(limit - 1) :] if limit > 1 else "…"


BUBBLE_FONT_FAMILIES = (
    "Segoe UI",
    "Cascadia Code",
    "Consolas",
    "Bahnschrift",
    "Arial",
)
BUBBLE_FONT_WEIGHTS = ("normal", "semibold", "bold")
BUBBLE_VISUAL_STYLES = ("minimal", "glitch_glass", "neon_edge", "high_contrast")
BUBBLE_TEXT_COLOR_MODES = ("theme", "accent", "bright")


@dataclass(frozen=True)
class BubbleVisualStyle:
    background: str
    border: str
    text: str
    state: str
    border_width: int


def bubble_font_spec(family: str, size: int, weight: str = "normal") -> tuple:
    selected_family = str(family or "").strip()
    if selected_family not in BUBBLE_FONT_FAMILIES:
        selected_family = "Segoe UI"
    selected_weight = str(weight or "").strip().casefold()
    if selected_weight not in BUBBLE_FONT_WEIGHTS:
        selected_weight = "normal"
    selected_size = min(max(int(size), 6), 48)
    if selected_weight == "semibold":
        if selected_family == "Segoe UI":
            return ("Segoe UI Semibold", selected_size)
        return (selected_family, selected_size, "bold")
    if selected_weight == "bold":
        return (selected_family, selected_size, "bold")
    return (selected_family, selected_size)


def resolve_bubble_visual_style(
    palette: ThemePalette,
    visual_style: str,
    text_color_mode: str,
    *,
    border_width: int = 1,
) -> BubbleVisualStyle:
    style = str(visual_style or "").strip().casefold()
    if style not in BUBBLE_VISUAL_STYLES:
        style = "glitch_glass"
    color_mode = str(text_color_mode or "").strip().casefold()
    if color_mode not in BUBBLE_TEXT_COLOR_MODES:
        color_mode = "theme"
    width = min(max(int(border_width), 0), 4)
    text = {
        "theme": palette.text,
        "accent": palette.accent_soft,
        "bright": "#FFFFFF" if palette.mode == "dark" else "#000000",
    }[color_mode]
    if style == "minimal":
        return BubbleVisualStyle(
            palette.surface, palette.border, text, palette.muted, min(width, 1)
        )
    if style == "neon_edge":
        return BubbleVisualStyle(
            "#080B12" if palette.mode == "dark" else palette.surface,
            palette.accent,
            text,
            palette.accent,
            max(2, width),
        )
    if style == "high_contrast":
        return BubbleVisualStyle(
            "#000000", "#FFFFFF", "#FFFFFF", palette.accent_soft, max(2, width)
        )
    return BubbleVisualStyle(
        palette.bubble, palette.accent, text, palette.accent_soft, width
    )


def show_window_no_activate(
    hwnd: int,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    user32=None,
) -> bool:
    """Show a topmost indicator without taking focus from the dictation target."""
    native_user32 = user32 is None
    if native_user32:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow.restype = wintypes.BOOL
        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL
    SW_SHOWNOACTIVATE = 4
    SWP_NOACTIVATE = 0x0010
    SWP_SHOWWINDOW = 0x0040
    HWND_TOPMOST = wintypes.HWND(-1) if native_user32 else -1
    user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
    return bool(
        user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            x,
            y,
            width,
            height,
            SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )
    )


class _MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


class DictationOverlay:
    WIDTH = 300
    HEIGHT = 92
    OPACITY = 0.97
    BRAND_NAME = "GLITCH FACTOR"
    BRAND_SUBTITLE = "GLITCHSCRIBE // PRIVATE AI"
    # Pulse animation (purely visual) — periods in ms.
    PULSE_TICK_MS = 700
    PULSE_FADE_MS = 220

    def __init__(
        self,
        root: tk.Tk,
        on_toggle: Callable[[], None],
        on_settings: Optional[Callable[[], None]] = None,
        *,
        position: Optional[tuple[int, int]] = None,
        on_position_changed: Optional[Callable[[int, int], None]] = None,
        size: Optional[tuple[int, int]] = None,
        on_size_changed: Optional[Callable[[int, int], None]] = None,
        on_mouse_event: Optional[Callable[[str], None]] = None,
        palette: Optional[ThemePalette] = None,
    ):
        self.root = root
        self.on_toggle = on_toggle
        self.on_settings = on_settings
        self.position = position
        self.on_position_changed = on_position_changed
        self.width, self.height = clamp_overlay_size(*(size or (self.WIDTH, self.HEIGHT)))
        self.on_size_changed = on_size_changed
        self.on_mouse_event = on_mouse_event
        self.palette = palette or build_palette("dark", "violet")
        self._drag_origin: Optional[tuple[int, int, int, int]] = None
        self._resize_origin: Optional[tuple[int, int, int, int]] = None
        self._recording = False
        # Pulse animation bookkeeping. We never start the cycle during __init__
        # so headless tests don't schedule Tk timers; callers (or update())
        # flip the cycle on.
        self._pulse_after_id: Optional[str] = None
        self._pulse_phase: int = 0
        self._pulse_colors: tuple[str, str] = (
            self.palette.success,
            self.palette.accent_soft,
        )
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.configure(bg=self.palette.window)
        self.window.attributes("-topmost", True)
        try:
            self.window.attributes("-alpha", self.OPACITY)
        except tk.TclError:
            pass

        self.shell = tk.Frame(
            self.window,
            bg=self.palette.surface,
            highlightbackground=self.palette.accent,
            highlightthickness=1,
            padx=10,
            pady=7,
            cursor="fleur",
            width=self.width,
            height=self.height,
        )
        self.shell.pack(fill="both", expand=True)
        self.shell.pack_propagate(False)

        self.brand = tk.Frame(
            self.shell,
            bg=self.palette.surface_alt,
            cursor="fleur",
            padx=6,
            pady=3,
        )
        self.brand.pack(fill="x", pady=(0, 6))
        self.brand_badge = tk.Label(
            self.brand,
            text="GF",
            bg=self.palette.accent,
            fg=self.palette.on_accent,
            font=("Segoe UI Black", 8),
            padx=5,
            pady=0,
            cursor="fleur",
        )
        self.brand_badge.pack(side="left", padx=(0, 7))
        self.brand_title = tk.Label(
            self.brand,
            text=self.BRAND_NAME,
            bg=self.palette.surface_alt,
            fg=self.palette.text,
            font=("Segoe UI Semibold", 9),
            cursor="fleur",
        )
        self.brand_title.pack(side="left")
        self.brand_subtitle = tk.Label(
            self.brand,
            text=self.BRAND_SUBTITLE,
            bg=self.palette.surface_alt,
            fg=self.palette.accent_soft,
            font=("Segoe UI", 7),
            cursor="fleur",
        )
        self.brand_subtitle.pack(side="right")

        self.top = tk.Frame(self.shell, bg=self.palette.surface, cursor="fleur")
        self.top.pack(fill="x")
        self.status_dot = tk.Label(
            self.top,
            text="●",
            bg=self.palette.surface,
            fg=self.palette.success,
            font=("Segoe UI", 11),
            cursor="fleur",
        )
        self.status_dot.pack(side="left", padx=(0, 6))
        self.state_var = tk.StringVar(value="READY")
        self.state_label = tk.Label(
            self.top,
            textvariable=self.state_var,
            bg=self.palette.surface,
            fg=self.palette.accent_soft,
            font=("Segoe UI Semibold", 10),
            cursor="fleur",
        )
        self.state_label.pack(side="left")

        self.hide_control = self._control_label("×", self.hide, "Hide indicator")
        self.hide_control.pack(side="right", padx=(5, 0))
        self.settings_control = self._control_label(
            "⚙", self._open_settings, "Open settings"
        )
        self.settings_control.pack(side="right", padx=(5, 0))
        self.action_control = self._control_label(
            "●", self._toggle, "Start or stop dictation"
        )
        self.action_control.pack(side="right", padx=(7, 0))

        self.text_var = tk.StringVar(value="Available")
        self.text_label = tk.Label(
            self.top,
            textvariable=self.text_var,
            bg=self.palette.surface,
            fg=self.palette.text,
            font=("Segoe UI", 10),
            anchor="e",
            cursor="fleur",
        )
        self.text_label.pack(side="right", fill="x", expand=True, padx=(8, 0))

        self.bottom = tk.Frame(self.shell, bg=self.palette.surface)
        self.bottom.pack(fill="x", pady=(8, 0), side="bottom")
        self.meter = tk.Canvas(
            self.bottom,
            height=4,
            bg=self.palette.meter_trough,
            highlightthickness=0,
            bd=0,
        )
        self.meter.pack(side="left", fill="x", expand=True)
        self._meter_fill = self.meter.create_rectangle(
            0, 0, 0, 4, fill=self.palette.accent, outline=""
        )
        self.resize_grip = tk.Label(
            self.bottom,
            text="◢",
            bg=self.palette.surface,
            fg=self.palette.muted,
            font=("Segoe UI Symbol", 8),
            cursor="size_nw_se",
            padx=1,
            pady=0,
        )
        self.resize_grip.pack(side="right", padx=(6, 0))
        self.resize_grip.bind("<ButtonPress-1>", self._resize_start)
        self.resize_grip.bind("<B1-Motion>", self._resize_motion)
        self.resize_grip.bind("<ButtonRelease-1>", self._resize_end)
        self.apply_palette(self.palette)
        self.window.update_idletasks()
        self._apply_no_activate_style()
        for drag_handle in (
            self.shell,
            self.brand,
            self.brand_badge,
            self.brand_title,
            self.brand_subtitle,
            self.top,
            self.status_dot,
            self.state_label,
            self.text_label,
        ):
            drag_handle.bind("<ButtonPress-1>", self._drag_start)
            drag_handle.bind("<B1-Motion>", self._drag_motion)
            drag_handle.bind("<ButtonRelease-1>", self._drag_end)

    def _notify_mouse_event(self, event_name: str) -> None:
        callback = self.on_mouse_event
        if callback is None:
            return
        try:
            callback(str(event_name))
        except Exception:
            # Pointer diagnostics must never break overlay interaction.
            pass

    def _begin_pointer_capture(self, event_name: str) -> None:
        self._notify_mouse_event(event_name + "_started")
        try:
            self.window.grab_set()
        except tk.TclError:
            self._notify_mouse_event(event_name + "_capture_failed")
        else:
            self._notify_mouse_event(event_name + "_capture_started")

    def _end_pointer_capture(self, event_name: str) -> None:
        try:
            self.window.grab_release()
        except tk.TclError:
            pass
        self._notify_mouse_event(event_name + "_ended")

    def _activate_control(
        self,
        callback: Callable[[], None],
        accessible_name: str,
    ) -> None:
        self._notify_mouse_event("control_click:" + accessible_name)
        callback()

    def _control_label(
        self, text: str, callback: Callable[[], None], accessible_name: str
    ) -> tk.Label:
        label = tk.Label(
            self.top,
            text=text,
            bg=self.palette.surface,
            fg=self.palette.muted,
            activebackground=self.palette.surface_alt,
            activeforeground=self.palette.text,
            font=("Segoe UI Symbol", 10),
            cursor="hand2",
            padx=2,
        )
        label.bind(
            "<Button-1>",
            lambda _event: self._activate_control(callback, accessible_name),
        )
        label.bind(
            "<Enter>",
            lambda _event: label.configure(bg=self.palette.surface_alt),
        )
        label.bind(
            "<Leave>",
            lambda _event: label.configure(bg=self.palette.surface),
        )
        label.accessible_name = accessible_name
        return label

    def apply_palette(self, palette: ThemePalette) -> None:
        self.palette = palette
        self.window.configure(bg=palette.window)
        self.shell.configure(
            bg=palette.surface,
            highlightbackground=palette.accent,
        )
        self.brand.configure(bg=palette.surface_alt)
        self.brand_badge.configure(bg=palette.accent, fg=palette.on_accent)
        self.brand_title.configure(bg=palette.surface_alt, fg=palette.text)
        self.brand_subtitle.configure(
            bg=palette.surface_alt,
            fg=palette.accent_soft,
        )
        self.top.configure(bg=palette.surface)
        self.bottom.configure(bg=palette.surface)
        self.status_dot.configure(bg=palette.surface)
        self.state_label.configure(bg=palette.surface, fg=palette.accent_soft)
        self.text_label.configure(bg=palette.surface, fg=palette.text)
        self.resize_grip.configure(bg=palette.surface, fg=palette.muted)
        self.meter.configure(bg=palette.meter_trough)
        self.meter.itemconfigure(self._meter_fill, fill=palette.accent)
        for control in (
            self.hide_control,
            self.settings_control,
            self.action_control,
        ):
            control.configure(
                bg=palette.surface,
                fg=palette.muted,
                activebackground=palette.surface_alt,
                activeforeground=palette.text,
            )

    def _toggle(self) -> None:
        self.on_toggle()

    def _open_settings(self) -> None:
        if self.on_settings is not None:
            self.on_settings()

    def _drag_start(self, event) -> None:
        self._drag_origin = (
            int(self.window.winfo_x()),
            int(self.window.winfo_y()),
            int(event.x_root),
            int(event.y_root),
        )
        self._begin_pointer_capture("drag")

    def _drag_motion(self, event) -> None:
        if self._drag_origin is None:
            return
        x, y = dragged_overlay_position(
            *self._drag_origin, int(event.x_root), int(event.y_root)
        )
        self.window.geometry(f"{self.width}x{self.height}+{x}+{y}")

    def _drag_end(self, event) -> None:
        if self._drag_origin is None:
            return
        x, y = dragged_overlay_position(
            *self._drag_origin, int(event.x_root), int(event.y_root)
        )
        x, y = clamp_overlay_position(
            x,
            y,
            0,
            0,
            self.window.winfo_screenwidth(),
            self.window.winfo_screenheight(),
            self.width,
            self.height,
        )
        self.position = (x, y)
        self.window.geometry(f"{self.width}x{self.height}+{x}+{y}")
        self._drag_origin = None
        self._end_pointer_capture("drag")
        if self.on_position_changed is not None:
            self.on_position_changed(x, y)

    def _resize_start(self, event) -> None:
        self._resize_origin = (
            self.width,
            self.height,
            int(event.x_root),
            int(event.y_root),
        )
        self._begin_pointer_capture("resize")

    def _resize_motion(self, event) -> None:
        if self._resize_origin is None:
            return
        width, height = resized_overlay_size(
            *self._resize_origin,
            int(event.x_root),
            int(event.y_root),
        )
        self.width, self.height = width, height
        self.shell.configure(width=width, height=height)
        x, y = int(self.window.winfo_x()), int(self.window.winfo_y())
        self.window.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")
        if os.name == "nt":
            show_window_no_activate(self._native_hwnd(), x, y, width, height)

    def _resize_end(self, event) -> None:
        if self._resize_origin is None:
            return
        self._resize_motion(event)
        self._resize_origin = None
        self._end_pointer_capture("resize")
        if self.on_size_changed is not None:
            self.on_size_changed(self.width, self.height)

    def _native_hwnd(self) -> int:
        hwnd = int(self.window.winfo_id())
        if os.name == "nt":
            parent = int(ctypes.windll.user32.GetParent(hwnd))
            if parent:
                hwnd = parent
        return hwnd

    def _apply_no_activate_style(self) -> None:
        if os.name != "nt":
            return
        # Defensive: test doubles in some modules (e.g. dictation_features)
        # construct ``DictationOverlay`` subclasses without a Tk ``window``;
        # we silently no-op in that case rather than crashing.
        if not hasattr(self, "window"):
            return
        try:
            hwnd = self._native_hwnd()
        except (tk.TclError, AttributeError):
            return
        GWL_EXSTYLE = -20
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_NOACTIVATE = 0x08000000
        style = int(ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
        ctypes.windll.user32.SetWindowLongW(
            hwnd,
            GWL_EXSTYLE,
            style | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        )

    def _position_for_target(self, target_window: int) -> tuple[int, int]:
        if self.position is not None:
            return self.position
        work_area = self._work_area_for_target(target_window)
        if work_area is not None:
            return overlay_position_in_work_area(
                *work_area,
                self.width,
                self.height,
            )
        return overlay_position(
            self.window.winfo_screenwidth(),
            self.window.winfo_screenheight(),
            self.width,
            self.height,
        )

    def _work_area_for_target(
        self, target_window: int
    ) -> Optional[tuple[int, int, int, int]]:
        if os.name == "nt" and target_window:
            MONITOR_DEFAULTTONEAREST = 0x00000002
            monitor = ctypes.windll.user32.MonitorFromWindow(
                target_window, MONITOR_DEFAULTTONEAREST
            )
            info = _MonitorInfo()
            info.cbSize = ctypes.sizeof(_MonitorInfo)
            if monitor and ctypes.windll.user32.GetMonitorInfoW(
                monitor, ctypes.byref(info)
            ):
                work = info.rcWork
                return (
                    work.left,
                    work.top,
                    work.right,
                    work.bottom,
                )
        return None

    def _show_at_saved_or_default(
        self,
        target_window: int = 0,
    ) -> None:
        x, y = self._position_for_target(target_window)
        geometry_x = x if x >= 0 else 0
        geometry_y = y if y >= 0 else 0
        self.window.geometry(
            f"{self.width}x{self.height}+{geometry_x}+{geometry_y}"
        )
        self._apply_no_activate_style()
        if os.name == "nt":
            show_window_no_activate(
                self._native_hwnd(), x, y, self.width, self.height
            )
        else:
            self.window.deiconify()

    def show_idle(self, hotkey: str = "Ctrl + Alt + Space") -> None:
        self._recording = False
        self.state_var.set("READY")
        self.text_var.set(hotkey)
        self.status_dot.configure(fg=self.palette.success)
        self.action_control.configure(text="●", fg=self.palette.success)
        self.set_meter(0)
        self._stop_pulse()
        self._show_at_saved_or_default()

    def show(
        self,
        target_window: int = 0,
        *,
        caret: Optional[tuple[int, int]] = None,
        caret_gap: int = 16,
    ) -> None:
        self._recording = True
        self.state_var.set("LISTENING")
        self.text_var.set("Speak now…")
        self.status_dot.configure(fg=self.palette.danger)
        self.action_control.configure(text="■", fg=self.palette.danger)
        self.set_meter(0)
        # Caret-adjacent recognition belongs to StreamingBubble. The main
        # control remains at its persisted position for the entire session.
        self._show_at_saved_or_default(target_window)
        # Idempotent: reapply no-activate style on every show in case Explorer
        # restarted (WS_EX_NOACTIVATE is reset when the taskbar is recreated).
        self._apply_no_activate_style()
        self._start_pulse()

    def update_partial(
        self,
        text: str,
        state: str = "LISTENING • LIVE",
        *,
        target_window: int = 0,
        caret: Optional[tuple[int, int]] = None,
        caret_gap: int = 16,
    ) -> None:
        self.state_var.set(state)
        self.text_var.set("Listening…")

    def set_meter(self, percent: float) -> None:
        self.meter.update_idletasks()
        width = max(1, self.meter.winfo_width())
        fill = int(width * max(0.0, min(100.0, float(percent))) / 100.0)
        self.meter.coords(self._meter_fill, 0, 0, fill, 4)

    def finalizing(self) -> None:
        self.state_var.set("FINALIZING")
        self.action_control.configure(text="…", fg=self.palette.warning)
        # Stop pulsing while we commit the final transcript — keeps the
        # indicator calm and signals "I'm not listening anymore".
        self._stop_pulse()

    def error(self, message: str) -> None:
        self.state_var.set("PAUSED")
        self.text_var.set(compact_overlay_text(message))
        self.status_dot.configure(fg=self.palette.warning)
        self.action_control.configure(text="●", fg=self.palette.warning)
        self._stop_pulse()

    def hide(self) -> None:
        self._stop_pulse()
        self.window.withdraw()

    def destroy(self) -> None:
        self._stop_pulse()
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    # ---------------------------------------------------------------- Pulse
    def _start_pulse(self) -> None:
        """Begin a gentle pulse on the status dot while recording."""
        if getattr(self, "_pulse_after_id", None) is not None:
            return  # already running
        window = getattr(self, "window", None)
        if window is None:
            return
        # We only schedule if Tk is alive (after() raises if window is gone).
        try:
            self._pulse_phase = 0
            self._pulse_after_id = window.after(
                self.PULSE_TICK_MS, self._pulse_step
            )
        except tk.TclError:
            self._pulse_after_id = None

    def _stop_pulse(self) -> None:
        if self._pulse_after_id is None:
            return
        try:
            self.window.after_cancel(self._pulse_after_id)
        except tk.TclError:
            pass
        self._pulse_after_id = None
        self._pulse_phase = 0
        # Restore solid dot color when we stop pulsing.
        try:
            color = self.palette.danger if self._recording else self.palette.success
            self.status_dot.configure(fg=color)
        except tk.TclError:
            pass

    def _pulse_step(self) -> None:
        """One tick of the pulse: swap dot color, schedule next."""
        self._pulse_after_id = None
        if not self._recording:
            return  # guard against stale tick after stop_pulse
        try:
            a, b = self._pulse_colors
            new_color = a if (self._pulse_phase % 2 == 0) else b
            self.status_dot.configure(fg=new_color)
            self._pulse_phase += 1
            self._pulse_after_id = self.window.after(
                self.PULSE_TICK_MS, self._pulse_step
            )
        except tk.TclError:
            # Window was destroyed mid-pulse; bail.
            self._pulse_after_id = None

    # ----------------------------------------------------- Keyboard hooks
    def bind_keyboard_shortcuts(self, root: Optional[tk.Misc] = None) -> None:
        """Wire accelerators to the three controls.

        Bindings target the bubble window directly so they only fire when it
        (or a child) has focus — we never steal global focus, so other apps
        remain fully interactive. ``root`` may be supplied if the caller wants
        the shortcuts available from the parent Tk as well.

        Keys:
          ``h``  — hide the bubble
          ``s``  — open settings
          ``Esc`` — hide (mirrors the universal "close" gesture)
        """
        targets: list[tk.Misc] = [self.window]
        if root is not None and root is not self.window:
            targets.append(root)
        for target in targets:
            target.bind("<KeyPress-h>", lambda _e: self.hide(), add="+")
            target.bind("<KeyPress-s>", lambda _e: self._open_settings(), add="+")
            target.bind("<Escape>", lambda _e: self.hide(), add="+")


class StreamingBubble:
    """Transient, no-activate speech preview anchored once per recording."""

    WIDTH = 420
    HEIGHT = 68
    FONT_SIZE = 10
    MAX_CHARACTERS = 120
    OPACITY = 0.82
    TRANSPARENT_KEY = "#010203"

    def __init__(
        self,
        root: tk.Tk,
        *,
        width: int = WIDTH,
        height: int = HEIGHT,
        font_size: int = FONT_SIZE,
        font_family: str = "Segoe UI",
        font_weight: str = "normal",
        max_characters: int = MAX_CHARACTERS,
        opacity: float = OPACITY,
        visual_style: str = "glitch_glass",
        text_color_mode: str = "theme",
        corner_radius: int = 20,
        border_width: int = 1,
        palette: Optional[ThemePalette] = None,
        on_offset_changed: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.root = root
        self.width = min(max(int(width), 280), 900)
        self.height = min(max(int(height), 56), 300)
        self.font_size = min(max(int(font_size), 8), 28)
        self.font_family = (
            font_family if font_family in BUBBLE_FONT_FAMILIES else "Segoe UI"
        )
        self.font_weight = (
            font_weight if font_weight in BUBBLE_FONT_WEIGHTS else "normal"
        )
        self.max_characters = min(max(int(max_characters), 40), 1000)
        self.opacity = min(max(float(opacity), 0.35), 0.98)
        self.visual_style = (
            visual_style if visual_style in BUBBLE_VISUAL_STYLES else "glitch_glass"
        )
        self.text_color_mode = (
            text_color_mode
            if text_color_mode in BUBBLE_TEXT_COLOR_MODES
            else "theme"
        )
        self.corner_radius = min(max(int(corner_radius), 4), 32)
        self.border_width = min(max(int(border_width), 0), 4)
        self.palette = palette or build_palette("dark", "violet")
        self.on_offset_changed = on_offset_changed
        self.position: Optional[tuple[int, int]] = None
        self._anchor: Optional[tuple[int, int]] = None
        self._work_area: Optional[tuple[int, int, int, int]] = None
        self._drag_origin: Optional[tuple[int, int, int, int]] = None
        self._visible = False

        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.configure(bg=self.TRANSPARENT_KEY)
        self.window.attributes("-topmost", True)
        try:
            self.window.attributes("-alpha", self.opacity)
            if os.name == "nt":
                self.window.wm_attributes("-transparentcolor", self.TRANSPARENT_KEY)
        except tk.TclError:
            pass

        self.canvas = tk.Canvas(
            self.window,
            width=self.width,
            height=self.height,
            bg=self.TRANSPARENT_KEY,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self._shape = self.canvas.create_polygon(
            self._rounded_points(
                self.width,
                self.height,
                min(self.corner_radius, self.width // 2, self.height // 2),
            ),
            smooth=True,
            splinesteps=18,
            fill=self.palette.bubble,
            outline=self.palette.accent,
            width=1,
        )
        self._state_text = self.canvas.create_text(
            18,
            17,
            anchor="w",
            text="LISTENING",
            fill=self.palette.accent_soft,
            font=bubble_font_spec(self.font_family, 8, "semibold"),
        )
        self._preview_text = self.canvas.create_text(
            18,
            31,
            anchor="nw",
            text="",
            width=max(80, self.width - 36),
            fill=self.palette.text,
            font=bubble_font_spec(
                self.font_family,
                self.font_size,
                self.font_weight,
            ),
        )
        self.apply_palette(self.palette)
        self.canvas.configure(cursor="fleur")
        self.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
        self.window.update_idletasks()
        self._apply_no_activate_style()

    @staticmethod
    def _rounded_points(width: int, height: int, radius: int) -> tuple[int, ...]:
        return (
            radius, 0, width - radius, 0, width, 0, width, radius,
            width, height - radius, width, height, width - radius, height,
            radius, height, 0, height, 0, height - radius, 0, radius, 0, 0,
        )

    def _native_hwnd(self) -> int:
        hwnd = int(self.window.winfo_id())
        if os.name == "nt":
            parent = int(ctypes.windll.user32.GetParent(hwnd))
            if parent:
                hwnd = parent
        return hwnd

    def _apply_no_activate_style(self) -> None:
        if os.name != "nt":
            return
        # Defensive: test doubles in some modules (e.g. dictation_features)
        # construct ``DictationOverlay`` subclasses without a Tk ``window``;
        # we silently no-op in that case rather than crashing.
        if not hasattr(self, "window"):
            return
        try:
            hwnd = self._native_hwnd()
        except (tk.TclError, AttributeError):
            return
        GWL_EXSTYLE = -20
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_NOACTIVATE = 0x08000000
        hwnd = self._native_hwnd()
        style = int(ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
        ctypes.windll.user32.SetWindowLongW(
            hwnd,
            GWL_EXSTYLE,
            style | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        )

    def _work_area_for_target(self, target_window: int) -> tuple[int, int, int, int]:
        if os.name == "nt" and target_window:
            MONITOR_DEFAULTTONEAREST = 0x00000002
            monitor = ctypes.windll.user32.MonitorFromWindow(
                target_window, MONITOR_DEFAULTTONEAREST
            )
            info = _MonitorInfo()
            info.cbSize = ctypes.sizeof(_MonitorInfo)
            if monitor and ctypes.windll.user32.GetMonitorInfoW(
                monitor, ctypes.byref(info)
            ):
                work = info.rcWork
                return work.left, work.top, work.right, work.bottom
        return 0, 0, self.window.winfo_screenwidth(), self.window.winfo_screenheight()

    def begin(
        self,
        target_window: int,
        *,
        caret: Optional[tuple[int, int]] = None,
        caret_gap: int = 16,
        fallback_position: Optional[tuple[int, int]] = None,
        offset: tuple[int, int] = (18, -88),
    ) -> None:
        work_area = self._work_area_for_target(target_window)
        self._work_area = work_area
        self._anchor = caret
        if caret is not None:
            self.position = anchored_bubble_position(
                *caret,
                *work_area,
                self.width,
                self.height,
                offset_x=int(offset[0]),
                offset_y=int(offset[1]),
            )
        elif fallback_position is not None:
            self.position = clamp_overlay_position(
                fallback_position[0],
                fallback_position[1] - self.height - 12,
                *work_area,
                self.width,
                self.height,
            )
        else:
            self.position = overlay_position_in_work_area(
                *work_area, self.width, self.height
            )
        self.canvas.itemconfigure(self._state_text, text="LISTENING")
        self.canvas.itemconfigure(self._preview_text, text="Listening…")
        x, y = self.position
        geometry_x = x if x >= 0 else 0
        geometry_y = y if y >= 0 else 0
        self.window.geometry(
            f"{self.width}x{self.height}+{geometry_x}+{geometry_y}"
        )
        self._apply_no_activate_style()
        if os.name == "nt":
            show_window_no_activate(
                self._native_hwnd(), x, y, self.width, self.height
            )
        else:
            self.window.deiconify()
        self._visible = True

    def _drag_start(self, event) -> None:
        if self.position is None:
            return
        self._drag_origin = (
            int(self.position[0]),
            int(self.position[1]),
            int(event.x_root),
            int(event.y_root),
        )

    def _drag_motion(self, event) -> None:
        if self._drag_origin is None or self._work_area is None:
            return
        x, y = dragged_overlay_position(
            *self._drag_origin,
            int(event.x_root),
            int(event.y_root),
        )
        x, y = clamp_overlay_position(
            x,
            y,
            *self._work_area,
            self.width,
            self.height,
        )
        self.position = (x, y)
        self.window.geometry(f"{self.width}x{self.height}+{max(0, x)}+{max(0, y)}")
        if os.name == "nt" and getattr(self, "_visible", False):
            show_window_no_activate(
                self._native_hwnd(), x, y, self.width, self.height
            )

    def _drag_end(self, event) -> None:
        if self._drag_origin is None:
            return
        self._drag_motion(event)
        self._drag_origin = None
        if (
            self.position is not None
            and self._anchor is not None
            and self.on_offset_changed is not None
        ):
            offset_x, offset_y = bubble_offset_for_position(
                *self.position,
                anchor_x=self._anchor[0],
                anchor_y=self._anchor[1],
            )
            self.on_offset_changed(offset_x, offset_y)

    def configure(
        self,
        *,
        width: int,
        height: int,
        font_size: int,
        max_characters: int,
        opacity: float,
        font_family: Optional[str] = None,
        font_weight: Optional[str] = None,
        visual_style: Optional[str] = None,
        text_color_mode: Optional[str] = None,
        corner_radius: Optional[int] = None,
        border_width: Optional[int] = None,
    ) -> None:
        self.width = min(max(int(width), 280), 900)
        self.height = min(max(int(height), 56), 300)
        self.font_size = min(max(int(font_size), 8), 28)
        self.max_characters = min(max(int(max_characters), 40), 1000)
        self.opacity = min(max(float(opacity), 0.35), 0.98)
        if font_family is not None:
            self.font_family = (
                font_family if font_family in BUBBLE_FONT_FAMILIES else "Segoe UI"
            )
        if font_weight is not None:
            self.font_weight = (
                font_weight if font_weight in BUBBLE_FONT_WEIGHTS else "normal"
            )
        if visual_style is not None:
            self.visual_style = (
                visual_style
                if visual_style in BUBBLE_VISUAL_STYLES
                else "glitch_glass"
            )
        if text_color_mode is not None:
            self.text_color_mode = (
                text_color_mode
                if text_color_mode in BUBBLE_TEXT_COLOR_MODES
                else "theme"
            )
        if corner_radius is not None:
            self.corner_radius = min(max(int(corner_radius), 4), 32)
        if border_width is not None:
            self.border_width = min(max(int(border_width), 0), 4)
        self.canvas.configure(width=self.width, height=self.height)
        self.canvas.coords(
            self._shape,
            *self._rounded_points(
                self.width,
                self.height,
                min(self.corner_radius, self.width // 2, self.height // 2),
            ),
        )
        self.canvas.itemconfigure(
            self._state_text,
            font=bubble_font_spec(self.font_family, 8, "semibold"),
        )
        self.canvas.itemconfigure(
            self._preview_text,
            width=max(80, self.width - 36),
            font=bubble_font_spec(
                self.font_family,
                self.font_size,
                self.font_weight,
            ),
        )
        self._apply_visual_style()
        try:
            self.window.attributes("-alpha", self.opacity)
        except tk.TclError:
            pass

    def apply_palette(self, palette: ThemePalette) -> None:
        self.palette = palette
        self._apply_visual_style()

    def _apply_visual_style(self) -> None:
        style = resolve_bubble_visual_style(
            self.palette,
            self.visual_style,
            self.text_color_mode,
            border_width=self.border_width,
        )
        self.canvas.itemconfigure(
            self._shape,
            fill=style.background,
            outline=style.border,
            width=style.border_width,
        )
        self.canvas.itemconfigure(self._state_text, fill=style.state)
        self.canvas.itemconfigure(self._preview_text, fill=style.text)

    def update(self, text: str, state: str = "LISTENING") -> None:
        if self.position is None:
            return
        preview = compact_overlay_text(
            text,
            max_chars=getattr(self, "max_characters", self.MAX_CHARACTERS),
        )
        if not str(text or "").strip():
            preview = "Listening…"
        # Always push canvas text so rapid partials never stick on a stale frame.
        self.canvas.itemconfigure(self._state_text, text=state)
        self.canvas.itemconfigure(self._preview_text, text=preview)
        update_idletasks = getattr(self.canvas, "update_idletasks", None)
        if callable(update_idletasks):
            try:
                update_idletasks()
            except tk.TclError:
                pass
        if not self._visible:
            x, y = self.position
            geometry_x = x if x >= 0 else 0
            geometry_y = y if y >= 0 else 0
            self.window.geometry(
                f"{self.width}x{self.height}+{geometry_x}+{geometry_y}"
            )
            self._apply_no_activate_style()
            if os.name == "nt":
                show_window_no_activate(
                    self._native_hwnd(), x, y, self.width, self.height
                )
            else:
                self.window.deiconify()
            self._visible = True

    def finalizing(self) -> None:
        if self._visible:
            self.canvas.itemconfigure(self._state_text, text="FINALIZING")

    def hide(self) -> None:
        self._visible = False
        self.position = None
        self.window.withdraw()

    def destroy(self) -> None:
        try:
            self.window.destroy()
        except tk.TclError:
            pass
