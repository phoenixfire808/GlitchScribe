"""GPU-independent, no-focus cursor halo and event burst for Windows."""

from __future__ import annotations

import ctypes
import math
import os
import tkinter as tk
from ctypes import wintypes
from typing import Optional

from dictation_overlay import show_window_no_activate
from drew_local_voice.cursor_fx import CURSOR_STYLE_CHOICES, clamp_intensity
from drew_local_voice.theme import ThemePalette

TRANSPARENT_KEY = "#010101"
WM_NCHITTEST = 0x0084
HTTRANSPARENT = -1
GWL_EXSTYLE = -20
GWL_WNDPROC = -4
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000


def ring_phase_geometry(size: int, phase: float) -> dict[str, tuple[float, ...]]:
    """Return deterministic halo geometry for structural/UI tests."""
    center = float(size) / 2.0
    phase = float(phase)
    pulse = (math.sin(phase) + 1.0) / 2.0
    sweep = (phase * 56.0) % 360.0
    outer = 68.0 + 10.0 * pulse
    mid = 54.0 + 6.0 * pulse
    inner = 24.0 + 2.0 * pulse
    return {
        "center": (center, center),
        "outer": (center - outer, center - outer, center + outer, center + outer),
        "mid": (center - mid, center - mid, center + mid, center + mid),
        "inner": (center - inner, center - inner, center + inner, center + inner),
        "arc_start": (-118.0 + sweep) % 360.0,
        "arc_extent": 198.0 + 22.0 * pulse,
        "comet_start": (42.0 + sweep * 1.7) % 360.0,
        "comet_extent": 58.0 + 8.0 * pulse,
        "orbit_radius": 72.0 + 4.0 * pulse,
    }


class MicrophoneActiveRing:
    """Cursor-centered status halo with one-shot sound-synchronized bursts."""

    SIZE = 240
    TICK_MS = 24

    def __init__(
        self,
        root: tk.Misc,
        palette: ThemePalette,
        *,
        style: str = "orbit",
        intensity: float = 1.0,
    ) -> None:
        self.root = root
        self.palette = palette
        self.window: Optional[tk.Toplevel] = None
        self.canvas: Optional[tk.Canvas] = None
        self._active = False
        self._tick_job: Optional[str] = None
        self._burst_job: Optional[str] = None
        self._phase = 0.0
        self._burst_phase = 0.0
        self._burst_active = False
        self._destroyed = False
        self._wndproc = None
        self._old_wndproc = None
        self._reduced_motion = False
        self.style = style if style in CURSOR_STYLE_CHOICES else "orbit"
        self.intensity = clamp_intensity(intensity)

    @property
    def active(self) -> bool:
        return self._active

    def configure(self, *, style: Optional[str] = None, intensity: Optional[float] = None) -> None:
        if style is not None:
            normalized = str(style).strip().casefold()
            self.style = normalized if normalized in CURSOR_STYLE_CHOICES else "orbit"
        if intensity is not None:
            self.intensity = clamp_intensity(intensity)
        if self._active:
            self._render()

    def trigger(
        self,
        *,
        style: Optional[str] = None,
        intensity: Optional[float] = None,
        reduced_motion: bool = False,
    ) -> None:
        """Start a visible expanding burst at the current cursor position."""
        self.configure(style=style, intensity=intensity)
        if self._destroyed or not self._ensure_window():
            return
        self._burst_active = True
        self._burst_phase = 0.0
        self._reduced_motion = bool(reduced_motion)
        try:
            self.window.deiconify()
            self.window.attributes("-topmost", True)
            self.window.update_idletasks()
            self._position()
            if self._burst_job is not None:
                self.window.after_cancel(self._burst_job)
            if self._reduced_motion:
                # Reduced motion keeps a stable confirmation flash rather than
                # animating expanding rings; the event still remains visible.
                self._burst_phase = 0.45
                self._render()
                self._burst_job = self.window.after(120, self._burst_tick)
            else:
                self._render()
                self._burst_job = self.window.after(0, self._burst_tick)
        except tk.TclError:
            self._burst_job = None

    def _burst_tick(self) -> None:
        self._burst_job = None
        if self._destroyed or not self._burst_active:
            return
        if self._reduced_motion:
            self._burst_active = False
            if not self._active and self.window is not None:
                try:
                    self.window.withdraw()
                except tk.TclError:
                    pass
            return
        self._burst_phase += 0.105
        self._position()
        self._render()
        if self._burst_phase >= 1.0:
            self._burst_active = False
            if not self._active and self.window is not None:
                try:
                    self.window.withdraw()
                except tk.TclError:
                    pass
            return
        try:
            if self.window is not None:
                self._burst_job = self.window.after(self.TICK_MS, self._burst_tick)
        except tk.TclError:
            self._burst_job = None

    def _native_hwnd(self) -> int:
        if self.window is None:
            return 0
        hwnd = int(self.window.winfo_id())
        if os.name == "nt":
            parent = int(ctypes.windll.user32.GetParent(hwnd) or 0)
            if parent:
                hwnd = parent
        return hwnd

    def _apply_click_through_style(self) -> None:
        if os.name != "nt" or self.window is None:
            return
        hwnd = self._native_hwnd()
        if not hwnd:
            return
        user32 = ctypes.windll.user32
        style = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
        user32.SetWindowLongW(
            hwnd,
            GWL_EXSTYLE,
            style | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_LAYERED | WS_EX_NOACTIVATE,
        )
        if self._wndproc is not None:
            return
        try:
            WNDPROC = ctypes.WINFUNCTYPE(
                ctypes.c_ssize_t,
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )

            user32.CallWindowProcW.argtypes = [
                ctypes.c_void_p,
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]
            user32.CallWindowProcW.restype = ctypes.c_ssize_t

            def _proc(window, message, wparam, lparam):
                if int(message) == WM_NCHITTEST:
                    return HTTRANSPARENT
                return user32.CallWindowProcW(
                    self._old_wndproc, window, message, wparam, lparam
                )

            self._wndproc = WNDPROC(_proc)
            user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.GetWindowLongPtrW.restype = ctypes.c_void_p
            user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
            user32.SetWindowLongPtrW.restype = ctypes.c_void_p
            self._old_wndproc = user32.GetWindowLongPtrW(hwnd, GWL_WNDPROC)
            user32.SetWindowLongPtrW(hwnd, GWL_WNDPROC, ctypes.cast(self._wndproc, ctypes.c_void_p))
        except (AttributeError, OSError, TypeError, ValueError):
            self._wndproc = None
            self._old_wndproc = None

    def _ensure_window(self) -> bool:
        if self._destroyed:
            return False
        if self.window is not None and self.canvas is not None:
            return True
        try:
            window = tk.Toplevel(self.root)
            window.withdraw()
            window.overrideredirect(True)
            window.attributes("-topmost", True)
            window.configure(bg=TRANSPARENT_KEY)
            try:
                window.attributes("-transparentcolor", TRANSPARENT_KEY)
            except tk.TclError:
                window.attributes("-alpha", 0.94)
            canvas = tk.Canvas(
                window, width=self.SIZE, height=self.SIZE, bg=TRANSPARENT_KEY,
                bd=0, highlightthickness=0, relief="flat",
            )
            canvas.pack(fill="both", expand=True)
            self.window = window
            self.canvas = canvas
            window.update_idletasks()
            self._apply_click_through_style()
            return True
        except tk.TclError:
            self.window = None
            self.canvas = None
            return False

    def _cursor_position(self) -> Optional[tuple[int, int]]:
        if os.name != "nt":
            return None
        try:
            point = wintypes.POINT()
            if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
                return None
            return int(point.x), int(point.y)
        except (AttributeError, OSError, TypeError, ValueError):
            return None

    @staticmethod
    def _box(cx: float, cy: float, radius: float) -> tuple[float, float, float, float]:
        return (cx - radius, cy - radius, cx + radius, cy + radius)

    def _render(self) -> None:
        if self.canvas is None:
            return
        size = float(self.SIZE)
        cx = cy = size / 2.0
        try:
            self.canvas.delete("cursor-fx")
            intensity = clamp_intensity(self.intensity)
            halo = 82.0 + (5.0 * intensity) * math.sin(self._phase)
            burst = min(max(self._burst_phase, 0.0), 1.0)
            primary_width = max(3, int(round(8.0 * intensity)))
            secondary_width = max(2, int(round(4.0 * intensity)))
            if self._burst_active:
                fade = 1.0 - burst
                for index, color in enumerate((self.palette.accent, self.palette.success, self.palette.accent_soft)):
                    radius = 18.0 + burst * (34.0 + index * 18.0)
                    self.canvas.create_oval(
                        *self._box(cx, cy, radius), outline=color,
                        width=max(1, int(round((5.0 - index) * fade * self.intensity))),
                        tags="cursor-fx",
                    )
            style = self.style
            if style == "ripple":
                radii = (48.0, 59.0, 70.0)
                colors = (self.palette.accent, self.palette.accent_soft, self.palette.success)
                for radius, color in zip(radii, colors):
                    self.canvas.create_oval(*self._box(cx, cy, radius), outline=color, width=2, tags="cursor-fx")
            elif style == "comet" and not self._reduced_motion:
                geom = ring_phase_geometry(self.SIZE, self._phase)
                self.canvas.create_arc(*geom["outer"], start=geom["arc_start"], extent=geom["arc_extent"], style="arc", outline=self.palette.accent, width=primary_width, tags="cursor-fx")
                self.canvas.create_arc(*geom["mid"], start=geom["comet_start"], extent=geom["comet_extent"], style="arc", outline=self.palette.success, width=secondary_width, tags="cursor-fx")
                comet_angle = self._phase * 1.8
                comet_x = cx + math.cos(comet_angle) * geom["orbit_radius"]
                comet_y = cy + math.sin(comet_angle) * geom["orbit_radius"]
                self.canvas.create_line(cx, cy, comet_x, comet_y, fill=self.palette.accent_soft, width=2, tags="cursor-fx")
                self.canvas.create_oval(comet_x - 5, comet_y - 5, comet_x + 5, comet_y + 5, fill=self.palette.warning, outline="", tags="cursor-fx")
            elif style == "spectrum" and not self._reduced_motion:
                for index, color in enumerate((self.palette.accent, self.palette.success, self.palette.danger, self.palette.accent_soft, self.palette.warning)):
                    radius = 39.0 + index * 8.0 + 2.0 * math.sin(self._phase + index)
                    self.canvas.create_arc(*self._box(cx, cy, radius), start=(self._phase * 55 + index * 72) % 360, extent=52, style="arc", outline=color, width=4 if index == 0 else 2, tags="cursor-fx")
            elif style == "minimal":
                self.canvas.create_oval(*self._box(cx, cy, 45.0), outline=self.palette.accent_soft, width=max(1, int(round(2.0 * intensity))), tags="cursor-fx")
                self.canvas.create_oval(*self._box(cx, cy, 10.0 + (2.0 * intensity) * math.sin(self._phase)), fill=self.palette.accent, outline="", tags="cursor-fx")
            else:
                geom = ring_phase_geometry(self.SIZE, self._phase)
                self.canvas.create_oval(*geom["outer"], outline=self.palette.accent_soft, width=max(2, int(round(2.5 * intensity))), tags="cursor-fx")
                self.canvas.create_arc(*geom["outer"], start=geom["arc_start"], extent=geom["arc_extent"], style="arc", outline=self.palette.accent, width=primary_width, tags="cursor-fx")
                self.canvas.create_arc(*geom["mid"], start=(geom["arc_start"] + 92.0) % 360.0, extent=118.0, style="arc", outline=self.palette.success, width=secondary_width, tags="cursor-fx")
                self.canvas.create_arc(*geom["inner"], start=(geom["arc_start"] + 214.0) % 360.0, extent=72.0, style="arc", outline=self.palette.warning, width=max(1, secondary_width - 1), tags="cursor-fx")
                self.canvas.create_oval(*geom["inner"], outline=self.palette.accent_soft, width=1, tags="cursor-fx")
                self.canvas.create_oval(*self._box(cx, cy, 13.0), fill=TRANSPARENT_KEY, outline="", tags="cursor-fx")
                if not self._reduced_motion:
                    orbit_radius = geom["orbit_radius"]
                    for index, color in enumerate((self.palette.accent, self.palette.success, self.palette.accent_soft, self.palette.danger)):
                        angle = self._phase * 1.6 + math.tau * index / 4.0
                        dot_x = cx + math.cos(angle) * orbit_radius
                        dot_y = cy + math.sin(angle) * orbit_radius
                        tail_x = cx + math.cos(angle - 0.2) * (orbit_radius - 10.0)
                        tail_y = cy + math.sin(angle - 0.2) * (orbit_radius - 10.0)
                        dot_size = 4.0 if index == 0 else 2.6
                        self.canvas.create_line(tail_x, tail_y, dot_x, dot_y, fill=color, width=2, capstyle="round", tags="cursor-fx")
                        self.canvas.create_oval(dot_x-dot_size, dot_y-dot_size, dot_x+dot_size, dot_y+dot_size, fill=color, outline="", tags="cursor-fx")
            if self._active:
                self.canvas.create_oval(*self._box(cx, cy, 5.0), fill=self.palette.danger, outline=self.palette.accent_soft, width=1, tags="cursor-fx")
                self.canvas.create_oval(*self._box(cx, cy, 1.5), fill=TRANSPARENT_KEY, outline="", tags="cursor-fx")
        except tk.TclError:
            pass

    def _position(self) -> None:
        if self.window is None:
            return
        cursor = self._cursor_position()
        if cursor is None:
            return
        x = int(cursor[0] - self.SIZE / 2)
        y = int(cursor[1] - self.SIZE / 2)
        try:
            self.window.geometry(f"{self.SIZE}x{self.SIZE}+{x}+{y}")
            if os.name == "nt":
                show_window_no_activate(self._native_hwnd(), x, y, self.SIZE, self.SIZE)
        except tk.TclError:
            pass

    def _tick(self) -> None:
        self._tick_job = None
        if not self._active or self._destroyed:
            return
        if not self._reduced_motion:
            self._phase += 0.26
        self._position()
        self._render()
        try:
            if self.window is not None:
                self._tick_job = self.window.after(self.TICK_MS, self._tick)
        except tk.TclError:
            self._tick_job = None

    def show(
        self,
        *,
        style: Optional[str] = None,
        intensity: Optional[float] = None,
        reduced_motion: bool = False,
    ) -> None:
        if self._destroyed or not self._ensure_window():
            return
        self.configure(style=style, intensity=intensity)
        self._active = True
        self._reduced_motion = bool(reduced_motion)
        try:
            self.window.deiconify()
            self.window.attributes("-topmost", True)
            self.window.update_idletasks()
        except tk.TclError:
            return
        self._position()
        self._render()
        try:
            self._tick_job = self.window.after(0, self._tick)
        except tk.TclError:
            self._tick_job = None

    def hide(self) -> None:
        self._active = False
        self._reduced_motion = False
        if self._tick_job is not None and self.window is not None:
            try:
                self.window.after_cancel(self._tick_job)
            except tk.TclError:
                pass
        self._tick_job = None
        if self._burst_job is not None and self.window is not None:
            try:
                self.window.after_cancel(self._burst_job)
            except tk.TclError:
                pass
        self._burst_job = None
        self._burst_active = False
        if self.window is not None:
            try:
                self.window.withdraw()
            except tk.TclError:
                pass

    def destroy(self) -> None:
        self._destroyed = True
        self.hide()
        if self.window is not None:
            try:
                self.window.destroy()
            except tk.TclError:
                pass
        self.window = None
        self.canvas = None


__all__ = ["MicrophoneActiveRing", "ring_phase_geometry"]
