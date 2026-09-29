"""Low-glare circular mouse-wheel feedback for the native Tk surface.

Tk does not render SVG natively on all supported Windows builds.  The visual
contract is therefore expressed as the same circle/arc geometry in a
Canvas-native renderer, with the SVG path retained as a design/source token
for future renderers.  The indicator never creates a window or captures focus.
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Optional

from drew_local_voice.theme import ThemePalette

WHEEL_INDICATOR_SVG = (
    '<svg viewBox="0 0 48 48" aria-hidden="true">'
    '<circle cx="24" cy="24" r="17" fill="none" stroke="currentColor" stroke-width="3"/>'
    '<path d="M24 7v8m0 18v8M7 24h8m18 0h8" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>'
    '</svg>'
)


def normalize_scroll_intensity(delta: int, units: int = 0) -> float:
    """Map wheel movement to a stable 0..1 visual intensity."""
    raw = max(abs(int(delta)) / 480.0, abs(int(units)) / 8.0)
    return max(0.08, min(1.0, raw))


class ScrollWheelIndicator:
    """Transient circular indicator drawn into an existing Tk Canvas."""

    def __init__(self, canvas: tk.Canvas, palette: ThemePalette) -> None:
        self.canvas = canvas
        self.palette = palette
        self._items: list[int] = []
        self._fade_job: Optional[str] = None
        self._destroyed = False

    def show(self, x: int, y: int, *, delta: int, units: int = 0) -> float:
        """Show a velocity-scaled ring at canvas coordinates and return intensity."""
        if self._destroyed:
            return 0.0
        intensity = normalize_scroll_intensity(delta, units)
        self._cancel_fade()
        self._clear()
        radius = 15 + round(7 * intensity)
        shadow_radius = radius + 7 + round(4 * intensity)
        tags = ("scroll-wheel-indicator",)
        try:
            # Two rings provide the subtle depth/shadow treatment without an
            # alpha dependency or a second transparent top-level window.
            self._items.append(
                self.canvas.create_oval(
                    x - shadow_radius,
                    y - shadow_radius,
                    x + shadow_radius,
                    y + shadow_radius,
                    outline=self.palette.border,
                    width=2 + round(2 * intensity),
                    state="normal",
                    tags=tags,
                )
            )
            self._items.append(
                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    outline=self.palette.accent_soft,
                    width=2 + round(2 * intensity),
                    fill=self.palette.surface_alt,
                    tags=tags,
                )
            )
            extent = max(35, min(330, round(80 + 230 * intensity)))
            start = -90 if delta <= 0 else 90
            self._items.append(
                self.canvas.create_arc(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    start=start,
                    extent=extent if delta <= 0 else -extent,
                    style="arc",
                    outline=self.palette.accent,
                    width=3 + round(2 * intensity),
                    tags=tags,
                )
            )
            hub = max(3, round(3 + 3 * intensity))
            self._items.append(
                self.canvas.create_oval(
                    x - hub,
                    y - hub,
                    x + hub,
                    y + hub,
                    fill=self.palette.accent,
                    outline="",
                    tags=tags,
                )
            )
            self._fade_job = self.canvas.after(260, self._fade)
        except tk.TclError:
            self._clear()
        return intensity

    def _fade(self) -> None:
        self._fade_job = None
        if self._destroyed:
            return
        try:
            # Short delayed removal creates a clean fade-like interaction
            # without relying on unsupported per-item alpha in Tk Canvas.
            for item in self._items[:1]:
                self.canvas.itemconfigure(item, outline=self.palette.surface_alt, width=1)
            self._fade_job = self.canvas.after(70, self._clear)
        except tk.TclError:
            self._clear()

    def _cancel_fade(self) -> None:
        if self._fade_job is not None:
            try:
                self.canvas.after_cancel(self._fade_job)
            except tk.TclError:
                pass
            self._fade_job = None

    def _clear(self) -> None:
        for item in self._items:
            try:
                self.canvas.delete(item)
            except tk.TclError:
                pass
        self._items.clear()

    def destroy(self) -> None:
        self._destroyed = True
        self._cancel_fade()
        self._clear()


__all__ = ["WHEEL_INDICATOR_SVG", "ScrollWheelIndicator", "normalize_scroll_intensity"]
