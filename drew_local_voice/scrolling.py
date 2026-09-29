"""Shared native Tk scrolling contracts for GlitchScribe surfaces.

The production UI has more than one scrollable surface.  Keeping the event
normalization and bind-tag installation here prevents one surface from fixing
Windows wheel events while another still loses events to child widget
bindtags.  The module is Tk-only, local, and metadata-free: callers own any
optional runtime logging.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import tkinter as tk


WHEEL_DELTA = 120


class WheelDeltaAccumulator:
    """Accumulate high-resolution wheel/trackpad deltas into Tk scroll units."""

    def __init__(self, wheel_delta: int = WHEEL_DELTA) -> None:
        self.wheel_delta = max(1, int(wheel_delta))
        self._remainder = 0

    @property
    def remainder(self) -> int:
        """Return the bounded raw-delta remainder awaiting a full unit."""
        return int(self._remainder)

    def reset(self) -> None:
        self._remainder = 0

    def consume(self, delta: int) -> int:
        """Consume a raw Tk delta and return signed Tk ``yview`` units.

        Windows mouse wheels normally deliver +/-120. Precision touchpads can
        deliver many smaller deltas; retaining the remainder prevents those
        events from becoming a full scroll unit each and makes the movement
        proportional instead of jumpy.
        """
        value = int(delta)
        if not value:
            return 0
        self._remainder += value
        units = int(self._remainder / self.wheel_delta)
        if units:
            self._remainder -= units * self.wheel_delta
        # Tk's positive yview direction is down while a positive wheel delta
        # means up, so invert the platform delta just like the legacy helper.
        return -units

    def consume_discrete(self, units: int) -> int:
        """Reset partial precision input and return a discrete button unit."""
        self.reset()
        return int(units)


def _walk_widgets(widget: tk.Misc) -> Iterator[tk.Misc]:
    yield widget
    try:
        children = tuple(widget.winfo_children())
    except tk.TclError:
        return
    for child in children:
        yield from _walk_widgets(child)


def install_scroll_bindings(
    container: tk.Misc,
    mousewheel_handler: Callable[[tk.Event], str | None],
    *,
    button4_handler: Callable[[tk.Event], str | None] | None = None,
    button5_handler: Callable[[tk.Event], str | None] | None = None,
) -> str:
    """Install a high-priority bind tag across a scrollable widget subtree.

    A toplevel/root binding can be skipped when a child widget's class binding
    returns ``break``.  This tag is placed before the widget/class/toplevel
    tags so scrolling remains consistent over labels, ttk controls, text
    fields, and canvases.  The returned tag can be passed to
    :func:`uninstall_scroll_bindings` when a dialog closes.
    """
    tag = f"GlitchScribeScroll-{id(container):x}"
    container.bind_class(tag, "<MouseWheel>", mousewheel_handler, add="+")
    container.bind_class(
        tag,
        "<Button-4>",
        button4_handler or mousewheel_handler,
        add="+",
    )
    container.bind_class(
        tag,
        "<Button-5>",
        button5_handler or mousewheel_handler,
        add="+",
    )
    for widget in _walk_widgets(container):
        try:
            tags = tuple(widget.bindtags())
            if tag not in tags:
                widget.bindtags((tag, *tags))
        except tk.TclError:
            continue
    return tag


def uninstall_scroll_bindings(container: tk.Misc, tag: str | None) -> None:
    """Remove a previously installed scroll bind tag without touching others."""
    if not tag:
        return
    for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        try:
            container.unbind_class(tag, sequence)
        except tk.TclError:
            pass
    for widget in _walk_widgets(container):
        try:
            widget.bindtags(tuple(item for item in widget.bindtags() if item != tag))
        except tk.TclError:
            continue


__all__ = [
    "WHEEL_DELTA",
    "WheelDeltaAccumulator",
    "install_scroll_bindings",
    "uninstall_scroll_bindings",
]
