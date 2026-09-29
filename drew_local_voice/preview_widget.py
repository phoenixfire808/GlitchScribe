"""GlitchScribe Control Lab live bubble preview widget.

This module renders the staged values from :class:`BubblePreviewModel` on an
embedded Tk ``Frame``/``Canvas``. It is deliberately split into two layers:

* :func:`build_preview_render_spec` is a pure function. It accepts a
  :class:`BubblePreviewModel` and returns an immutable :class:`PreviewRenderSpec`
  containing every visual input the widget needs (dimensions, palette, font,
  text, state). The pure spec is the source of truth for headless tests and
  for any future non-Tk renderer.
* :class:`PreviewWidget` is the embedded Tk widget. It consumes a render spec
  via :meth:`refresh` and draws it on a Canvas. The widget is an embedded
  ``Frame`` and never spawns a Toplevel window, never applies ``-topmost`` /
  ``-alpha`` / Win32 no-activate tricks, and never touches the microphone,
  audio, or network state.

The widget reuses production helpers from ``dictation_overlay`` (``bubble_font_spec``
and ``resolve_bubble_visual_style``) and :func:`drew_local_voice.theme.build_palette`
to keep its look-and-feel aligned with the real :class:`DictationOverlay`.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from dictation_overlay import (
    bubble_font_spec,
    resolve_bubble_visual_style,
)
from drew_local_voice.preview_model import (
    BubblePreviewModel,
    PREVIEW_STATE_LABELS,
)
from drew_local_voice.theme import ThemePalette, build_palette


# Preview envelope: keeps the embedded widget within the Control Lab layout.
PREVIEW_MIN_WIDTH = 240
PREVIEW_MAX_WIDTH = 720
PREVIEW_MIN_HEIGHT = 56
PREVIEW_MAX_HEIGHT = 240

# Visual padding inside the preview surface.
_PREVIEW_PAD_X = 14
_PREVIEW_PAD_Y = 10
_STATE_BAR_HEIGHT = 18


@dataclass(frozen=True)
class PreviewRenderSpec:
    """Pure, immutable description of what the preview widget should draw.

    All fields are pre-validated and safe to feed to a Canvas. ``font`` is the
    Tk font tuple produced by ``bubble_font_spec`` and is therefore directly
    accepted by ``canvas.itemconfigure(..., font=...)``.
    """

    width: int
    height: int
    corner_radius: int
    border_width: int
    font: tuple
    state_text: str
    display_text: str
    background: str
    border: str
    text: str
    state: str
    opacity: float
    theme: str
    accent: str
    visual_style: str
    text_color_mode: str


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(float(value), float(minimum)), float(maximum))


def build_preview_render_spec(model: BubblePreviewModel) -> PreviewRenderSpec:
    """Derive a deterministic render spec from a :class:`BubblePreviewModel`.

    The function never touches Tk, the microphone, audio, or network state.
    Runtime-only settings (``audio.microphone``, ``runtime.target_hwnd`` …)
    are filtered out by :func:`extract_bubble_settings` upstream and therefore
    cannot leak into the spec.
    """
    if not isinstance(model, BubblePreviewModel):
        raise TypeError(
            "build_preview_render_spec requires a BubblePreviewModel instance"
        )

    settings = model.settings
    palette: ThemePalette = build_palette(
        settings["ui.theme"],
        settings["ui.accent_color"],
    )
    visual = resolve_bubble_visual_style(
        palette,
        settings["ui.bubble_visual_style"],
        settings["ui.bubble_text_color_mode"],
        border_width=int(settings["ui.bubble_border_width"]),
    )

    width = int(
        _clamp(
            settings["ui.bubble_width"],
            PREVIEW_MIN_WIDTH,
            PREVIEW_MAX_WIDTH,
        )
    )
    height = int(
        _clamp(
            settings["ui.bubble_height"],
            PREVIEW_MIN_HEIGHT,
            PREVIEW_MAX_HEIGHT,
        )
    )
    corner_radius = int(_clamp(settings["ui.bubble_corner_radius"], 0, 64))
    font = bubble_font_spec(
        settings["ui.bubble_font_family"],
        int(settings["ui.bubble_font_size"]),
        settings["ui.bubble_font_weight"],
    )
    opacity = float(_clamp(settings["ui.bubble_opacity"], 0.0, 1.0))

    # ``model.display_text`` already routes through ``compact_overlay_text``
    # via the production helper. The state text is bounded by
    # :data:`PREVIEW_STATE_LABELS`, so direct passthrough is safe.
    display_text = model.display_text
    state_text = model.state_label
    if state_text not in PREVIEW_STATE_LABELS:
        state_text = PREVIEW_STATE_LABELS[0]

    return PreviewRenderSpec(
        width=width,
        height=height,
        corner_radius=corner_radius,
        border_width=int(visual.border_width),
        font=font,
        state_text=state_text,
        display_text=display_text,
        background=visual.background,
        border=visual.border,
        text=visual.text,
        state=visual.state,
        opacity=opacity,
        theme=palette.mode,
        accent=settings["ui.accent_color"],
        visual_style=settings["ui.bubble_visual_style"],
        text_color_mode=settings["ui.bubble_text_color_mode"],
    )


class _CanvasAdapter:
    """Small adapter that exposes only the Canvas methods the widget uses.

    Tests can supply a fake object satisfying this protocol to assert exactly
    which Canvas calls were made and with which arguments.
    """

    def configure(self, **_kwargs: Any) -> Any:  # pragma: no cover - exercised by tests via duck typing
        raise NotImplementedError

    def coords(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def itemconfigure(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def delete(self, *_args: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class PreviewWidget(tk.Frame):
    """Embedded Tk widget that renders a GlitchScribe live bubble preview.

    The widget is a :class:`tk.Frame` and only draws inside its own canvas; it
    never creates a Toplevel, never sets ``-topmost`` or ``-alpha``, never
    touches focus, and never queries the microphone, audio, or network state.
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        registry: Any = None,
        canvas: Optional[Any] = None,
        background: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, **kwargs)
        self._registry = registry
        # ``_render_spec`` is the canonical state of the widget. Tests can
        # inspect it directly to assert what *would* be drawn without spawning
        # a Canvas.
        self._render_spec: Optional[PreviewRenderSpec] = None
        self._background_hint = background
        if canvas is None:
            self.canvas = tk.Canvas(
                self,
                highlightthickness=0,
                borderwidth=0,
                bd=0,
            )
            self.canvas.pack(fill="both", expand=True)
        else:
            self.canvas = canvas
        # Bookkeeping for items we created. We deliberately use plain integer
        # ids returned by Canvas; the real Canvas and a fake adapter both honour
        # this contract.
        self._bg_item: Optional[int] = None
        self._border_item: Optional[int] = None
        self._state_item: Optional[int] = None
        self._text_item: Optional[int] = None

    # ------------------------------------------------------------------ API
    @property
    def render_spec(self) -> Optional[PreviewRenderSpec]:
        """The most recent spec consumed by :meth:`refresh` (or ``None``)."""
        return self._render_spec

    def refresh(self, model: BubblePreviewModel) -> PreviewRenderSpec:
        """Re-render the widget from ``model`` and return the new spec.

        ``refresh`` is the only public mutator the Control Lab needs to call.
        It builds a pure :class:`PreviewRenderSpec` and asks the Canvas to
        update its items, falling back to item creation when first drawn.
        """
        spec = build_preview_render_spec(model)
        self._render_spec = spec
        self._apply_spec(spec)
        return spec

    # ------------------------------------------------------------- Internals
    def _apply_spec(self, spec: PreviewRenderSpec) -> None:
        # Configure canvas size; this is what the parent layout sees.
        self.canvas.configure(
            width=spec.width,
            height=spec.height,
            bg=self._background_hint or spec.background,
            highlightthickness=0,
            borderwidth=0,
            bd=0,
        )
        self.configure(width=spec.width, height=spec.height)

        outer = (
            _PREVIEW_PAD_X // 2,
            _PREVIEW_PAD_Y // 2,
            spec.width - _PREVIEW_PAD_X // 2,
            spec.height - _PREVIEW_PAD_Y // 2,
        )
        if self._bg_item is None:
            self._bg_item = self.canvas.create_rectangle(
                *outer,
                fill=spec.background,
                outline="",
                width=0,
            )
        else:
            self.canvas.coords(self._bg_item, *outer)
            self.canvas.itemconfigure(self._bg_item, fill=spec.background)

        if self._border_item is None:
            if spec.border_width > 0:
                self._border_item = self.canvas.create_rectangle(
                    *outer,
                    fill="",
                    outline=spec.border,
                    width=spec.border_width,
                )
        else:
            self.canvas.coords(self._border_item, *outer)
            self.canvas.itemconfigure(
                self._border_item,
                outline=spec.border,
                width=spec.border_width,
            )

        # State bar sits in the upper-left inset; display text fills the rest.
        text_x = _PREVIEW_PAD_X
        text_y = _PREVIEW_PAD_Y + _STATE_BAR_HEIGHT
        if self._state_item is None:
            self._state_item = self.canvas.create_text(
                text_x,
                _PREVIEW_PAD_Y + 2,
                anchor="nw",
                text=spec.state_text,
                fill=spec.state,
                font=spec.font,
            )
        else:
            self.canvas.coords(self._state_item, text_x, _PREVIEW_PAD_Y + 2)
            self.canvas.itemconfigure(
                self._state_item,
                text=spec.state_text,
                fill=spec.state,
                font=spec.font,
            )

        if self._text_item is None:
            self._text_item = self.canvas.create_text(
                text_x,
                text_y,
                anchor="nw",
                text=spec.display_text,
                fill=spec.text,
                font=spec.font,
                width=max(20, spec.width - 2 * _PREVIEW_PAD_X),
            )
        else:
            self.canvas.coords(self._text_item, text_x, text_y)
            self.canvas.itemconfigure(
                self._text_item,
                text=spec.display_text,
                fill=spec.text,
                font=spec.font,
                width=max(20, spec.width - 2 * _PREVIEW_PAD_X),
            )

    # Convenience for the future Control Lab layout wiring. The widget never
    # invokes these on its own; it is purely a structural contract.
    def preview_settings(self) -> tuple[str, ...]:
        """Return the canonical preview keys handled by this widget."""
        from drew_local_voice.preview_model import preview_setting_keys

        if self._registry is None:  # pragma: no cover - defensive
            return ()
        return preview_setting_keys(self._registry)


__all__ = [
    "PREVIEW_MIN_WIDTH",
    "PREVIEW_MAX_WIDTH",
    "PREVIEW_MIN_HEIGHT",
    "PREVIEW_MAX_HEIGHT",
    "PreviewRenderSpec",
    "PreviewWidget",
    "build_preview_render_spec",
]
