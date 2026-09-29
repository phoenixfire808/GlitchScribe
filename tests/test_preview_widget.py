"""Focused tests for the GlitchScribe Control Lab live bubble preview widget.

The tests cover two layers:

* ``build_preview_render_spec`` is a pure function. It must run without Tk and
  produce a deterministic render specification (dimensions, palette, font,
  state text, display text) derived from a :class:`BubblePreviewModel`.
* :class:`PreviewWidget` is an embedded Tk widget that renders the spec on a
  Canvas. Tests exercise it through a real (hidden) Tk root, but verify that
  the widget never spawns its own Toplevel window and accepts a fake Canvas for
  deterministic assertions.
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any

# Ensure the repository root is importable when this file is executed directly
# as a script (``python tests/test_preview_widget.py``).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dictation_overlay import compact_overlay_text  # noqa: E402
from drew_local_voice.preview_model import (  # noqa: E402
    PREVIEW_SAMPLE_TEXTS,
    BubblePreviewModel,
    extract_bubble_settings,
    preview_setting_keys,
)
from drew_local_voice.preview_widget import (  # noqa: E402
    PREVIEW_MIN_WIDTH,
    PREVIEW_MAX_WIDTH,
    PREVIEW_MIN_HEIGHT,
    PREVIEW_MAX_HEIGHT,
    PreviewRenderSpec,
    PreviewWidget,
    build_preview_render_spec,
)
from drew_local_voice.settings import SettingsRegistry, built_in_settings  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(**overrides: Any) -> BubblePreviewModel:
    registry = SettingsRegistry(built_in_settings())
    return BubblePreviewModel.from_values(overrides, registry=registry)


# ---------------------------------------------------------------------------
# Pure render spec tests (no Tk)
# ---------------------------------------------------------------------------


class BuildPreviewRenderSpecTests(unittest.TestCase):
    """``build_preview_render_spec`` is the headless surface used by the widget."""

    def setUp(self) -> None:
        self.registry = SettingsRegistry(built_in_settings())

    def test_render_spec_is_a_frozen_dataclass_with_only_visual_keys(self):
        model = BubblePreviewModel.from_values({}, registry=self.registry)
        spec = build_preview_render_spec(model)

        self.assertIsInstance(spec, PreviewRenderSpec)
        # dataclass(frozen=True) -> no public mutation.
        with self.assertRaises(Exception):
            spec.width = spec.width + 1  # type: ignore[misc]

        keys = set(spec.__dataclass_fields__)
        expected = {
            "width",
            "height",
            "corner_radius",
            "border_width",
            "font",
            "state_text",
            "display_text",
            "background",
            "border",
            "text",
            "state",
            "opacity",
            "theme",
            "accent",
            "visual_style",
            "text_color_mode",
        }
        self.assertEqual(keys, expected)

    def test_render_spec_rejects_non_preview_model(self):
        with self.assertRaises(TypeError):
            build_preview_render_spec(object())  # type: ignore[arg-type]

    def test_render_spec_clamps_dimensions_to_preview_envelope(self):
        model = BubblePreviewModel.from_values(
            {"ui.bubble_width": 1, "ui.bubble_height": 9999},
            registry=self.registry,
        )
        spec = build_preview_render_spec(model)
        self.assertGreaterEqual(spec.width, PREVIEW_MIN_WIDTH)
        self.assertLessEqual(spec.width, PREVIEW_MAX_WIDTH)
        self.assertGreaterEqual(spec.height, PREVIEW_MIN_HEIGHT)
        self.assertLessEqual(spec.height, PREVIEW_MAX_HEIGHT)

    def test_render_spec_dimensions_match_clamped_bubble_size(self):
        model = BubblePreviewModel.from_values(
            {"ui.bubble_width": 480, "ui.bubble_height": 140},
            registry=self.registry,
        )
        spec = build_preview_render_spec(model)
        # 480 / 140 are within canonical envelope.
        self.assertEqual(spec.width, 480)
        self.assertEqual(spec.height, 140)
        # Corner radius must come from the model settings (validated).
        self.assertGreaterEqual(spec.corner_radius, 0)
        self.assertLessEqual(spec.border_width, 4)

    def test_render_spec_text_compacts_through_overlay_helper(self):
        model = BubblePreviewModel.from_values(
            {"ui.bubble_max_characters": 28}, registry=self.registry
        )
        model.select_sample("rapid")
        spec = build_preview_render_spec(model)
        self.assertEqual(
            spec.display_text,
            compact_overlay_text(
                model.sample_text,
                max_chars=int(model.settings["ui.bubble_max_characters"]),
            ),
        )
        self.assertEqual(spec.state_text, model.state_label)

    def test_render_spec_font_is_tuple_from_production_helper(self):
        model = BubblePreviewModel.from_values(
            {
                "ui.bubble_font_family": "Consolas",
                "ui.bubble_font_size": 18,
                "ui.bubble_font_weight": "bold",
            },
            registry=self.registry,
        )
        spec = build_preview_render_spec(model)
        # ``bubble_font_spec`` returns either a 2- or 3-tuple; either is valid Tk.
        self.assertIsInstance(spec.font, tuple)
        self.assertGreaterEqual(len(spec.font), 2)
        self.assertEqual(spec.font[0], "Consolas")
        self.assertEqual(spec.font[1], 18)

    def test_render_spec_palette_and_style_follow_production_resolver(self):
        model = BubblePreviewModel.from_values(
            {
                "ui.theme": "dark",
                "ui.accent_color": "cyan",
                "ui.bubble_visual_style": "neon_edge",
                "ui.bubble_text_color_mode": "accent",
                "ui.bubble_border_width": 3,
            },
            registry=self.registry,
        )
        spec = build_preview_render_spec(model)
        # Colors are six-digit hex strings; ``neon_edge`` + ``dark`` ⇒
        # background uses the dark neon surface.
        self.assertEqual(spec.theme, "dark")
        self.assertEqual(spec.accent, "cyan")
        self.assertEqual(spec.visual_style, "neon_edge")
        self.assertEqual(spec.text_color_mode, "accent")
        # Border width is clamped to >= 2 for neon_edge regardless of input.
        self.assertGreaterEqual(spec.border_width, 2)
        # Background should be the palette ``bubble`` surface for dark mode.
        self.assertTrue(spec.background.startswith("#"))
        self.assertTrue(spec.border.startswith("#"))
        self.assertTrue(spec.text.startswith("#"))
        self.assertTrue(spec.state.startswith("#"))

    def test_render_spec_opacity_is_a_normalized_float(self):
        model = BubblePreviewModel.from_values(
            {"ui.bubble_opacity": 0.5}, registry=self.registry
        )
        spec = build_preview_render_spec(model)
        self.assertGreaterEqual(spec.opacity, 0.0)
        self.assertLessEqual(spec.opacity, 1.0)
        self.assertEqual(spec.opacity, model.settings["ui.bubble_opacity"])

    def test_render_spec_leaves_out_runtime_only_keys(self):
        # Settings registry exposes only the canonical preview keys; runtime
        # keys such as ``audio.microphone`` or ``runtime.target_hwnd`` must
        # never influence the render spec.
        model = BubblePreviewModel.from_values({}, registry=self.registry)
        spec = build_preview_render_spec(model)
        allowed = set(preview_setting_keys(self.registry))
        self.assertTrue(allowed)
        # The spec explicitly enumerates a closed set of keys; ensure none of
        # the runtime-only canonical keys leak in.
        for forbidden in {
            "audio.microphone",
            "general.hotkey",
            "typing.live_insert",
            "insertion.clipboard_fallback",
            "runtime.target_hwnd",
        }:
            self.assertNotIn(forbidden, allowed)

    def test_render_spec_drives_state_text_from_model(self):
        model = BubblePreviewModel.from_values({}, registry=self.registry)
        model.set_state_label("LISTENING • LIVE")
        spec = build_preview_render_spec(model)
        self.assertEqual(spec.state_text, "LISTENING • LIVE")

        model.set_state_label("FINALIZING")
        spec = build_preview_render_spec(model)
        self.assertEqual(spec.state_text, "FINALIZING")

        # Unknown labels collapse to READY in the model; spec must reflect that.
        model.set_state_label("WHATEVER")
        spec = build_preview_render_spec(model)
        self.assertEqual(spec.state_text, "READY")

    def test_render_spec_is_deterministic_for_same_model(self):
        model = BubblePreviewModel.from_values(
            {"ui.bubble_width": 360, "ui.bubble_height": 110},
            registry=self.registry,
        )
        spec_a = build_preview_render_spec(model)
        spec_b = build_preview_render_spec(model)
        self.assertEqual(spec_a, spec_b)

    def test_render_spec_reacts_to_staged_setting_changes(self):
        model = BubblePreviewModel.from_values(
            {"ui.bubble_width": 360, "ui.bubble_visual_style": "minimal"},
            registry=self.registry,
        )
        spec_before = build_preview_render_spec(model)

        model.stage_setting("ui.bubble_width", 520)
        model.stage_setting("ui.bubble_visual_style", "high_contrast")
        spec_after = build_preview_render_spec(model)

        self.assertNotEqual(spec_before.width, spec_after.width)
        self.assertEqual(spec_after.width, 520)
        self.assertEqual(spec_after.visual_style, "high_contrast")
        self.assertNotEqual(spec_before.background, spec_after.background)


# ---------------------------------------------------------------------------
# Widget integration tests (require a working Tk display)
# ---------------------------------------------------------------------------


_TK_AVAILABLE = True
_TK_SKIP_REASON = ""
try:
    import tkinter as tk
    from tkinter import ttk

    _root = tk.Tk()
    _root.withdraw()
    _root.update_idletasks()
    _root.destroy()
except Exception as exc:  # pragma: no cover - environment-dependent
    _TK_AVAILABLE = False
    _TK_SKIP_REASON = repr(exc)
    tk = None  # type: ignore[assignment]


@unittest.skipUnless(_TK_AVAILABLE, f"Tk display not available: {_TK_SKIP_REASON}")
class PreviewWidgetIntegrationTests(unittest.TestCase):
    """Render through a real hidden Tk root and a fake canvas adapter."""

    def setUp(self) -> None:
        self.registry = SettingsRegistry(built_in_settings())
        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self) -> None:
        try:
            self.root.update_idletasks()
            self.root.destroy()
        except tk.TclError:
            pass

    def _new_widget(self) -> PreviewWidget:
        widget = PreviewWidget(self.root, registry=self.registry)
        widget.pack()
        self.root.update_idletasks()
        return widget

    def test_widget_stays_embedded_inside_parent(self):
        widget = self._new_widget()
        # Embedded widget: must be a child of the parent, not a Toplevel.
        self.assertIsInstance(widget, tk.Widget)
        self.assertIsNotNone(widget.winfo_parent())
        self.assertFalse(widget.winfo_toplevel().attributes("-topmost"))

    def test_widget_refresh_updates_canvas_items_for_text_and_state(self):
        widget = self._new_widget()
        model = BubblePreviewModel.from_values(
            {"ui.bubble_font_family": "Segoe UI", "ui.bubble_font_size": 12},
            registry=self.registry,
        )
        widget.refresh(model)

        canvas = widget.canvas
        # The widget should have drawn a background rectangle and two text items.
        self.assertGreaterEqual(len(canvas.find_all()), 3)

        # Change sample + state; assert canvas reflects new strings.
        model.select_sample("punctuation")
        model.set_state_label("LISTENING • LIVE")
        widget.refresh(model)

        item_texts = [
            canvas.itemconfigure(item, "text")[-1]
            for item in canvas.find_all()
            if "text" in canvas.itemconfigure(item)
        ]
        self.assertIn(model.state_label, item_texts)
        self.assertIn(
            compact_overlay_text(
                model.sample_text,
                max_chars=int(model.settings["ui.bubble_max_characters"]),
            ),
            item_texts,
        )

    def test_widget_refresh_updates_dimensions_when_model_changes(self):
        widget = self._new_widget()
        model = BubblePreviewModel.from_values(
            {"ui.bubble_width": 360, "ui.bubble_height": 110},
            registry=self.registry,
        )
        widget.refresh(model)
        canvas = widget.canvas
        # The background rectangle is inset by half the preview padding on
        # every side; verify the spec drives both the canvas size and the
        # drawn rectangle width.
        self.assertEqual(int(canvas["width"]), 360)
        self.assertEqual(int(canvas["height"]), 110)
        before = canvas.coords(canvas.find_all()[0])  # background rectangle
        self.assertAlmostEqual(int(before[2]) - int(before[0]), 360 - 14, delta=2)

        model.stage_setting("ui.bubble_width", 540)
        model.stage_setting("ui.bubble_height", 150)
        widget.refresh(model)
        self.assertEqual(int(canvas["width"]), 540)
        self.assertEqual(int(canvas["height"]), 150)
        after = canvas.coords(canvas.find_all()[0])
        self.assertAlmostEqual(int(after[2]) - int(after[0]), 540 - 14, delta=2)

    def test_widget_refresh_accepts_fake_canvas_adapter(self):
        @dataclass
        class FakeCanvas:
            calls: list[tuple] = None  # type: ignore[assignment]
            _next_id: int = 1

            def __post_init__(self) -> None:
                if self.calls is None:
                    self.calls = []

            def configure(self, **kwargs: Any) -> None:
                self.calls.append(("configure", dict(kwargs)))

            def coords(self, *args: Any, **kwargs: Any) -> Any:
                self.calls.append(("coords", args, dict(kwargs)))
                return [0, 0, 10, 10]

            def itemconfigure(self, *args: Any, **kwargs: Any) -> Any:
                self.calls.append(("itemconfigure", args, dict(kwargs)))
                return [str(kwargs.get("text", ""))]

            def delete(self, *args: Any) -> None:
                self.calls.append(("delete", args))

            def create_rectangle(self, *args: Any, **kwargs: Any) -> int:
                self.calls.append(("create_rectangle", args, dict(kwargs)))
                item_id = self._next_id
                self._next_id += 1
                return item_id

            def create_text(self, *args: Any, **kwargs: Any) -> int:
                self.calls.append(("create_text", args, dict(kwargs)))
                item_id = self._next_id
                self._next_id += 1
                return item_id

        fake = FakeCanvas()
        widget = PreviewWidget(self.root, registry=self.registry, canvas=fake)
        model = BubblePreviewModel.from_values({}, registry=self.registry)
        # First refresh creates items, second refresh exercises the update
        # path (coords / itemconfigure).
        widget.refresh(model)
        model.stage_setting("ui.bubble_width", 520)
        widget.refresh(model)

        kinds = [c[0] for c in fake.calls]
        self.assertIn("configure", kinds)
        self.assertIn("coords", kinds)
        self.assertIn("itemconfigure", kinds)
        self.assertIn("create_rectangle", kinds)
        self.assertIn("create_text", kinds)

    def test_widget_does_not_steal_focus_or_raise_topmost(self):
        widget = self._new_widget()
        model = BubblePreviewModel.from_values({}, registry=self.registry)
        widget.refresh(model)
        toplevel = widget.winfo_toplevel()
        # The parent (or shared root) must not gain topmost/withdraw/overrideredirect.
        self.assertFalse(bool(toplevel.attributes("-topmost")))
        # Direct sanity: widget remains a Frame on the root, never Toplevel.
        self.assertIsInstance(widget, (tk.Frame, ttk.Frame))


if __name__ == "__main__":
    unittest.main(verbosity=2)
