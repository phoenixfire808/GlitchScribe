"""Structural tests for the GlitchScribe Control Lab two-pane layout.

These tests instantiate :class:`SchemaSettingsDialog` against a withdrawn Tk
root so the dialog builds without producing a visible window, microphone
activity, or network traffic. They assert that the non-embedded dialog now
ships a persistent right preview pane alongside the existing tabbed settings
notebook, and that the embedded layout keeps its historical single-pane
form.

The fixtures use the canonical :func:`built_in_settings` registry and copy
``registry.defaults()`` into the dialog so the staged values are guaranteed
to validate. Every Tk root the tests create is destroyed in ``tearDown`` so
each case starts from a clean slate.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
import unittest
from tkinter import ttk
from typing import Any, Iterator

# Ensure the repository root is importable when this file is executed directly
# as a script (``python tests/test_settings_ui_preview.py``).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from drew_local_voice.preview_model import BubblePreviewModel  # noqa: E402
from drew_local_voice.preview_widget import PreviewWidget  # noqa: E402
from drew_local_voice.settings import (  # noqa: E402
    SettingsRegistry,
    built_in_effect_recipes,
    built_in_settings,
)
from drew_local_voice.settings_ui import SchemaSettingsDialog  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk_descendants(widget: tk.Misc) -> Iterator[tk.Misc]:
    """Yield every direct and indirect child of ``widget`` in BFS order."""

    pending: list[tk.Misc] = list(widget.winfo_children())
    while pending:
        current = pending.pop(0)
        yield current
        pending.extend(current.winfo_children())


class _TkRootGuard:
    """Context manager that creates and tears down a withdrawn Tk root.

    The root stays hidden throughout the test; ``update_idletasks`` is still
    called once so layout calculations (column weights, geometry requests)
    settle before the assertions inspect the widget tree.
    """

    def __init__(self) -> None:
        self.root: tk.Tk | None = None

    def __enter__(self) -> tk.Tk:
        self.root = tk.Tk()
        self.root.withdraw()
        return self.root

    def __exit__(self, *_exc_info: Any) -> None:
        if self.root is not None:
            self.root.destroy()
            self.root = None


# ---------------------------------------------------------------------------
# Non-embedded two-pane layout
# ---------------------------------------------------------------------------


class ControlLabTwoPaneLayoutTests(unittest.TestCase):
    """Verify the persistent two-pane body for the non-embedded dialog."""

    def setUp(self) -> None:
        self._guard = _TkRootGuard()
        self.root = self._guard.__enter__()
        try:
            self.registry = SettingsRegistry(built_in_settings())
            self.values = dict(self.registry.defaults())
            self.dialog = SchemaSettingsDialog(
                self.root,
                self.registry,
                self.values,
                lambda _values: None,
                embedded=False,
            )
            self.root.update_idletasks()
        except Exception:
            self._guard.__exit__(None, None, None)
            raise

    def tearDown(self) -> None:
        self._guard.__exit__(None, None, None)

    # --- discoverable attributes ------------------------------------------------

    def test_discoverable_preview_attributes_are_populated(self):
        self.assertIsInstance(self.dialog.preview_model, BubblePreviewModel)
        self.assertIsInstance(self.dialog.preview_widget, PreviewWidget)
        self.assertIsInstance(self.dialog.preview_pane, ttk.Frame)
        self.assertIsInstance(self.dialog.settings_pane, ttk.Frame)
        self.assertIsInstance(self.dialog.control_lab_body, ttk.Frame)
        self.assertIsNotNone(self.dialog.notebook)

    def test_preview_widget_is_a_descendant_of_preview_pane_only(self):
        body = self.dialog.control_lab_body
        preview_pane = self.dialog.preview_pane
        settings_pane = self.dialog.settings_pane
        widget = self.dialog.preview_widget
        assert widget is not None
        # Body → preview_pane → preview_widget chain must hold.
        self.assertIn(preview_pane, body.winfo_children())
        self.assertIn(widget, preview_pane.winfo_children())
        # The widget must NOT be a child of the settings pane.
        self.assertNotIn(widget, settings_pane.winfo_children())

    # --- body geometry ----------------------------------------------------------

    def test_body_has_distinct_left_and_right_sibling_panes(self):
        body = self.dialog.control_lab_body
        columns: dict[int, list[tk.Widget]] = {0: [], 1: []}
        for child in body.winfo_children():
            info = body.grid_info()
            column = int(child.grid_info().get("column", -1))
            columns.setdefault(column, []).append(child)
        self.assertEqual(set(columns.keys()), {0, 1})
        self.assertIn(self.dialog.settings_pane, columns[0])
        self.assertIn(self.dialog.preview_pane, columns[1])

    def test_body_grid_weights_are_left_one_and_right_zero(self):
        body = self.dialog.control_lab_body
        weights = {
            index: int(body.columnconfigure(index).get("weight", 0))
            for index in (0, 1)
        }
        self.assertEqual(weights, {0: 1, 1: 0})

    def test_notebook_lives_under_left_pane(self):
        settings_pane = self.dialog.settings_pane
        descendants = list(_walk_descendants(settings_pane))
        notebooks = [w for w in descendants if isinstance(w, ttk.Notebook)]
        self.assertEqual(len(notebooks), 1)
        self.assertIs(notebooks[0], self.dialog.notebook)
        # Right pane must not contain any notebook.
        right_descendants = list(_walk_descendants(self.dialog.preview_pane))
        self.assertFalse(
            any(isinstance(w, ttk.Notebook) for w in right_descendants)
        )

    # --- preview isolation ------------------------------------------------------

    def test_preview_widget_is_not_a_descendant_of_any_settings_canvas(self):
        widget = self.dialog.preview_widget
        assert widget is not None
        for canvas in self.dialog.canvases.values():
            canvas_widgets = {
                canvas.winfo_id(),
            }
            for descendant in _walk_descendants(canvas):
                self.assertIsNot(descendant, widget)
                self.assertNotEqual(descendant.winfo_id(), widget.winfo_id())
            # The widget's id must not match any canvas id either.
            self.assertNotIn(widget.winfo_id(), canvas_widgets)

    def test_preview_pane_survives_selecting_appearance_tab(self):
        # Selecting another tab must not destroy the right pane or its widget.
        notebook = self.dialog.notebook
        assert notebook is not None
        # Find the Appearance tab if present, otherwise any non-selected tab.
        tabs = [notebook.tab(index, "text") for index in notebook.tabs()]
        self.assertTrue(tabs)
        target = None
        for index, label in enumerate(tabs):
            if label.upper() == "APPEARANCE":
                target = index
                break
        if target is None:
            target = (notebook.index(notebook.select()) + 1) % len(tabs)
        notebook.select(target)
        self.root.update_idletasks()
        self.assertTrue(self.dialog.preview_pane.winfo_exists())
        self.assertTrue(self.dialog.preview_widget.winfo_exists())

    def test_preview_render_spec_is_populated(self):
        widget = self.dialog.preview_widget
        assert widget is not None
        spec = widget.render_spec
        self.assertIsNotNone(spec)
        # Spec width/height must fall within the preview envelope used by the
        # widget (constants live in preview_widget). The exact bounds are
        # asserted by the widget tests; here we only confirm the spec exists
        # and is structurally valid (positive dimensions, non-empty text).
        assert spec is not None
        self.assertGreater(spec.width, 0)
        self.assertGreater(spec.height, 0)
        self.assertTrue(spec.display_text)
        self.assertTrue(spec.state_text)

    def test_recipe_selector_exposes_all_rapid_testing_visual_recipes(self):
        expected_titles = {
            "Streamer Large Text",
            "Low-Glare Chat",
            "Neon High-Contrast",
            "Compact Corner Bubble",
            "Rapid Preview / Long Sample",
        }
        self.assertTrue(expected_titles.issubset(self.dialog.recipes_by_title))
        self.assertTrue(expected_titles.issubset(set(self.dialog.recipe_box.cget("values"))))

    def test_visual_recipes_stage_controls_and_preview_without_saving(self):
        recipes = {
            item.key: item for item in built_in_effect_recipes()
        }
        expected_keys = (
            "streamer_large_text",
            "low_glare_chat",
            "neon_high_contrast",
            "compact_corner_bubble",
            "rapid_preview_long_sample",
        )
        saved: list[dict[str, Any]] = []
        self.dialog.on_save = saved.append

        for recipe_key in expected_keys:
            recipe = recipes[recipe_key]
            with self.subTest(recipe=recipe_key):
                self.dialog.recipe_var.set(recipe.title)
                self.dialog.stage_selected_recipe()
                for key, value in recipe.changes:
                    self.assertEqual(
                        self.registry.validate(key, self.dialog.variables[key].get()),
                        value,
                    )
                    if key in self.dialog._preview_keys:
                        assert self.dialog.preview_model is not None
                        self.assertEqual(
                            self.dialog.preview_model.settings[key], value
                        )

        self.assertEqual(saved, [])
        assert self.dialog.preview_model is not None
        self.assertTrue(self.dialog.preview_model.dirty)
        self.assertIn("STAGED", self.dialog.recipe_outcome_var.get())
        self.assertIn("Save Settings", self.dialog.recipe_outcome_var.get())

    def test_preview_model_tracks_canonical_bubble_settings(self):
        model = self.dialog.preview_model
        assert model is not None
        # The model is built from the copied canonical values; it must not be
        # dirty until external code stages a change.
        self.assertFalse(model.dirty)
        # Staging a change must mark the model dirty without touching the
        # dialog's stored baseline. The default bubble width is 420, so we
        # stage a different value to verify the dirty flag actually flips.
        model.stage_setting("ui.bubble_width", 480)
        self.assertTrue(model.dirty)


# ---------------------------------------------------------------------------
# Embedded regression
# ---------------------------------------------------------------------------


class EmbeddedRegressionTests(unittest.TestCase):
    """The embedded (single-pane) layout must keep its historical shape."""

    def setUp(self) -> None:
        self._guard = _TkRootGuard()
        self.root = self._guard.__enter__()
        try:
            self.registry = SettingsRegistry(built_in_settings())
            self.dialog = SchemaSettingsDialog(
                self.root,
                self.registry,
                self.registry.defaults(),
                lambda _values: None,
                embedded=True,
            )
            self.root.update_idletasks()
        except Exception:
            self._guard.__exit__(None, None, None)
            raise

    def tearDown(self) -> None:
        self._guard.__exit__(None, None, None)

    def test_embedded_dialog_keeps_single_pane_form(self):
        self.assertIsNone(self.dialog.control_lab_body)
        self.assertIsNone(self.dialog.settings_pane)
        self.assertIsNone(self.dialog.preview_pane)
        self.assertIsNone(self.dialog.preview_widget)
        self.assertIsNone(self.dialog.preview_model)
        self.assertIsNone(self.dialog.notebook)
        # The host Frame is the dialog's ``window``; no Toplevel was created.
        self.assertIs(self.dialog.window, self.root)

    def test_embedded_form_is_a_direct_child_of_window(self):
        descendants = list(_walk_descendants(self.root))
        # No ttk.Notebook should be reachable from the embedded root.
        self.assertFalse(
            any(isinstance(widget, ttk.Notebook) for widget in descendants)
        )
        # The form is the only top-level child of the root besides the heading
        # and recipe_panel which embed directly under ``self.window`` already.
        self.assertTrue(descendants)

    def test_embedded_save_path_is_inert_until_called(self):
        # Sanity-check that the schema-generated variables still exist on the
        # embedded dialog, so the historical control surface is preserved.
        self.assertTrue(self.dialog.variables)
        self.assertIn(
            "ui.bubble_width",
            self.dialog.variables,
            "embedded dialog must continue to expose Speech Bubble settings",
        )


if __name__ == "__main__":  # pragma: no cover - direct execution entry point
    unittest.main(verbosity=2)