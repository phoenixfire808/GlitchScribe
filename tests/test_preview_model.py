from __future__ import annotations

import unittest

from dictation_overlay import compact_overlay_text
from drew_local_voice.preview_model import (
    BubblePreviewModel,
    PREVIEW_SAMPLE_TEXTS,
    PREVIEW_STATE_LABELS,
    choose_preview_sample,
    extract_bubble_settings,
    preview_display_text,
)
from drew_local_voice.settings import SettingsRegistry, built_in_settings


class PreviewModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = SettingsRegistry(built_in_settings())

    def test_extracts_only_canonical_bubble_and_palette_settings_without_mutation(self):
        values = {
            "ui.theme": "light",
            "ui.accent_color": "cyan",
            "ui.bubble_width": 510,
            "ui.bubble_font_family": "Consolas",
            "audio.microphone": "private microphone",
            "general.hotkey": "Ctrl + Shift + Space",
            "typing.live_insert": True,
            "insertion.clipboard_fallback": False,
            "runtime.target_hwnd": 1234,
        }
        original = dict(values)

        extracted = extract_bubble_settings(values, self.registry)

        self.assertEqual(values, original)
        self.assertEqual(extracted["ui.theme"], "light")
        self.assertEqual(extracted["ui.accent_color"], "cyan")
        self.assertEqual(extracted["ui.bubble_width"], 510.0)
        self.assertEqual(extracted["ui.bubble_font_family"], "Consolas")
        self.assertNotIn("audio.microphone", extracted)
        self.assertNotIn("general.hotkey", extracted)
        self.assertNotIn("typing.live_insert", extracted)
        self.assertNotIn("insertion.clipboard_fallback", extracted)
        self.assertNotIn("runtime.target_hwnd", extracted)

    def test_missing_and_invalid_values_fail_closed_to_registry_defaults(self):
        values = {
            "ui.bubble_width": 9_999,
            "ui.bubble_opacity": "opaque",
            "ui.bubble_font_family": "Unregistered Font",
            "ui.theme": None,
        }

        extracted = extract_bubble_settings(values, self.registry)

        for key in values:
            self.assertEqual(extracted[key], self.registry.get(key).default)
        self.assertEqual(
            extracted["ui.bubble_height"],
            self.registry.get("ui.bubble_height").default,
        )

    def test_preview_helpers_use_bounded_synthetic_samples_and_overlay_compaction(self):
        self.assertEqual(choose_preview_sample("short"), PREVIEW_SAMPLE_TEXTS["short"])
        self.assertEqual(choose_preview_sample("unknown"), PREVIEW_SAMPLE_TEXTS["short"])
        self.assertLessEqual(len(PREVIEW_SAMPLE_TEXTS), 8)
        self.assertTrue(all(PREVIEW_SAMPLE_TEXTS.values()))

        sample = PREVIEW_SAMPLE_TEXTS["rapid"]
        self.assertEqual(
            preview_display_text(sample, 40),
            compact_overlay_text(sample, max_chars=40),
        )

    def test_model_holds_copied_validated_staged_state_and_tracks_dirty(self):
        values = {"ui.bubble_width": 500, "audio.microphone": "private microphone"}

        model = BubblePreviewModel.from_values(values, registry=self.registry)
        values["ui.bubble_width"] = 300

        self.assertEqual(model.settings["ui.bubble_width"], 500.0)
        self.assertEqual(model.sample_key, "short")
        self.assertEqual(model.sample_text, PREVIEW_SAMPLE_TEXTS["short"])
        self.assertEqual(model.state_label, "READY")
        self.assertFalse(model.dirty)
        self.assertNotIn("audio.microphone", model.settings)

        model.stage_setting("ui.bubble_width", 620)
        self.assertEqual(model.settings["ui.bubble_width"], 620.0)
        self.assertTrue(model.dirty)

    def test_model_rejects_non_preview_keys_and_fails_closed_on_invalid_updates(self):
        model = BubblePreviewModel.from_values({}, registry=self.registry)

        with self.assertRaises(KeyError):
            model.stage_setting("audio.microphone", "private microphone")

        model.stage_setting("ui.bubble_opacity", 99)
        self.assertEqual(
            model.settings["ui.bubble_opacity"],
            self.registry.get("ui.bubble_opacity").default,
        )
        self.assertFalse(model.dirty)

    def test_model_bounds_state_sample_and_custom_text_and_derives_display_text(self):
        model = BubblePreviewModel.from_values({}, registry=self.registry)

        model.select_sample("rapid")
        model.set_state_label("LISTENING • LIVE")
        self.assertEqual(model.sample_text, PREVIEW_SAMPLE_TEXTS["rapid"])
        self.assertEqual(model.state_label, "LISTENING • LIVE")
        self.assertTrue(model.dirty)

        model.set_state_label("UNBOUNDED RUNTIME STATE")
        self.assertEqual(model.state_label, "READY")

        oversized = "x" * 5_000
        model.set_sample_text(oversized)
        self.assertLessEqual(len(model.sample_text), model.MAX_SAMPLE_CHARACTERS)
        self.assertEqual(
            model.display_text,
            compact_overlay_text(
                model.sample_text,
                max_chars=int(model.settings["ui.bubble_max_characters"]),
            ),
        )
        self.assertEqual(
            PREVIEW_STATE_LABELS,
            ("READY", "LISTENING • LIVE", "FINALIZING", "ERROR"),
        )


if __name__ == "__main__":
    unittest.main()
