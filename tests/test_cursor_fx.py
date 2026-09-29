from __future__ import annotations

import unittest

from drew_local_voice.cursor_fx import (
    CURSOR_STYLE_CHOICES,
    POWER_RING_STYLE,
    SOUND_PROFILE_CHOICES,
    SOUND_PROFILES,
    clamp_intensity,
    sound_profile,
)
from drew_local_voice.sound_cues import (
    tones_for_event,
    normalize_sound_profile,
)
from drew_local_voice.settings import built_in_settings


class CursorFxCatalogTests(unittest.TestCase):
    def test_has_multiple_unbranded_sound_profiles(self):
        self.assertGreaterEqual(len(SOUND_PROFILE_CHOICES), 6)
        self.assertIn("silent", SOUND_PROFILES)
        self.assertEqual(sound_profile("missing").key, "pulse")
        for profile in SOUND_PROFILES.values():
            self.assertNotIn("glitchscribe", profile.label.casefold())
            self.assertNotIn("glitchscribe", profile.description.casefold())

    def test_has_multiple_cursor_styles(self):
        self.assertEqual(
            set(CURSOR_STYLE_CHOICES),
            {"orbit", "ripple", "comet", "spectrum", "minimal"},
        )
        self.assertEqual(POWER_RING_STYLE, "power")

    def test_intensity_is_bounded(self):
        self.assertEqual(clamp_intensity(-10), 0.35)
        self.assertEqual(clamp_intensity(4), 1.5)
        self.assertEqual(clamp_intensity("bad"), 1.0)

    def test_profiles_include_stop_cues_for_optional_stop_feedback(self):
        self.assertTrue(any(sound_profile(key).stop for key in SOUND_PROFILE_CHOICES if key != "silent"))
        self.assertFalse(sound_profile("silent").stop)

    def test_sound_and_ring_events_are_independently_configurable(self):
        specs = {spec.key: spec for spec in built_in_settings()}
        for key in (
            "ui.sound_start",
            "ui.sound_stop",
            "ui.sound_complete",
            "ui.sound_error",
            "ui.sound_preview",
            "ui.ring_start",
            "ui.ring_stop",
            "ui.ring_complete",
            "ui.ring_error",
            "ui.ring_preview",
        ):
            self.assertIn(key, specs)
            self.assertEqual(specs[key].kind.value, "boolean")
        self.assertFalse(specs["ui.sound_stop"].default)
        self.assertFalse(specs["ui.ring_stop"].default)

    def test_compatibility_facade_uses_active_catalog(self):
        self.assertEqual(normalize_sound_profile("soft"), "soft")
        self.assertEqual(
            len(tones_for_event("soft", "stop")),
            len(SOUND_PROFILES["soft"].stop),
        )

    def test_catalog_entries_remain_data_only(self):
        for profile in SOUND_PROFILES.values():
            self.assertIsInstance(profile.start, tuple)
            self.assertIsInstance(profile.stop, tuple)
            self.assertIsInstance(profile.complete, tuple)
            self.assertIsInstance(profile.error, tuple)
        self.assertEqual(sound_profile("soft").key, "soft")


if __name__ == "__main__":
    unittest.main()
