"""Tests for the dictation-overlay pulse animation and CLI overrides.

These tests follow the AGENTS.md rule that synthetic Tk/preview tests must
not exercise the microphone, audio device, or focus. We construct a real
``DictationOverlay`` against a hidden ``tk.Tk()`` root, drive it through
``show_idle`` / ``show`` / ``error`` / ``finalizing`` / ``hide``, and assert
that the pulse animation bookkeeping behaves correctly without ever moving
real focus or scheduling actual audio work.
"""

from __future__ import annotations

import os
import tkinter as tk
import unittest

from drew_local_voice.doctor import parse_cli_args, parse_cli_overrides


def _require_tk():
    if os.environ.get("GLSCRIBE_TEST_NO_TK") == "1":
        raise unittest.SkipTest("Tk disabled via GLSCRIBE_TEST_NO_TK")
    try:
        root = tk.Tk()
        root.withdraw()
        return root
    except tk.TclError as exc:  # pragma: no cover - depends on host display
        raise unittest.SkipTest(f"Tk unavailable: {exc}") from exc


class CliOverridesTests(unittest.TestCase):
    """``parse_cli_overrides`` is a pure function: no Tk required."""

    def test_empty_argv_returns_empty(self):
        self.assertEqual(parse_cli_overrides([]), {})

    def test_unknown_flag_is_silently_ignored(self):
        self.assertEqual(parse_cli_overrides(["--garbage", "x"]), {})

    def test_known_flags_are_returned_with_canonical_keys(self):
        result = parse_cli_overrides(
            [
                "--bubble-style", "neon_edge",
                "--theme", "dark",
                "--accent", "cyan",
                "--opacity", "0.9",
            ]
        )
        self.assertEqual(
            result,
            {
                "ui.bubble_visual_style": "neon_edge",
                "ui.theme": "dark",
                "ui.accent_color": "cyan",
                "ui.bubble_opacity": "0.900",
            },
        )

    def test_invalid_values_are_dropped_not_raised(self):
        # An unknown style must not crash and must not appear in overrides.
        result = parse_cli_overrides(
            ["--bubble-style", "plasma", "--opacity", "2.0", "--opacity", "0.5"]
        )
        self.assertNotIn("ui.bubble_visual_style", result)
        self.assertEqual(result["ui.bubble_opacity"], "0.500")

    def test_case_insensitive_flag_match(self):
        result = parse_cli_overrides(["--Bubble-Style", "Minimal"])
        self.assertEqual(result["ui.bubble_visual_style"], "minimal")

    def test_missing_value_after_flag_is_ignored(self):
        # --bubble-style with nothing after it must not crash or partial-write.
        self.assertEqual(parse_cli_overrides(["--bubble-style"]), {})

    def test_existing_parse_cli_args_contract_is_unchanged(self):
        # Defensive: existing call sites unpack a 3-tuple. Make sure the
        # new helper didn't shadow or break the old one.
        want_doctor, want_selftest, consumed = parse_cli_args(
            ["doctor", "--selftest"]
        )
        self.assertTrue(want_doctor)
        self.assertTrue(want_selftest)
        self.assertTrue(consumed)


class DictationOverlayPulseTests(unittest.TestCase):
    """Pulse is a UI-only animation; verify it starts, stops, and is idempotent."""

    def setUp(self):
        self.root = _require_tk()
        # Lazy import so unit-test discovery on machines without Tk still works
        # (the CliOverridesTests above don't need Tk).
        from dictation_overlay import DictationOverlay

        self.overlay = DictationOverlay(self.root, on_toggle=lambda: None)

    def tearDown(self):
        try:
            self.overlay.destroy()
        except tk.TclError:
            pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_pulse_does_not_start_during_init(self):
        # Headless tests must not schedule timers.
        self.assertIsNone(self.overlay._pulse_after_id)

    def test_show_starts_pulse(self):
        self.overlay.show(target_window=0)
        self.assertTrue(self.overlay._recording)
        self.assertIsNotNone(self.overlay._pulse_after_id)
        # Tearing down must cancel the pending after() so the test process
        # does not leak live timers.
        self.overlay.hide()
        self.assertIsNone(self.overlay._pulse_after_id)

    def test_show_idle_stops_pulse(self):
        self.overlay.show(target_window=0)
        self.assertIsNotNone(self.overlay._pulse_after_id)
        self.overlay.show_idle()
        self.assertFalse(self.overlay._recording)
        self.assertIsNone(self.overlay._pulse_after_id)

    def test_finalizing_stops_pulse(self):
        self.overlay.show(target_window=0)
        self.assertIsNotNone(self.overlay._pulse_after_id)
        self.overlay.finalizing()
        self.assertIsNone(self.overlay._pulse_after_id)

    def test_error_stops_pulse(self):
        self.overlay.show(target_window=0)
        self.assertIsNotNone(self.overlay._pulse_after_id)
        self.overlay.error("mic disconnected")
        self.assertIsNone(self.overlay._pulse_after_id)

    def test_double_start_is_idempotent(self):
        # Calling start twice must not schedule two parallel timers.
        self.overlay.show(target_window=0)
        first_id = self.overlay._pulse_after_id
        self.overlay._start_pulse()
        second_id = self.overlay._pulse_after_id
        self.assertEqual(first_id, second_id)
        self.overlay.hide()

    def test_destroy_cancels_pulse(self):
        self.overlay.show(target_window=0)
        # destroy() must be safe to call while the pulse is running.
        self.overlay.destroy()
        self.assertIsNone(self.overlay._pulse_after_id)


if __name__ == "__main__":
    unittest.main()