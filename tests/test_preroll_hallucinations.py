from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from drew_local_voice.audio import (
    AmbientPreRollCapture,
    PreRollRingBuffer,
    prepend_preroll,
    samples_for_duration_ms,
)
from drew_local_voice.hallucinations import (
    DEFAULT_HALLUCINATION_BLOCKLIST,
    filter_hallucinations,
)
from drew_local_voice.settings import built_in_settings
from local_voice import LocalVoiceApp


class PreRollBufferTests(unittest.TestCase):
    def test_samples_for_duration_math(self):
        self.assertEqual(samples_for_duration_ms(400, 16_000), 6_400)
        self.assertEqual(samples_for_duration_ms(0, 16_000), 0)
        self.assertEqual(samples_for_duration_ms(500, 48_000), 24_000)

    def test_ring_trims_to_capacity_and_keeps_newest(self):
        ring = PreRollRingBuffer(capacity_samples=8, sample_rate=16_000)
        ring.write(np.arange(5, dtype=np.float32))
        ring.write(np.arange(10, 20, dtype=np.float32))
        snap = ring.snapshot()
        self.assertEqual(snap.size, 8)
        self.assertTrue(np.allclose(snap, np.arange(12, 20, dtype=np.float32)))
        taken = ring.take()
        self.assertEqual(taken.size, 8)
        self.assertEqual(ring.stored_samples, 0)

    def test_from_preroll_ms_capacity(self):
        ring = PreRollRingBuffer.from_preroll_ms(400, sample_rate=16_000)
        self.assertEqual(ring.capacity_samples, 6_400)

    def test_prepend_preroll_concatenates_head_then_body(self):
        head = np.array([1.0, 2.0], dtype=np.float32)
        body = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        out = prepend_preroll(body, head)
        self.assertTrue(np.allclose(out, [1.0, 2.0, 3.0, 4.0, 5.0]))
        clipped = prepend_preroll(body, head, max_preroll_samples=1)
        self.assertTrue(np.allclose(clipped, [2.0, 3.0, 4.0, 5.0]))


class AmbientPreRollCaptureHandoffTests(unittest.TestCase):
    """Guards against the same-device close-then-reopen WASAPI stall.

    Regression: closing the ambient pre-roll stream and immediately opening
    a fresh session AudioRecorder stream on the same physical device made
    WASAPI renegotiate the shared endpoint, which was observed to stall for
    several seconds (worst case ~19s on Voicemeeter) before the microphone
    actually started -- the "hotkey pressed, nothing happens" complaint.
    peek_for_session() must read the buffered pre-roll without stopping the
    stream, so a caller can open the new stream first and only stop the
    ambient one afterward.
    """

    def _capture(self):
        capture = AmbientPreRollCapture(
            device_index=7,
            preroll_ms=400.0,
            sample_rate=16_000,
            capture_rate=16_000,
        )
        # Populate the ring directly; unit tests do not open a real stream.
        capture._ring.write(np.linspace(-1.0, 1.0, num=800, dtype=np.float32))
        capture._running = True
        capture._stream = object()
        return capture

    def test_peek_does_not_stop_the_stream(self):
        capture = self._capture()
        seed = capture.peek_for_session(16_000)
        self.assertEqual(seed.size, 800)
        self.assertTrue(capture.is_running)
        self.assertIsNotNone(capture._stream)

    def test_peek_is_idempotent_until_take(self):
        capture = self._capture()
        first = capture.peek_for_session(16_000)
        second = capture.peek_for_session(16_000)
        self.assertTrue(np.allclose(first, second))
        self.assertTrue(capture.is_running)

    def test_take_after_peek_still_stops_and_clears(self):
        capture = self._capture()
        capture.peek_for_session(16_000)
        taken = capture.take_for_session(16_000)
        self.assertEqual(taken.size, 800)
        self.assertFalse(capture.is_running)
        self.assertEqual(capture._ring.stored_samples, 0)

    def test_peek_resamples_like_take(self):
        capture = AmbientPreRollCapture(
            device_index=7,
            preroll_ms=400.0,
            sample_rate=16_000,
            capture_rate=48_000,
        )
        capture._ring.write(np.linspace(-1.0, 1.0, num=2400, dtype=np.float32))
        capture._running = True
        capture._stream = object()
        peeked = capture.peek_for_session(16_000)
        self.assertEqual(peeked.size, 800)


class HallucinationFilterTests(unittest.TestCase):
    def test_drops_thanks_for_watching(self):
        outcome = filter_hallucinations("Thanks for watching!")
        self.assertTrue(outcome.removed)
        self.assertEqual(outcome.text, "")

    def test_drops_punctuation_only(self):
        outcome = filter_hallucinations("....")
        self.assertTrue(outcome.removed)
        self.assertEqual(outcome.reason, "punctuation_only")

    def test_keeps_real_speech(self):
        outcome = filter_hallucinations("Please open the settings window")
        self.assertFalse(outcome.removed)
        self.assertEqual(outcome.text, "Please open the settings window")

    def test_custom_blocklist_extra(self):
        outcome = filter_hallucinations(
            "beep boop ghost",
            blocklist=DEFAULT_HALLUCINATION_BLOCKLIST + ("beep boop ghost",),
        )
        self.assertTrue(outcome.removed)

    def test_trailing_outro_stripped(self):
        outcome = filter_hallucinations(
            "Ship the feature tomorrow. Thanks for watching."
        )
        self.assertTrue(outcome.removed)
        self.assertIn("Ship the feature tomorrow", outcome.text)
        self.assertNotIn("watching", outcome.text.casefold())

    def test_disabled_passthrough(self):
        outcome = filter_hallucinations("Thanks for watching", enabled=False)
        self.assertFalse(outcome.removed)
        self.assertEqual(outcome.text, "Thanks for watching")

    def test_settings_defaults_present(self):
        specs = {spec.key: spec for spec in built_in_settings()}
        self.assertIn("audio.preroll_ms", specs)
        self.assertEqual(specs["audio.preroll_ms"].default, 400.0)
        self.assertTrue(specs["cleanup.hallucination_filter_enabled"].default)
        self.assertIn("cleanup.hallucination_blocklist", specs)

    def test_app_cleanup_applies_hallucination_before_disabled_ai(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = SimpleNamespace(
            advanced_settings={
                "ai.cleanup_enabled": False,
                "cleanup.hallucination_filter_enabled": True,
            },
            custom_vocabulary="",
            live_insert=False,
            performance_preset="reliable_stream",
        )
        app.active_dictation_profile = SimpleNamespace(id="general", overrides={})
        app.performance_presets = None
        app._transcript_cleanup = Mock()
        outcome = app._cleanup_final_transcript("Thank you for watching")
        self.assertEqual(outcome.text, "")
        self.assertEqual(outcome.reason, "cleanup_disabled")
        app._transcript_cleanup.cleanup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
