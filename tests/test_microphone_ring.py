from __future__ import annotations

import unittest

from drew_local_voice.microphone_ring import (
    HTTRANSPARENT,
    WM_NCHITTEST,
    MicrophoneActiveRing,
    ring_phase_geometry,
)


class MicrophoneRingTests(unittest.TestCase):
    def test_geometry_is_centered_and_pulses_without_content(self):
        first = ring_phase_geometry(MicrophoneActiveRing.SIZE, 0.0)
        second = ring_phase_geometry(MicrophoneActiveRing.SIZE, 1.0)
        self.assertEqual(first["center"], (120.0, 120.0))
        self.assertEqual(len(first["outer"]), 4)
        self.assertIn("mid", first)
        self.assertIn("comet_start", first)
        self.assertIn("orbit_radius", first)
        self.assertNotEqual(first["outer"], second["outer"])
        self.assertNotEqual(first["arc_start"], second["arc_start"])
        self.assertGreater(first["arc_extent"], 198.0 - 1e-6)
        self.assertGreater(first["outer"][2] - first["center"][0], 68.0)

    def test_click_through_contract_is_explicit(self):
        self.assertEqual(WM_NCHITTEST, 0x84)
        self.assertEqual(HTTRANSPARENT, -1)
        self.assertEqual(MicrophoneActiveRing.TICK_MS, 24)
        self.assertEqual(MicrophoneActiveRing.SIZE, 240)


if __name__ == "__main__":
    unittest.main()
