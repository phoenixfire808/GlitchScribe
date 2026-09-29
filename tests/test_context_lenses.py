from __future__ import annotations

import importlib
import unittest
from unittest.mock import Mock


class ContextLensTests(unittest.TestCase):
    def test_selected_text_lens_returns_bounded_provenance(self):
        context = importlib.import_module("drew_local_voice.context")
        lens_type = getattr(context, "SelectedTextLens", None)
        probe_type = getattr(context, "SelectionProbe", None)
        self.assertIsNotNone(lens_type)
        self.assertIsNotNone(probe_type)

        probe = probe_type(
            accessible=True,
            is_password=False,
            text="alpha beta gamma",
            source="uia_text_pattern",
        )
        self.assertTrue(hasattr(probe, "control_identity"))
        probe = probe_type(
            accessible=True,
            is_password=False,
            text="alpha beta gamma",
            source="uia_text_pattern",
            control_identity="control-7",
        )
        reader = Mock(return_value=probe)
        lens = lens_type(max_characters=10, clock=lambda: 1234.5)
        capture = lens.capture(
            77,
            foreground_getter=lambda: 77,
            selection_reader=reader,
        )

        self.assertTrue(capture.included)
        self.assertEqual(capture.content, "alpha beta")
        self.assertEqual(capture.lens_id, "selected_text")
        self.assertEqual(capture.source, "uia_text_pattern")
        self.assertEqual(capture.target_window, 77)
        self.assertEqual(capture.captured_at, 1234.5)
        self.assertEqual(capture.sensitivity, "private")
        self.assertEqual(capture.original_characters, 16)
        self.assertTrue(capture.truncated)
        self.assertEqual(capture.blocked_reason, "")
        self.assertEqual(capture.control_identity, "control-7")


if __name__ == "__main__":
    unittest.main()
