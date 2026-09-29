from __future__ import annotations

import unittest

from drew_local_voice.ai.transformation_controls import (
    TransformationControls,
    build_modifier_contract,
    normalize_amplification,
    normalize_execution,
    normalize_length_scale,
    normalize_style,
)
from drew_local_voice.scrolling import WheelDeltaAccumulator
from drew_local_voice.wheel_indicator import normalize_scroll_intensity


class AdaptiveTransformationControlTests(unittest.TestCase):
    def test_values_are_bounded_and_unknown_values_fail_closed(self):
        self.assertEqual(normalize_amplification("99"), 4.0)
        self.assertEqual(normalize_amplification("bad"), 1.0)
        self.assertEqual(normalize_length_scale(-5), 0.5)
        self.assertEqual(normalize_length_scale(9), 2.0)
        self.assertEqual(normalize_style("Technical"), "Technical")
        self.assertEqual(normalize_style("untrusted"), "Preserve voice")
        self.assertEqual(normalize_execution("parallel"), "parallel")
        self.assertEqual(normalize_execution("exec"), "current")

    def test_modified_contract_contains_no_source_text_and_scales_output(self):
        contract = build_modifier_contract(
            TransformationControls(
                amplification=2,
                style="Formal",
                length_scale=1.5,
                execution="modified",
            )
        )
        self.assertEqual(contract["amplification"], 2.0)
        self.assertEqual(contract["style"], "Formal")
        self.assertEqual(contract["length_scale"], 1.5)
        self.assertEqual(contract["variant_count"], 1)
        self.assertGreater(contract["output_multiplier"], 1.0)
        self.assertIn("Amplification target", contract["instruction_suffix"])
        self.assertIn("Style target", contract["instruction_suffix"])
        self.assertNotIn("selected", contract["instruction_suffix"].lower())

    def test_variant_contracts_are_bounded_and_review_only(self):
        parallel = build_modifier_contract({"execution": "parallel"})
        batch = build_modifier_contract({"execution": "batch"})
        self.assertEqual(parallel["variant_count"], 2)
        self.assertEqual(batch["variant_count"], 3)
        self.assertLessEqual(batch["variant_count"], 3)
        self.assertIn("review", parallel["system_suffix"])
        self.assertIn("review", batch["system_suffix"])


class WheelIndicatorTests(unittest.TestCase):
    def test_scroll_intensity_is_stable_and_bounded(self):
        self.assertEqual(normalize_scroll_intensity(0, 0), 0.08)
        self.assertGreater(normalize_scroll_intensity(120, 1), 0.08)
        self.assertEqual(normalize_scroll_intensity(9999, 9999), 1.0)
        self.assertGreaterEqual(normalize_scroll_intensity(-240, -2), 0.08)
        self.assertLessEqual(normalize_scroll_intensity(-240, -2), 1.0)


class WheelDeltaAccumulatorTests(unittest.TestCase):
    def test_precision_deltas_accumulate_without_jumps(self):
        accumulator = WheelDeltaAccumulator()
        self.assertEqual(accumulator.consume(30), 0)
        self.assertEqual(accumulator.remainder, 30)
        self.assertEqual(accumulator.consume(90), -1)
        self.assertEqual(accumulator.remainder, 0)
        self.assertEqual(accumulator.consume(-240), 2)
        self.assertEqual(accumulator.remainder, 0)

    def test_discrete_button_input_resets_partial_precision_state(self):
        accumulator = WheelDeltaAccumulator()
        self.assertEqual(accumulator.consume(60), 0)
        self.assertEqual(accumulator.consume_discrete(-1), -1)
        self.assertEqual(accumulator.remainder, 0)
        self.assertEqual(accumulator.consume(60), 0)


if __name__ == "__main__":
    unittest.main()
