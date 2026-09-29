import unittest

from apps.call_response.call_response import (
    DEFAULT_ENDPOINT,
    RESPONSE_SYSTEM_PROMPT,
    build_response_messages,
    decode_ctc_ids,
    strip_reasoning,
    validate_loopback_endpoint,
)


class CallResponsePureTests(unittest.TestCase):
    def test_loopback_endpoint_rejects_remote_origin(self):
        with self.assertRaises(ValueError):
            validate_loopback_endpoint("https://example.invalid")

    def test_loopback_endpoint_accepts_local_origin(self):
        self.assertEqual(validate_loopback_endpoint(DEFAULT_ENDPOINT), DEFAULT_ENDPOINT)

    def test_reasoning_wrappers_are_not_shown(self):
        self.assertEqual(
            strip_reasoning("<think>private reasoning</think> Say this instead."),
            "Say this instead.",
        )

    def test_ctc_decode_collapses_blanks_and_sentencepiece_boundary(self):
        tokens = {1: "\u2581Please", 2: " wait", 3: "<blk>"}
        self.assertEqual(
            decode_ctc_ids([1, 1, 3, 2, 2], tokens, blank_id=3),
            "Please wait",
        )

    def test_prompt_preserves_economic_goal_without_inventing_amounts(self):
        messages = build_response_messages([], "They asked what I want.")
        system = messages[0]["content"]
        self.assertIn("complete written negotiated separation", system)
        self.assertIn("Never invent", system)
        self.assertNotIn("$", system)
        self.assertEqual(messages[-1]["content"], "They asked what I want.")


if __name__ == "__main__":
    unittest.main()
