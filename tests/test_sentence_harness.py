from __future__ import annotations

import unittest
from unittest.mock import Mock

from drew_local_voice.ai.providers import (
    AIResponse,
    ProviderError,
    ProviderTimeout,
)
from drew_local_voice.ai.sentence_harness import (
    DEFAULT_HARNESS_MODES,
    SentenceHarness,
    get_mode,
    list_modes,
    run_mode,
)


REQUIRED_MODE_IDS = {
    "literal_cleanup",
    "beef_up_natural",
    "coding_prompt",
    "cursor_agent_task",
    "implementation_plan",
    "make_shorter",
    "bullets",
    "fix_grammar",
    "explain",
    "extract_todos",
    "formal_email",
    "casual_chat",
}


class _FakeProvider:
    def __init__(self, text: str = "", *, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list = []
        self.label = "Local • synthetic"

    def complete(self, request, *, cancellation=None):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return AIResponse(
            text=self.text,
            provider_label=self.label,
            model="synthetic",
        )


class SentenceHarnessTests(unittest.TestCase):
    def test_registry_includes_required_modes_with_profiles(self):
        modes = list_modes()
        ids = {m.id for m in modes}
        self.assertTrue(REQUIRED_MODE_IDS <= ids)
        self.assertEqual(len(modes), len(DEFAULT_HARNESS_MODES))
        for mode in modes:
            self.assertIn(mode.safety_profile, {"literal", "expand", "rewrite"})
            self.assertTrue(mode.label)
            self.assertTrue(mode.description)
            if mode.id != "literal_cleanup":
                self.assertTrue(mode.instruction.strip())
                self.assertTrue(mode.system_prompt.strip())

    def test_get_mode_known_and_unknown(self):
        mode = get_mode("coding_prompt")
        self.assertIsNotNone(mode)
        self.assertEqual(mode.safety_profile, "expand")
        self.assertIsNone(get_mode("not_a_real_mode"))
        self.assertIsNone(get_mode(""))

    def test_unknown_mode_returns_original(self):
        provider = _FakeProvider("should not run")
        raw = "hello world"
        outcome = run_mode(provider, "nope", raw)
        self.assertEqual(outcome.text, raw)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "unknown_mode")
        self.assertEqual(outcome.mode_id, "nope")
        self.assertEqual(provider.calls, [])

    def test_empty_text_skips_provider(self):
        provider = _FakeProvider("x")
        outcome = run_mode(provider, "beef_up_natural", "   ")
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "empty_transcript")
        self.assertEqual(provider.calls, [])

    def test_literal_cleanup_delegates_and_rejects_preamble(self):
        provider = _FakeProvider(
            "Sure, here's a polished email you can send tomorrow."
        )
        raw = "write an email to the design team about tomorrow"
        outcome = run_mode(provider, "literal_cleanup", raw)
        self.assertEqual(outcome.text, raw)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "assistant_preamble_detected")
        self.assertEqual(outcome.mode_id, "literal_cleanup")
        self.assertEqual(len(provider.calls), 1)

    def test_literal_cleanup_accepts_sanitized_cleanup(self):
        provider = _FakeProvider('<think>fix</think>\n"Hello, team."')
        outcome = run_mode(provider, "literal_cleanup", "hello team")
        self.assertEqual(outcome.text, "Hello, team.")
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.reason, "accepted_cleanup")

    def test_beef_up_accepts_structured_expansion(self):
        source = "make the wheel open faster near the cursor"
        upgraded = (
            "Goal: Open the Action Wheel faster near the cursor.\n\n"
            "Context:\n"
            "- User wants snappier wheel open near cursor\n\n"
            "Steps:\n"
            "1. Profile wheel show path\n"
            "2. Reduce work before first paint\n\n"
            "Acceptance criteria:\n"
            "- Wheel appears quickly at cursor\n"
        )
        provider = _FakeProvider(upgraded)
        outcome = run_mode(
            provider,
            "beef_up_natural",
            source,
            vocabulary=("Action Wheel",),
            nearby_context="pie menu code",
        )
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.mode_id, "beef_up_natural")
        self.assertIn(outcome.reason, {"accepted", "unchanged"})
        self.assertIn("wheel", outcome.text.casefold())
        self.assertEqual(len(provider.calls), 1)
        req = provider.calls[0]
        self.assertIn("SOURCE_TEXT", req.context)
        self.assertIn(source, req.context)
        self.assertEqual(req.context_label, "selected_text")

    def test_coding_prompt_rejects_assistant_preamble(self):
        provider = _FakeProvider(
            "Sure, I'd be happy to write a full multi-file refactor plan for you."
        )
        raw = "refactor the insert path to restore focus"
        outcome = run_mode(provider, "coding_prompt", raw)
        self.assertEqual(outcome.text, raw)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "assistant_preamble_detected")

    def test_cursor_agent_task_accepts_goal_structure(self):
        source = "add unit tests for the sentence harness modes"
        structured = (
            "Goal:\n"
            "Add unit tests for the sentence harness modes.\n\n"
            "Context:\n"
            "- sentence harness modes need coverage\n\n"
            "Steps:\n"
            "1. Inspect sentence_harness.py\n"
            "2. Add fake provider tests\n"
            "3. Run unittest module\n\n"
            "Acceptance criteria:\n"
            "- Required modes registered\n"
            "- Safety rejects preambles\n\n"
            "Constraints:\n"
            "- Stay local / offline where possible\n"
        )
        provider = _FakeProvider(structured)
        outcome = run_mode(provider, "cursor_agent_task", source)
        self.assertTrue(outcome.applied)
        self.assertIn("Goal:", outcome.text)
        self.assertEqual(outcome.mode_id, "cursor_agent_task")

    def test_expand_rejects_unrelated_long_drift(self):
        provider = _FakeProvider(
            "Quantum pastry recipes require laminated dough, "
            "tempered chocolate, and seasonal fruit reductions "
            "served with espresso foam art."
        )
        raw = "wire the action wheel to the local sentence harness API"
        outcome = run_mode(provider, "coding_prompt", raw)
        self.assertEqual(outcome.text, raw)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "meaning_drift")

    def test_provider_timeout_and_error_fallback(self):
        raw = "keep this sentence intact"
        for err, reason in (
            (ProviderTimeout("t"), "provider_timeout"),
            (ProviderError("e"), "provider_error"),
        ):
            with self.subTest(reason=reason):
                outcome = run_mode(_FakeProvider(error=err), "make_shorter", raw)
                self.assertEqual(outcome.text, raw)
                self.assertFalse(outcome.applied)
                self.assertEqual(outcome.reason, reason)

    def test_empty_model_output_falls_back(self):
        provider = _FakeProvider("<think>nothing</think>")
        raw = "something spoken out loud"
        outcome = run_mode(provider, "bullets", raw)
        self.assertEqual(outcome.text, raw)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "empty_output")

    def test_short_refusal_rejected(self):
        provider = _FakeProvider("I'm unable to help with that request.")
        raw = "draft a casual chat message about lunch"
        outcome = run_mode(provider, "casual_chat", raw)
        self.assertEqual(outcome.text, raw)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "refusal_detected")

    def test_facade_run_matches_module_api(self):
        provider = _FakeProvider("- ship the feature\n- write tests")
        harness = SentenceHarness(provider)
        self.assertEqual(len(harness.list_modes()), len(list_modes()))
        self.assertIsNotNone(harness.get_mode("extract_todos"))
        outcome = harness.run(
            "extract_todos",
            "we need to ship the feature and write tests",
        )
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.mode_id, "extract_todos")

    def test_timeout_forwarded_on_request(self):
        provider = _FakeProvider("Short text.")
        run_mode(
            provider,
            "make_shorter",
            "this is a longer bit of text that should shrink",
            timeout_seconds=17.0,
        )
        self.assertEqual(provider.calls[0].timeout_seconds, 17.0)

    def test_fix_grammar_literal_profile(self):
        self.assertEqual(get_mode("fix_grammar").safety_profile, "literal")
        provider = _FakeProvider("Hello, team. Let's ship tomorrow.")
        outcome = run_mode(provider, "fix_grammar", "hello team lets ship tomorrow")
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.reason, "accepted")


if __name__ == "__main__":
    unittest.main()
