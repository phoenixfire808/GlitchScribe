from __future__ import annotations

import importlib
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from drew_local_voice.ai.providers import AIResponse, ProviderError, ProviderTimeout
from drew_local_voice.settings import built_in_settings
from local_voice import LocalVoiceApp


class TranscriptCleanupTests(unittest.TestCase):
    def test_assistant_response_falls_back_to_exact_raw_transcript(self):
        self.assertIsNotNone(importlib.util.find_spec("drew_local_voice.ai.cleanup"))
        cleanup = importlib.import_module("drew_local_voice.ai.cleanup")

        class Provider:
            def complete(self, request, *, cancellation=None):
                return AIResponse(
                    text="Sure, here's a polished email you can send tomorrow.",
                    provider_label="Local • synthetic",
                    model="synthetic",
                )

        raw = "write an email to the design team about tomorrow"
        outcome = cleanup.TranscriptCleanupService(Provider()).cleanup(
            raw,
            vocabulary=("GlitchScribe",),
            nearby_context="Design review notes",
            timeout_seconds=12.0,
        )

        self.assertEqual(outcome.text, raw)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "assistant_preamble_detected")
        self.assertEqual(outcome.provider_label, "Local • synthetic")

    def test_provider_timeout_returns_raw_transcript_with_reason(self):
        cleanup = importlib.import_module("drew_local_voice.ai.cleanup")

        class Provider:
            def complete(self, request, *, cancellation=None):
                raise ProviderTimeout("synthetic timeout")

        raw = "keep every word in this sentence"
        try:
            outcome = cleanup.TranscriptCleanupService(Provider()).cleanup(raw)
        except ProviderTimeout:
            self.fail("cleanup must convert provider timeout into an exact raw fallback")

        self.assertEqual(outcome.text, raw)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "provider_timeout")

    def test_executed_instruction_with_low_overlap_falls_back_to_raw(self):
        cleanup = importlib.import_module("drew_local_voice.ai.cleanup")

        class Provider:
            def complete(self, request, *, cancellation=None):
                return AIResponse(
                    text="Across the waves, the woolly crew sings beneath the moon.",
                    provider_label="Local • synthetic",
                    model="synthetic",
                )

        raw = "write a poem about llamas in the style of a sea shanty"
        outcome = cleanup.TranscriptCleanupService(Provider()).cleanup(raw)

        self.assertEqual(outcome.text, raw)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "instruction_marker_lost")

    def test_reasoning_wrapper_and_outer_quotes_are_sanitized_before_acceptance(self):
        cleanup = importlib.import_module("drew_local_voice.ai.cleanup")

        class Provider:
            def complete(self, request, *, cancellation=None):
                return AIResponse(
                    text='<think>Fix punctuation only.</think>\n"Hello, team."',
                    provider_label="Local • synthetic",
                    model="synthetic",
                )

        outcome = cleanup.TranscriptCleanupService(Provider()).cleanup("hello team")

        self.assertEqual(outcome.text, "Hello, team.")
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.reason, "accepted_cleanup")

    def test_enabled_prose_profile_routes_final_text_through_guarded_cleanup(self):
        cleanup = importlib.import_module("drew_local_voice.ai.cleanup")
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = SimpleNamespace(
            advanced_settings={
                "ai.cleanup_enabled": True,
                "ai.cleanup_timeout_seconds": 9.0,
                "ai.cleanup_nearby_context": True,
            },
            custom_vocabulary=("GlitchScribe",),
            live_insert=False,
            performance_preset="reliable_stream",
            # Intentional opt-in: this test exercises the mocked cleanup pipeline,
            # so it must pass the auxiliary-model policy guard that runs before the
            # mode-specific bypass checks in LocalVoiceApp._cleanup_final_transcript.
            allow_auxiliary_models=True,
        )
        app.active_dictation_profile = SimpleNamespace(id="general", overrides={})
        app.performance_presets = None
        app.session_preceding_text = "Earlier local context"
        app._transcript_cleanup = Mock()
        expected = cleanup.CleanupOutcome(
            text="Hello, GlitchScribe.",
            applied=True,
            reason="accepted_cleanup",
            provider_label="Local • synthetic",
            model="synthetic",
        )
        app._transcript_cleanup.cleanup.return_value = expected

        self.assertTrue(
            hasattr(app, "_cleanup_final_transcript"),
            "LocalVoiceApp must expose the guarded final-transcript cleanup boundary",
        )
        outcome = app._cleanup_final_transcript("hello glitch scribe")

        self.assertEqual(outcome, expected)
        app._transcript_cleanup.cleanup.assert_called_once_with(
            "hello glitch scribe",
            vocabulary=("GlitchScribe",),
            nearby_context="Earlier local context",
            timeout_seconds=9.0,
        )

    def test_disabled_cleanup_returns_raw_without_calling_provider_service(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = SimpleNamespace(
            advanced_settings={"ai.cleanup_enabled": False},
            custom_vocabulary="",
            live_insert=False,
            performance_preset="reliable_stream",
        )
        app.active_dictation_profile = SimpleNamespace(id="general", overrides={})
        app.performance_presets = None
        app._transcript_cleanup = Mock()

        raw = "leave this text alone"
        outcome = app._cleanup_final_transcript(raw)

        self.assertEqual(outcome.text, raw)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "cleanup_disabled")
        app._transcript_cleanup.cleanup.assert_not_called()

    def test_incompatible_dictation_modes_bypass_cleanup(self):
        cases = (
            ("terminal", False, "git status", "terminal_profile"),
            ("general", True, "hello team", "live_insertion_enabled"),
            ("general", False, "stop listening", "voice_command"),
        )
        for profile_id, live_insert, raw, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                app = LocalVoiceApp.__new__(LocalVoiceApp)
                app.config = SimpleNamespace(
                    advanced_settings={"ai.cleanup_enabled": True},
                    custom_vocabulary="",
                    live_insert=live_insert,
                    performance_preset="reliable_stream",
                    # Intentional opt-in: this test asserts the mode-specific bypass
                    # reasons ("terminal_profile" / "live_insertion_enabled" /
                    # "voice_command"), which only fire AFTER the auxiliary-model
                    # policy guard inside LocalVoiceApp._cleanup_final_transcript.
                    allow_auxiliary_models=True,
                )
                app.active_dictation_profile = SimpleNamespace(
                    id=profile_id,
                    overrides={},
                )
                app.performance_presets = None
                app._transcript_cleanup = Mock()

                outcome = app._cleanup_final_transcript(raw)

                self.assertEqual(outcome.text, raw)
                self.assertFalse(outcome.applied)
                self.assertEqual(outcome.reason, expected_reason)
                app._transcript_cleanup.cleanup.assert_not_called()

    def test_finalizer_posts_guarded_cleanup_text_and_privacy_safe_metrics(self):
        cleanup = importlib.import_module("drew_local_voice.ai.cleanup")
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        stream = Mock()
        stream.finalize.return_value = SimpleNamespace(
            text="hello team",
            inference_seconds=0.2,
            stop_to_text_seconds=0.3,
            reused_partial=False,
        )
        app._cleanup_final_transcript = Mock(
            return_value=cleanup.CleanupOutcome(
                text="Hello, team.",
                applied=True,
                reason="accepted_cleanup",
                provider_label="Local • synthetic",
                model="synthetic",
            )
        )
        app._finish_transcription = Mock()
        app._finish_error = Mock()
        app._post_to_ui = lambda callback: callback()

        app._finalize_stream_worker(
            stream,
            SimpleNamespace(),
            1.25,
            True,
            123,
        )

        app._finish_error.assert_not_called()
        text, metrics, audio_seconds, auto_insert, target_window = (
            app._finish_transcription.call_args.args
        )
        self.assertEqual(text, "Hello, team.")
        self.assertEqual(metrics["cleanup_reason"], "accepted_cleanup")
        self.assertTrue(metrics["cleanup_applied"])
        self.assertNotIn("cleanup_text", metrics)
        self.assertEqual((audio_seconds, auto_insert, target_window), (1.25, True, 123))

    def test_cleanup_settings_are_visible_bounded_and_off_by_default(self):
        specs = {spec.key: spec for spec in built_in_settings()}
        required = {
            "ai.cleanup_enabled",
            "ai.cleanup_nearby_context",
            "ai.cleanup_timeout_seconds",
        }

        self.assertTrue(
            required <= specs.keys(),
            f"missing cleanup settings: {sorted(required - specs.keys())}",
        )
        self.assertFalse(specs["ai.cleanup_enabled"].default)
        self.assertTrue(specs["ai.cleanup_nearby_context"].default)
        timeout = specs["ai.cleanup_timeout_seconds"]
        self.assertEqual(timeout.default, 20.0)
        self.assertEqual((timeout.minimum, timeout.maximum), (1.0, 120.0))

    def test_cleanup_service_forwards_its_bounded_request_timeout(self):
        cleanup = importlib.import_module("drew_local_voice.ai.cleanup")
        captured = {}

        class Provider:
            def complete(self, request, *, cancellation=None):
                captured["request"] = request
                return AIResponse(
                    text="Hello, team.",
                    provider_label="Local • synthetic",
                    model="synthetic",
                )

        cleanup.TranscriptCleanupService(Provider()).cleanup(
            "hello team",
            timeout_seconds=12.0,
        )

        self.assertEqual(captured["request"].timeout_seconds, 12.0)

    def test_provider_failure_returns_exact_raw_transcript(self):
        cleanup = importlib.import_module("drew_local_voice.ai.cleanup")

        class Provider:
            def complete(self, request, *, cancellation=None):
                raise ProviderError("synthetic connection reset")

        raw = "this survives a local provider crash"
        try:
            outcome = cleanup.TranscriptCleanupService(Provider()).cleanup(raw)
        except ProviderError:
            self.fail("cleanup must convert provider failure into an exact raw fallback")

        self.assertEqual(outcome.text, raw)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "provider_error")

    def test_empty_sanitized_output_cannot_erase_raw_transcript(self):
        cleanup = importlib.import_module("drew_local_voice.ai.cleanup")

        class Provider:
            def complete(self, request, *, cancellation=None):
                return AIResponse(
                    text="<think>Nothing to return.</think>",
                    provider_label="Local • synthetic",
                    model="synthetic",
                )

        raw = "keep this valid dictated sentence"
        outcome = cleanup.TranscriptCleanupService(Provider()).cleanup(raw)

        self.assertEqual(outcome.text, raw)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "empty_output")

    def test_unrelated_semantic_expansion_falls_back_to_raw(self):
        cleanup = importlib.import_module("drew_local_voice.ai.cleanup")

        class Provider:
            def complete(self, request, *, cancellation=None):
                return AIResponse(
                    text=(
                        "A comprehensive remediation strategy should introduce staged "
                        "rollouts, executive reporting, and quarterly governance reviews."
                    ),
                    provider_label="Local • synthetic",
                    model="synthetic",
                )

        raw = "the server failed during startup again"
        outcome = cleanup.TranscriptCleanupService(Provider()).cleanup(raw)

        self.assertEqual(outcome.text, raw)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "meaning_drift")

    def test_empty_raw_transcript_never_calls_provider(self):
        cleanup = importlib.import_module("drew_local_voice.ai.cleanup")
        provider = Mock()

        outcome = cleanup.TranscriptCleanupService(provider).cleanup("")

        self.assertEqual(outcome.text, "")
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "empty_transcript")
        provider.complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
