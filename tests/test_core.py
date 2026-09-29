import ctypes
import json
import tempfile
import threading
import time
import unittest
import wave
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import local_voice as local_voice_module
from dictation_features import LiveCommitter, VoiceCommand
from streaming_dictation import StreamResult
from drew_local_voice.performance import (
    PerformancePresetRegistry,
    built_in_performance_presets,
)
from drew_local_voice.context import ContextLensCapture
from drew_local_voice.ai.preview import PreviewDraft, PreviewGate
from drew_local_voice.ai.providers import (
    AIResponse,
    CancellationSignal,
    ProviderTimeout,
)
from drew_local_voice.profiles import ProfileRegistry, built_in_profiles
from drew_local_voice.settings import SettingsRegistry, built_in_settings
from drew_local_voice.workflows import PromptCompiler, SnippetStore, built_in_snippets

from local_voice import (
    AppConfig,
    append_runtime_event,
    AudioRecorder,
    CaretContext,
    FocusedTargetInfo,
    INPUT,
    LocalVoiceApp,
    RuntimeProfile,
    SpeechEndDetector,
    TRAY_ICON_PATH,
    WhisperBackend,
    create_uia_automation,
    choose_runtime_profile,
    choose_session_preceding_text,
    choose_saved_device,
    insertion_target_is_safe,
    insertion_target_safety,
    main,
    normalize_transcript,
    uia_control_is_editable,
    _uia_rect_anchor,
    focused_text_target_anchor,
    validated_text_target_anchor,
    probe_gpu_free_vram_mb,
    preferred_bubble_anchor,
    replace_hotkey,
    resample_audio,
    resolve_hotkey,
    tail_is_quiet,
    write_wav,
)


class CoreTests(unittest.TestCase):
    def test_taskbar_notification_icon_uses_the_packaged_glitchscribe_asset(self):
        self.assertEqual(TRAY_ICON_PATH.name, "drew-local-voice-tray.ico")
        self.assertTrue(TRAY_ICON_PATH.is_file())

    def test_bubble_anchor_prefers_text_caret_then_falls_back_to_mouse(self):
        self.assertEqual(
            preferred_bubble_anchor((600, 500), (900, 700)),
            (600, 500),
        )
        self.assertEqual(preferred_bubble_anchor(None, (900, 700)), (900, 700))
        self.assertIsNone(preferred_bubble_anchor(None, None))

    def test_validated_text_target_anchor_fails_closed_without_target(self):
        anchor, reason = validated_text_target_anchor(0)
        self.assertIsNone(anchor)
        self.assertEqual(reason, "missing_target")

    def test_validated_text_target_anchor_requires_editable_uia_target(self):
        focused = FocusedTargetInfo(
            accessible=True,
            is_password=False,
            is_editable=False,
            ime_active=False,
            keyboard_focusable=True,
            control_type=50057,
        )
        with patch.object(local_voice_module, "_window_belongs_to_target", return_value=True), patch.object(
            local_voice_module, "_inspect_focused_control", return_value=focused
        ):
            anchor, reason = validated_text_target_anchor(101, focused_window=202)
        self.assertIsNone(anchor)
        self.assertEqual(reason, "unsupported_control")

    def test_selection_anchor_uses_final_visible_line_bottom_center(self):
        self.assertEqual(
            _uia_rect_anchor((10, 20, 100, 12, 10, 40, 50, 12)),
            (35, 52),
        )

    def test_selection_geometry_failure_does_not_fall_back_to_native_caret(self):
        with patch.object(
            local_voice_module,
            "uia_text_anchor_screen_position",
            return_value=(None, "selection_geometry_unavailable"),
        ), patch.object(
            local_voice_module,
            "native_caret_screen_position",
            return_value=(999, 999),
        ):
            anchor, reason = focused_text_target_anchor(101)
        self.assertIsNone(anchor)
        self.assertEqual(reason, "selection_geometry_unavailable")

    def test_app_preparation_suppresses_inferred_period_but_preserves_internal_dot(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = AppConfig(
            automatic_punctuation=True,
            suppress_inferred_trailing_period=True,
        )
        app.active_dictation_profile = None

        self.assertEqual(
            app._prepare_dictated_text("python script.py."),
            "python script.py",
        )

    def test_exact_slash_final_insertion_has_no_automatic_space(self):
        class Variable:
            def set(self, _value):
                pass

        class Widget:
            def configure(self, **_kwargs):
                pass

            def delete(self, *_args):
                pass

            def insert(self, *_args):
                pass

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = SimpleNamespace(
            show_overlay=False,
            hotkey="Ctrl + Alt + Space",
            filter_profanity=False,
            automatic_punctuation=True,
            custom_replacements="",
            custom_vocabulary="",
            smart_continuation=True,
            live_insert=True,
        )
        app.record_button = Widget()
        app.metrics_var = Variable()
        app.transcript_label_var = Variable()
        app.text = Widget()
        app.live_committer = LiveCommitter(holdback_words=0)
        app.live_insertion_suspended = False
        app.session_live_payload = ""
        app.session_preceding_text = ""
        app.last_inserted_payload = ""
        app.last_inserted_target = 0
        app.last_inserted_at = 0.0
        app._target_is_active = lambda _target: True
        app._set_status = lambda *_args: None

        from drew_local_voice.insertion import InsertionResult

        app._insert_text_with_policy = Mock(
            return_value=InsertionResult(1, 1, reason="ok", method="unicode")
        )
        app._remember_last_transcript = Mock()
        app.pipeline_observer = Mock()
        app.pipeline_observer.summary.return_value = {"stage_count": 0, "elapsed_ms": 0}
        app._hide_streaming_bubble = Mock()
        app.busy = True
        app.finalizer_thread = None
        app.streaming = None
        app.finalization_cancelled = __import__("threading").Event()
        with patch("local_voice.append_runtime_event"):
            app._finish_transcription(
                "slash",
                {"stop_to_text_seconds": 0.01, "reused_partial": False},
                0.5,
                True,
                41,
            )

        app._insert_text_with_policy.assert_called_once()
        self.assertEqual(app._insert_text_with_policy.call_args.args[0], "/")

    def test_direct_empty_caret_context_beats_stale_same_window_history(self):
        self.assertEqual(
            choose_session_preceding_text(
                CaretContext(supported=True, preceding_text=""),
                same_recent_target=True,
                last_inserted_payload="stale prior text ",
            ),
            "",
        )

    def test_recent_same_target_history_is_only_a_context_fallback(self):
        self.assertEqual(
            choose_session_preceding_text(
                CaretContext(supported=False),
                same_recent_target=True,
                last_inserted_payload="prior fragment ",
            ),
            "prior fragment ",
        )
        self.assertEqual(
            choose_session_preceding_text(
                CaretContext(supported=False),
                same_recent_target=False,
                last_inserted_payload="prior fragment ",
            ),
            "",
        )

    def test_custom_uia_control_requires_edit_pattern_and_keyboard_focus(self):
        self.assertTrue(
            uia_control_is_editable(
                50025,
                keyboard_focusable=True,
                text_pattern_available=True,
                value_pattern_available=False,
                read_only=False,
            )
        )
        self.assertFalse(
            uia_control_is_editable(
                50025,
                keyboard_focusable=True,
                text_pattern_available=False,
                value_pattern_available=False,
                read_only=False,
            )
        )
        self.assertFalse(
            uia_control_is_editable(
                50025,
                keyboard_focusable=True,
                text_pattern_available=True,
                value_pattern_available=False,
                read_only=True,
            )
        )
        self.assertTrue(
            uia_control_is_editable(
                50020,
                keyboard_focusable=True,
                text_pattern_available=True,
                value_pattern_available=False,
                read_only=False,
            )
        )

    def test_insertion_safety_reports_specific_rejection_reason(self):
        safe, reason = insertion_target_safety(
            41,
            inspector=lambda: FocusedTargetInfo(True, False, False),
            foreground_getter=lambda: 41,
        )
        self.assertFalse(safe)
        self.assertEqual(reason, "unsupported_control")

        safe, reason = insertion_target_safety(
            41,
            inspector=lambda: FocusedTargetInfo(True, False, True),
            foreground_getter=lambda: 99,
        )
        self.assertFalse(safe)
        self.assertEqual(reason, "foreground_changed")

    def test_insertion_safety_allows_keyboard_focused_cursor_custom_editor(self):
        safe, reason = insertion_target_safety(
            41,
            inspector=lambda: FocusedTargetInfo(
                True,
                False,
                False,
                ime_active=False,
                keyboard_focusable=True,
                control_type=50025,
            ),
            foreground_getter=lambda: 41,
            process_metadata_getter=lambda _hwnd: ("Cursor.exe", ""),
        )

        self.assertTrue(safe)
        self.assertEqual(reason, "safe")

    def test_insertion_safety_restores_foreground_when_allowed(self):
        restored = {"called": False}

        def restorer(_hwnd: int) -> bool:
            restored["called"] = True
            return True

        safe, reason = insertion_target_safety(
            41,
            inspector=lambda: FocusedTargetInfo(True, False, True),
            foreground_getter=lambda: 41 if restored["called"] else 99,
            allow_foreground_restore=True,
            foreground_restorer=restorer,
        )
        self.assertTrue(restored["called"])
        self.assertTrue(safe)
        self.assertEqual(reason, "safe")

    def test_insertion_safety_process_fallback_when_uia_unavailable(self):
        safe, reason = insertion_target_safety(
            41,
            inspector=lambda: FocusedTargetInfo(False, False, False),
            foreground_getter=lambda: 41,
            process_metadata_getter=lambda _hwnd: ("cursor.exe", "editor"),
        )
        self.assertTrue(safe)
        self.assertEqual(reason, "safe_process_fallback")

    def test_insertion_safety_password_still_fail_closed(self):
        safe, reason = insertion_target_safety(
            41,
            inspector=lambda: FocusedTargetInfo(
                True, True, True, ime_active=False, keyboard_focusable=True
            ),
            foreground_getter=lambda: 41,
            process_metadata_getter=lambda _hwnd: ("chrome.exe", ""),
        )
        self.assertFalse(safe)
        self.assertEqual(reason, "password_field")

    def test_insertion_safety_keeps_unknown_custom_controls_blocked(self):
        safe, reason = insertion_target_safety(
            41,
            inspector=lambda: FocusedTargetInfo(
                True,
                False,
                False,
                ime_active=False,
                keyboard_focusable=True,
                control_type=50025,
            ),
            foreground_getter=lambda: 41,
            process_metadata_getter=lambda _hwnd: ("unknown.exe", ""),
        )

        self.assertFalse(safe)
        self.assertEqual(reason, "unsupported_control")

    def test_overlay_position_defaults_are_persistable(self):
        # Contract: gpu_vram_reserve_mb default is 512 MiB (Nemotron budget room).
        # The previous 1024 MiB default predates the explicit-model contract.
        config = AppConfig()
        self.assertIsNone(config.overlay_x)
        self.assertIsNone(config.overlay_y)
        self.assertEqual((config.overlay_width, config.overlay_height), (300, 92))
        self.assertTrue(config.streaming_bubble)
        self.assertEqual(config.bubble_width, 420)
        self.assertEqual(getattr(config, "bubble_height", None), 68)
        self.assertEqual(getattr(config, "bubble_font_size", None), 10)
        self.assertEqual(getattr(config, "bubble_font_family", None), "Segoe UI")
        self.assertEqual(getattr(config, "bubble_font_weight", None), "normal")
        self.assertEqual(getattr(config, "bubble_visual_style", None), "glitch_glass")
        self.assertEqual(getattr(config, "bubble_text_color_mode", None), "theme")
        self.assertEqual(getattr(config, "bubble_corner_radius", None), 20)
        self.assertEqual(getattr(config, "bubble_border_width", None), 1)
        self.assertEqual(getattr(config, "bubble_max_characters", None), 120)
        self.assertAlmostEqual(config.bubble_opacity, 0.82)
        self.assertEqual((config.bubble_offset_x, config.bubble_offset_y), (18, -88))
        self.assertEqual(config.theme_mode, "dark")
        self.assertEqual(config.accent_color, "violet")
        self.assertEqual(config.gpu_device_choice, "automatic")
        self.assertEqual(config.gpu_vram_reserve_mb, 512)

    def test_gpu_vram_reserve_is_clamped_when_config_is_loaded(self):
        for saved_value, expected in ((-1, 0), (99999, 8192)):
            with self.subTest(saved_value=saved_value):
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "config.json"
                    path.write_text(
                        json.dumps({"gpu_vram_reserve_mb": saved_value}),
                        encoding="utf-8",
                    )
                    with patch("local_voice.CONFIG_PATH", path):
                        config = AppConfig.load()
                self.assertEqual(config.gpu_vram_reserve_mb, expected)

    def test_config_accepts_any_nonnegative_cuda_index_and_rejects_malformed_values(self):
        cases = (
            ("cuda:7", "cuda:7"),
            ("CUDA:003", "cuda:3"),
            ("cuda", "cuda:0"),
            ("cuda:-1", "automatic"),
            ("cuda:abc", "automatic"),
        )
        for saved_value, expected in cases:
            with self.subTest(saved_value=saved_value):
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "config.json"
                    path.write_text(
                        json.dumps({"gpu_device_choice": saved_value}),
                        encoding="utf-8",
                    )
                    with patch("local_voice.CONFIG_PATH", path):
                        config = AppConfig.load()
                self.assertEqual(config.gpu_device_choice, expected)

    def test_code_profile_disables_prose_continuation_period_and_spacing(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = AppConfig(
            smart_continuation=True,
            suppress_inferred_trailing_period=False,
            advanced_settings={"typing.trailing_space": True},
        )
        app.performance_presets = None
        app.active_dictation_profile = ProfileRegistry(built_in_profiles()).get("code")

        self.assertFalse(app._profile_setting("typing.smart_continuation", True))
        self.assertTrue(
            app._profile_setting("typing.suppress_inferred_trailing_period", False)
        )
        self.assertFalse(app._profile_setting("typing.trailing_space", True))

    def test_workflow_voice_commands_insert_exact_local_payloads(self):
        with tempfile.TemporaryDirectory() as folder:
            app = LocalVoiceApp.__new__(LocalVoiceApp)
            app._target_is_active = lambda _target: True
            app.last_inserted_payload = ""
            app.last_inserted_target = 0
            app.snippet_store = SnippetStore(
                Path(folder) / "snippets.json",
                defaults=built_in_snippets(),
            )
            app.snippet_store.initialize()
            app.prompt_compiler = PromptCompiler()

            with patch("local_voice.type_unicode") as type_unicode_mock:
                self.assertTrue(
                    app._execute_voice_command(
                        VoiceCommand("snippet", "Python Main"),
                        321,
                    )
                )
                snippet_payload = 'if __name__ == "__main__":\n    main()'
                type_unicode_mock.assert_called_once_with(snippet_payload, 321)
                self.assertEqual(app.last_inserted_payload, snippet_payload)

                type_unicode_mock.reset_mock()
                self.assertTrue(
                    app._execute_voice_command(
                        VoiceCommand("compile_prompt", "Add retry handling"),
                        321,
                    )
                )
                compiled_payload = type_unicode_mock.call_args.args[0]
                self.assertIn("IMPLEMENTATION BRIEF", compiled_payload)
                self.assertIn("Add retry handling", compiled_payload)
                self.assertFalse(compiled_payload.endswith(" "))
                self.assertEqual(app.last_inserted_payload, compiled_payload)

                type_unicode_mock.reset_mock()
                self.assertFalse(
                    app._execute_voice_command(
                        VoiceCommand("snippet", "does not exist"),
                        321,
                    )
                )
                type_unicode_mock.assert_not_called()

    def test_capture_selection_command_stages_context_without_typing(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app._target_is_active = lambda _target: True
        app.selected_text_context = None
        app.selected_text_lens = Mock()
        captured = ContextLensCapture(
            lens_id="selected_text",
            included=True,
            content="selected API error",
            source="uia_text_pattern",
            target_window=654,
            captured_at=123.0,
            original_characters=18,
        )
        app.selected_text_lens.capture.return_value = captured

        with patch("local_voice.type_unicode") as type_unicode_mock:
            self.assertTrue(
                app._execute_voice_command(VoiceCommand("capture_selection"), 654)
            )
            type_unicode_mock.assert_not_called()
        self.assertEqual(app.selected_text_context, captured)

        app.selected_text_lens.capture.return_value = ContextLensCapture(
            lens_id="selected_text",
            included=False,
            content="",
            source="uia_unavailable",
            target_window=654,
            captured_at=124.0,
            blocked_reason="empty_selection",
        )
        self.assertFalse(
            app._execute_voice_command(VoiceCommand("capture_selection"), 654)
        )
        self.assertIsNone(app.selected_text_context)

    def test_explain_selection_stages_preview_without_typing_or_applying(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = SimpleNamespace(
            allow_auxiliary_models=True,
            auxiliary_model_name="qwen3-coder-local",
        )
        app._target_is_active = lambda _target: True
        app.selected_text_context = ContextLensCapture(
            lens_id="selected_text",
            included=True,
            content="ValueError: invalid token",
            source="uia_text_pattern",
            target_window=654,
            captured_at=123.0,
            original_characters=25,
        )
        app.ai_provider = Mock()
        app.ai_provider.complete.return_value = AIResponse(
            text="The parser rejected an invalid token.",
            provider_label="Local • qwen3-coder-local",
            model="qwen3-coder-local",
        )
        app.preview_gate = PreviewGate()
        app._show_ai_preview = Mock()
        app.root = SimpleNamespace(after=lambda _delay, callback: callback())

        with patch("local_voice.type_unicode") as type_unicode_mock:
            self.assertTrue(
                app._execute_voice_command(VoiceCommand("explain_selection"), 654)
            )
            type_unicode_mock.assert_not_called()

        request = app.ai_provider.complete.call_args.args[0]
        self.assertEqual(request.instruction, "Explain the selected code or error.")
        self.assertEqual(request.context, "ValueError: invalid token")
        draft = app.preview_gate.pending
        self.assertIsNotNone(draft)
        self.assertEqual(draft.original, request.context)
        self.assertEqual(draft.result, "The parser rejected an invalid token.")
        self.assertEqual(draft.context_lenses, ("Selected Text",))
        app._show_ai_preview.assert_called_once_with(draft)

        app.selected_text_context = None
        app.ai_provider.reset_mock()
        self.assertFalse(
            app._execute_voice_command(VoiceCommand("explain_selection"), 654)
        )
        app.ai_provider.complete.assert_not_called()

    def test_edit_selection_captures_and_stages_a_local_transform_preview(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = SimpleNamespace(
            allow_auxiliary_models=True,
            auxiliary_model_name="qwen3-coder-local",
        )
        app._target_is_active = lambda _target: True
        app.selected_text_context = None
        app.selected_text_lens = Mock()
        captured = ContextLensCapture(
            lens_id="selected_text",
            included=True,
            content="This sentence is much longer than it needs to be.",
            source="uia_text_pattern",
            target_window=654,
            captured_at=123.0,
            original_characters=49,
            control_identity="control-654",
        )
        app.selected_text_lens.capture.return_value = captured
        app.ai_provider = Mock()
        app.ai_provider.complete.return_value = AIResponse(
            text="This sentence is too long.",
            provider_label="Local • qwen3-coder-local",
            model="qwen3-coder-local",
        )
        app.ai_cancellation = None
        app.preview_gate = PreviewGate()
        app._show_ai_preview = Mock()
        app.root = SimpleNamespace(after=lambda _delay, callback: callback())

        with patch("local_voice.type_unicode") as type_unicode_mock:
            self.assertTrue(
                app._execute_voice_command(
                    VoiceCommand("edit_selection", "Make this shorter."),
                    654,
                )
            )
            type_unicode_mock.assert_not_called()

        app.selected_text_lens.capture.assert_called_once_with(654)
        request = app.ai_provider.complete.call_args.args[0]
        self.assertEqual(request.instruction, "Make this shorter.")
        self.assertEqual(request.context, captured.content)
        self.assertIn("only source material", request.system_prompt)
        draft = app.preview_gate.pending
        self.assertIsNotNone(draft)
        self.assertEqual(draft.original, captured.content)
        self.assertEqual(draft.result, "This sentence is too long.")
        self.assertEqual(draft.selection_identity, "control-654")
        app._show_ai_preview.assert_called_once_with(draft)

    def test_ai_preview_stage_consumes_exact_draft_once_into_review(self):
        draft = PreviewDraft.create(
            original="selected error",
            instruction="Explain the selected error.",
            result="Local explanation",
            provider_label="Local • qwen3-coder-local",
            context_lenses=("Selected Text",),
            target_window=654,
            clock=lambda: 100.0,
            id_factory=lambda: "preview-stage-1",
        )
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.preview_gate = PreviewGate()
        app.preview_gate.stage(draft)
        app.text = Mock()
        app.transcript_label_var = Mock()
        app.root = Mock()
        app._set_status = Mock()

        app._stage_ai_preview(draft.transaction_id, draft.fingerprint)
        app.text.delete.assert_called_once_with("1.0", "end")
        app.text.insert.assert_called_once_with("1.0", draft.result)
        self.assertEqual(app.last_text, draft.result)
        self.assertIsNone(app.preview_gate.pending)

        app.text.reset_mock()
        app._stage_ai_preview(draft.transaction_id, draft.fingerprint)
        app.text.insert.assert_not_called()
        app._set_status.assert_called_with(
            "AI preview changed or was already consumed",
            "error",
        )

    def test_ai_preview_refuses_to_stage_unchanged_result(self):
        draft = PreviewDraft.create(
            original="value = 1\r\n",
            instruction="Simplify selection.",
            result="value = 1\n",
            provider_label="Local • qwen3-coder-local",
            context_lenses=("Selected Text",),
            target_window=654,
            clock=lambda: 100.0,
            id_factory=lambda: "preview-unchanged-1",
        )
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.preview_gate = PreviewGate()
        app.preview_gate.stage(draft)
        app.text = Mock()
        app.transcript_label_var = Mock()
        app.root = Mock()
        app._set_status = Mock()

        app._stage_ai_preview(draft.transaction_id, draft.fingerprint)

        app.text.insert.assert_not_called()
        self.assertIsNone(app.preview_gate.pending)
        app._set_status.assert_called_once_with(
            "Already good — cloud AI proposed no meaningful change",
            "ready",
        )

    def test_ai_preview_replaces_exact_recaptured_selection_once(self):
        self.assertTrue(hasattr(LocalVoiceApp, "_replace_ai_preview"))
        draft = PreviewDraft.create(
            original="old_name",
            instruction="Rename selection.",
            result="better_name",
            provider_label="Local • qwen3-coder-local",
            context_lenses=("Selected Text",),
            target_window=654,
            selection_identity="control-1",
            clock=lambda: 100.0,
            id_factory=lambda: "preview-replace-1",
        )
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.preview_gate = PreviewGate()
        app.preview_gate.stage(draft)
        app._target_is_active = Mock(return_value=True)
        app.selected_text_lens = Mock()
        app.selected_text_lens.capture.return_value = ContextLensCapture(
            lens_id="selected_text",
            included=True,
            content="old_name",
            source="uia_text_pattern",
            target_window=654,
            captured_at=101.0,
            original_characters=8,
            control_identity="control-1",
        )
        app._set_status = Mock()

        with patch("local_voice.type_unicode") as type_unicode_mock:
            with patch("local_voice.time.time", return_value=101.0):
                app._replace_ai_preview(draft.transaction_id, draft.fingerprint)
            type_unicode_mock.assert_called_once_with("better_name", 654)

        self.assertIsNone(app.preview_gate.pending)
        self.assertEqual(app.last_inserted_payload, "better_name")
        self.assertEqual(app.last_inserted_target, 654)
        app._set_status.assert_called_once_with(
            "Selection replaced from one-time approved preview",
            "ready",
        )

        with patch("local_voice.type_unicode") as type_unicode_mock:
            app._replace_ai_preview(draft.transaction_id, draft.fingerprint)
            type_unicode_mock.assert_not_called()

    def test_ai_preview_refuses_expired_selection_replacement(self):
        draft = PreviewDraft.create(
            original="old_name",
            instruction="Rename selection.",
            result="better_name",
            provider_label="Local • qwen3-coder-local",
            context_lenses=("Selected Text",),
            target_window=654,
            selection_identity="control-1",
            clock=lambda: 100.0,
            id_factory=lambda: "preview-expired-1",
        )
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.preview_gate = PreviewGate()
        app.preview_gate.stage(draft)
        app._target_is_active = Mock(return_value=True)
        app.selected_text_lens = Mock()
        app.selected_text_lens.capture.return_value = ContextLensCapture(
            lens_id="selected_text",
            included=True,
            content="old_name",
            source="uia_text_pattern",
            target_window=654,
            captured_at=401.0,
            original_characters=8,
            control_identity="control-1",
        )
        app._set_status = Mock()

        with patch("local_voice.time.time", return_value=401.0), patch(
            "local_voice.type_unicode"
        ) as type_unicode_mock:
            app._replace_ai_preview(draft.transaction_id, draft.fingerprint)
            type_unicode_mock.assert_not_called()

        self.assertIsNone(app.preview_gate.pending)
        app.selected_text_lens.capture.assert_not_called()
        app._set_status.assert_called_once_with(
            "Selection replacement approval expired — generate a new preview",
            "error",
        )

    def test_escape_cancels_exact_active_local_ai_request_once(self):
        self.assertTrue(hasattr(LocalVoiceApp, "_cancel_ai_request"))
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        signal = CancellationSignal()
        app.ai_cancellation = signal
        app._set_status = Mock()

        self.assertEqual(app._cancel_ai_request(), "break")
        self.assertTrue(signal.cancelled)
        app._set_status.assert_called_once_with(
            "Cancelling cloud AI request…",
            "working",
        )
        self.assertIsNone(app._cancel_ai_request())

    def test_explain_selection_records_timeout_without_staging_preview(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = SimpleNamespace(
            allow_auxiliary_models=True,
            auxiliary_model_name="qwen3-coder-local",
        )
        app._target_is_active = lambda _target: True
        app.selected_text_context = ContextLensCapture(
            lens_id="selected_text",
            included=True,
            content="ValueError: invalid token",
            source="uia_text_pattern",
            target_window=654,
            captured_at=123.0,
            original_characters=25,
        )
        app.ai_provider = Mock()
        app.ai_provider.complete.side_effect = ProviderTimeout(
            "Local AI timed out after 30 seconds"
        )
        app.ai_cancellation = None
        app.last_ai_request_timed_out = False
        app.preview_gate = PreviewGate()
        app.root = Mock()

        self.assertFalse(
            app._execute_voice_command(VoiceCommand("explain_selection"), 654)
        )
        self.assertTrue(app.last_ai_request_timed_out)
        self.assertIsNone(app.preview_gate.pending)
        app.root.after.assert_not_called()

    def test_bubble_visual_config_loads_with_safe_choices_and_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "bubble_font_family": "Missing Font",
                        "bubble_font_weight": "extra-heavy",
                        "bubble_visual_style": "unknown",
                        "bubble_text_color_mode": "invisible",
                        "bubble_corner_radius": 100,
                        "bubble_border_width": -7,
                    }
                ),
                encoding="utf-8",
            )

            with patch("local_voice.CONFIG_PATH", path):
                config = AppConfig.load()

        self.assertEqual(config.bubble_font_family, "Segoe UI")
        self.assertEqual(config.bubble_font_weight, "normal")
        self.assertEqual(config.bubble_visual_style, "glitch_glass")
        self.assertEqual(config.bubble_text_color_mode, "theme")
        self.assertEqual(config.bubble_corner_radius, 32)
        self.assertEqual(config.bubble_border_width, 0)

    def test_bubble_visual_settings_round_trip_through_live_runtime(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = AppConfig()
        app.settings_registry = SettingsRegistry(built_in_settings())
        app.performance_presets = PerformancePresetRegistry(
            built_in_performance_presets()
        )
        app.model_box = SimpleNamespace(configure=lambda **_kwargs: None)
        app.model_var = SimpleNamespace(set=lambda _value: None)
        app.hotkey_var = SimpleNamespace(set=lambda _value: None)
        app._change_model = lambda: None
        app._change_hotkey = lambda: None
        app._apply_theme = lambda: None
        app._select_config_values = lambda: None
        app.custom_vocabulary_var = SimpleNamespace(set=lambda _value: None)
        app.custom_replacements_var = SimpleNamespace(set=lambda _value: None)
        app.backend = SimpleNamespace(hotwords="")
        app.overlay = SimpleNamespace(width=0, height=0)
        app.streaming_bubble = unittest.mock.Mock()
        app.recording = False
        app.engine_ready = threading.Event()

        values = app._advanced_setting_values()
        values.update(
            {
                "recognition.device": "cuda:1",
                "recognition.vram_reserve_mb": 2048,
                "ui.bubble_font_family": "Cascadia Code",
                "ui.bubble_font_weight": "bold",
                "ui.bubble_visual_style": "neon_edge",
                "ui.bubble_text_color_mode": "accent",
                "ui.bubble_corner_radius": 28.0,
                "ui.bubble_border_width": 3.0,
            }
        )

        with patch.object(AppConfig, "save"):
            app._save_advanced_settings(values)

        self.assertEqual(app.config.bubble_font_family, "Cascadia Code")
        self.assertEqual(app.config.gpu_device_choice, "cuda:1")
        self.assertEqual(app.config.gpu_vram_reserve_mb, 2048)
        self.assertEqual(app.config.bubble_font_weight, "bold")
        self.assertEqual(app.config.bubble_visual_style, "neon_edge")
        self.assertEqual(app.config.bubble_text_color_mode, "accent")
        self.assertEqual(app.config.bubble_corner_radius, 28)
        self.assertEqual(app.config.bubble_border_width, 3)
        app.streaming_bubble.configure.assert_called_once_with(
            width=420,
            height=68,
            font_size=10,
            font_family="Cascadia Code",
            font_weight="bold",
            max_characters=120,
            opacity=0.82,
            visual_style="neon_edge",
            text_color_mode="accent",
            corner_radius=28,
            border_width=3,
        )

    def test_sendinput_structure_matches_native_windows_abi(self):
        expected_size = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        self.assertEqual(ctypes.sizeof(INPUT), expected_size)

    def test_runtime_trace_records_structured_metadata_without_transcript(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "runtime-events.jsonl"

            append_runtime_event(
                "partial",
                path=path,
                target_window=41,
                text_chars=12,
                target_safe=True,
            )

            event = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(event["event"], "partial")
            self.assertEqual(event["target_window"], 41)
            self.assertEqual(event["text_chars"], 12)
            self.assertNotIn("text", event)

    def test_runtime_events_log_rotates_instead_of_growing_unbounded(self):
        # Regression: runtime-events.jsonl had no size cap and was observed
        # to grow past 480MB in production from continuous UI-handoff and
        # resource-panel polling events. An unbounded append-only file that
        # size makes every write slower (and is a bigger target for
        # antivirus real-time scanning), which was implicated in some of
        # the multi-second hotkey/stop stalls Drew reported. This asserts
        # the active file is kept at or under the configured cap once it
        # would otherwise be exceeded, mirroring the sibling
        # TroubleshootingLog's existing rotation behavior.
        import local_voice as lv

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "runtime-events.jsonl"
            original_max_bytes = lv.RUNTIME_EVENTS_MAX_BYTES
            lv.RUNTIME_EVENTS_MAX_BYTES = 2048
            try:
                for index in range(400):
                    append_runtime_event(
                        "ui_handoff_completed",
                        path=path,
                        callback="LocalVoiceApp._refresh_resource_panel",
                        index=index,
                    )
                self.assertLessEqual(path.stat().st_size, lv.RUNTIME_EVENTS_MAX_BYTES)
                archive = path.with_suffix(path.suffix + ".1")
                self.assertTrue(archive.exists())
                # The most recent event must still be present in the active
                # (post-rotation) file, not lost.
                last_line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
                last_event = json.loads(last_line)
                self.assertEqual(last_event["index"], 399)
            finally:
                lv.RUNTIME_EVENTS_MAX_BYTES = original_max_bytes

    def test_runtime_events_rotation_disabled_when_cap_is_zero(self):
        import local_voice as lv

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "runtime-events.jsonl"
            original_max_bytes = lv.RUNTIME_EVENTS_MAX_BYTES
            lv.RUNTIME_EVENTS_MAX_BYTES = 0
            try:
                for index in range(50):
                    append_runtime_event("ui_handoff_completed", path=path, index=index)
                archive = path.with_suffix(path.suffix + ".1")
                self.assertFalse(archive.exists())
                lines = path.read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(lines), 50)
            finally:
                lv.RUNTIME_EVENTS_MAX_BYTES = original_max_bytes

    def test_engine_ready_diagnostics_record_vram_decision_without_content(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app._pending_model_rollback = None
        app.pending_hotkey_start = None
        app.runtime_profile = RuntimeProfile(
            "distil-large-v3.5",
            "cuda",
            "int8_float16",
            1,
            "High accuracy — test",
            free_vram_mb=6600,
            required_vram_mb=6144,
            vram_reserve_mb=2048,
        )
        app.model_var = SimpleNamespace(set=lambda _value: None)
        app.metrics_var = SimpleNamespace(set=lambda _value: None)
        app.config = SimpleNamespace(hotkey="Ctrl + Alt + Space", show_overlay=False)
        app._set_status = Mock()

        with patch("local_voice.append_runtime_event") as append:
            app._engine_ready_ui(
                {"load_seconds": 0.5, "warm_seconds": 0.25},
                "",
            )

        append.assert_called_once_with(
            "engine_ready",
            pid=os.getpid(),
            model="distil-large-v3.5",
            backend="faster-whisper",
            device="cuda",
            device_index=1,
            free_vram_mb=6600,
            required_vram_mb=6144,
            vram_reserve_mb=2048,
            used_fallback=False,
        )

    def test_gpu_quick_selection_persists_canonical_value_and_previews_next_start(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = AppConfig(model_name="tiny.en", gpu_device_choice="automatic")
        app.config.advanced_settings = {}
        app.gpu_inventory = local_voice_module.parse_nvidia_smi_gpu_inventory(
            "0, NVIDIA A, GPU-A, 8000, 1000, 7000, 0\n"
            "2, NVIDIA B, GPU-B, 12000, 2000, 10000, 0\n"
        )
        app.gpu_display_to_value = {"CUDA 2 — NVIDIA B": "cuda:2"}
        app.gpu_value_to_display = {
            "automatic": "Automatic (best available)",
            "cuda:2": "CUDA 2 — NVIDIA B",
        }
        app.gpu_device_var = SimpleNamespace(
            get=lambda: "CUDA 2 — NVIDIA B",
            set=Mock(),
        )
        app.gpu_runtime_var = SimpleNamespace(set=Mock())
        app.recording = False
        app.busy = False
        app._set_status = Mock()

        with patch.object(AppConfig, "save") as save:
            app._change_gpu_device()

        self.assertEqual(app.config.gpu_device_choice, "cuda:2")
        self.assertEqual(app.config.advanced_settings["recognition.device"], "cuda:2")
        save.assert_called_once_with()
        app.gpu_runtime_var.set.assert_called_once()
        self.assertIn("NEXT START", app.gpu_runtime_var.set.call_args.args[0])
        app._set_status.assert_called_once()
        self.assertIn("restart", app._set_status.call_args.args[0].lower())

    def test_gpu_quick_selection_rejects_changes_while_busy(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = AppConfig(gpu_device_choice="automatic")
        app.gpu_display_to_value = {"CPU only": "cpu"}
        app.gpu_value_to_display = {
            "automatic": "Automatic (best available)",
        }
        app.gpu_device_var = SimpleNamespace(get=lambda: "CPU only", set=Mock())
        app.recording = False
        app.busy = True
        app._set_status = Mock()

        with patch.object(AppConfig, "save") as save:
            app._change_gpu_device()

        self.assertEqual(app.config.gpu_device_choice, "automatic")
        app.gpu_device_var.set.assert_called_once_with("Automatic (best available)")
        save.assert_not_called()
        self.assertEqual(app._set_status.call_args.args[1], "error")

    def test_engine_ready_diagnostics_marks_resolver_fallback_but_not_explicit_cpu(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app._pending_model_rollback = None
        app.pending_hotkey_start = None
        app.model_var = SimpleNamespace(set=lambda _value: None)
        app.metrics_var = SimpleNamespace(set=lambda _value: None)
        app.gpu_runtime_var = SimpleNamespace(set=Mock())
        app.gpu_inventory = ()
        app.config = SimpleNamespace(hotkey="Ctrl + Alt + Space", show_overlay=False)
        app._set_status = Mock()
        fallback = RuntimeProfile(
            "tiny.en",
            "cpu",
            "int8",
            None,
            "CPU fallback",
            fallback_reason="No CUDA device is available",
        )
        explicit = RuntimeProfile("tiny.en", "cpu", "int8", None, "CPU selected")

        with patch("local_voice.append_runtime_event") as append:
            app.runtime_profile = fallback
            app._engine_ready_ui(
                {"load_seconds": 0.1, "warm_seconds": 0.1},
                "",
            )
            self.assertTrue(append.call_args.kwargs["used_fallback"])
            app.runtime_profile = explicit
            app._engine_ready_ui(
                {"load_seconds": 0.1, "warm_seconds": 0.1},
                "",
            )
            self.assertFalse(append.call_args.kwargs["used_fallback"])

    def test_stable_partial_streams_into_original_active_chat_target(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.recording = True
        app.speech_end_detector = None
        app.last_recognized_partial = ""
        app._prepare_dictated_text = lambda text: text
        app.transcript_label_var = SimpleNamespace(set=lambda value: None)
        app.text = SimpleNamespace(
            delete=lambda *args: None,
            insert=lambda *args: None,
        )
        app.metrics_var = SimpleNamespace(set=lambda value: None)
        app.config = SimpleNamespace(
            show_overlay=False,
            hotkey_auto_insert=True,
            live_insert=True,
        )
        app.command_debouncer = SimpleNamespace(update=lambda text: None)
        app.hotkey_started_recording = True
        app.live_insertion_suspended = False
        app.live_committer = SimpleNamespace(
            accept_stable=lambda text: SimpleNamespace(
                conflict=False,
                text_to_insert="hello",
            )
        )
        app.target_window = 41
        app._target_is_active = lambda target: target == 41
        app.session_live_payload = ""

        result = StreamResult(
            text="hello world",
            processed_samples=32_000,
            inference_seconds=0.08,
            completed_at=1.0,
            stable_text="hello",
        )
        with patch("local_voice.type_unicode") as type_unicode:
            app._show_partial(result)

        type_unicode.assert_called_once_with("hello", 41)
        self.assertEqual(app.session_live_payload, "hello")

    def test_review_mode_previews_partial_without_typing_then_commits_complete_final(self):
        class Widget:
            def configure(self, **_kwargs):
                pass

            def delete(self, *_args):
                pass

            def insert(self, *_args):
                pass

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.recording = True
        app.speech_end_detector = None
        app.last_recognized_partial = ""
        app.transcript_label_var = SimpleNamespace(set=lambda _value: None)
        app.metrics_var = SimpleNamespace(set=lambda _value: None)
        app.text = Widget()
        app.record_button = Widget()
        app.config = AppConfig(show_overlay=False, live_insert=False)
        app.active_dictation_profile = None
        app.command_debouncer = SimpleNamespace(update=lambda _text: None)
        app.hotkey_started_recording = True
        app.live_insertion_suspended = False
        app.live_committer = LiveCommitter(holdback_words=2)
        app.target_window = 41
        app._target_is_active = lambda target: target == 41
        app._set_status = lambda *_args: None
        app.session_live_payload = ""
        app.session_preceding_text = ""
        app.last_inserted_payload = ""
        app.last_inserted_target = 0
        app.last_inserted_at = 0.0

        partial = StreamResult(
            text="all of the visible ending",
            processed_samples=32_000,
            inference_seconds=0.08,
            completed_at=1.0,
            stable_text="all of the",
        )
        with patch("local_voice.type_unicode") as send:
            app._show_partial(partial)
            send.assert_not_called()
            app._finish_transcription(
                "all of the visible ending is committed",
                {"stop_to_text_seconds": 0.01, "reused_partial": False},
                0.5,
                True,
                41,
            )

        send.assert_called_once_with(
            "all of the visible ending is committed ",
            41,
        )

    def test_main_hides_large_window_before_building_application(self):
        events = []

        class Root:
            def withdraw(self):
                events.append("withdraw")

            def mainloop(self):
                events.append("mainloop")

        root = Root()
        with patch("local_voice.tk.Tk", return_value=root), patch(
            "local_voice.LocalVoiceApp",
            side_effect=lambda current: events.append("build"),
        ):
            main()

        self.assertEqual(events, ["withdraw", "build", "mainloop"])

    def test_silence_auto_stop_never_arms_before_real_speech(self):
        detector = SpeechEndDetector(
            silence_seconds=2.0,
            min_speech_seconds=0.3,
            rms_threshold=0.001,
        )

        self.assertFalse(detector.observe(0.0002, now=0.0))
        self.assertFalse(detector.observe(0.0002, now=10.0))

    def test_default_speech_threshold_rejects_measured_headset_idle_level(self):
        detector = SpeechEndDetector(silence_seconds=1.0)

        for step in range(20):
            self.assertFalse(detector.observe(0.0011, now=step * 0.1))

        self.assertFalse(detector.speech_started)

    def test_recognizer_armed_detector_ignores_noise_until_text_exists(self):
        detector = SpeechEndDetector(
            silence_seconds=1.0,
            require_recognized_speech=True,
        )

        for step in range(20):
            self.assertFalse(detector.observe(0.02, now=step * 0.1))
        self.assertFalse(detector.speech_started)

        detector.mark_recognized_speech(now=2.0)
        self.assertTrue(detector.speech_started)
        self.assertFalse(detector.observe(0.0001, now=2.9))
        self.assertTrue(detector.observe(0.0001, now=3.01))

    def test_silence_auto_stop_requires_sustained_speech_then_quiet(self):
        detector = SpeechEndDetector(
            silence_seconds=2.0,
            min_speech_seconds=0.3,
            rms_threshold=0.001,
        )

        self.assertFalse(detector.observe(0.005, now=0.0))
        self.assertFalse(detector.observe(0.005, now=0.2))
        self.assertFalse(detector.observe(0.005, now=0.35))
        self.assertFalse(detector.observe(0.0002, now=2.30))
        self.assertTrue(detector.observe(0.0002, now=2.36))

    def test_silence_auto_stop_ignores_short_noise_and_resets_for_more_speech(self):
        detector = SpeechEndDetector(
            silence_seconds=2.0,
            min_speech_seconds=0.3,
            rms_threshold=0.001,
        )

        detector.observe(0.005, now=0.0)
        detector.observe(0.0002, now=0.1)
        self.assertFalse(detector.observe(0.0002, now=5.0))
        detector.observe(0.005, now=6.0)
        detector.observe(0.005, now=6.31)
        self.assertFalse(detector.observe(0.0002, now=8.0))
        self.assertFalse(detector.observe(0.005, now=8.1))
        self.assertFalse(detector.observe(0.0002, now=10.0))
        self.assertTrue(detector.observe(0.0002, now=10.11))

    def test_hotkey_choices_resolve_to_windows_modifiers_and_virtual_keys(self):
        self.assertEqual(resolve_hotkey("Ctrl + Alt + Space"), (0x0003, 0x20))
        self.assertEqual(resolve_hotkey("Ctrl + Shift + A"), (0x0006, 0x41))
        self.assertEqual(resolve_hotkey("F8"), (0, 0x77))
        self.assertEqual(resolve_hotkey("F12"), (0, 0x7B))
        with self.assertRaises(ValueError):
            resolve_hotkey("Ctrl + Shift + Mystery")

    def test_hotkey_replacement_restores_previous_binding_on_conflict(self):
        events = []

        class FakeHotkey:
            def __init__(self, callback, label):
                self.label = label
                self.error = "already used" if label == "F8" else None

            def start(self):
                events.append(("start", self.label))

            def stop(self):
                events.append(("stop", self.label))

        current = FakeHotkey(lambda: None, "Ctrl + Alt + Space")
        active, applied, error = replace_hotkey(
            current,
            "F8",
            "Ctrl + Alt + Space",
            lambda: None,
            factory=FakeHotkey,
        )

        self.assertEqual(applied, "Ctrl + Alt + Space")
        self.assertEqual(active.label, "Ctrl + Alt + Space")
        self.assertEqual(error, "already used")
        self.assertEqual(
            events,
            [
                ("stop", "Ctrl + Alt + Space"),
                ("start", "F8"),
                ("stop", "F8"),
                ("start", "Ctrl + Alt + Space"),
            ],
        )

    def test_hotkey_replacement_keeps_available_requested_binding(self):
        class FakeHotkey:
            def __init__(self, callback, label):
                self.label = label
                self.error = None

            def start(self):
                pass

            def stop(self):
                pass

        current = FakeHotkey(lambda: None, "Ctrl + Alt + Space")
        active, applied, error = replace_hotkey(
            current,
            "F9",
            "Ctrl + Alt + Space",
            lambda: None,
            factory=FakeHotkey,
        )

        self.assertEqual(active.label, "F9")
        self.assertEqual(applied, "F9")
        self.assertIsNone(error)

    @unittest.skipUnless(os.name == "nt", "Windows UI Automation only")
    def test_native_uia_constructor_is_available_without_a_progid(self):
        automation = create_uia_automation()

        self.assertIsNotNone(automation)

    def test_live_insert_defaults_to_review_then_complete_commit(self):
        self.assertFalse(AppConfig().live_insert)

    def test_save_settings_preserves_selected_timestamp_live_value(self):
        class Value:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        saved = []
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.device_var = Value("missing")
        app.devices = []
        app.runtime_profile = SimpleNamespace(model_name="base.en")
        app.live_insert_var = Value(True)
        app.overlay_var = Value(True)
        app.auto_punctuation_var = Value(True)
        app.profanity_var = Value(False)
        app.auto_stop_var = Value(True)
        app.silence_wait_var = Value("Normal (2.0 s)")
        app.command_wait_var = Value("Fast")
        app.config = AppConfig()
        app.config.save = lambda: saved.append(True)

        app._save_settings()

        self.assertTrue(app.config.live_insert)
        self.assertTrue(app.live_insert_var.get())
        self.assertEqual(saved, [True])

    def test_insertion_target_fails_closed_for_password_or_focus_change(self):
        foreground = [41, 41]

        self.assertFalse(
            insertion_target_is_safe(
                41,
                inspector=lambda: FocusedTargetInfo(True, True, True),
                foreground_getter=lambda: foreground.pop(0),
            )
        )
        self.assertFalse(
            insertion_target_is_safe(
                41,
                inspector=lambda: FocusedTargetInfo(False, False, False),
                foreground_getter=lambda: 41,
            )
        )
        changed = iter((41, 99))
        self.assertFalse(
            insertion_target_is_safe(
                41,
                inspector=lambda: FocusedTargetInfo(True, False, True),
                foreground_getter=lambda: next(changed),
            )
        )

    def test_insertion_target_allows_same_accessible_edit_control(self):
        self.assertTrue(
            insertion_target_is_safe(
                41,
                inspector=lambda: FocusedTargetInfo(True, False, True),
                foreground_getter=lambda: 41,
            )
        )

    def test_insertion_target_rejects_active_ime_composition(self):
        self.assertFalse(
            insertion_target_is_safe(
                41,
                inspector=lambda: FocusedTargetInfo(True, False, True, True),
                foreground_getter=lambda: 41,
            )
        )

    def test_backend_instances_share_one_inference_lock(self):
        state_lock = threading.Lock()
        release = threading.Event()
        state = {"concurrent": 0, "maximum": 0}

        class SlowModel:
            def transcribe(self, audio, **kwargs):
                with state_lock:
                    state["concurrent"] += 1
                    state["maximum"] = max(state["maximum"], state["concurrent"])
                release.wait(timeout=1)
                with state_lock:
                    state["concurrent"] -= 1
                return [], SimpleNamespace(language="en", language_probability=1.0)

        profile = RuntimeProfile("base.en", "cpu", "int8", None, "test")
        backends = [WhisperBackend(model_factory=lambda *a, **k: SlowModel()) for _ in range(2)]
        workers = [
            threading.Thread(
                target=backend.transcribe_samples,
                args=(np.zeros(100, dtype=np.float32), profile),
            )
            for backend in backends
        ]
        for worker in workers:
            worker.start()
        time.sleep(0.05)
        release.set()
        for worker in workers:
            worker.join(timeout=1)

        self.assertEqual(state["maximum"], 1)

    def test_saved_microphone_name_wins_over_stale_device_index(self):
        devices = [
            {
                "index": 91,
                "name": "Microphone (3- onn Wired Gaming Headset)",
                "api": "Windows WASAPI",
            },
            {
                "index": 93,
                "name": "Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)",
                "api": "Windows WASAPI",
            },
        ]

        selected = choose_saved_device(
            devices,
            saved_name="Microphone (onn Wired Gaming Headset)",
            saved_index=93,
        )

        self.assertEqual(selected, 91)

    def test_resample_audio_converts_native_48k_capture_to_16k(self):
        source = np.full(4_800, 0.25, dtype=np.float32)

        converted = resample_audio(source, source_rate=48_000, target_rate=16_000)

        self.assertEqual(converted.dtype, np.float32)
        self.assertEqual(converted.size, 1_600)
        np.testing.assert_allclose(converted, 0.25, atol=1e-6)

    def test_recorder_snapshot_resamples_native_capture_rate(self):
        recorder = AudioRecorder(
            device_index=91, sample_rate=16_000, capture_rate=48_000
        )
        recorder._callback(
            np.ones((4_800, 1), dtype=np.float32), 4_800, None, None
        )

        snapshot = recorder.snapshot()

        self.assertEqual(snapshot.size, 1_600)

    def test_gpu_probe_parses_free_memory_and_fails_closed(self):
        class Result:
            stdout = "8104\n"

        self.assertEqual(
            probe_gpu_free_vram_mb(runner=lambda *args, **kwargs: Result()), 8104
        )

        def failed_runner(*args, **kwargs):
            raise OSError("nvidia-smi unavailable")

        self.assertEqual(probe_gpu_free_vram_mb(runner=failed_runner), 0)

    def test_audio_recorder_snapshot_is_ordered_and_non_destructive(self):
        recorder = AudioRecorder(device_index=0)
        recorder._callback(
            np.array([[0.1], [0.2]], dtype=np.float32), 2, None, None
        )
        recorder._callback(
            np.array([[0.3], [0.4]], dtype=np.float32), 2, None, None
        )

        first = recorder.snapshot()
        second = recorder.snapshot()

        np.testing.assert_allclose(first, [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_equal(second, first)
        self.assertIsNot(first, second)

    def test_audio_recorder_incremental_downsample_keeps_samples_across_chunk_boundaries(self):
        recorder = AudioRecorder(
            device_index=0,
            sample_rate=16_000,
            capture_rate=48_000,
        )
        first = np.arange(2_999, dtype=np.float32).reshape(-1, 1)
        second = np.arange(2_999, 6_002, dtype=np.float32).reshape(-1, 1)

        recorder._callback(first, len(first), None, None)
        first_snapshot = recorder.snapshot()
        recorder._callback(second, len(second), None, None)
        second_snapshot = recorder.snapshot()

        expected = np.arange(6_000, dtype=np.float32).reshape(-1, 3).mean(axis=1)
        self.assertEqual(first_snapshot.size, 999)
        self.assertEqual(second_snapshot.size, 2_000)
        np.testing.assert_allclose(second_snapshot, expected)

    def test_backend_transcribes_raw_float32_audio_with_profile(self):
        calls = []

        class FakeModel:
            def transcribe(self, audio, **kwargs):
                calls.append((audio, kwargs))
                return [
                    SimpleNamespace(
                        text=" instant local text ",
                        words=[
                            SimpleNamespace(
                                word=" instant", start=0.0, end=0.3, probability=0.98
                            ),
                            SimpleNamespace(
                                word=" local", start=0.3, end=0.6, probability=0.97
                            ),
                            SimpleNamespace(
                                word=" text", start=0.6, end=0.9, probability=0.96
                            ),
                        ],
                    )
                ], SimpleNamespace(
                    language="en", language_probability=1.0
                )

        factory_calls = []

        def factory(model_name, **kwargs):
            factory_calls.append((model_name, kwargs))
            return FakeModel()

        profile = RuntimeProfile("base.en", "cuda", "int8_float16", 0, "Instant")
        backend = WhisperBackend(model_factory=factory)
        text, metrics = backend.transcribe_samples(
            np.ones(8_000, dtype=np.float64), profile, fast=True
        )

        self.assertEqual(text, "instant local text")
        self.assertEqual(factory_calls[0][0], "base.en")
        self.assertEqual(factory_calls[0][1]["device"], "cuda")
        self.assertEqual(factory_calls[0][1]["device_index"], 0)
        self.assertEqual(factory_calls[0][1]["compute_type"], "int8_float16")
        self.assertIsInstance(calls[0][0], np.ndarray)
        self.assertEqual(calls[0][0].dtype, np.float32)
        self.assertEqual(calls[0][1]["beam_size"], 1)
        self.assertTrue(calls[0][1]["word_timestamps"])
        self.assertFalse(calls[0][1]["without_timestamps"])
        self.assertEqual(
            metrics["words"],
            (
                {"word": " instant", "start": 0.0, "end": 0.3, "probability": 0.98},
                {"word": " local", "start": 0.3, "end": 0.6, "probability": 0.97},
                {"word": " text", "start": 0.6, "end": 0.9, "probability": 0.96},
            ),
        )
        self.assertGreaterEqual(metrics["load_seconds"], 0.0)

    def test_backend_preload_reaches_ready_without_synthetic_inference(self):
        calls = []

        class FakeModel:
            def transcribe(self, audio, **kwargs):
                calls.append(np.asarray(audio))
                return [], SimpleNamespace(language="en", language_probability=1.0)

        profile = RuntimeProfile("base.en", "cuda", "int8_float16", 0, "Instant")
        backend = WhisperBackend(model_factory=lambda *args, **kwargs: FakeModel())
        metrics = backend.preload_and_warm(profile)

        self.assertEqual(calls, [])
        self.assertTrue(metrics["ready"])
        self.assertEqual(metrics["warm_seconds"], 0.0)

    def test_fast_core_defaults_to_nemotron_with_lean_core_streaming(self):
        # Contract: Nemotron is the explicit default primary model; the
        # lean_core preset still ships with progressive streaming enabled.
        # The "tiny" whisper model is no longer the auto-selected default.
        from drew_local_voice.models import NEMOTRON_MODEL_ID

        specs = {spec.key: spec for spec in built_in_settings()}

        self.assertEqual(AppConfig().model_name, NEMOTRON_MODEL_ID)
        self.assertEqual(AppConfig().performance_preset, "lean_core")
        self.assertEqual(specs["recognition.model"].default, NEMOTRON_MODEL_ID)
        self.assertEqual(specs["performance.preset"].default, "lean_core")
        self.assertTrue(specs["streaming.enabled"].default)

    def test_runtime_profile_uses_guarded_gpu_for_instant_mode(self):
        profile = choose_runtime_profile(free_vram_mb=4096, cuda_available=True)
        self.assertEqual(profile.device, "cuda")
        self.assertEqual(profile.device_index, 0)
        self.assertEqual(profile.compute_type, "int8_float16")
        self.assertEqual(profile.model_name, "base.en")

    def test_gpu_inventory_parses_devices_and_selects_only_safe_headroom(self):
        parse_inventory = getattr(
            local_voice_module, "parse_nvidia_smi_gpu_inventory", None
        )
        select_device = getattr(local_voice_module, "select_gpu_device", None)
        self.assertIsNotNone(parse_inventory)
        self.assertIsNotNone(select_device)
        devices = parse_inventory(
            "0, NVIDIA GeForce RTX 5060 Ti, GPU-A, 16311, 11795, 4516, 61\n"
            "1, NVIDIA GeForce RTX 2070 SUPER, GPU-B, 8192, 7520, 672, 0\n"
        )

        self.assertEqual([device.index for device in devices], [0, 1])
        self.assertEqual(devices[1].name, "NVIDIA GeForce RTX 2070 SUPER")
        automatic, reason = select_device(
            devices, "automatic", required_vram_mb=4096
        )
        blocked_2070, blocked_reason = select_device(
            devices, "cuda:1", required_vram_mb=4096
        )

        self.assertEqual(automatic.index, 0)
        self.assertIn("5060 Ti", reason)
        self.assertIsNone(blocked_2070)
        self.assertIn("672 MiB free", blocked_reason)
        self.assertIn("4096 MiB required", blocked_reason)

    def test_gpu_probes_hide_nvidia_smi_console_on_windows(self):
        inventory_runner = Mock(
            return_value=SimpleNamespace(
                stdout="0, NVIDIA A, GPU-A, 8000, 1000, 7000, 0\n"
            )
        )
        free_runner = Mock(return_value=SimpleNamespace(stdout="7000\n"))

        local_voice_module.probe_gpu_inventory(runner=inventory_runner)
        local_voice_module.probe_gpu_free_vram_mb(0, runner=free_runner)

        expected = local_voice_module.subprocess.CREATE_NO_WINDOW
        self.assertEqual(
            inventory_runner.call_args.kwargs["creationflags"] & expected,
            expected,
        )
        self.assertEqual(
            free_runner.call_args.kwargs["creationflags"] & expected,
            expected,
        )

    def test_gpu_setting_enumerates_every_detected_card_without_fixed_indices(self):
        devices = local_voice_module.parse_nvidia_smi_gpu_inventory(
            "0, NVIDIA A, GPU-A, 8000, 1000, 7000, 0\n"
            "2, NVIDIA B, GPU-B, 12000, 2000, 10000, 0\n"
            "7, NVIDIA C, GPU-C, 24000, 3000, 21000, 0\n"
        )
        registry = SettingsRegistry(built_in_settings())

        choices = local_voice_module.configure_gpu_device_setting(
            registry,
            devices,
        )

        self.assertEqual(
            choices,
            ("automatic", "cuda:0", "cuda:2", "cuda:7", "cpu"),
        )
        spec = registry.get("recognition.device")
        self.assertEqual(spec.choices, choices)
        self.assertIn("CUDA 7: NVIDIA C", spec.guidance)

        self.assertEqual(
            local_voice_module.gpu_device_display_options(devices),
            (
                ("Automatic (best available)", "automatic"),
                ("CUDA 0 — NVIDIA A", "cuda:0"),
                ("CUDA 2 — NVIDIA B", "cuda:2"),
                ("CUDA 7 — NVIDIA C", "cuda:7"),
                ("CPU only", "cpu"),
            ),
        )

    def test_runtime_device_status_names_active_card_and_honest_cpu_fallback(self):
        devices = local_voice_module.parse_nvidia_smi_gpu_inventory(
            "0, NVIDIA A, GPU-A, 8000, 1000, 7000, 0\n"
            "2, NVIDIA B, GPU-B, 12000, 2000, 10000, 0\n"
        )
        active = RuntimeProfile(
            "distil-large-v3.5",
            "cuda",
            "int8_float16",
            2,
            "High accuracy",
            free_vram_mb=10000,
            required_vram_mb=6144,
            vram_reserve_mb=2048,
        )
        blocked = local_voice_module.choose_runtime_profile_from_inventory(
            devices,
            "cuda:0",
            model_name="distil-large-v3.5",
            vram_reserve_mb=4096,
        )
        explicit_cpu = local_voice_module.choose_runtime_profile_from_inventory(
            devices,
            "cpu",
            model_name="distil-large-v3.5",
        )

        active_status = local_voice_module.runtime_profile_device_status(
            active,
            devices,
        )
        blocked_status = local_voice_module.runtime_profile_device_status(
            blocked,
            devices,
        )
        self.assertTrue(active_status.startswith("ACTIVE •"))
        self.assertIn("NVIDIA B", active_status)
        self.assertIn("CUDA 2", active_status)
        self.assertIn("10000 MiB free / 6144 MiB required", active_status)
        self.assertTrue(blocked.fallback_reason)
        self.assertEqual(
            (blocked.device, blocked.model_name),
            ("blocked", "distil-large-v3.5"),
        )
        self.assertFalse(blocked.runnable)
        self.assertTrue(blocked_status.startswith("BLOCKED •"))
        self.assertIn("7000 MiB free", blocked_status)
        self.assertIn("distil", blocked_status.lower())
        self.assertEqual(explicit_cpu.fallback_reason, "")

    def test_runtime_resolver_honors_explicit_cpu_and_reports_blocked_gpu(self):
        resolve_profile = getattr(
            local_voice_module, "choose_runtime_profile_from_inventory", None
        )
        self.assertIsNotNone(resolve_profile)
        devices = local_voice_module.parse_nvidia_smi_gpu_inventory(
            "0, NVIDIA GeForce RTX 5060 Ti, GPU-A, 16311, 11795, 4516, 61\n"
            "1, NVIDIA GeForce RTX 2070 SUPER, GPU-B, 8192, 7520, 672, 0\n"
        )

        automatic = resolve_profile(
            devices, "automatic", model_name="distil-large-v3.5"
        )
        explicit_cpu = resolve_profile(
            devices, "cpu", model_name="distil-large-v3.5"
        )
        blocked = resolve_profile(
            devices, "cuda:1", model_name="distil-large-v3.5"
        )

        self.assertEqual((automatic.device, automatic.device_index), ("cuda", 0))
        self.assertEqual(
            (explicit_cpu.device, explicit_cpu.model_name),
            ("cpu", "distil-large-v3.5"),
        )
        self.assertEqual(blocked.device, "blocked")
        self.assertEqual(blocked.model_name, "distil-large-v3.5")
        self.assertFalse(blocked.runnable)
        self.assertIn("672 MiB free", blocked.label)

    def test_runtime_resolver_adds_configured_reserve_to_model_estimate(self):
        devices = local_voice_module.parse_nvidia_smi_gpu_inventory(
            "0, NVIDIA GeForce RTX 5060 Ti, GPU-A, 16311, 10311, 6000, 20\n"
            "1, NVIDIA GeForce RTX 2070 SUPER, GPU-B, 8192, 1792, 6400, 0\n"
        )

        automatic = local_voice_module.choose_runtime_profile_from_inventory(
            devices,
            "automatic",
            model_name="distil-large-v3.5",
            vram_reserve_mb=2048,
        )
        blocked = local_voice_module.choose_runtime_profile_from_inventory(
            devices,
            "cuda:0",
            model_name="distil-large-v3.5",
            vram_reserve_mb=2048,
        )

        self.assertEqual(
            (automatic.device, automatic.model_name),
            ("cuda", "distil-large-v3.5"),
        )
        self.assertEqual(automatic.device_index, 0)
        self.assertEqual(automatic.fallback_reason, "")
        self.assertEqual(automatic.required_vram_mb, 5120)
        self.assertEqual(automatic.free_vram_mb, 6000)
        self.assertEqual(automatic.vram_reserve_mb, 2048)
        self.assertIn("RTX 5060 Ti", automatic.label)
        self.assertEqual(
            (blocked.device, blocked.model_name),
            ("cuda", "distil-large-v3.5"),
        )
        self.assertEqual(blocked.device_index, 0)
        self.assertEqual(blocked.required_vram_mb, 5120)
        self.assertEqual(blocked.free_vram_mb, 6000)

    def test_runtime_resolver_quarantines_distil_on_rtx_2070_super(self):
        devices = local_voice_module.parse_nvidia_smi_gpu_inventory(
            "0, NVIDIA GeForce RTX 5060 Ti, GPU-A, 16311, 13000, 3311, 20\n"
            "1, NVIDIA GeForce RTX 2070 SUPER, GPU-B, 8192, 500, 7692, 0\n"
        )

        explicit = local_voice_module.choose_runtime_profile_from_inventory(
            devices,
            "cuda:1",
            model_name="distil-large-v3.5",
        )
        automatic = local_voice_module.choose_runtime_profile_from_inventory(
            devices,
            "automatic",
            model_name="distil-large-v3.5",
        )
        safe_small_model = local_voice_module.choose_runtime_profile_from_inventory(
            devices,
            "cuda:1",
            model_name="tiny.en",
        )

        self.assertEqual(
            (explicit.device, explicit.model_name),
            ("blocked", "distil-large-v3.5"),
        )
        self.assertIsNone(explicit.device_index)
        self.assertFalse(explicit.runnable)
        self.assertTrue(explicit.fallback_reason)
        self.assertEqual(
            (automatic.device, automatic.model_name),
            ("cuda", "distil-large-v3.5"),
        )
        self.assertEqual(automatic.device_index, 0)
        self.assertEqual(automatic.fallback_reason, "")
        self.assertEqual(
            (safe_small_model.device, safe_small_model.device_index),
            ("cuda", 1),
        )

    def test_runtime_resolver_uses_safe_alternate_gpu_for_automatic_distil(self):
        devices = local_voice_module.parse_nvidia_smi_gpu_inventory(
            "0, NVIDIA GeForce RTX 5060 Ti, GPU-A, 16311, 9000, 7311, 20\n"
            "1, NVIDIA GeForce RTX 2070 SUPER, GPU-B, 8192, 500, 7692, 0\n"
        )

        automatic = local_voice_module.choose_runtime_profile_from_inventory(
            devices,
            "automatic",
            model_name="distil-large-v3.5",
        )

        self.assertEqual(
            (automatic.device, automatic.device_index),
            ("cuda", 0),
        )

    def test_runtime_resolver_keeps_distil_on_gpu_with_corrected_footprint(self):
        devices = local_voice_module.parse_nvidia_smi_gpu_inventory(
            "0, NVIDIA GeForce RTX 5060 Ti, GPU-0, 16311, 11611, 4700, 29\n"
            "1, NVIDIA GeForce RTX 2070 SUPER, GPU-1, 8192, 7722, 470, 0\n"
        )

        profile = local_voice_module.choose_runtime_profile_from_inventory(
            devices=devices,
            requested_device="automatic",
            model_name="distil-large-v3.5",
            minimum_free_vram_mb=2048,
            vram_reserve_mb=1024,
        )

        self.assertEqual(profile.device, "cuda")
        self.assertEqual(profile.device_index, 0)
        self.assertEqual(profile.model_name, "distil-large-v3.5")
        self.assertEqual(profile.fallback_reason, "")

    def test_runtime_profile_uses_selected_distil_model_with_larger_guard(self):
        profile = choose_runtime_profile(
            free_vram_mb=7893,
            cuda_available=True,
            model_name="distil-large-v3.5",
        )
        self.assertEqual(profile.device, "cuda")
        self.assertEqual(profile.device_index, 0)
        self.assertEqual(profile.compute_type, "int8_float16")
        self.assertEqual(profile.model_name, "distil-large-v3.5")

        fallback = choose_runtime_profile(
            free_vram_mb=2500,
            cuda_available=True,
            model_name="distil-large-v3.5",
        )
        self.assertEqual(fallback.device, "blocked")
        self.assertEqual(fallback.model_name, "distil-large-v3.5")
        self.assertFalse(fallback.runnable)
        self.assertTrue(fallback.fallback_reason)

    def test_runtime_profile_names_and_uses_an_explicit_cuda_device(self):
        profile = choose_runtime_profile(
            free_vram_mb=6000,
            cuda_available=True,
            model_name="distil-large-v3.5",
            device_index=1,
            device_name="NVIDIA GeForce RTX 2070 SUPER",
        )

        self.assertEqual(profile.device, "cuda")
        self.assertEqual(profile.device_index, 1)
        self.assertIn("RTX 2070 SUPER", profile.label)
        self.assertIn("CUDA 1", profile.label)

    def test_app_runtime_profile_reprobes_the_configured_gpu(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = AppConfig(
            gpu_device_choice="cuda:1",
            gpu_vram_reserve_mb=2048,
        )
        devices = local_voice_module.parse_nvidia_smi_gpu_inventory(
            "0, NVIDIA GeForce RTX 5060 Ti, GPU-A, 16311, 11795, 4516, 61\n"
            "1, NVIDIA GeForce RTX 3070, GPU-B, 8192, 1592, 6600, 0\n"
        )

        with patch("local_voice.probe_gpu_inventory", return_value=devices) as probe:
            profile = app._runtime_profile_for_model("distil-large-v3.5")

        probe.assert_called_once_with()
        self.assertEqual((profile.device, profile.device_index), ("cuda", 1))
        self.assertEqual(profile.required_vram_mb, 5120)
        self.assertEqual(profile.vram_reserve_mb, 2048)
        self.assertIn("RTX 3070", profile.label)

    def test_start_recording_queues_hotkey_input_until_engine_is_ready(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.engine_ready = threading.Event()
        app.config = SimpleNamespace(show_overlay=True)
        app.overlay = Mock()
        app.hold_shortcut = Mock()
        app.hold_latched = True
        app._pipeline = Mock()
        app._set_status = Mock()
        app.pending_hotkey_start_job = Mock()

        with patch("local_voice.append_runtime_event") as runtime_event:
            app.start_recording(True)

        app._pipeline.assert_not_called()
        app.hold_shortcut.reset_session.assert_not_called()
        self.assertTrue(app.hold_latched)
        self.assertTrue(app.pending_hotkey_start[0])
        app._set_status.assert_called_once_with(
            "Speech engine is reloading — recording will start when READY",
            "working",
        )
        event_names = [call.args[0] for call in runtime_event.call_args_list]
        self.assertIn("recording_start_entered", event_names)
        self.assertIn("recording_queued_until_engine_ready", event_names)
        queued_event = next(
            call
            for call in runtime_event.call_args_list
            if call.args[0] == "recording_queued_until_engine_ready"
        )
        self.assertEqual(queued_event.kwargs["reason"], "engine_warming")
        self.assertEqual(queued_event.kwargs["deadline_seconds"], 90.0)

    def test_worker_ui_callbacks_are_drained_only_by_the_tk_poller(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.closing = threading.Event()
        app.ui_callbacks = local_voice_module.queue.SimpleQueue()
        app.ui_callback_poll_job = None
        app.root = Mock()
        app.root.after.return_value = "poll-2"
        callback = Mock()

        self.assertTrue(app._post_to_ui(callback))
        app.root.after.assert_not_called()
        callback.assert_not_called()

        app._drain_ui_callbacks()

        callback.assert_called_once_with()
        app.root.after.assert_called_once_with(15, app._drain_ui_callbacks)
        self.assertEqual(app.ui_callback_poll_job, "poll-2")

    def test_runtime_profile_accepts_a_valid_custom_model_identifier(self):
        profile = choose_runtime_profile(
            free_vram_mb=6000,
            cuda_available=True,
            model_name="Systran/faster-whisper-large-v3",
        )

        self.assertEqual(profile.device, "cuda")
        self.assertEqual(profile.model_name, "Systran/faster-whisper-large-v3")

    def test_backend_passes_personal_vocabulary_and_english_to_distil(self):
        calls = []

        class Segment:
            text = " Drew uses Hermes"
            words = ()

        class Model:
            def transcribe(self, audio, **kwargs):
                calls.append(kwargs)
                return iter([Segment()]), SimpleNamespace(
                    language="en", language_probability=1.0
                )

        backend = WhisperBackend(hotwords="Drew, Hermes, Glitch Factor")
        backend._model = Model()
        backend._signature = (
            "distil-large-v3.5",
            "cuda",
            "int8_float16",
            0,
        )
        profile = RuntimeProfile(
            "distil-large-v3.5", "cuda", "int8_float16", 0, "test"
        )

        backend.transcribe_samples(np.zeros(16_000, dtype=np.float32), profile)

        self.assertEqual(calls[0]["language"], "en")
        self.assertEqual(calls[0]["hotwords"], "Drew, Hermes, Glitch Factor")
        self.assertIn("Drew", calls[0]["initial_prompt"])

    def test_runtime_profile_falls_back_when_gpu_guard_fails(self):
        for free_vram_mb, cuda_available in ((511, True), (8192, False)):
            with self.subTest(free_vram_mb=free_vram_mb, cuda_available=cuda_available):
                profile = choose_runtime_profile(free_vram_mb, cuda_available)
                self.assertEqual(profile.device, "blocked")
                self.assertIsNone(profile.device_index)
                self.assertFalse(profile.runnable)
                self.assertTrue(profile.fallback_reason)

    def test_quiet_short_tail_can_reuse_latest_partial(self):
        samples = np.zeros(16_000, dtype=np.float32)
        self.assertTrue(
            tail_is_quiet(samples, processed_samples=12_000, max_tail_seconds=0.3)
        )

    def test_spoken_or_stale_tail_requires_final_inference(self):
        speech_tail = np.zeros(16_000, dtype=np.float32)
        speech_tail[12_000:] = 0.02
        self.assertFalse(
            tail_is_quiet(speech_tail, processed_samples=12_000, max_tail_seconds=0.3)
        )
        self.assertFalse(
            tail_is_quiet(np.zeros(16_000, dtype=np.float32), processed_samples=0, max_tail_seconds=0.3)
        )

    def test_normalize_transcript(self):
        self.assertEqual(normalize_transcript("  hello\n  local   voice  "), "hello local voice")

    def test_write_wav_is_16k_mono_pcm(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.wav"
            samples = np.linspace(-0.25, 0.25, 1600, dtype=np.float32)
            write_wav(path, samples)
            with wave.open(str(path), "rb") as wav:
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(wav.getnframes(), 1600)


if __name__ == "__main__":
    unittest.main()
