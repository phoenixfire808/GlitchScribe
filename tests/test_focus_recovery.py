from __future__ import annotations

import importlib
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import local_voice
from drew_local_voice import context as context_module
from drew_local_voice.settings import built_in_settings
from local_voice import LocalVoiceApp


class FocusRecoveryTests(unittest.TestCase):
    def test_close_cancels_pending_retry_before_destroying_tk(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.closing = threading.Event()
        app.pending_final_insertion = object()
        app.pending_final_retry_job = "job-1"
        app.root = SimpleNamespace(after_cancel=Mock(), destroy=Mock())
        app._stop_project_ear = Mock()
        app.ai_cancellation = None
        app.streaming = None
        app.recording = False
        app.hotkey = Mock()
        app.tray_icon = None
        app.overlay = None
        app.streaming_bubble = None

        app.close()

        app.root.after_cancel.assert_called_once_with("job-1")
        self.assertIsNone(app.pending_final_retry_job)
        self.assertIsNone(app.pending_final_insertion)
        app.root.destroy.assert_called_once_with()

    def test_focus_recovery_window_is_canonical_and_bounded(self):
        specs = {spec.key: spec for spec in built_in_settings()}
        self.assertIn("typing.focus_recovery_seconds", specs)
        spec = specs["typing.focus_recovery_seconds"]
        self.assertEqual(spec.default, 30.0)
        self.assertEqual((spec.minimum, spec.maximum), (1.0, 120.0))

    def test_control_identity_probe_is_content_free_and_platform_bounded(self):
        probe_reader = getattr(
            context_module,
            "read_uia_focused_control_identity",
            None,
        )
        self.assertIsNotNone(
            probe_reader,
            "focus recovery needs a content-free focused-control probe",
        )
        with patch("drew_local_voice.context.os.name", "posix"):
            probe = probe_reader()

        self.assertFalse(probe.accessible)
        self.assertFalse(probe.is_password)
        self.assertEqual(probe.text, "")
        self.assertEqual(probe.source, "platform_unsupported")

    def test_finalization_queues_complete_payload_when_popup_owns_focus(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.busy = True
        app.finalizer_thread = object()
        app.streaming = object()
        app.live_insertion_suspended = False
        app.session_live_payload = ""
        app.last_text = ""
        app.config = SimpleNamespace(
            show_overlay=False,
            hotkey="ctrl+alt+space",
            live_insert=False,
        )
        app._hide_streaming_bubble = Mock()
        app.record_button = Mock()
        app._effective_replacements = Mock(return_value={})
        app._prepare_dictated_text = Mock(side_effect=lambda text: text)
        app._contextualize_prepared_text = Mock(
            return_value=SimpleNamespace(text="Complete message.", prefix="")
        )
        app.metrics_var = Mock()
        app.transcript_label_var = Mock()
        app.text = Mock()
        app.live_committer = Mock()
        app.live_committer.finalize.return_value = SimpleNamespace(
            conflict=False,
            text_to_insert="Complete message.",
        )
        app._target_is_active = Mock(return_value=False)
        app._profile_setting = Mock(side_effect=lambda _key, default: default)
        app._queue_pending_final_insertion = Mock(return_value=True)
        app._set_status = Mock()
        app.overlay = Mock()

        fake_winsound = SimpleNamespace(MessageBeep=Mock())
        with (
            patch("local_voice.os.name", "nt"),
            patch("local_voice.type_unicode") as type_unicode_mock,
            patch.dict(sys.modules, {"winsound": fake_winsound}),
            patch("local_voice.append_runtime_event"),
        ):
            app._finish_transcription(
                "Complete message.",
                {"stop_to_text_seconds": 0.2, "reused_partial": False},
                1.0,
                True,
                321,
            )

        type_unicode_mock.assert_not_called()
        app._queue_pending_final_insertion.assert_called_once_with(
            "Complete message. ",
            321,
            "Complete message.",
            timeout_seconds=30.0,
        )
        app._set_status.assert_called_with(
            "Transcript ready; waiting for the original field to return",
            "working",
        )

    def test_hotkey_recording_freezes_target_control_before_device_setup(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = SimpleNamespace(
            device_index=None,
            live_insert=False,
            hotkey_auto_insert=True,
        )
        app._freeze_session_target_control = Mock(return_value="control-321")
        app._save_settings = Mock()
        app._resolve_profile_for_target = Mock()
        app._set_status = Mock()

        app.start_recording(True, target_window_override=321)

        app._freeze_session_target_control.assert_called_once_with(321)
        app._set_status.assert_called_once_with("No microphone is available", "error")

    def test_final_payload_is_queued_against_frozen_original_control(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.session_target_control_identity = "control-321"
        app.pending_final_insertion = None
        app.pending_final_retry_job = None
        app.root = SimpleNamespace(after=Mock(return_value="job-1"))

        queued = getattr(app, "_queue_pending_final_insertion", None)
        self.assertIsNotNone(queued, "finalization needs a bounded retry queue")
        accepted = queued(
            "Complete message. ",
            321,
            "Complete message.",
            timeout_seconds=30.0,
            clock=lambda: 100.0,
        )

        self.assertTrue(accepted)
        self.assertEqual(app.pending_final_insertion.payload, "Complete message. ")
        self.assertEqual(app.pending_final_insertion.target_window, 321)
        self.assertEqual(
            app.pending_final_insertion.control_identity,
            "control-321",
        )
        self.assertEqual(app.pending_final_insertion.expires_at, 130.0)
        self.assertEqual(app.pending_final_retry_job, "job-1")
        app.root.after.assert_called_once_with(
            250,
            app._retry_pending_final_insertion,
        )

    def test_matching_original_control_receives_retained_payload_once(self):
        pending_type = getattr(local_voice, "PendingFinalInsertion", None)
        self.assertIsNotNone(
            pending_type,
            "focus recovery requires an immutable pending insertion contract",
        )
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.pending_final_insertion = pending_type(
            payload="Complete message. ",
            target_window=321,
            control_identity="control-321",
            prepared_text="Complete message.",
            expires_at=130.0,
        )
        app.pending_final_retry_job = "job-1"
        app.closing = threading.Event()
        app._target_is_active = lambda target: target == 321
        app._current_control_identity = lambda: "control-321"
        app._set_status = Mock()
        app.session_live_payload = ""
        app.last_inserted_payload = ""
        app.last_inserted_target = 0
        app.last_inserted_at = 0.0
        app.root = SimpleNamespace(after=Mock(return_value="job-2"))

        from drew_local_voice.insertion import InsertionResult

        app._insert_text_with_policy = Mock(
            return_value=InsertionResult(
                len("Complete message. "),
                len("Complete message. "),
                reason="ok",
                method="unicode",
            )
        )
        with patch("local_voice.append_runtime_event"):
            app._retry_pending_final_insertion(clock=lambda: 101.0)
            app._retry_pending_final_insertion(clock=lambda: 102.0)

        app._insert_text_with_policy.assert_called_once_with(
            "Complete message. ",
            321,
            source="pending_final",
        )
        self.assertIsNone(app.pending_final_insertion)
        self.assertIsNone(app.pending_final_retry_job)
        self.assertEqual(app.last_inserted_payload, "Complete message. ")
        self.assertEqual(app.last_inserted_target, 321)
        app.root.after.assert_not_called()


if __name__ == "__main__":
    unittest.main()
