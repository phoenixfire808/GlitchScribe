from __future__ import annotations

import importlib.util
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch


class PhysicalShortcutHookTests(unittest.TestCase):

    def test_action_wheel_open_path_completes_timing_receipt(self):
        from local_voice import LocalVoiceApp
        from drew_local_voice.theme import build_palette
        from types import SimpleNamespace

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.root = Mock()
        app.palette = build_palette("dark", "violet")
        app.config = SimpleNamespace(advanced_settings={})
        app._action_wheel = None
        app._action_wheel_cache = None
        app._action_wheel_session = None
        app._handle_action_wheel_choice = Mock()
        app._handle_action_wheel_error = Mock()
        app._clear_action_wheel_session = Mock()
        wheel = Mock()
        wheel_mod = SimpleNamespace(
            load_wheel_actions_from_config=Mock(return_value=()),
            actions_for_wheel_mode=Mock(return_value=()),
            is_action_wheel_enabled=Mock(return_value=True),
            ActionWheel=Mock(return_value=wheel),
        )
        session = SimpleNamespace(
            target_window=0,
            source_kind="review_buffer",
            mode="review",
            has_selection=False,
            anchor=(500, 500),
            request_id="wheel-test-open",
        )
        with patch("local_voice._lazy_import", return_value=wheel_mod), patch(
            "local_voice.append_runtime_event"
        ):
            app._open_action_wheel_at(
                500,
                500,
                session=session,
                requested_at=0.0,
            )
        wheel.show_at.assert_called_once_with(500, 500)
        self.assertIs(app._action_wheel, wheel)

    def test_global_action_wheel_hotkey_preserves_selection_mode(self):
        import inspect
        from local_voice import LocalVoiceApp

        source = inspect.getsource(LocalVoiceApp.__init__)
        start = source.index("def _dispatch_action_wheel_hotkey")
        end = source.index("self._action_wheel_hotkey_callback", start)
        hotkey_dispatch = source[start:end]
        self.assertIn("requested_at=started", hotkey_dispatch)
        self.assertIn("utility_fast=False", hotkey_dispatch)
        self.assertNotIn("utility_fast=True", hotkey_dispatch)

    def test_runtime_profile_never_falls_back_to_cpu_when_gpu_is_blocked(self):
        from local_voice import GPUDeviceInfo, LocalVoiceApp

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = SimpleNamespace(
            gpu_device_choice="cuda:0",
            gpu_vram_reserve_mb=512,
            compute_type="int8_float16",
        )
        devices = (
            GPUDeviceInfo(
                0,
                "NVIDIA GeForce RTX 5060 Ti",
                "GPU-A",
                16311,
                12901,
                3410,
                55,
            ),
        )
        with patch("local_voice.probe_gpu_inventory", return_value=devices), patch(
            "local_voice.append_runtime_event"
        ) as append_event:
            profile = app._runtime_profile_for_model("distil-large-v3.5")
        self.assertEqual(profile.device, "blocked")
        self.assertFalse(profile.runnable)
        self.assertTrue(
            any(call.args and call.args[0] == "runtime_profile_gpu_blocked" for call in append_event.call_args_list)
        )
        self.assertFalse(
            any(call.args and call.args[0] == "runtime_profile_cpu_fallback" for call in append_event.call_args_list)
        )

    def test_runtime_profile_reselects_another_eligible_gpu_instead_of_cpu(self):
        from local_voice import GPUDeviceInfo, LocalVoiceApp

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = SimpleNamespace(
            gpu_device_choice="cuda:1",
            gpu_vram_reserve_mb=512,
            compute_type="int8_float16",
        )
        devices = (
            GPUDeviceInfo(0, "NVIDIA GeForce RTX 5060 Ti", "GPU-A", 16311, 7000, 9311, 20),
            GPUDeviceInfo(1, "NVIDIA GeForce RTX 2070 SUPER", "GPU-B", 8192, 7000, 1192, 0),
        )
        with patch("local_voice.probe_gpu_inventory", return_value=devices), patch(
            "local_voice.append_runtime_event"
        ) as append_event:
            profile = app._runtime_profile_for_model("distil-large-v3.5")
        self.assertEqual((profile.device, profile.device_index), ("cuda", 0))
        self.assertTrue(profile.runnable)
        self.assertTrue(
            any(call.args and call.args[0] == "runtime_profile_gpu_reselected" for call in append_event.call_args_list)
        )

        from local_voice import GlobalHotkey, MOD_NOREPEAT

        hotkey = GlobalHotkey(lambda: None, "F8")

        self.assertNotEqual(hotkey.modifiers & MOD_NOREPEAT, 0)

    def test_app_starts_escape_monitor_when_hold_to_talk_is_disabled(self):
        from local_voice import AppConfig, LocalVoiceApp

        listener = Mock(error=None)
        factory = Mock(return_value=listener)
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = AppConfig(hold_to_talk_enabled=False)
        app.recording = True
        app.busy = False
        app.ai_cancellation = None
        app.pending_final_insertion = None
        app._post_to_ui = Mock(return_value=True)
        app._handle_hold_shortcut_action = Mock()

        app._start_hold_shortcut(factory=factory)

        listener.start.assert_called_once_with()
        self.assertIs(app.hold_shortcut, listener)
        kwargs = factory.call_args.kwargs
        self.assertFalse(kwargs["hold_enabled"])
        self.assertTrue(kwargs["escape_callback"]())
        app._post_to_ui.assert_called_once_with(app._cancel_active_operation)

    def test_global_escape_cancels_local_ai_without_waiting_for_tk(self):
        from local_voice import AppConfig, LocalVoiceApp

        listener = Mock(error=None)
        factory = Mock(return_value=listener)
        signal = Mock()
        signal.cancel.return_value = True
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = AppConfig(hold_to_talk_enabled=False)
        app.recording = False
        app.busy = False
        app.ai_cancellation = signal
        app.pending_final_insertion = None
        app._post_to_ui = Mock(return_value=True)
        app._set_status = Mock()
        app._handle_hold_shortcut_action = Mock()

        app._start_hold_shortcut(factory=factory)
        escape_callback = factory.call_args.kwargs["escape_callback"]

        self.assertTrue(escape_callback())
        signal.cancel.assert_called_once_with()
        posted = app._post_to_ui.call_args.args[0]
        posted()
        app._set_status.assert_called_once_with(
            "Cancelling cloud AI request…",
            "working",
        )

    def test_escape_is_consumed_only_when_active_operation_cancels(self):
        from drew_local_voice.shortcut_hook import PhysicalShortcutHook
        from drew_local_voice.shortcut_sessions import ShortcutAction

        cancelled = []
        hook = PhysicalShortcutHook(
            callback=lambda _action, _target: None,
            foreground_reader=lambda: 777,
            escape_callback=lambda: cancelled.append(True) or True,
            hold_enabled=False,
        )

        self.assertEqual(
            hook.handle_key_event(0x1B, is_down=True),
            ShortcutAction.CANCEL,
        )
        self.assertEqual(
            hook.handle_key_event(0x1B, is_down=True),
            ShortcutAction.CANCEL,
        )
        self.assertEqual(
            hook.handle_key_event(0x1B, is_down=False),
            ShortcutAction.CANCEL,
        )
        self.assertEqual(cancelled, [True])

        hook.escape_callback = lambda: False
        self.assertEqual(
            hook.handle_key_event(0x1B, is_down=True),
            ShortcutAction.NONE,
        )
        self.assertEqual(
            hook.handle_key_event(0x1B, is_down=False),
            ShortcutAction.NONE,
        )

    def test_app_escape_discards_active_recording_without_finalizing(self):
        from local_voice import LocalVoiceApp

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.recording = True
        app.busy = False
        app.stop_pending = False
        app.hold_latched = True
        recorder = Mock()
        recorder.stop.return_value = ([], 1.0)
        app.recorder = recorder
        stream = Mock()
        app.streaming = stream
        app.hold_shortcut = Mock()
        app.finalization_cancelled = threading.Event()
        app.ai_cancellation = None
        app.meter_var = Mock()
        app.record_button = Mock()
        app._hide_streaming_bubble = Mock()
        app._set_status = Mock()
        app.config = SimpleNamespace(show_overlay=True, hotkey="Ctrl + Alt + Space")
        app.overlay = Mock()
        app.pipeline_observer = Mock()
        app.stop_recording = Mock()

        with patch("local_voice.append_runtime_event") as runtime_event:
            self.assertTrue(app._cancel_active_operation())

        stream.cancel.assert_called_once_with()
        recorder.stop.assert_called_once_with()
        app.stop_recording.assert_not_called()
        self.assertFalse(app.recording)
        self.assertFalse(app.busy)
        self.assertFalse(app.hold_latched)
        app.record_button.configure.assert_called_once_with(
            text="START DICTATION",
            state="normal",
        )
        app.overlay.show_idle.assert_called_once_with(app.config.hotkey)
        runtime_event.assert_called_once_with("recording_cancelled", pid=ANY)

    def test_app_escape_marks_inflight_finalization_for_safe_discard(self):
        from local_voice import LocalVoiceApp

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.recording = False
        app.busy = True
        app.finalizer_thread = Mock()
        app.finalization_cancelled = threading.Event()
        app.ai_cancellation = None
        app._set_status = Mock()

        self.assertTrue(app._cancel_active_operation())
        self.assertTrue(app.finalization_cancelled.is_set())
        app._set_status.assert_called_once_with(
            "Cancelling transcription…",
            "working",
        )

    def test_cancelled_finalization_never_reaches_text_insertion(self):
        from local_voice import LocalVoiceApp

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.busy = True
        app.finalizer_thread = Mock()
        app.streaming = Mock()
        app.finalization_cancelled = threading.Event()
        app.finalization_cancelled.set()
        app.pipeline_observer = Mock()
        app.record_button = Mock()
        app._hide_streaming_bubble = Mock()
        app._set_status = Mock()
        app.config = SimpleNamespace(show_overlay=True, hotkey="Ctrl + Alt + Space")
        app.overlay = Mock()

        with (
            patch("local_voice.type_unicode") as type_unicode,
            patch("local_voice.append_runtime_event") as runtime_event,
        ):
            app._finish_transcription(
                "must not be inserted",
                {"stop_to_text_seconds": 0.1},
                1.0,
                True,
                777,
            )

        type_unicode.assert_not_called()
        event_names = [call.args[0] for call in runtime_event.call_args_list]
        self.assertEqual(
            event_names,
            ["transcription_ui_finish_entered", "transcription_cancelled"],
        )
        app.pipeline_observer.record.assert_called_once_with(
            "complete",
            "skipped",
            reason="user_cancelled_transcription",
        )
        app.overlay.show_idle.assert_called_once_with(app.config.hotkey)
        app._set_status.assert_called_once_with(
            "Transcription cancelled — no text inserted",
            "ready",
        )

    def test_app_starts_configured_hold_listener_and_marshals_to_tk(self):
        from drew_local_voice.shortcut_sessions import ShortcutAction
        from local_voice import AppConfig, LocalVoiceApp

        listener = Mock(error=None)
        factory = Mock(return_value=listener)
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = AppConfig(
            hotkey="Ctrl + Alt + Space",
            hold_to_talk_enabled=True,
            hold_shortcut="F10",
            hold_latch_modifier="Alt",
        )
        app._post_to_ui = Mock()
        app._handle_hold_shortcut_action = Mock()
        app._set_status = Mock()

        app._start_hold_shortcut(factory=factory)

        listener.start.assert_called_once_with()
        self.assertIs(app.hold_shortcut, listener)
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs["hold_key"], "F10")
        self.assertEqual(kwargs["latch_modifier"], "ALT")

        kwargs["callback"](ShortcutAction.START_HOLD, 777)
        posted = app._post_to_ui.call_args.args[0]
        posted()
        app._handle_hold_shortcut_action.assert_called_once_with(
            ShortcutAction.START_HOLD,
            777,
        )

    def test_hold_latch_save_validator_updates_persisted_config(self):
        from local_voice import AppConfig, apply_hold_shortcut_settings

        config = AppConfig()
        apply_hold_shortcut_settings(
            config,
            {
                "general.hold_to_talk_enabled": True,
                "general.hold_shortcut": "F10",
                "general.hold_latch_modifier": "Alt",
            },
        )

        self.assertTrue(config.hold_to_talk_enabled)
        self.assertEqual(config.hold_shortcut, "F10")
        self.assertEqual(config.hold_latch_modifier, "Alt")

    def test_app_projects_persisted_hold_latch_values_into_settings_ui(self):
        from drew_local_voice.performance import (
            PerformancePresetRegistry,
            built_in_performance_presets,
        )
        from drew_local_voice.settings import SettingsRegistry, built_in_settings
        from local_voice import AppConfig, LocalVoiceApp

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.config = AppConfig(
            hold_to_talk_enabled=True,
            hold_shortcut="F10",
            hold_latch_modifier="Alt",
        )
        app.settings_registry = SettingsRegistry(built_in_settings())
        app.performance_presets = PerformancePresetRegistry(
            built_in_performance_presets()
        )

        values = app._advanced_setting_values()

        self.assertTrue(values["general.hold_to_talk_enabled"])
        self.assertEqual(values["general.hold_shortcut"], "F10")
        self.assertEqual(values["general.hold_latch_modifier"], "Alt")

    def test_hold_latch_settings_are_canonical_and_opt_in(self):
        from drew_local_voice.settings import built_in_settings
        from local_voice import AppConfig

        config = AppConfig()
        self.assertFalse(config.hold_to_talk_enabled)
        self.assertEqual(config.hold_shortcut, "F8")
        self.assertEqual(config.hold_latch_modifier, "Shift")

        specs = {spec.key: spec for spec in built_in_settings()}
        self.assertFalse(specs["general.hold_to_talk_enabled"].default)
        self.assertEqual(specs["general.hold_shortcut"].choices, ("F8", "F9", "F10", "F11", "F12"))
        self.assertEqual(specs["general.hold_latch_modifier"].choices, ("Shift", "Ctrl", "Alt"))

    def test_app_hold_release_uses_existing_stop_and_finalize_pipeline(self):
        from drew_local_voice.shortcut_sessions import ShortcutAction
        from local_voice import LocalVoiceApp

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.busy = False
        app.recording = True
        app.hold_latched = False
        app.stop_recording = Mock()

        app._handle_hold_shortcut_action(ShortcutAction.STOP_HOLD, 0)

        app.stop_recording.assert_called_once_with()

    def test_app_fresh_toggle_chord_stops_latched_recording_once(self):
        from drew_local_voice.shortcut_sessions import ShortcutAction
        from local_voice import LocalVoiceApp

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.busy = False
        app.recording = True
        app.hold_latched = True
        app.stop_recording = Mock()

        app._handle_hold_shortcut_action(ShortcutAction.STOP_LATCHED, 0)

        self.assertFalse(app.hold_latched)
        app.stop_recording.assert_called_once_with()

    def test_app_latch_marks_visible_state_without_restarting_recorder(self):
        from drew_local_voice.shortcut_sessions import ShortcutAction
        from local_voice import LocalVoiceApp

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.busy = False
        app.recording = True
        app.hold_latched = False
        app.hold_shortcut = Mock()
        app.start_recording = Mock()
        app.stop_recording = Mock()
        app._set_status = Mock()
        app.config = SimpleNamespace(show_overlay=True)
        app.overlay = Mock()

        app._handle_hold_shortcut_action(ShortcutAction.LATCH, 0)

        self.assertTrue(app.hold_latched)
        app.start_recording.assert_not_called()
        app.stop_recording.assert_not_called()
        app._set_status.assert_called_once_with(
            "Recording locally — latched; use the toggle shortcut to stop",
            "recording",
        )
        app.overlay.update_partial.assert_called_once_with(
            "",
            state="LISTENING • LATCHED",
        )

    def test_app_hold_start_uses_existing_recording_pipeline_and_target(self):
        from drew_local_voice.shortcut_sessions import ShortcutAction
        from local_voice import LocalVoiceApp

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.busy = False
        app.recording = False
        app.hold_latched = False
        app.hold_shortcut = Mock()
        app.start_recording = Mock()
        app._set_status = Mock()
        app.config = SimpleNamespace(show_overlay=True)
        app.overlay = Mock()

        app._handle_hold_shortcut_action(ShortcutAction.START_HOLD, 777)

        app.start_recording.assert_called_once_with(
            True,
            target_window_override=777,
        )
        self.assertFalse(app.hold_latched)

    def test_start_and_stop_join_hook_thread_and_post_quit(self):
        from drew_local_voice.shortcut_hook import PhysicalShortcutHook

        release = threading.Event()

        class FakeApi:
            def __init__(self):
                self.posted = []
                self.uninstalled = []

            def current_thread_id(self):
                return 42

            def install(self, _callback):
                return 9001

            def message_loop(self):
                release.wait(timeout=2.0)

            def post_quit(self, thread_id):
                self.posted.append(thread_id)
                release.set()

            def uninstall(self, handle):
                self.uninstalled.append(handle)

        api = FakeApi()
        hook = PhysicalShortcutHook(
            callback=lambda _action, _target: None,
            foreground_reader=lambda: 777,
            api_factory=lambda: api,
        )

        hook.start()
        self.assertTrue(hook.registered.is_set())
        self.assertTrue(hook.thread.is_alive())

        hook.stop()

        self.assertEqual(api.posted, [42])
        self.assertEqual(api.uninstalled, [9001])
        self.assertFalse(hook.thread.is_alive())

    def test_hook_loop_installs_dispatches_and_uninstalls_through_adapter(self):
        from drew_local_voice.shortcut_hook import PhysicalShortcutHook
        from drew_local_voice.shortcut_sessions import ShortcutAction

        events = []

        class FakeApi:
            def __init__(self):
                self.callback = None
                self.uninstalled = []

            def current_thread_id(self):
                return 42

            def install(self, callback):
                self.callback = callback
                return 9001

            def message_loop(self):
                self.callback(0x77, True, 0)

            def uninstall(self, handle):
                self.uninstalled.append(handle)

        api = FakeApi()
        hook = PhysicalShortcutHook(
            callback=lambda action, target: events.append((action, target)),
            hold_key="F8",
            latch_modifier="SHIFT",
            foreground_reader=lambda: 777,
            api_factory=lambda: api,
        )

        hook._run()

        self.assertTrue(hook.registered.is_set())
        self.assertEqual(events, [(ShortcutAction.START_HOLD, 777)])
        self.assertEqual(api.uninstalled, [9001])
        self.assertIsNone(hook.hook_handle)

    def test_physical_hold_dispatches_start_with_frozen_foreground_target(self):
        self.assertIsNotNone(
            importlib.util.find_spec("drew_local_voice.shortcut_hook"),
            "runtime hold/latch integration needs a physical-event adapter",
        )
        from drew_local_voice.shortcut_hook import PhysicalShortcutHook
        from drew_local_voice.shortcut_sessions import ShortcutAction

        dispatched = []
        hook = PhysicalShortcutHook(
            callback=lambda action, target: dispatched.append((action, target)),
            hold_key="F8",
            latch_modifier="SHIFT",
            foreground_reader=lambda: 777,
        )

        self.assertEqual(
            hook.handle_key_event(0x77, is_down=True, flags=0),
            ShortcutAction.START_HOLD,
        )
        self.assertEqual(dispatched, [(ShortcutAction.START_HOLD, 777)])

        hook.reset_session()
        self.assertEqual(
            hook.handle_key_event(0x77, is_down=True, flags=0x10),
            ShortcutAction.NONE,
        )
        self.assertEqual(dispatched, [(ShortcutAction.START_HOLD, 777)])


if __name__ == "__main__":
    unittest.main()
