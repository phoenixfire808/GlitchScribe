from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

_PRODUCTION_LOCALAPPDATA = os.environ.get("LOCALAPPDATA")
_ISOLATED_LOCALAPPDATA = tempfile.TemporaryDirectory(
    prefix="glitchscribe-engine-warmup-tests-"
)
os.environ["LOCALAPPDATA"] = _ISOLATED_LOCALAPPDATA.name
try:
    import local_voice
finally:
    if _PRODUCTION_LOCALAPPDATA is None:
        os.environ.pop("LOCALAPPDATA", None)
    else:
        os.environ["LOCALAPPDATA"] = _PRODUCTION_LOCALAPPDATA


class FakeRoot:
    def __init__(self) -> None:
        self.jobs: dict[str, tuple[int, object]] = {}
        self.cancelled: list[str] = []
        self._next_id = 0

    def after(self, delay_ms: int, callback):
        self._next_id += 1
        job_id = f"job-{self._next_id}"
        self.jobs[job_id] = (int(delay_ms), callback)
        return job_id

    def after_cancel(self, job_id: str) -> None:
        self.cancelled.append(str(job_id))
        self.jobs.pop(str(job_id), None)

    def run(self, job_id: str) -> None:
        _delay_ms, callback = self.jobs.pop(str(job_id))
        callback()


class ControlledBackend:
    def is_loaded(self):
        return True

    def __init__(self, *, gate: threading.Event | None = None, error=None) -> None:
        self.gate = gate
        self.error = error
        self.entered = threading.Event()

    def preload_and_warm(self, _profile):
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(timeout=30.0)
        if self.error is not None:
            raise self.error
        return {"load_seconds": 0.1, "warm_seconds": 0.2}


class BlockingBackend(ControlledBackend):
    def preload_and_warm(self, _profile):
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(timeout=30.0)
        return {"load_seconds": 0.1, "warm_seconds": 0.2}


class FailingRoot(FakeRoot):
    def after(self, _delay_ms: int, _callback):
        raise RuntimeError("fixture scheduler unavailable")


class EngineWarmupLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_events = patch("local_voice.append_runtime_event")
        self.append_runtime_event = self.runtime_events.start()
        self.addCleanup(self.runtime_events.stop)
        self.apps: list[LocalAppFixture] = []

    def tearDown(self) -> None:
        for fixture in self.apps:
            fixture.release()

    def make_app(
        self,
        *,
        backend: ControlledBackend | None = None,
        gate: threading.Event | None = None,
    ):
        root = FakeRoot()
        backend = backend or ControlledBackend(gate=gate)
        app = local_voice.LocalVoiceApp.__new__(local_voice.LocalVoiceApp)
        app.root = root
        app.backend = backend
        app.runtime_profile = SimpleNamespace(
            runnable=True,
            fallback_reason="",
            label="Fixture engine",
            model_name="fixture-model",
            backend="fixture-backend",
            device="cpu",
            device_index=0,
            compute_type="int8",
            free_vram_mb=0,
            required_vram_mb=0,
            vram_reserve_mb=0,
        )
        app.config = SimpleNamespace(
            show_overlay=False,
            hotkey="Ctrl + Alt + Space",
        )
        app.engine_ready = threading.Event()
        app.closing = threading.Event()
        app.ui_callbacks = queue.SimpleQueue()
        app.engine_warmup_thread = None
        app._engine_warmup_lock = threading.RLock()
        app.pending_hotkey_start = None
        app.pending_hotkey_start_job = None
        app.pending_hotkey_start_deadline = 0.0
        app.engine_warmup_deadline_seconds = 45.0
        app.recording = False
        app.busy = False
        app._model_idle_unloaded = False
        app._last_dictation_activity = 0.0
        app._set_status = Mock()
        app._refresh_resource_panel = Mock()
        app._ensure_ambient_preroll = Mock()
        app._touch_dictation_activity = Mock()
        app.metrics_var = Mock()
        app.model_var = Mock()
        app.gpu_runtime_var = Mock()
        app.record_button = Mock()
        app.overlay = Mock()
        app.troubleshooting_log = None
        self.apps.append(LocalAppFixture(app, backend, root))
        return app, backend, root

    @staticmethod
    def drain_ui(app) -> None:
        while True:
            try:
                callback = app.ui_callbacks.get_nowait()
            except queue.Empty:
                return
            callback()

    def test_warmup_start_is_single_flight_while_worker_is_live(self):
        gate = threading.Event()
        app, backend, _root = self.make_app(gate=gate)

        self.assertTrue(app._start_engine_warmup())
        self.assertTrue(backend.entered.wait(timeout=1.0))
        first_thread = app.engine_warmup_thread

        self.assertFalse(app._start_engine_warmup())
        self.assertIs(first_thread, app.engine_warmup_thread)
        self.assertEqual(first_thread.name, "instant-engine-warmup")

        gate.set()
        first_thread.join(timeout=2.0)
        self.assertFalse(first_thread.is_alive())

    def test_timeout_terminalizes_once_clears_pending_start_and_blocks_retry(self):
        gate = threading.Event()
        app, backend, root = self.make_app(gate=gate)
        app.pending_hotkey_start = (True, 0)
        app.pending_hotkey_start_job = root.after(150, lambda: None)

        self.assertTrue(app._start_engine_warmup())
        attempt_id = app._engine_warmup_attempt["attempt_id"]
        app._engine_warmup_watchdog(attempt_id)
        app._engine_warmup_watchdog(attempt_id)

        self.assertEqual(app._engine_warmup_attempt["outcome"], "timed_out")
        self.assertIsNone(app.pending_hotkey_start)
        self.assertIsNone(app.pending_hotkey_start_job)
        self.assertFalse(app.engine_ready.is_set())
        self.assertTrue(app._engine_warmup_retry_blocked)
        self.assertFalse(app._start_engine_warmup())
        terminal_calls = [
            call
            for call in self.append_runtime_event.call_args_list
            if call.args and call.args[0] == "engine_warmup_terminal"
        ]
        self.assertEqual(len(terminal_calls), 1)
        self.assertTrue(
            any("restart" in str(call).casefold() for call in app._set_status.call_args_list)
        )

        gate.set()
        app.engine_warmup_thread.join(timeout=2.0)
        app.backend = ControlledBackend()
        self.assertFalse(app._start_engine_warmup(reset=True))

    def test_late_success_after_timeout_cannot_set_readiness_or_dispatch(self):
        app, backend, _root = self.make_app()
        app.pending_hotkey_start = (True, 0)
        app.pending_hotkey_start_deadline = local_voice.time.monotonic() + 90.0
        app.start_recording = Mock()

        self.assertTrue(app._start_engine_warmup())
        attempt_id = app._engine_warmup_attempt["attempt_id"]
        app._engine_warmup_watchdog(attempt_id)
        app.engine_warmup_thread.join(timeout=2.0)
        self.drain_ui(app)

        self.assertFalse(app.engine_ready.is_set())
        self.assertEqual(app._engine_warmup_attempt["outcome"], "timed_out")
        app.start_recording.assert_not_called()
        self.assertTrue(
            any(
                call.args
                and call.args[0] == "engine_warmup_late_result_ignored"
                for call in self.append_runtime_event.call_args_list
            )
        )

    def test_current_success_sets_readiness_and_dispatches_pending_start_once(self):
        app, backend, root = self.make_app()
        app.pending_hotkey_start = (True, 0)
        app.pending_hotkey_start_deadline = local_voice.time.monotonic() + 90.0
        app.start_recording = Mock()

        self.assertTrue(app._start_engine_warmup())
        app.engine_warmup_thread.join(timeout=2.0)
        self.drain_ui(app)

        self.assertTrue(app.engine_ready.is_set())
        self.assertEqual(app._engine_warmup_attempt["outcome"], "ready")
        pending_job = app.pending_hotkey_start_job
        self.assertIsNotNone(pending_job)
        root.run(pending_job)
        app.start_recording.assert_called_once_with(True, target_window_override=0)
        self.assertIsNone(app.pending_hotkey_start)

    def test_system_exit_is_terminalized_without_exception_text(self):
        app, backend, _root = self.make_app(
            backend=ControlledBackend(error=SystemExit("secret native detail"))
        )

        self.assertTrue(app._start_engine_warmup())
        app.engine_warmup_thread.join(timeout=2.0)
        self.drain_ui(app)

        self.assertEqual(app._engine_warmup_attempt["outcome"], "failed")
        terminal_calls = [
            call
            for call in self.append_runtime_event.call_args_list
            if call.args and call.args[0] == "engine_warmup_terminal"
        ]
        self.assertEqual(len(terminal_calls), 1)
        terminal_fields = terminal_calls[0].kwargs
        self.assertEqual(terminal_fields["error_type"], "SystemExit")
        self.assertNotIn("error_message", terminal_fields)
        self.assertNotIn(
            "secret native detail",
            repr(self.append_runtime_event.call_args_list),
        )

    def test_failed_attempt_blocks_ordinary_retry_until_explicit_reset(self):
        backend = ControlledBackend(error=RuntimeError("fixture failure"))
        app, _backend, root = self.make_app(backend=backend)
        app.pending_hotkey_start = (True, 0)
        app.pending_hotkey_start_job = root.after(150, lambda: None)

        self.assertTrue(app._start_engine_warmup())
        app.engine_warmup_thread.join(timeout=2.0)
        self.drain_ui(app)

        self.assertEqual(app._engine_warmup_attempt["outcome"], "failed")
        self.assertIsNone(app.pending_hotkey_start)
        self.assertIsNone(app.pending_hotkey_start_job)
        first_thread = app.engine_warmup_thread
        self.assertFalse(app._start_engine_warmup())
        self.assertIs(first_thread, app.engine_warmup_thread)

        app.backend = ControlledBackend()
        self.assertTrue(app._start_engine_warmup(reset=True))
        app.engine_warmup_thread.join(timeout=2.0)
        self.drain_ui(app)
        self.assertEqual(app._engine_warmup_attempt["outcome"], "ready")

    def test_ui_handoff_rejection_terminalizes_attempt(self):
        app, backend, root = self.make_app()
        app.pending_hotkey_start = (True, 0)
        app.pending_hotkey_start_job = root.after(150, lambda: None)
        app._post_to_ui = Mock(return_value=False)

        self.assertTrue(app._start_engine_warmup())
        app.engine_warmup_thread.join(timeout=2.0)

        self.assertEqual(app._engine_warmup_attempt["outcome"], "failed")
        self.assertEqual(app._engine_warmup_attempt["terminal_reason"], "ui_handoff_failed")
        self.assertFalse(app.engine_ready.is_set())
        self.assertIsNone(app.pending_hotkey_start)
        self.assertIsNone(app.pending_hotkey_start_job)

    def test_timeout_preserves_backend_stage_in_terminal_receipt(self):
        gate = threading.Event()
        app, backend, _root = self.make_app(gate=gate)

        self.assertTrue(app._start_engine_warmup())
        self.assertTrue(backend.entered.wait(timeout=1.0))
        attempt_id = app._engine_warmup_attempt["attempt_id"]
        app._engine_warmup_watchdog(attempt_id)

        terminal = [
            call
            for call in self.append_runtime_event.call_args_list
            if call.args and call.args[0] == "engine_warmup_terminal"
        ]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0].kwargs["last_stage"], "backend_call")
        self.assertNotIn("error_message", terminal[0].kwargs)
        gate.set()
        app.engine_warmup_thread.join(timeout=2.0)

    def test_watchdog_schedule_failure_never_starts_backend_worker(self):
        app, backend, _root = self.make_app()
        app.root = FailingRoot()

        self.assertFalse(app._start_engine_warmup())
        self.assertFalse(backend.entered.is_set())
        self.assertFalse(app.engine_warmup_thread.is_alive())
        self.assertIsNone(app.engine_warmup_watchdog_job)
        self.assertEqual(app._engine_warmup_attempt["outcome"], "failed")
        terminal = [
            call
            for call in self.append_runtime_event.call_args_list
            if call.args and call.args[0] == "engine_warmup_terminal"
        ]
        self.assertEqual(len(terminal), 1)

    def test_shutdown_quarantines_live_worker_and_skips_backend_unload(self):
        gate = threading.Event()
        backend = BlockingBackend(gate=gate)
        app, _backend, root = self.make_app(backend=backend)
        app._stop_project_ear = Mock()
        app._stop_ambient_preroll = Mock()
        app._unload_backend = Mock()
        app.hotkey = Mock()
        app.hold_shortcut = None
        app.action_wheel_hotkey = None
        app.tray_icon = None
        app.overlay = None
        app.streaming_bubble = None
        app.streaming = None
        app.recording = False
        app.recorder = None
        root.destroy = Mock()

        self.assertTrue(app._start_engine_warmup())
        self.assertTrue(backend.entered.wait(timeout=1.0))
        worker = app.engine_warmup_thread
        started = time.monotonic()
        app.close()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 1.5)
        self.assertTrue(worker.daemon)
        self.assertTrue(worker.is_alive())
        app._unload_backend.assert_not_called()
        self.assertTrue(
            any(
                call.args
                and call.args[0] == "engine_warmup_backend_unload_skipped"
                for call in self.append_runtime_event.call_args_list
            )
        )
        gate.set()
        worker.join(timeout=2.0)

    def test_concurrent_terminalizers_claim_one_terminal_outcome(self):
        app, _backend, _root = self.make_app()
        app._engine_warmup_attempt = {
            "attempt_id": 1,
            "started_monotonic": local_voice.time.monotonic(),
            "stage": "backend_call",
            "outcome": None,
        }
        barrier = threading.Barrier(2)
        results = []

        def terminalize(outcome):
            barrier.wait(timeout=2.0)
            results.append(
                app._terminalize_engine_warmup(
                    1,
                    outcome,
                    reason="concurrent_fixture",
                )
            )

        first = threading.Thread(target=terminalize, args=("failed",))
        second = threading.Thread(target=terminalize, args=("timed_out",))
        first.start()
        second.start()
        first.join(timeout=2.0)
        second.join(timeout=2.0)

        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(sum(bool(result) for result in results), 1)
        terminal = [
            call
            for call in self.append_runtime_event.call_args_list
            if call.args and call.args[0] == "engine_warmup_terminal"
        ]
        self.assertEqual(len(terminal), 1)
        self.assertIn(app._engine_warmup_attempt["outcome"], {"failed", "timed_out"})

    def test_model_change_resets_only_after_backend_acceptance(self):
        def make_model_app(create_backend):
            app = local_voice.LocalVoiceApp.__new__(local_voice.LocalVoiceApp)
            app.model_var = Mock()
            app.model_var.get.return_value = "fixture-new"
            app.model_box = Mock()
            app.config = SimpleNamespace(
                model_name="fixture-old",
                custom_model_ids="",
                show_overlay=False,
                save=Mock(),
            )
            app.recording = False
            app.busy = False
            app.engine_warmup_thread = None
            app.backend = object()
            app.runtime_profile = SimpleNamespace(model_name="fixture-old")
            app.engine_ready = threading.Event()
            app._pending_model_rollback = None
            app._runtime_profile_for_model = Mock(
                return_value=SimpleNamespace(
                    model_name="fixture-new",
                    runnable=True,
                    label="Fixture new",
                )
            )
            app._create_backend_for_profile = Mock(side_effect=create_backend)
            app._unload_backend = Mock()
            app._set_status = Mock()
            app._start_engine_warmup = Mock()
            return app

        with patch("local_voice.validate_model_identifier", return_value="fixture-new"), patch(
            "local_voice.configured_model_choices", return_value=("fixture-new",)
        ):
            app = make_model_app(lambda _profile: object())
            app._change_model()
            app._start_engine_warmup.assert_called_once_with(reset=True)

            failed = make_model_app(
                lambda _profile: (_ for _ in ()).throw(RuntimeError("backend rejected"))
            )
            failed._change_model()
            failed._start_engine_warmup.assert_not_called()
            self.assertEqual(failed.config.model_name, "fixture-old")

    def test_backend_call_start_event_precedes_terminal_result(self):
        app, backend, _root = self.make_app()

        self.assertTrue(app._start_engine_warmup())
        app.engine_warmup_thread.join(timeout=2.0)
        self.drain_ui(app)

        names = [call.args[0] for call in self.append_runtime_event.call_args_list]
        self.assertIn("engine_warmup_backend_call_started", names)
        self.assertIn("engine_warmup_terminal", names)
        self.assertLess(
            names.index("engine_warmup_backend_call_started"),
            names.index("engine_warmup_terminal"),
        )

    def test_runtime_event_sink_is_injected_for_the_complete_lifecycle(self):
        app, backend, _root = self.make_app()

        self.assertTrue(app._start_engine_warmup())
        app.engine_warmup_thread.join(timeout=2.0)
        self.drain_ui(app)

        self.assertTrue(self.append_runtime_event.called)
        self.assertTrue(
            all(call.args for call in self.append_runtime_event.call_args_list)
        )


class LocalAppFixture:
    def __init__(self, app, backend, root) -> None:
        self.app = app
        self.backend = backend
        self.root = root

    def release(self) -> None:
        gate = getattr(self.backend, "gate", None)
        if gate is not None:
            gate.set()
        thread = getattr(self.app, "engine_warmup_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
