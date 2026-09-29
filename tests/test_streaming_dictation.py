import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from local_voice import LocalVoiceApp, RuntimeProfile
from streaming_dictation import (
    StreamingDictation,
    TimedWord,
    TimestampCommitter,
    reconcile_latest_tail,
)


PROFILE = RuntimeProfile("base.en", "cuda", "int8_float16", 0, "Instant")


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.concurrent = 0
        self.max_concurrent = 0
        self.gate = None
        self.error = None

    def transcribe_samples(self, samples, profile, fast=True):
        if self.error is not None:
            raise self.error
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        if self.gate is not None:
            self.gate.wait(timeout=1)
        audio = np.asarray(samples).copy()
        self.calls.append(audio)
        self.concurrent -= 1
        return f"partial {audio.size}", {"transcribe_seconds": 0.05}


class FakeRecorder:
    def __init__(self, samples):
        self.samples = np.asarray(samples, dtype=np.float32)

    def snapshot(self):
        return self.samples.copy()


class StreamingDictationTests(unittest.TestCase):
    def test_reconciles_suffix_visible_in_latest_partial_but_missing_from_final(self):
        merged, reconciled = reconcile_latest_tail(
            "This is a long dictated message and the final pass ends here.",
            "the final pass ends here with the words that were visible in the bubble.",
        )

        self.assertTrue(reconciled)
        self.assertEqual(
            merged,
            "This is a long dictated message and the final pass ends here with the words that were visible in the bubble.",
        )

    def test_tail_reconciliation_rejects_real_word_changes_and_terminal_operators(self):
        changed, changed_ok = reconcile_latest_tail(
            "alpha beta gamma delta",
            "beta gamma different ending",
        )
        terminal, terminal_ok = reconcile_latest_tail(
            "run git status in the terminal",
            "git status in the terminal && press enter",
        )

        self.assertFalse(changed_ok)
        self.assertEqual(changed, "alpha beta gamma delta")
        self.assertFalse(terminal_ok)
        self.assertEqual(terminal, "run git status in the terminal")

    def test_finalize_recovers_only_a_near_stop_partial_tail(self):
        class SequencedBackend:
            def __init__(self):
                self.outputs = [
                    "the final pass ends here with the missing final words",
                    "A long message whose final pass ends here.",
                ]

            def transcribe_samples(self, samples, profile, fast=True):
                return self.outputs.pop(0), {"transcribe_seconds": 0.01}

        stream = StreamingDictation(
            SequencedBackend(),
            PROFILE,
            min_audio_seconds=0.1,
            min_new_audio_seconds=0.01,
        )
        audio = np.ones(16_000, dtype=np.float32)
        stream.process_snapshot(audio)

        result = stream.finalize(np.concatenate([audio, np.zeros(800, dtype=np.float32)]))

        self.assertTrue(result.reconciled_tail)
        self.assertEqual(
            result.text,
            "A long message whose final pass ends here with the missing final words",
        )
        self.assertGreater(result.reconciled_tail_characters, 0)

    def test_finalize_does_not_merge_a_partial_far_before_stop(self):
        class SequencedBackend:
            def __init__(self):
                self.outputs = [
                    "the final pass ends here with an obsolete guess",
                    "A long message whose final pass ends here.",
                ]

            def transcribe_samples(self, samples, profile, fast=True):
                return self.outputs.pop(0), {"transcribe_seconds": 0.01}

        stream = StreamingDictation(
            SequencedBackend(),
            PROFILE,
            min_audio_seconds=0.1,
            min_new_audio_seconds=0.01,
        )
        stream.process_snapshot(np.ones(16_000, dtype=np.float32))

        result = stream.finalize(np.ones(64_001, dtype=np.float32))

        self.assertFalse(result.reconciled_tail)
        self.assertEqual(result.text, "A long message whose final pass ends here.")

    def test_stream_accepts_low_latency_timestamp_commit_profile(self):
        stream = StreamingDictation(
            backend=object(),
            profile=object(),
            interval_seconds=0.35,
            min_audio_seconds=0.75,
            min_new_audio_seconds=0.30,
            timestamp_stability_lag_seconds=0.45,
            timestamp_holdback_words=2,
        )

        self.assertEqual(stream.interval_seconds, 0.35)
        self.assertEqual(stream._timestamp_committer.stability_lag_seconds, 0.45)
        self.assertEqual(stream._timestamp_committer.holdback_words, 2)

    def test_timestamp_committer_requires_two_stable_passes_and_holds_tail(self):
        committer = TimestampCommitter(stability_lag_seconds=1.0, holdback_words=4)
        words = tuple(
            TimedWord(word=f" {word}", start=index * 0.4, end=(index + 1) * 0.4)
            for index, word in enumerate("one two three four five six seven eight".split())
        )

        self.assertEqual(committer.update(words, audio_seconds=5.0), "")
        self.assertEqual(
            committer.update(words, audio_seconds=5.5),
            "one two three four",
        )

    def test_timestamp_committer_never_commits_a_revised_early_phrase(self):
        committer = TimestampCommitter(stability_lag_seconds=0.5, holdback_words=1)
        first = (
            TimedWord(" I'm", 0.0, 0.4),
            TimedWord(" going", 0.4, 0.8),
            TimedWord(" home", 0.8, 1.2),
        )
        revised = (
            TimedWord(" I", 0.0, 0.2),
            TimedWord(" am", 0.2, 0.4),
            TimedWord(" going", 0.4, 0.8),
            TimedWord(" home", 0.8, 1.2),
        )

        self.assertEqual(committer.update(first, audio_seconds=3.0), "")
        self.assertEqual(committer.update(revised, audio_seconds=3.5), "")

    def test_timestamp_committer_does_not_commit_recent_words(self):
        committer = TimestampCommitter(stability_lag_seconds=1.0, holdback_words=0)
        words = (TimedWord(" hello", 0.6, 1.0),)

        self.assertEqual(committer.update(words, audio_seconds=1.5), "")
        self.assertEqual(committer.update(words, audio_seconds=1.9), "")
        self.assertEqual(committer.update(words, audio_seconds=2.1), "hello")

    def test_windowed_word_timestamps_are_offset_to_full_recording_time(self):
        class WordBackend:
            def transcribe_samples(self, audio, profile, fast=True):
                return "hello", {
                    "transcribe_seconds": 0.01,
                    "words": (
                        {
                            "word": " hello",
                            "start": 0.1,
                            "end": 0.2,
                            "probability": 0.9,
                        },
                    ),
                }

        stream = StreamingDictation(
            WordBackend(),
            PROFILE,
            min_audio_seconds=0.1,
            min_new_audio_seconds=0.0,
            max_partial_audio_seconds=1.0,
            sample_rate=10,
        )
        result = stream.process_snapshot(np.zeros(20, dtype=np.float32))

        self.assertTrue(result.windowed)
        self.assertAlmostEqual(result.words[0].start, 1.1)
        self.assertAlmostEqual(result.words[0].end, 1.2)

    def test_app_live_insertion_uses_only_timestamp_stable_text(self):
        seen = []

        class Value:
            def set(self, value):
                pass

        class Text:
            def delete(self, *args):
                pass

            def insert(self, *args):
                pass

        class Committer:
            def accept_stable(self, text):
                seen.append(text)
                return SimpleNamespace(
                    conflict=False,
                    text_to_insert=text,
                )

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.recording = True
        app.speech_end_detector = None
        app.last_recognized_partial = ""
        app._prepare_dictated_text = lambda text: text
        app.transcript_label_var = Value()
        app.text = Text()
        app.metrics_var = Value()
        app.config = SimpleNamespace(
            show_overlay=False,
            hotkey_auto_insert=True,
            live_insert=True,
        )
        app.command_debouncer = SimpleNamespace(update=lambda text: None)
        app.hotkey_started_recording = True
        app.live_insertion_suspended = False
        app.live_committer = Committer()
        app.target_window = 41
        app._target_is_active = lambda target: True
        app.session_live_payload = ""
        app._set_status = lambda *args: None
        result = SimpleNamespace(
            text="unstable draft words",
            stable_text="stable words",
            inference_seconds=0.01,
        )

        with patch("local_voice.type_unicode") as send:
            app._show_partial(result)

        self.assertEqual(seen, ["stable words"])
        send.assert_called_once_with("stable words", 41)

    def test_app_live_insertion_pauses_when_target_focus_changes(self):
        class Value:
            def set(self, value):
                pass

        class Text:
            def delete(self, *args):
                pass

            def insert(self, *args):
                pass

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.recording = True
        app.speech_end_detector = None
        app.last_recognized_partial = ""
        app._prepare_dictated_text = lambda text: text
        app.transcript_label_var = Value()
        app.text = Text()
        app.metrics_var = Value()
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
                text_to_insert=text,
            )
        )
        app.target_window = 41
        app._target_is_active = lambda target: False
        app.session_live_payload = ""
        app._set_status = lambda *args: None
        result = SimpleNamespace(
            text="stable words plus draft",
            stable_text="stable words",
            inference_seconds=0.01,
        )

        with patch("local_voice.type_unicode") as send:
            app._show_partial(result)

        self.assertTrue(app.live_insertion_suspended)
        self.assertEqual(app.session_live_payload, "")
        send.assert_not_called()

    def test_transient_status_does_not_rebuild_control_lab_state(self):
        updates = []

        class Value:
            def set(self, value):
                updates.append(value)

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.status_var = Value()
        app._status_kind = "recording"
        app._refresh_control_lab_state = unittest.mock.Mock()
        app.status_label = SimpleNamespace(configure=unittest.mock.Mock())
        app.tray_icon = None

        app._set_transient_status("Recording locally… 1.0s", "recording")

        self.assertEqual(updates, ["Recording locally… 1.0s"])
        app._refresh_control_lab_state.assert_not_called()
        app.status_label.configure.assert_not_called()

    def test_partial_runtime_events_are_throttled_without_transcript_content(self):
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        with patch("local_voice.time.monotonic", side_effect=(100.0, 100.2, 101.2)) as clock:
            with patch("local_voice.append_runtime_event") as record:
                app._record_partial_runtime_event("hello", 41, 3, 2)
                app._record_partial_runtime_event("hello again", 41, 8, 4)
                app._record_partial_runtime_event("hello again", 41, 8, 4)

        self.assertEqual(clock.call_count, 3)
        self.assertEqual(record.call_count, 2)
        self.assertNotIn("text", record.call_args.kwargs)
        self.assertNotIn("transcript", record.call_args.kwargs)
        self.assertEqual(record.call_args.kwargs["text_chars"], 8)

    def test_meter_tick_routes_detected_silence_through_normal_stop_path(self):
        events = []

        class Value:
            def set(self, value):
                events.append(("meter", value))

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.recording = True
        app.recorder = SimpleNamespace(
            peak=0.01,
            latest_rms=0.0001,
            started_at=time.monotonic() - 3.0,
        )
        app.config = SimpleNamespace(
            show_overlay=False,
            auto_stop_on_silence=True,
        )
        app.speech_end_detector = SimpleNamespace(
            speech_started=True,
            observe=lambda rms: True,
        )
        app.meter_var = Value()
        app.root = SimpleNamespace(after=lambda *args: events.append(("after", args)))
        app._set_status = lambda text, kind: events.append(("status", kind))
        app.stop_recording = lambda: events.append(("stop", None))

        app._meter_tick()

        self.assertIn(("stop", None), events)
        self.assertFalse(any(name == "after" for name, _ in events))

    def test_stop_recording_waits_for_capture_tail_grace_before_snapshot(self):
        events = []

        class Button:
            def configure(self, **kwargs):
                events.append(("button", kwargs))

        class Recorder:
            def stop(self):
                events.append(("recorder-stop", None))
                return np.ones(16_000, dtype=np.float32), 1.0

        class Root:
            def after(self, delay, callback):
                events.append(("after", delay, callback))

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.recorder = Recorder()
        app.recording = True
        app.busy = False
        app.stop_pending = False
        app.config = SimpleNamespace(capture_tail_grace_seconds=0.12)
        app.record_button = Button()
        app.root = Root()
        app._complete_stop_recording = lambda: events.append(("complete-stop", None))

        app.stop_recording()

        self.assertFalse(app.recording)
        self.assertTrue(app.busy)
        self.assertTrue(app.stop_pending)
        self.assertFalse(any(item[0] == "recorder-stop" for item in events))
        scheduled = next(item for item in events if item[0] == "after")
        self.assertEqual(scheduled[1], 120)
        scheduled[2]()
        self.assertIn(("complete-stop", None), events)

    def test_partial_inference_is_bounded_and_window_is_not_reused_as_full_final(self):
        backend = FakeBackend()
        stream = StreamingDictation(
            backend,
            PROFILE,
            min_audio_seconds=0.1,
            max_partial_audio_seconds=2.0,
        )
        audio = np.zeros(5 * 16_000, dtype=np.float32)

        partial = stream.process_snapshot(audio)
        final = stream.finalize(audio)

        self.assertEqual(backend.calls[0].size, 2 * 16_000)
        self.assertEqual(partial.processed_samples, 5 * 16_000)
        self.assertTrue(partial.windowed)
        self.assertFalse(final.reused_partial)
        self.assertEqual(backend.calls[1].size, 5 * 16_000)

    def test_cancel_keeps_live_thread_visible_until_inference_finishes(self):
        backend = FakeBackend()
        backend.gate = threading.Event()
        recorder = FakeRecorder(np.zeros(16_000, dtype=np.float32))
        stream = StreamingDictation(
            backend, PROFILE, min_audio_seconds=0.1, interval_seconds=0.01
        )
        stream.start(recorder)
        deadline = time.monotonic() + 1.0
        while backend.concurrent == 0 and time.monotonic() < deadline:
            time.sleep(0.005)

        stopped = stream.cancel(timeout=0.01)

        self.assertFalse(stopped)
        self.assertTrue(stream.is_running)
        backend.gate.set()
        self.assertTrue(stream.cancel(timeout=1.0))
        self.assertFalse(stream.is_running)

    def test_app_close_waits_for_finalizer_before_destroying_ui(self):
        events = []

        class FakeRoot:
            def destroy(self):
                events.append("destroyed")

        class FakeHotkey:
            def stop(self):
                events.append("hotkey-stopped")

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.closing = threading.Event()
        app.streaming = None
        app.recording = False
        app.recorder = None
        app.root = FakeRoot()
        app.hotkey = FakeHotkey()

        def finish_work():
            time.sleep(0.04)
            events.append("finalizer-finished")

        app.finalizer_thread = threading.Thread(target=finish_work, daemon=False)
        app.finalizer_thread.start()

        app.close()

        self.assertEqual(
            events,
            ["finalizer-finished", "hotkey-stopped", "destroyed"],
        )

    def test_background_worker_reports_partial_failure(self):
        backend = FakeBackend()
        backend.error = RuntimeError("partial failed")
        errors = []
        recorder = FakeRecorder(np.zeros(16_000, dtype=np.float32))
        stream = StreamingDictation(
            backend,
            PROFILE,
            min_audio_seconds=0.1,
            interval_seconds=0.01,
            on_error=errors.append,
        )

        stream.start(recorder)
        time.sleep(0.04)
        stream.cancel()

        self.assertEqual(str(errors[0]), "partial failed")

    def test_ignores_audio_before_minimum_window(self):
        backend = FakeBackend()
        stream = StreamingDictation(backend, PROFILE, min_audio_seconds=1.0)

        result = stream.process_snapshot(np.zeros(15_999, dtype=np.float32))

        self.assertIsNone(result)
        self.assertEqual(backend.calls, [])

    def test_publishes_partial_after_minimum_window(self):
        backend = FakeBackend()
        partials = []
        stream = StreamingDictation(
            backend, PROFILE, min_audio_seconds=1.0, on_partial=partials.append
        )

        result = stream.process_snapshot(np.zeros(16_000, dtype=np.float32))

        self.assertEqual(result.text, "partial 16000")
        self.assertEqual(result.processed_samples, 16_000)
        self.assertEqual(partials[-1].text, "partial 16000")

    def test_model_calls_never_overlap(self):
        backend = FakeBackend()
        backend.gate = threading.Event()
        stream = StreamingDictation(backend, PROFILE, min_audio_seconds=0.1)
        audio = np.zeros(16_000, dtype=np.float32)
        threads = [threading.Thread(target=stream.process_snapshot, args=(audio,)) for _ in range(2)]
        for thread in threads:
            thread.start()
        time.sleep(0.05)
        backend.gate.set()
        for thread in threads:
            thread.join(timeout=1)

        self.assertEqual(backend.max_concurrent, 1)

    def test_partial_and_concurrent_finalize_never_overlap(self):
        backend = FakeBackend()
        backend.gate = threading.Event()
        stream = StreamingDictation(backend, PROFILE, min_audio_seconds=0.1)
        partial_audio = np.zeros(16_000, dtype=np.float32)
        final_audio = np.full(20_000, 0.02, dtype=np.float32)
        partial_thread = threading.Thread(
            target=stream.process_snapshot, args=(partial_audio,)
        )
        final_thread = threading.Thread(target=stream.finalize, args=(final_audio,))

        partial_thread.start()
        deadline = time.monotonic() + 1.0
        while backend.concurrent == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        final_thread.start()
        time.sleep(0.03)

        self.assertEqual(backend.max_concurrent, 1)
        backend.gate.set()
        partial_thread.join(timeout=1)
        final_thread.join(timeout=1)
        self.assertEqual(backend.max_concurrent, 1)

    def test_quiet_two_hundred_ms_tail_forces_final_inference(self):
        backend = FakeBackend()
        stream = StreamingDictation(backend, PROFILE, min_audio_seconds=0.1)
        speech = np.full(16_000, 0.02, dtype=np.float32)
        stream.process_snapshot(speech)
        final_audio = np.concatenate([speech, np.zeros(3_200, dtype=np.float32)])

        result = stream.finalize(final_audio)

        self.assertFalse(result.reused_partial)
        self.assertEqual(result.text, "partial 19200")
        self.assertEqual(len(backend.calls), 2)

    def test_negligible_quiet_tail_can_reuse_latest_partial(self):
        backend = FakeBackend()
        stream = StreamingDictation(
            backend,
            PROFILE,
            min_audio_seconds=0.1,
            allow_quiet_tail_reuse=True,
        )
        speech = np.full(16_000, 0.02, dtype=np.float32)
        partial = stream.process_snapshot(speech)
        final_audio = np.concatenate([speech, np.zeros(640, dtype=np.float32)])

        result = stream.finalize(final_audio)

        self.assertTrue(result.reused_partial)
        self.assertEqual(result.text, partial.text)
        self.assertEqual(len(backend.calls), 1)

    def test_reliable_default_forces_complete_final_pass_for_quiet_tail(self):
        backend = FakeBackend()
        stream = StreamingDictation(backend, PROFILE, min_audio_seconds=0.1)
        speech = np.full(16_000, 0.02, dtype=np.float32)
        stream.process_snapshot(speech)
        final_audio = np.concatenate([speech, np.zeros(640, dtype=np.float32)])

        result = stream.finalize(final_audio)

        self.assertFalse(result.reused_partial)
        self.assertEqual(result.text, "partial 16640")
        self.assertEqual(len(backend.calls), 2)

    def test_accurate_final_mode_requests_non_fast_full_buffer_decode(self):
        fast_values = []

        class DecodingBackend:
            def transcribe_samples(self, samples, profile, fast=True):
                fast_values.append(fast)
                return "complete tail", {"transcribe_seconds": 0.01}

        stream = StreamingDictation(
            DecodingBackend(),
            PROFILE,
            min_audio_seconds=0.1,
            final_decoding_mode="accurate_full_buffer",
        )

        result = stream.finalize(np.ones(16_000, dtype=np.float32))

        self.assertEqual(result.text, "complete tail")
        self.assertEqual(fast_values, [False])

    def test_spoken_tail_forces_final_inference(self):
        backend = FakeBackend()
        stream = StreamingDictation(backend, PROFILE, min_audio_seconds=0.1)
        first = np.full(16_000, 0.02, dtype=np.float32)
        stream.process_snapshot(first)
        final_audio = np.concatenate([first, np.full(3_200, 0.02, dtype=np.float32)])

        result = stream.finalize(final_audio)

        self.assertFalse(result.reused_partial)
        self.assertEqual(result.text, "partial 19200")
        self.assertEqual(len(backend.calls), 2)

    def test_background_worker_can_be_cancelled(self):
        backend = FakeBackend()
        recorder = FakeRecorder(np.zeros(16_000, dtype=np.float32))
        stream = StreamingDictation(
            backend, PROFILE, min_audio_seconds=0.1, interval_seconds=0.01
        )
        stream.start(recorder)
        self.assertFalse(stream._thread.daemon)
        time.sleep(0.04)

        stream.cancel()

        self.assertFalse(stream.is_running)


if __name__ == "__main__":
    unittest.main()
