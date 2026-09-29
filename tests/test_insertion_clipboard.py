from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from drew_local_voice.insertion import (
    CF_TEXT,
    CF_UNICODETEXT,
    ClipboardFormatSnapshot,
    ClipboardSnapshot,
    InsertionPolicy,
    LastTranscriptStore,
    SecondaryHotkeyTracker,
    default_insertion_policy,
    insert_text_with_fallback,
    normalize_inject_delay_ms,
    parse_secondary_hotkey,
    paste_text_via_clipboard,
    restore_clipboard,
    set_clipboard_text,
    should_prefer_clipboard,
    snapshot_clipboard,
)


class FakeClipboardApi:
    def __init__(self, initial: dict[int, bytes] | None = None, *, fail_open_times: int = 0):
        self.store: dict[int, bytes] = dict(initial or {})
        self.opened = False
        self.fail_open_times = fail_open_times
        self.open_calls = 0
        self.empty_calls = 0

    def open(self, owner: int = 0) -> bool:
        self.open_calls += 1
        if self.fail_open_times > 0:
            self.fail_open_times -= 1
            return False
        self.opened = True
        return True

    def close(self) -> None:
        self.opened = False

    def empty(self) -> bool:
        self.empty_calls += 1
        self.store.clear()
        return True

    def sequence_number(self) -> int:
        return len(self.store)

    def enum_formats(self) -> list[int]:
        return list(self.store.keys())

    def get_format_bytes(self, format_id: int) -> bytes | None:
        if format_id not in self.store:
            return None
        return self.store[format_id]

    def set_format_bytes(self, format_id: int, data: bytes) -> bool:
        self.store[int(format_id)] = bytes(data)
        return True

    def set_unicode_text(self, text: str) -> bool:
        payload = text.encode("utf-16-le") + b"\x00\x00"
        return self.set_format_bytes(CF_UNICODETEXT, payload)


class InsertionClipboardTests(unittest.TestCase):
    def test_threshold_prefers_clipboard(self):
        policy = default_insertion_policy(clipboard_threshold_chars=10)
        self.assertFalse(should_prefer_clipboard("short", policy))
        self.assertTrue(should_prefer_clipboard("0123456789X", policy))

    def test_inject_delay_is_normalized_and_applied_before_unicode(self):
        self.assertEqual(normalize_inject_delay_ms(None, default=25), 25)
        self.assertEqual(normalize_inject_delay_ms(-1), 0)
        self.assertEqual(normalize_inject_delay_ms(5000), 2000)

        events = []
        result = insert_text_with_fallback(
            "hello",
            type_unicode=lambda text: events.append(("typed", text)),
            policy=default_insertion_policy(
                clipboard_fallback=False,
                inject_delay_ms=125,
            ),
            sleep=lambda seconds: events.append(("sleep", seconds)),
        )
        self.assertTrue(result.complete)
        self.assertEqual(events, [("sleep", 0.125), ("typed", "hello")])

    def test_clipboard_delay_runs_before_paste_and_preserves_settle(self):
        api = FakeClipboardApi()
        events = []
        result = paste_text_via_clipboard(
            "delayed paste",
            api=api,
            policy=InsertionPolicy(
                inject_delay_ms=250,
                paste_settle_ms=40,
                clipboard_open_retries=1,
                clipboard_open_retry_ms=0,
            ),
            send_paste=lambda: events.append(("paste", None)),
            sleep=lambda seconds: events.append(("sleep", seconds)),
        )
        self.assertTrue(result.complete)
        self.assertEqual(
            events,
            [("sleep", 0.25), ("paste", None), ("sleep", 0.04)],
        )

    def test_snapshot_and_restore_round_trip(self):
        api = FakeClipboardApi(
            {
                CF_UNICODETEXT: "prior".encode("utf-16-le") + b"\x00\x00",
                CF_TEXT: b"prior\x00",
            }
        )
        snap = snapshot_clipboard(api, open_retries=1, retry_ms=0)
        self.assertFalse(snap.empty)
        api.store.clear()
        self.assertTrue(restore_clipboard(snap, api, open_retries=1, retry_ms=0))
        self.assertIn(CF_UNICODETEXT, api.store)
        self.assertIn(CF_TEXT, api.store)

    def test_paste_restores_prior_clipboard(self):
        api = FakeClipboardApi(
            {CF_UNICODETEXT: "keep-me".encode("utf-16-le") + b"\x00\x00"}
        )
        paste_events = []

        result = paste_text_via_clipboard(
            "dictated text",
            api=api,
            policy=InsertionPolicy(paste_settle_ms=0, clipboard_open_retries=1, clipboard_open_retry_ms=0),
            send_paste=lambda: paste_events.append(api.store.get(CF_UNICODETEXT)),
            sleep=lambda _s: None,
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.method, "clipboard")
        self.assertEqual(len(paste_events), 1)
        # After paste, prior clipboard restored.
        restored = api.store[CF_UNICODETEXT].decode("utf-16-le").rstrip("\x00")
        self.assertEqual(restored, "keep-me")

    def test_fallback_uses_clipboard_when_unicode_fails(self):
        paste_calls = []

        def boom(_text: str) -> None:
            raise OSError("SendInput failed")

        def fake_paste(text: str):
            paste_calls.append(text)
            from drew_local_voice.insertion import InsertionResult

            return InsertionResult(len(text), len(text), reason="clipboard_paste", method="clipboard")

        result = insert_text_with_fallback(
            "hello",
            type_unicode=boom,
            policy=default_insertion_policy(clipboard_fallback=True, clipboard_threshold_chars=9999),
            paste_via_clipboard=fake_paste,
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.reason, "clipboard_paste")
        self.assertEqual(paste_calls, ["hello"])

    def test_threshold_uses_clipboard_first(self):
        typed = []
        pasted = []

        def type_fn(text: str) -> None:
            typed.append(text)

        def fake_paste(text: str):
            pasted.append(text)
            from drew_local_voice.insertion import InsertionResult

            return InsertionResult(len(text), len(text), reason="clipboard_paste", method="clipboard")

        long_text = "x" * 50
        result = insert_text_with_fallback(
            long_text,
            type_unicode=type_fn,
            policy=default_insertion_policy(clipboard_threshold_chars=10),
            paste_via_clipboard=fake_paste,
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.reason, "clipboard_threshold")
        self.assertEqual(pasted, [long_text])
        self.assertEqual(typed, [])

    def test_last_transcript_store(self):
        store = LastTranscriptStore()
        self.assertFalse(store.available)
        store.remember("  hi there  ", target_window=7, source="final")
        self.assertTrue(store.available)
        self.assertEqual(store.text, "  hi there  ")
        self.assertEqual(store.target_window, 7)
        store.clear()
        self.assertFalse(store.available)

    def test_parse_secondary_hotkey(self):
        self.assertEqual(parse_secondary_hotkey("Ctrl + Shift + V"), ({"CTRL", "SHIFT"}, "V"))
        self.assertIsNone(parse_secondary_hotkey("None"))
        self.assertIsNone(parse_secondary_hotkey(""))

    def test_secondary_hotkey_tracker_edge_trigger(self):
        tracker = SecondaryHotkeyTracker("Ctrl + Shift + V")
        self.assertTrue(tracker.enabled)
        self.assertFalse(tracker.handle("CTRL", is_down=True))
        self.assertFalse(tracker.handle("SHIFT", is_down=True))
        self.assertTrue(tracker.handle("V", is_down=True))
        # hold should not re-fire
        self.assertFalse(tracker.handle("V", is_down=True))
        tracker.handle("V", is_down=False)
        self.assertTrue(tracker.handle("V", is_down=True))


class PasteLastAppTests(unittest.TestCase):
    def test_paste_last_uses_store_and_insert_policy(self):
        try:
            from local_voice import LocalVoiceApp
        except ModuleNotFoundError as exc:
            self.skipTest(f"local_voice deps unavailable: {exc}")
        from drew_local_voice.insertion import InsertionResult, LastTranscriptStore

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        app.last_transcript_store = LastTranscriptStore()
        app.last_transcript_store.remember("prior phrase", target_window=1)
        app.last_text = ""
        app.last_inserted_payload = ""
        app._target_is_active = Mock(return_value=True)
        app._set_status = Mock()
        app._insert_text_with_policy = Mock(
            return_value=InsertionResult(12, 12, reason="ok", method="unicode")
        )
        with patch("local_voice.append_runtime_event"):
            ok = app._paste_last_transcript(99)
        self.assertTrue(ok)
        app._insert_text_with_policy.assert_called_once()
        args = app._insert_text_with_policy.call_args
        self.assertEqual(args.args[0], "prior phrase")
        self.assertEqual(args.args[1], 99)

    def test_settings_include_clipboard_keys(self):
        from drew_local_voice.settings import built_in_settings

        keys = {spec.key for spec in built_in_settings()}
        self.assertIn("insertion.clipboard_fallback", keys)
        self.assertIn("insertion.clipboard_threshold_chars", keys)
        self.assertIn("general.paste_last_hotkey", keys)


if __name__ == "__main__":
    unittest.main()
