from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from drew_local_voice.discord_push_to_mute import DiscordPushToMuteKey


class DiscordPushToMuteKeyTests(unittest.TestCase):
    def test_hold_and_release_send_matching_f7_events(self):
        events: list[tuple[str, dict]] = []
        calls: list[bool] = []
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "f7.lock"

            def start_watchdog():
                lock.write_text("{}", encoding="utf-8")
                return lock

            key = DiscordPushToMuteKey(
                event_sink=lambda event, **fields: events.append((event, fields))
            )
            with patch.object(key, "_start_watchdog", side_effect=start_watchdog), patch(
                "drew_local_voice.discord_push_to_mute._send_f7",
                side_effect=lambda *, key_up: calls.append(bool(key_up)) or True,
            ):
                self.assertTrue(key.hold())
                self.assertTrue(key.held)
                self.assertTrue(key.hold())
                self.assertTrue(key.release())
                self.assertFalse(key.held)
                self.assertFalse(lock.exists())

        self.assertEqual(calls, [False, True])
        self.assertEqual(
            [event for event, _fields in events],
            ["discord_push_to_mute_held", "discord_push_to_mute_released"],
        )

    def test_key_down_failure_removes_watchdog_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "f7.lock"

            def start_watchdog():
                lock.write_text("{}", encoding="utf-8")
                return lock

            key = DiscordPushToMuteKey()
            with patch.object(key, "_start_watchdog", side_effect=start_watchdog), patch(
                "drew_local_voice.discord_push_to_mute._send_f7",
                return_value=False,
            ):
                self.assertFalse(key.hold())
                self.assertFalse(key.held)
                self.assertFalse(lock.exists())


if __name__ == "__main__":
    unittest.main()
