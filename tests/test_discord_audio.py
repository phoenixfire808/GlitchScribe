from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from drew_local_voice.discord_audio import (
    WindowsDiscordCaptureSessions,
    _DiscordCaptureSession,
)


class FakeVolume:
    def __init__(self, muted: bool) -> None:
        self.muted = bool(muted)
        self.set_calls: list[bool] = []

    def GetMute(self) -> bool:
        return self.muted

    def SetMute(self, muted: bool, _event_context) -> None:
        self.set_calls.append(bool(muted))
        self.muted = bool(muted)


class FailOnMuteVolume(FakeVolume):
    def SetMute(self, muted: bool, event_context) -> None:
        if muted:
            raise OSError("simulated mute failure")
        super().SetMute(muted, event_context)


class DiscordCaptureSessionTests(unittest.TestCase):
    def make_backend(self, sessions: list[_DiscordCaptureSession]):
        backend = WindowsDiscordCaptureSessions.__new__(
            WindowsDiscordCaptureSessions
        )
        backend._restore = {}
        backend._closed = False
        backend._discord_sessions = lambda: list(sessions)
        return backend

    def test_only_installed_discord_path_matches(self):
        with patch.dict(os.environ, {"LOCALAPPDATA": r"C:\Users\TestUser\AppData\Local"}):
            self.assertTrue(
                WindowsDiscordCaptureSessions._is_installed_discord(
                    r"C:\Users\TestUser\AppData\Local\Discord\app-1.0.9251\Discord.exe"
                )
            )
            self.assertFalse(
                WindowsDiscordCaptureSessions._is_installed_discord(
                    r"C:\Temp\Discord.exe"
                )
            )
            self.assertFalse(
                WindowsDiscordCaptureSessions._is_installed_discord(
                    r"C:\Users\TestUser\AppData\Local\Discord\app-1.0.9251\Other.exe"
                )
            )

    def test_reconcile_preserves_each_sessions_prior_state(self):
        first = FakeVolume(False)
        second = FakeVolume(True)
        sessions = [
            _DiscordCaptureSession(100, 1001, first),
            _DiscordCaptureSession(200, 2001, second),
        ]
        backend = self.make_backend(sessions)

        count, newly_owned, verified = backend.reconcile_muted()
        self.assertEqual((count, newly_owned, verified), (2, 2, True))
        self.assertTrue(first.muted)
        self.assertTrue(second.muted)

        self.assertTrue(backend.set_muted(False))
        self.assertFalse(first.muted)
        self.assertTrue(second.muted)

    def test_reconcile_owns_session_that_appears_later(self):
        first = FakeVolume(False)
        sessions = [_DiscordCaptureSession(100, 1001, first)]
        backend = self.make_backend(sessions)

        self.assertEqual(backend.reconcile_muted(), (1, 1, True))

        second = FakeVolume(False)
        sessions.append(_DiscordCaptureSession(200, 2001, second))
        self.assertEqual(backend.reconcile_muted(), (2, 1, True))
        self.assertTrue(second.muted)

        self.assertTrue(backend.set_muted(False))
        self.assertFalse(first.muted)
        self.assertFalse(second.muted)

    def test_partial_acquisition_failure_rolls_back_owned_sessions(self):
        first = FakeVolume(False)
        second = FailOnMuteVolume(False)
        sessions = [
            _DiscordCaptureSession(100, 1001, first),
            _DiscordCaptureSession(200, 2001, second),
        ]
        backend = self.make_backend(sessions)

        self.assertEqual(backend.reconcile_muted(), (2, 2, False))
        self.assertTrue(first.muted)
        self.assertFalse(second.muted)

        self.assertTrue(backend.set_muted(False))
        self.assertFalse(first.muted)
        self.assertFalse(second.muted)

    def test_vanished_session_is_resolved_during_restore(self):
        first = FakeVolume(False)
        sessions = [_DiscordCaptureSession(100, 1001, first)]
        backend = self.make_backend(sessions)

        self.assertEqual(backend.reconcile_muted(), (1, 1, True))
        sessions.clear()
        self.assertTrue(backend.set_muted(False))
        self.assertEqual(backend._restore, {})


if __name__ == "__main__":
    unittest.main()
