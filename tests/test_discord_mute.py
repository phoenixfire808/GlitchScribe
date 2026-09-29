from __future__ import annotations

import time
import unittest

from drew_local_voice.discord_mute import DiscordMuteCoordinator


class FakeDiscordBackend:
    def __init__(self, state: bool | None = False, *, fail_set: bool = False) -> None:
        self.state = state
        self.fail_set = fail_set
        self.set_calls: list[bool] = []
        self.closed = False
        self._owns_state = False
        self._prior_state: bool | None = None

    def muted_state(self) -> bool | None:
        return self.state

    def set_muted(self, muted: bool) -> bool:
        self.set_calls.append(bool(muted))
        if self.fail_set:
            return False
        if muted:
            if self.state is None:
                return False
            if not self._owns_state:
                self._prior_state = bool(self.state)
                self._owns_state = True
            self.state = True
        elif self._owns_state:
            self.state = self._prior_state
            self._owns_state = False
            self._prior_state = None
        else:
            self.state = False
        return True

    def reconcile_muted(self) -> tuple[int, int, bool]:
        if self.state is None:
            return 0, 0, True
        newly_owned = 0 if self._owns_state else 1
        verified = self.set_muted(True)
        return 1, newly_owned, verified

    def close(self) -> None:
        self.closed = True


class PartialAcquireBackend(FakeDiscordBackend):
    def reconcile_muted(self) -> tuple[int, int, bool]:
        if not self._owns_state:
            self._prior_state = bool(self.state)
            self._owns_state = True
        self.state = True
        return 1, 1, False


class RestoreRetryBackend(FakeDiscordBackend):
    def __init__(self) -> None:
        super().__init__(False)
        self.restore_attempts = 0

    def set_muted(self, muted: bool) -> bool:
        if not muted:
            self.restore_attempts += 1
            if self.restore_attempts == 1:
                return False
        return super().set_muted(muted)


class DiscordMuteCoordinatorTests(unittest.TestCase):
    def make_coordinator(self, backend: FakeDiscordBackend, *, enabled: bool = True):
        events: list[tuple[str, dict]] = []
        coordinator = DiscordMuteCoordinator(
            enabled=enabled,
            event_sink=lambda event, **fields: events.append((event, fields)),
            backend_factory=lambda: backend,
        )
        return coordinator, events

    def test_unmuted_discord_is_muted_then_restored(self):
        backend = FakeDiscordBackend(False)
        coordinator, events = self.make_coordinator(backend)
        try:
            begin = coordinator.begin_recording(wait_timeout=1.0)
            self.assertEqual(begin.status, "muted")
            self.assertTrue(begin.changed)
            self.assertTrue(backend.state)

            end = coordinator.end_recording(wait_timeout=1.0)
            self.assertEqual(end.status, "restored")
            self.assertFalse(backend.state)
            self.assertEqual(backend.set_calls, [True, False])
            self.assertIn("discord_mute_begin_completed", [event for event, _ in events])
            self.assertIn("discord_mute_end_completed", [event for event, _ in events])
        finally:
            coordinator.close(wait_timeout=1.0)

    def test_already_muted_discord_stays_muted(self):
        backend = FakeDiscordBackend(True)
        coordinator, _events = self.make_coordinator(backend)
        try:
            begin = coordinator.begin_recording(wait_timeout=1.0)
            self.assertEqual(begin.status, "already_muted")
            self.assertFalse(begin.changed)
            end = coordinator.end_recording(wait_timeout=1.0)
            self.assertEqual(end.status, "restored")
            self.assertFalse(end.changed)
            self.assertTrue(backend.state)
            self.assertEqual(backend.set_calls, [True, False])
        finally:
            coordinator.close(wait_timeout=1.0)

    def test_external_unmute_during_recording_is_not_toggled_on_release(self):
        backend = FakeDiscordBackend(False)
        coordinator, _events = self.make_coordinator(backend)
        try:
            coordinator.begin_recording(wait_timeout=1.0)
            backend.state = False
            end = coordinator.end_recording(wait_timeout=1.0)
            self.assertEqual(end.status, "restored")
            self.assertFalse(backend.state)
            self.assertEqual(backend.set_calls, [True, False])
        finally:
            coordinator.close(wait_timeout=1.0)

    def test_unavailable_control_never_blocks_dictation_lifecycle(self):
        backend = FakeDiscordBackend(None)
        coordinator, _events = self.make_coordinator(backend)
        try:
            begin = coordinator.begin_recording(wait_timeout=1.0)
            self.assertEqual(begin.status, "control_unavailable")
            end = coordinator.end_recording(wait_timeout=1.0)
            self.assertEqual(end.status, "unchanged")
            self.assertEqual(backend.set_calls, [])
        finally:
            coordinator.close(wait_timeout=1.0)

    def test_new_discord_session_is_isolated_while_recording(self):
        backend = FakeDiscordBackend(None)
        coordinator, events = self.make_coordinator(backend)
        try:
            begin = coordinator.begin_recording(wait_timeout=1.0)
            self.assertEqual(begin.status, "control_unavailable")

            backend.state = False
            time.sleep(0.35)
            self.assertTrue(backend.state)

            end = coordinator.end_recording(wait_timeout=1.0)
            self.assertEqual(end.status, "restored")
            self.assertFalse(backend.state)
            self.assertIn(
                "discord_isolation_new_session_muted",
                [event for event, _fields in events],
            )
        finally:
            coordinator.close(wait_timeout=1.0)

    def test_disabled_setting_does_not_touch_discord(self):
        backend = FakeDiscordBackend(False)
        coordinator, events = self.make_coordinator(backend, enabled=False)
        try:
            begin = coordinator.begin_recording(wait_timeout=1.0)
            self.assertEqual(begin.status, "disabled")
            self.assertEqual(backend.set_calls, [])
            self.assertIn("discord_mute_skipped", [event for event, _ in events])
        finally:
            coordinator.close(wait_timeout=1.0)

    def test_partial_acquisition_failure_is_owned_and_rolled_back(self):
        backend = PartialAcquireBackend(False)
        coordinator, _events = self.make_coordinator(backend)
        try:
            begin = coordinator.begin_recording(wait_timeout=1.0)
            self.assertEqual(begin.status, "mute_failed")
            self.assertTrue(backend.state)

            end = coordinator.end_recording(wait_timeout=1.0)
            self.assertEqual(end.status, "restored")
            self.assertFalse(backend.state)
        finally:
            coordinator.close(wait_timeout=1.0)

    def test_restore_failure_is_retried_on_close(self):
        backend = RestoreRetryBackend()
        coordinator, _events = self.make_coordinator(backend)
        begin = coordinator.begin_recording(wait_timeout=1.0)
        self.assertEqual(begin.status, "muted")

        end = coordinator.end_recording(wait_timeout=1.0)
        self.assertEqual(end.status, "restore_failed")
        self.assertTrue(backend.state)

        coordinator.close(wait_timeout=1.0)
        self.assertFalse(backend.state)
        self.assertGreaterEqual(backend.restore_attempts, 2)

    def test_close_restores_an_active_integration_mute(self):
        backend = FakeDiscordBackend(False)
        coordinator, _events = self.make_coordinator(backend)
        coordinator.begin_recording(wait_timeout=1.0)
        receipt = coordinator.close(wait_timeout=1.0)
        self.assertEqual(receipt.status, "restored")
        self.assertFalse(backend.state)
        self.assertTrue(backend.closed)
        self.assertEqual(backend.set_calls, [True, False])


if __name__ == "__main__":
    unittest.main()
