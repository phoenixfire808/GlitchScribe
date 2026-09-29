import tempfile
import unittest
from pathlib import Path

from drew_local_voice.window_shield import (
    BlockEvent,
    EventLog,
    NoOpAdapter,
    ShieldConfig,
    WindowInfo,
    WindowShield,
    load_config,
    save_config,
)


class WindowShieldTests(unittest.TestCase):
    def make_shield(self, windows):
        adapter = NoOpAdapter()
        adapter.set_visible_windows(windows)
        shield = WindowShield(config=ShieldConfig(), adapter=adapter)
        return shield, adapter

    def test_terminal_executable_is_hidden_and_logged(self):
        shield, adapter = self.make_shield([
            WindowInfo(101, 42, "powershell.exe", "PowerShell", True),
        ])

        events = shield.tick()

        self.assertEqual(1, len(events))
        self.assertEqual(101, events[0].hwnd)
        self.assertEqual("both", events[0].matched_by)
        self.assertEqual(1, len(shield.events))
        self.assertFalse(adapter.is_window_visible(101))

    def test_title_matching_can_be_disabled(self):
        shield, adapter = self.make_shield([
            WindowInfo(102, 42, "custom.exe", "Build Terminal", True),
        ])
        shield.update_config(use_title_keyword_match=False)

        self.assertEqual([], shield.tick())
        self.assertTrue(adapter.is_window_visible(102))

    def test_never_hide_and_self_pid_are_excluded(self):
        adapter = NoOpAdapter()
        adapter.set_visible_windows([
            WindowInfo(103, 42, "explorer.exe", "File Explorer", True),
            WindowInfo(104, adapter.current_pid(), "powershell.exe", "PowerShell", True),
        ])
        shield = WindowShield(config=ShieldConfig(), adapter=adapter)

        self.assertEqual([], shield.tick())
        self.assertTrue(adapter.is_window_visible(103))
        self.assertTrue(adapter.is_window_visible(104))

    def test_allowlist_suppresses_matching_process(self):
        shield, adapter = self.make_shield([
            WindowInfo(105, 42, "cmd.exe", "Command Prompt", True),
        ])
        shield.update_config(allowlist_executables=["cmd.exe"])

        self.assertEqual([], shield.tick())
        self.assertTrue(adapter.is_window_visible(105))

    def test_unblock_is_not_immediately_reblocked(self):
        shield, adapter = self.make_shield([
            WindowInfo(106, 42, "wt.exe", "Windows Terminal", True),
        ])
        shield.tick()

        self.assertTrue(shield.unblock_hwnd(106))
        self.assertTrue(adapter.is_window_visible(106))
        self.assertEqual([], shield.tick())
        self.assertTrue(adapter.is_window_visible(106))

    def test_pause_resume_and_restore_all(self):
        shield, adapter = self.make_shield([
            WindowInfo(107, 42, "pwsh.exe", "PowerShell", True),
        ])
        shield.pause()
        self.assertEqual([], shield.tick())
        shield.resume()
        shield.tick()
        self.assertEqual(1, shield.restore_all())
        self.assertTrue(adapter.is_window_visible(107))

    def test_config_round_trip_and_bad_file_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = ShieldConfig(allowlist_executables=["PWsh.exe"], poll_interval_seconds=20)
            save_config(config, path=path)
            loaded = load_config(path=path)
            self.assertEqual(["pwsh.exe"], loaded.allowlist_executables)
            self.assertEqual(10.0, loaded.poll_interval_seconds)

            path.write_text("not json", encoding="utf-8")
            fallback = load_config(path=path)
            self.assertIn("pwsh.exe", fallback.blocked_executables)

    def test_event_log_is_bounded(self):
        log = EventLog(max_events=50)
        for index in range(60):
            log.append(BlockEvent(
                timestamp=str(index), hwnd=index, pid=index, process_name="cmd.exe", title="", matched_by="executable"
            ))
        self.assertEqual(50, len(log))
        self.assertEqual("10", log.snapshot()[0].timestamp)


if __name__ == "__main__":
    unittest.main()
