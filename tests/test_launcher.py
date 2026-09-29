import unittest
from pathlib import Path
from unittest.mock import Mock

import launch_glitchscribe as launcher
from drew_local_voice.tray import SystemTrayIcon


class GlitchScribeLauncherTests(unittest.TestCase):
    def test_launcher_posts_open_message_to_existing_tray_window(self):
        post_message = Mock()

        self.assertTrue(
            launcher.request_open_settings(
                find_window=lambda: 321,
                post_message=post_message,
            )
        )

        post_message.assert_called_once_with(
            321,
            launcher.EXTERNAL_OPEN_SETTINGS_MESSAGE,
            0,
            0,
        )

    def test_tray_callback_uses_thread_safe_ui_dispatcher(self):
        root = Mock()
        post_to_ui = Mock(return_value=True)
        callback = Mock()
        tray = SystemTrayIcon(
            root,
            Path("unused.ico"),
            on_toggle_dictation=Mock(),
            on_open_settings=Mock(),
            on_toggle_flyout=Mock(),
            post_to_ui=post_to_ui,
        )

        tray._post_to_tk(callback)

        post_to_ui.assert_called_once_with(callback)
        root.after.assert_not_called()


if __name__ == "__main__":
    unittest.main()
