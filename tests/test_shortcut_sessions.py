from __future__ import annotations

import importlib
import importlib.util
import unittest


class ShortcutSessionTests(unittest.TestCase):
    def test_latched_session_stops_only_after_toggle_chord_is_released_and_repressed(self):
        shortcuts = importlib.import_module("drew_local_voice.shortcut_sessions")
        controller = shortcuts.ShortcutSessionController(
            hold_chord={"CTRL", "SPACE"},
            latch_chord={"CTRL", "SHIFT", "SPACE"},
        )

        controller.handle("SHIFT", is_down=True)
        controller.handle("CTRL", is_down=True)
        self.assertEqual(
            controller.handle("SPACE", is_down=True),
            shortcuts.ShortcutAction.START_LATCHED,
        )
        self.assertEqual(
            controller.handle("SPACE", is_down=True),
            shortcuts.ShortcutAction.NONE,
        )
        controller.handle("SPACE", is_down=False)
        controller.handle("CTRL", is_down=False)
        controller.handle("SHIFT", is_down=False)

        controller.handle("CTRL", is_down=True)
        controller.handle("SHIFT", is_down=True)
        self.assertEqual(
            controller.handle("SPACE", is_down=True),
            shortcuts.ShortcutAction.STOP_LATCHED,
        )
        self.assertEqual(controller.state, shortcuts.ShortcutState.IDLE)

    def test_latching_a_hold_does_not_stop_until_a_fresh_toggle_activation(self):
        shortcuts = importlib.import_module("drew_local_voice.shortcut_sessions")
        controller = shortcuts.ShortcutSessionController(
            hold_chord={"CTRL", "SPACE"},
            latch_chord={"CTRL", "SHIFT", "SPACE"},
        )

        controller.handle("CTRL", is_down=True)
        self.assertEqual(
            controller.handle("SPACE", is_down=True),
            shortcuts.ShortcutAction.START_HOLD,
        )
        self.assertEqual(
            controller.handle("SHIFT", is_down=True),
            shortcuts.ShortcutAction.LATCH,
        )
        controller.handle("SPACE", is_down=False)
        controller.handle("CTRL", is_down=False)
        controller.handle("SHIFT", is_down=False)
        controller.handle("SHIFT", is_down=True)
        controller.handle("CTRL", is_down=True)

        self.assertEqual(
            controller.handle("SPACE", is_down=True),
            shortcuts.ShortcutAction.STOP_LATCHED,
        )

    def test_reset_preserves_held_inputs_until_physical_release(self):
        shortcuts = importlib.import_module("drew_local_voice.shortcut_sessions")
        controller = shortcuts.ShortcutSessionController(
            hold_chord={"F8"},
            latch_chord={"F8", "SHIFT"},
        )

        controller.handle("SHIFT", is_down=True)
        self.assertEqual(
            controller.handle("F8", is_down=True),
            shortcuts.ShortcutAction.START_LATCHED,
        )
        controller.handle("F8", is_down=False)
        controller.handle("SHIFT", is_down=False)
        controller.handle("SHIFT", is_down=True)
        self.assertEqual(
            controller.handle("F8", is_down=True),
            shortcuts.ShortcutAction.STOP_LATCHED,
        )

        controller.reset()
        self.assertEqual(
            controller.handle("F8", is_down=True),
            shortcuts.ShortcutAction.NONE,
        )
        controller.handle("F8", is_down=False)
        controller.handle("SHIFT", is_down=False)
        self.assertEqual(
            controller.handle("F8", is_down=True),
            shortcuts.ShortcutAction.START_HOLD,
        )

    def test_extended_chord_modifier_order_starts_latched_exactly_once(self):
        shortcuts = importlib.import_module("drew_local_voice.shortcut_sessions")
        controller = shortcuts.ShortcutSessionController(
            hold_chord={"CTRL", "SPACE"},
            latch_chord={"CTRL", "SHIFT", "SPACE"},
        )

        self.assertEqual(
            controller.handle("SHIFT", is_down=True),
            shortcuts.ShortcutAction.NONE,
        )
        self.assertEqual(
            controller.handle("CTRL", is_down=True),
            shortcuts.ShortcutAction.NONE,
        )
        self.assertEqual(
            controller.handle("SPACE", is_down=True),
            shortcuts.ShortcutAction.START_LATCHED,
        )
        self.assertEqual(
            controller.handle("SPACE", is_down=True),
            shortcuts.ShortcutAction.NONE,
        )
        self.assertEqual(controller.state, shortcuts.ShortcutState.LATCHED)

    def test_extra_modifier_latches_existing_hold_without_stop(self):
        shortcuts = importlib.import_module("drew_local_voice.shortcut_sessions")
        controller = shortcuts.ShortcutSessionController(
            hold_chord={"CTRL", "SPACE"},
            latch_chord={"CTRL", "SHIFT", "SPACE"},
        )

        self.assertEqual(
            controller.handle("CTRL", is_down=True),
            shortcuts.ShortcutAction.NONE,
        )
        self.assertEqual(
            controller.handle("SPACE", is_down=True),
            shortcuts.ShortcutAction.START_HOLD,
        )
        self.assertEqual(
            controller.handle("SHIFT", is_down=True),
            shortcuts.ShortcutAction.LATCH,
        )
        self.assertEqual(controller.state, shortcuts.ShortcutState.LATCHED)
        self.assertEqual(
            controller.handle("SPACE", is_down=False),
            shortcuts.ShortcutAction.NONE,
        )
        self.assertEqual(
            controller.handle("CTRL", is_down=False),
            shortcuts.ShortcutAction.NONE,
        )
        self.assertEqual(controller.state, shortcuts.ShortcutState.LATCHED)

    def test_hold_chord_starts_on_completion_and_stops_on_release(self):
        self.assertIsNotNone(
            importlib.util.find_spec("drew_local_voice.shortcut_sessions"),
            "hold/latch behavior needs a reusable session controller",
        )
        shortcuts = importlib.import_module("drew_local_voice.shortcut_sessions")
        controller_type = getattr(shortcuts, "ShortcutSessionController", None)
        action = getattr(shortcuts, "ShortcutAction", None)
        self.assertIsNotNone(controller_type)
        self.assertIsNotNone(action)

        controller = controller_type(
            hold_chord={"CTRL", "SPACE"},
            latch_chord={"CTRL", "SHIFT", "SPACE"},
        )

        self.assertEqual(controller.handle("CTRL", is_down=True), action.NONE)
        self.assertEqual(controller.handle("SPACE", is_down=True), action.START_HOLD)
        self.assertEqual(controller.handle("SPACE", is_down=False), action.STOP_HOLD)
        self.assertEqual(controller.state.value, "idle")


if __name__ == "__main__":
    unittest.main()
