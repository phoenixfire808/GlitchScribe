"""Action Wheel v2 — pie geometry, config, defaults, popup surface."""

from __future__ import annotations

import math
import tkinter as tk
import unittest
from unittest.mock import Mock

from drew_local_voice.ai.action_wheel import (
    ACTION_WHEEL_HOTKEY_CHOICES,
    ACTION_WHEEL_NEXT_PAGE_ID,
    ACTION_WHEEL_PREVIOUS_PAGE_ID,
    ActionWheelSession,
    CONFIG_KEY_ACTIONS,
    CONFIG_KEY_ENABLED,
    DEFAULT_WHEEL_ACTIONS,
    ActionWheel,
    WheelAction,
    add_wheel_action,
    action_wheel_hotkey_label,
    actions_for_wheel_mode,
    decode_wheel_actions_from_settings,
    default_action_ids,
    encode_wheel_actions_for_settings,
    hit_test_pie,
    is_action_wheel_enabled,
    load_wheel_actions_from_config,
    make_custom_ai_wheel_action,
    move_wheel_action,
    paginate_wheel_actions,
    remove_wheel_action,
    reorder_wheel_actions,
    save_wheel_actions_to_config,
    set_action_wheel_enabled,
    slice_index_for_angle,
    stub_wheel_result,
    suggest_custom_wheel_action_id,
)
from drew_local_voice.theme import build_palette


class ActionWheelTests(unittest.TestCase):
    def test_real_canvas_click_dispatches_action_callback(self):
        root = tk.Tk()
        root.withdraw()
        received = Mock()
        try:
            action = DEFAULT_WHEEL_ACTIONS[0]
            wheel = ActionWheel(
                root,
                palette=build_palette("dark", "violet"),
                actions=(action,),
                on_action=received,
                on_dismiss=Mock(),
                reusable=True,
            )
            wheel.show_at(500, 500)
            root.update_idletasks()
            root.update()
            x = wheel.cx
            y = wheel.cy - 100
            wheel.canvas.event_generate("<Motion>", x=x, y=y)
            wheel.canvas.event_generate("<ButtonPress-1>", x=x, y=y)
            wheel.canvas.event_generate("<ButtonRelease-1>", x=x, y=y)
            root.update_idletasks()
            root.update()
            received.assert_called_once_with(action)
        finally:
            try:
                wheel.dismiss()
            except (NameError, tk.TclError):
                pass
            root.destroy()

    def test_action_wheel_session_freezes_target_and_source(self):
        session = ActionWheelSession(
            target_window=101,
            focused_window=202,
            source_text="selected prompt",
            selection_identity="control-1",
            source_kind="uia_selection",
            anchor=(400, 300),
            request_id="wheel-test-001",
        )
        self.assertEqual(session.target_window, 101)
        self.assertEqual(session.focused_window, 202)
        self.assertEqual(session.source_text, "selected prompt")
        self.assertTrue(session.has_source)
        self.assertTrue(session.has_selection)
        self.assertEqual(session.request_id, "wheel-test-001")

    def test_utility_target_session_is_anchor_safe_without_selection(self):
        session = ActionWheelSession(
            target_window=101,
            focused_window=202,
            source_kind="utility_target",
            anchor=(400, 300),
            request_id="wheel-utility-001",
        )
        self.assertEqual(session.mode, "utility")
        self.assertFalse(session.has_selection)
        self.assertEqual(session.anchor, (400, 300))

    def test_action_wheel_session_without_identity_is_not_external_selection(self):
        session = ActionWheelSession(source_text="review text", source_kind="review_buffer")
        self.assertTrue(session.has_source)
        self.assertFalse(session.has_selection)
        self.assertEqual(session.mode, "review")
        self.assertEqual(
            ActionWheelSession(
                source_text="selected",
                selection_identity="control-1",
                source_kind="uia_selection",
            ).mode,
            "selection",
        )
        self.assertEqual(ActionWheelSession().mode, "utility")

    def test_action_wheel_session_freezes_target_and_source(self):
        target_window = 101
        selected_source = "original selection"
        session = ActionWheelSession(
            target_window=target_window,
            source_text=selected_source,
            selection_identity="control-1",
            source_kind="uia_selection",
        )
        target_window = 202
        selected_source = "post-popup selection"
        self.assertEqual(session.target_window, 101)
        self.assertEqual(session.source_text, "original selection")
        self.assertEqual(session.selection_identity, "control-1")

    def test_action_wheel_hotkey_uses_supported_value_and_safe_fallback(self):
        self.assertEqual(
            action_wheel_hotkey_label({"action_wheel.hotkey": "F10"}),
            "F10",
        )
        self.assertEqual(action_wheel_hotkey_label({"action_wheel.hotkey": "bad"}), "Ctrl + Shift + A")
        self.assertIn("Ctrl + Shift + A", ACTION_WHEEL_HOTKEY_CHOICES)

    def test_action_wheel_paginates_large_action_sets(self):
        first, first_index, page_count = paginate_wheel_actions(DEFAULT_WHEEL_ACTIONS, 0)
        second, second_index, _ = paginate_wheel_actions(DEFAULT_WHEEL_ACTIONS, 1)
        expected_pages = math.ceil(len(DEFAULT_WHEEL_ACTIONS) / 6)
        self.assertEqual((first_index, second_index, page_count), (0, 1, expected_pages))
        self.assertEqual(len(first), 7)
        self.assertEqual(first[-1].id, ACTION_WHEEL_NEXT_PAGE_ID)
        expected_second_actions = min(6, max(0, len(DEFAULT_WHEEL_ACTIONS) - 6))
        expected_second_navigation = 1 + int(page_count > 2)
        self.assertEqual(
            len(second),
            expected_second_actions + expected_second_navigation,
        )
        self.assertEqual(second[0].id, ACTION_WHEEL_PREVIOUS_PAGE_ID)

    def test_utility_mode_hides_selection_actions(self):
        utility = actions_for_wheel_mode(DEFAULT_WHEEL_ACTIONS, "utility")
        self.assertTrue(utility)
        self.assertTrue(all(not action.requires_selection for action in utility))
        self.assertEqual(
            len(actions_for_wheel_mode(DEFAULT_WHEEL_ACTIONS, "review")),
            len(DEFAULT_WHEEL_ACTIONS),
        )

    def test_default_actions_cover_requested_slots(self):
        ids = set(default_action_ids())
        required = {
            "upgrade_prompt",
            "cursor_agent_task",
            "make_shorter",
            "bullets",
            "fix_grammar",
            "explain_selection",
            "extract_todos",
            "insert_last_transcript",
            "copy_last_transcript",
            "paste_last_transcript",
            "open_review",
            "open_preview",
        }
        self.assertTrue(required.issubset(ids))
        self.assertEqual(ids, {a.id for a in DEFAULT_WHEEL_ACTIONS})

    def test_selection_mode_exposes_a_broad_local_transform_pack(self):
        expected = {
            "rewrite_clearer",
            "expand_detail",
            "summarize_selection",
            "make_formal",
            "make_casual",
            "improve_flow",
            "key_points",
            "checklist",
            "action_items",
            "email_draft",
            "format_markdown",
            "explain_simple",
            "generate_title",
            "add_code_comments",
            "generate_tests",
            "commit_message",
        }
        actions = {
            action.id: action
            for action in DEFAULT_WHEEL_ACTIONS
            if action.requires_selection
        }
        self.assertTrue(expected.issubset(actions))
        self.assertGreaterEqual(len(actions), 20)
        for action_id in expected:
            self.assertTrue(actions[action_id].instruction.strip())
            self.assertEqual(actions[action_id].kind, "ai")

    def test_stub_upgrade_cursor_and_bullets_are_offline(self):
        upgrade = next(a for a in DEFAULT_WHEEL_ACTIONS if a.id == "upgrade_prompt")
        bullets = next(a for a in DEFAULT_WHEEL_ACTIONS if a.id == "bullets")
        cursor = next(a for a in DEFAULT_WHEEL_ACTIONS if a.id == "cursor_agent_task")
        source = "fix the login bug. ship tonight."
        upgraded = stub_wheel_result(upgrade, source)
        listed = stub_wheel_result(bullets, source)
        agent = stub_wheel_result(cursor, source)
        self.assertIn("Task:", upgraded)
        self.assertIn(source, upgraded)
        self.assertTrue(listed.startswith("- "))
        self.assertIn("Goal:", agent)
        self.assertIn("Steps:", agent)

    def test_hit_test_pie_by_angle_and_radius(self):
        cx = cy = 100.0
        inner, outer = 40.0, 90.0
        count = 4
        # North of center (y smaller) -> first slice (index 0) when start=-pi/2
        north = hit_test_pie(
            cx, cy - 60, cx=cx, cy=cy, inner_radius=inner, outer_radius=outer, count=count
        )
        east = hit_test_pie(
            cx + 60, cy, cx=cx, cy=cy, inner_radius=inner, outer_radius=outer, count=count
        )
        south = hit_test_pie(
            cx, cy + 60, cx=cx, cy=cy, inner_radius=inner, outer_radius=outer, count=count
        )
        west = hit_test_pie(
            cx - 60, cy, cx=cx, cy=cy, inner_radius=inner, outer_radius=outer, count=count
        )
        self.assertEqual(north, 0)
        self.assertEqual(east, 1)
        self.assertEqual(south, 2)
        self.assertEqual(west, 3)
        # Hub / outside
        self.assertIsNone(
            hit_test_pie(
                cx, cy, cx=cx, cy=cy, inner_radius=inner, outer_radius=outer, count=count
            )
        )
        self.assertIsNone(
            hit_test_pie(
                cx,
                cy - 120,
                cx=cx,
                cy=cy,
                inner_radius=inner,
                outer_radius=outer,
                count=count,
            )
        )

    def test_slice_index_wraps_full_circle(self):
        # Just before full circle lands on last slice
        almost = -math.pi / 2 + (2 * math.pi) - 0.001
        self.assertEqual(slice_index_for_angle(almost, 8), 7)
        self.assertEqual(slice_index_for_angle(-math.pi / 2, 8), 0)

    def test_config_load_save_and_helpers(self):
        bag: dict = {}
        self.assertTrue(is_action_wheel_enabled(bag))
        set_action_wheel_enabled(bag, False)
        self.assertFalse(is_action_wheel_enabled(bag))
        set_action_wheel_enabled(bag, True)

        custom = (
            WheelAction(
                id="custom_one",
                label="Custom",
                description="d",
                instruction="i",
                kind="ai",
            ),
            WheelAction(
                id="custom_two",
                label="Two",
                description="d2",
                instruction="",
                requires_selection=False,
                mutates_selection=False,
                kind="copy",
            ),
        )
        save_wheel_actions_to_config(bag, custom)
        self.assertIn(CONFIG_KEY_ACTIONS, bag)
        loaded = load_wheel_actions_from_config(bag)
        self.assertEqual([a.id for a in loaded], ["custom_one", "custom_two"])

        # Invalid falls back
        self.assertEqual(
            load_wheel_actions_from_config({CONFIG_KEY_ACTIONS: "nope"}),
            DEFAULT_WHEEL_ACTIONS,
        )
        self.assertEqual(
            load_wheel_actions_from_config({CONFIG_KEY_ACTIONS: []}),
            DEFAULT_WHEEL_ACTIONS,
        )

        with_extra = add_wheel_action(
            custom,
            WheelAction(id="z", label="Z", description="", instruction=""),
            index=0,
        )
        self.assertEqual(with_extra[0].id, "z")
        trimmed = remove_wheel_action(with_extra, "custom_one")
        self.assertEqual([a.id for a in trimmed], ["z", "custom_two"])
        reordered = reorder_wheel_actions(trimmed, ["custom_two", "z"])
        self.assertEqual([a.id for a in reordered], ["custom_two", "z"])

    def test_control_lab_action_helpers_round_trip_and_validate(self):
        actions = (
            WheelAction(
                id="custom_fix",
                label="Fix",
                description="Local cleanup",
                instruction="Fix grammar only.",
                kind="ai",
            ),
            WheelAction(
                id="copy_last",
                label="Copy",
                description="Copy locally",
                instruction="",
                requires_selection=False,
                mutates_selection=False,
                kind="copy",
            ),
        )
        encoded = encode_wheel_actions_for_settings(actions)
        decoded = decode_wheel_actions_from_settings(encoded)
        self.assertEqual(decoded, actions)
        self.assertEqual(
            decode_wheel_actions_from_settings("not-json"),
            DEFAULT_WHEEL_ACTIONS,
        )

        moved = move_wheel_action(actions, "copy_last", -1)
        self.assertEqual([action.id for action in moved], ["copy_last", "custom_fix"])
        self.assertEqual(move_wheel_action(moved, "copy_last", -1), moved)
        self.assertEqual(move_wheel_action(moved, "missing", 1), moved)

        self.assertEqual(
            suggest_custom_wheel_action_id("Fix grammar", ["custom_fix_grammar"]),
            "custom_fix_grammar_2",
        )
        custom = make_custom_ai_wheel_action(
            action_id="custom_release_notes",
            label="Release notes",
            instruction="Draft concise release notes.",
        )
        self.assertIsNotNone(custom)
        self.assertEqual(custom.kind, "ai")
        self.assertTrue(custom.requires_selection)
        self.assertTrue(custom.mutates_selection)
        self.assertIsNone(
            make_custom_ai_wheel_action(
                action_id="bad id",
                label="Bad",
                instruction="Should be rejected.",
            )
        )

    def test_action_wheel_builds_pie_and_dismisses(self):
        root = tk.Tk()
        root.withdraw()
        chosen = []
        try:
            wheel = ActionWheel(
                root,
                palette=build_palette("dark", "violet"),
                on_action=lambda action: chosen.append(action.id),
            )
            wheel.show_at(120, 140)
            self.assertTrue(wheel.window.winfo_exists())
            self.assertEqual(len(wheel.actions), len(DEFAULT_WHEEL_ACTIONS))
            self.assertEqual(len(wheel._slice_ids), len(DEFAULT_WHEEL_ACTIONS))
            wheel._move_highlight(1)
            self.assertEqual(wheel._highlight_index, 0)
            wheel._move_highlight(1)
            self.assertEqual(wheel._highlight_index, 1)
            # Center is hub — no hit
            self.assertIsNone(wheel.hit_test(wheel.cx, wheel.cy))
            # North ring should hit first default action
            hit = wheel.hit_test(wheel.cx, wheel.cy - 100)
            self.assertEqual(hit, 0)
            wheel.dismiss()
            self.assertFalse(wheel.window.winfo_exists())
        finally:
            root.destroy()

    def test_transient_unknown_focus_does_not_dismiss_wheel(self):
        wheel = ActionWheel.__new__(ActionWheel)
        wheel._dead = False
        wheel.window = type(
            "WindowDouble",
            (),
            {
                "winfo_exists": lambda _self: True,
                "focus_get": lambda _self: None,
            },
        )()
        wheel.dismiss = lambda: self.fail("transient unknown focus dismissed wheel")

        wheel._dismiss_if_unfocused()


if __name__ == "__main__":
    unittest.main()
