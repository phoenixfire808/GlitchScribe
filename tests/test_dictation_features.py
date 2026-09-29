import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import dictation_features as dictation_features_module

from dictation_features import (
    CommandDebouncer,
    LiveCommitter,
    mask_profanity,
    parse_custom_replacements,
    parse_voice_command,
    prepare_transcript,
    contextualize_transcript,
    render_spoken_punctuation,
    spoken_replacement_is_exact,
    strip_automatic_punctuation,
    suppress_inferred_trailing_period,
    transcript_needs_trailing_separator,
)
from dictation_overlay import (
    DictationOverlay,
    StreamingBubble,
    anchored_bubble_position,
    bubble_font_spec,
    bubble_offset_for_position,
    caret_overlay_position,
    clamp_overlay_size,
    resolve_bubble_visual_style,
    clamp_overlay_position,
    compact_overlay_text,
    dragged_overlay_position,
    resized_overlay_size,
    overlay_position,
    overlay_position_in_work_area,
    show_window_no_activate,
)
from drew_local_voice.theme import build_palette


class DictationLanguageTests(unittest.TestCase):
    def test_inferred_final_period_is_suppressed_without_damaging_command_dots(self):
        self.assertEqual(
            suppress_inferred_trailing_period("git status.", "git status."),
            "git status",
        )
        self.assertEqual(
            suppress_inferred_trailing_period(
                "python script.py.",
                "python script.py.",
            ),
            "python script.py",
        )

    def test_explicit_spoken_period_is_preserved(self):
        self.assertEqual(
            suppress_inferred_trailing_period("That is complete.", "that is complete period"),
            "That is complete.",
        )

    def test_terminal_spoken_symbols_render_as_literal_characters(self):
        self.assertEqual(render_spoken_punctuation("slash"), "/")
        self.assertEqual(render_spoken_punctuation("back slash"), "\\")
        self.assertEqual(render_spoken_punctuation("pipe"), "|")
        self.assertEqual(render_spoken_punctuation("double ampersand"), "&&")
        self.assertEqual(render_spoken_punctuation("double dash version"), "--version")

    def test_custom_literal_replacements_are_case_insensitive_and_longest_first(self):
        replacements = parse_custom_replacements(
            "home path=~/; kube control=kubectl; kube=wrong"
        )

        self.assertEqual(
            render_spoken_punctuation(
                "Kube Control get pods", custom_replacements=replacements
            ),
            "kubectl get pods",
        )
        self.assertEqual(
            render_spoken_punctuation(
                "cd home path", custom_replacements=replacements
            ),
            "cd ~/",
        )

    def test_correction_map_adds_updates_and_removes_local_misrecognitions(self):
        upsert = getattr(dictation_features_module, "upsert_custom_replacement", None)
        remove = getattr(dictation_features_module, "remove_custom_replacement", None)
        self.assertIsNotNone(upsert)
        self.assertIsNotNone(remove)

        specification = upsert("", "  glitch   scribe ", "GlitchScribe")
        specification = upsert(specification, "her mees", "Hermes")
        specification = upsert(specification, "HER MEES", "Hermes Agent")

        self.assertEqual(
            parse_custom_replacements(specification),
            {"glitch scribe": "GlitchScribe", "her mees": "Hermes Agent"},
        )
        self.assertEqual(
            render_spoken_punctuation(
                "Open her mees in glitch scribe",
                custom_replacements=parse_custom_replacements(specification),
            ),
            "Open Hermes Agent in GlitchScribe",
        )
        self.assertEqual(
            parse_custom_replacements(remove(specification, "her mees")),
            {"glitch scribe": "GlitchScribe"},
        )

    def test_exact_literal_symbol_does_not_receive_automatic_trailing_space(self):
        self.assertTrue(spoken_replacement_is_exact("slash"))
        self.assertFalse(transcript_needs_trailing_separator("/", exact_literal=True))
        self.assertFalse(transcript_needs_trailing_separator("~/", exact_literal=False))
        self.assertTrue(transcript_needs_trailing_separator("normal words"))

    def test_contextualize_continuation_adds_one_space_and_lowercases_fragment(self):
        result = contextualize_transcript("This continues", "hello")

        self.assertEqual(result.prefix, " ")
        self.assertEqual(result.text, "this continues")

    def test_contextualize_continuation_respects_existing_space(self):
        result = contextualize_transcript("This continues", "hello ")

        self.assertEqual(result.prefix, "")
        self.assertEqual(result.text, "this continues")

    def test_contextualize_keeps_capital_after_sentence_boundary(self):
        result = contextualize_transcript("This starts", "finished.")

        self.assertEqual(result.prefix, " ")
        self.assertEqual(result.text, "This starts")

    def test_contextualize_protects_personal_vocabulary_and_pronoun_i(self):
        drew = contextualize_transcript(
            "Drew is speaking", "and", protected_terms=("Drew",)
        )
        pronoun = contextualize_transcript("I agree", "and")

        self.assertEqual(drew.text, "Drew is speaking")
        self.assertEqual(pronoun.text, "I agree")

    def test_contextualize_does_not_space_before_punctuation(self):
        result = contextualize_transcript(", then", "hello")

        self.assertEqual(result.prefix, "")
        self.assertEqual(result.text, ", then")

    def test_command_must_remain_stable_for_configured_wait(self):
        now = [10.0]
        debounce = CommandDebouncer(0.6, clock=lambda: now[0])
        self.assertIsNone(debounce.update("stop listening"))
        now[0] += 0.4
        self.assertIsNone(debounce.update("stop listening"))
        now[0] += 0.3
        self.assertEqual(debounce.update("stop listening").kind, "stop")

    def test_non_command_resets_command_debounce(self):
        now = [10.0]
        debounce = CommandDebouncer(0.2, clock=lambda: now[0])
        debounce.update("stop listening")
        now[0] += 1.0
        self.assertIsNone(debounce.update("stop listening to music"))
        self.assertIsNone(debounce.update("stop listening"))

    def test_prepares_transcript_using_punctuation_and_filter_settings(self):
        self.assertEqual(
            prepare_transcript(
                "Damn comma this works period",
                automatic_punctuation=True,
                profanity_terms={"damn"},
            ),
            "****, this works.",
        )

    def test_overlay_is_compact_like_a_dictation_flyout(self):
        self.assertLessEqual(DictationOverlay.WIDTH, 320)
        self.assertLessEqual(DictationOverlay.HEIGHT, 100)

    def test_overlay_text_keeps_only_a_short_recent_hint(self):
        compact = compact_overlay_text("one two three four five six seven", 18)

        self.assertLessEqual(len(compact), 18)
        self.assertTrue(compact.startswith("…"))
        self.assertTrue(compact.endswith("six seven"))

    def test_overlay_defaults_near_bottom_right_instead_of_center(self):
        self.assertEqual(
            overlay_position(1920, 1080, 620, 150, margin=48),
            (1252, 882),
        )

    def test_overlay_follows_a_non_primary_monitor_work_area(self):
        self.assertEqual(
            overlay_position_in_work_area(-1920, 0, 0, 1080, 620, 150, 48),
            (-668, 882),
        )

    def test_saved_overlay_position_is_clamped_inside_work_area(self):
        self.assertEqual(
            clamp_overlay_position(4000, -100, 0, 0, 1920, 1080, 300, 64),
            (1612, 8),
        )

    def test_overlay_drag_uses_pointer_delta_from_original_position(self):
        self.assertEqual(
            dragged_overlay_position(100, 200, 400, 500, 420, 535),
            (120, 235),
        )

    def test_main_overlay_resize_uses_pointer_delta_and_safe_bounds(self):
        self.assertEqual(
            resized_overlay_size(300, 64, 400, 500, 520, 580),
            (420, 144),
        )
        self.assertEqual(clamp_overlay_size(120, 30), (260, 92))
        self.assertEqual(clamp_overlay_size(1200, 600), (720, 240))

    def test_floating_control_has_professional_glitch_factor_identity(self):
        self.assertEqual(DictationOverlay.BRAND_NAME, "GLITCH FACTOR")
        self.assertIn("GLITCHSCRIBE", DictationOverlay.BRAND_SUBTITLE)
        self.assertIn("PRIVATE", DictationOverlay.BRAND_SUBTITLE)
        self.assertGreaterEqual(DictationOverlay.HEIGHT, 92)

    def test_streaming_bubble_is_separate_and_more_translucent_than_main_control(self):
        self.assertIsNot(StreamingBubble, DictationOverlay)
        self.assertGreaterEqual(StreamingBubble.WIDTH, 340)
        self.assertLessEqual(StreamingBubble.HEIGHT, 72)
        self.assertLess(StreamingBubble.OPACITY, DictationOverlay.OPACITY)

    def test_streaming_bubble_appears_immediately_when_recording_begins(self):
        bubble = StreamingBubble.__new__(StreamingBubble)
        bubble.width = 420
        bubble.height = 68
        bubble.position = None
        bubble._work_area = None
        bubble._anchor = None
        bubble.canvas = Mock()
        bubble.window = Mock()
        bubble._state_text = 1
        bubble._preview_text = 2
        bubble._work_area_for_target = Mock(return_value=(0, 0, 1920, 1080))
        bubble._apply_no_activate_style = Mock()
        bubble._native_hwnd = Mock(return_value=77)

        with patch("dictation_overlay.os.name", "nt"), patch(
            "dictation_overlay.show_window_no_activate"
        ) as show:
            bubble.begin(41, caret=(500, 500))

        self.assertTrue(bubble._visible)
        bubble.canvas.itemconfigure.assert_any_call(1, text="LISTENING")
        bubble.canvas.itemconfigure.assert_any_call(2, text="Listening…")
        show.assert_called_once()

    def test_streaming_bubble_reconfigures_size_text_and_visual_design(self):
        parameters = inspect.signature(StreamingBubble.configure).parameters
        for name in (
            "width", "height", "font_size", "max_characters", "opacity",
            "font_family", "font_weight", "visual_style", "text_color_mode",
            "corner_radius", "border_width",
        ):
            self.assertIn(name, parameters)

        bubble = StreamingBubble.__new__(StreamingBubble)
        bubble.width = 420
        bubble.height = 68
        bubble.font_size = 10
        bubble.max_characters = 120
        bubble.opacity = 0.82
        bubble.font_family = "Segoe UI"
        bubble.font_weight = "normal"
        bubble.visual_style = "glitch_glass"
        bubble.text_color_mode = "theme"
        bubble.corner_radius = 20
        bubble.border_width = 1
        bubble.palette = build_palette("dark", "violet")
        bubble.canvas = Mock()
        bubble.window = Mock()
        bubble._shape = 1
        bubble._state_text = 3
        bubble._preview_text = 2

        bubble.configure(
            width=540,
            height=104,
            font_size=16,
            max_characters=260,
            opacity=0.90,
            font_family="Cascadia Code",
            font_weight="bold",
            visual_style="neon_edge",
            text_color_mode="accent",
            corner_radius=28,
            border_width=3,
        )

        self.assertEqual(
            (
                bubble.width, bubble.height, bubble.font_size,
                bubble.max_characters, bubble.font_family, bubble.font_weight,
                bubble.visual_style, bubble.text_color_mode,
                bubble.corner_radius, bubble.border_width,
            ),
            (540, 104, 16, 260, "Cascadia Code", "bold", "neon_edge", "accent", 28, 3),
        )
        self.assertAlmostEqual(bubble.opacity, 0.90)
        bubble.canvas.configure.assert_called_once_with(width=540, height=104)
        self.assertIn(
            ((2,), {"width": 504, "font": ("Cascadia Code", 16, "bold")}),
            bubble.canvas.itemconfigure.call_args_list,
        )

    def test_bubble_visual_presets_and_font_specs_are_deterministic(self):
        palette = build_palette("dark", "cyan")
        neon = resolve_bubble_visual_style(
            palette,
            "neon_edge",
            "accent",
            border_width=3,
        )
        contrast = resolve_bubble_visual_style(
            palette,
            "high_contrast",
            "bright",
            border_width=2,
        )

        self.assertEqual(neon.border, palette.accent)
        self.assertEqual(neon.text, palette.accent_soft)
        self.assertEqual(neon.border_width, 3)
        self.assertEqual((contrast.background, contrast.text), ("#000000", "#FFFFFF"))
        self.assertEqual(bubble_font_spec("Segoe UI", 14, "semibold"), ("Segoe UI Semibold", 14))
        self.assertEqual(bubble_font_spec("Consolas", 14, "bold"), ("Consolas", 14, "bold"))

    def test_streaming_bubble_uses_configured_visible_text_capacity(self):
        bubble = StreamingBubble.__new__(StreamingBubble)
        bubble.position = (560, 420)
        bubble._visible = True
        bubble._state_text = 1
        bubble._preview_text = 2
        bubble.max_characters = 40
        bubble.canvas = Mock()

        bubble.update("one two three four five six seven eight nine ten eleven twelve")

        preview_calls = [
            call
            for call in bubble.canvas.itemconfigure.call_args_list
            if call.args == (2,)
        ]
        rendered = preview_calls[-1].kwargs["text"]
        self.assertLessEqual(len(rendered), 40)
        self.assertTrue(rendered.endswith("eleven twelve"))

    def test_main_control_ignores_caret_coordinates_during_recording(self):
        overlay = DictationOverlay.__new__(DictationOverlay)
        overlay.state_var = SimpleNamespace(set=lambda _value: None)
        overlay.text_var = SimpleNamespace(set=lambda _value: None)
        overlay.status_dot = SimpleNamespace(configure=lambda **_kwargs: None)
        overlay.action_control = SimpleNamespace(configure=lambda **_kwargs: None)
        overlay.palette = build_palette("dark", "violet")
        overlay.set_meter = lambda _value: None
        overlay._show_at_saved_or_default = Mock()

        overlay.show(41, caret=(600, 500), caret_gap=16)

        overlay._show_at_saved_or_default.assert_called_once_with(41)

    def test_streaming_text_update_keeps_the_session_bubble_anchor_fixed(self):
        bubble = StreamingBubble.__new__(StreamingBubble)
        bubble.position = (560, 420)
        bubble._visible = True
        bubble._state_text = 1
        bubble._preview_text = 2
        bubble.canvas = SimpleNamespace(itemconfigure=lambda *_args, **_kwargs: None)

        bubble.update("first provisional phrase")
        bubble.update("a longer provisional phrase after text insertion")

        self.assertEqual(bubble.position, (560, 420))

    def test_bubble_uses_adjustable_offset_from_session_caret_or_mouse_anchor(self):
        self.assertEqual(
            anchored_bubble_position(
                600, 500, 0, 0, 1920, 1080, 420, 68, offset_x=18, offset_y=-88
            ),
            (618, 412),
        )
        self.assertEqual(
            anchored_bubble_position(
                1900, 1070, 0, 0, 1920, 1080, 420, 68, offset_x=18, offset_y=12
            ),
            (1492, 1004),
        )

    def test_dragged_bubble_position_persists_as_anchor_relative_offset(self):
        self.assertEqual(
            bubble_offset_for_position(710, 360, anchor_x=600, anchor_y=500),
            (110, -140),
        )

    def test_dragging_visible_bubble_updates_persisted_anchor_offset(self):
        saved = Mock()
        geometry = Mock()
        bubble = StreamingBubble.__new__(StreamingBubble)
        bubble.width = 420
        bubble.height = 68
        bubble.position = (618, 412)
        bubble._anchor = (600, 500)
        bubble._work_area = (0, 0, 1920, 1080)
        bubble._drag_origin = (618, 412, 100, 100)
        bubble.on_offset_changed = saved
        bubble.window = SimpleNamespace(geometry=geometry)

        bubble._drag_end(SimpleNamespace(x_root=120, y_root=90))

        self.assertEqual(bubble.position, (638, 402))
        saved.assert_called_once_with(38, -98)

    def test_caret_preview_prefers_above_and_clamps_to_monitor(self):
        self.assertEqual(
            caret_overlay_position(
                600, 500, 0, 0, 1920, 1080, 300, 64, gap=16
            ),
            (560, 420),
        )
        self.assertEqual(
            caret_overlay_position(
                20, 30, 0, 0, 1920, 1080, 300, 64, gap=16
            ),
            (8, 46),
        )

    def test_overlay_native_show_never_activates_target_window(self):
        calls = []

        class User32:
            def ShowWindow(self, hwnd, command):
                calls.append(("show", hwnd, command))
                return 1

            def SetWindowPos(self, hwnd, order, x, y, width, height, flags):
                calls.append(
                    ("position", hwnd, order, x, y, width, height, flags)
                )
                return 1

        show_window_no_activate(41, 100, 200, 300, 64, user32=User32())

        self.assertEqual(calls[0], ("show", 41, 4))  # SW_SHOWNOACTIVATE
        self.assertEqual(calls[1][0], "position")
        self.assertTrue(calls[1][-1] & 0x0010)  # SWP_NOACTIVATE
        self.assertTrue(calls[1][-1] & 0x0040)  # SWP_SHOWWINDOW

    def test_renders_common_punctuation_and_paragraph_commands(self):
        rendered = render_spoken_punctuation(
            "Hello comma how are you question mark new paragraph I am fine exclamation point"
        )
        self.assertEqual(rendered, "Hello, how are you?\n\nI am fine!")

    def test_renders_symbols_and_apostrophe_s_without_extra_spaces(self):
        rendered = render_spoken_punctuation(
            "Drew apostrophe s email is drew at sign example period com"
        )
        self.assertEqual(rendered, "Drew's email is drew@example.com")

    def test_parses_core_voice_typing_commands_only_when_exact(self):
        self.assertEqual(parse_voice_command("Stop listening").kind, "stop")
        self.assertEqual(parse_voice_command("scratch that").kind, "delete_that")
        self.assertEqual(parse_voice_command("press enter").kind, "key")
        self.assertEqual(parse_voice_command("press enter").value, "enter")
        self.assertEqual(parse_voice_command("revert").kind, "undo")
        self.assertIsNone(parse_voice_command("please stop listening to the song"))

    def test_literal_and_text_shaping_prefixes(self):
        literal = parse_voice_command("type stop listening")
        caps = parse_voice_command("caps drew local voice")
        no_space = parse_voice_command("no space drew at example dot com")
        self.assertEqual((literal.kind, literal.value), ("insert", "stop listening"))
        self.assertEqual((caps.kind, caps.value), ("insert", "Drew Local Voice"))
        self.assertEqual((no_space.kind, no_space.value), ("insert", "drewatexampledotcom"))

    def test_explicit_identifier_shaping_commands_are_deterministic(self):
        expected = {
            "snake case HTTP response code": "http_response_code",
            "camel case HTTP response code": "httpResponseCode",
            "pascal case HTTP response code": "HttpResponseCode",
            "kebab case HTTP response code": "http-response-code",
            "constant case HTTP response code": "HTTP_RESPONSE_CODE",
        }
        for spoken, rendered in expected.items():
            with self.subTest(spoken=spoken):
                command = parse_voice_command(spoken)
                self.assertIsNotNone(command)
                self.assertEqual((command.kind, command.value), ("insert", rendered))
        self.assertIsNone(parse_voice_command("snake case"))

    def test_snippet_and_prompt_compiler_commands_preserve_payload_text(self):
        snippet = parse_voice_command("snippet Python Main")
        compiler = parse_voice_command(
            "compile prompt Add SHA256 validation to API responses"
        )

        self.assertIsNotNone(snippet)
        self.assertEqual((snippet.kind, snippet.value), ("snippet", "Python Main"))
        self.assertIsNotNone(compiler)
        self.assertEqual(
            (compiler.kind, compiler.value),
            ("compile_prompt", "Add SHA256 validation to API responses"),
        )
        self.assertIsNone(parse_voice_command("snippet"))
        self.assertIsNone(parse_voice_command("compile prompt"))

    def test_capture_selection_is_an_exact_read_only_command(self):
        command = parse_voice_command("Capture selection.")
        self.assertIsNotNone(command)
        self.assertEqual((command.kind, command.value), ("capture_selection", None))
        self.assertIsNone(
            parse_voice_command("capture selection after explaining the change")
        )

    def test_explain_selection_is_an_exact_local_ai_command(self):
        command = parse_voice_command("Explain selection.")
        self.assertIsNotNone(command)
        self.assertEqual((command.kind, command.value), ("explain_selection", None))
        self.assertIsNone(parse_voice_command("please explain selection when finished"))

    def test_edit_selection_preserves_the_bounded_spoken_instruction(self):
        command = parse_voice_command(
            "edit selection Make this shorter and keep OAuth unchanged"
        )
        self.assertIsNotNone(command)
        self.assertEqual(
            (command.kind, command.value),
            ("edit_selection", "Make this shorter and keep OAuth unchanged"),
        )
        self.assertIsNone(parse_voice_command("edit selection"))
        self.assertIsNone(parse_voice_command("please edit selection make it shorter"))

    def test_automatic_punctuation_can_be_removed_without_breaking_apostrophes(self):
        self.assertEqual(
            strip_automatic_punctuation("Drew's ready, right? Yes."),
            "Drew's ready right Yes",
        )

    def test_local_profanity_filter_is_optional_and_case_insensitive(self):
        self.assertEqual(
            mask_profanity("Damn, that damn thing", {"damn"}),
            "****, that **** thing",
        )
        self.assertEqual(mask_profanity("Damn", set()), "Damn")


class LiveCommitterTests(unittest.TestCase):
    def test_timestamp_stable_text_commits_on_first_delivery(self):
        committer = LiveCommitter(holdback_words=0)

        stable = committer.accept_stable("This is already stable")
        final = committer.finalize("This is already stable and final")

        self.assertFalse(stable.conflict)
        self.assertEqual(stable.text_to_insert, "This is already stable")
        self.assertEqual(final.text_to_insert, " and final")

    def test_commits_only_stable_prefix_and_holds_three_trailing_words(self):
        live = LiveCommitter(holdback_words=3)
        self.assertEqual(live.update("one two three four five").text_to_insert, "")

        decision = live.update("one two three four five six")

        self.assertEqual(decision.text_to_insert, "one two")
        self.assertEqual(decision.committed_text, "one two")

    def test_finalization_appends_only_uncommitted_suffix(self):
        live = LiveCommitter(holdback_words=2)
        live.update("hello from drew local voice")
        partial = live.update("hello from drew local voice today")
        self.assertEqual(partial.text_to_insert, "hello from drew")

        final = live.finalize("hello from drew local voice today")

        self.assertEqual(final.text_to_insert, " local voice today")
        self.assertFalse(final.conflict)
        self.assertEqual(final.committed_text, "hello from drew local voice today")

    def test_conflicting_final_never_retracts_or_guesses(self):
        live = LiveCommitter(holdback_words=1)
        live.update("the quick brown fox")
        live.update("the quick brown fox jumps")
        self.assertEqual(live.committed_text, "the quick brown")

        final = live.finalize("the quick blue fox jumps")

        self.assertTrue(final.conflict)
        self.assertEqual(final.text_to_insert, "")
        self.assertEqual(final.committed_text, "the quick brown")

    def test_finalization_appends_tail_after_harmless_formatting_revision(self):
        live = LiveCommitter(holdback_words=0)
        committed = live.accept_stable("okay this is working")
        self.assertEqual(committed.text_to_insert, "okay this is working")

        final = live.finalize("Okay, this is working now")

        self.assertFalse(final.conflict)
        self.assertEqual(final.text_to_insert, " now")
        self.assertEqual(final.committed_text, "okay this is working now")

    def test_finalization_preserves_new_punctuation_before_uncommitted_tail(self):
        live = LiveCommitter(holdback_words=0)
        live.accept_stable("hello world")

        final = live.finalize("Hello world, again")

        self.assertFalse(final.conflict)
        self.assertEqual(final.text_to_insert, ", again")

    def test_short_command_phrase_never_streams_before_final(self):
        live = LiveCommitter(holdback_words=3)
        live.update("delete that")
        decision = live.update("delete that")
        self.assertEqual(decision.text_to_insert, "")

    def test_punctuation_delta_does_not_insert_a_leading_space(self):
        live = LiveCommitter(holdback_words=0)
        live.update("Hello")
        first = live.update("Hello comma")
        second = live.update("Hello comma world")
        third = live.finalize("Hello comma world")

        self.assertEqual(first.text_to_insert, "Hello")
        self.assertEqual(second.text_to_insert, ",")
        self.assertEqual(third.text_to_insert, " world")


if __name__ == "__main__":
    unittest.main()
