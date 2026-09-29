from __future__ import annotations

import tempfile
import tkinter as tk
import unittest
from tkinter import ttk

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from drew_local_voice.commands import CommandDefinition, CommandRegistry
from drew_local_voice.plugins import ExtensionRegistry, PluginManager
from drew_local_voice.profiles import DictationProfile, ProfileRegistry, built_in_profiles
from drew_local_voice.performance import (
    PerformancePresetRegistry,
    built_in_performance_presets,
)
from drew_local_voice.models import (
    configured_model_choices,
    estimated_model_vram_mb,
    validate_model_identifier,
)
from drew_local_voice.settings import (
    EffectRecipe,
    SettingKind,
    SettingSpec,
    SettingsRegistry,
    built_in_settings,
    built_in_effect_recipes,
)
from drew_local_voice.settings_ui import (
    SECTION_ORDER,
    SchemaSettingsDialog,
    settings_tab_for_section,
    slider_step_for_spec,
    wheel_scroll_units,
)
from drew_local_voice import settings_ui as settings_ui_module
from drew_local_voice import theme as theme_module
from drew_local_voice.theme import build_palette
from dictation_features import parse_custom_replacements
from local_voice import (
    AppConfig,
    ControlLabRuntimeLayer,
    LocalVoiceApp,
    build_control_lab_view_state,
)


class ModularPlatformTests(unittest.TestCase):
    class _Variable:
        def __init__(self, value=""):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    class _NativeWindow:
        def __init__(self):
            self.updated = False

        def update_idletasks(self):
            self.updated = True

        def winfo_id(self):
            return 101

    class _User32:
        @staticmethod
        def GetParent(_hwnd):
            return 202

    class _DwmApi:
        def __init__(self):
            self.calls = []

        def DwmSetWindowAttribute(self, hwnd, attribute, value, size):
            self.calls.append((hwnd, attribute, value._obj.value, size))
            return 1 if attribute == 20 else 0

    def test_control_lab_state_matrix_keeps_primary_workflow_truthful(self):
        effective = ControlLabRuntimeLayer(
            "distil-large-v3.5", "cuda:0", "int8_float16"
        )
        saved = ControlLabRuntimeLayer("distil-large-v3.5", "automatic", "int8")
        staged = ControlLabRuntimeLayer("distil-large-v3.5", "cpu", "int8")

        expected = (
            ("ready", "Ready", "START DICTATION", True),
            ("recording", "Recording", "STOP & TRANSCRIBE", True),
            ("processing", "Processing", "PROCESSING...", False),
            ("transcript", "Transcript ready", "START NEW DICTATION", True),
            ("blocked", "Blocked", "START DICTATION", False),
        )
        for scenario, phase, action, enabled in expected:
            with self.subTest(scenario=scenario):
                state = build_control_lab_view_state(
                    ready=scenario in {"ready", "transcript"},
                    recording=scenario == "recording",
                    processing=scenario == "processing",
                    transcript_ready=scenario == "transcript",
                    blocked_reason=(
                        "Microphone unavailable" if scenario == "blocked" else ""
                    ),
                    effective=effective,
                    saved=saved,
                    staged=staged,
                )
                self.assertEqual(state.phase, phase)
                self.assertEqual(state.primary_action, action)
                self.assertEqual(state.primary_enabled, enabled)
                self.assertTrue(state.runtime_drift)
                self.assertTrue(state.detail)
                self.assertTrue(state.next_action)

    def test_control_lab_runtime_layers_report_sync_without_false_drift(self):
        layer = ControlLabRuntimeLayer(
            "distil-large-v3.5", "cuda:0", "int8_float16"
        )
        state = build_control_lab_view_state(
            ready=True,
            effective=layer,
            saved=layer,
            staged=layer,
        )
        self.assertEqual(
            layer.summary(),
            "distil-large-v3.5 / cuda:0 / int8_float16",
        )
        self.assertFalse(state.runtime_drift)

    def test_control_lab_unchanged_runtime_staging_is_not_pending(self):
        app = object.__new__(LocalVoiceApp)
        app.config = AppConfig(
            model_name="distil-large-v3.5",
            gpu_device_choice="automatic",
            compute_type="int8",
        )
        app._staged_runtime_values = {
            "recognition.model": "distil-large-v3.5",
            "recognition.device": "automatic",
            "recognition.compute_type": "int8",
        }

        self.assertIsNone(app._control_lab_staged_layer())

        app._staged_runtime_values["recognition.device"] = "cpu"
        staged = app._control_lab_staged_layer()
        self.assertIsNotNone(staged)
        self.assertEqual(staged.device, "cpu")

    def test_control_lab_dialog_lifecycle_distinguishes_open_clean_from_dirty(self):
        app = object.__new__(LocalVoiceApp)
        app.config = AppConfig(
            model_name="distil-large-v3.5",
            gpu_device_choice="automatic",
            compute_type="int8",
        )
        app._advanced_settings_open = False
        app._staged_runtime_values = None
        app._refresh_control_lab_state = Mock()
        saved = app._control_lab_saved_layer()
        values = {
            "recognition.model": saved.model,
            "recognition.device": saved.device,
            "recognition.compute_type": saved.compute,
        }

        app._on_advanced_settings_staged(values)
        self.assertTrue(app._advanced_settings_open)
        self.assertIsNone(app._staged_runtime_values)

        app._on_advanced_settings_staged(
            {**values, "recognition.device": "cpu"}
        )
        self.assertTrue(app._advanced_settings_open)
        self.assertIsNotNone(app._staged_runtime_values)

        app._on_advanced_settings_staged(None)
        self.assertFalse(app._advanced_settings_open)
        self.assertIsNone(app._staged_runtime_values)

    def test_every_bounded_numeric_setting_has_a_slider_step(self):
        numeric_specs = tuple(
            spec
            for spec in built_in_settings()
            if spec.kind is SettingKind.NUMBER
            and spec.minimum is not None
            and spec.maximum is not None
        )

        self.assertTrue(numeric_specs)
        for spec in numeric_specs:
            with self.subTest(key=spec.key):
                step = slider_step_for_spec(spec)
                self.assertIsNotNone(step)
                self.assertGreater(step, 0)
                self.assertLessEqual(step, spec.maximum - spec.minimum)

    def test_settings_sections_are_split_into_compact_tabs(self):
        expected = {
            "General": "General",
            "Performance Lab": "General",
            "Recognition": "Recognition",
            "Local AI": "Recognition",
            "Streaming": "Streaming",
            "Audio": "Streaming",
            "Typing": "Dictation",
            "Endpointing": "Dictation",
            "Finalization": "Dictation",
            "Commands": "Commands",
            "Corrections": "Commands",
            "Interface": "Appearance",
            "Speech Bubble": "Appearance",
            "Privacy": "System",
            "Plugins": "System",
            "Compatibility": "System",
        }

        self.assertEqual(
            {section: settings_tab_for_section(section) for section in expected},
            expected,
        )

    def test_effect_recipes_stage_coordinated_settings_for_named_goals(self):
        settings = SettingsRegistry(built_in_settings())
        recipes = {recipe.key: recipe for recipe in built_in_effect_recipes()}

        expected_recipes = {
            "streamer_large_text": "Streamer Large Text",
            "low_glare_chat": "Low-Glare Chat",
            "neon_high_contrast": "Neon High-Contrast",
            "compact_corner_bubble": "Compact Corner Bubble",
            "rapid_preview_long_sample": "Rapid Preview / Long Sample",
        }
        for key in (
            "protect_every_final_word",
            "fast_bubble_preview",
            "terminal_safe_dictation",
            "bubble_near_cursor",
        ):
            self.assertIn(key, recipes)
        self.assertEqual(
            {key: recipes[key].title for key in expected_recipes},
            expected_recipes,
        )
        for recipe in recipes.values():
            self.assertIsInstance(recipe, EffectRecipe)
            self.assertTrue(recipe.outcome)
            self.assertGreaterEqual(len(recipe.changes), 2)
            for key, value in recipe.changes:
                self.assertEqual(settings.validate(key, value), value)

        rapid_preview = recipes["rapid_preview_long_sample"]
        self.assertIn("long synthetic sample", rapid_preview.outcome.casefold())
        self.assertFalse(
            any("sample" in key.casefold() for key, _value in rapid_preview.changes)
        )

    def test_native_titlebar_theme_uses_dark_caption_with_windows_10_fallback(self):
        apply_titlebar = getattr(theme_module, "apply_windows_titlebar_theme", None)
        self.assertIsNotNone(apply_titlebar)
        window = self._NativeWindow()
        user32 = self._User32()
        dwmapi = self._DwmApi()

        applied = apply_titlebar(
            window,
            build_palette("dark", "violet"),
            user32=user32,
            dwmapi=dwmapi,
        )

        self.assertTrue(applied)
        self.assertTrue(window.updated)
        attributes = [call[1] for call in dwmapi.calls]
        self.assertEqual(attributes[:2], [20, 19])
        self.assertIn(35, attributes)  # caption color
        self.assertIn(36, attributes)  # caption text color
        self.assertIn(34, attributes)  # border color
        self.assertEqual(dwmapi.calls[0][0], 202)
        self.assertEqual(dwmapi.calls[0][2], 1)

    def test_effect_recipe_stages_values_without_saving_automatically(self):
        recipe = built_in_effect_recipes()[0]
        variables = {key: Mock() for key, _value in recipe.changes}
        outcome = Mock()
        dialog = SchemaSettingsDialog.__new__(SchemaSettingsDialog)
        dialog.recipes_by_title = {recipe.title: recipe}
        dialog.recipe_var = SimpleNamespace(get=lambda: recipe.title)
        dialog.recipe_outcome_var = outcome
        dialog.variables = variables

        dialog.stage_selected_recipe()

        for key, value in recipe.changes:
            variables[key].set.assert_called_once_with(value)
        self.assertIn("STAGED", outcome.set.call_args.args[0])
        self.assertIn("Save Settings", outcome.set.call_args.args[0])

    def test_interface_settings_are_immediately_reachable(self):
        self.assertEqual(SECTION_ORDER[:2], ("General", "Interface"))

    def test_mouse_wheel_and_trackpad_deltas_produce_vertical_scroll_units(self):
        self.assertEqual(wheel_scroll_units(120), -1)
        self.assertEqual(wheel_scroll_units(-120), 1)
        self.assertEqual(wheel_scroll_units(240), -2)
        self.assertEqual(wheel_scroll_units(1), -1)
        self.assertEqual(wheel_scroll_units(0), 0)

    def test_slider_values_snap_to_steps_and_stay_inside_bounds(self):
        snap = getattr(settings_ui_module, "snap_slider_value", None)
        self.assertIsNotNone(snap)
        self.assertEqual(snap(417, 280, 900, 10), 420)
        self.assertEqual(snap(51, 56, 300, 4), 56)
        self.assertEqual(snap(1004, 40, 1000, 10), 1000)
        self.assertAlmostEqual(snap(0.826, 0.35, 0.98, 0.01), 0.83)

    def test_settings_dialog_routes_wheel_page_and_edge_navigation_to_selected_tab(self):
        dialog = SchemaSettingsDialog.__new__(SchemaSettingsDialog)
        canvas = Mock()
        dialog.notebook = SimpleNamespace(select=lambda: "selected-tab")
        dialog.canvases = {"selected-tab": canvas}

        self.assertEqual(
            dialog._on_mousewheel(SimpleNamespace(delta=-120)),
            "break",
        )
        dialog._on_button_scroll(SimpleNamespace(), -1)
        dialog._scroll_pages(SimpleNamespace(), 1)
        dialog._scroll_edge(SimpleNamespace(), 1.0)

        self.assertEqual(
            canvas.yview_scroll.call_args_list,
            [
                call(1, "units"),
                call(-1, "units"),
                call(1, "pages"),
            ],
        )
        canvas.yview_moveto.assert_called_once_with(1.0)

    def test_embedded_settings_are_one_sectioned_page_without_notebook_or_close(self):
        root = tk.Tk()
        root.withdraw()
        try:
            host = ttk.Frame(root)
            registry = SettingsRegistry(
                (
                    SettingSpec(
                        key="general.hotkey",
                        label="Hotkey",
                        section="General",
                        kind=SettingKind.CHOICE,
                        default="Ctrl+Win",
                        choices=("Ctrl+Win", "Alt+Win"),
                    ),
                    SettingSpec(
                        key="typing.smart_continuation",
                        label="Smart continuation",
                        section="Typing",
                        kind=SettingKind.BOOLEAN,
                        default=True,
                    ),
                )
            )
            panel = SchemaSettingsDialog(
                host,
                registry,
                registry.defaults(),
                lambda _values: None,
                embedded=True,
                excluded_keys=("general.hotkey",),
            )
            root.update_idletasks()
            descendants = []
            pending = list(host.winfo_children())
            while pending:
                widget = pending.pop()
                descendants.append(widget)
                pending.extend(widget.winfo_children())
            button_text = {
                widget.cget("text")
                for widget in descendants
                if isinstance(widget, (tk.Button, ttk.Button))
            }

            self.assertIs(panel.window, host)
            self.assertIsNone(panel.notebook)
            self.assertNotIn("general.hotkey", panel.variables)
            self.assertIn("typing.smart_continuation", panel.variables)
            self.assertFalse(any(isinstance(widget, ttk.Notebook) for widget in descendants))
            self.assertIn("SAVE SETTINGS", button_text)
            self.assertNotIn("CLOSE", button_text)
        finally:
            root.destroy()

    def test_embedded_save_refreshes_excluded_values_from_canonical_provider(self):
        root = tk.Tk()
        root.withdraw()
        saved = []
        try:
            host = ttk.Frame(root)
            registry = SettingsRegistry(
                (
                    SettingSpec(
                        key="general.hotkey",
                        label="Hotkey",
                        section="General",
                        kind=SettingKind.CHOICE,
                        default="old",
                        choices=("old", "new"),
                    ),
                    SettingSpec(
                        key="typing.smart_continuation",
                        label="Smart continuation",
                        section="Typing",
                        kind=SettingKind.BOOLEAN,
                        default=True,
                    ),
                )
            )
            panel = SchemaSettingsDialog(
                host,
                registry,
                registry.defaults(),
                saved.append,
                embedded=True,
                excluded_keys=("general.hotkey",),
                values_provider=lambda: {
                    "general.hotkey": "new",
                    "typing.smart_continuation": True,
                },
            )
            panel.variables["typing.smart_continuation"].set(False)
            with patch("drew_local_voice.settings_ui.messagebox.showinfo"):
                panel.save()

            self.assertEqual(saved[-1]["general.hotkey"], "new")
            self.assertFalse(saved[-1]["typing.smart_continuation"])
        finally:
            root.destroy()

    def test_theme_palettes_support_dark_light_system_resolution_and_accents(self):
        dark = build_palette("dark", "violet")
        light = build_palette("light", "violet")
        system_dark = build_palette("system", "cyan", system_uses_light=False)
        cyan_dark = build_palette("dark", "cyan")

        self.assertNotEqual(dark.window, light.window)
        self.assertNotEqual(dark.text, light.text)
        self.assertEqual(system_dark.mode, "dark")
        self.assertEqual(system_dark.accent, cyan_dark.accent)
        self.assertNotEqual(dark.accent, cyan_dark.accent)
    def test_custom_model_catalog_accepts_ids_and_local_paths_without_duplicates(self):
        choices = configured_model_choices(
            "Systran/faster-whisper-large-v3, base.en, C:/models/my-ct2"
        )

        self.assertIn("Systran/faster-whisper-large-v3", choices)
        self.assertIn("C:/models/my-ct2", choices)
        self.assertEqual(choices.count("base.en"), 1)
        self.assertEqual(
            validate_model_identifier("  distil-whisper/distil-large-v3.5-ct2  "),
            "distil-whisper/distil-large-v3.5-ct2",
        )
        with self.assertRaises(ValueError):
            validate_model_identifier("bad\nmodel")

    def test_model_vram_estimate_is_conservative_for_unrecognized_large_models(self):
        self.assertGreaterEqual(
            estimated_model_vram_mb("org/custom-large-v3-ct2"),
            estimated_model_vram_mb("small.en"),
        )

    def test_builtin_model_vram_estimates_match_lean_runtime_footprints(self):
        self.assertEqual(estimated_model_vram_mb("tiny.en"), 512)
        self.assertEqual(estimated_model_vram_mb("base.en"), 1024)
        self.assertEqual(estimated_model_vram_mb("small.en"), 2048)
        self.assertEqual(estimated_model_vram_mb("distil-large-v3.5"), 3072)

    def test_built_in_schema_exposes_a_large_runtime_tuning_surface(self):
        settings = {spec.key: spec for spec in built_in_settings()}

        self.assertGreaterEqual(len(settings), 60)
        for key in (
            "performance.preset",
            "streaming.minimum_new_audio_seconds",
            "streaming.timestamp_tolerance_seconds",
            "endpoint.minimum_speech_seconds",
            "finalization.decoding_mode",
            "audio.callback_block_ms",
        ):
            self.assertIn(key, settings)

    def test_every_built_in_setting_has_explanation_and_practical_guidance(self):
        settings = built_in_settings()

        missing_descriptions = [spec.key for spec in settings if not spec.description.strip()]
        missing_guidance = [spec.key for spec in settings if not spec.guidance.strip()]

        self.assertEqual(missing_descriptions, [])
        self.assertEqual(missing_guidance, [])
        self.assertIn("final word", {
            spec.key: spec.guidance.casefold() for spec in settings
        }["endpoint.silence_seconds"])

    def test_settings_schema_includes_custom_models_and_caret_preview_controls(self):
        settings = {spec.key: spec for spec in built_in_settings()}

        self.assertIn("recognition.additional_models", settings)
        self.assertIn("ui.overlay_width", settings)
        self.assertIn("ui.overlay_height", settings)
        self.assertIn("ui.theme", settings)
        self.assertIn("ui.accent_color", settings)
        self.assertIn("ui.streaming_bubble", settings)
        self.assertIn("ui.bubble_width", settings)
        self.assertIn("ui.bubble_font_family", settings)
        self.assertIn("ui.bubble_font_weight", settings)
        self.assertIn("ui.bubble_visual_style", settings)
        self.assertIn("ui.bubble_text_color_mode", settings)
        self.assertIn("ui.bubble_corner_radius", settings)
        self.assertIn("ui.bubble_border_width", settings)
        self.assertIn("ui.bubble_opacity", settings)
        self.assertIn("ui.preview_near_caret", settings)
        self.assertIn("ui.preview_offset_pixels", settings)
        self.assertIn("does not chase", settings["ui.preview_near_caret"].guidance)
        self.assertEqual(
            settings["recognition.device"].choices,
            ("automatic", "cuda:0", "cuda:1", "cpu"),
        )
        self.assertEqual(settings["recognition.vram_reserve_mb"].default, 512)
        self.assertEqual(settings["recognition.vram_reserve_mb"].minimum, 0)
        self.assertEqual(settings["recognition.vram_reserve_mb"].maximum, 8192)
        self.assertEqual(settings["recognition.vram_reserve_mb"].step, 256)
        self.assertTrue(settings["recognition.vram_reserve_mb"].restart_required)

    def test_speech_bubble_size_text_and_opacity_settings_use_sliders(self):
        settings = {spec.key: spec for spec in built_in_settings()}
        expected_steps = {
            "ui.bubble_width": 10.0,
            "ui.bubble_height": 4.0,
            "ui.bubble_font_size": 1.0,
            "ui.bubble_max_characters": 10.0,
            "ui.bubble_opacity": 0.01,
            "ui.bubble_corner_radius": 1.0,
            "ui.bubble_border_width": 1.0,
        }

        for key, step in expected_steps.items():
            spec = settings.get(key)
            self.assertIsNotNone(spec, key)
            self.assertEqual(getattr(spec, "presentation", None), "slider")
            self.assertEqual(getattr(spec, "step", None), step)

    def test_custom_replacements_use_a_dedicated_correction_editor(self):
        settings = {spec.key: spec for spec in built_in_settings()}
        correction = settings["commands.custom_replacements"]

        self.assertEqual(correction.section, "Corrections")
        self.assertEqual(correction.presentation, "correction_editor")
        self.assertIn("wrong", correction.description.casefold())
        self.assertIn("intended", correction.guidance.casefold())

    def test_correction_editor_stages_mapping_and_promotes_intended_vocabulary(self):
        add = getattr(SchemaSettingsDialog, "add_or_update_correction", None)
        self.assertIsNotNone(add)
        dialog = SchemaSettingsDialog.__new__(SchemaSettingsDialog)
        dialog.variables = {
            "commands.custom_replacements": self._Variable(""),
            "recognition.custom_vocabulary": self._Variable("Drew"),
        }
        dialog.correction_heard_var = self._Variable("her mees")
        dialog.correction_intended_var = self._Variable("Hermes Agent")
        dialog.correction_status_var = self._Variable("")
        dialog._refresh_correction_tree = Mock()

        add(dialog)

        self.assertEqual(
            parse_custom_replacements(
                dialog.variables["commands.custom_replacements"].get()
            ),
            {"her mees": "Hermes Agent"},
        )
        self.assertEqual(
            dialog.variables["recognition.custom_vocabulary"].get(),
            "Drew, Hermes Agent",
        )
        self.assertEqual(dialog.correction_heard_var.get(), "")
        self.assertEqual(dialog.correction_intended_var.get(), "")
        self.assertIn("Save Settings", dialog.correction_status_var.get())

    def test_performance_presets_expose_safe_and_experimental_latency_profiles(self):
        registry = PerformancePresetRegistry(built_in_performance_presets())

        lean = registry.get("lean_core")
        reliable = registry.get("reliable_stream")
        instant = registry.get("instant")
        turbo = registry.get("experimental_turbo")

        self.assertTrue(lean.settings["streaming.enabled"])
        self.assertLessEqual(lean.settings["streaming.interval_seconds"], 0.22)
        self.assertFalse(lean.settings["ai.cleanup_enabled"])
        self.assertFalse(lean.settings["typing.live_insert"])
        self.assertTrue(reliable.settings["streaming.enabled"])
        self.assertFalse(reliable.settings["finalization.allow_quiet_tail_reuse"])
        self.assertGreaterEqual(
            reliable.settings["endpoint.capture_tail_grace_seconds"],
            0.25,
        )
        self.assertTrue(instant.settings["streaming.enabled"])
        self.assertLessEqual(instant.settings["streaming.interval_seconds"], 0.18)
        self.assertTrue(turbo.experimental)
        self.assertTrue(turbo.settings["finalization.allow_quiet_tail_reuse"])
        self.assertLess(
            turbo.settings["streaming.interval_seconds"],
            reliable.settings["streaming.interval_seconds"],
        )

    def test_settings_registry_validates_and_serializes_public_schema(self):
        registry = SettingsRegistry()
        registry.register(
            SettingSpec(
                key="typing.smart_continuation",
                label="Smart continuation",
                section="Typing",
                kind=SettingKind.BOOLEAN,
                default=True,
            )
        )
        registry.register(
            SettingSpec(
                key="endpoint.silence_seconds",
                label="Silence timeout",
                section="Endpointing",
                kind=SettingKind.NUMBER,
                default=2.0,
                minimum=0.5,
                maximum=10.0,
            )
        )

        self.assertIs(registry.validate("typing.smart_continuation", False), False)
        self.assertEqual(registry.validate("endpoint.silence_seconds", 1.25), 1.25)
        with self.assertRaises(ValueError):
            registry.validate("endpoint.silence_seconds", 0.1)
        schema = registry.to_schema()
        self.assertEqual(schema[0]["key"], "typing.smart_continuation")
        self.assertEqual(schema[1]["minimum"], 0.5)

    def test_profile_registry_resolves_explicit_and_application_profiles(self):
        registry = ProfileRegistry(built_in_profiles())
        registry.register(
            DictationProfile(
                id="powershell",
                label="PowerShell",
                mode="terminal",
                executable_patterns=("pwsh.exe", "powershell.exe"),
                overrides={"typing.automatic_punctuation": False},
            )
        )

        self.assertEqual(registry.resolve(explicit_id="code").id, "code")
        self.assertEqual(registry.resolve(executable="pwsh.exe").id, "powershell")
        self.assertEqual(registry.resolve(executable="unknown.exe").id, "general")

    def test_command_registry_supports_literal_key_and_custom_actions(self):
        registry = CommandRegistry()
        registry.register_replacement("slash", "/", source="core")
        registry.register_command(
            CommandDefinition(
                phrase="press escape",
                action="key",
                value="escape",
                source="core",
            )
        )

        self.assertEqual(registry.replacements()["slash"], "/")
        self.assertEqual(registry.command_for("Press Escape.").value, "escape")
        self.assertIsNone(registry.command_for("please press escape later"))

    def test_local_plugin_registers_extensions_and_failure_is_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "good.py").write_text(
                "def register(registry):\n"
                "    registry.register_spoken_replacement('home path', '~/')\n"
                "    registry.register_profile(id='shell-x', label='Shell X', mode='terminal')\n",
                encoding="utf-8",
            )
            (root / "broken.py").write_text(
                "def register(registry):\n    raise RuntimeError('boom')\n",
                encoding="utf-8",
            )
            extensions = ExtensionRegistry()
            report = PluginManager(extensions).load_directory(root)

        self.assertEqual(extensions.commands.replacements()["home path"], "~/")
        self.assertEqual(extensions.profiles.get("shell-x").mode, "terminal")
        self.assertEqual(report.loaded, ("good",))
        self.assertEqual(report.failed[0].plugin, "broken")
        self.assertEqual(report.failed[0].error_type, "RuntimeError")


if __name__ == "__main__":
    unittest.main()
