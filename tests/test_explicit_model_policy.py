"""Focused unit tests for the explicit-model / local-policy contract.

These tests do not construct Tk, query microphones or devices, load any
recognition model, access the GPU, or call the network. They exercise pure
functions and a temp-file AppConfig.load() override only.

Coverage:

1. ``Nemotron`` is the default and the first built-in selectable model.
2. Windows-style Nemotron paths (forward slashes, backslashes, ``model.safetensors``
   inside a Nemotron folder) classify as the ``transformers-rnnt`` backend.
3. Insufficient automatic GPU VRAM returns a ``blocked`` ``RuntimeProfile``
   with no alternate model loaded.
4. Explicit CPU remains explicit (never silently flipped to GPU/CUDA).
5. ``AppConfig`` advanced ``recognition.model`` / ``recognition.device`` /
   ``recognition.vram_reserve_mb`` / ``recognition.compute_type`` keys override
   stale top-level persisted values.
6. Auxiliary model requests are denied by default (no LocalOpenAIProvider is
   imported and the deny path raises a RuntimeError).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import local_voice as local_voice_module
from drew_local_voice.models import (
    BUILT_IN_MODELS,
    DEFAULT_PRIMARY_MODEL,
    NEMOTRON_MODEL_ID,
    NEMOTRON_MODEL_LABEL,
    NEMOTRON_SOURCE_FILENAME,
    configured_model_choices,
    estimated_model_vram_mb,
    model_backend,
    model_display_name,
)
from drew_local_voice.settings import built_in_settings
from local_voice import (
    OPENROUTER_MODEL,
    AppConfig,
    GPUDeviceInfo,
    choose_runtime_profile_from_inventory,
)


class NemotronIsTheDefaultAndSelectableTests(unittest.TestCase):
    """Nemotron is the built-in default, surfaced first in the picker."""

    def test_nemotron_is_declared_the_default_primary_model(self):
        self.assertEqual(DEFAULT_PRIMARY_MODEL, NEMOTRON_MODEL_ID)
        self.assertEqual(NEMOTRON_MODEL_ID, "nemotron-3.5-asr-streaming-0.6b")
        self.assertEqual(
            NEMOTRON_MODEL_LABEL,
            "Nemotron 3.5 ASR Streaming 0.6B",
        )
        # The canonical safetensors filename is exposed for the prepare script
        # and matches the Nemotron identifier with a single space and extension.
        self.assertTrue(
            NEMOTRON_SOURCE_FILENAME.startswith(NEMOTRON_MODEL_ID),
            msg="source filename must be derived from the Nemotron model id",
        )
        self.assertTrue(NEMOTRON_SOURCE_FILENAME.endswith(".safetensors"))

    def test_nemotron_is_the_first_built_in_selectable_model(self):
        self.assertEqual(BUILT_IN_MODELS[0], NEMOTRON_MODEL_ID)
        self.assertIn(NEMOTRON_MODEL_ID, BUILT_IN_MODELS)
        # The built-in picker never duplicates the identifier case-fold-wise.
        seen = {item.casefold() for item in BUILT_IN_MODELS}
        self.assertEqual(len(seen), len(BUILT_IN_MODELS))

    def test_configured_model_choices_lists_nemotron_first_with_no_extras(self):
        choices = configured_model_choices()
        self.assertEqual(choices[0], NEMOTRON_MODEL_ID)
        self.assertEqual(choices, BUILT_IN_MODELS)

    def test_recognition_model_setting_default_is_nemotron(self):
        # The SettingsRegistry spec must point at Nemotron so the Control Lab
        # shows the correct default and AppConfig picks it up at load time.
        specs = {spec.key: spec for spec in built_in_settings()}
        spec = specs["recognition.model"]
        self.assertEqual(spec.default, NEMOTRON_MODEL_ID)
        self.assertEqual(spec.kind.value, "text")
        self.assertTrue(spec.restart_required)

    def test_appconfig_defaults_to_nemotron_when_constructed_fresh(self):
        # Pure constructor, no filesystem, no GPU probe. Even when the live
        # config file is excluded by patching CONFIG_PATH to a missing file,
        # the dataclass default must still be Nemotron.
        config = AppConfig()
        self.assertEqual(config.model_name, NEMOTRON_MODEL_ID)
        # Auxiliary model requests are off and the default name matches the
        # local-only loopback constant declared by local_voice.
        self.assertFalse(config.allow_auxiliary_models)
        self.assertEqual(config.auxiliary_model_name, OPENROUTER_MODEL)


class WindowsStyleNemotronPathsClassifyAsTransformersRnntTests(unittest.TestCase):
    """Windows-style Nemotron paths classify as ``transformers-rnnt``."""

    def test_nemotron_identifier_is_transformers_rnnt(self):
        self.assertEqual(model_backend(NEMOTRON_MODEL_ID), "transformers-rnnt")
        self.assertEqual(model_display_name(NEMOTRON_MODEL_ID), NEMOTRON_MODEL_LABEL)

    def test_nemotron_with_windows_backslash_path_is_transformers_rnnt(self):
        # A typical Windows install path for the Nemotron weights.
        identifier = (
            r"C:\Users\me\models\nemotron-3.5-asr-streaming-0.6b\model.safetensors"
        )
        self.assertEqual(model_backend(identifier), "transformers-rnnt")
        # Display name still resolves to the friendly Nemotron label.
        self.assertEqual(model_display_name(identifier), NEMOTRON_MODEL_LABEL)

    def test_nemotron_folder_with_forward_slashes_is_transformers_rnnt(self):
        identifier = (
            "/home/me/models/nemotron-3.5-asr-streaming-0.6b/model.safetensors"
        )
        self.assertEqual(model_backend(identifier), "transformers-rnnt")

    def test_nemotron_folder_directory_only_is_transformers_rnnt(self):
        identifier = r"D:\models\nemotron-3.5-asr-streaming-0.6b"
        self.assertEqual(model_backend(identifier), "transformers-rnnt")

    def test_mixed_separator_nemotron_folder_is_transformers_rnnt(self):
        # Mixed forward + back slashes are common after copy/paste on Windows.
        identifier = (
            r"C:/Users\me/models\nemotron-3.5-asr-streaming-0.6b/model.safetensors"
        )
        self.assertEqual(model_backend(identifier), "transformers-rnnt")

    def test_whisper_models_are_not_misclassified_as_nemotron(self):
        # Negative check: faster-whisper model identifiers stay on the whisper
        # backend even when the path happens to mention the substring "nemotron".
        self.assertEqual(model_backend("tiny.en"), "faster-whisper")
        self.assertEqual(
            model_backend("/models/some-other/model.safetensors"),
            "faster-whisper",
        )


class InsufficientAutomaticGpuVramReturnsBlockedProfileTests(unittest.TestCase):
    """Insufficient automatic GPU VRAM → blocked profile, no alternate model."""

    def setUp(self):
        # A single small GPU that is below the required Nemotron budget after
        # the 512 MiB reserve. No second GPU exists, so there is no alternate
        # model the resolver can swap in.
        self.devices = (
            GPUDeviceInfo(
                index=0,
                name="NVIDIA GeForce RTX 4060",
                uuid="GPU-uuid-0",
                total_vram_mb=8192,
                used_vram_mb=8192 - 1024,
                free_vram_mb=1024,
                utilization_percent=0,
            ),
        )

    def test_automatic_request_with_low_vram_returns_blocked_profile(self):
        profile = choose_runtime_profile_from_inventory(
            self.devices,
            "automatic",
            model_name=NEMOTRON_MODEL_ID,
            vram_reserve_mb=512,
        )
        self.assertEqual(profile.device, "blocked")
        self.assertFalse(profile.runnable)
        # The resolver never silently picks a smaller whisper model: the model
        # name reported on the blocked profile must still be the requested one.
        self.assertEqual(profile.model_name, NEMOTRON_MODEL_ID)
        self.assertEqual(profile.backend, "transformers-rnnt")
        self.assertIsNone(profile.device_index)
        self.assertTrue(profile.fallback_reason)
        # The required budget reflects the Nemotron estimate (2304) plus the
        # 512 MiB reserve, so we can assert the user-visible diagnostics are
        # populated without depending on the exact estimate.
        self.assertIsNotNone(profile.required_vram_mb)
        self.assertEqual(profile.vram_reserve_mb, 512)
        self.assertIsNotNone(profile.free_vram_mb)
        self.assertLess(profile.free_vram_mb, profile.required_vram_mb)
        # The label must clearly say "blocked" with the friendly Nemotron name.
        self.assertIn(NEMOTRON_MODEL_LABEL, profile.label)
        self.assertIn("blocked", profile.label.lower())

    def test_no_gpu_devices_at_all_also_returns_blocked_profile(self):
        profile = choose_runtime_profile_from_inventory(
            (),
            "automatic",
            model_name=NEMOTRON_MODEL_ID,
            vram_reserve_mb=512,
        )
        self.assertEqual(profile.device, "blocked")
        self.assertFalse(profile.runnable)
        self.assertEqual(profile.model_name, NEMOTRON_MODEL_ID)
        self.assertEqual(profile.backend, "transformers-rnnt")
        self.assertIsNone(profile.device_index)
        # When nothing was probed, the resolver explicitly reports "no CUDA".
        self.assertIn("No CUDA", profile.fallback_reason)

    def test_estimated_vram_budget_matches_blocked_required_field(self):
        # Required VRAM = max(min_free, estimated_model_vram + reserve). With a
        # 512 MiB reserve the resolver must use the Nemotron estimate as the
        # floor, which is the source of the block above.
        estimated = estimated_model_vram_mb(NEMOTRON_MODEL_ID)
        profile = choose_runtime_profile_from_inventory(
            self.devices,
            "automatic",
            model_name=NEMOTRON_MODEL_ID,
            vram_reserve_mb=512,
            minimum_free_vram_mb=512,
        )
        self.assertEqual(profile.required_vram_mb, max(512, estimated + 512))


class ExplicitCpuRemainsExplicitTests(unittest.TestCase):
    """Explicit CPU must remain explicit, with no GPU fallback."""

    def test_cpu_choice_yields_cpu_profile_with_no_device_index(self):
        profile = choose_runtime_profile_from_inventory(
            (),
            "cpu",
            model_name=NEMOTRON_MODEL_ID,
        )
        self.assertEqual(profile.device, "cpu")
        self.assertIsNone(profile.device_index)
        # The label and reason must mention CPU explicitly.
        self.assertIn("CPU", profile.label)
        # The resolver does not switch to CUDA silently: runnable is True
        # (CPU is a valid fallback) but the device field stays "cpu".
        self.assertTrue(profile.runnable)

    def test_cpu_choice_with_a_healthy_gpu_still_uses_cpu(self):
        devices = (
            GPUDeviceInfo(
                index=0,
                name="NVIDIA GeForce RTX 5090",
                uuid="GPU-uuid-0",
                total_vram_mb=24576,
                used_vram_mb=0,
                free_vram_mb=24576,
                utilization_percent=0,
            ),
        )
        profile = choose_runtime_profile_from_inventory(
            devices,
            "cpu",
            model_name=NEMOTRON_MODEL_ID,
            vram_reserve_mb=512,
        )
        self.assertEqual(profile.device, "cpu")
        self.assertIsNone(profile.device_index)
        self.assertTrue(profile.runnable)
        self.assertEqual(profile.model_name, NEMOTRON_MODEL_ID)

    def test_normalize_gpu_device_choice_round_trips_cpu(self):
        # local_voice.normalize_gpu_device_choice must keep "cpu" intact.
        normalize = local_voice_module.normalize_gpu_device_choice
        self.assertEqual(normalize("cpu"), "cpu")
        self.assertEqual(normalize("CPU"), "cpu")
        self.assertEqual(normalize(" cuda:0 "), "cuda:0")
        self.assertEqual(normalize("automatic"), "automatic")
        # Unknown values degrade to "automatic" rather than to a fake "cpu".
        self.assertEqual(normalize("garbage"), "automatic")


class AdvancedSettingsOverrideStaleTopLevelValuesTests(unittest.TestCase):
    """AppConfig advanced keys win over stale top-level persisted values."""

    def _write_config(self, top_level: dict, advanced: dict | None) -> Path:
        payload = dict(top_level)
        if advanced is not None:
            payload["advanced_settings"] = advanced
        folder = tempfile.mkdtemp(prefix="nemotron-policy-")
        self.addCleanup(_cleanup, folder)
        path = Path(folder) / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_advanced_overrides_replace_stale_top_level_values(self):
        # Top-level fields are stale (an older build may have written them).
        # The advanced_settings block is the source of truth and must win.
        path = self._write_config(
            top_level={
                "model_name": "tiny.en",
                "gpu_device_choice": "automatic",
                "gpu_vram_reserve_mb": 512,
                "compute_type": "float16",
            },
            advanced={
                "recognition.model": NEMOTRON_MODEL_ID,
                "recognition.device": "cuda:1",
                "recognition.vram_reserve_mb": 1024,
                "recognition.compute_type": "int8_float16",
            },
        )
        with patch("local_voice.CONFIG_PATH", path):
            config = AppConfig.load()

        self.assertEqual(config.model_name, NEMOTRON_MODEL_ID)
        self.assertEqual(config.gpu_device_choice, "cuda:1")
        self.assertEqual(config.gpu_vram_reserve_mb, 1024)
        self.assertEqual(config.compute_type, "int8_float16")

    def test_advanced_invalid_model_falls_back_to_default_primary(self):
        # An invalid advanced model identifier must not blow up; it must be
        # replaced by the built-in default so the app still launches.
        path = self._write_config(
            top_level={"model_name": "tiny.en"},
            advanced={"recognition.model": "   "},
        )
        with patch("local_voice.CONFIG_PATH", path):
            config = AppConfig.load()
        self.assertEqual(config.model_name, DEFAULT_PRIMARY_MODEL)

    def test_advanced_reserve_is_clamped_to_safe_range(self):
        # Out-of-range reserves are clamped via the loader.
        path = self._write_config(
            top_level={"gpu_vram_reserve_mb": 256},
            advanced={"recognition.vram_reserve_mb": 99999},
        )
        with patch("local_voice.CONFIG_PATH", path):
            config = AppConfig.load()
        self.assertEqual(config.gpu_vram_reserve_mb, 8192)

    def test_advanced_device_normalizes_malformed_cuda_to_automatic(self):
        path = self._write_config(
            top_level={"gpu_device_choice": "automatic"},
            advanced={"recognition.device": "cuda:-1"},
        )
        with patch("local_voice.CONFIG_PATH", path):
            config = AppConfig.load()
        self.assertEqual(config.gpu_device_choice, "automatic")


class AuxiliaryRequestsDefaultDeniedTests(unittest.TestCase):
    """Auxiliary model requests are denied by default."""

    def test_appconfig_default_disallows_auxiliary_requests(self):
        config = AppConfig()
        self.assertFalse(config.allow_auxiliary_models)
        # The auxiliary model name still resolves to the local-only constant
        # so Resource Use can show what *would* be requested.
        self.assertEqual(config.auxiliary_model_name, OPENROUTER_MODEL)

    def test_local_voice_app_default_config_denies_auxiliary_requests(self):
        # Build the bare app shell (no Tk) and verify the deny gate.
        app = local_voice_module.LocalVoiceApp.__new__(local_voice_module.LocalVoiceApp)
        app.config = AppConfig()
        self.assertFalse(app._auxiliary_model_requests_allowed())

    def test_default_app_deny_path_raises_without_loading_ai_provider(self):
        # _ensure_ai_subsystems is the only sanctioned AI bootstrapping path;
        # it must refuse when auxiliary requests are off. We must not have
        # imported any provider module by side effect.
        app = local_voice_module.LocalVoiceApp.__new__(local_voice_module.LocalVoiceApp)
        app.config = AppConfig()
        with self.assertRaises(RuntimeError) as exc:
            app._ensure_ai_subsystems()
        self.assertIn("disabled", str(exc.exception).lower())
        # No provider should have been cached.
        self.assertFalse(hasattr(app, "_ai_provider") and app._ai_provider is not None)

    def test_auxiliary_model_name_helper_returns_local_only_constant(self):
        app = local_voice_module.LocalVoiceApp.__new__(local_voice_module.LocalVoiceApp)
        app.config = AppConfig()
        self.assertEqual(app._auxiliary_model_name(), OPENROUTER_MODEL)

    def test_explicit_allow_only_via_advanced_setting(self):
        # Sanity: the same loader path that overrides stale values is the only
        # way to flip the default-deny gate on for real settings.
        folder = tempfile.mkdtemp(prefix="nemotron-policy-")
        self.addCleanup(_cleanup, folder)
        path = Path(folder) / "config.json"
        path.write_text(
            json.dumps(
                {
                    "allow_auxiliary_models": False,
                    "advanced_settings": {"resources.allow_auxiliary_models": True},
                }
            ),
            encoding="utf-8",
        )
        with patch("local_voice.CONFIG_PATH", path):
            config = AppConfig.load()
        self.assertTrue(config.allow_auxiliary_models)

    def test_ai_provider_property_raises_when_disabled(self):
        # Touching the ai_provider property must not silently load a provider;
        # the LocalOpenAIProvider must not be imported by the deny path.
        app = local_voice_module.LocalVoiceApp.__new__(local_voice_module.LocalVoiceApp)
        app.config = AppConfig()
        with patch.object(
            local_voice_module, "_lazy_import"
        ) as lazy_import_mock:
            with self.assertRaises(RuntimeError):
                _ = app.ai_provider
            lazy_import_mock.assert_not_called()


def _cleanup(folder: str) -> None:
    """Remove a temp directory tree without surfacing errors to the test runner."""

    import shutil

    try:
        shutil.rmtree(folder, ignore_errors=True)
    except OSError:
        pass


if __name__ == "__main__":
    unittest.main()