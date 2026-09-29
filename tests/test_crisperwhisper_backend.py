from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import tempfile
import unittest

import numpy as np

from drew_local_voice.crisperwhisper_backend import (
    CrisperWhisperBackend,
    CrisperWhisperBackendError,
    resolve_local_model_directory,
)
from drew_local_voice.models import model_backend, model_display_name
from drew_local_voice.settings import SettingsRegistry, built_in_settings


@dataclass
class Profile:
    model_name: str = "crisperwhisper:small"
    device: str = "cpu"
    device_index: int | None = None
    compute_type: str = "float32"
    label: str = "CrisperWhisper 2.0 Small — CPU"


class CrisperWhisperBackendTests(unittest.TestCase):
    def test_crisper_model_registry_is_explicit(self):
        self.assertEqual(
            model_backend("crisperwhisper:small"),
            "crisperwhisper-transformers",
        )
        self.assertEqual(
            model_display_name("crisperwhisper:turbo"),
            "CrisperWhisper 2.0 Turbo",
        )

    def test_missing_weights_fail_closed_without_factory(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(
                CrisperWhisperBackendError,
                "not installed locally",
            ):
                resolve_local_model_directory(
                    "crisperwhisper:small",
                    model_root=Path(folder),
                )

    def test_backend_uses_local_model_and_maps_word_timings(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp_path = Path(folder)
            model_dir = tmp_path / "CrisperWhisper2.0_small"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            calls: list[tuple[str, dict]] = []
            test_case = self

            class FakeModel:
                def transcribe(self, audio, **kwargs):
                    test_case.assertIsInstance(audio, np.ndarray)
                    calls.append(("transcribe", kwargs))
                    return SimpleNamespace(
                        text="[um] we ship it",
                        language="en",
                        words=[
                            SimpleNamespace(word="[um]", start=0.1, end=0.3),
                            SimpleNamespace(word="we", start=0.4, end=0.6),
                            SimpleNamespace(word="ship", start=0.7, end=0.9),
                            SimpleNamespace(word="it", start=1.0, end=1.1),
                        ],
                    )

            def factory(path: str, **kwargs):
                calls.append((path, kwargs))
                self.assertEqual(path, str(model_dir))
                self.assertEqual(kwargs["backend"], "transformers")
                self.assertEqual(kwargs["device"], "cpu")
                return FakeModel()

            backend = CrisperWhisperBackend(
                model_name="crisperwhisper:small",
                model_root=tmp_path,
                mode="verbatim",
                model_factory=factory,
            )
            warm = backend.preload_and_warm(Profile())
            text, metrics = backend.transcribe_samples(
                np.zeros(16_000, dtype=np.float32), Profile()
            )

            self.assertIs(warm["ready"], True)
            self.assertEqual(warm["backend"], "crisperwhisper-transformers")
            self.assertEqual(text, "[um] we ship it")
            self.assertEqual(metrics["mode"], "verbatim")
            self.assertEqual(
                [word["word"] for word in metrics["words"]],
                ["[um]", "we", "ship", "it"],
            )
            self.assertIs(calls[1][1]["word_timestamps"], True)
            self.assertEqual(calls[1][1]["mode"], "verbatim")

    def test_crisper_settings_are_in_registry(self):
        registry = SettingsRegistry(built_in_settings())
        defaults = registry.defaults()
        self.assertEqual(defaults["recognition.transcription_mode"], "verbatim")
        self.assertIn("recognition.crisperwhisper_model_path", defaults)
