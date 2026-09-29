"""Focused tests for ``drew_local_voice.nemotron_backend``.

These tests MUST NOT install packages, download model weights, query
CUDA devices, open UI, or load a real model. They exercise the
Nemotron backend entirely through fakes and temp directories.

The point is to lock in the *contract* the rest of GlitchScribe depends
on:

* the staged directory is validated (lone weights file is rejected),
* torch/transformers stay out of the import path,
* the device/dtype are pinned (no silent CPU fallback),
* ``preload_and_warm`` returns the metrics ``local_voice.py`` reads,
* ``transcribe_samples`` returns ``(text, metrics)`` with the same shape
  the streaming layer expects,
* ``language`` is always ``"en-US"`` and ``words`` is empty when the
  RNNT output has no timestamps,
* ``unload`` / ``is_loaded`` round-trip correctly,
* ``resource_metadata`` exposes a small, stable dict.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_MODULE = "drew_local_voice.nemotron_backend"


def _ensure_repo_root_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class _StageDir:
    """Create a temp directory mimicking the Nemotron staged layout."""

    def __init__(self, *, missing: tuple[str, ...] = ()):
        _ensure_repo_root_on_path()
        from drew_local_voice.nemotron_backend import (
            REQUIRED_COMPANION_FILENAMES,
            REQUIRED_WEIGHTS_FILENAME,
        )

        self._required_weights = REQUIRED_WEIGHTS_FILENAME
        self._required_companions = REQUIRED_COMPANION_FILENAMES

        handle = tempfile.TemporaryDirectory(prefix="nemotron-stage-")
        self.path = Path(handle.__enter__())
        self._handle = handle

        # We always create the weights file even when "missing" is asked
        # for some other file, because the validation order requires the
        # weights file to exist before companion-file validation runs.
        (self.path / self._required_weights).write_bytes(b"fake-weights")
        for name in self._required_companions:
            if name in missing:
                continue
            (self.path / name).write_text("{}", encoding="utf-8")

    def add_file(self, name: str, content: bytes | str = b"") -> Path:
        target = self.path / name
        if isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            target.write_bytes(content)
        return target

    def remove_weights(self) -> None:
        (self.path / self._required_weights).unlink()

    def cleanup(self) -> None:
        self._handle.__exit__(None, None, None)


class NemotronDirectoryValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        _ensure_repo_root_on_path()
        from drew_local_voice.nemotron_backend import NemotronBackend

        self.NemotronBackend = NemotronBackend

    def test_rejects_missing_directory(self) -> None:
        missing_path = Path(tempfile.gettempdir()) / "nemotron-no-such-dir-xyz"
        profile = SimpleNamespace(device="cpu", compute_type="float32")
        with self.assertRaises(Exception) as ctx:
            self.NemotronBackend(model_dir=missing_path).preload_and_warm(profile)
        # Any error class from our backend module is acceptable as long
        # as the path is called out.
        self.assertIn("no-such-dir-xyz", str(ctx.exception))

    def test_rejects_directory_with_only_weights(self) -> None:
        stage = _StageDir(missing=(
            "config.json",
            "generation_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ))
        try:
            profile = SimpleNamespace(device="cpu", compute_type="float32")
            with self.assertRaises(Exception) as ctx:
                self.NemotronBackend(model_dir=stage.path).preload_and_warm(profile)
            self.assertIn("incomplete", str(ctx.exception).lower())
        finally:
            stage.cleanup()

    def test_rejects_directory_missing_a_single_companion(self) -> None:
        stage = _StageDir()
        try:
            stage.remove_weights()  # drop weights so validation order fails
            profile = SimpleNamespace(device="cpu", compute_type="float32")
            with self.assertRaises(Exception) as ctx:
                self.NemotronBackend(model_dir=stage.path).preload_and_warm(profile)
            self.assertIn(self.NemotronBackend.__module__, type(ctx.exception).__module__)
        finally:
            stage.cleanup()

    def test_accepts_complete_staged_directory(self) -> None:
        stage = _StageDir()
        try:
            backend = self.NemotronBackend(model_dir=stage.path)
            self.assertEqual(backend.model_dir, stage.path)
            self.assertFalse(backend.is_loaded())
        finally:
            stage.cleanup()

    def test_accepts_model_path_alias(self) -> None:
        stage = _StageDir()
        try:
            backend = self.NemotronBackend(model_path=stage.path)
            self.assertEqual(backend.model_dir, stage.path)
        finally:
            stage.cleanup()

    def test_accepts_model_directory_alias(self) -> None:
        stage = _StageDir()
        try:
            backend = self.NemotronBackend(model_directory=stage.path)
            self.assertEqual(backend.model_dir, stage.path)
        finally:
            stage.cleanup()

    def test_rejects_directory_pointer_argument_missing(self) -> None:
        with self.assertRaises(Exception) as ctx:
            self.NemotronBackend()
        self.assertIn("directory", str(ctx.exception).lower())

    def test_path_pointer_argument_is_not_a_directory(self) -> None:
        with tempfile.NamedTemporaryFile(
            prefix="nemotron-pointer-", suffix=".txt", delete=False
        ) as handle:
            handle.write(b"x")
            pointer_path = Path(handle.name)
        try:
            profile = SimpleNamespace(device="cpu", compute_type="float32")
            with self.assertRaises(Exception) as ctx:
                self.NemotronBackend(model_dir=pointer_path).preload_and_warm(profile)
            self.assertIn("not a directory", str(ctx.exception))
        finally:
            pointer_path.unlink(missing_ok=True)


class NemotronDeviceAndDtypeTests(unittest.TestCase):
    """Device and dtype must be pinned. No silent CPU fallback."""

    def setUp(self) -> None:
        _ensure_repo_root_on_path()
        from drew_local_voice.nemotron_backend import (
            _resolve_device,
            _resolve_dtype,
        )

        self._resolve_device = _resolve_device
        self._resolve_dtype = _resolve_dtype

    def test_device_pins_cpu(self) -> None:
        self.assertEqual(self._resolve_device("cpu"), "cpu")
        self.assertEqual(self._resolve_device("CPU"), "cpu")
        self.assertEqual(self._resolve_device(" cpu "), "cpu")

    def test_device_pins_cuda_index(self) -> None:
        self.assertEqual(self._resolve_device("cuda:0"), "cuda:0")
        self.assertEqual(self._resolve_device("cuda:2"), "cuda:2")

    def test_device_rejects_plain_cuda_without_index(self) -> None:
        with self.assertRaises(ValueError):
            self._resolve_device("cuda")

    def test_device_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            self._resolve_device("metal")

    def test_device_rejects_non_string(self) -> None:
        with self.assertRaises(ValueError):
            self._resolve_device(0)  # type: ignore[arg-type]

    def test_dtype_pins_known_values(self) -> None:
        self.assertEqual(self._resolve_dtype("float16"), "float16")
        self.assertEqual(self._resolve_dtype("float32"), "float32")
        self.assertEqual(self._resolve_dtype("bfloat16"), "bfloat16")

    def test_dtype_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            self._resolve_dtype("int8")

    def test_dtype_rejects_non_string(self) -> None:
        with self.assertRaises(ValueError):
            self._resolve_dtype(1)  # type: ignore[arg-type]


class NemotronImportSafetyTests(unittest.TestCase):
    """``torch`` and ``transformers`` must stay out of the import path."""

    def setUp(self) -> None:
        _ensure_repo_root_on_path()

    def test_module_does_not_trigger_torch_or_transformers_at_import(self) -> None:
        # Block these modules so an accidental eager import is loud.
        blocked = {"torch": None, "transformers": None}
        original_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name in blocked or name.split(".", 1)[0] in blocked:
                raise AssertionError(
                    f"nemotron_backend must not import {name!r} at module load"
                )
            return original_import(name, *args, **kwargs)

        # Drop any cached version to make the test honest.
        for cached in list(sys.modules):
            if cached == BACKEND_MODULE or cached.startswith(BACKEND_MODULE + "."):
                sys.modules.pop(cached, None)

        with patch("builtins.__import__", side_effect=guarded_import):
            importlib.import_module(BACKEND_MODULE)

    def test_module_only_pulls_heavy_libs_on_preload(self) -> None:
        # Verify that simply constructing the backend (with a valid
        # staged directory) does NOT import transformers/torch.
        heavy_modules_before = {
            name
            for name in sys.modules
            if name.split(".", 1)[0] in {"torch", "transformers"}
        }
        stage = _StageDir()
        try:
            from drew_local_voice.nemotron_backend import NemotronBackend

            backend = NemotronBackend(model_dir=stage.path)
            heavy_modules_after = {
                name
                for name in sys.modules
                if name.split(".", 1)[0] in {"torch", "transformers"}
            }
            self.assertEqual(heavy_modules_after, heavy_modules_before)
            self.assertFalse(backend.is_loaded())
        finally:
            stage.cleanup()


class NemotronLifecycleTests(unittest.TestCase):
    """Preload / transcribe / unload round-trip with fakes."""

    def setUp(self) -> None:
        _ensure_repo_root_on_path()
        from drew_local_voice.nemotron_backend import NemotronBackend

        self.NemotronBackend = NemotronBackend
        self.stage = _StageDir()
        self.profile = SimpleNamespace(
            model_name="nemotron-3.5-asr-streaming-0.6b",
            device="cpu",
            compute_type="float32",
            device_index=None,
            label="cpu · float32",
            backend="transformers-rnnt",
        )

    def tearDown(self) -> None:
        self.stage.cleanup()

    def test_preload_uses_local_files_only(self) -> None:
        with self._patched_transformers() as (fake_model, fake_processor):
            backend = self.NemotronBackend(model_dir=self.stage.path)
            metrics = backend.preload_and_warm(self.profile)
            self.assertTrue(metrics["ready"])
            self.assertEqual(metrics["profile"], self.profile.label)
            self.assertGreaterEqual(metrics["load_seconds"], 0.0)
            # AutoModelForRNNT.from_pretrained was called with local_files_only
            self.assertTrue(fake_model.calls, "model factory was not called")
            kwargs = fake_model.calls[0]
            self.assertTrue(kwargs["local_files_only"])
            self.assertEqual(kwargs["torch_dtype"], fake_model.float32)
            # AutoProcessor.from_pretrained was called with local_files_only
            self.assertTrue(fake_processor.calls)
            self.assertTrue(fake_processor.calls[0]["local_files_only"])
            # Explicit device move happened.
            self.assertEqual(fake_model.moved_to, "cpu")
            self.assertTrue(fake_model.entered_eval)
            self.assertTrue(backend.is_loaded())

    def test_preload_reuses_signature_when_matching(self) -> None:
        with self._patched_transformers():
            backend = self.NemotronBackend(model_dir=self.stage.path)
            first = backend.preload_and_warm(self.profile)
            self.assertGreater(first["load_seconds"], 0.0)
            second = backend.preload_and_warm(self.profile)
            self.assertEqual(second["load_seconds"], 0.0)

    def test_preload_rejects_dtype_mismatch(self) -> None:
        with self._patched_transformers():
            backend = self.NemotronBackend(model_dir=self.stage.path)
            backend.preload_and_warm(self.profile)
            bad_profile = SimpleNamespace(
                model_name="nemotron-3.5-asr-streaming-0.6b",
                device="cpu",
                compute_type="int8",
                device_index=None,
                label="cpu · int8",
                backend="transformers-rnnt",
            )
            with self.assertRaises(ValueError):
                backend.transcribe_samples(self._silence(), bad_profile)

    def test_unload_round_trip(self) -> None:
        with self._patched_transformers():
            backend = self.NemotronBackend(model_dir=self.stage.path)
            backend.preload_and_warm(self.profile)
            self.assertTrue(backend.is_loaded())
            self.assertTrue(backend.unload())
            self.assertFalse(backend.is_loaded())
            # A second unload is a no-op.
            self.assertFalse(backend.unload())

    def test_resource_metadata_shape(self) -> None:
        with self._patched_transformers():
            backend = self.NemotronBackend(
                model_dir=self.stage.path,
                hotwords="alpha bravo",
                language="fr-FR",
            )
            meta = backend.resource_metadata()
            self.assertEqual(meta["backend"], "transformers-rnnt")
            self.assertEqual(meta["model_dir"], str(self.stage.path))
            self.assertEqual(meta["language"], "en-US")
            self.assertEqual(meta["hotwords"], "alpha bravo")
            self.assertFalse(meta["loaded"])
            backend.preload_and_warm(self.profile)
            meta2 = backend.resource_metadata()
            self.assertTrue(meta2["loaded"])
            self.assertIsNotNone(meta2["signature"])
            backend.unload()
            self.assertFalse(backend.resource_metadata()["loaded"])

    def test_language_normalizes_to_en_us_when_empty(self) -> None:
        with self._patched_transformers():
            backend = self.NemotronBackend(model_dir=self.stage.path)
            self.assertEqual(backend.language, "en-US")

    def test_transcribe_returns_text_and_metrics(self) -> None:
        with self._patched_transformers() as (fake_model, _fake_proc):
            fake_model.timed_output = [
                (101, 0.0),
                (202, 0.4),
                (303, 0.8),
            ]
            # Set up a fake tokenizer.decode that returns recognizable words.
            tokenizer = _FakeTokenizer(
                {
                    101: " hello",
                    202: " there",
                    303: " world",
                }
            )
            _fake_proc.tokenizer = tokenizer

            backend = self.NemotronBackend(model_dir=self.stage.path)
            backend.preload_and_warm(self.profile)
            text, metrics = backend.transcribe_samples(
                self._silence(), self.profile, fast=True
            )

            self.assertIn("hello", text)
            self.assertIn("there", text)
            self.assertIn("world", text)
            self.assertEqual(metrics["language"], "en-US")
            self.assertEqual(metrics["language_probability"], 1.0)
            self.assertEqual(metrics["profile"], self.profile.label)
            self.assertEqual(metrics["backend"], "transformers-rnnt")
            self.assertGreaterEqual(metrics["transcribe_seconds"], 0.0)
            self.assertGreaterEqual(metrics["load_seconds"], 0.0)
            self.assertEqual(metrics["fast"], True)
            self.assertTrue(fake_model.generate_calls)
            self.assertTrue(
                fake_model.generate_calls[-1]["return_dict_in_generate"]
            )
            self.assertTrue(fake_model.generate_calls[-1]["is_streaming"])
            # Three word dicts with start/end/probability.
            self.assertEqual(len(metrics["words"]), 3)
            for word in metrics["words"]:
                self.assertIn("word", word)
                self.assertIn("start", word)
                self.assertIn("end", word)
                self.assertIn("probability", word)

    def test_transcribe_leaves_words_empty_when_no_timestamps(self) -> None:
        with self._patched_transformers() as (fake_model, _fake_proc):
            # A tensor-like object with no .timed_tokens and no .sequences.
            fake_model.last_generate = object()
            backend = self.NemotronBackend(model_dir=self.stage.path)
            backend.preload_and_warm(self.profile)
            text, metrics = backend.transcribe_samples(
                self._silence(), self.profile, fast=True
            )
            self.assertEqual(metrics["words"], ())
            self.assertEqual(text, "")
            # Language still normalized.
            self.assertEqual(metrics["language"], "en-US")

    def test_transcribe_retries_without_unsupported_streaming_kwarg(self) -> None:
        with self._patched_transformers() as (fake_model, _fake_proc):
            fake_model.reject_streaming_keyword = True
            backend = self.NemotronBackend(model_dir=self.stage.path)
            backend.preload_and_warm(self.profile)
            text, metrics = backend.transcribe_samples(
                self._silence(), self.profile, fast=True
            )
            self.assertEqual(text, "")
            self.assertEqual(metrics["words"], ())
            self.assertEqual(len(fake_model.generate_calls), 2)
            self.assertTrue(fake_model.generate_calls[0]["is_streaming"])
            self.assertNotIn("is_streaming", fake_model.generate_calls[1])

    def test_transcribe_triggers_preload_if_not_loaded(self) -> None:
        with self._patched_transformers():
            backend = self.NemotronBackend(model_dir=self.stage.path)
            self.assertFalse(backend.is_loaded())
            text, _metrics = backend.transcribe_samples(
                self._silence(), self.profile, fast=True
            )
            self.assertTrue(backend.is_loaded())
            self.assertEqual(text, "")

    def test_transcribe_rejects_unloaded_then_mismatched_signature(self) -> None:
        with self._patched_transformers():
            backend = self.NemotronBackend(model_dir=self.stage.path)
            backend.preload_and_warm(self.profile)
            mismatched = SimpleNamespace(
                model_name="nemotron-3.5-asr-streaming-0.6b",
                device="cuda:0",
                compute_type="float16",
                device_index=0,
                label="cuda:0 · float16",
                backend="transformers-rnnt",
            )
            with self.assertRaises(Exception):
                backend.transcribe_samples(self._silence(), mismatched)

    def test_hotwords_setter_round_trips(self) -> None:
        with self._patched_transformers():
            backend = self.NemotronBackend(model_dir=self.stage.path)
            self.assertEqual(backend.hotwords, "")
            backend.hotwords = " alpha,beta "
            self.assertEqual(backend.hotwords, "alpha,beta")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _silence(self):
        import numpy as np

        return np.zeros(1600, dtype=np.float32)

    def _patched_transformers(self):
        """Patch ``transformers.AutoModelForRNNT`` / ``AutoProcessor`` so
        no real model is loaded.

        Returns an instance of ``_TransformersPatch`` which is both a
        context manager and exposes ``.model`` and ``.processor``
        namespaces for assertions. Use it as::

            with self._patched_transformers() as patch:
                patch.model.timed_output = [...]
                backend.transcribe_samples(...)
        """
        return _TransformersPatch()


class _FakeTensor:
    """A stand-in for a torch.Tensor that supports ``.to(device)``."""

    def __init__(self, size: int):
        self._size = size

    def to(self, device):
        return self

    def __len__(self) -> int:
        return self._size

    def __iter__(self):
        return iter(())


class _FakeTokenizer:
    """Maps token ids to strings for fake decoding."""

    def __init__(self, mapping: dict[int, str]):
        self._mapping = mapping

    def decode(self, ids, skip_special_tokens=True):
        if not ids:
            return ""
        if isinstance(ids, (list, tuple)):
            return "".join(self._mapping.get(int(token), "") for token in ids)
        return self._mapping.get(int(ids), "")


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeProcessor:
    def __init__(self):
        self.calls = []
        self.feature_extractor = SimpleNamespace(sampling_rate=16_000)
        self.tokenizer = _FakeTokenizer({})

    def __call__(self, audio, *, sampling_rate, language, return_tensors):
        self.calls.append(
            {
                "sampling_rate": sampling_rate,
                "language": language,
                "return_tensors": return_tensors,
            }
        )
        return {"input_values": _FakeTensor(len(audio))}

    def decode(self, sequences, skip_special_tokens=True):
        if sequences is None:
            return ""
        if isinstance(sequences, (list, tuple)) and sequences and isinstance(
            sequences[0], (list, tuple)
        ):
            sequences = sequences[0]
        return self.tokenizer.decode(
            list(sequences) if isinstance(sequences, (list, tuple)) else sequences,
            skip_special_tokens=skip_special_tokens,
        )


class _FakeModel:
    def __init__(self, dtype):
        self.calls = []
        self.float32 = dtype
        self.moved_to = None
        self.entered_eval = False
        self.timed_output = None
        self.last_generate = None
        self.generate_calls = []
        self.reject_streaming_keyword = False
        self.dtype = dtype

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.entered_eval = True
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.reject_streaming_keyword and "is_streaming" in kwargs:
            raise ValueError("unused model kwargs: ['is_streaming']")
        if self.last_generate is not None:
            return self.last_generate
        if self.timed_output is not None:
            return SimpleNamespace(
                sequences=[[101, 202, 303]],
                timed_tokens=tuple(self.timed_output),
            )
        return SimpleNamespace(sequences=[[]])


class _TransformersPatch:
    """Install inert torch/transformers modules for backend contract tests."""

    def __init__(self):
        self._previous = {}
        self.model = None
        self.processor = None

    def __enter__(self):
        fake_torch = ModuleType("torch")
        fake_torch.float32 = object()
        fake_torch.float16 = object()
        fake_torch.bfloat16 = object()
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_torch.no_grad = lambda: _NullContext()

        self.model = _FakeModel(fake_torch.float32)
        self.processor = _FakeProcessor()
        fake_model_factory = SimpleNamespace(
            from_pretrained=lambda directory, **kwargs: self._load_model(
                directory, **kwargs
            ),
        )
        fake_processor_factory = SimpleNamespace(
            from_pretrained=lambda directory, **kwargs: self._load_processor(
                directory, **kwargs
            ),
        )
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModelForRNNT = fake_model_factory
        fake_transformers.AutoProcessor = fake_processor_factory

        for name, module in (("torch", fake_torch), ("transformers", fake_transformers)):
            self._previous[name] = sys.modules.get(name)
            sys.modules[name] = module
        return self.model, self.processor

    def _load_model(self, _directory, **kwargs):
        self.model.calls.append(kwargs)
        return self.model

    def _load_processor(self, _directory, **kwargs):
        self.processor.calls.append(kwargs)
        return self.processor

    def __exit__(self, exc_type, exc, tb):
        for name, previous in self._previous.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        return False


if __name__ == "__main__":  # pragma: no cover
    unittest.main()