"""Nemotron 3.5 ASR Streaming 0.6B backend (transformers RNNT, local-only).

This module implements the only backend allowed to interpret the explicit
Nemotron model identifier. It deliberately keeps ``torch`` and
``transformers`` out of the module import path so the cold-start cost of
importing ``drew_local_voice`` is not paid when the Nemotron backend is
not selected. The heavy libraries are only imported on demand inside
``_import_transformers`` and ``preload_and_warm``.

Loading contract (matches ``local_voice.py``):

* Reads weights and processor from an explicit staged directory only.
  No fallback to Hugging Face Hub, no ``local_files_only=False`` path.
* Validates that the complete directory exists with companion files:
  ``model.safetensors``, ``config.json``, ``generation_config.json``,
  ``processor_config.json``, ``tokenizer.json``, ``tokenizer_config.json``.
  A lone ``.safetensors`` or missing companions is rejected explicitly
  before any heavy import is attempted.
* Pin the device to either ``"cpu"`` or ``"cuda:<index>"`` and pin the
  dtype to a single explicit value. No silent CPU fallback when CUDA
  is requested and unavailable.
* ``preload_and_warm(profile)``, ``transcribe_samples(samples, profile,
  fast=True)`` returning ``(text, metrics)``, ``unload()``,
  ``is_loaded()``, and ``resource_metadata()`` are exposed. ``language``
  is normalized to ``"en-US"``; ``words`` is left empty when the RNNT
  output has no timestamps.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

# torch and transformers are deliberately NOT imported at module load.
# They are pulled in lazily inside _import_transformers() and the
# preload path so simply importing drew_local_voice stays cheap.

REQUIRED_WEIGHTS_FILENAME = "model.safetensors"
REQUIRED_COMPANION_FILENAMES = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
DEFAULT_LANGUAGE = "en-US"
SUPPORTED_DEVICES = ("cpu", "cuda")
SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")
SUPPORTED_MODEL_KIND = "transformers-rnnt"

# Names that local_voice.py may pass via inspect.signature. We accept any
# of them but only use the first one we recognize for the staged
# directory. This keeps the construction call tolerant of renames
# without forcing local_voice.py to change.
DIRECTORY_PARAM_NAMES = ("model_dir", "model_directory", "model_path")
SOURCE_PARAM_NAMES = ("source_path",)
HOTWORDS_PARAM_NAMES = ("hotwords",)
LANGUAGE_PARAM_NAMES = ("language",)


class NemotronBackendError(ValueError):
    """Raised when the Nemotron runtime cannot be started or used."""


def _resolve_dtype(value: Any) -> str:
    """Pin the dtype to a single explicit string; reject anything else."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in SUPPORTED_DTYPES:
            return normalized
    raise NemotronBackendError(
        "compute_type must be one of "
        + ", ".join(SUPPORTED_DTYPES)
        + f" (got {value!r})"
    )


def _resolve_device(value: Any) -> str:
    """Pin the device to either ``cpu`` or ``cuda:<index>`` exactly.

    Anything else is rejected: no silent CPU fallback, no ``cuda``
    without an index, no typos. ``local_voice.py`` normalizes the user
    choice upstream; we only accept what is already explicit.
    """
    if not isinstance(value, str):
        raise NemotronBackendError(
            f"device must be a string (got {type(value).__name__})"
        )
    lowered = value.strip().lower()
    if lowered == "cpu":
        return "cpu"
    if lowered.startswith("cuda"):
        suffix = lowered[len("cuda") :]
        if suffix.startswith(":"):
            index_text = suffix[1:]
            if index_text.isdigit():
                return f"cuda:{int(index_text)}"
    raise NemotronBackendError(
        "device must be 'cpu' or 'cuda:<index>' "
        f"(got {value!r})"
    )


def _validate_directory(directory: Path) -> Path:
    """Reject a missing directory or any incomplete staged layout.

    A lone ``model.safetensors`` is explicitly rejected even when the
    weight file itself is present and well-formed: the runtime cannot
    start without the processor and tokenizer companions.
    """
    if directory is None:
        raise NemotronBackendError("model directory is required")
    if not isinstance(directory, Path):
        directory = Path(directory)
    if not directory.exists():
        raise NemotronBackendError(
            f"Nemotron model directory does not exist: {directory}"
        )
    if not directory.is_dir():
        raise NemotronBackendError(
            f"Nemotron model path is not a directory: {directory}"
        )

    weights_path = directory / REQUIRED_WEIGHTS_FILENAME
    if not weights_path.is_file():
        raise NemotronBackendError(
            f"Nemotron weights file missing: {weights_path}"
        )

    missing = [
        name
        for name in REQUIRED_COMPANION_FILENAMES
        if not (directory / name).is_file()
    ]
    if missing:
        raise NemotronBackendError(
            "Nemotron model directory is incomplete: missing "
            + ", ".join(missing)
        )

    return directory


def _first_matching(values: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in values and values[name] not in (None, ""):
            return values[name]
    return None


def _import_transformers():
    """Lazy import of transformers + torch for the heavy preload path.

    Returns a tuple ``(transformers_module, torch_module)``. Callers must
    already have validated the staged directory before calling this.
    """
    try:
        transformers = importlib.import_module("transformers")
    except Exception as exc:  # pragma: no cover - import error path
        raise NemotronBackendError(
            "transformers is required to load the Nemotron backend: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - import error path
        raise NemotronBackendError(
            "torch is required to load the Nemotron backend: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return transformers, torch


def _torch_dtype(torch_module, name: str):
    """Resolve an explicit torch dtype by name; reject anything fuzzy."""
    dtype = getattr(torch_module, name, None)
    if dtype is None:
        raise NemotronBackendError(
            f"torch has no dtype named {name!r}; expected one of "
            + ", ".join(SUPPORTED_DTYPES)
        )
    return dtype


class NemotronBackend:
    """Local-files-only Nemotron RNNT backend.

    The constructor pins everything that can be wrong before the heavy
    imports: the staged directory must exist with the complete
    companion layout, the device must be ``cpu`` or ``cuda:<index>``,
    and the dtype must be one of the explicit supported values.
    """

    def __init__(
        self,
        model_dir: Path | str | None = None,
        model_directory: Path | str | None = None,
        model_path: Path | str | None = None,
        source_path: Path | str | None = None,
        hotwords: str | None = None,
        language: str | None = None,
    ):
        chosen = _first_matching(
            {
                "model_dir": model_dir,
                "model_directory": model_directory,
                "model_path": model_path,
            },
            DIRECTORY_PARAM_NAMES,
        )
        if chosen is None:
            raise NemotronBackendError(
                "NemotronBackend requires a model directory argument"
            )
        # Keep construction side-effect-free and allow the UI to start even
        # when a configured artifact is missing. The same strict validation
        # runs at the beginning of preload_and_warm(), before any heavy
        # runtime import, so the warmup path can report MODEL BLOCKED.
        self._model_dir: Path = Path(chosen)

        self._source_path = (
            Path(source_path) if source_path not in (None, "") else None
        )
        self._hotwords = str(hotwords or "").strip()
        # Per spec: this backend is English-only and always reports the
        # canonical language tag, regardless of a stale caller setting.
        self._language = DEFAULT_LANGUAGE

        # Runtime state. None means "not loaded yet".
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._signature: tuple[str, str, str] | None = None
        self._load_seconds: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def hotwords(self) -> str:
        return self._hotwords

    @hotwords.setter
    def hotwords(self, value: str | None) -> None:
        self._hotwords = str(value or "").strip()

    @property
    def language(self) -> str:
        return self._language

    @property
    def backend(self) -> str:
        return SUPPORTED_MODEL_KIND

    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def resource_metadata(self) -> dict[str, Any]:
        """Return a small, JSON-safe description of the runtime state.

        The resource ledger in ``resource_ledger.py`` reads this; tests
        rely on the keys being stable.
        """
        return {
            "backend": self.backend,
            "model_dir": str(self._model_dir),
            "weights_filename": REQUIRED_WEIGHTS_FILENAME,
            "companions": list(REQUIRED_COMPANION_FILENAMES),
            "language": self._language,
            "hotwords": self._hotwords,
            "loaded": self.is_loaded(),
            "signature": self._signature,
            "load_seconds": round(float(self._load_seconds), 6),
            "source_path": (
                str(self._source_path) if self._source_path is not None else None
            ),
        }

    # ------------------------------------------------------------------
    # Loading / unloading
    # ------------------------------------------------------------------
    def _signature_from_profile(self, profile: Any) -> tuple[str, str, str]:
        requested_device = getattr(profile, "device", None)
        # The existing Faster-Whisper profile contract represents CUDA as
        # ``device="cuda"`` plus ``device_index``. Convert that pair to the
        # explicit Transformers device string without permitting a CPU
        # fallback or an unindexed CUDA request.
        if str(requested_device).strip().lower() == "cuda":
            index = getattr(profile, "device_index", None)
            if index is None:
                raise NemotronBackendError(
                    "cuda profile requires an explicit device_index"
                )
            requested_device = f"cuda:{int(index)}"
        device = _resolve_device(requested_device)
        dtype = _resolve_dtype(getattr(profile, "compute_type", None))
        return (self._model_dir.as_posix(), device, dtype)

    def preload_and_warm(self, profile: Any) -> dict[str, Any]:
        """Load the model on the profile's device and return warmup metrics.

        Heavy imports happen here (not at module import time). The
        signature must match the profile's pinned (device, dtype) or we
        refuse to silently downgrade.
        """
        with self._lock:
            _validate_directory(self._model_dir)
            signature = self._signature_from_profile(profile)
            started = time.perf_counter()
            if (
                self._signature == signature
                and self._model is not None
                and self._processor is not None
            ):
                # Already loaded with the right signature; report zero.
                self._load_seconds = 0.0
                return {
                    "ready": True,
                    "profile": getattr(profile, "label", ""),
                    "load_seconds": 0.0,
                    "warm_seconds": 0.0,
                    "backend": self.backend,
                }
            # Drop any previously loaded model that does not match.
            if self._model is not None or self._processor is not None:
                self._drop_model_unlocked()

            transformers, torch_module = _import_transformers()
            self._torch = torch_module
            dtype_name = signature[2]
            torch_dtype = _torch_dtype(torch_module, dtype_name)
            device = signature[1]

            # local_files_only=True is non-negotiable; we never fall back
            # to a remote download.
            try:
                processor = transformers.AutoProcessor.from_pretrained(
                    str(self._model_dir),
                    local_files_only=True,
                )
            except Exception as exc:
                raise NemotronBackendError(
                    "Failed to load Nemotron processor from "
                    f"{self._model_dir}: {type(exc).__name__}: {exc}"
                ) from exc
            try:
                model = transformers.AutoModelForRNNT.from_pretrained(
                    str(self._model_dir),
                    local_files_only=True,
                    torch_dtype=torch_dtype,
                )
            except Exception as exc:
                raise NemotronBackendError(
                    "Failed to load Nemotron model from "
                    f"{self._model_dir}: {type(exc).__name__}: {exc}"
                ) from exc

            # Explicit device move with no fallback.
            try:
                model.to(device)
            except Exception as exc:
                raise NemotronBackendError(
                    f"Failed to move Nemotron model to {device}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            try:
                model.eval()
            except Exception as exc:
                raise NemotronBackendError(
                    "Failed to set Nemotron model to eval mode: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            self._model = model
            self._processor = processor
            self._signature = signature
            self._load_seconds = time.perf_counter() - started

            return {
                "ready": True,
                "profile": getattr(profile, "label", ""),
                "load_seconds": float(self._load_seconds),
                "warm_seconds": 0.0,
                "backend": self.backend,
            }

    def _drop_model_unlocked(self) -> None:
        """Release model + processor. Must be called with the lock held."""
        previous_model = self._model
        previous_processor = self._processor
        self._model = None
        self._processor = None
        self._signature = None
        if previous_model is not None:
            del previous_model
        if previous_processor is not None:
            del previous_processor
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        if self._torch is not None:
            try:
                cuda = getattr(self._torch, "cuda", None)
                if cuda is not None and bool(getattr(cuda, "is_available", lambda: False)()):
                    empty_cache = getattr(cuda, "empty_cache", None)
                    if callable(empty_cache):
                        empty_cache()
            except Exception:
                pass

    def unload(self) -> bool:
        """Drop the resident model and processor; return True if anything
        was actually unloaded."""
        with self._lock:
            if self._model is None and self._processor is None:
                return False
            self._drop_model_unlocked()
            return True

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------
    def transcribe_samples(
        self,
        samples: Any,
        profile: Any,
        fast: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        """Transcribe a float32 PCM mono buffer.

        Returns ``(text, metrics)``. ``metrics`` has the same shape used
        by ``WhisperBackend``: ``load_seconds``, ``transcribe_seconds``,
        ``words``, ``language``, ``language_probability``, ``profile``.
        ``language`` is always ``"en-US"`` per spec; ``words`` is empty
        when the RNNT output carries no timestamps.
        """
        # Local imports kept inside the call so the test module can be
        # imported without numpy/torch present.
        try:
            import numpy as _np  # type: ignore
        except Exception as exc:  # pragma: no cover - numpy is required
            raise NemotronBackendError(
                "numpy is required to transcribe: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not self.is_loaded():
            self.preload_and_warm(profile)

        audio = _np.ascontiguousarray(
            _np.asarray(samples, dtype=_np.float32).reshape(-1)
        )

        with self._lock:
            signature = self._signature_from_profile(profile)
            if (
                self._signature != signature
                or self._model is None
                or self._processor is None
            ):
                raise NemotronBackendError(
                    "Nemotron model is not loaded for the requested "
                    f"profile signature {signature}; "
                    f"current signature is {self._signature}"
                )

            load_seconds = float(self._load_seconds)

            started = time.perf_counter()
            text, words = self._run_inference(audio, fast=bool(fast))
            transcribe_seconds = time.perf_counter() - started

        text = " ".join(str(text or "").split())

        return text, {
            "load_seconds": load_seconds,
            "transcribe_seconds": transcribe_seconds,
            "words": words,
            "language": self._language,
            "language_probability": 1.0,
            "profile": getattr(profile, "label", ""),
            "backend": self.backend,
            "fast": bool(fast),
        }

    def _run_inference(
        self,
        audio,
        *,
        fast: bool,
    ) -> tuple[str, tuple[dict[str, Any], ...]]:
        """Run the actual model and return words.

        When the RNNT output has no timestamp metadata, ``words`` is empty
        while decoded text is still returned.
        """
        model = self._model
        processor = self._processor
        torch_module = self._torch
        if model is None or processor is None or torch_module is None:
            raise NemotronBackendError("Nemotron backend is not loaded")

        try:
            inputs = processor(
                audio,
                sampling_rate=getattr(processor, "feature_extractor", None)
                and getattr(processor.feature_extractor, "sampling_rate", 16_000)
                or 16_000,
                language=self._language,
                return_tensors="pt",
            )
        except Exception as exc:
            raise NemotronBackendError(
                "Failed to preprocess audio for Nemotron: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        device = self._signature[1] if self._signature else "cpu"
        try:
            batch_to = getattr(inputs, "to", None)
            if callable(batch_to):
                model_dtype = getattr(model, "dtype", None)
                if model_dtype is None:
                    inputs = batch_to(device)
                else:
                    try:
                        inputs = batch_to(device=device, dtype=model_dtype)
                    except TypeError:
                        inputs = batch_to(device)
            else:
                inputs = {
                    key: value.to(device) for key, value in inputs.items()
                }
        except Exception as exc:
            raise NemotronBackendError(
                f"Failed to move Nemotron inputs to {device}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            with torch_module.no_grad():
                generation_kwargs = {"return_dict_in_generate": True}
                if fast:
                    generation_kwargs["is_streaming"] = True
                try:
                    generated = model.generate(
                        **inputs,
                        **generation_kwargs,
                    )
                except (TypeError, ValueError) as exc:
                    # Transformers generation APIs have differed on whether
                    # the optional streaming kwarg is accepted by the model
                    # class. Keep the streaming request for implementations
                    # that support it, but retry once without it when the
                    # installed generation validator explicitly rejects it.
                    if (
                        not fast
                        or "is_streaming" not in generation_kwargs
                        or "is_streaming" not in str(exc)
                    ):
                        raise
                    generation_kwargs.pop("is_streaming", None)
                    generated = model.generate(
                        **inputs,
                        **generation_kwargs,
                    )
        except Exception as exc:
            raise NemotronBackendError(
                "Nemotron inference failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        text = _decode_generated_text(generated, processor)
        words = _extract_words(
            generated=generated,
            processor=processor,
            torch_module=torch_module,
        )
        return text, words


def _generated_value(generated: Any, name: str) -> Any:
    if isinstance(generated, Mapping):
        return generated.get(name)
    return getattr(generated, name, None)


def _decode_generated_text(generated: Any, processor: Any) -> str:
    """Decode RNNT ``sequences`` without requiring timestamp metadata."""
    sequences = _generated_value(generated, "sequences")
    if sequences is None:
        return ""
    decode = getattr(processor, "decode", None)
    if not callable(decode):
        decode = getattr(getattr(processor, "tokenizer", None), "decode", None)
    if not callable(decode):
        return ""
    try:
        value = decode(sequences, skip_special_tokens=True)
    except TypeError:
        value = decode(sequences)
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    return str(value or "")


def _extract_words(
    *,
    generated: Any,
    processor: Any,
    torch_module: Any,
) -> tuple[dict[str, Any], ...]:
    """Extract timestamped words from a Nemotron generate() result.

    When timestamps are absent, return an empty tuple; callers still decode
    ``sequences`` separately for transcript text.
    """
    timed = _generated_value(generated, "timed_tokens")
    if timed is None:
        # RNNT sequences can be decoded to text, but they are not word
        # timestamps. Keep the public word list empty unless timestamps are
        # explicitly present in the generation result.
        return ()

    words: list[dict[str, Any]] = []
    for entry in timed:
        token_id, time_value = _split_timed_entry(entry, torch_module)
        if token_id is None:
            continue
        text = _decode_token(token_id, processor)
        if not text.strip():
            continue
        start_seconds, end_seconds = _time_to_window(time_value)
        words.append(
            {
                "word": text,
                "start": float(start_seconds),
                "end": float(end_seconds),
                "probability": 1.0,
            }
        )
    return tuple(words)


def _split_timed_entry(entry: Any, torch_module: Any) -> tuple[Any, Any]:
    """Split a timed-tokens entry into ``(token_id, time)``."""
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return entry[0], entry[1]
    return getattr(entry, "token", entry), getattr(entry, "time", None)


def _time_to_window(value: Any) -> tuple[float, float]:
    """Convert a single time value to a ``(start, end)`` window.

    When the model gives us a single value per token we treat it as the
    end time of that token and the start of the next. For the last
    token we collapse both ends onto the same value to keep the window
    legal.
    """
    if value is None:
        return 0.0, 0.0
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.0, 0.0
    return seconds, seconds


def _decode_token(token_id: Any, processor: Any) -> str:
    """Best-effort decode of a single token id using the loaded processor."""
    decode = getattr(getattr(processor, "tokenizer", None), "decode", None)
    if not callable(decode):
        return ""
    try:
        return str(decode([token_id], skip_special_tokens=True))
    except Exception:
        return ""


def _tokens_to_words(
    *, tokens: Any, processor: Any
) -> tuple[dict[str, Any], ...]:
    """Decode a flat tensor/sequence of token ids into words.

    Used as the fallback when the RNNT output does not include
    timestamps. Per spec, when there are no timestamps we leave the
    ``words`` list empty.
    """
    if tokens is None:
        return ()
    return ()


__all__ = [
    "NemotronBackend",
    "NemotronBackendError",
    "REQUIRED_WEIGHTS_FILENAME",
    "REQUIRED_COMPANION_FILENAMES",
    "DEFAULT_LANGUAGE",
    "SUPPORTED_DEVICES",
    "SUPPORTED_DTYPES",
    "SUPPORTED_MODEL_KIND",
]