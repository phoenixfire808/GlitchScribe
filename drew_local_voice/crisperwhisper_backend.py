"""Local-only CrisperWhisper 2.0 backend for the primary ASR path.

The package's Transformers backend is supported on Windows.  This adapter
keeps the existing ``transcribe_samples`` contract used by the streaming
pipeline, exposes word timings, and refuses to construct a model unless the
weights already exist in an explicit local directory.  It never calls
Hugging Face download helpers implicitly.
"""

from __future__ import annotations

import gc
import importlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

SAMPLE_RATE = 16_000
_BACKEND_NAME = "crisperwhisper-transformers"
_VALID_SIZES = {"large", "turbo", "medium", "small"}


class CrisperWhisperBackendError(RuntimeError):
    """Raised when the explicit local CrisperWhisper runtime is unavailable."""


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _size_from_model_name(model_name: str) -> str:
    value = str(model_name or "").strip().casefold()
    if not value.startswith("crisperwhisper:"):
        raise CrisperWhisperBackendError(
            f"Unsupported CrisperWhisper model identifier: {model_name!r}"
        )
    size = value.split(":", 1)[1]
    if size not in _VALID_SIZES:
        raise CrisperWhisperBackendError(
            f"CrisperWhisper size must be one of {sorted(_VALID_SIZES)}"
        )
    return size


def _candidate_model_directories(
    model_name: str,
    model_root: Path,
    configured_path: str,
) -> tuple[Path, ...]:
    size = _size_from_model_name(model_name)
    candidates: list[Path] = []
    if configured_path.strip():
        configured = Path(os.path.expandvars(os.path.expanduser(configured_path)))
        candidates.extend(
            (
                configured,
                configured / f"CrisperWhisper2.0_{size}",
                configured / size,
            )
        )
    candidates.extend(
        (
            model_root / f"CrisperWhisper2.0_{size}",
            model_root / size,
        )
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False)).casefold()
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return tuple(unique)


def resolve_local_model_directory(
    model_name: str,
    *,
    model_root: Path,
    configured_path: str = "",
) -> Path:
    """Resolve a model directory without contacting the Hub."""
    candidates = _candidate_model_directories(model_name, model_root, configured_path)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise CrisperWhisperBackendError(
        "CrisperWhisper weights are not installed locally. Searched: "
        f"{searched}. Download the selected standard model explicitly, then "
        "set recognition.crisperwhisper_model_path or place it under "
        "%LOCALAPPDATA%/DrewLocalVoice/models/crisperwhisper. No model "
        "fallback or automatic download was attempted."
    )


class CrisperWhisperBackend:
    """Adapter exposing CrisperWhisper 2.0 through GlitchScribe's ASR API."""

    def __init__(
        self,
        *,
        model_name: str,
        model_root: Path,
        configured_path: str = "",
        language: str = "en",
        mode: str = "verbatim",
        hotwords: str = "",
        model_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.model_name = str(model_name)
        self.model_root = Path(model_root)
        self.configured_path = str(configured_path or "")
        self.language = _normalise_text(language) or "en"
        self.mode = str(mode or "verbatim").strip().casefold()
        if self.mode not in {"verbatim", "intended"}:
            raise CrisperWhisperBackendError(
                "recognition.transcription_mode must be verbatim or intended"
            )
        self.hotwords = _normalise_text(hotwords)
        self._model_factory = model_factory
        self._model: Any = None
        self._signature: tuple[str, str, str, Optional[int]] | None = None
        self._model_path: Path | None = None
        self._lock = threading.RLock()

    @property
    def backend(self) -> str:
        return _BACKEND_NAME

    @property
    def model(self) -> Any:
        return self._model

    @property
    def model_path(self) -> Path | None:
        return self._model_path

    def is_loaded(self) -> bool:
        return self._model is not None

    def resource_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "model_path": str(self._model_path) if self._model_path else None,
            "language": self.language,
            "mode": self.mode,
            "hotwords": self.hotwords,
            "loaded": self.is_loaded(),
            "local_only": True,
        }

    def _factory(self) -> Callable[..., Any]:
        if self._model_factory is not None:
            return self._model_factory
        try:
            module = importlib.import_module("crisperwhisper")
            factory = getattr(module, "CrisperWhisperModel")
        except Exception as exc:  # pragma: no cover - environment dependent
            raise CrisperWhisperBackendError(
                "crisperwhisper[transformers] is required for this ASR backend: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._model_factory = factory
        return factory

    @staticmethod
    def _device_for_profile(profile: Any) -> tuple[str, Optional[int]]:
        device = str(getattr(profile, "device", "cpu") or "cpu").strip().casefold()
        if device == "cuda":
            index = getattr(profile, "device_index", None)
            if index is None:
                raise CrisperWhisperBackendError(
                    "CrisperWhisper CUDA profiles require an explicit device index"
                )
            return "cuda", int(index)
        if device == "cpu":
            return "cpu", None
        raise CrisperWhisperBackendError(
            f"CrisperWhisper supports only explicit cpu/cuda profiles, got {device!r}"
        )

    @staticmethod
    def _compute_type(profile: Any, device: str) -> str:
        requested = str(getattr(profile, "compute_type", "float32") or "float32")
        if device == "cpu":
            return "float32"
        if requested in {"float16", "int8_float16", "int8", "float32"}:
            return "float16" if requested in {"int8", "int8_float16"} else requested
        raise CrisperWhisperBackendError(
            f"Unsupported CrisperWhisper compute type: {requested!r}"
        )

    def _load_profile(self, profile: Any) -> float:
        model_path = resolve_local_model_directory(
            self.model_name,
            model_root=self.model_root,
            configured_path=self.configured_path,
        )
        device, device_index = self._device_for_profile(profile)
        compute_type = self._compute_type(profile, device)
        signature = (str(model_path), device, compute_type, device_index)
        if self._model is not None and signature == self._signature:
            self._model_path = model_path
            return 0.0
        if self._model is not None:
            self.unload()
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "backend": "transformers",
            "compute_type": compute_type,
            "device": device,
        }
        if device_index is not None:
            kwargs["device_index"] = device_index
        try:
            self._model = self._factory()(str(model_path), **kwargs)
        except Exception as exc:
            self._model = None
            self._signature = None
            raise CrisperWhisperBackendError(
                f"Failed to load local CrisperWhisper model {model_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._model_path = model_path
        self._signature = signature
        return time.perf_counter() - started

    def preload_and_warm(self, profile: Any, *, run_inference: bool = False) -> dict[str, Any]:
        with self._lock:
            load_seconds = self._load_profile(profile)
            warm_seconds = 0.0
            if run_inference:
                started = time.perf_counter()
                self.transcribe_samples(
                    np.zeros(int(SAMPLE_RATE * 1.5), dtype=np.float32),
                    profile,
                    fast=True,
                )
                warm_seconds = time.perf_counter() - started
            return {
                "ready": True,
                "profile": getattr(profile, "label", ""),
                "load_seconds": float(load_seconds),
                "warm_seconds": float(warm_seconds),
                "backend": self.backend,
                "model_path": str(self._model_path),
                "mode": self.mode,
            }

    def _transcribe_unlocked(self, audio: np.ndarray, profile: Any, fast: bool) -> tuple[str, dict]:
        load_seconds = self._load_profile(profile)
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "language": self.language,
            "mode": self.mode,
            "sr": SAMPLE_RATE,
            "longform_strategy": "continuation",
            "temperature_fallback": True,
            "hallucination_mitigation": True,
            "early_eot_recovery": True,
            "word_timestamps": True,
            "max_new_tokens": 256 if fast else 384,
        }
        if self.hotwords:
            kwargs["hotwords"] = [item.strip() for item in self.hotwords.split(",") if item.strip()]
        try:
            result = self._model.transcribe(audio, **kwargs)
        except Exception as exc:
            raise CrisperWhisperBackendError(
                f"CrisperWhisper transcription failed: {type(exc).__name__}: {exc}"
            ) from exc
        words = tuple(
            {
                "word": str(getattr(word, "word", "")),
                "start": float(getattr(word, "start", 0.0) or 0.0),
                "end": float(getattr(word, "end", 0.0) or 0.0),
                "probability": float(getattr(word, "probability", 0.0) or 0.0),
            }
            for word in (getattr(result, "words", None) or ())
            if str(getattr(word, "word", "")).strip()
            and getattr(word, "start", None) is not None
            and getattr(word, "end", None) is not None
        )
        elapsed = time.perf_counter() - started
        text = _normalise_text(getattr(result, "text", ""))
        return text, {
            "load_seconds": float(load_seconds),
            "transcribe_seconds": float(elapsed),
            "words": words,
            "language": getattr(result, "language", self.language),
            "profile": getattr(profile, "label", ""),
            "backend": self.backend,
            "mode": self.mode,
        }

    def transcribe_samples(
        self,
        samples: np.ndarray,
        profile: Any,
        fast: bool = True,
    ) -> tuple[str, dict]:
        audio = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
        with self._lock:
            return self._transcribe_unlocked(audio, profile, fast)

    def transcribe(self, wav_path: Path, model_name: str, device: str, compute_type: str) -> tuple[str, dict]:
        import soundfile as sf

        audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        if int(sample_rate) != SAMPLE_RATE:
            raise CrisperWhisperBackendError(
                f"Expected {SAMPLE_RATE} Hz audio, received {sample_rate} Hz"
            )
        profile = type("Profile", (), {
            "model_name": model_name,
            "device": device,
            "device_index": None,
            "compute_type": compute_type,
            "label": model_name,
        })()
        return self.transcribe_samples(np.asarray(audio, dtype=np.float32), profile, fast=False)

    def unload(self) -> bool:
        with self._lock:
            if self._model is None:
                return False
            model = self._model
            self._model = None
            self._signature = None
            del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return True
