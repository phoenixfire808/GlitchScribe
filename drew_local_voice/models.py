from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

NEMOTRON_MODEL_ID = "nemotron-3.5-asr-streaming-0.6b"
NEMOTRON_MODEL_LABEL = "Nemotron 3.5 ASR Streaming 0.6B"
NEMOTRON_SOURCE_FILENAME = "nemotron-3.5-asr-streaming-0.6b .safetensors"
CRISPERWHISPER_MODEL_PREFIX = "crisperwhisper:"
CRISPERWHISPER_MODEL_IDS = (
    "crisperwhisper:large",
    "crisperwhisper:turbo",
    "crisperwhisper:medium",
    "crisperwhisper:small",
)
DEFAULT_PRIMARY_MODEL = NEMOTRON_MODEL_ID

BUILT_IN_MODELS = (
    NEMOTRON_MODEL_ID,
    "tiny.en",
    "base.en",
    "small.en",
    "distil-large-v3.5",
    *CRISPERWHISPER_MODEL_IDS,
)


def validate_model_identifier(value: str) -> str:
    identifier = str(value or "").strip()
    if not identifier:
        raise ValueError("Model ID or local model path cannot be empty")
    if len(identifier) > 500:
        raise ValueError("Model ID or local model path is too long")
    if any(ord(character) < 32 for character in identifier):
        raise ValueError("Model ID or local model path contains a control character")
    return identifier


def parse_model_identifiers(values: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        candidates = re.split(r"[,;\n]+", values)
    else:
        candidates = list(values)
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate).strip()
        if not text:
            continue
        identifier = validate_model_identifier(text)
        identity = identifier.casefold()
        if identity not in seen:
            result.append(identifier)
            seen.add(identity)
    return tuple(result)


def configured_model_choices(
    additional_models: str | Iterable[str] = (),
) -> tuple[str, ...]:
    result = list(BUILT_IN_MODELS)
    seen = {item.casefold() for item in result}
    for identifier in parse_model_identifiers(additional_models):
        if identifier.casefold() not in seen:
            result.append(identifier)
            seen.add(identifier.casefold())
    return tuple(result)


def model_backend(model_identifier: str) -> str:
    """Return the only backend allowed to interpret this explicit model ID."""
    identifier = validate_model_identifier(model_identifier)
    normalized = identifier.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1].casefold()
    parent = normalized.rsplit("/", 1)[0].casefold() if "/" in normalized else ""
    if identifier.casefold().startswith(CRISPERWHISPER_MODEL_PREFIX):
        suffix = identifier.casefold().split(":", 1)[1]
        if suffix in {"large", "turbo", "medium", "small"}:
            return "crisperwhisper-transformers"
    if (
        identifier.casefold() == NEMOTRON_MODEL_ID
        or name == NEMOTRON_MODEL_ID
        or name == NEMOTRON_SOURCE_FILENAME.casefold()
        or (
            name == "model.safetensors"
            and NEMOTRON_MODEL_ID in parent
        )
    ):
        return "transformers-rnnt"
    return "faster-whisper"


def model_display_name(model_identifier: str) -> str:
    identifier = validate_model_identifier(model_identifier)
    backend = model_backend(identifier)
    if backend == "transformers-rnnt":
        return NEMOTRON_MODEL_LABEL
    if backend == "crisperwhisper-transformers":
        size = identifier.split(":", 1)[1].replace("_", " ").title()
        return f"CrisperWhisper 2.0 {size}"
    return identifier


def estimated_model_vram_mb(model_identifier: str) -> int:
    identifier = validate_model_identifier(model_identifier)
    backend = model_backend(identifier)
    if backend == "transformers-rnnt":
        # 638M FP32 checkpoint is cast to FP16 on CUDA. This budget includes
        # weights plus a conservative single-stream activation/runtime margin.
        return 2304
    if backend == "crisperwhisper-transformers":
        size = identifier.split(":", 1)[1].casefold()
        return {
            "small": 2048,
            "medium": 3072,
            "turbo": 3072,
            "large": 4096,
        }[size]
    identifier = identifier.casefold()
    name = Path(identifier).name
    if "distil" in name:
        return 3072
    if "large" in name:
        return 4096
    if "medium" in name:
        return 3072
    if "small" in name:
        return 2048
    if "base" in name:
        return 1024
    if "tiny" in name:
        return 512
    return 4096


# Default idle unload (minutes) for Hermes / multi-agent VRAM coexistence.
# Wired via AppConfig.idle_unload_minutes and advanced key model.idle_unload_minutes.
DEFAULT_IDLE_UNLOAD_MINUTES = 10.0
