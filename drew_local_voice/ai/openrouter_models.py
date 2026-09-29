"""OpenRouter model catalog and free-route selection helpers.

This module is deliberately UI- and provider-agnostic. It parses the public
OpenRouter Models API, filters to text-capable zero-price models, and exposes a
conservative Fastest Free policy. It never handles or logs API credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Callable, Iterable, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_AUTO_MODEL_ID = "openrouter/auto"
OPENROUTER_AUTO_LABEL = "OpenRouter Auto Router (task/cost/quality)"
FASTEST_FREE_MODEL_ID = "__fastest_free__"
FASTEST_FREE_LABEL = "Auto-select fastest free (measured or metadata-ranked)"
MAX_CATALOG_BYTES = 5_000_000

# Family hints are only used to make a compact, human-useful dropdown from the
# live public catalog. They are not claims about benchmark quality or latency.
_SOLID_FAMILY_HINTS: tuple[str, ...] = (
    "openai/gpt-5.6-luna",
    "openai/gpt-5.4-mini",
    "anthropic/claude-sonnet",
    "google/gemini-3.6-flash",
    "google/gemini-3.5-flash",
    "qwen/qwen3.7-flash",
    "qwen/qwen3.5-35b",
    "moonshotai/kimi-k2.7",
    "minimax/minimax-m2.7",
    "mistralai/mistral-medium",
    "nvidia/nemotron-3-super",
    "openai/gpt-oss-20b",
    "poolside/laguna-s-2.1",
)


class ModelCatalogError(RuntimeError):
    """A bounded, user-displayable model-catalog failure."""


@dataclass(frozen=True)
class OpenRouterModelInfo:
    model_id: str
    name: str
    context_length: int
    max_completion_tokens: Optional[int]
    prompt_price: str
    completion_price: str
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    supported_parameters: tuple[str, ...]
    created: Optional[int] = None

    @property
    def is_free(self) -> bool:
        return _is_zero_price(self.prompt_price) and _is_zero_price(self.completion_price)

    @property
    def is_free_variant(self) -> bool:
        return self.model_id.endswith(":free")

    @property
    def is_text_capable(self) -> bool:
        return "text" in self.input_modalities and "text" in self.output_modalities

    @property
    def supports_generation(self) -> bool:
        return "max_tokens" in self.supported_parameters

    @property
    def display_name(self) -> str:
        label = self.name.strip() or self.model_id
        return f"{label}  [{self.model_id}]"

    @property
    def metadata_sort_key(self) -> tuple:
        # This is an eligibility/routing tie-breaker, not a latency claim.
        # Prefer text-only generation with a normal max-token control and a
        # sufficiently large context. Stable ID ordering makes the fallback
        # deterministic when OpenRouter metadata is incomplete.
        return (
            0 if self.supports_generation else 1,
            0 if self.input_modalities == ("text",) else 1,
            -min(max(self.context_length, 0), 1_000_000),
            -(self.max_completion_tokens or 0),
            self.model_id.casefold(),
        )


@dataclass(frozen=True)
class ModelSelection:
    model: OpenRouterModelInfo
    route: str
    reason: str
    measured_latency_seconds: Optional[float] = None


def _is_zero_price(value: object) -> bool:
    try:
        return math.isclose(float(str(value or "0").strip()), 0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def _as_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(item).strip().lower() for item in value if str(item).strip()))


def _optional_int(value: object) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def parse_model_catalog(payload: Mapping[str, object], *, max_models: int = 1_000) -> tuple[OpenRouterModelInfo, ...]:
    """Parse the public Models API envelope without trusting arbitrary fields."""
    entries = payload.get("data")
    if not isinstance(entries, list):
        raise ModelCatalogError("OpenRouter model catalog returned invalid data")
    models: list[OpenRouterModelInfo] = []
    seen: set[str] = set()
    for entry in entries[:max_models]:
        if not isinstance(entry, Mapping):
            continue
        model_id = str(entry.get("id", "")).strip()
        if not model_id or model_id in seen:
            continue
        architecture = entry.get("architecture")
        architecture = architecture if isinstance(architecture, Mapping) else {}
        pricing = entry.get("pricing")
        pricing = pricing if isinstance(pricing, Mapping) else {}
        top_provider = entry.get("top_provider")
        top_provider = top_provider if isinstance(top_provider, Mapping) else {}
        context_length = _optional_int(entry.get("context_length")) or 0
        provider_context = _optional_int(top_provider.get("context_length"))
        models.append(
            OpenRouterModelInfo(
                model_id=model_id,
                name=str(entry.get("name", "")).strip(),
                context_length=provider_context or context_length,
                max_completion_tokens=_optional_int(top_provider.get("max_completion_tokens")),
                prompt_price=str(pricing.get("prompt", "")),
                completion_price=str(pricing.get("completion", "")),
                input_modalities=_as_tuple(
                    architecture.get("input_modalities")
                ) or _as_tuple(entry.get("input_modalities")),
                output_modalities=_as_tuple(
                    architecture.get("output_modalities")
                ) or _as_tuple(entry.get("output_modalities")),
                supported_parameters=_as_tuple(entry.get("supported_parameters")),
                created=_optional_int(entry.get("created")),
            )
        )
        seen.add(model_id)
    if not models:
        raise ModelCatalogError("OpenRouter model catalog contained no usable models")
    return tuple(models)


def eligible_free_text_models(
    models: Iterable[OpenRouterModelInfo],
    *,
    minimum_context_length: int = 8_192,
    minimum_output_tokens: int = 256,
) -> tuple[OpenRouterModelInfo, ...]:
    """Return zero-price text generation models suitable for transformations."""
    result = []
    for model in models:
        if not model.is_free_variant or not model.is_free or not model.is_text_capable or not model.supports_generation:
            continue
        if model.context_length and model.context_length < minimum_context_length:
            continue
        if model.max_completion_tokens is not None and model.max_completion_tokens < minimum_output_tokens:
            continue
        result.append(model)
    return tuple(sorted(result, key=lambda item: item.metadata_sort_key))


def solid_text_model_options(
    models: Iterable[OpenRouterModelInfo],
    *,
    max_models: int = 16,
) -> tuple[OpenRouterModelInfo, ...]:
    """Return a compact list of current, explicit text-generation choices.

    The list is derived from the live catalog, excludes batch/image/safety-only
    variants, and prefers recognizable model families. It is a presentation
    helper, not a quality or latency benchmark.
    """
    candidates = []
    for model in models:
        lowered = model.model_id.casefold()
        if not model.is_text_capable or not model.supports_generation:
            continue
        if model.context_length and model.context_length < 8_192:
            continue
        if any(
            token in lowered
            for token in (":batch", "-image", "content-safety", "openrouter/free")
        ):
            continue
        if "preview" in lowered and not lowered.endswith(":free"):
            continue
        candidates.append(model)
    chosen: list[OpenRouterModelInfo] = []
    seen: set[str] = set()
    for hint in _SOLID_FAMILY_HINTS:
        matches = sorted(
            (
                model
                for model in candidates
                if model.model_id.casefold().startswith(hint)
            ),
            key=lambda model: (model.model_id.casefold().count("-pro"), model.model_id),
        )
        for model in matches[:1]:
            if model.model_id not in seen:
                chosen.append(model)
                seen.add(model.model_id)
    for model in sorted(
        candidates,
        key=lambda item: (1 if item.is_free else 0, item.metadata_sort_key),
    ):
        if model.model_id not in seen:
            chosen.append(model)
            seen.add(model.model_id)
        if len(chosen) >= max(1, int(max_models)):
            break
    return tuple(chosen[: max(1, int(max_models))])


def select_fastest_free(
    models: Iterable[OpenRouterModelInfo],
    *,
    measured_latencies: Optional[Mapping[str, float]] = None,
) -> ModelSelection:
    """Select a free text model using measurements when available.

    A metadata-ranked result is explicitly labeled as such. It is not described
    as globally fastest because public model metadata does not prove local
    request latency.
    """
    eligible = eligible_free_text_models(models)
    if not eligible:
        raise ModelCatalogError("OpenRouter advertised no eligible free text models")
    measured: dict[str, float] = {}
    for model_id, seconds in (measured_latencies or {}).items():
        try:
            rendered = float(seconds)
        except (TypeError, ValueError):
            continue
        if math.isfinite(rendered) and rendered > 0.0:
            measured[str(model_id)] = rendered
    measured_candidates = [model for model in eligible if model.model_id in measured]
    if measured_candidates:
        selected = min(
            measured_candidates,
            key=lambda model: (measured[model.model_id], model.metadata_sort_key),
        )
        return ModelSelection(
            model=selected,
            route="measured",
            reason="fastest successful bounded probe among eligible free text models",
            measured_latency_seconds=measured[selected.model_id],
        )
    selected = eligible[0]
    return ModelSelection(
        model=selected,
        route="metadata-ranked",
        reason="no recent successful probe; deterministic eligibility and metadata ranking",
    )


def probe_model_latencies(
    models: Iterable[OpenRouterModelInfo],
    *,
    api_key: str,
    endpoint: str = "https://openrouter.ai/api/v1",
    timeout_seconds: float = 20.0,
    max_candidates: int = 3,
    opener: Optional[Callable] = None,
) -> dict[str, float]:
    """Measure bounded completion latency for a few eligible free candidates.

    This is intentionally opt-in and never called by import or application
    startup. It sends only a synthetic speed probe, never selected text.
    """
    key = str(api_key or "").strip()
    if not key:
        return {}
    parsed = urlparse(str(endpoint).strip().rstrip("/"))
    if parsed.scheme != "https" or parsed.hostname != "openrouter.ai":
        raise ModelCatalogError("OpenRouter probe endpoint is invalid")
    candidates = list(eligible_free_text_models(models))
    if not candidates:
        return {}
    preferred_tokens = ("tiny", "mini", "26b", "31b", "xs")

    def probe_key(model: OpenRouterModelInfo) -> tuple:
        lowered = model.model_id.casefold()
        specialized = int("content-safety" in lowered or "code" in lowered)
        preferred = 0 if any(token in lowered for token in preferred_tokens) else 1
        return (specialized, preferred, model.metadata_sort_key)

    candidates.sort(key=probe_key)
    candidates = candidates[: max(1, min(int(max_candidates), 3))]
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://drewlocalvoice.local",
        "X-Title": "GlitchScribe",
    }
    open_url = opener or build_opener(HTTPRedirectHandler()).open
    timeout = max(3.0, min(float(timeout_seconds), 20.0))
    latencies: dict[str, float] = {}
    for model in candidates:
        payload = {
            "model": model.model_id,
            "messages": [
                {"role": "user", "content": "Reply with the single word OK."}
            ],
            "max_tokens": 8,
            "stream": False,
        }
        request = Request(
            f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            response = open_url(request, timeout=timeout)
            with response:
                raw = response.read(256_001)
            if len(raw) > 256_000:
                continue
            decoded = json.loads(raw.decode("utf-8"))
            text = str(decoded["choices"][0]["message"]["content"]).strip()
            if not text:
                continue
        except (HTTPError, URLError, OSError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
        latencies[model.model_id] = max(0.001, time.perf_counter() - started)
    return latencies


def fetch_model_catalog(
    *,
    api_key: Optional[str] = None,
    timeout_seconds: float = 10.0,
    opener: Optional[Callable] = None,
) -> tuple[OpenRouterModelInfo, ...]:
    """Fetch the public model catalog; credentials are used only as a header."""
    parsed = urlparse(OPENROUTER_MODELS_URL)
    if parsed.scheme != "https" or parsed.hostname != "openrouter.ai":
        raise ModelCatalogError("OpenRouter model endpoint is invalid")
    headers = {
        "Accept": "application/json",
        "User-Agent": "GlitchScribe/ai-model-catalog",
    }
    key = str(api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(OPENROUTER_MODELS_URL, headers=headers, method="GET")
    open_url = opener or build_opener(HTTPRedirectHandler()).open
    timeout = max(2.0, min(float(timeout_seconds), 30.0))
    try:
        response = open_url(request, timeout=timeout)
        with response:
            raw = response.read(MAX_CATALOG_BYTES + 1)
    except HTTPError as exc:
        raise ModelCatalogError(f"OpenRouter model discovery failed: HTTP {int(exc.code)}") from exc
    except (URLError, OSError, TimeoutError) as exc:
        raise ModelCatalogError(
            f"OpenRouter model discovery failed: {type(exc).__name__}"
        ) from exc
    if len(raw) > MAX_CATALOG_BYTES:
        raise ModelCatalogError("OpenRouter model catalog exceeded the transport limit")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelCatalogError("OpenRouter model catalog was not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ModelCatalogError("OpenRouter model catalog returned an invalid envelope")
    return parse_model_catalog(decoded)


__all__ = [
    "FASTEST_FREE_LABEL",
    "FASTEST_FREE_MODEL_ID",
    "OPENROUTER_AUTO_LABEL",
    "OPENROUTER_AUTO_MODEL_ID",
    "MAX_CATALOG_BYTES",
    "OPENROUTER_MODELS_URL",
    "ModelCatalogError",
    "ModelSelection",
    "OpenRouterModelInfo",
    "eligible_free_text_models",
    "fetch_model_catalog",
    "parse_model_catalog",
    "probe_model_latencies",
    "select_fastest_free",
    "solid_text_model_options",
]
