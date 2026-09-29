"""Structured task outputs with Pydantic validation + IAL circuit breakers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar, Generic
from pydantic import BaseModel, ValidationError, create_model

from drew_local_voice.ai.providers import AIRequest, AIResponse


# ---------------------------------------------------------------------------
# Task output modes
# ---------------------------------------------------------------------------


class OutputMode(str, Enum):
    RAW = "raw"           # plain text
    JSON_DICT = "json"   # parseable dict
    PYDANTIC = "pydantic"  # Pydantic-validated instance


@dataclass
class TaskOutput:
    """Result of a task executed by an agent."""

    agent_id: str
    task_description: str
    mode: OutputMode = OutputMode.RAW
    raw_text: str = ""
    parsed: dict | None = None
    instance: Any = None  # Pydantic model instance
    success: bool = False
    truncated: bool = False
    errors: tuple[str, ...] = ()
    iterations: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.errors, list):
            object.__setattr__(self, "errors", tuple(self.errors))

    @property
    def text(self) -> str:
        """Best-effort text representation."""
        if self.instance is not None:
            return self.instance.model_dump_json(indent=2)
        if self.parsed is not None:
            import json
            return json.dumps(self.parsed, indent=2)
        return self.raw_text

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "task_description": self.task_description,
            "mode": self.mode.value,
            "text": self.text,
            "parsed": self.parsed,
            "success": self.success,
            "truncated": self.truncated,
            "errors": list(self.errors),
            "iterations": self.iterations,
        }


# ---------------------------------------------------------------------------
# Structured output builder
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Failed to produce a structured output."""
    pass


class OutputSchema:
    """Schema descriptor for structured task output.

    Wraps a Pydantic model and provides validation + retry logic.
    """

    def __init__(self, model: type[BaseModel] | None = None, json_keys: list[str] | None = None) -> None:
        self._model = model
        self._json_keys = json_keys or []

    @property
    def mode(self) -> OutputMode:
        if self._model is not None:
            return OutputMode.PYDANTIC
        if self._json_keys:
            return OutputMode.JSON_DICT
        return OutputMode.RAW

    def parse(self, text: str) -> TaskOutput:
        """Parse text according to the schema.

        Returns a TaskOutput with parsed/instance populated on success.
        Raises StructuredOutputError on validation failure.
        """
        import json
        import re

        output = TaskOutput(agent_id="", task_description="", mode=self.mode, raw_text=text)

        if self.mode == OutputMode.RAW:
            output.success = True
            return output

        if self.mode == OutputMode.JSON_DICT:
            # Extract JSON from text (handle markdown code blocks)
            json_text = text
            code_match = re.search(r"```(?:json)?\s*({\s*.*?\s*})\s*```", text, re.DOTALL)
            if code_match:
                json_text = code_match.group(1)
            else:
                # Try to find the first { ... } block
                brace_start = text.find("{")
                brace_end = text.rfind("}")
                if brace_start >= 0 and brace_end > brace_start:
                    json_text = text[brace_start:brace_end + 1]

            try:
                data = json.loads(json_text)
                # Validate required keys
                missing = [k for k in self._json_keys if k not in data]
                if missing:
                    raise StructuredOutputError(f"missing keys: {missing}")
                output.parsed = data
                output.success = True
            except (json.JSONDecodeError, StructuredOutputError) as exc:
                output.errors = (f"JSON parse error: {exc}",)
                raise StructuredOutputError(f"failed to parse JSON: {exc}") from exc
            return output

        if self._model is not None:
            # Try direct parse
            model_text = text
            code_match = re.search(r"```(?:json)?\s*({.*})\s*```", text, re.DOTALL)
            if code_match:
                model_text = code_match.group(1)

            try:
                output.instance = self._model.model_validate_json(model_text)
                output.success = True
            except (json.JSONDecodeError, ValidationError) as exc:
                output.errors = (f"Pydantic validation error: {exc}",)
                raise StructuredOutputError(f"validation failed: {exc}") from exc
            return output

        return output


def output_pydantic(model: type[BaseModel]) -> OutputSchema:
    """Decorator-style helper to create a Pydantic output schema."""
    return OutputSchema(model=model)


def output_json(*required_keys: str) -> OutputSchema:
    """Decorator-style helper to create a JSON dict output schema with required keys."""
    return OutputSchema(json_keys=list(required_keys))


# ---------------------------------------------------------------------------
# IAL (Iterative Action Loop) Circuit Breaker
# ---------------------------------------------------------------------------


@dataclass
class IALBreakerConfig:
    """Configuration for the IAL circuit breaker."""

    max_iterations: int = 5
    convergence_window: int = 3        # look back this many iterations
    drift_threshold: float = 0.15     # output similarity below this = divergence
    max_retries: int = 2
    retry_delay_seconds: float = 0.5


class IALBreaker:
    """Circuit breaker that halts delegation loops when output stops improving.

    Uses n-gram Jaccard similarity to detect semantic loops. When similarity
    between consecutive outputs drops below the drift threshold, the loop is
    considered divergent and halted.
    """

    def __init__(self, config: IALBreakerConfig | None = None) -> None:
        self._config = config or IALBreakerConfig()
        self._iter_count: int = 0
        self._outputs: list[str] = []
        self._tripped: bool = False
        self._trip_reason: str = ""

    def record(self, output: str) -> bool:
        """Record one iteration's output.

        Returns True if the loop should continue, False if the breaker has tripped.
        """
        self._iter_count += 1
        self._outputs.append(output[:2000])  # cap stored output

        # Check iteration cap
        if self._iter_count >= self._config.max_iterations:
            self._trip("max_iterations", f"reached {self._iter_count} iterations (max {self._config.max_iterations})")
            return False

        # Check convergence (output similarity)
        if self._iter_count >= self._config.convergence_window + 1:
            recent = self._outputs[-(self._config.convergence_window + 1):]
            similarity = self._jaccard_similarity(recent[-1], recent[-2])
            if similarity < self._config.drift_threshold:
                self._trip("divergence", f"output similarity {similarity:.2f} below threshold {self._config.drift_threshold}")
                return False

        return True

    def _trip(self, reason: str, detail: str) -> None:
        self._tripped = True
        self._trip_reason = f"{reason}: {detail}"

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    @property
    def iter_count(self) -> int:
        return self._iter_count

    def reset(self) -> None:
        self._iter_count = 0
        self._outputs.clear()
        self._tripped = False
        self._trip_reason = ""

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """N-gram (trigram) Jaccard similarity between two strings."""
        def ngrams(s: str, n: int = 3) -> set[str]:
            s = s.lower()
            return set(s[i:i + n] for i in range(max(1, len(s) - n + 1)))

        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        grams_a = ngrams(a)
        grams_b = ngrams(b)
        if not grams_a or not grams_b:
            return 0.0
        intersection = len(grams_a & grams_b)
        union = len(grams_a | grams_b)
        return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Retry loop with structured output
# ---------------------------------------------------------------------------


@dataclass
class RetryResult:
    """Result of a retry loop."""

    output: str
    success: bool
    attempts: int
    final_error: str = ""


def run_with_retry(
    make_request: callable,
    schema: OutputSchema,
    *,
    max_retries: int = 2,
    retry_delay: float = 0.5,
) -> RetryResult:
    """Call make_request() up to max_retries times, parsing output against schema each time.

    make_request: callable that returns raw text string
    schema: OutputSchema to validate against

    Returns RetryResult with best attempt on final failure.
    """
    import time

    last_error = ""
    last_output = ""

    for attempt in range(max_retries + 1):
        try:
            raw = make_request()
            last_output = raw
            result = schema.parse(raw)
            return RetryResult(output=raw, success=True, attempts=attempt + 1)
        except Exception as exc:  # broad: schema parse error or request error
            last_error = str(exc)[:200]
            if attempt < max_retries:
                time.sleep(retry_delay * (attempt + 1))

    return RetryResult(
        output=last_output,
        success=False,
        attempts=max_retries + 1,
        final_error=last_error,
    )
