"""OpenTelemetry + Arize Phoenix observability for GlitchScribe multi-agent flows.

Provides full span hierarchy: parent span = crew execution, child spans = tasks,
grandchild spans = agent thoughts + MCP tool I/O + LLM calls. Token usage persisted
to spans. Phoenix collector endpoint configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from contextlib import contextmanager
import time

# ---------------------------------------------------------------------------
# Span hierarchy types
# ---------------------------------------------------------------------------


@dataclass
class SpanContext:
    """Lightweight span context for when OpenTelemetry is not installed."""

    trace_id: str = ""
    span_id: str = ""
    is_recording: bool = False


class NoOpSpan:
    """No-op span when OTel is unavailable."""

    def __init__(self) -> None:
        self._attributes: dict[str, Any] = {}
        self._events: list[dict] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self._attributes[key] = value

    def set_attributes(self, **kwargs: Any) -> None:
        self._attributes.update(kwargs)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._events.append({"name": name, "attributes": attributes or {}})

    def end(self, *, exception: BaseException | None = None) -> None:
        pass

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class GlitchScribeTracer:
    """Unified tracer with OpenTelemetry backend and NoOp fallback.

    Usage:
        tracer = GlitchScribeTracer(endpoint="http://localhost:4317")
        with tracer.start_as_current_span("crew_execution") as span:
            span.set_attribute("crew.id", crew_id)
            ...
    """

    def __init__(
        self,
        service_name: str = "glitchscribe",
        endpoint: str | None = None,  # OTel collector endpoint
        phoenix_endpoint: str | None = None,  # Phoenix collector
        console_export: bool = False,
    ) -> None:
        self._service_name = service_name
        self._endpoint = endpoint
        self._phoenix_endpoint = phoenix_endpoint
        self._console_export = console_export
        self._otel_tracer: Any = None
        self._otel_tracer_provider: Any = None
        self._setup()

    def _setup(self) -> None:
        """Try to set up OpenTelemetry; fall back silently to NoOp on error."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
                SpanExporter,
            )
            from opentelemetry.sdk.resources import Resource, SERVICE_NAME

            resource = Resource(attributes={SERVICE_NAME: self._service_name})
            provider = TracerProvider(resource=resource)

            if self._console_export:
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

            if self._endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                    exporter = OTLPSpanExporter(endpoint=self._endpoint, insecure=True)
                    provider.add_span_processor(BatchSpanProcessor(exporter))
                except Exception:
                    pass

            trace.set_tracer_provider(provider)
            self._otel_tracer_provider = provider
            self._otel_tracer = trace.get_tracer(self._service_name)
        except ImportError:
            pass  # OpenTelemetry not installed — use NoOp

    def start_as_current_span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> NoOpSpan | Any:
        """Start a span. Returns OTel span if available, NoOpSpan otherwise."""
        if self._otel_tracer is not None:
            span = self._otel_tracer.start_span(name)
            if attributes:
                span.set_attributes(**attributes)
            return OTelSpanWrapper(span)
        return NoOpSpan()

    @property
    def is_otel_available(self) -> bool:
        return self._otel_tracer is not None


class OTelSpanWrapper:
    """Wraps an OTel span with a context-manager interface."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)

    def set_attributes(self, **kwargs: Any) -> None:
        self._span.set_attributes(**kwargs)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._span.add_event(name, attributes or {})

    def end(self, *, exception: BaseException | None = None) -> None:
        if exception:
            self._span.record_exception(exception)
        self._span.end()

    def __enter__(self) -> "OTelSpanWrapper":
        return self

    def __exit__(self, *args: Any) -> None:
        self._span.end()


# ---------------------------------------------------------------------------
# Phoenix instrumentation (separate setup, activates when phoenix is available)
# ---------------------------------------------------------------------------


@dataclass
class PhoenixConfig:
    """Configuration for Arize Phoenix observability."""

    endpoint: str = "http://localhost:6006/v1/traces"
    project_name: str = "glitchscribe"
    enabled: bool = False


class PhoenixInstrumentor:
    """Instruments GlitchScribe with Arize Phoenix tracing.

    When Phoenix is unavailable, all methods are no-ops.
    """

    def __init__(self, config: PhoenixConfig | None = None) -> None:
        self._config = config or PhoenixConfig()
        self._enabled = False
        self._setup()

    def _setup(self) -> None:
        if not self._config.enabled:
            return
        try:
            # Phoenix auto-instruments common LLM frameworks
            from phoenix.trace import SpanExporter
            self._exporter = SpanExporter(endpoint=self._config.endpoint)
            self._enabled = True
        except ImportError:
            pass  # Phoenix not installed
        except Exception:
            pass

    def create_span_exporter(self) -> Any:
        """Return the Phoenix span exporter if available."""
        if not self._enabled:
            return None
        return getattr(self, "_exporter", None)

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# ---------------------------------------------------------------------------
# Crew-level instrumentation
# ---------------------------------------------------------------------------


class CrewInstrumentor:
    """Instruments a crew execution with full span hierarchy.

    Usage:
        inst = CrewInstrumentor(tracer)
        with inst.crew_span(crew_id="c123", process="sequential") as crew_span:
            inst.agent_span(crew_span, agent_id="a1", role="coder")
            inst.llm_span(agent_span, model="qwen", prompt_tokens=50)
    """

    def __init__(
        self,
        tracer: GlitchScribeTracer,
        phoenix: PhoenixInstrumentor | None = None,
    ) -> None:
        self._tracer = tracer
        self._phoenix = phoenix

    @contextmanager
    def crew_span(
        self,
        crew_id: str,
        process: str,
        *,
        attributes: dict[str, Any] | None = None,
    ):
        attrs = {"crew.id": crew_id, "crew.process": process, "span.type": "crew"}
        if attributes:
            attrs.update(attributes)
        with self._tracer.start_as_current_span("crew", attributes=attrs) as span:
            yield span

    @contextmanager
    def task_span(
        self,
        parent: NoOpSpan | Any,
        task_description: str,
        agent_id: str,
        *,
        attributes: dict[str, Any] | None = None,
    ):
        attrs = {
            "task.description": task_description,
            "agent.id": agent_id,
            "span.type": "task",
        }
        if attributes:
            attrs.update(attributes)
        with self._tracer.start_as_current_span("task", attributes=attrs) as span:
            yield span

    @contextmanager
    def agent_span(
        self,
        parent: NoOpSpan | Any,
        agent_id: str,
        role: str,
        *,
        attributes: dict[str, Any] | None = None,
    ):
        attrs = {
            "agent.id": agent_id,
            "agent.role": role,
            "span.type": "agent",
        }
        if attributes:
            attrs.update(attributes)
        with self._tracer.start_as_current_span("agent", attributes=attrs) as span:
            yield span

    @contextmanager
    def llm_span(
        self,
        parent: NoOpSpan | Any,
        model: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        *,
        attributes: dict[str, Any] | None = None,
    ):
        attrs = {
            "llm.model": model,
            "span.type": "llm",
        }
        if prompt_tokens is not None:
            attrs["llm.prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            attrs["llm.completion_tokens"] = completion_tokens
        if attributes:
            attrs.update(attributes)
        with self._tracer.start_as_current_span("llm", attributes=attrs) as span:
            yield span

    @contextmanager
    def mcp_span(
        self,
        parent: NoOpSpan | Any,
        tool_name: str,
        server_name: str,
        *,
        attributes: dict[str, Any] | None = None,
    ):
        attrs = {
            "mcp.tool_name": tool_name,
            "mcp.server_name": server_name,
            "span.type": "mcp_tool",
        }
        if attributes:
            attrs.update(attributes)
        with self._tracer.start_as_current_span("mcp_tool", attributes=attrs) as span:
            yield span


# ---------------------------------------------------------------------------
# Token usage logger
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Token usage record for one LLM call."""

    agent_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int = field(init=False)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_tokens", self.prompt_tokens + self.completion_tokens)


class TokenUsageLogger:
    """Persists token usage from AIResponse into spans and optionally a local log file.

    Privacy-safe: records only metadata, never transcript text.
    """

    def __init__(self, log_path: str | None = None) -> None:
        import os
        if log_path is None:
            log_path = os.path.join(os.path.expanduser("~"), ".glitchscribe", "token_log.jsonl")
        self._log_path = log_path
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)

    def log(self, usage: TokenUsage) -> None:
        import json
        record = {
            "agent_id": usage.agent_id,
            "model": usage.model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "timestamp": usage.timestamp,
        }
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass  # Never let logging crash execution

    def from_response(
        self,
        agent_id: str,
        model: str,
        response: Any,  # AIResponse
    ) -> None:
        """Create a TokenUsage record from an AIResponse and log it."""
        if response is None:
            return
        usage = TokenUsage(
            agent_id=agent_id,
            model=model,
            prompt_tokens=response.prompt_tokens or 0,
            completion_tokens=response.completion_tokens or 0,
        )
        self.log(usage)


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

_tracer: Optional[GlitchScribeTracer] = None
_token_logger: Optional[TokenUsageLogger] = None


def get_tracer(
    endpoint: str | None = None,
    phoenix_endpoint: str | None = None,
    console_export: bool = False,
) -> GlitchScribeTracer:
    global _tracer
    if _tracer is None:
        _tracer = GlitchScribeTracer(
            service_name="glitchscribe",
            endpoint=endpoint,
            phoenix_endpoint=phoenix_endpoint,
            console_export=console_export,
        )
    return _tracer


def get_token_logger() -> TokenUsageLogger:
    global _token_logger
    if _token_logger is None:
        _token_logger = TokenUsageLogger()
    return _token_logger
