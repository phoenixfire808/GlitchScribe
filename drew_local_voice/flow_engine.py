"""Homegrown event-driven Flow engine — no CrewAI dependency.

Implements @start, @listen, @router decorators with lazy FlowDefinition compilation,
Pydantic state, UUID-tracked identity, and async kickoff entry points.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    ClassVar,
    FrozenSet,
    TypeVar,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Public decorators
# ---------------------------------------------------------------------------


def start(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a method as the entry point of a Flow.

    A Flow must have exactly one @start method. It is invoked first when
    kickoff() or kickoff_async() is called.
    """
    func.__flow_role__ = "start"  # type: ignore[attr-defined]
    return func


def listen(trigger: str | Callable[..., Any]) -> Callable[..., Any]:
    """Mark a method as listening for the completion of another method.

    Usage:
        @listen("method_name")   — listen for a specific named method
        @listen(other_method)     — listen for another decorated method
    """
    trigger_name: str
    if callable(trigger):
        trigger_name = trigger.__name__
    else:
        trigger_name = str(trigger)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func.__flow_role__ = "listen"  # type: ignore[attr-defined]
        func.__flow_trigger__ = trigger_name  # type: ignore[attr-defined]
        return func

    return decorator


def router(func: Callable[..., str]) -> Callable[..., Any]:
    """Mark a method as a router — returns the name of the next branch to execute.

    The return value must be the name of another @listen method.
    """
    func.__flow_role__ = "router"  # type: ignore[attr-defined]
    return func


# ---------------------------------------------------------------------------
# Flow definition (compiled once, lazy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowDefinition:
    """Immutable snapshot of a Flow class's structure — compiled once on first access."""

    start_method: str
    listen_map: dict[str, str]  # method_name -> trigger_name
    router_methods: FrozenSet[str]
    all_methods: FrozenSet[str]


class FlowMeta(type):
    """Metaclass that collects @start/@listen/@router metadata into FlowDefinition."""

    _definitions: dict[type, FlowDefinition] = {}

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> "FlowMeta":
        cls = super().__new__(mcs, name, bases, namespace)

        # Collect decorated methods
        start_method: str | None = None
        listen_map: dict[str, str] = {}
        router_methods: set[str] = set()
        all_methods: set[str] = set()

        for attr_name, attr_value in namespace.items():
            if not callable(attr_value):
                continue
            role = getattr(attr_value, "__flow_role__", None)
            if role == "start":
                if start_method is not None:
                    raise TypeError(f"Flow {name} has multiple @start methods: {start_method}, {attr_name}")
                start_method = attr_name
            elif role == "listen":
                trigger = getattr(attr_value, "__flow_trigger__", None)
                if not trigger:
                    raise TypeError(f"@listen method {attr_name} has no trigger")
                listen_map[attr_name] = trigger
            elif role == "router":
                router_methods.add(attr_name)

        # Only abstract Flows (those with __flow_role__ set) get a definition
        if start_method or listen_map or router_methods:
            if start_method is None:
                raise TypeError(f"Flow {name} has @listen/@router but no @start method")
            all_methods = {n for n in listen_map} | router_methods | {start_method}
            mcs._definitions[cls] = FlowDefinition(
                start_method=start_method,
                listen_map=dict(listen_map),
                router_methods=frozenset(router_methods),
                all_methods=frozenset(all_methods),
            )

        return cls

    @classmethod
    def get_definition(mcs, cls: type) -> FlowDefinition | None:
        return mcs._definitions.get(cls)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class FlowEvent(str, Enum):
    STARTED = "flow_started"
    STEP_STARTED = "flow_step_started"
    STEP_COMPLETED = "flow_step_completed"
    ROUTED = "flow_routed"
    STATE_UPDATED = "flow_state_updated"
    FINISHED = "flow_finished"
    ERROR = "flow_error"


@dataclass(frozen=True)
class FlowStartedEvent:
    flow_id: str
    state_type: str


@dataclass(frozen=True)
class FlowStepEvent:
    flow_id: str
    method: str
    status: str  # started | completed | error
    duration_ms: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class FlowState:
    """Base class for flow-specific state.

    Subclass with Pydantic-style fields or plain dataclass fields.
    Automatically assigned a UUID that persists across updates.
    """

    _flow_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    errors: list[str] = field(default_factory=list)
    _completion_pct: float = field(default=0.0, init=False)

    @property
    def flow_id(self) -> str:
        return self._flow_id

    def update_progress(self, completed: int, total: int) -> float:
        """Update completion percentage. Returns the new percentage (0.0–100.0)."""
        if total <= 0:
            return 0.0
        self._completion_pct = min(100.0, (completed / total) * 100.0)
        return self._completion_pct

    @property
    def completion_pct(self) -> float:
        return self._completion_pct

    def add_error(self, error: str) -> None:
        self.errors.append(str(error)[:500])

    def to_dict(self) -> dict:
        result: dict[str, Any] = {"_flow_id": self._flow_id, "errors": list(self.errors), "_completion_pct": self._completion_pct}
        for key, val in self.__dict__.items():
            if key.startswith("_") and key not in ("_flow_id", "_completion_pct"):
                continue
            if key in ("errors", "_flow_id", "_completion_pct"):
                continue
            result[key] = val
        return result


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=FlowState)


class Flow(metaclass=FlowMeta):
    """Event-driven workflow base class.

    Subclass with @start, @listen, and @router decorated methods.

    Usage:
        class MyFlow(Flow[MyState]):
            @start
            def begin(self) -> MyState:
                return MyState(counter=0)

            @listen("begin")
            def process(self) -> None:
                self.state.counter += 1

            @router
            def route(self) -> str:
                if self.state.counter < 3:
                    return "process"
                return "_done"

        flow = MyFlow()
        result = flow.kickoff()
    """

    _state: FlowState | None = None
    _events: list[FlowEvent] = field(default_factory=list)
    _event_handlers: dict[FlowEvent, list[Callable[..., None]]] = field(default_factory=dict)
    _definition: FlowDefinition | None = field(default=None, init=False)

    # Subclasses MUST define a state class via the generic parameter
    STATE_CLASS: ClassVar[type[FlowState] | None] = None

    def __init__(self) -> None:
        self._state = None
        self._events = []
        self._event_handlers = {}
        self._definition = None

    # -------------------------------------------------------------------------
    # State
    # -------------------------------------------------------------------------

    @property
    def state(self) -> FlowState:
        if self._state is None:
            raise RuntimeError("flow state not initialized — call kickoff() first")
        return self._state

    def _init_state(self, input_data: dict[str, Any] | None = None) -> None:
        """Initialize state before kickoff."""
        state_cls = self.STATE_CLASS or FlowState
        if is_dataclass(state_cls):
            import dataclasses

            # Build kwargs from input_data, handling partial init
            kwargs: dict[str, Any] = {}
            for f in dataclasses.fields(state_cls):
                if f.name in ("_flow_id", "errors", "_completion_pct"):
                    continue
                if input_data and f.name in input_data:
                    kwargs[f.name] = input_data[f.name]
            self._state = state_cls(**kwargs)
        else:
            self._state = state_cls()  # type: ignore[assignment]

    # -------------------------------------------------------------------------
    # Event system
    # -------------------------------------------------------------------------

    def on(self, event: FlowEvent, handler: Callable[..., None]) -> Callable[..., None]:
        """Register a handler for a flow event. Returns the handler for convenience."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
        return handler

    def _emit(self, event: FlowEvent, **kwargs: Any) -> None:
        self._events.append(event)
        for handler in self._event_handlers.get(event, []):
            try:
                handler(**kwargs)
            except Exception:
                pass  # never let event handlers crash the flow

    # -------------------------------------------------------------------------
    # Definition (lazy)
    # -------------------------------------------------------------------------

    @property
    def definition(self) -> FlowDefinition:
        if self._definition is None:
            self._definition = FlowMeta.get_definition(type(self))
            if self._definition is None:
                raise RuntimeError(
                    f"Flow {type(self).__name__} has no flow definition — "
                    "add @start, @listen, or @router methods"
                )
        return self._definition

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def kickoff(self, **input_data: Any) -> FlowState:
        """Synchronous entry point. Runs the flow in the current thread."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.kickoff_async(**input_data))
        finally:
            loop.close()

    async def kickoff_async(self, **input_data: Any) -> FlowState:
        """Async entry point. Restores prior state, merges new input, dispatches @start."""
        self._init_state(input_data)

        self._emit(
            FlowEvent.STARTED,
            flow_id=self.state.flow_id,
            state_type=type(self.state).__name__,
        )

        # Run the start method
        start_name = self.definition.start_method
        start_method = getattr(self, start_name)
        result = start_method()

        # If the start method returned state, use it
        if result is not None and isinstance(result, FlowState):
            self._state = result
        elif result is not None:
            # Allow start to return a dict to update state
            if isinstance(result, dict):
                for k, v in result.items():
                    if hasattr(self.state, k):
                        setattr(self.state, k, v)

        self._emit(
            FlowEvent.STARTED,
            flow_id=self.state.flow_id,
            state_type=type(self.state).__name__,
        )

        # Build reverse map: trigger_name -> [method_name, ...]
        trigger_to_listeners: dict[str, list[str]] = {}
        for method_name, trigger in self.definition.listen_map.items():
            trigger_to_listeners.setdefault(trigger, []).append(method_name)

        # Dispatch listeners for the start method
        await self._dispatch_listeners(
            trigger_to_listeners.get(start_name, []),
            trigger_to_listeners,
        )

        self._emit(FlowEvent.FINISHED, flow_id=self.state.flow_id)
        return self.state

    async def _dispatch_listeners(
        self,
        listener_names: list[str],
        trigger_to_listeners: dict[str, list[str]],
    ) -> None:
        """Dispatch all listeners for a completed step. Router runs first if present."""
        for name in listener_names:
            if name in self.definition.router_methods:
                # Run router, then dispatch the chosen branch
                router_method = getattr(self, name)
                branch = router_method()
                self._emit(
                    FlowEvent.ROUTED,
                    flow_id=self.state.flow_id,
                    method=name,
                    target=branch,
                )
                # Find listeners for the router's target
                next_listeners = trigger_to_listeners.get(branch, [])
                for next_name in next_listeners:
                    await self._run_step(next_name, trigger_to_listeners)
            else:
                await self._run_step(name, trigger_to_listeners)

    async def _run_step(
        self,
        method_name: str,
        trigger_to_listeners: dict[str, list[str]],
    ) -> None:
        """Execute one flow step."""
        import time

        t0 = time.monotonic()
        self._emit(
            FlowEvent.STEP_STARTED,
            flow_id=self.state.flow_id,
            method=method_name,
            status="started",
        )
        try:
            method = getattr(self, method_name)
            result = method()

            # Support async methods
            if inspect.isawaitable(result):
                result = await result

            self._emit(
                FlowEvent.STEP_COMPLETED,
                flow_id=self.state.flow_id,
                method=method_name,
                status="completed",
                duration_ms=(time.monotonic() - t0) * 1000,
            )

            # Trigger downstream listeners
            downstream = trigger_to_listeners.get(method_name, [])
            if downstream:
                await self._dispatch_listeners(downstream, trigger_to_listeners)

        except Exception as exc:
            self.state.add_error(f"{method_name}: {exc}")
            self._emit(
                FlowEvent.ERROR,
                flow_id=self.state.flow_id,
                method=method_name,
                status="error",
                error=str(exc)[:200],
                duration_ms=(time.monotonic() - t0) * 1000,
            )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def is_flow_subclass(cls: type) -> bool:
    return FlowMeta.get_definition(cls) is not None


def get_flow_definition(cls: type) -> FlowDefinition | None:
    return FlowMeta.get_definition(cls)
