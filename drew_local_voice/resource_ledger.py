"""Resource ledger — a lean, deterministic, dataclass-only view of resources.

Purpose
-------
This module gives :mod:`local_voice` (and its tests / doctor / overlays) a
small, frozen, *side-effect-free* view of every model, plugin, device,
process, and network policy the application is using. It is the single
resource-ledger contract consumed by the runtime and its tests:

* **One row per concern.** The orchestrator asked for explicit rows for
  the primary ASR, the auxiliary model, plugins, process, and network
  policy. ``ResourceLedger`` returns exactly those rows, in a fixed order,
  so downstream code can render or compare them without re-deriving order.
* **No transcript / audio / clipboard / credential content.** Rows carry
  only identifiers, paths, byte counts, load states, and policy flags.
* **No live side effects at import.** The module never imports Tk,
  :mod:`subprocess`, :mod:`socket`, :mod:`urllib`, or any model runtime.
  Every probe is an injectable callable with a deterministic stub default.
* **Frozen dataclasses.** Rows are immutable; the ledger itself is a
  tuple of rows, not a mutable collection.

Integration contract
--------------------
:func:`build_resource_ledger` accepts the same keyword-only knobs a
caller might already have on hand (an ASR identifier + source, an
auxiliary model spec, a list of plugin records, etc.) plus injectable
probes for anything that touches the host. The default ``probe_enabled``
flag is ``False`` so the function is deterministic at cold start and
from tests.

:func:`evaluate_auxiliary_deny` is a pure-function policy helper. It
returns a :class:`AuxiliaryDenyReason` with a stable ``code`` (string
enum) and a human-readable ``message``. It never raises — failure modes
are returned, not thrown.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Role labels used to tag rows in a :class:`ResourceLedger`.
ROLE_PRIMARY_ASR = "primary_asr"
ROLE_AUXILIARY_MODEL = "auxiliary_model"
ROLE_PLUGIN = "plugin"
ROLE_PROCESS = "process"
ROLE_NETWORK = "network"
ROLE_DEVICE = "device"

#: Known role codes, in the canonical row order.
ROLE_ORDER: Tuple[str, ...] = (
    ROLE_PRIMARY_ASR,
    ROLE_AUXILIARY_MODEL,
    ROLE_PLUGIN,
    ROLE_PROCESS,
    ROLE_NETWORK,
    ROLE_DEVICE,
)

#: Source provenance for a model row.
SOURCE_BUNDLED = "bundled"
SOURCE_HUGGINGFACE = "huggingface"
SOURCE_LOCAL_PATH = "local_path"
SOURCE_LOOPBACK = "loopback"
KNOWN_MODEL_SOURCES: Tuple[str, ...] = (
    SOURCE_BUNDLED,
    SOURCE_HUGGINGFACE,
    SOURCE_LOCAL_PATH,
    SOURCE_LOOPBACK,
)

#: Plugin origins.
PLUGIN_ORIGIN_BUILTIN = "builtin"
PLUGIN_ORIGIN_ENTRY_POINT = "entry_point"
PLUGIN_ORIGIN_USER_DIRECTORY = "user_directory"
KNOWN_PLUGIN_ORIGINS: Tuple[str, ...] = (
    PLUGIN_ORIGIN_BUILTIN,
    PLUGIN_ORIGIN_ENTRY_POINT,
    PLUGIN_ORIGIN_USER_DIRECTORY,
)

#: Network policies.
NETWORK_POLICY_OFFLINE_ONLY = "offline_only"
NETWORK_POLICY_LOOPBACK_ONLY = "loopback_only"
NETWORK_POLICY_ALLOW_REMOTE = "allow_remote"
KNOWN_NETWORK_POLICIES: Tuple[str, ...] = (
    NETWORK_POLICY_OFFLINE_ONLY,
    NETWORK_POLICY_LOOPBACK_ONLY,
    NETWORK_POLICY_ALLOW_REMOTE,
)

#: Auxiliary-model deny policy codes. These are *stable strings* the UI
#: and structured logs may key off, so they must not change without a
#: migration plan.
AUX_DENY_ALLOWED = "aux_allowed"
AUX_DENY_POLICY_DISABLED = "aux_policy_disabled"
AUX_DENY_ENDPOINT_NOT_LOOPBACK = "aux_endpoint_not_loopback"
AUX_DENY_ENDPOINT_NOT_ALLOWLISTED = "aux_endpoint_not_allowlisted"
AUX_DENY_PATH_NOT_LOCAL = "aux_path_not_local"
AUX_DENY_PATH_NOT_ALLOWLISTED = "aux_path_not_allowlisted"
AUX_DENY_MODEL_NOT_ALLOWLISTED = "aux_model_not_allowlisted"
AUX_DENY_MODEL_NOT_DECLARED = "aux_model_not_declared"
AUX_DENY_TRANSPORT_DENIED = "aux_transport_denied"
AUX_DENY_PAYLOAD_DENIED = "aux_payload_denied"
KNOWN_AUX_DENY_CODES: Tuple[str, ...] = (
    AUX_DENY_ALLOWED,
    AUX_DENY_POLICY_DISABLED,
    AUX_DENY_ENDPOINT_NOT_LOOPBACK,
    AUX_DENY_ENDPOINT_NOT_ALLOWLISTED,
    AUX_DENY_PATH_NOT_LOCAL,
    AUX_DENY_PATH_NOT_ALLOWLISTED,
    AUX_DENY_MODEL_NOT_ALLOWLISTED,
    AUX_DENY_MODEL_NOT_DECLARED,
    AUX_DENY_TRANSPORT_DENIED,
    AUX_DENY_PAYLOAD_DENIED,
)


# ---------------------------------------------------------------------------
# Bytes / MiB helpers
# ---------------------------------------------------------------------------


_BYTES_PER_KIB = 1024
_MIB_PER_BYTE = 1.0 / (1024 * 1024)


def format_bytes(value: Optional[int]) -> str:
    """Format a byte count deterministically.

    Returns ``"unknown"`` for ``None`` or any negative value so callers do
    not have to special-case probes that fail to measure.
    """
    if value is None:
        return "unknown"
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if number < 0:
        return "unknown"
    if number < _BYTES_PER_KIB:
        return f"{number} B"
    units: Tuple[Tuple[str, int], ...] = (
        ("KiB", 1024),
        ("MiB", 1024 * 1024),
        ("GiB", 1024 * 1024 * 1024),
        ("TiB", 1024 * 1024 * 1024 * 1024),
    )
    for label, scale in reversed(units):
        if number >= scale:
            scaled = number / scale
            # Trim trailing zeros while keeping at least one decimal.
            text = f"{scaled:.2f}".rstrip("0").rstrip(".")
            if not text:
                text = "0"
            return f"{text} {label}"
    return f"{number} B"


def format_mib(value: Optional[int]) -> str:
    """Format a MiB count deterministically (integer-only, no rounding)."""
    if value is None:
        return "unknown"
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if number < 0:
        return "unknown"
    return f"{number} MiB"


def bytes_to_mib(value: Optional[int]) -> Optional[int]:
    """Convert a byte count to integer MiB (``bytes / 1024 / 1024``).

    Returns ``None`` for ``None`` / negative inputs. Integer truncation
    is intentional — callers that need more precision should keep the
    byte value.
    """
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number >> 20  # 1024 * 1024 == 1 << 20


# ---------------------------------------------------------------------------
# Row dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceRecord:
    """A single deterministic row in the resource ledger.

    ``role`` is one of :data:`ROLE_ORDER` so consumers can sort and
    filter rows by purpose without consulting a second table.

    ``path_or_repository`` is intentionally a single string slot that
    carries either a filesystem path (for ``local_path`` / bundled /
    plugin sources) or a repository / endpoint identifier (for huggingface
    / loopback). It must never contain audio, transcript, or credential
    contents.
    """

    role: str
    purpose: str
    path_or_repository: str
    source: str
    backend: str
    device: str
    dtype: str
    load_state: str          # "loaded" | "not_loaded" | "failed" | "n/a"
    artifact_bytes: int      # -1 if unknown
    estimated_vram_mb: Optional[int] = None
    runtime_vram_mb: Optional[int] = None
    process_rss_bytes: Optional[int] = None
    plugin_origin: Optional[str] = None
    plugin_enabled: Optional[bool] = None
    network_policy: Optional[str] = None
    auxiliary_deny_code: Optional[str] = None
    notes: Tuple[str, ...] = ()

    def as_row(self) -> Tuple[Any, ...]:
        """Return a deterministic tuple suitable for tabular rendering."""
        return (
            self.role,
            self.purpose,
            self.path_or_repository,
            self.source,
            self.backend,
            self.device,
            self.dtype,
            self.load_state,
            self.artifact_bytes,
            self.estimated_vram_mb,
            self.runtime_vram_mb,
            self.process_rss_bytes,
            self.plugin_origin,
            self.plugin_enabled,
            self.network_policy,
            self.auxiliary_deny_code,
            "|".join(self.notes),
        )


@dataclass(frozen=True)
class ResourceLedger:
    """An immutable, deterministic ledger of all resource rows.

    ``records`` is always ordered so the primary ASR row appears first,
    the auxiliary model row second (when present), the plugin rows next
    (sorted by ``(origin, name)``), the process row, the network policy
    row, and finally the device row.
    """

    records: Tuple[ResourceRecord, ...]
    generated_at: Optional[str] = None  # ISO-8601 UTC string, or None
    schema_version: str = "1"

    def primary_asr(self) -> Optional[ResourceRecord]:
        """Return the primary ASR row, or ``None`` if absent."""
        for record in self.records:
            if record.role == ROLE_PRIMARY_ASR:
                return record
        return None

    def auxiliary_model(self) -> Optional[ResourceRecord]:
        """Return the auxiliary model row, or ``None`` if absent."""
        for record in self.records:
            if record.role == ROLE_AUXILIARY_MODEL:
                return record
        return None

    def plugin_rows(self) -> Tuple[ResourceRecord, ...]:
        """Return just the plugin rows (sorted)."""
        return tuple(r for r in self.records if r.role == ROLE_PLUGIN)

    def process_row(self) -> Optional[ResourceRecord]:
        for record in self.records:
            if record.role == ROLE_PROCESS:
                return record
        return None

    def network_row(self) -> Optional[ResourceRecord]:
        for record in self.records:
            if record.role == ROLE_NETWORK:
                return record
        return None

    def device_row(self) -> Optional[ResourceRecord]:
        for record in self.records:
            if record.role == ROLE_DEVICE:
                return record
        return None

    def rows(self) -> Tuple[Tuple[Any, ...], ...]:
        """Return every record as a deterministic tuple row."""
        return tuple(record.as_row() for record in self.records)

    def summary(self) -> Mapping[str, Any]:
        """Return a small, JSON-serializable summary mapping.

        The mapping is intentionally flat and free of any nested mutable
        types so callers can drop it straight into ``json.dumps``.
        """
        asr = self.primary_asr()
        aux = self.auxiliary_model()
        proc = self.process_row()
        net = self.network_row()
        dev = self.device_row()
        plugins = self.plugin_rows()
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "primary_asr": _record_to_summary(asr),
            "auxiliary_model": _record_to_summary(aux),
            "plugin_count": len(plugins),
            "plugin_enabled_count": sum(
                1 for p in plugins if bool(p.plugin_enabled)
            ),
            "process": _record_to_summary(proc),
            "network": _record_to_summary(net),
            "device": _record_to_summary(dev),
            "rows": [list(row) for row in self.rows()],
        }


def _record_to_summary(record: Optional[ResourceRecord]) -> Optional[Mapping[str, Any]]:
    if record is None:
        return None
    return {
        "role": record.role,
        "purpose": record.purpose,
        "path_or_repository": record.path_or_repository,
        "source": record.source,
        "backend": record.backend,
        "device": record.device,
        "dtype": record.dtype,
        "load_state": record.load_state,
        "artifact_bytes": record.artifact_bytes,
        "artifact_human": format_bytes(record.artifact_bytes),
        "estimated_vram_mb": record.estimated_vram_mb,
        "runtime_vram_mb": record.runtime_vram_mb,
        "process_rss_bytes": record.process_rss_bytes,
        "process_rss_human": format_bytes(record.process_rss_bytes),
        "plugin_origin": record.plugin_origin,
        "plugin_enabled": record.plugin_enabled,
        "network_policy": record.network_policy,
        "auxiliary_deny_code": record.auxiliary_deny_code,
        "notes": list(record.notes),
    }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PluginInput:
    """Minimal input for a plugin row."""

    name: str
    origin: str = PLUGIN_ORIGIN_BUILTIN
    enabled: bool = True
    loaded: bool = True
    failure_type: Optional[str] = None
    failure_message: Optional[str] = None

    def load_state(self) -> str:
        if self.failure_type is not None:
            return "failed"
        if self.loaded:
            return "loaded"
        if self.enabled:
            return "not_loaded"
        return "n/a"


@dataclass(frozen=True)
class ModelInput:
    """Minimal input for a model row."""

    purpose: str
    identifier: str
    source: str = SOURCE_BUNDLED
    backend: str = ""
    device: str = "cpu"
    dtype: str = "float32"
    is_loaded: bool = False
    artifact_bytes: int = -1
    estimated_vram_mb: Optional[int] = None
    runtime_vram_mb: Optional[int] = None
    notes: Tuple[str, ...] = ()
    path_or_repository: str = ""


# ---------------------------------------------------------------------------
# Probe contract (injectable)
# ---------------------------------------------------------------------------


#: A process-RSS probe. Must return ``None`` or a non-negative int.
ProcessRssProbe = Callable[[], Optional[int]]
#: An artifact-size probe, given the artifact name. Returns bytes (or
#: ``-1`` for unknown).
ArtifactSizeProbe = Callable[[str], int]
#: A runtime-VRAM probe, given the model identifier. Returns MiB or None.
RuntimeVramProbe = Callable[[str], Optional[int]]


def _no_process_rss() -> Optional[int]:
    """Default process-RSS probe: returns ``None`` (unknown)."""
    return None


def _no_artifact_size(_name: str) -> int:
    """Default artifact-size probe: returns ``-1`` (unknown)."""
    return -1


def _no_runtime_vram(_identifier: str) -> Optional[int]:
    """Default runtime-VRAM probe: returns ``None`` (unmeasured)."""
    return None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _load_state(is_loaded: bool) -> str:
    return "loaded" if is_loaded else "not_loaded"


def _coerce_model_source(source: str) -> str:
    normalized = (source or "").strip().lower() or SOURCE_BUNDLED
    if normalized not in KNOWN_MODEL_SOURCES:
        return SOURCE_BUNDLED
    return normalized


def _coerce_plugin_origin(origin: str) -> str:
    normalized = (origin or "").strip().lower() or PLUGIN_ORIGIN_BUILTIN
    if normalized not in KNOWN_PLUGIN_ORIGINS:
        return PLUGIN_ORIGIN_BUILTIN
    return normalized


def _coerce_network_policy(policy: str) -> str:
    normalized = (policy or "").strip().lower() or NETWORK_POLICY_LOOPBACK_ONLY
    if normalized not in KNOWN_NETWORK_POLICIES:
        return NETWORK_POLICY_LOOPBACK_ONLY
    return normalized


def build_resource_ledger(
    *,
    primary_asr: ModelInput,
    auxiliary_model: Optional[ModelInput] = None,
    plugins: Sequence[PluginInput] = (),
    process_role_label: str = "primary_process",
    process_device: str = "cpu",
    network_policy: str = NETWORK_POLICY_LOOPBACK_ONLY,
    network_loopback_endpoints: Sequence[str] = (),
    network_allow_remote_hosts: Sequence[str] = (),
    device_label: str = "cpu",
    device_kind: str = "cpu",
    device_dtype: str = "float32",
    device_index: int = -1,
    device_total_vram_mb: Optional[int] = None,
    device_free_vram_mb: Optional[int] = None,
    generated_at: Optional[str] = None,
    schema_version: str = "1",
    # Live probes — all optional and side-effect-free defaults.
    process_rss_probe: ProcessRssProbe = _no_process_rss,
    artifact_size_probe: ArtifactSizeProbe = _no_artifact_size,
    runtime_vram_probe: RuntimeVramProbe = _no_runtime_vram,
    probe_enabled: bool = False,
) -> ResourceLedger:
    """Build a :class:`ResourceLedger` from explicit values and optional probes.

    ``probe_enabled`` is the master switch for live probes. When ``False``
    (the default) every probe is ignored and the explicit values are used
    verbatim — the helper is then deterministic and safe to call from
    tests or during cold start.

    The returned ledger always contains, in order: primary ASR row,
    auxiliary model row (when present), plugin rows (sorted by
    ``(origin, name)``), process row, network row, and device row.
    """
    primary_path = primary_asr.path_or_repository or primary_asr.identifier
    primary_record = ResourceRecord(
        role=ROLE_PRIMARY_ASR,
        purpose=primary_asr.purpose,
        path_or_repository=primary_path,
        source=_coerce_model_source(primary_asr.source),
        backend=primary_asr.backend or "unknown",
        device=primary_asr.device,
        dtype=primary_asr.dtype,
        load_state=_load_state(bool(primary_asr.is_loaded)),
        artifact_bytes=int(primary_asr.artifact_bytes),
        estimated_vram_mb=primary_asr.estimated_vram_mb,
        runtime_vram_mb=primary_asr.runtime_vram_mb,
        notes=tuple(primary_asr.notes),
    )

    aux_record: Optional[ResourceRecord] = None
    if auxiliary_model is not None:
        aux_path = auxiliary_model.path_or_repository or auxiliary_model.identifier
        aux_record = ResourceRecord(
            role=ROLE_AUXILIARY_MODEL,
            purpose=auxiliary_model.purpose,
            path_or_repository=aux_path,
            source=_coerce_model_source(auxiliary_model.source),
            backend=auxiliary_model.backend or "unknown",
            device=auxiliary_model.device,
            dtype=auxiliary_model.dtype,
            load_state=_load_state(bool(auxiliary_model.is_loaded)),
            artifact_bytes=int(auxiliary_model.artifact_bytes),
            estimated_vram_mb=auxiliary_model.estimated_vram_mb,
            runtime_vram_mb=auxiliary_model.runtime_vram_mb,
            notes=tuple(auxiliary_model.notes),
        )

    plugin_records: list[ResourceRecord] = []
    for plugin in plugins:
        origin = _coerce_plugin_origin(plugin.origin)
        plugin_records.append(
            ResourceRecord(
                role=ROLE_PLUGIN,
                purpose=f"plugin:{plugin.name}",
                path_or_repository=plugin.name,
                source=origin,
                backend="plugin",
                device="n/a",
                dtype="n/a",
                load_state=plugin.load_state(),
                artifact_bytes=-1,
                estimated_vram_mb=None,
                runtime_vram_mb=None,
                plugin_origin=origin,
                plugin_enabled=bool(plugin.enabled),
                notes=(
                    (f"failure:{plugin.failure_type}",)
                    if plugin.failure_type
                    else ()
                ),
            )
        )
    plugin_records.sort(key=lambda r: (r.plugin_origin or "", r.path_or_repository))

    # Process row: RSS comes from explicit value or probe.
    resolved_rss: Optional[int] = None
    if probe_enabled:
        try:
            probed = process_rss_probe()
        except Exception:
            probed = None
        if probed is not None:
            try:
                resolved_rss = int(probed)
            except (TypeError, ValueError):
                resolved_rss = None

    process_record = ResourceRecord(
        role=ROLE_PROCESS,
        purpose=process_role_label,
        path_or_repository="",
        source="process",
        backend="python",
        device=process_device,
        dtype="n/a",
        load_state="loaded",
        artifact_bytes=-1,
        estimated_vram_mb=None,
        runtime_vram_mb=None,
        process_rss_bytes=resolved_rss,
        notes=(
            (f"rss={format_bytes(resolved_rss)}",) if resolved_rss is not None else ()
        ),
    )

    net_policy = _coerce_network_policy(network_policy)
    loopback_notes = tuple(
        f"loopback:{endpoint}" for endpoint in network_loopback_endpoints
    )
    remote_notes = tuple(
        f"remote:{host}" for host in network_allow_remote_hosts
    )
    network_record = ResourceRecord(
        role=ROLE_NETWORK,
        purpose="network_policy",
        path_or_repository="",
        source="policy",
        backend="network",
        device="n/a",
        dtype="n/a",
        load_state="n/a",
        artifact_bytes=-1,
        estimated_vram_mb=None,
        runtime_vram_mb=None,
        network_policy=net_policy,
        notes=loopback_notes + remote_notes,
    )

    device_record = ResourceRecord(
        role=ROLE_DEVICE,
        purpose="compute_device",
        path_or_repository="",
        source="device",
        backend="compute",
        device=device_kind,
        dtype=device_dtype,
        load_state="n/a",
        artifact_bytes=-1,
        estimated_vram_mb=device_free_vram_mb,
        runtime_vram_mb=device_total_vram_mb,
        notes=(f"index={device_index}", f"label={device_label}"),
    )

    # Optional runtime-VRAM probe for the primary ASR. We only fire the
    # probe when the model is actually loaded — there is no meaningful
    # runtime VRAM to measure for a not-yet-loaded (or failed) model, and
    # probing in that state would risk invoking model runtimes / GPU
    # queries from inside the ledger builder, which the contract forbids.
    if (
        probe_enabled
        and primary_record.runtime_vram_mb is None
        and primary_record.load_state == "loaded"
    ):
        try:
            probed = runtime_vram_probe(primary_record.path_or_repository)
        except Exception:
            probed = None
        if probed is not None:
            try:
                coerced = int(probed)
            except (TypeError, ValueError):
                coerced = None
            if coerced is not None:
                primary_record = _replace(primary_record, runtime_vram_mb=coerced)

    records_list: list[ResourceRecord] = [primary_record]
    if aux_record is not None:
        records_list.append(aux_record)
    records_list.extend(plugin_records)
    records_list.append(process_record)
    records_list.append(network_record)
    records_list.append(device_record)

    return ResourceLedger(
        records=tuple(records_list),
        generated_at=generated_at,
        schema_version=schema_version,
    )


def _replace(record: ResourceRecord, **changes: Any) -> ResourceRecord:
    """Replace fields on a frozen :class:`ResourceRecord` (small helper)."""
    from dataclasses import replace as _replace_dc

    return _replace_dc(record, **changes)


# Tiny helper so the runtime-VRAM probe path can check load state
# without leaking implementation details onto the row dataclass.
def _is_loaded_state_known(record: ResourceRecord) -> bool:
    return record.load_state in ("loaded", "not_loaded")


# ---------------------------------------------------------------------------
# Auxiliary-model deny policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuxiliaryPolicyConfig:
    """The deny-policy configuration the ledger applies to auxiliary models."""

    policy: str = NETWORK_POLICY_LOOPBACK_ONLY
    allowed_loopback_endpoints: Tuple[str, ...] = ()
    allowed_local_paths: Tuple[str, ...] = ()
    allowed_model_identifiers: Tuple[str, ...] = ()

    def with_endpoint(self, endpoint: str) -> "AuxiliaryPolicyConfig":
        text = (endpoint or "").strip().lower()
        if not text:
            return self
        if text in (e.lower() for e in self.allowed_loopback_endpoints):
            return self
        return AuxiliaryPolicyConfig(
            policy=self.policy,
            allowed_loopback_endpoints=tuple(self.allowed_loopback_endpoints) + (text,),
            allowed_local_paths=self.allowed_local_paths,
            allowed_model_identifiers=self.allowed_model_identifiers,
        )


@dataclass(frozen=True)
class AuxiliaryDenyReason:
    """Structured result of an auxiliary-model policy evaluation.

    ``code`` is one of :data:`KNOWN_AUX_DENY_CODES`. ``allowed`` is the
    inverse of "deny": ``True`` means the request is permitted, ``False``
    means it must be refused.
    """

    code: str
    allowed: bool
    message: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def is_denied(self) -> bool:
        return not self.allowed

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "code": self.code,
            "allowed": self.allowed,
            "message": self.message,
            "detail": dict(self.detail),
        }


_ALLOWED_REASON = AuxiliaryDenyReason(
    code=AUX_DENY_ALLOWED,
    allowed=True,
    message="Auxiliary request is permitted by the current policy.",
    detail={},
)


def evaluate_auxiliary_deny(
    config: AuxiliaryPolicyConfig,
    *,
    requested_endpoint: Optional[str] = None,
    requested_local_path: Optional[str] = None,
    requested_model_identifier: Optional[str] = None,
    auxiliary_declared: Optional[ModelInput] = None,
    payload_classification: str = "ok",
) -> AuxiliaryDenyReason:
    """Decide whether an auxiliary-model request must be **denied**.

    This is a pure function — no network or filesystem access, no side
    effects, never raises. The returned :class:`AuxiliaryDenyReason`
    carries a stable ``code`` (one of :data:`KNOWN_AUX_DENY_CODES`) plus
    a human-readable ``message`` and a ``detail`` mapping that callers
    may log or surface in the UI.

    The deny ladder, in order:

    1. ``aux_policy_disabled`` — ``config.policy`` is
       :data:`NETWORK_POLICY_OFFLINE_ONLY` (or unknown).
    2. ``aux_transport_denied`` — the network policy is
       :data:`NETWORK_POLICY_OFFLINE_ONLY` and *any* auxiliary request
       is being attempted.
    3. ``aux_payload_denied`` — ``payload_classification`` is not
       ``"ok"``. ``detail["payload_classification"]`` carries the
       offending label.
    4. ``aux_endpoint_not_loopback`` / ``aux_endpoint_not_allowlisted``
       — endpoint rules.
    5. ``aux_path_not_local`` / ``aux_path_not_allowlisted`` — local
       path rules.
    6. ``aux_model_not_allowlisted`` — model-identifier rules.
    7. ``aux_model_not_declared`` — caller asked for an auxiliary model
       but the ledger declares none.
    8. ``aux_allowed`` — request is permitted.
    """
    policy = (config.policy or "").strip().lower()

    if policy not in (
        NETWORK_POLICY_OFFLINE_ONLY,
        NETWORK_POLICY_LOOPBACK_ONLY,
        NETWORK_POLICY_ALLOW_REMOTE,
    ):
        return AuxiliaryDenyReason(
            code=AUX_DENY_POLICY_DISABLED,
            allowed=False,
            message=(
                f"Auxiliary policy {policy!r} is not recognised; "
                "denying all auxiliary requests."
            ),
            detail={"policy": policy},
        )

    if policy == NETWORK_POLICY_OFFLINE_ONLY:
        return AuxiliaryDenyReason(
            code=AUX_DENY_TRANSPORT_DENIED,
            allowed=False,
            message=(
                "Network policy is offline-only; auxiliary requests are "
                "denied by transport."
            ),
            detail={"policy": policy},
        )

    if payload_classification and payload_classification != "ok":
        return AuxiliaryDenyReason(
            code=AUX_DENY_PAYLOAD_DENIED,
            allowed=False,
            message=(
                "Auxiliary payload classification is not 'ok'; refusing "
                "to forward user content to the auxiliary model."
            ),
            detail={"payload_classification": payload_classification},
        )

    if requested_endpoint is not None:
        return _evaluate_endpoint(config, requested_endpoint)
    if requested_local_path is not None:
        return _evaluate_local_path(config, requested_local_path)
    if requested_model_identifier is not None:
        return _evaluate_model_identifier(config, requested_model_identifier)

    if auxiliary_declared is None:
        return AuxiliaryDenyReason(
            code=AUX_DENY_MODEL_NOT_DECLARED,
            allowed=False,
            message=(
                "Caller asked about an auxiliary request but the ledger "
                "declares no auxiliary model."
            ),
            detail={},
        )

    return _ALLOWED_REASON


def _evaluate_endpoint(
    config: AuxiliaryPolicyConfig,
    requested_endpoint: Optional[str],
) -> AuxiliaryDenyReason:
    endpoint = (requested_endpoint or "").strip()
    if not endpoint:
        return AuxiliaryDenyReason(
            code=AUX_DENY_ENDPOINT_NOT_ALLOWLISTED,
            allowed=False,
            message="Refusing empty auxiliary endpoint.",
            detail={"endpoint": endpoint},
        )
    lowered = endpoint.lower()
    host = _endpoint_host(lowered)
    if host is None or (
        not host.startswith("127.") and host != "localhost" and host != "::1"
    ):
        return AuxiliaryDenyReason(
            code=AUX_DENY_ENDPOINT_NOT_LOOPBACK,
            allowed=False,
            message=(
                f"Auxiliary endpoint {endpoint!r} is not a loopback host; "
                "remote endpoints are denied."
            ),
            detail={"endpoint": endpoint, "host": host},
        )
    loopback_only = (config.policy or "").strip().lower() == NETWORK_POLICY_LOOPBACK_ONLY
    if loopback_only:
        if not config.allowed_loopback_endpoints:
            return AuxiliaryDenyReason(
                code=AUX_DENY_ENDPOINT_NOT_ALLOWLISTED,
                allowed=False,
                message=(
                    "No loopback endpoints are allow-listed for the "
                    "auxiliary policy."
                ),
                detail={"endpoint": endpoint},
            )
        if not any(
            candidate.strip().lower() == lowered
            for candidate in config.allowed_loopback_endpoints
        ):
            return AuxiliaryDenyReason(
                code=AUX_DENY_ENDPOINT_NOT_ALLOWLISTED,
                allowed=False,
                message=(
                    f"Auxiliary endpoint {endpoint!r} is not in the "
                    "allow-list of loopback endpoints."
                ),
                detail={"endpoint": endpoint},
            )
    return AuxiliaryDenyReason(
        code=AUX_DENY_ALLOWED,
        allowed=True,
        message=f"Auxiliary endpoint {endpoint!r} is permitted.",
        detail={"endpoint": endpoint, "loopback_only": loopback_only},
    )


def _evaluate_local_path(
    config: AuxiliaryPolicyConfig,
    requested_local_path: Optional[str],
) -> AuxiliaryDenyReason:
    path = (requested_local_path or "").strip()
    if not path:
        return AuxiliaryDenyReason(
            code=AUX_DENY_PATH_NOT_ALLOWLISTED,
            allowed=False,
            message="Refusing empty auxiliary local path.",
            detail={"path": path},
        )
    if not _looks_like_local_path(path):
        return AuxiliaryDenyReason(
            code=AUX_DENY_PATH_NOT_LOCAL,
            allowed=False,
            message=(
                f"Auxiliary local path {path!r} is not a local filesystem "
                "path; remote schemes are denied."
            ),
            detail={"path": path},
        )
    if (config.policy or "").strip().lower() == NETWORK_POLICY_LOOPBACK_ONLY:
        return AuxiliaryDenyReason(
            code=AUX_DENY_PATH_NOT_ALLOWLISTED,
            allowed=False,
            message=(
                "Auxiliary policy is loopback-only; local paths are "
                "denied by transport."
            ),
            detail={"path": path},
        )
    if not config.allowed_local_paths:
        return AuxiliaryDenyReason(
            code=AUX_DENY_PATH_NOT_ALLOWLISTED,
            allowed=False,
            message="No local paths are allow-listed for the auxiliary policy.",
            detail={"path": path},
        )
    if not any(
        _paths_match(candidate, path) for candidate in config.allowed_local_paths
    ):
        return AuxiliaryDenyReason(
            code=AUX_DENY_PATH_NOT_ALLOWLISTED,
            allowed=False,
            message=(
                f"Auxiliary local path {path!r} is not in the allow-list."
            ),
            detail={"path": path},
        )
    return AuxiliaryDenyReason(
        code=AUX_DENY_ALLOWED,
        allowed=True,
        message=f"Auxiliary local path {path!r} is permitted.",
        detail={"path": path},
    )


def _evaluate_model_identifier(
    config: AuxiliaryPolicyConfig,
    requested_model_identifier: Optional[str],
) -> AuxiliaryDenyReason:
    identifier = (requested_model_identifier or "").strip()
    if not identifier:
        return AuxiliaryDenyReason(
            code=AUX_DENY_MODEL_NOT_ALLOWLISTED,
            allowed=False,
            message="Refusing empty auxiliary model identifier.",
            detail={"identifier": identifier},
        )
    if not config.allowed_model_identifiers:
        return AuxiliaryDenyReason(
            code=AUX_DENY_MODEL_NOT_ALLOWLISTED,
            allowed=False,
            message="No auxiliary model identifiers are allow-listed.",
            detail={"identifier": identifier},
        )
    lowered = identifier.casefold()
    if not any(
        candidate.casefold() == lowered
        for candidate in config.allowed_model_identifiers
    ):
        return AuxiliaryDenyReason(
            code=AUX_DENY_MODEL_NOT_ALLOWLISTED,
            allowed=False,
            message=(
                f"Auxiliary model identifier {identifier!r} is not in "
                "the allow-list."
            ),
            detail={"identifier": identifier},
        )
    return AuxiliaryDenyReason(
        code=AUX_DENY_ALLOWED,
        allowed=True,
        message=f"Auxiliary model {identifier!r} is permitted.",
        detail={"identifier": identifier},
    )


def _endpoint_host(endpoint: str) -> Optional[str]:
    """Return the host part of a loopback-ish URL, or ``None`` if not parseable."""
    candidate = endpoint.strip()
    if not candidate:
        return None
    if "://" not in candidate:
        # Bare host:port.
        if "/" in candidate:
            return None
        host = candidate.split(":", 1)[0]
        return host or None
    scheme, _, rest = candidate.partition("://")
    if not scheme or not rest:
        return None
    if "/" in rest:
        host = rest.split("/", 1)[0]
    else:
        host = rest
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if host.startswith("[") and "]" in host:
        # IPv6 literal
        host = host[1 : host.index("]")]
    elif ":" in host:
        host = host.split(":", 1)[0]
    return host or None


def _looks_like_local_path(path: str) -> bool:
    if not path:
        return False
    lowered = path.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return False
    if "://" in path:
        return False
    return True


def _paths_match(allow: str, requested: str) -> bool:
    """Compare two paths case-insensitively; Windows-aware."""
    allow_norm = os.path.normpath(allow).replace("\\", "/").rstrip("/").lower()
    requested_norm = (
        os.path.normpath(requested).replace("\\", "/").rstrip("/").lower()
    )
    return allow_norm == requested_norm


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def render_ledger_table(
    ledger: ResourceLedger,
    *,
    include_header: bool = True,
    indent: str = "  ",
) -> str:
    """Render a deterministic, plain-text view of the ledger.

    Output is suitable for embedding in logs, the doctor report, or the
    settings UI — no colours, no rich rendering, no live probes.
    """
    lines: list[str] = []
    if include_header:
        lines.append(
            f"ResourceLedger schema={ledger.schema_version} rows={len(ledger.records)}"
        )
    for record in ledger.records:
        lines.append(_render_record(record, indent=indent))
    return "\n".join(lines)


def _render_record(record: ResourceRecord, *, indent: str) -> str:
    bits: list[str] = [
        f"role={record.role}",
        f"purpose={record.purpose}",
    ]
    if record.path_or_repository:
        bits.append(f"path={record.path_or_repository}")
    if record.source:
        bits.append(f"source={record.source}")
    if record.backend:
        bits.append(f"backend={record.backend}")
    if record.device and record.device != "n/a":
        bits.append(f"device={record.device}")
    if record.dtype and record.dtype != "n/a":
        bits.append(f"dtype={record.dtype}")
    bits.append(f"load_state={record.load_state}")
    if record.artifact_bytes >= 0:
        bits.append(f"artifact={format_bytes(record.artifact_bytes)}")
    if record.estimated_vram_mb is not None:
        bits.append(f"est_vram={format_mib(record.estimated_vram_mb)}")
    if record.runtime_vram_mb is not None:
        bits.append(f"runtime_vram={format_mib(record.runtime_vram_mb)}")
    if record.process_rss_bytes is not None:
        bits.append(f"rss={format_bytes(record.process_rss_bytes)}")
    if record.network_policy:
        bits.append(f"net_policy={record.network_policy}")
    if record.auxiliary_deny_code:
        bits.append(f"aux_deny={record.auxiliary_deny_code}")
    if record.notes:
        bits.append("notes=[" + ", ".join(record.notes) + "]")
    return indent + " | ".join(bits)


def render_ledger_summary(ledger: ResourceLedger) -> str:
    """Render a one-line-per-section deterministic summary."""
    lines: list[str] = []
    asr = ledger.primary_asr()
    if asr is not None:
        lines.append(
            "primary_asr: "
            f"{asr.path_or_repository} ({asr.backend}/{asr.device}/{asr.dtype}) "
            f"loaded={asr.load_state} "
            f"artifact={format_bytes(asr.artifact_bytes)}"
        )
    aux = ledger.auxiliary_model()
    if aux is not None:
        lines.append(
            "auxiliary_model: "
            f"{aux.path_or_repository} ({aux.backend}/{aux.device}/{aux.dtype}) "
            f"loaded={aux.load_state} "
            f"artifact={format_bytes(aux.artifact_bytes)}"
        )
    plugins = ledger.plugin_rows()
    lines.append(
        "plugins: "
        f"{len(plugins)} total, "
        f"{sum(1 for p in plugins if p.plugin_enabled)} enabled, "
        f"{sum(1 for p in plugins if p.load_state == 'failed')} failed"
    )
    proc = ledger.process_row()
    if proc is not None:
        lines.append(
            "process: "
            f"rss={format_bytes(proc.process_rss_bytes)}"
        )
    net = ledger.network_row()
    if net is not None:
        lines.append(f"network_policy: {net.network_policy}")
    dev = ledger.device_row()
    if dev is not None:
        bits = [f"device={dev.device}", f"dtype={dev.dtype}"]
        if dev.runtime_vram_mb is not None:
            bits.append(f"total_vram={format_mib(dev.runtime_vram_mb)}")
        if dev.estimated_vram_mb is not None:
            bits.append(f"free_vram={format_mib(dev.estimated_vram_mb)}")
        lines.append("compute_device: " + ", ".join(bits))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    # Constants
    "ROLE_PRIMARY_ASR",
    "ROLE_AUXILIARY_MODEL",
    "ROLE_PLUGIN",
    "ROLE_PROCESS",
    "ROLE_NETWORK",
    "ROLE_DEVICE",
    "ROLE_ORDER",
    "SOURCE_BUNDLED",
    "SOURCE_HUGGINGFACE",
    "SOURCE_LOCAL_PATH",
    "SOURCE_LOOPBACK",
    "KNOWN_MODEL_SOURCES",
    "PLUGIN_ORIGIN_BUILTIN",
    "PLUGIN_ORIGIN_ENTRY_POINT",
    "PLUGIN_ORIGIN_USER_DIRECTORY",
    "KNOWN_PLUGIN_ORIGINS",
    "NETWORK_POLICY_OFFLINE_ONLY",
    "NETWORK_POLICY_LOOPBACK_ONLY",
    "NETWORK_POLICY_ALLOW_REMOTE",
    "KNOWN_NETWORK_POLICIES",
    "AUX_DENY_ALLOWED",
    "AUX_DENY_POLICY_DISABLED",
    "AUX_DENY_ENDPOINT_NOT_LOOPBACK",
    "AUX_DENY_ENDPOINT_NOT_ALLOWLISTED",
    "AUX_DENY_PATH_NOT_LOCAL",
    "AUX_DENY_PATH_NOT_ALLOWLISTED",
    "AUX_DENY_MODEL_NOT_ALLOWLISTED",
    "AUX_DENY_MODEL_NOT_DECLARED",
    "AUX_DENY_TRANSPORT_DENIED",
    "AUX_DENY_PAYLOAD_DENIED",
    "KNOWN_AUX_DENY_CODES",
    # Types
    "ResourceRecord",
    "ResourceLedger",
    "ModelInput",
    "PluginInput",
    "AuxiliaryPolicyConfig",
    "AuxiliaryDenyReason",
    # Probe contracts
    "ProcessRssProbe",
    "ArtifactSizeProbe",
    "RuntimeVramProbe",
    # Helpers
    "format_bytes",
    "format_mib",
    "bytes_to_mib",
    "build_resource_ledger",
    "evaluate_auxiliary_deny",
    "render_ledger_table",
    "render_ledger_summary",
]


def _suppress_unused() -> Iterable[Any]:
    """Keep ``Iterable`` referenced for static analysers."""
    return (Iterable,)
