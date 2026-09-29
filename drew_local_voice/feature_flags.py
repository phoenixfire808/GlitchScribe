"""Lean Core feature flags.

Single source of truth for the default critical path. GlitchScribe defaults
to the **lean_core** path: rolling-stream bubble previews with a single
final insert (live_insert still off), no synthetic warm decode, no eagerly
imported AI/plugins/theme/workflows/project_ear tree on startup, and
chunked SendInput for reliable Cursor/Monaco insertion.

All optional behavior is preserved — these flags only move work **off the
default critical path**, they never delete it. A user (or env var) can flip
flags back on individually without touching any other code.

Override precedence (highest first):
    1. Environment variables (``GLITCHSCRIBE_<FLAG>=0|1``)
    2. Persisted ``config.json`` (``lean_core.*`` keys, written by Settings UI)
    3. Built-in defaults

Use :func:`resolve_flags` once at startup and then read fields off the
returned :class:`FeatureFlags` instance. The function is cheap (one dict
allocation + a handful of string lookups) so callers may also call it
defensively from hot paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class FeatureFlags:
    """Resolved lean_core flags used by the runtime.

    Every field is a boolean. ``True`` keeps the lean default behavior on;
    flipping a flag to ``False`` opts back into the legacy/optional path.
    """

    enabled: bool = True
    skip_synthetic_warm: bool = True
    lazy_import_ai: bool = True
    lazy_import_project_ear: bool = True
    lazy_import_workflows: bool = True
    lazy_import_shortcut_hook: bool = True
    lazy_import_settings_ui: bool = True
    lazy_import_theme: bool = True
    skip_streaming_when_disabled: bool = True
    bubble_show_first: bool = True
    chunked_insertion: bool = True
    skip_runtime_event_logging: bool = False
    skip_pipeline_observer: bool = False

    # Internal bookkeeping only; not a flag.
    sources: dict[str, str] = field(default_factory=dict)


_ENV_PREFIX = "GLITCHSCRIBE_"
_FLAG_FIELDS: tuple[str, ...] = tuple(field.name for field in fields(FeatureFlags) if field.name != "sources")


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Coerce env/json values to bool without raising."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on", "enable", "enabled"):
            return True
        if normalized in ("0", "false", "no", "off", "disable", "disabled", ""):
            return False
    return default


def _env_key(flag: str) -> str:
    return f"{_ENV_PREFIX}{flag.upper()}"


def resolve_flags(
    persisted: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> FeatureFlags:
    """Resolve lean_core feature flags from env > persisted > defaults.

    Parameters
    ----------
    persisted
        Mapping loaded from ``config.json`` (``advanced_settings`` is a
        natural home). ``None`` means "no user overrides".
    env
        Mapping of environment variables. ``None`` means "use ``os.environ``".
    """
    persisted = persisted or {}
    env_source: Mapping[str, str] = os.environ if env is None else env
    defaults = FeatureFlags()
    sources: dict[str, str] = {}
    overrides: dict[str, bool] = {}

    for flag in _FLAG_FIELDS:
        default = bool(getattr(defaults, flag))
        env_key = _env_key(flag)
        if env_key in env_source:
            overrides[flag] = _coerce_bool(env_source[env_key], default=default)
            sources[flag] = f"env:{env_key}"
            continue
        json_key = f"lean_core.{flag}"
        if json_key in persisted:
            overrides[flag] = _coerce_bool(persisted[json_key], default=default)
            sources[flag] = f"config:{json_key}"
            continue
        # No override; the default stands.

    if not overrides:
        return defaults

    kwargs = {flag: overrides.get(flag, getattr(defaults, flag)) for flag in _FLAG_FIELDS}
    return FeatureFlags(sources=sources, **kwargs)


def flags_from_advanced_settings(advanced: Mapping[str, Any] | None) -> FeatureFlags:
    """Convenience wrapper that reads from the standard advanced_settings bag."""
    return resolve_flags(persisted=advanced or {})


def iter_persisted_keys(flags: FeatureFlags | None = None) -> Iterable[str]:
    """Yield the ``lean_core.*`` config keys that exist in storage."""
    return tuple(f"lean_core.{name}" for name in _FLAG_FIELDS)


__all__ = [
    "FeatureFlags",
    "resolve_flags",
    "flags_from_advanced_settings",
    "iter_persisted_keys",
]