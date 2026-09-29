"""Public package for GlitchScribe.

Top-level imports are intentionally minimal so the lean_core startup path
does not pay for AI providers, project-ear, workflows, theme, or
settings_ui chain imports on cold boot. Importers that need the heavier
registries should reach into submodules directly:

    from drew_local_voice.api import CommandRegistry
    from drew_local_voice.performance import PerformancePresetRegistry
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .feature_flags import FeatureFlags, resolve_flags

__all__ = [
    "FeatureFlags",
    "resolve_flags",
]

__version__ = "0.2.0"


if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from .api import (
        CommandDefinition,
        CommandRegistry,
        DictationProfile,
        ExtensionRegistry,
        PluginManager,
        PerformancePreset,
        PerformancePresetRegistry,
        ProfileRegistry,
        SettingKind,
        SettingSpec,
        SettingsRegistry,
    )


def __getattr__(name: str):
    """Lazy attribute access for legacy top-level names.

    Modules that historically imported the registries from
    ``drew_local_voice`` continue to work, but only when those names are
    actually requested — keeping the lean_core cold-start cost down.
    """
    legacy_map = {
        "CommandDefinition": "drew_local_voice.api",
        "CommandRegistry": "drew_local_voice.api",
        "DictationProfile": "drew_local_voice.api",
        "ExtensionRegistry": "drew_local_voice.api",
        "PluginManager": "drew_local_voice.api",
        "PerformancePreset": "drew_local_voice.performance",
        "PerformancePresetRegistry": "drew_local_voice.performance",
        "ProfileRegistry": "drew_local_voice.api",
        "SettingKind": "drew_local_voice.api",
        "SettingSpec": "drew_local_voice.api",
        "SettingsRegistry": "drew_local_voice.api",
    }
    module_name = legacy_map.get(name)
    if module_name is None:
        raise AttributeError(f"module 'drew_local_voice' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
