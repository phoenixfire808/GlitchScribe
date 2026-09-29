"""Stable public imports for extension authors."""

from .commands import CommandDefinition, CommandRegistry
from .models import (
    BUILT_IN_MODELS,
    configured_model_choices,
    estimated_model_vram_mb,
    validate_model_identifier,
)
from .plugins import ExtensionRegistry, PluginManager
from .performance import PerformancePreset, PerformancePresetRegistry
from .profiles import DictationProfile, ProfileRegistry
from .settings import (
    EffectRecipe,
    SettingKind,
    SettingSpec,
    SettingsRegistry,
    built_in_effect_recipes,
)

__all__ = [
    "CommandDefinition",
    "CommandRegistry",
    "BUILT_IN_MODELS",
    "DictationProfile",
    "EffectRecipe",
    "ExtensionRegistry",
    "PluginManager",
    "PerformancePreset",
    "PerformancePresetRegistry",
    "ProfileRegistry",
    "SettingKind",
    "SettingSpec",
    "SettingsRegistry",
    "built_in_effect_recipes",
    "configured_model_choices",
    "estimated_model_vram_mb",
    "validate_model_identifier",
]
