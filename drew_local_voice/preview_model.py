from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from dictation_overlay import compact_overlay_text

from .settings import SettingsRegistry


PREVIEW_STATE_LABELS = (
    "READY",
    "LISTENING • LIVE",
    "FINALIZING",
    "ERROR",
)

PREVIEW_SAMPLE_TEXTS: Mapping[str, str] = MappingProxyType(
    {
        "short": "Testing the GlitchScribe overlay.",
        "long": (
            "GlitchScribe keeps a realistic local preview beside the controls so "
            "width, wrapping, font size, opacity, and the visible text limit can be "
            "tuned before any setting is saved."
        ),
        "punctuation": (
            "Punctuation check: commas, semicolons; quotes “like this”; paths "
            r"C:\Projects\GlitchScribe; and symbols /, \, |, &&, --version."
        ),
        "rapid": (
            "Drew, test GlitchScribe at 9:42 PM—names, numbers 17 and 3.5, "
            "quick clauses, terminal-safe symbols, and a rapid final phrase!"
        ),
    }
)

_DEFAULT_SAMPLE_KEY = "short"
_PALETTE_SETTING_KEYS = frozenset(("ui.theme", "ui.accent_color"))
_MISSING = object()


def preview_setting_keys(registry: SettingsRegistry) -> tuple[str, ...]:
    """Return preview-relevant keys from the supplied canonical registry."""
    return tuple(
        row["key"]
        for row in registry.to_schema()
        if row["section"] == "Speech Bubble" or row["key"] in _PALETTE_SETTING_KEYS
    )


def _validated_or_default(
    registry: SettingsRegistry,
    key: str,
    value: Any = _MISSING,
) -> Any:
    spec = registry.get(key)
    candidate = spec.default if value is _MISSING else value
    try:
        return registry.validate(key, candidate)
    except (TypeError, ValueError):
        return registry.validate(key, spec.default)


def extract_bubble_settings(
    values: Mapping[str, Any],
    registry: SettingsRegistry,
) -> dict[str, Any]:
    """Copy and validate only settings used by the staged bubble preview.

    Missing or invalid caller values fall back to their corresponding canonical
    ``SettingSpec.default``. The caller's mapping is read only and is never
    retained by the returned dictionary.
    """
    return {
        key: _validated_or_default(registry, key, values.get(key, _MISSING))
        for key in preview_setting_keys(registry)
    }


def choose_preview_sample(sample_key: str = _DEFAULT_SAMPLE_KEY) -> str:
    """Choose a bounded, synthetic sample without consulting transcript history."""
    key = str(sample_key or "").strip().casefold()
    return PREVIEW_SAMPLE_TEXTS.get(key, PREVIEW_SAMPLE_TEXTS[_DEFAULT_SAMPLE_KEY])


def preview_display_text(text: str, max_characters: int | float) -> str:
    """Render preview text through the production overlay's compaction contract."""
    return compact_overlay_text(text, max_chars=max_characters)


def _preview_state_label(label: str) -> str:
    requested = str(label or "").strip().casefold()
    for state in PREVIEW_STATE_LABELS:
        if state.casefold() == requested:
            return state
    return PREVIEW_STATE_LABELS[0]


@dataclass(frozen=True)
class BubblePreviewState:
    """Immutable copy of the current pure preview state."""

    settings: dict[str, Any]
    sample_key: str
    sample_text: str
    state_label: str
    dirty: bool

    @property
    def display_text(self) -> str:
        return preview_display_text(
            self.sample_text,
            self.settings["ui.bubble_max_characters"],
        )


@dataclass(init=False)
class BubblePreviewModel:
    """Tk-free model for values staged inside the GlitchScribe Control Lab."""

    MAX_SAMPLE_CHARACTERS = 2_000

    registry: SettingsRegistry = field(repr=False)
    _settings: dict[str, Any] = field(repr=False)
    sample_key: str
    sample_text: str
    state_label: str
    dirty: bool
    _baseline_settings: dict[str, Any] = field(repr=False)
    _baseline_sample_key: str = field(repr=False)
    _baseline_sample_text: str = field(repr=False)
    _baseline_state_label: str = field(repr=False)

    def __init__(
        self,
        values: Mapping[str, Any] | None = None,
        *,
        registry: SettingsRegistry,
        sample_key: str = _DEFAULT_SAMPLE_KEY,
        sample_text: str | None = None,
        state_label: str = PREVIEW_STATE_LABELS[0],
    ) -> None:
        self.registry = registry
        self._settings = extract_bubble_settings(values or {}, registry)
        normalized_key = str(sample_key or "").strip().casefold()
        if normalized_key not in PREVIEW_SAMPLE_TEXTS:
            normalized_key = _DEFAULT_SAMPLE_KEY
        self.sample_key = normalized_key
        selected_text = (
            choose_preview_sample(normalized_key)
            if sample_text is None
            else self._bounded_sample_text(sample_text)
        )
        self.sample_text = selected_text
        self.state_label = _preview_state_label(state_label)
        self._baseline_settings = dict(self._settings)
        self._baseline_sample_key = self.sample_key
        self._baseline_sample_text = self.sample_text
        self._baseline_state_label = self.state_label
        self.dirty = False

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
        *,
        registry: SettingsRegistry,
        sample_key: str = _DEFAULT_SAMPLE_KEY,
        state_label: str = PREVIEW_STATE_LABELS[0],
    ) -> BubblePreviewModel:
        return cls(
            values,
            registry=registry,
            sample_key=sample_key,
            state_label=state_label,
        )

    @property
    def settings(self) -> dict[str, Any]:
        return dict(self._settings)

    @property
    def display_text(self) -> str:
        return preview_display_text(
            self.sample_text,
            self._settings["ui.bubble_max_characters"],
        )

    @property
    def state(self) -> BubblePreviewState:
        return BubblePreviewState(
            settings=dict(self._settings),
            sample_key=self.sample_key,
            sample_text=self.sample_text,
            state_label=self.state_label,
            dirty=self.dirty,
        )

    def stage_setting(self, key: str, value: Any) -> Any:
        if key not in self._settings:
            raise KeyError(f"setting is not available to the bubble preview: {key}")
        validated = _validated_or_default(self.registry, key, value)
        self._settings[key] = validated
        self._refresh_dirty()
        return validated

    def stage_values(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Stage only supplied preview keys, rejecting runtime-only keys."""
        staged: dict[str, Any] = {}
        for key, value in values.items():
            staged[key] = self.stage_setting(key, value)
        return staged

    def select_sample(self, sample_key: str) -> str:
        normalized_key = str(sample_key or "").strip().casefold()
        if normalized_key not in PREVIEW_SAMPLE_TEXTS:
            normalized_key = _DEFAULT_SAMPLE_KEY
        self.sample_key = normalized_key
        self.sample_text = choose_preview_sample(normalized_key)
        self._refresh_dirty()
        return self.sample_text

    def set_sample_text(self, text: str) -> str:
        self.sample_key = "custom"
        self.sample_text = self._bounded_sample_text(text)
        self._refresh_dirty()
        return self.sample_text

    def set_state_label(self, label: str) -> str:
        self.state_label = _preview_state_label(label)
        self._refresh_dirty()
        return self.state_label

    def revert(self) -> None:
        self._settings = dict(self._baseline_settings)
        self.sample_key = self._baseline_sample_key
        self.sample_text = self._baseline_sample_text
        self.state_label = self._baseline_state_label
        self.dirty = False

    def accept_staged(self) -> None:
        """Make the current pure state the new baseline after an external save."""
        self._baseline_settings = dict(self._settings)
        self._baseline_sample_key = self.sample_key
        self._baseline_sample_text = self.sample_text
        self._baseline_state_label = self.state_label
        self.dirty = False

    def _bounded_sample_text(self, text: str) -> str:
        if not isinstance(text, str):
            return choose_preview_sample(_DEFAULT_SAMPLE_KEY)
        return text[: self.MAX_SAMPLE_CHARACTERS]

    def _refresh_dirty(self) -> None:
        self.dirty = (
            self._settings != self._baseline_settings
            or self.sample_key != self._baseline_sample_key
            or self.sample_text != self._baseline_sample_text
            or self.state_label != self._baseline_state_label
        )
