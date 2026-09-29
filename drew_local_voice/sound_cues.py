"""Compatibility policy facade for local GlitchScribe feedback cues.

The active catalog lives in ``cursor_fx.py`` so settings, playback, and visual
feedback cannot drift into different profile vocabularies. This module keeps a
small platform-neutral policy API for callers that imported the earlier
``sound_cues`` name; it never opens an audio device or touches the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .cursor_fx import SOUND_PROFILES as CURSOR_SOUND_PROFILES
from .cursor_fx import sound_profile

FEEDBACK_EVENTS = (
    "start",
    "stop",
    "complete",
    "error",
    "action_open",
    "action_select",
    "preview",
    "preview_ready",
    "cancel",
)

SOUND_PROFILES = tuple(CURSOR_SOUND_PROFILES)
DEFAULT_SOUND_EVENTS = ("start", "complete", "error")
DEFAULT_RING_EVENTS = ("start", "complete", "error")


@dataclass(frozen=True)
class Tone:
    frequency_hz: int
    duration_ms: int
    gap_ms: int = 12


def normalize_event_list(
    value: Iterable[object] | object,
    *,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Normalize a persisted string/list into known, de-duplicated event keys."""
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_items = list(value)
    else:
        raw_items = []
    normalized: list[str] = []
    for item in raw_items:
        event = str(item).strip().casefold()
        if event in FEEDBACK_EVENTS and event not in normalized:
            normalized.append(event)
    return tuple(normalized) if normalized else tuple(default)


def normalize_sound_profile(value: object) -> str:
    return sound_profile(str(value or "pulse")).key


def tones_for_event(profile: object, event: object) -> tuple[Tone, ...]:
    """Return the active catalog recipe using the compatibility Tone shape."""
    recipe = getattr(sound_profile(str(profile)), str(event).strip().casefold(), ())
    return tuple(Tone(int(frequency), int(duration)) for frequency, duration in recipe)


def event_enabled(
    event: object,
    *,
    master_enabled: object,
    configured_events: Iterable[object] | object,
    default_events: tuple[str, ...] = DEFAULT_SOUND_EVENTS,
) -> bool:
    if not bool(master_enabled):
        return False
    normalized_event = str(event).strip().casefold()
    events = normalize_event_list(configured_events, default=default_events)
    return normalized_event in events


__all__ = [
    "DEFAULT_RING_EVENTS",
    "DEFAULT_SOUND_EVENTS",
    "FEEDBACK_EVENTS",
    "SOUND_PROFILES",
    "Tone",
    "event_enabled",
    "normalize_event_list",
    "normalize_sound_profile",
    "tones_for_event",
]
