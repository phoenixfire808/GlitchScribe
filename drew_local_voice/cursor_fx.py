"""Local cursor feedback catalog: lightweight tones and status-only visual styles.

The catalog is intentionally data-only. Audio is synthesized by the Windows
backend without external files, while the cursor layer renders geometry only.
No transcript, clipboard, target-window, or product-name content belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SoundProfile:
    key: str
    label: str
    description: str
    start: tuple[tuple[int, int], ...]
    stop: tuple[tuple[int, int], ...]
    complete: tuple[tuple[int, int], ...]
    error: tuple[tuple[int, int], ...]


SOUND_PROFILES: Mapping[str, SoundProfile] = {
    "pulse": SoundProfile(
        "pulse",
        "Pulse",
        "Bright rising start and soft falling completion.",
        ((392, 34), (587, 42), (784, 52)),
        ((587, 30), (392, 42)),
        ((784, 34), (587, 42), (440, 52)),
        ((330, 70), (247, 90)),
    ),
    "chime": SoundProfile(
        "chime",
        "Chime",
        "Airy fifths with a clean, spacious finish.",
        ((523, 40), (659, 42), (784, 72)),
        ((659, 38), (523, 52)),
        ((784, 40), (659, 42), (1047, 80)),
        ((392, 65), (311, 90)),
    ),
    "synth": SoundProfile(
        "synth",
        "Synth Rise",
        "Tighter electronic sweep for a more technical feel.",
        ((220, 28), (330, 28), (494, 32), (740, 42)),
        ((494, 28), (330, 38)),
        ((740, 28), (494, 28), (330, 32), (220, 42)),
        ((185, 48), (277, 48), (185, 80)),
    ),
    "arcade": SoundProfile(
        "arcade",
        "Arcade Pop",
        "Playful two-step confirmation without spoken branding.",
        ((660, 35), (990, 48)),
        ((990, 30), (660, 42)),
        ((990, 35), (1320, 58)),
        ((262, 55), (196, 70)),
    ),
    "soft": SoundProfile(
        "soft",
        "Soft Tap",
        "Low-distraction rounded cue for meetings and streams.",
        ((440, 30), (554, 48)),
        ((554, 28), (440, 44)),
        ((554, 30), (440, 48)),
        ((277, 70),),
    ),
    "silent": SoundProfile(
        "silent",
        "Silent",
        "Visual cursor effects only.",
        (),
        (),
        (),
        (),
    ),
}

SOUND_PROFILE_CHOICES = tuple(SOUND_PROFILES)
CURSOR_STYLE_CHOICES = ("orbit", "ripple", "comet", "spectrum", "minimal")
CURSOR_STYLE_LABELS = {
    "orbit": "Orbit Halo",
    "ripple": "Ripple Rings",
    "comet": "Comet Sweep",
    "spectrum": "Spectrum Burst",
    "minimal": "Minimal Pulse",
}

POWER_RING_STYLE = "power"


def sound_profile(key: str) -> SoundProfile:
    return SOUND_PROFILES.get(str(key).strip().casefold(), SOUND_PROFILES["pulse"])


def clamp_intensity(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(max(number, 0.35), 1.5)


__all__ = [
    "CURSOR_STYLE_CHOICES",
    "CURSOR_STYLE_LABELS",
    "POWER_RING_STYLE",
    "SOUND_PROFILE_CHOICES",
    "SOUND_PROFILES",
    "SoundProfile",
    "clamp_intensity",
    "sound_profile",
]
