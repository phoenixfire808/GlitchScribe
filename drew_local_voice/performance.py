from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class PerformancePreset:
    id: str
    label: str
    description: str
    settings: dict[str, Any]
    experimental: bool = False


class PerformancePresetRegistry:
    def __init__(self, presets: Iterable[PerformancePreset] = ()) -> None:
        self._presets: dict[str, PerformancePreset] = {}
        for preset in presets:
            self.register(preset)

    def register(self, preset: PerformancePreset, *, replace: bool = False) -> None:
        if not preset.id:
            raise ValueError("performance preset id is required")
        if not replace and preset.id in self._presets:
            raise KeyError(f"performance preset already registered: {preset.id}")
        self._presets[preset.id] = preset

    def get(self, preset_id: str) -> PerformancePreset:
        return self._presets[preset_id]

    def all(self) -> tuple[PerformancePreset, ...]:
        return tuple(self._presets.values())

    def settings_for(self, preset_id: str) -> dict[str, Any]:
        return dict(self.get(preset_id).settings)


def built_in_performance_presets() -> tuple[PerformancePreset, ...]:
    return (
        PerformancePreset(
            "lean_core",
            "Lean Core",
            "Fast dependable path; record once, run one final pass, then insert once.",
            {
                "streaming.enabled": True,
                "streaming.interval_seconds": 0.18,
                "streaming.minimum_audio_seconds": 0.45,
                "streaming.minimum_new_audio_seconds": 0.14,
                "streaming.stability_lag_seconds": 0.32,
                "streaming.holdback_words": 1,
                "streaming.maximum_partial_audio_seconds": 28.0,
                "typing.live_insert": False,
                "ai.cleanup_enabled": False,
                "ai.cleanup_nearby_context": False,
                "endpoint.silence_seconds": 1.5,
                "endpoint.minimum_speech_seconds": 0.22,
                "endpoint.rms_threshold": 0.002,
                "endpoint.capture_tail_grace_seconds": 0.12,
                "finalization.decoding_mode": "fast_full_buffer",
                "finalization.allow_quiet_tail_reuse": False,
                "ui.meter_interval_ms": 60.0,
                "ui.streaming_bubble": True,
                "audio.callback_block_ms": 60.0,
            },
        ),
        PerformancePreset(
            "reliable_stream",
            "Reliable Stream",
            "Protect sentence tails and always verify the complete recording.",
            {
                "streaming.enabled": True,
                "streaming.interval_seconds": 0.28,
                "streaming.minimum_audio_seconds": 0.60,
                "streaming.minimum_new_audio_seconds": 0.22,
                "streaming.stability_lag_seconds": 0.40,
                "streaming.holdback_words": 2,
                "streaming.maximum_partial_audio_seconds": 40.0,
                "endpoint.silence_seconds": 3.0,
                "endpoint.minimum_speech_seconds": 0.30,
                "endpoint.rms_threshold": 0.002,
                "endpoint.capture_tail_grace_seconds": 0.40,
                "finalization.decoding_mode": "fast_full_buffer",
                "finalization.allow_quiet_tail_reuse": False,
                "ui.meter_interval_ms": 70.0,
                "audio.callback_block_ms": 80.0,
            },
        ),
        PerformancePreset(
            "balanced",
            "Balanced",
            "Faster live updates with complete final-buffer verification.",
            {
                "streaming.enabled": True,
                "streaming.interval_seconds": 0.22,
                "streaming.minimum_audio_seconds": 0.50,
                "streaming.minimum_new_audio_seconds": 0.16,
                "streaming.stability_lag_seconds": 0.32,
                "streaming.holdback_words": 1,
                "streaming.maximum_partial_audio_seconds": 32.0,
                "endpoint.silence_seconds": 1.8,
                "endpoint.minimum_speech_seconds": 0.25,
                "endpoint.rms_threshold": 0.002,
                "endpoint.capture_tail_grace_seconds": 0.22,
                "finalization.decoding_mode": "fast_full_buffer",
                "finalization.allow_quiet_tail_reuse": False,
                "ui.meter_interval_ms": 70.0,
                "audio.callback_block_ms": 80.0,
            },
        ),
        PerformancePreset(
            "instant",
            "Instant",
            "Aggressive live cadence while retaining a complete final pass.",
            {
                "streaming.enabled": True,
                "streaming.interval_seconds": 0.15,
                "streaming.minimum_audio_seconds": 0.40,
                "streaming.minimum_new_audio_seconds": 0.12,
                "streaming.stability_lag_seconds": 0.24,
                "streaming.holdback_words": 1,
                "streaming.maximum_partial_audio_seconds": 28.0,
                "endpoint.silence_seconds": 1.5,
                "endpoint.minimum_speech_seconds": 0.22,
                "endpoint.rms_threshold": 0.002,
                "endpoint.capture_tail_grace_seconds": 0.18,
                "finalization.decoding_mode": "fast_full_buffer",
                "finalization.allow_quiet_tail_reuse": False,
                "ui.meter_interval_ms": 50.0,
                "ui.streaming_bubble": True,
                "audio.callback_block_ms": 50.0,
            },
        ),
        PerformancePreset(
            "experimental_turbo",
            "Experimental Turbo",
            "Minimum latency. Quiet-partial reuse may omit a final word; not recommended on stream.",
            {
                "streaming.enabled": True,
                "streaming.interval_seconds": 0.12,
                "streaming.minimum_audio_seconds": 0.30,
                "streaming.minimum_new_audio_seconds": 0.08,
                "streaming.stability_lag_seconds": 0.12,
                "streaming.holdback_words": 0,
                "streaming.maximum_partial_audio_seconds": 20.0,
                "endpoint.silence_seconds": 1.0,
                "endpoint.minimum_speech_seconds": 0.15,
                "endpoint.rms_threshold": 0.002,
                "endpoint.capture_tail_grace_seconds": 0.10,
                "finalization.decoding_mode": "reuse_when_quiet",
                "finalization.allow_quiet_tail_reuse": True,
                "ui.meter_interval_ms": 45.0,
                "audio.callback_block_ms": 40.0,
            },
            experimental=True,
        ),
    )
