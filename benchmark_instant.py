from __future__ import annotations

import json
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from faster_whisper.audio import decode_audio

from local_voice import (
    WhisperBackend,
    choose_runtime_profile,
    normalize_transcript,
    probe_gpu_free_vram_mb,
)
from streaming_dictation import StreamingDictation

ROOT = Path(__file__).resolve().parent
EXPECTED = normalize_transcript(
    "And so, my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country."
)


def main() -> None:
    free_before = probe_gpu_free_vram_mb(0)
    profile = choose_runtime_profile(free_before, free_before > 0)
    if profile.device != "cuda":
        raise RuntimeError(f"Instant GPU profile unavailable: {profile}")

    backend = WhisperBackend()
    warm = backend.preload_and_warm(profile)
    audio = decode_audio(str(ROOT / "artifacts" / "jfk.wav"), sampling_rate=16_000)
    slices = []
    for seconds in (1.5, 2.5, 5.0, 11.0):
        samples = np.asarray(audio[: int(seconds * 16_000)], dtype=np.float32)
        text, metrics = backend.transcribe_samples(samples, profile, fast=True)
        slices.append(
            {
                "audio_seconds": seconds,
                "inference_seconds": round(metrics["transcribe_seconds"], 4),
                "text": text,
            }
        )

    stream = StreamingDictation(backend, profile, min_audio_seconds=0.1)
    full = np.asarray(audio, dtype=np.float32)
    partial = stream.process_snapshot(full)
    quiet_final_audio = np.concatenate(
        [full, np.zeros(int(0.2 * 16_000), dtype=np.float32)]
    )
    quiet_final = stream.finalize(quiet_final_audio)

    forced = StreamingDictation(backend, profile, min_audio_seconds=0.1)
    forced.process_snapshot(full[: int(10.7 * 16_000)])
    forced_started = time.perf_counter()
    forced_final = forced.finalize(full)
    forced_wall = time.perf_counter() - forced_started

    ratio = SequenceMatcher(
        None, EXPECTED.casefold(), quiet_final.text.casefold()
    ).ratio()
    payload = {
        "profile": profile.__dict__,
        "free_vram_before_mb": free_before,
        "warmup": {key: round(value, 4) if isinstance(value, float) else value for key, value in warm.items()},
        "warm_slices": slices,
        "streaming": {
            "partial_text": partial.text if partial else "",
            "quiet_tail_reused": quiet_final.reused_partial,
            "quiet_stop_to_text_seconds": round(quiet_final.stop_to_text_seconds, 6),
            "forced_final_reused": forced_final.reused_partial,
            "forced_stop_to_text_seconds": round(forced_final.stop_to_text_seconds, 4),
            "forced_wall_seconds": round(forced_wall, 4),
            "final_text": quiet_final.text,
            "match_ratio": round(ratio, 4),
        },
    }

    by_seconds = {row["audio_seconds"]: row for row in slices}
    assert by_seconds[2.5]["inference_seconds"] < 0.35, payload
    assert by_seconds[11.0]["inference_seconds"] < 0.8, payload
    assert quiet_final.reused_partial, payload
    assert quiet_final.stop_to_text_seconds < 0.05, payload
    assert ratio >= 0.97, payload

    output = ROOT / "artifacts" / "instant-benchmark.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
