from __future__ import annotations

import argparse
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

from local_voice import WhisperBackend, normalize_transcript

EXPECTED = normalize_transcript(
    "And so, my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country."
)


def score(actual: str) -> float:
    return round(SequenceMatcher(None, EXPECTED.casefold(), actual.casefold()).ratio(), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model-benchmark.json"))
    args = parser.parse_args()
    results = []
    for model_name in ("tiny.en", "base.en", "small.en"):
        backend = WhisperBackend()
        started = time.perf_counter()
        text, metrics = backend.transcribe(args.audio, model_name, "cpu", "int8")
        results.append(
            {
                "model": model_name,
                "text": text,
                "match_ratio": score(text),
                "load_seconds": round(metrics["load_seconds"], 3),
                "transcribe_seconds": round(metrics["transcribe_seconds"], 3),
                "wall_seconds": round(time.perf_counter() - started, 3),
            }
        )
    payload = {"expected": EXPECTED, "device": "cpu", "compute_type": "int8", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
