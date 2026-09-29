from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

MODEL_ID = "nemotron-3.5-asr-streaming-0.6b"
DISPLAY_NAME = "Nemotron 3.5 ASR Streaming 0.6B"
REPOSITORY = f"nvidia/{MODEL_ID}"
REVISION = "1c8deaecc64b91f034d73e08dd8b64625eb3395d"
WEIGHTS_BYTES = 2_552_062_944
WEIGHTS_SHA256 = "9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174"
COMPANION_FILES = {
    "config.json": (
        1_376,
        "62d186fd91f518e00e7867500f1f5819225e8ee95ea3e21b546514bf2048e845",
    ),
    "generation_config.json": (
        193,
        "993e5d4cb74a6fe9d6e7084a76b3313c1446740679be4676570c23b664fdc07e",
    ),
    "processor_config.json": (
        2_519,
        "ec47870f1091ea4f25539208387b45b902c92d0e3f997a30061ef88f73437ab0",
    ),
    "tokenizer.json": (
        752_051,
        "3f3d481deb073b64c2082e8c7860d487a3a62774bf4e9e4faac83007e181f246",
    ),
    "tokenizer_config.json": (
        881,
        "5c641c5b3f50702a60082690d27c1ce7fcb5a92c4a624793bcae0f21eda3d6e0",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path.name}: expected {expected_bytes} bytes, got {path.stat().st_size}"
        )
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{path.name}: SHA-256 mismatch; expected {expected_sha256}, got {actual}"
        )


def atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def stage_weights(source: Path, target: Path) -> str:
    destination = target / "model.safetensors"
    if destination.exists():
        verify_file(destination, WEIGHTS_BYTES, WEIGHTS_SHA256)
        return "existing"
    try:
        os.link(source, destination)
        storage = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        storage = "copy"
    verify_file(destination, WEIGHTS_BYTES, WEIGHTS_SHA256)
    return storage


def stage_companions(target: Path, *, offline: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for name, (expected_bytes, expected_sha256) in COMPANION_FILES.items():
        destination = target / name
        if not destination.exists():
            if offline:
                raise FileNotFoundError(
                    f"{destination} is required; rerun without --offline to fetch it"
                )
            url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(expected_bytes + 1)
            if len(data) != expected_bytes:
                raise ValueError(
                    f"{name}: expected {expected_bytes} bytes, received {len(data)}"
                )
            json.loads(data.decode("utf-8"))
            atomic_write(destination, data)
        verify_file(destination, expected_bytes, expected_sha256)
        records.append(
            {
                "name": name,
                "bytes": expected_bytes,
                "sha256": expected_sha256,
            }
        )
    return records


def main() -> int:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    parser = argparse.ArgumentParser(
        description="Stage the explicitly supplied Nemotron checkpoint for local-only use."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home()
        / "Downloads"
        / "nemotron-3.5-asr-streaming-0.6b .safetensors",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=local_app_data / "DrewLocalVoice" / "models" / MODEL_ID,
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Verify already-staged files without any network access.",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    target = args.target.expanduser().resolve()
    verify_file(source, WEIGHTS_BYTES, WEIGHTS_SHA256)
    target.mkdir(parents=True, exist_ok=True)
    storage = stage_weights(source, target)
    companions = stage_companions(target, offline=args.offline)
    manifest = {
        "schema_version": 1,
        "display_name": DISPLAY_NAME,
        "purpose": "primary_speech_recognition",
        "backend": "transformers.AutoModelForRNNT",
        "network_at_runtime": "blocked_local_files_only",
        "source_repository": REPOSITORY,
        "source_revision": REVISION,
        "source_weights_path": str(source),
        "runtime_model_directory": str(target),
        "weights": {
            "name": "model.safetensors",
            "bytes": WEIGHTS_BYTES,
            "sha256": WEIGHTS_SHA256,
            "storage": storage,
        },
        "companion_files": companions,
    }
    atomic_write(
        target / "GLITCHSCRIBE_MODEL_MANIFEST.json",
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "model": DISPLAY_NAME,
                "source": str(source),
                "runtime_directory": str(target),
                "weights_storage": storage,
                "runtime_network": "blocked_local_files_only",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
