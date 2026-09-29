from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

from drew_local_voice.model_lab import (
    GPULabSnapshot,
    RuntimeCapabilities,
    evaluate_speech_backends,
)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _probe_gpu(index: int) -> tuple[GPULabSnapshot, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=index,name,uuid,compute_cap,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    parts = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
    if len(parts) != 7:
        raise RuntimeError("nvidia-smi returned an unexpected GPU record")
    return (
        GPULabSnapshot(
            index=int(parts[0]),
            name=parts[1],
            compute_capability=parts[3],
            total_vram_mb=int(parts[4]),
            used_vram_mb=int(parts[5]),
            free_vram_mb=int(parts[6]),
        ),
        parts[2],
    )


def _probe_runtime(gpu_index: int) -> RuntimeCapabilities:
    ctranslate_version = _package_version("ctranslate2")
    compute_types: tuple[str, ...] = ()
    if ctranslate_version:
        import ctranslate2

        compute_types = tuple(
            sorted(ctranslate2.get_supported_compute_types("cuda", device_index=gpu_index))
        )
    providers: tuple[str, ...] = ()
    onnx_version = _package_version("onnxruntime-gpu") or _package_version("onnxruntime")
    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime

        providers = tuple(onnxruntime.get_available_providers())
    return RuntimeCapabilities(
        ctranslate2_version=ctranslate_version,
        ctranslate2_cuda_compute_types=compute_types,
        onnxruntime_version=onnx_version,
        onnxruntime_providers=providers,
        sherpa_onnx_version=_package_version("sherpa-onnx"),
        nemo_version=_package_version("nemo-toolkit"),
    )


def _probe_compute_owners(gpu_uuid: str) -> list[dict[str, object]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    owners: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4 or parts[0] != gpu_uuid:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        try:
            used_memory_mb: int | None = int(parts[3])
        except ValueError:
            used_memory_mb = None
        owners.append(
            {
                "pid": pid,
                "process_name": Path(parts[2]).name,
                "used_memory_mb": used_memory_mb,
            }
        )
    return owners


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe RTX speech backend eligibility without loading a model."
    )
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/rtx2070-speech-backend-evaluation.json"),
    )
    args = parser.parse_args()

    gpu, gpu_uuid = _probe_gpu(args.gpu_index)
    runtime = _probe_runtime(args.gpu_index)
    report = evaluate_speech_backends(gpu=gpu, runtime=runtime)
    payload = report.to_dict()
    payload.update(
        {
            "schema_version": 1,
            "observed_at_unix": time.time(),
            "target_gpu_uuid": gpu_uuid,
            "compute_owners": _probe_compute_owners(gpu_uuid),
            "model_load_performed": False,
            "downloads_performed": False,
            "existing_workloads_evicted": False,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
