"""Deterministic, no-model-load speech backend eligibility planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPULabSnapshot:
    index: int
    name: str
    compute_capability: str
    total_vram_mb: int
    used_vram_mb: int
    free_vram_mb: int


@dataclass(frozen=True)
class RuntimeCapabilities:
    ctranslate2_version: str = ""
    ctranslate2_cuda_compute_types: tuple[str, ...] = ()
    onnxruntime_version: str = ""
    onnxruntime_providers: tuple[str, ...] = ()
    sherpa_onnx_version: str = ""
    nemo_version: str = ""


@dataclass(frozen=True)
class BackendCandidate:
    candidate_id: str
    backend: str
    model: str
    required_free_vram_mb: int
    artifact_size_mb: int
    streaming_mode: str
    hardware_compatible: bool
    runtime_ready: bool
    status: str
    blockers: tuple[str, ...]
    evidence_urls: tuple[str, ...]


@dataclass(frozen=True)
class SpeechBackendReport:
    gpu: GPULabSnapshot
    runtime: RuntimeCapabilities
    candidates: tuple[BackendCandidate, ...]
    any_model_load_allowed: bool
    decision: str

    def to_dict(self) -> dict:
        return asdict(self)


_CTRANSLATE_EVIDENCE = (
    "https://github.com/SYSTRAN/faster-whisper/blob/master/README.md",
)
_PARAKEET_EVIDENCE = (
    "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
    "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html",
    "https://k2-fsa.github.io/sherpa/onnx/install/windows/build-cuda.html",
)


def _compute_capability_at_least(value: str, minimum: tuple[int, int]) -> bool:
    try:
        major_text, minor_text = str(value).strip().split(".", 1)
        current = (int(major_text), int(minor_text))
    except (TypeError, ValueError):
        return False
    return current >= minimum


def _candidate(
    *,
    gpu: GPULabSnapshot,
    candidate_id: str,
    backend: str,
    model: str,
    required_free_vram_mb: int,
    artifact_size_mb: int,
    streaming_mode: str,
    runtime_ready: bool,
    runtime_blockers: tuple[str, ...],
    evidence_urls: tuple[str, ...],
) -> BackendCandidate:
    hardware_compatible = _compute_capability_at_least(
        gpu.compute_capability,
        (7, 5),
    )
    blockers: list[str] = []
    if gpu.free_vram_mb < required_free_vram_mb:
        blockers.append(
            f"CUDA {gpu.index} has {gpu.free_vram_mb} MiB free; "
            f"{required_free_vram_mb} MiB guarded free VRAM required"
        )
    if not hardware_compatible:
        blockers.append("GPU compute capability 7.5 or newer is required")
    blockers.extend(runtime_blockers)
    if gpu.free_vram_mb < required_free_vram_mb:
        status = "blocked_headroom"
    elif not hardware_compatible:
        status = "unsupported_hardware"
    elif not runtime_ready:
        status = "runtime_missing"
    else:
        status = "eligible"
    return BackendCandidate(
        candidate_id=candidate_id,
        backend=backend,
        model=model,
        required_free_vram_mb=required_free_vram_mb,
        artifact_size_mb=artifact_size_mb,
        streaming_mode=streaming_mode,
        hardware_compatible=hardware_compatible,
        runtime_ready=runtime_ready,
        status=status,
        blockers=tuple(blockers),
        evidence_urls=evidence_urls,
    )


def evaluate_speech_backends(
    *,
    gpu: GPULabSnapshot,
    runtime: RuntimeCapabilities,
) -> SpeechBackendReport:
    """Classify candidates without downloading, loading, or evicting any model."""
    compute_types = set(runtime.ctranslate2_cuda_compute_types)
    ctranslate_ready = bool(runtime.ctranslate2_version) and bool(
        {"int8_float16", "float16", "int8"} & compute_types
    )
    ctranslate_blockers: list[str] = []
    if not runtime.ctranslate2_version:
        ctranslate_blockers.append("CTranslate2 runtime is not installed")
    if not ({"int8_float16", "float16", "int8"} & compute_types):
        ctranslate_blockers.append(
            "CTranslate2 exposes no supported CUDA speech compute type"
        )

    sherpa_ready = bool(runtime.sherpa_onnx_version) and (
        "CUDAExecutionProvider" in runtime.onnxruntime_providers
    )
    sherpa_blockers: list[str] = []
    if not runtime.sherpa_onnx_version:
        sherpa_blockers.append("sherpa-onnx runtime is not installed")
    if "CUDAExecutionProvider" not in runtime.onnxruntime_providers:
        sherpa_blockers.append("CUDAExecutionProvider is unavailable")

    candidates = (
        _candidate(
            gpu=gpu,
            candidate_id="faster-whisper-base-en",
            backend="faster-whisper/CTranslate2",
            model="base.en",
            required_free_vram_mb=2560,
            artifact_size_mb=148,
            streaming_mode="snapshot_incremental",
            runtime_ready=ctranslate_ready,
            runtime_blockers=tuple(ctranslate_blockers),
            evidence_urls=_CTRANSLATE_EVIDENCE,
        ),
        _candidate(
            gpu=gpu,
            candidate_id="faster-whisper-distil-large-v3.5",
            backend="faster-whisper/CTranslate2",
            model="distil-large-v3.5",
            required_free_vram_mb=5120,
            artifact_size_mb=1600,
            streaming_mode="snapshot_incremental",
            runtime_ready=ctranslate_ready,
            runtime_blockers=tuple(ctranslate_blockers),
            evidence_urls=_CTRANSLATE_EVIDENCE,
        ),
        _candidate(
            gpu=gpu,
            candidate_id="sherpa-parakeet-tdt-0.6b-v3-int8",
            backend="sherpa-onnx/ONNX Runtime CUDA",
            model="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            required_free_vram_mb=3072,
            artifact_size_mb=640,
            streaming_mode="simulated_streaming_from_offline_model",
            runtime_ready=sherpa_ready,
            runtime_blockers=tuple(sherpa_blockers),
            evidence_urls=_PARAKEET_EVIDENCE,
        ),
    )
    any_model_load_allowed = any(
        candidate.status == "eligible" for candidate in candidates
    )
    decision = (
        "eligible_candidates_require_explicit_benchmark"
        if any_model_load_allowed
        else "no_go_preserve_existing_workloads"
    )
    return SpeechBackendReport(
        gpu=gpu,
        runtime=runtime,
        candidates=candidates,
        any_model_load_allowed=any_model_load_allowed,
        decision=decision,
    )
