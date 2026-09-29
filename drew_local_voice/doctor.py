from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, TextIO


DEFAULT_IDLE_UNLOAD_MINUTES = 10.0


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    critical: bool = True

    def line(self) -> str:
        mark = "OK" if self.ok else "FAIL"
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    @property
    def ok(self) -> bool:
        return all(not c.critical or c.ok for c in self.checks)

    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def format_text(self) -> str:
        lines = [c.line() for c in self.checks]
        lines.append("")
        lines.append("RESULT: PASS" if self.ok else "RESULT: FAIL")
        return "\n".join(lines)


def normalize_idle_unload_minutes(value: Any, default: float = DEFAULT_IDLE_UNLOAD_MINUTES) -> float:
    """Clamp idle unload minutes; 0 disables idle unload."""
    try:
        minutes = float(value)
    except (TypeError, ValueError):
        return float(default)
    if minutes < 0:
        return 0.0
    if minutes > 24 * 60:
        return float(24 * 60)
    return minutes


def idle_unload_due(
    *,
    last_activity_monotonic: float,
    now_monotonic: float,
    idle_unload_minutes: float,
    model_loaded: bool,
    recording: bool = False,
    busy: bool = False,
) -> bool:
    """Pure timer: True when weights should be unloaded for Hermes coexistence."""
    minutes = normalize_idle_unload_minutes(idle_unload_minutes)
    if minutes <= 0 or not model_loaded or recording or busy:
        return False
    elapsed = float(now_monotonic) - float(last_activity_monotonic)
    return elapsed >= minutes * 60.0


def resolve_config_path(env: Optional[dict[str, str]] = None) -> Path:
    environ = env if env is not None else os.environ
    base = Path(environ.get("LOCALAPPDATA") or Path.home()) / "DrewLocalVoice"
    return base / "config.json"


def resolve_runtime_events_path(env: Optional[dict[str, str]] = None) -> Path:
    return resolve_config_path(env).with_name("runtime-events.jsonl")


def huggingface_model_cache_candidates(model_name: str) -> tuple[Path, ...]:
    """Likely local cache locations for a Faster-Whisper / HF model id."""
    name = str(model_name or "").strip()
    if not name:
        return ()
    path = Path(name)
    if path.exists():
        return (path.resolve(),)

    home = Path.home()
    hf_home = Path(os.environ.get("HF_HOME") or (home / ".cache" / "huggingface"))
    hub = hf_home / "hub"
    # faster-whisper uses Systran/faster-whisper-* style repos for built-ins
    slug = name.replace("/", "--")
    candidates = [
        hub / f"models--Systran--faster-whisper-{name}",
        hub / f"models--{slug}",
        Path(os.environ.get("LOCALAPPDATA", "")) / "DrewLocalVoice" / "models" / name,
    ]
    return tuple(p for p in candidates if str(p).strip("."))


def model_appears_loadable(model_name: str, *, path_exists: Callable[[Path], bool] | None = None) -> tuple[bool, str]:
    """Non-destructive: check local path or HF cache presence (does not load weights)."""
    exists = path_exists or (lambda p: p.exists())
    name = str(model_name or "").strip()
    if not name:
        return False, "model name is empty"
    direct = Path(name)
    if exists(direct):
        return True, f"local path exists: {direct}"
    hits = [p for p in huggingface_model_cache_candidates(name) if exists(p)]
    if hits:
        return True, f"cache present: {hits[0]}"
    return (
        False,
        f"no local path or HF cache found for {name!r} (first load may download if allowed)",
    )


def parse_hotkey_label(label: str) -> tuple[bool, str]:
    text = str(label or "").strip()
    if not text:
        return False, "hotkey is empty"
    if text == "Ctrl + Alt + Space":
        return True, "Ctrl+Alt+Space maps to MOD_CONTROL|MOD_ALT + VK_SPACE"
    import re

    if re.fullmatch(r"F(?:[8-9]|1[0-2])", text):
        return True, f"{text} is a supported function-key hotkey"
    return False, f"unsupported hotkey label: {text}"


def tail_text_file(path: Path, *, max_lines: int = 20, max_bytes: int = 32_768) -> str:
    if not path.is_file():
        return ""
    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def collect_mic_checks(
    devices: Sequence[dict[str, Any]],
    *,
    default_index: Optional[int],
    configured_index: Optional[int],
    configured_name: str = "",
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if not devices:
        checks.append(
            CheckResult("microphone", False, "no input devices found", critical=True)
        )
        return checks
    listing = "; ".join(
        f"[{d.get('index')}] {d.get('name')} ({d.get('api', '?')})" for d in devices[:12]
    )
    if len(devices) > 12:
        listing += f"; … +{len(devices) - 12} more"
    checks.append(
        CheckResult(
            "microphone.devices",
            True,
            f"{len(devices)} input device(s): {listing}",
            critical=False,
        )
    )
    checks.append(
        CheckResult(
            "microphone.default",
            default_index is not None,
            (
                f"index {default_index}"
                if default_index is not None
                else "no default input"
            ),
            critical=True,
        )
    )
    if configured_index is not None:
        match = next((d for d in devices if int(d.get("index", -1)) == int(configured_index)), None)
        ok = match is not None
        detail = (
            f"configured index {configured_index}"
            + (f" name={match.get('name')!r}" if match else " (missing)")
        )
        if configured_name:
            detail += f" saved_name={configured_name!r}"
        checks.append(CheckResult("microphone.configured", ok, detail, critical=True))
    return checks


def collect_compute_checks(
    *,
    cuda_devices: Sequence[Any] = (),
    torch_cuda: Optional[bool] = None,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if cuda_devices:
        names = []
        for d in cuda_devices:
            idx = getattr(d, "index", None)
            name = getattr(d, "name", str(d))
            free = getattr(d, "free_vram_mb", None)
            names.append(f"cuda:{idx} {name}" + (f" free={free}MiB" if free is not None else ""))
        checks.append(
            CheckResult(
                "compute.cuda",
                True,
                "; ".join(names),
                critical=False,
            )
        )
    else:
        checks.append(
            CheckResult(
                "compute.cuda",
                True,
                "no NVIDIA GPU via nvidia-smi (CPU path available)",
                critical=False,
            )
        )
    if torch_cuda is not None:
        checks.append(
            CheckResult(
                "compute.torch_cuda",
                True,
                "available" if torch_cuda else "not available",
                critical=False,
            )
        )
    checks.append(CheckResult("compute.cpu", True, "CPU available", critical=False))
    return checks


def build_doctor_report(
    *,
    config_path: Path,
    config_data: Optional[dict[str, Any]],
    devices: Sequence[dict[str, Any]],
    default_device_index: Optional[int],
    model_name: str,
    hotkey: str,
    cuda_devices: Sequence[Any] = (),
    runtime_events_path: Optional[Path] = None,
    faster_whisper_importable: bool = True,
    selftest: bool = False,
) -> DoctorReport:
    report = DoctorReport()
    report.add(
        CheckResult(
            "config.path",
            True,
            str(config_path),
            critical=False,
        )
    )
    if config_data is None and not config_path.is_file():
        report.add(
            CheckResult(
                "config.file",
                True,
                "missing (defaults will be used on first run)",
                critical=False,
            )
        )
    elif config_data is None:
        report.add(
            CheckResult("config.file", False, f"unreadable or invalid JSON: {config_path}")
        )
    else:
        report.add(
            CheckResult(
                "config.file",
                True,
                f"loaded ({len(config_data)} keys)",
                critical=False,
            )
        )

    configured_index = None
    configured_name = ""
    if isinstance(config_data, dict):
        raw_idx = config_data.get("device_index")
        if raw_idx is not None:
            try:
                configured_index = int(raw_idx)
            except (TypeError, ValueError):
                configured_index = None
        configured_name = str(config_data.get("device_name") or "")
        if not model_name:
            model_name = str(config_data.get("model_name") or "tiny.en")
        if not hotkey:
            hotkey = str(config_data.get("hotkey") or "Ctrl + Alt + Space")
        idle = config_data.get("idle_unload_minutes")
        if idle is None and isinstance(config_data.get("advanced_settings"), dict):
            idle = config_data["advanced_settings"].get("model.idle_unload_minutes")
        report.add(
            CheckResult(
                "model.idle_unload_minutes",
                True,
                str(normalize_idle_unload_minutes(idle if idle is not None else DEFAULT_IDLE_UNLOAD_MINUTES)),
                critical=False,
            )
        )

    report.checks.extend(
        collect_mic_checks(
            devices,
            default_index=default_device_index,
            configured_index=configured_index,
            configured_name=configured_name,
        )
    )

    loadable, load_detail = model_appears_loadable(model_name or "tiny.en")
    # Missing cache is warning for doctor (first-run download), critical only in selftest strict? keep non-critical
    report.add(
        CheckResult(
            "model.path",
            True if loadable else (not selftest),
            f"{model_name or 'tiny.en'} — {load_detail}",
            critical=selftest and not loadable,
        )
    )
    report.add(
        CheckResult(
            "model.faster_whisper",
            faster_whisper_importable,
            "importable" if faster_whisper_importable else "faster-whisper is not installed",
            critical=True,
        )
    )

    hk_ok, hk_detail = parse_hotkey_label(hotkey or "Ctrl + Alt + Space")
    report.add(CheckResult("hotkey.label", hk_ok, hk_detail, critical=True))

    report.checks.extend(collect_compute_checks(cuda_devices=cuda_devices))

    events_path = runtime_events_path or resolve_runtime_events_path()
    tail = tail_text_file(events_path, max_lines=12)
    if tail:
        report.add(
            CheckResult(
                "runtime_events.tail",
                True,
                f"{events_path}\n{tail}",
                critical=False,
            )
        )
    else:
        report.add(
            CheckResult(
                "runtime_events.tail",
                True,
                f"no events at {events_path}",
                critical=False,
            )
        )

    if selftest:
        report.add(
            CheckResult(
                "selftest.mode",
                True,
                "non-destructive checks only (no model load, no hotkey register)",
                critical=False,
            )
        )
    return report


def load_config_dict(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def run_doctor(
    *,
    selftest: bool = False,
    stream: Optional[TextIO] = None,
    list_devices: Optional[Callable[[], list[dict[str, Any]]]] = None,
    choose_default: Optional[Callable[[list[dict[str, Any]]], Optional[int]]] = None,
    probe_gpu: Optional[Callable[[], Sequence[Any]]] = None,
) -> int:
    """Run diagnostics and print a human-readable report. Returns process exit code."""
    out = stream or sys.stdout
    config_path = resolve_config_path()
    config_data = load_config_dict(config_path)

    devices: list[dict[str, Any]] = []
    default_index: Optional[int] = None
    try:
        if list_devices is not None:
            devices = list(list_devices())
        else:
            import sounddevice as sd  # local import keeps unit tests light

            from local_voice import choose_default_device, list_input_devices

            devices = list_input_devices()
            default_index = choose_default_device(devices)
    except Exception as exc:  # pragma: no cover - hardware dependent
        devices = []
        default_index = None
        _device_error = str(exc)
    else:
        _device_error = ""
        if choose_default is not None:
            default_index = choose_default(devices)

    cuda_devices: Sequence[Any] = ()
    try:
        if probe_gpu is not None:
            cuda_devices = probe_gpu()
        else:
            from local_voice import probe_gpu_inventory

            cuda_devices = probe_gpu_inventory()
    except Exception:
        cuda_devices = ()

    try:
        import faster_whisper  # noqa: F401

        fw_ok = True
    except ImportError:
        fw_ok = False

    model_name = ""
    hotkey = ""
    if config_data:
        model_name = str(config_data.get("model_name") or "")
        hotkey = str(config_data.get("hotkey") or "")

    report = build_doctor_report(
        config_path=config_path,
        config_data=config_data,
        devices=devices,
        default_device_index=default_index,
        model_name=model_name or "tiny.en",
        hotkey=hotkey or "Ctrl + Alt + Space",
        cuda_devices=cuda_devices,
        faster_whisper_importable=fw_ok,
        selftest=selftest,
    )
    if _device_error:
        report.add(
            CheckResult(
                "microphone.query",
                False,
                _device_error,
                critical=True,
            )
        )

    title = "GlitchScribe selftest" if selftest else "GlitchScribe doctor"
    print(f"{title}\n{'=' * len(title)}", file=out)
    print(report.format_text(), file=out)
    return report.exit_code()


def parse_cli_args(argv: Optional[Sequence[str]] = None) -> tuple[bool, bool, bool]:
    """Return (want_doctor, want_selftest, consumed).

    Supports:
      python -m drew_local_voice doctor
      python -m drew_local_voice selftest
      local_voice.py --doctor
      local_voice.py --selftest
    """
    args = list(sys.argv[1:] if argv is None else argv)
    want_doctor = False
    want_selftest = False
    for arg in args:
        lowered = arg.strip().casefold()
        if lowered in {"doctor", "--doctor"}:
            want_doctor = True
        elif lowered in {"selftest", "--selftest"}:
            want_selftest = True
            want_doctor = True
    return want_doctor, want_selftest, want_doctor or want_selftest


# Valid choices for the CLI overrides below. Kept in lock-step with the values
# already exposed in ``settings.py`` / ``dictation_overlay.py`` so we never
# accept an unknown style at the command line.
_VALID_BUBBLE_STYLES = {
    "minimal",
    "glitch_glass",
    "neon_edge",
    "high_contrast",
}
_VALID_THEMES = {"dark", "light", "system"}
_VALID_ACCENTS = {"violet", "cyan", "blue", "emerald", "rose"}


def parse_cli_overrides(
    argv: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Return a dict of UI overrides parsed from ``argv``.

    Supported flags (all optional, all safe to ignore if absent):

      ``--bubble-style <name>``  — one of ``minimal`` / ``glitch_glass`` /
                                   ``neon_edge`` / ``high_contrast``.
      ``--theme <name>``         — one of ``dark`` / ``light`` / ``system``.
      ``--accent <name>``        — one of ``violet`` / ``cyan`` / ``blue`` /
                                   ``emerald`` / ``rose``.
      ``--opacity <0..1>``       — bubble opacity as a float in [0, 1].

    Unknown values are silently dropped (caller falls back to saved settings).
    The returned dict uses settings-key names (``ui.bubble_visual_style``,
    ``ui.theme``, ``ui.accent_color``, ``ui.bubble_opacity``) so it can be
    applied directly to a ``SettingsRegistry``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    overrides: dict[str, str] = {}
    index = 0
    while index < len(args):
        raw = args[index]
        flag = str(raw).strip().casefold()
        if flag == "--bubble-style" and index + 1 < len(args):
            value = str(args[index + 1]).strip().casefold()
            if value in _VALID_BUBBLE_STYLES:
                overrides["ui.bubble_visual_style"] = value
            index += 2
            continue
        if flag == "--theme" and index + 1 < len(args):
            value = str(args[index + 1]).strip().casefold()
            if value in _VALID_THEMES:
                overrides["ui.theme"] = value
            index += 2
            continue
        if flag == "--accent" and index + 1 < len(args):
            value = str(args[index + 1]).strip().casefold()
            if value in _VALID_ACCENTS:
                overrides["ui.accent_color"] = value
            index += 2
            continue
        if flag == "--opacity" and index + 1 < len(args):
            try:
                parsed = float(str(args[index + 1]).strip())
            except (TypeError, ValueError):
                index += 2
                continue
            if 0.0 <= parsed <= 1.0:
                overrides["ui.bubble_opacity"] = f"{parsed:.3f}"
            index += 2
            continue
        index += 1
    return overrides
