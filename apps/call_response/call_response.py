"""Draft-only local phone-call response companion for GlitchScribe.

The phone remains the call transport. This module captures audio from an
explicitly selected Windows input endpoint, transcribes bounded utterances with
the local Parakeet ONNX model, and asks an already-running loopback llama.cpp
server for a short response draft. It never records to disk, sends a message,
speaks a response, or inserts text automatically.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np


SAMPLE_RATE = 16_000
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
DEFAULT_LLM_MODEL = "nanbeige4.2-3b-local"
DEFAULT_PARakeet_DIR = (
    Path.home()
    / ".buzz"
    / "models"
    / "parakeet-tdt-ctc-110m-en"
)
DEFAULT_PARakeet_MODEL = DEFAULT_PARakeet_DIR / "model.int8.onnx"
DEFAULT_PARakeet_TOKENS = DEFAULT_PARakeet_DIR / "tokens.txt"
DEFAULT_RMS_THRESHOLD = 0.006
DEFAULT_SILENCE_SECONDS = 0.80
DEFAULT_MIN_UTTERANCE_SECONDS = 0.45
MAX_DRAFT_CHARACTERS = 2_000


RESPONSE_SYSTEM_PROMPT = """You are Drew's private, local, draft-only assistant during a high-stakes OSHA retaliation and ADR phone call.

The user's stated objective is a complete written negotiated separation and economic resolution, while preserving the ability to review any return-to-work proposal. Help Drew protect that objective without making admissions or accepting terms verbally.

Rules:
- Return one concise sentence or two that Drew can say aloud, followed by at most one short question when useful.
- Draft only. Never send, call, record, speak, insert, settle, withdraw, release, resign, or agree on Drew's behalf.
- Use only facts in the supplied conversation. Never invent facts, legal conclusions, deadlines, dollar amounts, losses, or employer positions.
- Distinguish documented loss, an estimate, an opening demand, a proposal, a counteroffer, a private threshold, and an accepted term. Never disclose a private threshold.
- If a number or term is unclear, ask for the complete proposal in writing instead of guessing.
- Preserve optionality: do not say that Drew refuses to return; say that complete written terms are needed and separation is preferred.
- Do not treat a partial wage figure as the entire resolution. Keep non-money terms and a complete written proposal open.
- Prefer calm, factual language and avoid speculation, threats, exaggeration, or legal advice.
"""


def validate_loopback_endpoint(endpoint: str) -> str:
    parsed = urlparse(str(endpoint).strip())
    if parsed.scheme != "http":
        raise ValueError("the local response endpoint must use plain HTTP")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the response endpoint must be loopback-only")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("the response endpoint cannot contain credentials or query data")
    if parsed.path not in {"", "/"}:
        raise ValueError("the response endpoint must be an origin without a path")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("the response endpoint has an invalid port") from exc
    return str(endpoint).strip().rstrip("/")


def strip_reasoning(text: str) -> str:
    """Remove hidden reasoning wrappers before a draft reaches the UI."""
    value = str(text or "")
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<analysis>.*?</analysis>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"^\s*<think>.*$", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:MAX_DRAFT_CHARACTERS]


def decode_ctc_ids(
    ids: Iterable[int],
    tokens: dict[int, str],
    *,
    blank_id: int,
) -> str:
    """Greedy CTC collapse for the local Parakeet vocabulary."""
    pieces: list[str] = []
    previous: Optional[int] = None
    for raw_id in ids:
        token_id = int(raw_id)
        if token_id != previous and token_id != int(blank_id):
            token = tokens.get(token_id, "")
            if token not in {"<unk>", "<blk>"}:
                pieces.append(token)
        previous = token_id
    text = "".join(pieces).replace("\u2581", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_tokens(path: Path) -> tuple[dict[int, str], int]:
    tokens: dict[int, str] = {}
    blank_id: Optional[int] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        token, raw_id = line.rsplit(" ", 1)
        token_id = int(raw_id)
        tokens[token_id] = token
        if token == "<blk>":
            blank_id = token_id
    if blank_id is None:
        raise ValueError("Parakeet token file does not define <blk>")
    return tokens, blank_id


def resample_audio(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    source = max(1, int(source_rate))
    target = max(1, int(target_rate))
    if not audio.size or source == target:
        return np.ascontiguousarray(audio, dtype=np.float32)
    target_size = max(1, int(round(audio.size * target / source)))
    source_x = np.arange(audio.size, dtype=np.float32)
    target_x = np.linspace(0, max(0, audio.size - 1), target_size, dtype=np.float32)
    return np.ascontiguousarray(np.interp(target_x, source_x, audio), dtype=np.float32)


def build_response_messages(
    history: Iterable[dict[str, str]],
    latest_transcript: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": RESPONSE_SYSTEM_PROMPT}
    ]
    for item in list(history)[-8:]:
        role = item.get("role", "user")
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", "")).strip()
        if content:
            messages.append({"role": role, "content": content[:1_200]})
    latest = str(latest_transcript or "").strip()
    if latest:
        messages.append({"role": "user", "content": latest[:1_200]})
    return messages


class ParakeetCTC:
    """Small CPU-capable Parakeet ONNX CTC wrapper."""

    def __init__(self, model_path: Path, tokens_path: Path) -> None:
        self.model_path = Path(model_path)
        self.tokens_path = Path(tokens_path)
        self._session = None
        self._extractor = None
        self._tokens: dict[int, str] = {}
        self._blank_id = 1024
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        if self.loaded:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Parakeet model not found: {self.model_path}")
        if not self.tokens_path.is_file():
            raise FileNotFoundError(f"Parakeet tokens not found: {self.tokens_path}")
        import onnxruntime as ort
        from faster_whisper.feature_extractor import FeatureExtractor

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(8, os.cpu_count() or 4))
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._extractor = FeatureExtractor(
            feature_size=80,
            sampling_rate=SAMPLE_RATE,
            hop_length=160,
            chunk_length=30,
            n_fft=400,
        )
        self._tokens, self._blank_id = load_tokens(self.tokens_path)

    def transcribe(self, samples: np.ndarray, source_rate: int) -> tuple[str, float]:
        audio = resample_audio(samples, source_rate, SAMPLE_RATE)
        if not audio.size:
            return "", 0.0
        audio = audio[-30 * SAMPLE_RATE :]
        with self._lock:
            self.load()
            started = time.perf_counter()
            features = np.asarray(self._extractor(audio), dtype=np.float32)[None, :, :]
            output = self._session.run(
                None,
                {
                    "audio_signal": features,
                    "length": np.asarray([features.shape[2]], dtype=np.int64),
                },
            )[0]
            ids = np.asarray(output).argmax(axis=-1).reshape(-1)
            text = decode_ctc_ids(ids, self._tokens, blank_id=self._blank_id)
            return text, time.perf_counter() - started


class CaptureBuffer:
    """In-memory mono capture with explicit rotate/discard boundaries."""

    def __init__(self, device_index: int, *, max_segment_seconds: float = 60.0) -> None:
        self.device_index = int(device_index)
        self.max_segment_seconds = max(5.0, float(max_segment_seconds))
        self.capture_rate = SAMPLE_RATE
        self._chunks: list[np.ndarray] = []
        self._sample_count = 0
        self._lock = threading.Lock()
        self._stream = None
        self._latest_rms = 0.0

    @property
    def latest_rms(self) -> float:
        with self._lock:
            return float(self._latest_rms)

    @property
    def duration_seconds(self) -> float:
        with self._lock:
            return self._sample_count / float(max(1, self.capture_rate))

    def _callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info, status
        chunk = np.asarray(indata[:, 0], dtype=np.float32).copy()
        if not chunk.size:
            return
        with self._lock:
            self._chunks.append(chunk)
            self._sample_count += int(chunk.size)
            self._latest_rms = float(np.sqrt(np.mean(np.square(chunk))))
            max_samples = int(self.capture_rate * self.max_segment_seconds)
            while self._sample_count > max_samples and self._chunks:
                removed = self._chunks.pop(0)
                self._sample_count -= int(removed.size)

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        info = sd.query_devices(self.device_index)
        self.capture_rate = int(round(float(info.get("default_samplerate") or SAMPLE_RATE)))
        self._stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.capture_rate,
            dtype="float32",
            blocksize=max(1, int(self.capture_rate * 0.05)),
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

    def rotate(self) -> tuple[np.ndarray, int]:
        with self._lock:
            chunks = self._chunks
            self._chunks = []
            self._sample_count = 0
            self._latest_rms = 0.0
            rate = self.capture_rate
        if not chunks:
            return np.zeros(0, dtype=np.float32), rate
        return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32), rate

    def discard(self) -> None:
        self.rotate()


def list_input_devices() -> list[tuple[int, str]]:
    import sounddevice as sd

    devices: list[tuple[int, str]] = []
    for index, item in enumerate(sd.query_devices()):
        if int(item.get("max_input_channels", 0)) > 0:
            devices.append((index, str(item.get("name", f"Input {index}"))))
    return devices


def preferred_input_index(devices: list[tuple[int, str]]) -> Optional[int]:
    preferences = (
        "Voicemeeter Out B2",
        "Voicemeeter Out B1",
        "Microphone (onn",
        "Microphone (fifine",
        "Microphone (HD Pro Webcam",
    )
    for needle in preferences:
        for index, name in devices:
            if needle.casefold() in name.casefold():
                return index
    return devices[0][0] if devices else None


class LocalDraftClient:
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_LLM_MODEL,
        *,
        timeout_seconds: float = 12.0,
    ) -> None:
        self.endpoint = validate_loopback_endpoint(endpoint)
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("local response model is required")
        self.timeout_seconds = max(2.0, min(30.0, float(timeout_seconds)))

    def health(self) -> str:
        request = Request(f"{self.endpoint}/v1/models", method="GET")
        with urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        rows = data.get("data") or data.get("models") or []
        names = [str(row.get("id") or row.get("name")) for row in rows if isinstance(row, dict)]
        if names and self.model not in names:
            raise RuntimeError(
                f"local model {self.model!r} is not advertised; available models are local-only"
            )
        return self.model

    def draft(
        self,
        history: Iterable[dict[str, str]],
        transcript: str,
    ) -> tuple[str, float]:
        messages = build_response_messages(history, transcript)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.15,
            "max_tokens": 220,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = Request(
            f"{self.endpoint}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        raw = data["choices"][0]["message"]["content"]
        draft = strip_reasoning(raw)
        if not draft:
            raise RuntimeError("local model returned no visible draft")
        return draft, time.perf_counter() - started


class CallResponseEngine:
    """Capture, segment, transcribe, and draft without disk or outbound actions."""

    def __init__(
        self,
        *,
        device_index: int,
        model_path: Path = DEFAULT_PARakeet_MODEL,
        tokens_path: Path = DEFAULT_PARakeet_TOKENS,
        endpoint: str = DEFAULT_ENDPOINT,
        llm_model: str = DEFAULT_LLM_MODEL,
        rms_threshold: float = DEFAULT_RMS_THRESHOLD,
        silence_seconds: float = DEFAULT_SILENCE_SECONDS,
        min_utterance_seconds: float = DEFAULT_MIN_UTTERANCE_SECONDS,
        on_status: Optional[Callable[[str], None]] = None,
        on_transcript: Optional[Callable[[str, float], None]] = None,
        on_draft: Optional[Callable[[str, float], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.capture = CaptureBuffer(device_index)
        self.stt = ParakeetCTC(model_path, tokens_path)
        self.client = LocalDraftClient(endpoint, llm_model)
        self.rms_threshold = max(0.0005, float(rms_threshold))
        self.silence_seconds = max(0.25, float(silence_seconds))
        self.min_utterance_seconds = max(0.2, float(min_utterance_seconds))
        self.on_status = on_status
        self.on_transcript = on_transcript
        self.on_draft = on_draft
        self.on_error = on_error
        self._history: deque[dict[str, str]] = deque(maxlen=8)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._speech_seen = False
        self._last_voice_at = 0.0

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _emit_status(self, value: str) -> None:
        if self.on_status:
            self.on_status(str(value))

    def _emit_error(self, value: str) -> None:
        if self.on_error:
            self.on_error(str(value))

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._speech_seen = False
        self._last_voice_at = 0.0
        self._thread = threading.Thread(
            target=self._run,
            name="glitchscribe-call-response",
            daemon=False,
        )
        self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        self._stop.set()
        self.capture.stop()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, float(timeout)))
        self._thread = None if not thread or not thread.is_alive() else thread

    def _run(self) -> None:
        try:
            self._emit_status("LOADING LOCAL STT")
            self.stt.load()
            if self._stop.is_set():
                return
            self._emit_status("CHECKING LOOPBACK RESPONSE MODEL")
            self.client.health()
            if self._stop.is_set():
                return
            self.capture.start()
            self._emit_status("LISTENING • DRAFT ONLY")
            while not self._stop.wait(0.10):
                now = time.monotonic()
                if self.capture.latest_rms >= self.rms_threshold:
                    self._speech_seen = True
                    self._last_voice_at = now
                duration = self.capture.duration_seconds
                if self._speech_seen and now - self._last_voice_at >= self.silence_seconds:
                    self._flush_segment()
                elif not self._speech_seen and duration >= 2.0:
                    self.capture.discard()
                elif duration >= 55.0:
                    self._flush_segment()
            if self._speech_seen:
                self._flush_segment()
        except (HTTPError, URLError, OSError, RuntimeError, ValueError) as exc:
            self._emit_error(f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            self._emit_error(f"{type(exc).__name__}: local call session failed")
        finally:
            self.capture.stop()
            self._emit_status("STOPPED")

    def _flush_segment(self) -> None:
        audio, rate = self.capture.rotate()
        was_speech = self._speech_seen
        self._speech_seen = False
        self._last_voice_at = 0.0
        if not was_speech or audio.size / float(max(1, rate)) < self.min_utterance_seconds:
            return
        self._emit_status("TRANSCRIBING LOCALLY")
        transcript, stt_seconds = self.stt.transcribe(audio, rate)
        if not transcript:
            self._emit_status("LISTENING • NO CLEAR SPEECH")
            return
        if self.on_transcript:
            self.on_transcript(transcript, stt_seconds)
        self._emit_status("DRAFTING LOCALLY")
        draft, llm_seconds = self.client.draft(self._history, transcript)
        self._history.append({"role": "user", "content": transcript})
        self._history.append({"role": "assistant", "content": draft})
        if self.on_draft:
            self.on_draft(draft, llm_seconds)
        self._emit_status("LISTENING • DRAFT READY")


@dataclass
class _UiDevice:
    index: int
    label: str


class CallResponseWindow:
    def __init__(
        self,
        root,
        *,
        model_path: Path,
        tokens_path: Path,
        endpoint: str,
        llm_model: str,
        requested_device: Optional[int] = None,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.root.title("GlitchScribe — OSHA Call Response — DRAFT ONLY")
        self.root.geometry("760x700")
        self.root.minsize(640, 560)
        self.model_path = Path(model_path)
        self.tokens_path = Path(tokens_path)
        self.endpoint = endpoint
        self.llm_model = llm_model
        self.engine: Optional[CallResponseEngine] = None
        self.devices = [
            _UiDevice(index, name) for index, name in list_input_devices()
        ]
        self.device_by_label = {f"{item.index}: {item.label}": item for item in self.devices}
        self.device_var = tk.StringVar()
        self.status_var = tk.StringVar(value="READY • no microphone capture")
        self.consent_var = tk.BooleanVar(value=False)

        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="LOCAL CALL RESPONSE — DRAFT ONLY",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Phone stays on speaker. Audio is processed in memory; nothing is "
                "recorded, spoken, sent, or inserted automatically."
            ),
            wraplength=700,
        ).pack(anchor="w", pady=(4, 12))

        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        ttk.Label(controls, text="Input endpoint:").pack(side="left")
        self.device_box = ttk.Combobox(
            controls,
            textvariable=self.device_var,
            values=list(self.device_by_label),
            state="readonly",
            width=72,
        )
        self.device_box.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(controls, text="Refresh", command=self._refresh_devices).pack(side="left")

        model_text = (
            f"STT: Parakeet ONNX (CPU)  •  response: {llm_model} via {endpoint}"
        )
        ttk.Label(outer, text=model_text, wraplength=700).pack(anchor="w", pady=(8, 4))
        ttk.Checkbutton(
            outer,
            text="I have confirmed that I may capture/transcribe this call.",
            variable=self.consent_var,
            command=self._update_buttons,
        ).pack(anchor="w", pady=(0, 8))

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(0, 8))
        self.start_button = ttk.Button(buttons, text="Start session", command=self._start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(
            buttons, text="Stop session", command=self._stop, state="disabled"
        )
        self.stop_button.pack(side="left", padx=8)
        ttk.Button(buttons, text="Copy current draft", command=self._copy_draft).pack(
            side="left"
        )
        ttk.Label(buttons, textvariable=self.status_var).pack(side="right")

        ttk.Label(outer, text="Latest caller transcript").pack(anchor="w")
        self.transcript_text = tk.Text(outer, height=8, wrap="word", state="disabled")
        self.transcript_text.pack(fill="both", expand=True, pady=(2, 8))
        ttk.Label(outer, text="Suggested response to review and say yourself").pack(
            anchor="w"
        )
        self.draft_text = tk.Text(outer, height=7, wrap="word", state="disabled")
        self.draft_text.pack(fill="both", expand=True, pady=(2, 0))

        if requested_device is not None:
            selected = next(
                (item for item in self.devices if item.index == int(requested_device)),
                None,
            )
        else:
            preferred = preferred_input_index([(item.index, item.label) for item in self.devices])
            selected = next(
                (item for item in self.devices if item.index == preferred),
                None,
            )
        if selected is not None:
            self.device_var.set(f"{selected.index}: {selected.label}")
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _refresh_devices(self) -> None:
        current = self.device_var.get()
        self.devices = [_UiDevice(index, name) for index, name in list_input_devices()]
        self.device_by_label = {
            f"{item.index}: {item.label}": item for item in self.devices
        }
        self.device_box["values"] = list(self.device_by_label)
        if current in self.device_by_label:
            self.device_var.set(current)
        elif self.devices:
            preferred = preferred_input_index([(item.index, item.label) for item in self.devices])
            selected = next((item for item in self.devices if item.index == preferred), self.devices[0])
            self.device_var.set(f"{selected.index}: {selected.label}")

    def _update_buttons(self) -> None:
        if self.engine is None or not self.engine.is_running:
            self.start_button.configure(
                state="normal" if self.consent_var.get() and self.device_var.get() else "disabled"
            )

    def _append(self, widget, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _ui(self, callback: Callable[[], None]) -> None:
        try:
            self.root.after(0, callback)
        except Exception:
            pass

    def _set_status(self, value: str) -> None:
        self._ui(lambda: self.status_var.set(value))

    def _set_transcript(self, text: str, seconds: float) -> None:
        self._ui(
            lambda: self._append(
                self.transcript_text,
                f"{text}\n\n(local STT: {seconds:.2f}s)",
            )
        )

    def _set_draft(self, text: str, seconds: float) -> None:
        self._ui(
            lambda: self._append(
                self.draft_text,
                f"{text}\n\n(local response: {seconds:.2f}s)",
            )
        )

    def _start(self) -> None:
        if not self.consent_var.get():
            self.status_var.set("CHECK THE CONSENT BOX BEFORE CAPTURE")
            return
        item = self.device_by_label.get(self.device_var.get())
        if item is None:
            self.status_var.set("SELECT AN INPUT ENDPOINT")
            return
        self.engine = CallResponseEngine(
            device_index=item.index,
            model_path=self.model_path,
            tokens_path=self.tokens_path,
            endpoint=self.endpoint,
            llm_model=self.llm_model,
            on_status=self._set_status,
            on_transcript=self._set_transcript,
            on_draft=self._set_draft,
            on_error=lambda value: self._set_status(f"ERROR • {value}"),
        )
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.engine.start()

    def _stop(self) -> None:
        engine = self.engine
        if engine is not None:
            engine.stop()
        self.start_button.configure(
            state="normal" if self.consent_var.get() and self.device_var.get() else "disabled"
        )
        self.stop_button.configure(state="disabled")
        self.status_var.set("READY • no microphone capture")

    def _copy_draft(self) -> None:
        content = self.draft_text.get("1.0", "end").strip()
        if not content:
            self.status_var.set("NO DRAFT TO COPY")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status_var.set("DRAFT COPIED — REVIEW BEFORE SPEAKING")

    def _close(self) -> None:
        self._stop()
        self.root.destroy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--check", action="store_true", help="load local STT and check loopback model")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--stt-model", type=Path, default=DEFAULT_PARakeet_MODEL)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_PARakeet_TOKENS)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_devices:
        print(json.dumps([{"index": i, "name": n} for i, n in list_input_devices()], indent=2))
        return
    if args.check:
        stt = ParakeetCTC(args.stt_model, args.tokens)
        stt.load()
        model = LocalDraftClient(args.endpoint, args.llm_model).health()
        print(json.dumps({"stt": "ready", "response_model": model, "endpoint": validate_loopback_endpoint(args.endpoint)}))
        return
    import tkinter as tk

    root = tk.Tk()
    CallResponseWindow(
        root,
        model_path=args.stt_model,
        tokens_path=args.tokens,
        endpoint=args.endpoint,
        llm_model=args.llm_model,
        requested_device=args.device,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
