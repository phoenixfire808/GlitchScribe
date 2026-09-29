from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class AudioSnapshot:
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float


class AudioSource(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def latest_rms(self) -> float: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def snapshot(self) -> np.ndarray: ...


def samples_for_duration_ms(duration_ms: float, sample_rate: int) -> int:
    """Return non-negative sample count for a duration at the given rate."""
    rate = max(1, int(sample_rate))
    ms = max(0.0, float(duration_ms))
    return int(round(rate * ms / 1000.0))


class PreRollRingBuffer:
    """Fixed-capacity mono float32 ring of recent PCM (pre-roll before record).

    Independent reimplementation inspired by whisper-local's pre-roll idea:
    capacity is sample-count based; writers append chunks and readers take a
    contiguous copy of the most recent window.
    """

    def __init__(
        self,
        *,
        capacity_samples: int = 0,
        sample_rate: int = 16_000,
        preroll_ms: float = 400.0,
    ) -> None:
        rate = max(1, int(sample_rate))
        if capacity_samples <= 0:
            capacity_samples = max(1, samples_for_duration_ms(preroll_ms, rate))
        self.sample_rate = rate
        self.capacity_samples = max(1, int(capacity_samples))
        self._chunks: Deque[np.ndarray] = deque()
        self._stored = 0

    @classmethod
    def from_preroll_ms(
        cls,
        preroll_ms: float,
        sample_rate: int = 16_000,
    ) -> "PreRollRingBuffer":
        return cls(
            capacity_samples=samples_for_duration_ms(preroll_ms, sample_rate),
            sample_rate=sample_rate,
            preroll_ms=preroll_ms,
        )

    @property
    def stored_samples(self) -> int:
        return int(self._stored)

    @property
    def duration_ms(self) -> float:
        return 1000.0 * self._stored / float(self.sample_rate)

    def clear(self) -> None:
        self._chunks.clear()
        self._stored = 0

    def write(self, samples: np.ndarray | Iterable[float]) -> None:
        chunk = np.ascontiguousarray(
            np.asarray(samples, dtype=np.float32).reshape(-1)
        )
        if not chunk.size:
            return
        self._chunks.append(chunk)
        self._stored += int(chunk.size)
        self._trim()

    def _trim(self) -> None:
        while self._stored > self.capacity_samples and self._chunks:
            head = self._chunks[0]
            overflow = self._stored - self.capacity_samples
            if head.size <= overflow:
                self._chunks.popleft()
                self._stored -= int(head.size)
            else:
                self._chunks[0] = head[overflow:]
                self._stored -= int(overflow)
                break

    def snapshot(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self._chunks) == 1:
            return np.ascontiguousarray(self._chunks[0], dtype=np.float32)
        return np.ascontiguousarray(
            np.concatenate(list(self._chunks)), dtype=np.float32
        )

    def take(self) -> np.ndarray:
        """Return and clear the buffered window."""
        audio = self.snapshot()
        self.clear()
        return audio


def prepend_preroll(
    session_audio: np.ndarray,
    preroll: np.ndarray,
    *,
    max_preroll_samples: Optional[int] = None,
) -> np.ndarray:
    """Prepend pre-roll PCM ahead of the official session capture."""
    body = np.ascontiguousarray(
        np.asarray(session_audio, dtype=np.float32).reshape(-1)
    )
    head = np.ascontiguousarray(np.asarray(preroll, dtype=np.float32).reshape(-1))
    if max_preroll_samples is not None and max_preroll_samples >= 0:
        head = head[-int(max_preroll_samples) :]
    if not head.size:
        return body
    if not body.size:
        return head
    return np.concatenate([head, body])


class AmbientPreRollCapture:
    """Idle microphone ring used only to seed the next dictation session.

    Opens a low-duty InputStream while the app is READY and not recording.
    Call ``take_for_session()`` when official recording starts, then ``stop()``
    so the session AudioRecorder owns the device exclusively.
    """

    def __init__(
        self,
        *,
        device_index: int,
        preroll_ms: float = 400.0,
        sample_rate: int = 16_000,
        capture_rate: Optional[int] = None,
        callback_block_ms: float = 50.0,
        input_stream_factory=None,
        query_devices=None,
    ) -> None:
        self.device_index = int(device_index)
        self.preroll_ms = max(0.0, float(preroll_ms))
        self.sample_rate = max(1, int(sample_rate))
        self.capture_rate = int(capture_rate or self.sample_rate)
        self.callback_block_ms = max(10.0, float(callback_block_ms))
        self._input_stream_factory = input_stream_factory
        self._query_devices = query_devices
        self._stream = None
        self._ring = PreRollRingBuffer.from_preroll_ms(
            self.preroll_ms,
            sample_rate=self.sample_rate,
        )
        self._running = False

    @property
    def is_running(self) -> bool:
        return bool(self._running and self._stream is not None)

    def configure(
        self,
        *,
        device_index: Optional[int] = None,
        preroll_ms: Optional[float] = None,
    ) -> None:
        if device_index is not None:
            self.device_index = int(device_index)
        if preroll_ms is not None:
            self.preroll_ms = max(0.0, float(preroll_ms))
            self._ring = PreRollRingBuffer.from_preroll_ms(
                self.preroll_ms,
                sample_rate=self.sample_rate,
            )

    def start(self) -> None:
        if self.preroll_ms <= 0:
            return
        if self.is_running:
            return
        factory = self._input_stream_factory
        query = self._query_devices
        if factory is None or query is None:
            import sounddevice as sd

            factory = factory or sd.InputStream
            query = query or sd.query_devices
        try:
            device = query(self.device_index)
            self.capture_rate = int(float(device["default_samplerate"]))
        except Exception:
            self.capture_rate = self.sample_rate
        # Store at capture rate; convert on take for session.
        self._ring = PreRollRingBuffer(
            capacity_samples=samples_for_duration_ms(
                self.preroll_ms, self.capture_rate
            ),
            sample_rate=self.capture_rate,
        )

        def _callback(indata, frames, time_info, status) -> None:
            del frames, time_info, status
            chunk = np.asarray(indata[:, 0], dtype=np.float32).copy()
            if chunk.size:
                self._ring.write(chunk)

        self._stream = factory(
            device=self.device_index,
            channels=1,
            samplerate=self.capture_rate,
            dtype="float32",
            callback=_callback,
            blocksize=max(
                1, int(self.capture_rate * self.callback_block_ms / 1000.0)
            ),
        )
        self._stream.start()
        self._running = True

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        self._running = False
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _resampled(self, audio: np.ndarray, target_rate: int) -> np.ndarray:
        if not audio.size:
            return np.zeros(0, dtype=np.float32)
        if int(self.capture_rate) == int(target_rate):
            return np.ascontiguousarray(audio, dtype=np.float32)
        # Lightweight linear resample (same idea as local_voice.resample_audio).
        source = np.ascontiguousarray(audio, dtype=np.float32)
        duration = source.size / float(self.capture_rate)
        target_count = max(1, int(round(duration * target_rate)))
        if source.size == 1:
            return np.full(target_count, source[0], dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, num=source.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_count, endpoint=False)
        return np.interp(x_new, x_old, source).astype(np.float32)

    def peek_for_session(self, target_rate: int = 16_000) -> np.ndarray:
        """Return pre-roll resampled to target_rate without stopping capture.

        Closing this stream and immediately opening a fresh session stream on
        the same physical device forces WASAPI to renegotiate the shared
        endpoint, which can stall for several seconds (worse on virtual
        devices such as Voicemeeter). Callers should keep the ambient stream
        alive until the new session stream is confirmed running, then call
        ``stop()`` once the handoff is safe.
        """
        return self._resampled(self._ring.snapshot(), target_rate)

    def take_for_session(self, target_rate: int = 16_000) -> np.ndarray:
        """Stop ambient capture and return pre-roll resampled to target_rate."""
        audio = self._ring.take()
        self.stop()
        return self._resampled(audio, target_rate)
