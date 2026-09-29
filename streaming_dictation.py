from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

SAMPLE_RATE = 16_000
_TERMINAL_OPERATOR_PATTERN = re.compile(r"(?:&&|--|[|/\\])")
_SEMANTIC_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def reconcile_latest_tail(
    final_text: str,
    latest_text: str,
    *,
    minimum_overlap_words: int = 3,
    maximum_tail_words: int = 32,
) -> tuple[str, bool]:
    """Append only a semantically aligned suffix omitted by the final pass."""
    final = str(final_text or "").strip()
    latest = str(latest_text or "").strip()
    if (
        not final
        or not latest
        or final == latest
        or _TERMINAL_OPERATOR_PATTERN.search(final)
        or _TERMINAL_OPERATOR_PATTERN.search(latest)
    ):
        return final, False
    final_words = [
        (match.group(0).casefold(), match.start(), match.end())
        for match in _SEMANTIC_WORD_PATTERN.finditer(final)
    ]
    latest_words = [
        (match.group(0).casefold(), match.start(), match.end())
        for match in _SEMANTIC_WORD_PATTERN.finditer(latest)
    ]
    minimum = max(2, int(minimum_overlap_words))
    maximum_overlap = min(len(final_words), len(latest_words) - 1)
    for overlap in range(maximum_overlap, minimum - 1, -1):
        final_suffix = [word[0] for word in final_words[-overlap:]]
        last_start = len(latest_words) - overlap - 1
        for start in range(last_start, -1, -1):
            candidate = [
                word[0] for word in latest_words[start : start + overlap]
            ]
            if candidate != final_suffix:
                continue
            tail_words = latest_words[start + overlap :]
            if not tail_words or len(tail_words) > max(1, int(maximum_tail_words)):
                continue
            tail_start = latest_words[start + overlap - 1][2]
            tail = latest[tail_start:]
            if not tail.strip() or len(tail) > 2_000:
                continue
            base = re.sub(r"[,.;:!?…]+$", "", final.rstrip())
            if tail[0].isalnum() and base and not base[-1].isspace():
                tail = " " + tail
            return (base + tail).strip(), True
    return final, False


def tail_is_quiet(
    samples: np.ndarray,
    processed_samples: int,
    max_tail_seconds: float = 0.3,
    rms_threshold: float = 0.003,
    sample_rate: int = SAMPLE_RATE,
) -> bool:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    processed = max(0, min(int(processed_samples), int(audio.size)))
    tail = audio[processed:]
    if tail.size > int(max_tail_seconds * sample_rate):
        return False
    if not tail.size:
        return True
    rms = float(np.sqrt(np.mean(np.square(tail, dtype=np.float32))))
    return rms <= rms_threshold


@dataclass(frozen=True)
class TimedWord:
    word: str
    start: float
    end: float
    probability: float = 0.0


class TimestampCommitter:
    """Expose only old, repeated, timestamp-consistent words for live insertion."""

    def __init__(
        self,
        *,
stability_lag_seconds: float = 0.32,
        holdback_words: int = 1,
        timestamp_tolerance_seconds: float = 0.45,
    ):
        self.stability_lag_seconds = max(0.0, float(stability_lag_seconds))
        self.holdback_words = max(0, int(holdback_words))
        self.timestamp_tolerance_seconds = max(
            0.0, float(timestamp_tolerance_seconds)
        )
        self._previous: tuple[TimedWord, ...] = ()
        self._committed_end = -1.0
        self._committed_text = ""

    @staticmethod
    def _token(word: TimedWord) -> str:
        return re.sub(r"\s+", "", word.word).casefold()

    def _same_word(self, left: TimedWord, right: TimedWord) -> bool:
        return bool(
            self._token(left) == self._token(right)
            and abs(left.start - right.start) <= self.timestamp_tolerance_seconds
            and abs(left.end - right.end) <= self.timestamp_tolerance_seconds
        )

    def reset(self) -> None:
        self._previous = ()
        self._committed_end = -1.0
        self._committed_text = ""

    def update(
        self, words: tuple[TimedWord, ...], *, audio_seconds: float
    ) -> str:
        current = tuple(
            word
            for word in words
            if word.word.strip() and word.end > self._committed_end + 0.02
        )
        stable_count = 0
        for previous, latest in zip(self._previous, current):
            if not self._same_word(previous, latest):
                break
            stable_count += 1

        cutoff = float(audio_seconds) - self.stability_lag_seconds
        time_eligible = 0
        for word in current:
            if word.end > cutoff:
                break
            time_eligible += 1
        tail_eligible = max(0, len(current) - self.holdback_words)
        commit_count = min(stable_count, time_eligible, tail_eligible)
        if commit_count:
            committed_words = current[:commit_count]
            piece = "".join(word.word for word in committed_words)
            if (
                self._committed_text
                and piece
                and not piece[0].isspace()
                and piece[0] not in ",.;:!?%)]}"
            ):
                piece = " " + piece
            self._committed_text = (self._committed_text + piece).strip()
            self._committed_end = committed_words[-1].end
            current = current[commit_count:]

        self._previous = current
        return self._committed_text


@dataclass(frozen=True)
class StreamResult:
    text: str
    processed_samples: int
    inference_seconds: float
    completed_at: float
    reused_partial: bool = False
    stop_to_text_seconds: float = 0.0
    windowed: bool = False
    words: tuple[TimedWord, ...] = ()
    stable_text: str = ""
    reconciled_tail: bool = False
    reconciled_tail_characters: int = 0


class StreamingDictation:
    """Single-flight rolling transcription with guarded cached finalization."""

    def __init__(
        self,
        backend,
        profile,
        *,
on_partial: Optional[Callable[[StreamResult], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        interval_seconds: float = 0.18,
        min_audio_seconds: float = 0.45,
        min_new_audio_seconds: float = 0.14,
        max_quiet_tail_seconds: float = 0.05,
        quiet_rms_threshold: float = 0.003,
        allow_quiet_tail_reuse: bool = False,
        max_partial_audio_seconds: float = 28.0,
        timestamp_stability_lag_seconds: float = 0.32,
        timestamp_holdback_words: int = 1,
        timestamp_tolerance_seconds: float = 0.45,
        final_decoding_mode: str = "fast_full_buffer",
        final_worker_wait_seconds: float = 3.0,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.backend = backend
        self.profile = profile
        self.on_partial = on_partial
        self.on_error = on_error
        self.interval_seconds = interval_seconds
        self.min_audio_samples = int(min_audio_seconds * sample_rate)
        self.min_new_samples = int(min_new_audio_seconds * sample_rate)
        self.max_quiet_tail_seconds = max_quiet_tail_seconds
        self.quiet_rms_threshold = quiet_rms_threshold
        self.allow_quiet_tail_reuse = bool(allow_quiet_tail_reuse)
        self.final_decoding_mode = str(final_decoding_mode)
        self.final_worker_wait_seconds = max(0.1, float(final_worker_wait_seconds))
        self.sample_rate = sample_rate
        self.max_partial_samples = max(
            self.min_audio_samples,
            int(max_partial_audio_seconds * sample_rate),
        )
        self._inference_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._recorder = None
        self._latest: Optional[StreamResult] = None
        self._timestamp_committer = TimestampCommitter(
            stability_lag_seconds=timestamp_stability_lag_seconds,
            holdback_words=timestamp_holdback_words,
            timestamp_tolerance_seconds=timestamp_tolerance_seconds,
        )

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def latest(self) -> Optional[StreamResult]:
        with self._state_lock:
            return self._latest

    def _transcribe(
        self,
        audio: np.ndarray,
        reused_partial: bool = False,
        *,
        fast: bool = True,
        processed_samples: Optional[int] = None,
        windowed: bool = False,
        window_start_samples: int = 0,
    ) -> StreamResult:
        text, metrics = self.backend.transcribe_samples(audio, self.profile, fast=fast)
        offset_seconds = max(0, int(window_start_samples)) / self.sample_rate
        words = tuple(
            TimedWord(
                word=str(item.get("word", "")),
                start=float(item.get("start", 0.0)) + offset_seconds,
                end=float(item.get("end", 0.0)) + offset_seconds,
                probability=float(item.get("probability", 0.0)),
            )
            for item in (metrics.get("words", ()) or ())
            if str(item.get("word", "")).strip()
        )
        total_samples = int(
            audio.size if processed_samples is None else processed_samples
        )
        stable_text = self._timestamp_committer.update(
            words,
            audio_seconds=total_samples / self.sample_rate,
        )
        return StreamResult(
            text=text,
            processed_samples=total_samples,
            inference_seconds=float(metrics.get("transcribe_seconds", 0.0)),
            completed_at=time.monotonic(),
            reused_partial=reused_partial,
            windowed=windowed,
            words=words,
            stable_text=stable_text,
        )

    def process_snapshot(self, samples: np.ndarray) -> Optional[StreamResult]:
        audio = np.ascontiguousarray(
            np.asarray(samples, dtype=np.float32).reshape(-1)
        )
        if audio.size < self.min_audio_samples:
            return None
        with self._inference_lock:
            latest = self.latest
            if latest and audio.size - latest.processed_samples < self.min_new_samples:
                return latest
            processed_samples = int(audio.size)
            windowed = processed_samples > self.max_partial_samples
            inference_audio = (
                audio[-self.max_partial_samples :] if windowed else audio
            )
            result = self._transcribe(
                inference_audio,
                processed_samples=processed_samples,
                windowed=windowed,
                window_start_samples=processed_samples - int(inference_audio.size),
            )
            with self._state_lock:
                self._latest = result
            # Bubble preview should refresh on every completed pass, including
            # empty text (clears stale provisional phrases) and repeated
            # snapshots that still carry the latest transcript.
            if self.on_partial:
                self.on_partial(result)
            return result

    def start(self, recorder) -> None:
        if not self.cancel():
            raise RuntimeError("Previous streaming worker is still stopping")
        self._recorder = recorder
        with self._state_lock:
            self._latest = None
        self._timestamp_committer.reset()
        self._cancel_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=False,
            name="instant-streaming-stt",
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._cancel_event.wait(self.interval_seconds):
            recorder = self._recorder
            if recorder is None:
                return
            try:
                self.process_snapshot(recorder.snapshot())
            except Exception as exc:
                if self.on_error:
                    self.on_error(exc)
                # Finalization still gets one explicit pass and surfaces its error.
                return

    def cancel(self, timeout: float = 3.0) -> bool:
        self._cancel_event.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._recorder = None
        if thread and thread.is_alive():
            return False
        self._thread = None
        return True

    def finalize(self, samples: np.ndarray) -> StreamResult:
        stop_started = time.monotonic()
        self.cancel(timeout=self.final_worker_wait_seconds)
        audio = np.ascontiguousarray(
            np.asarray(samples, dtype=np.float32).reshape(-1)
        )
        latest = self.latest
        if self.allow_quiet_tail_reuse and latest and not latest.windowed and latest.text and tail_is_quiet(
            audio,
            latest.processed_samples,
            max_tail_seconds=self.max_quiet_tail_seconds,
            rms_threshold=self.quiet_rms_threshold,
            sample_rate=self.sample_rate,
        ):
            return StreamResult(
                text=latest.text,
                processed_samples=int(audio.size),
                inference_seconds=latest.inference_seconds,
                completed_at=time.monotonic(),
                reused_partial=True,
                stop_to_text_seconds=time.monotonic() - stop_started,
            )
        with self._inference_lock:
            result = self._transcribe(
                audio,
                fast=self.final_decoding_mode != "accurate_full_buffer",
            )
        final_text = result.text
        reconciled_tail = False
        near_stop_limit = max(
            int(2.0 * self.sample_rate),
            self.min_new_samples * 2,
        )
        if (
            latest is not None
            and latest.text
            and 0 <= int(audio.size) - latest.processed_samples <= near_stop_limit
        ):
            final_text, reconciled_tail = reconcile_latest_tail(
                final_text,
                latest.text,
            )
        return StreamResult(
            text=final_text,
            processed_samples=result.processed_samples,
            inference_seconds=result.inference_seconds,
            completed_at=result.completed_at,
            reused_partial=False,
            stop_to_text_seconds=time.monotonic() - stop_started,
            reconciled_tail=reconciled_tail,
            reconciled_tail_characters=max(0, len(final_text) - len(result.text)),
        )
