"""Deterministic Whisper ghost / hallucination cleanup (WinWhisp-style patterns).

Independent reimplementation: exact-phrase and regex blocklist only. No GPL
source was copied. Safe to run on every final transcript before insertion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


# Common empty-room / YouTube outro ghosts from English Whisper models.
DEFAULT_HALLUCINATION_BLOCKLIST: tuple[str, ...] = (
    "thank you for watching",
    "thanks for watching",
    "thank you for listening",
    "thanks for listening",
    "please subscribe",
    "subscribe to the channel",
    "like and subscribe",
    "don't forget to subscribe",
    "see you next time",
    "see you in the next video",
    "thanks for tuning in",
    "music playing",
    "applause",
    "laughter",
    "foreign",
    "subtitles by the amara.org community",
    "subtitles by",
    "captioning made possible by",
    "translated by",
    "www.youtube.com",
    "you",
)


_PUNCT_ONLY = re.compile(r"^[\s\W_]+$", re.UNICODE)
_REPEATED_PUNCT = re.compile(r"^([.!?…,\-–—])\1{2,}$")
_MULTI_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class HallucinationFilterResult:
    text: str
    removed: bool
    reason: str = ""


def _normalize_phrase(text: str) -> str:
    value = _MULTI_SPACE.sub(" ", str(text or "").strip().casefold())
    value = value.strip(" \t\r\n\"'`“”‘’")
    value = value.rstrip(".,!?;:…")
    return value.strip()


def compile_blocklist(phrases: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for phrase in phrases:
        normalized = _normalize_phrase(phrase)
        if normalized and normalized not in seen:
            seen.append(normalized)
    # Longest first so multi-word ghosts win over short tokens like "you".
    return tuple(sorted(seen, key=len, reverse=True))


def is_punctuation_only_ghost(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if _REPEATED_PUNCT.match(stripped):
        return True
    if _PUNCT_ONLY.match(stripped) and not any(ch.isalnum() for ch in stripped):
        return True
    return False


def filter_hallucinations(
    text: str,
    *,
    enabled: bool = True,
    blocklist: Optional[Sequence[str]] = None,
) -> HallucinationFilterResult:
    """Drop known Whisper ghosts; leave ordinary speech untouched."""
    raw = str(text or "")
    if not enabled:
        return HallucinationFilterResult(text=raw, removed=False, reason="disabled")
    if not raw.strip():
        return HallucinationFilterResult(text=raw, removed=False, reason="empty")

    if is_punctuation_only_ghost(raw):
        return HallucinationFilterResult(text="", removed=True, reason="punctuation_only")

    phrases = compile_blocklist(
        blocklist if blocklist is not None else DEFAULT_HALLUCINATION_BLOCKLIST
    )
    normalized = _normalize_phrase(raw)
    if not normalized:
        return HallucinationFilterResult(text="", removed=True, reason="empty_normalized")

    # Whole-utterance match (common for silent / music clips).
    if normalized in phrases:
        return HallucinationFilterResult(text="", removed=True, reason="blocklist_exact")

    # Utterance that is only a blocklist phrase plus trivial filler.
    for phrase in phrases:
        if normalized == phrase:
            return HallucinationFilterResult(text="", removed=True, reason="blocklist_exact")
        # Short clips that are solely the ghost plus optional "uh"/"um".
        pattern = rf"^(?:uh+|um+|ah+|oh+\s+)*{re.escape(phrase)}(?:\s+(?:uh+|um+|ah+|oh+))*$"
        if re.fullmatch(pattern, normalized):
            return HallucinationFilterResult(
                text="",
                removed=True,
                reason="blocklist_filled",
            )

    # Strip a trailing outro ghost when it is the entire trailing clause.
    for phrase in phrases:
        if len(phrase) < 8:
            continue
        trailing = re.compile(
            rf"(?:^|[.!?]\s+){re.escape(phrase)}\s*[.!?…]*$",
            re.IGNORECASE,
        )
        if trailing.search(raw.strip()):
            cleaned = trailing.sub("", raw.strip()).strip()
            if cleaned and cleaned != raw.strip():
                return HallucinationFilterResult(
                    text=cleaned,
                    removed=True,
                    reason="blocklist_trailing",
                )
            if not cleaned:
                return HallucinationFilterResult(
                    text="",
                    removed=True,
                    reason="blocklist_trailing",
                )

    return HallucinationFilterResult(text=raw, removed=False, reason="kept")
