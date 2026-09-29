"""Immutable AI preview transactions and a one-time approval gate."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class PreviewDraft:
    transaction_id: str
    original: str
    instruction: str
    result: str
    provider_label: str
    context_lenses: tuple[str, ...]
    target_window: int
    created_at: float
    unchanged: bool
    fingerprint: str
    selection_identity: str = ""
    expires_at: float = 0.0
    retry_system_prompt: str = ""
    context_label: str = "selected_text"
    action_id: str = ""

    @classmethod
    def create(
        cls,
        *,
        original: str,
        instruction: str,
        result: str,
        provider_label: str,
        context_lenses: tuple[str, ...],
        target_window: int,
        selection_identity: str = "",
        retry_system_prompt: str = "",
        context_label: str = "selected_text",
        action_id: str = "",
        approval_ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> "PreviewDraft":
        transaction_id = str(id_factory())
        created_at = float(clock())
        expires_at = created_at + max(30.0, min(float(approval_ttl_seconds), 900.0))
        unchanged = _comparison_text(str(original)) == _comparison_text(str(result))
        material = {
            "version": 4,
            "transaction_id": transaction_id,
            "original": str(original),
            "instruction": str(instruction),
            "result": str(result),
            "provider_label": str(provider_label),
            "context_lenses": list(context_lenses),
            "target_window": int(target_window),
            "selection_identity": str(selection_identity),
            "retry_system_prompt": str(retry_system_prompt),
            "context_label": str(context_label),
            "action_id": str(action_id),
            "created_at": created_at,
            "expires_at": expires_at,
            "unchanged": unchanged,
        }
        if not material["original"] or not material["result"]:
            raise ValueError("preview original and result cannot be empty")
        canonical = json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(canonical).hexdigest()
        return cls(
            transaction_id=transaction_id,
            original=material["original"],
            instruction=material["instruction"],
            result=material["result"],
            provider_label=material["provider_label"],
            context_lenses=tuple(context_lenses),
            target_window=material["target_window"],
            created_at=created_at,
            unchanged=unchanged,
            fingerprint=fingerprint,
            selection_identity=material["selection_identity"],
            expires_at=expires_at,
            retry_system_prompt=material["retry_system_prompt"],
            context_label=material["context_label"],
            action_id=material["action_id"],
        )


def _comparison_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).rstrip("\n")


class PreviewGate:
    """Stages one immutable draft and consumes exact approval at most once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Optional[PreviewDraft] = None

    @property
    def pending(self) -> Optional[PreviewDraft]:
        with self._lock:
            return self._pending

    def stage(self, draft: PreviewDraft) -> None:
        with self._lock:
            self._pending = draft

    def claim(self, transaction_id: str, fingerprint: str) -> Optional[PreviewDraft]:
        with self._lock:
            draft = self._pending
            if draft is None:
                return None
            if (
                draft.transaction_id != str(transaction_id)
                or draft.fingerprint != str(fingerprint)
            ):
                return None
            self._pending = None
            return draft

    def cancel(self, transaction_id: str) -> bool:
        with self._lock:
            if self._pending is None or self._pending.transaction_id != str(transaction_id):
                return False
            self._pending = None
            return True
