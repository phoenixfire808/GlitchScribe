from __future__ import annotations

import ctypes
import time
from contextlib import contextmanager
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol

# Win32 clipboard / paste constants (reimplemented; no GPL source).
CF_UNICODETEXT = 13
CF_TEXT = 1
CF_HDROP = 15
CF_BITMAP = 2
CF_DIB = 8
CF_DIBV5 = 17
GMEM_MOVEABLE = 0x0002
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_V = 0x56

# Prefer restoring these formats after a temporary paste (best-effort).
_RESTORE_FORMAT_PRIORITY: tuple[int, ...] = (
    CF_UNICODETEXT,
    CF_TEXT,
    CF_HDROP,
    CF_DIBV5,
    CF_DIB,
    CF_BITMAP,
)


@dataclass(frozen=True)
class InsertionTarget:
    window_handle: int
    control_identity: str = ""
    executable: str = ""
    window_title: str = ""
    control_type: int | None = None
    editable: bool = False
    password: bool = False
    ime_active: bool = False


@dataclass(frozen=True)
class InsertionResult:
    requested_characters: int
    inserted_characters: int
    reason: str = "ok"
    method: str = "unicode"

    @property
    def complete(self) -> bool:
        return self.requested_characters == self.inserted_characters and self.reason in {
            "ok",
            "clipboard_paste",
            "clipboard_threshold",
        }


@dataclass(frozen=True)
class ClipboardFormatSnapshot:
    format_id: int
    data: bytes


@dataclass(frozen=True)
class ClipboardSnapshot:
    formats: tuple[ClipboardFormatSnapshot, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.formats


@dataclass(frozen=True)
class InsertionPolicy:
    """Runtime knobs for SendInput vs clipboard paste fallback."""

    clipboard_fallback: bool = True
    clipboard_threshold_chars: int = 400
    clipboard_open_retries: int = 8
    clipboard_open_retry_ms: int = 15
    paste_settle_ms: int = 40
    inject_delay_ms: int = 0


class TextInserter(Protocol):
    def inspect_target(self) -> InsertionTarget: ...

    def insert(self, text: str, target: InsertionTarget) -> InsertionResult: ...


def normalize_inject_delay_ms(value: object, default: int = 0) -> int:
    """Clamp optional pre-insert delay (Murmur-style inject_delay_ms)."""
    try:
        delay = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return max(0, int(default))
    return max(0, min(delay, 2000))


def default_insertion_policy(
    *,
    clipboard_fallback: bool = True,
    clipboard_threshold_chars: int = 400,
    inject_delay_ms: int = 0,
    paste_settle_ms: int = 40,
) -> InsertionPolicy:
    threshold = max(1, int(clipboard_threshold_chars))
    return InsertionPolicy(
        clipboard_fallback=bool(clipboard_fallback),
        clipboard_threshold_chars=threshold,
        inject_delay_ms=normalize_inject_delay_ms(inject_delay_ms),
        paste_settle_ms=max(0, int(paste_settle_ms)),
    )


def should_prefer_clipboard(text: str, policy: InsertionPolicy) -> bool:
    if not policy.clipboard_fallback:
        return False
    return len(text) >= int(policy.clipboard_threshold_chars)


class Win32ClipboardApi:
    """Thin adapter around user32/kernel32 clipboard APIs (injectable for tests)."""

    def __init__(self) -> None:
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure()

    def _configure(self) -> None:
        u = self.user32
        k = self.kernel32
        u.OpenClipboard.argtypes = (wintypes.HWND,)
        u.OpenClipboard.restype = wintypes.BOOL
        u.CloseClipboard.argtypes = ()
        u.CloseClipboard.restype = wintypes.BOOL
        u.EmptyClipboard.argtypes = ()
        u.EmptyClipboard.restype = wintypes.BOOL
        u.IsClipboardFormatAvailable.argtypes = (wintypes.UINT,)
        u.IsClipboardFormatAvailable.restype = wintypes.BOOL
        u.GetClipboardData.argtypes = (wintypes.UINT,)
        u.GetClipboardData.restype = wintypes.HANDLE
        u.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
        u.SetClipboardData.restype = wintypes.HANDLE
        u.EnumClipboardFormats.argtypes = (wintypes.UINT,)
        u.EnumClipboardFormats.restype = wintypes.UINT
        u.GetClipboardSequenceNumber.argtypes = ()
        u.GetClipboardSequenceNumber.restype = wintypes.DWORD
        k.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
        k.GlobalAlloc.restype = wintypes.HGLOBAL
        k.GlobalLock.argtypes = (wintypes.HGLOBAL,)
        k.GlobalLock.restype = ctypes.c_void_p
        k.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
        k.GlobalUnlock.restype = wintypes.BOOL
        k.GlobalSize.argtypes = (wintypes.HGLOBAL,)
        k.GlobalSize.restype = ctypes.c_size_t
        k.GlobalFree.argtypes = (wintypes.HGLOBAL,)
        k.GlobalFree.restype = wintypes.HGLOBAL

    def open(self, owner: int = 0) -> bool:
        return bool(self.user32.OpenClipboard(int(owner or 0)))

    def close(self) -> None:
        self.user32.CloseClipboard()

    def empty(self) -> bool:
        return bool(self.user32.EmptyClipboard())

    def sequence_number(self) -> int:
        return int(self.user32.GetClipboardSequenceNumber())

    def enum_formats(self) -> list[int]:
        formats: list[int] = []
        current = 0
        while True:
            current = int(self.user32.EnumClipboardFormats(current))
            if current == 0:
                break
            formats.append(current)
        return formats

    def get_format_bytes(self, format_id: int) -> bytes | None:
        handle = self.user32.GetClipboardData(int(format_id))
        if not handle:
            return None
        size = int(self.kernel32.GlobalSize(handle))
        if size <= 0:
            return b""
        locked = self.kernel32.GlobalLock(handle)
        if not locked:
            return None
        try:
            return ctypes.string_at(locked, size)
        finally:
            self.kernel32.GlobalUnlock(handle)

    def set_format_bytes(self, format_id: int, data: bytes) -> bool:
        size = len(data)
        handle = self.kernel32.GlobalAlloc(GMEM_MOVEABLE, size if size else 1)
        if not handle:
            return False
        locked = self.kernel32.GlobalLock(handle)
        if not locked:
            self.kernel32.GlobalFree(handle)
            return False
        try:
            if size:
                ctypes.memmove(locked, data, size)
            else:
                ctypes.memset(locked, 0, 1)
        finally:
            self.kernel32.GlobalUnlock(handle)
        result = self.user32.SetClipboardData(int(format_id), handle)
        if not result:
            self.kernel32.GlobalFree(handle)
            return False
        return True

    def set_unicode_text(self, text: str) -> bool:
        payload = text.encode("utf-16-le") + b"\x00\x00"
        return self.set_format_bytes(CF_UNICODETEXT, payload)


def snapshot_clipboard(
    api: Win32ClipboardApi | None = None,
    *,
    open_retries: int = 8,
    retry_ms: int = 15,
) -> ClipboardSnapshot:
    clip = api or Win32ClipboardApi()
    if not _open_clipboard_with_retry(clip, open_retries=open_retries, retry_ms=retry_ms):
        return ClipboardSnapshot()
    try:
        formats = clip.enum_formats()
        ordered = _order_formats_for_restore(formats)
        snapshots: list[ClipboardFormatSnapshot] = []
        for format_id in ordered:
            data = clip.get_format_bytes(format_id)
            if data is None:
                continue
            snapshots.append(ClipboardFormatSnapshot(format_id=format_id, data=data))
        return ClipboardSnapshot(formats=tuple(snapshots))
    finally:
        clip.close()


def restore_clipboard(
    snapshot: ClipboardSnapshot,
    api: Win32ClipboardApi | None = None,
    *,
    open_retries: int = 8,
    retry_ms: int = 15,
) -> bool:
    clip = api or Win32ClipboardApi()
    if not _open_clipboard_with_retry(clip, open_retries=open_retries, retry_ms=retry_ms):
        return False
    try:
        if not clip.empty():
            return False
        if snapshot.empty:
            return True
        ok = True
        for item in snapshot.formats:
            if not clip.set_format_bytes(item.format_id, item.data):
                ok = False
        return ok
    finally:
        clip.close()


def set_clipboard_text(
    text: str,
    api: Win32ClipboardApi | None = None,
    *,
    open_retries: int = 8,
    retry_ms: int = 15,
) -> bool:
    clip = api or Win32ClipboardApi()
    if not _open_clipboard_with_retry(clip, open_retries=open_retries, retry_ms=retry_ms):
        return False
    try:
        if not clip.empty():
            return False
        return bool(clip.set_unicode_text(text))
    finally:
        clip.close()


def send_ctrl_v(
    *,
    keybd_event: Callable[[int, int, int, int], None] | None = None,
) -> None:
    """Inject Ctrl+V via keybd_event (same path as existing send_keypress)."""
    if keybd_event is None:
        user32 = ctypes.windll.user32
        keybd_event = user32.keybd_event
    keybd_event(VK_CONTROL, 0, 0, 0)
    try:
        keybd_event(VK_V, 0, 0, 0)
        keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
    finally:
        keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def _sleep_inject_delay(
    policy: InsertionPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    delay_ms = normalize_inject_delay_ms(getattr(policy, "inject_delay_ms", 0))
    if delay_ms > 0:
        sleep(delay_ms / 1000.0)


def paste_text_via_clipboard(
    text: str,
    *,
    api: Win32ClipboardApi | None = None,
    policy: InsertionPolicy | None = None,
    send_paste: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> InsertionResult:
    """Paste text with full prior-clipboard restore (text + best-effort formats)."""
    requested = len(text)
    if not text:
        return InsertionResult(0, 0, reason="empty", method="clipboard")
    policy = policy or default_insertion_policy()
    clip = api or Win32ClipboardApi()
    prior = snapshot_clipboard(
        clip,
        open_retries=policy.clipboard_open_retries,
        retry_ms=policy.clipboard_open_retry_ms,
    )
    if not set_clipboard_text(
        text,
        clip,
        open_retries=policy.clipboard_open_retries,
        retry_ms=policy.clipboard_open_retry_ms,
    ):
        restore_clipboard(
            prior,
            clip,
            open_retries=policy.clipboard_open_retries,
            retry_ms=policy.clipboard_open_retry_ms,
        )
        return InsertionResult(
            requested,
            0,
            reason="clipboard_open_failed",
            method="clipboard",
        )
    try:
        _sleep_inject_delay(policy, sleep=sleep)
        (send_paste or send_ctrl_v)()
        settle = max(0, int(policy.paste_settle_ms)) / 1000.0
        if settle:
            sleep(settle)
    finally:
        restore_clipboard(
            prior,
            clip,
            open_retries=policy.clipboard_open_retries,
            retry_ms=policy.clipboard_open_retry_ms,
        )
    return InsertionResult(
        requested,
        requested,
        reason="clipboard_paste",
        method="clipboard",
    )


def insert_text_with_fallback(
    text: str,
    *,
    type_unicode: Callable[[str], None],
    policy: InsertionPolicy | None = None,
    paste_via_clipboard: Callable[[str], InsertionResult] | None = None,
    prefer_clipboard: bool | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> InsertionResult:
    """Prefer Unicode SendInput; clipboard paste on length threshold or failure."""
    requested = len(text)
    if not text:
        return InsertionResult(0, 0, reason="empty", method="none")
    policy = policy or default_insertion_policy()
    use_clipboard_first = (
        bool(prefer_clipboard)
        if prefer_clipboard is not None
        else should_prefer_clipboard(text, policy)
    )
    paste = paste_via_clipboard or (
        lambda value: paste_text_via_clipboard(value, policy=policy, sleep=sleep)
    )

    if use_clipboard_first and policy.clipboard_fallback:
        result = paste(text)
        if result.complete:
            return InsertionResult(
                requested,
                result.inserted_characters,
                reason="clipboard_threshold",
                method="clipboard",
            )

    try:
        _sleep_inject_delay(policy, sleep=sleep)
        type_unicode(text)
        return InsertionResult(
            requested,
            requested,
            reason="ok",
            method="unicode",
        )
    except (OSError, PermissionError) as exc:
        if not policy.clipboard_fallback:
            return InsertionResult(
                requested,
                0,
                reason=type(exc).__name__,
                method="unicode",
            )
        result = paste(text)
        if result.complete:
            return InsertionResult(
                requested,
                result.inserted_characters,
                reason="clipboard_paste",
                method="clipboard",
            )
        return InsertionResult(
            requested,
            0,
            reason=result.reason or type(exc).__name__,
            method="clipboard",
        )


@dataclass
class LastTranscriptStore:
    """Remembers the last successful transcript for paste-last hotkey."""

    text: str = ""
    target_window: int = 0
    inserted_at: float = 0.0
    source: str = ""

    def remember(
        self,
        text: str,
        *,
        target_window: int = 0,
        source: str = "insert",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        cleaned = str(text or "")
        if not cleaned.strip():
            return
        self.text = cleaned
        self.target_window = int(target_window or 0)
        self.inserted_at = float(clock())
        self.source = str(source or "insert")

    def clear(self) -> None:
        self.text = ""
        self.target_window = 0
        self.inserted_at = 0.0
        self.source = ""

    @property
    def available(self) -> bool:
        return bool(self.text.strip())


def parse_secondary_hotkey(label: str) -> tuple[set[str], str] | None:
    """Parse labels like 'Ctrl + Shift + V' into ({CTRL, SHIFT}, V)."""
    raw = str(label or "").strip()
    if not raw or raw.lower() in {"none", "off", "disabled", "-"}:
        return None
    parts = [part.strip().upper() for part in raw.replace("-", "+").split("+") if part.strip()]
    if not parts:
        return None
    keys = []
    mods: set[str] = set()
    for part in parts:
        if part in {"CONTROL", "CTRL", "CTL"}:
            mods.add("CTRL")
        elif part == "SHIFT":
            mods.add("SHIFT")
        elif part in {"ALT", "MENU"}:
            mods.add("ALT")
        elif part in {"WIN", "WINDOWS", "CMD", "SUPER"}:
            mods.add("WIN")
        else:
            keys.append(part)
    if len(keys) != 1:
        return None
    key = keys[0]
    if key.startswith("F") and key[1:].isdigit():
        return mods, key
    if len(key) == 1 and key.isalnum():
        return mods, key
    aliases = {
        "V": "V",
        "INSERT": "INSERT",
        "INS": "INSERT",
        "SPACE": "SPACE",
    }
    if key in aliases:
        return mods, aliases[key]
    return None


@dataclass
class SecondaryHotkeyTracker:
    """Edge-triggered secondary chord detector for the low-level keyboard hook."""

    chord_label: str
    _down_keys: set[str] = field(default_factory=set)
    _fired: bool = False
    _parsed: tuple[set[str], str] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._parsed = parse_secondary_hotkey(self.chord_label)

    @property
    def enabled(self) -> bool:
        return self._parsed is not None

    def reconfigure(self, chord_label: str) -> None:
        self.chord_label = chord_label
        self._parsed = parse_secondary_hotkey(chord_label)
        self._down_keys.clear()
        self._fired = False

    def handle(self, key_name: str, *, is_down: bool) -> bool:
        """Return True on the rising edge when the full chord is satisfied."""
        if self._parsed is None or not key_name:
            return False
        name = key_name.upper()
        if name == "CONTROL":
            name = "CTRL"
        if name == "WINDOWS":
            name = "WIN"
        mods, primary = self._parsed
        if is_down:
            self._down_keys.add(name)
            if self._fired:
                return False
            if primary not in self._down_keys:
                return False
            if not mods.issubset(self._down_keys):
                return False
            held_mods = self._down_keys & {"CTRL", "SHIFT", "ALT", "WIN"}
            if held_mods != mods:
                return False
            self._fired = True
            return True
        self._down_keys.discard(name)
        if name == primary or name in mods:
            self._fired = False
        return False


def _order_formats_for_restore(formats: list[int]) -> list[int]:
    priority = {fmt: index for index, fmt in enumerate(_RESTORE_FORMAT_PRIORITY)}
    known = [fmt for fmt in formats if fmt in priority]
    known.sort(key=lambda fmt: priority[fmt])
    extras = [
        fmt
        for fmt in formats
        if fmt not in priority and fmt < 0xC000
    ]
    return known + extras[:6]


def _open_clipboard_with_retry(
    api: Win32ClipboardApi,
    *,
    open_retries: int,
    retry_ms: int,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    attempts = max(1, int(open_retries))
    delay = max(0, int(retry_ms)) / 1000.0
    for attempt in range(attempts):
        if api.open(0):
            return True
        if attempt + 1 < attempts and delay:
            sleep(delay)
    return False


@contextmanager
def temporary_clipboard_text(
    text: str,
    *,
    api: Win32ClipboardApi | None = None,
    policy: InsertionPolicy | None = None,
) -> Iterator[bool]:
    """Context manager that sets clipboard text and always restores prior contents."""
    policy = policy or default_insertion_policy()
    clip = api or Win32ClipboardApi()
    prior = snapshot_clipboard(
        clip,
        open_retries=policy.clipboard_open_retries,
        retry_ms=policy.clipboard_open_retry_ms,
    )
    ok = set_clipboard_text(
        text,
        clip,
        open_retries=policy.clipboard_open_retries,
        retry_ms=policy.clipboard_open_retry_ms,
    )
    try:
        yield ok
    finally:
        restore_clipboard(
            prior,
            clip,
            open_retries=policy.clipboard_open_retries,
            retry_ms=policy.clipboard_open_retry_ms,
        )
