from __future__ import annotations

import ctypes
import hashlib
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from .insertion import InsertionTarget


@dataclass(frozen=True)
class CaretContext:
    supported: bool = False
    preceding_text: str = ""
    following_text: str = ""
    selected_text: str = ""
    source: str = "unavailable"

    @property
    def has_selection(self) -> bool:
        return bool(self.selected_text)


class ContextProvider(Protocol):
    def read(self, target: InsertionTarget, *, max_characters: int = 256) -> CaretContext: ...


@dataclass(frozen=True)
class SelectionProbe:
    accessible: bool
    is_password: bool
    text: str = ""
    source: str = "unavailable"
    control_identity: str = ""


@dataclass(frozen=True)
class ContextLensCapture:
    lens_id: str
    included: bool
    content: str
    source: str
    target_window: int
    captured_at: float
    sensitivity: str = "private"
    original_characters: int = 0
    truncated: bool = False
    blocked_reason: str = ""
    control_identity: str = ""


def read_uia_focused_control_identity() -> SelectionProbe:
    """Read only non-content identity metadata for the focused UIA control."""
    if os.name != "nt":
        return SelectionProbe(False, False, source="platform_unsupported")
    try:
        import comtypes
        from comtypes.client import CreateObject, GetModule

        comtypes.CoInitialize()
        GetModule("UIAutomationCore.dll")
        from comtypes.gen import UIAutomationClient

        automation = CreateObject(
            UIAutomationClient.CUIAutomation,
            interface=UIAutomationClient.IUIAutomation,
        )
        element = automation.GetFocusedElement()
        if element is None:
            return SelectionProbe(False, False, source="uia_unavailable")
        return SelectionProbe(
            accessible=True,
            is_password=bool(element.CurrentIsPassword),
            source="uia_focused_element",
            control_identity=_uia_control_identity(element),
        )
    except Exception:
        return SelectionProbe(False, False, source="uia_unavailable")


def _blocked_capture(
    target_window: int,
    captured_at: float,
    reason: str,
    *,
    source: str = "unavailable",
) -> ContextLensCapture:
    return ContextLensCapture(
        lens_id="selected_text",
        included=False,
        content="",
        source=source,
        target_window=int(target_window or 0),
        captured_at=captured_at,
        blocked_reason=reason,
    )


def read_uia_selected_text(max_characters: int) -> SelectionProbe:
    """Read the focused UIA selection without copying or modifying clipboard state."""
    if os.name != "nt":
        return SelectionProbe(False, False, source="platform_unsupported")
    try:
        import comtypes
        from comtypes.client import CreateObject, GetModule

        comtypes.CoInitialize()
        GetModule("UIAutomationCore.dll")
        from comtypes.gen import UIAutomationClient

        automation = CreateObject(
            UIAutomationClient.CUIAutomation,
            interface=UIAutomationClient.IUIAutomation,
        )
        element = automation.GetFocusedElement()
        if element is None:
            return SelectionProbe(False, False, source="uia_unavailable")
        control_identity = _uia_control_identity(element)
        if bool(element.CurrentIsPassword):
            return SelectionProbe(
                True,
                True,
                source="uia_text_pattern",
                control_identity=control_identity,
            )
        pattern = element.GetCurrentPattern(UIAutomationClient.UIA_TextPatternId)
        text_pattern = pattern.QueryInterface(
            UIAutomationClient.IUIAutomationTextPattern
        )
        selection = text_pattern.GetSelection()
        if selection is None or int(selection.Length) <= 0:
            return SelectionProbe(
                True,
                False,
                source="uia_text_pattern",
                control_identity=control_identity,
            )
        text_range = selection.GetElement(0)
        text = str(text_range.GetText(max(1, int(max_characters))) or "")
        return SelectionProbe(
            True,
            False,
            text=text,
            source="uia_text_pattern",
            control_identity=control_identity,
        )
    except Exception:
        return SelectionProbe(False, False, source="uia_unavailable")


def _uia_control_identity(element) -> str:
    """Hash stable, non-content UIA metadata for later selection revalidation."""
    try:
        runtime_id = tuple(int(value) for value in (element.GetRuntimeId() or ()))
    except Exception:
        runtime_id = ()
    fields = (
        str(getattr(element, "CurrentProcessId", "")),
        str(getattr(element, "CurrentNativeWindowHandle", "")),
        str(getattr(element, "CurrentControlType", "")),
        str(getattr(element, "CurrentAutomationId", "")),
        str(getattr(element, "CurrentClassName", "")),
        ",".join(str(value) for value in runtime_id),
    )
    if not any(fields):
        return ""
    return hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()


class SelectedTextLens:
    id = "selected_text"

    def __init__(
        self,
        *,
        max_characters: int = 8_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.max_characters = max(1, min(int(max_characters), 100_000))
        self.clock = clock

    def capture(
        self,
        target_window: int,
        *,
        foreground_getter: Optional[Callable[[], int]] = None,
        selection_reader: Callable[[int], SelectionProbe] = read_uia_selected_text,
    ) -> ContextLensCapture:
        captured_at = float(self.clock())
        target = int(target_window or 0)
        if not target:
            return _blocked_capture(target, captured_at, "missing_target")
        if foreground_getter is None:
            if os.name != "nt":
                return _blocked_capture(target, captured_at, "platform_unsupported")
            foreground_getter = lambda: int(ctypes.windll.user32.GetForegroundWindow())
        if int(foreground_getter()) != target:
            return _blocked_capture(target, captured_at, "foreground_changed")

        probe = selection_reader(self.max_characters + 1)
        if int(foreground_getter()) != target:
            return _blocked_capture(
                target,
                captured_at,
                "focus_changed_during_capture",
                source=probe.source,
            )
        if not probe.accessible:
            return _blocked_capture(
                target,
                captured_at,
                "uia_unavailable",
                source=probe.source,
            )
        if probe.is_password:
            return _blocked_capture(
                target,
                captured_at,
                "password_field",
                source=probe.source,
            )
        if not probe.text:
            return _blocked_capture(
                target,
                captured_at,
                "empty_selection",
                source=probe.source,
            )

        original_characters = len(probe.text)
        content = probe.text[: self.max_characters]
        return ContextLensCapture(
            lens_id=self.id,
            included=True,
            content=content,
            source=probe.source,
            target_window=target,
            captured_at=captured_at,
            original_characters=original_characters,
            truncated=original_characters > len(content),
            control_identity=probe.control_identity,
        )
