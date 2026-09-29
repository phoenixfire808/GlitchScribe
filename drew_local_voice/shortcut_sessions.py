"""Pure shortcut-session transitions for hold-to-talk and latching."""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class ShortcutState(str, Enum):
    IDLE = "idle"
    HELD = "held"
    LATCHED = "latched"


class ShortcutAction(str, Enum):
    NONE = "none"
    START_HOLD = "start_hold"
    START_LATCHED = "start_latched"
    LATCH = "latch"
    STOP_HOLD = "stop_hold"
    STOP_LATCHED = "stop_latched"
    CANCEL = "cancel"


class ShortcutSessionController:
    def __init__(
        self,
        *,
        hold_chord: Iterable[str],
        latch_chord: Iterable[str],
    ) -> None:
        self.hold_chord = frozenset(str(key).upper() for key in hold_chord)
        self.latch_chord = frozenset(str(key).upper() for key in latch_chord)
        if not self.hold_chord:
            raise ValueError("hold chord cannot be empty")
        if not self.hold_chord < self.latch_chord:
            raise ValueError("latch chord must strictly extend the hold chord")
        self.pressed_keys: set[str] = set()
        self.state = ShortcutState.IDLE
        self.toggle_stop_armed = False

    def handle(
        self,
        key: str,
        *,
        is_down: bool,
        injected: bool = False,
    ) -> ShortcutAction:
        normalized = str(key).upper()
        if injected:
            return ShortcutAction.NONE
        if is_down:
            if normalized in self.pressed_keys:
                return ShortcutAction.NONE
            self.pressed_keys.add(normalized)
            if (
                self.state is ShortcutState.LATCHED
                and self.toggle_stop_armed
                and self.latch_chord <= self.pressed_keys
            ):
                self.state = ShortcutState.IDLE
                self.toggle_stop_armed = False
                return ShortcutAction.STOP_LATCHED
            if (
                self.state is ShortcutState.IDLE
                and self.latch_chord <= self.pressed_keys
            ):
                self.state = ShortcutState.LATCHED
                self.toggle_stop_armed = False
                return ShortcutAction.START_LATCHED
            if self.state is ShortcutState.IDLE and self.hold_chord <= self.pressed_keys:
                self.state = ShortcutState.HELD
                return ShortcutAction.START_HOLD
            if (
                self.state is ShortcutState.HELD
                and self.latch_chord <= self.pressed_keys
            ):
                self.state = ShortcutState.LATCHED
                self.toggle_stop_armed = False
                return ShortcutAction.LATCH
            return ShortcutAction.NONE
        self.pressed_keys.discard(normalized)
        if (
            self.state is ShortcutState.LATCHED
            and not self.latch_chord <= self.pressed_keys
        ):
            self.toggle_stop_armed = True
        if self.state is ShortcutState.HELD and normalized in self.hold_chord:
            self.state = ShortcutState.IDLE
            return ShortcutAction.STOP_HOLD
        return ShortcutAction.NONE

    def reset(self) -> None:
        # Reset the logical recording session without inventing key-up events.
        # Keeping physically held inputs prevents an auto-repeat from being
        # mistaken for a fresh shortcut activation before real release events
        # arrive from the hook.
        self.state = ShortcutState.IDLE
        self.toggle_stop_armed = False
