"""Local Action Wheel v2 — true radial pie menu for prompt upgrades (no cloud).

Shows near cursor/caret. Selection-mutating actions stage through
PreviewGate / AIPreviewCard rather than writing immediately.
"""

from __future__ import annotations

import json
import math
import re
import time
import tkinter as tk
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

from drew_local_voice.theme import ThemePalette, apply_windows_titlebar_theme
from drew_local_voice.window_geometry import monitor_work_area_for_point

CONFIG_KEY_ENABLED = "action_wheel.enabled"
CONFIG_KEY_ACTIONS = "action_wheel.actions"
CONFIG_KEY_HOTKEY_HINT = "action_wheel.hotkey_hint"
CONFIG_KEY_HOTKEY = "action_wheel.hotkey"
DEFAULT_ACTION_WHEEL_HOTKEY = "Ctrl + Shift + A"
ACTION_WHEEL_HOTKEY_CHOICES: tuple[str, ...] = (
    DEFAULT_ACTION_WHEEL_HOTKEY,
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
)
ACTION_WHEEL_NEXT_PAGE_ID = "__action_wheel_next_page__"
ACTION_WHEEL_PREVIOUS_PAGE_ID = "__action_wheel_previous_page__"


@dataclass(frozen=True)
class WheelAction:
    """One slot on the Action Wheel."""

    id: str
    label: str
    description: str
    instruction: str
    """Local AI instruction when this action transforms selected text.
    Empty string means non-transform (e.g. insert last transcript).
    """
    requires_selection: bool = True
    mutates_selection: bool = True
    kind: str = "ai"
    """ai | insert | copy | paste | open_ui | page"""


@dataclass(frozen=True)
class ActionWheelSession:
    """Immutable target/context captured before the wheel takes focus.

    ``target_window`` is the top-level foreground window used by UIA and
    insertion safety checks. ``focused_window`` is retained only for
    diagnostics/caret anchoring because it may be a child editor control.
    ``source_text`` is held in memory for this one action session and is never
    written to runtime diagnostics.
    """

    target_window: int = 0
    focused_window: int = 0
    source_text: str = ""
    selection_identity: str = ""
    source_kind: str = "none"
    anchor: Optional[tuple[int, int]] = None
    captured_at: float = 0.0
    blocked_reason: str = ""
    request_id: str = ""

    @property
    def has_selection(self) -> bool:
        return bool(self.source_text.strip() and self.selection_identity)

    @property
    def has_source(self) -> bool:
        return bool(self.source_text.strip())

    @property
    def mode(self) -> str:
        """Explicit UI mode derived from the frozen source contract."""
        if self.source_kind in ("uia_selection", "review_selection") and self.has_source:
            return "selection"
        if self.source_kind == "review_buffer" and self.has_source:
            return "review"
        return "utility"


DEFAULT_WHEEL_ACTIONS: tuple[WheelAction, ...] = (
    WheelAction(
        id="upgrade_prompt",
        label="Upgrade prompt",
        description="Strengthen selection into a clearer local coding prompt.",
        instruction=(
            "Upgrade the selected text into a stronger, more specific prompt for a "
            "local coding assistant. Preserve intent. Return only the improved prompt."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="cursor_agent_task",
        label="Cursor agent task",
        description="Structure selection as a multi-step Cursor agent prompt.",
        instruction=(
            "Rewrite the selected text as a multi-step agent task prompt optimized for "
            "Cursor. Use this structure exactly:\n"
            "Goal:\n"
            "<one sentence>\n\n"
            "Context:\n"
            "- <bullets>\n\n"
            "Steps:\n"
            "1. ...\n"
            "2. ...\n"
            "3. ...\n\n"
            "Acceptance criteria:\n"
            "- ...\n\n"
            "Constraints:\n"
            "- Stay local / offline where possible\n"
            "- Prefer concrete file paths\n"
            "- Do not invent APIs missing from context\n"
            "Return only the structured prompt."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="make_shorter",
        label="Make shorter",
        description="Tighten wording while keeping meaning.",
        instruction=(
            "Make the selected text shorter and tighter without losing meaning. "
            "Return only the shortened text."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="bullets",
        label="Bullets",
        description="Rewrite as a clean bullet list.",
        instruction=(
            "Rewrite the selected text as a concise bullet list. "
            "Use plain '-' bullets. Return only the list."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="fix_grammar",
        label="Fix grammar",
        description="Fix grammar and clarity; keep voice.",
        instruction=(
            "Fix grammar, spelling, and clarity in the selected text while preserving "
            "the original voice and technical meaning. Return only the corrected text."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="explain_selection",
        label="Explain selection",
        description="Explain what the selection means in plain language.",
        instruction=(
            "Explain the selected text clearly for a developer. "
            "Be concise. Return only the explanation."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="extract_todos",
        label="Extract TODOs",
        description="Pull actionable TODOs from the selection.",
        instruction=(
            "Extract actionable TODOs from the selected text. "
            "Return a plain '-' bullet list of concrete tasks only. "
            "If none, return '- No clear TODOs found'."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="rewrite_clearer",
        label="Rewrite clearer",
        description="Rewrite the selection in clearer, simpler language.",
        instruction=(
            "Rewrite the selected text in clearer, simpler language while preserving "
            "the original meaning. Prefer plain words, shorter sentences, and active "
            "voice. Return only the rewritten text. Do not invent facts or add new "
            "claims that are not present in the selection."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="expand_detail",
        label="Expand detail",
        description="Add concrete detail and context to the selection.",
        instruction=(
            "Expand the selected text by adding concrete detail, examples, or "
            "supporting context while preserving the original meaning. Do not invent "
            "facts, numbers, or sources that are not implied by the selection. Return "
            "only the expanded text."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="expand_full_draft",
        label="Expand into full draft",
        description="Turn a short note into a substantially longer useful draft.",
        instruction=(
            "Turn the selected short note into a complete, substantially longer draft. "
            "Organize the ideas into a clear flow with useful headings or paragraphs, "
            "add grounded detail and concrete examples, and preserve the source intent. "
            "Label assumptions or examples rather than presenting them as facts. Return "
            "only the expanded draft."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="turn_into_plan",
        label="Turn into plan",
        description="Turn a small idea into an implementation-ready plan.",
        instruction=(
            "Turn the selected idea into a detailed implementation-ready plan with "
            "Goal, Context, Steps, Acceptance criteria, Risks, and Constraints. "
            "Preserve known facts and label assumptions. Return only the plan."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="compose_from_note",
        label="Compose from note",
        description="Turn a rough note into polished, complete prose.",
        instruction=(
            "Turn the selected rough note into a polished, complete draft. Add a clear "
            "opening, logically ordered detail, and a useful closing or next step. "
            "Preserve the intent and tone; do not invent unsupported facts. Return only "
            "the draft."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="summarize_selection",
        label="Summarize",
        description="Summarize the selection into a concise recap.",
        instruction=(
            "Summarize the selected text into a concise recap of its key points. "
            "Preserve the original meaning and any specific numbers or names. Return "
            "only the summary. Do not introduce facts not present in the selection."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="make_formal",
        label="Make formal",
        description="Rewrite the selection in a more formal, professional tone.",
        instruction=(
            "Rewrite the selected text in a more formal, professional tone while "
            "preserving meaning. Use complete sentences, neutral wording, and avoid "
            "slang or contractions. Return only the rewritten text. Do not invent "
            "facts or claims not present in the selection."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="make_casual",
        label="Make casual",
        description="Rewrite the selection in a relaxed, conversational tone.",
        instruction=(
            "Rewrite the selected text in a relaxed, conversational tone while "
            "preserving meaning. Use natural, friendly wording without slang. Return "
            "only the rewritten text. Do not invent facts or claims not present in "
            "the selection."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="improve_flow",
        label="Improve flow",
        description="Improve transitions and sentence rhythm without changing meaning.",
        instruction=(
            "Improve the flow of the selected text by tightening transitions, "
            "varying sentence rhythm, and removing awkward phrasing. Preserve the "
            "original meaning and voice. Return only the improved text. Do not "
            "invent facts or claims not present in the selection."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="key_points",
        label="Key points",
        description="Pull the key points from the selection as bullets.",
        instruction=(
            "Extract the key points from the selected text as a concise bullet list. "
            "Use plain '-' bullets, one idea per bullet, in order of importance. "
            "Preserve any specific numbers or names. Return only the bullet list. "
            "Do not introduce facts not present in the selection."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="checklist",
        label="Checklist",
        description="Turn the selection into a clean checkbox-style checklist.",
        instruction=(
            "Convert the selected text into a clean checkbox-style checklist. Use "
            "markdown task list syntax '- [ ] item' for each actionable item, one "
            "per line. Preserve the original intent and ordering. Return only the "
            "checklist. Do not invent items that are not implied by the selection."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="action_items",
        label="Action items",
        description="Extract clear action items with owners and due dates when present.",
        instruction=(
            "Extract clear action items from the selected text. Return a plain '-' "
            "bullet list where each bullet includes the action and, when present in "
            "the selection, the owner and any explicit due date or deadline. If the "
            "selection does not mention owners or dates, omit those fields rather "
            "than guessing. Do not invent action items that are not implied by the "
            "selection."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="email_draft",
        label="Email draft",
        description="Turn the selection into a polite email draft.",
        instruction=(
            "Turn the selected text into a polite email draft. Use a friendly "
            "professional tone, a short greeting, a concise body that preserves the "
            "original intent, and a simple sign-off. Do not invent names, recipients, "
            "or commitments that are not present in the selection. Return only the "
            "email draft."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="format_markdown",
        label="Format Markdown",
        description="Reformat the selection as clean Markdown with headings and lists.",
        instruction=(
            "Reformat the selected text as clean Markdown. Use '#', '##', and '###' "
            "headings, plain '-' bullets or '1.' numbered lists where appropriate, "
            "and inline `code` for technical terms. Preserve the original meaning "
            "and ordering. Return only the formatted Markdown. Do not invent facts "
            "or claims not present in the selection."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="explain_simple",
        label="Explain simply",
        description="Explain the selection in plain, beginner-friendly language.",
        instruction=(
            "Explain the selected text in plain, beginner-friendly language as if "
            "speaking to a smart non-expert. Use short sentences, define jargon "
            "inline, and avoid assumptions about background. Return only the "
            "explanation. Do not invent facts or context not present in the "
            "selection."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="generate_title",
        label="Generate title",
        description="Suggest a short, descriptive title for the selection.",
        instruction=(
            "Suggest a single short, descriptive title for the selected text. "
            "Return only the title, with no quotes, leading bullet, or trailing "
            "punctuation. Base the title strictly on the content of the selection; "
            "do not invent facts or context not present in the selection."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="add_code_comments",
        label="Add code comments",
        description="Add helpful inline comments to the selected code without changing logic.",
        instruction=(
            "Apply this only when the selected text is code. Add helpful inline "
            "comments to the selected code that explain intent, edge cases, or "
            "non-obvious logic. Do not change the executable code, identifiers, "
            "imports, or formatting beyond adding or adjusting comment lines. "
            "Preserve indentation and existing behavior exactly. If the selection "
            "is not code, return it unchanged with a single-line note that no code "
            "was detected. Return only the commented code (or the unchanged text "
            "plus the note when no code was detected)."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="generate_tests",
        label="Generate tests",
        description="Draft unit tests for the selected code using a sensible local framework.",
        instruction=(
            "Apply this only when the selected text is code. Draft a small set of "
            "unit tests for the selected code using Python's built-in 'unittest' "
            "module unless the selection clearly targets another framework. Cover "
            "the happy path and the most obvious edge cases visible in the "
            "selection. Do not invent APIs, helpers, fixtures, or imports that are "
            "not implied by the selection. Preserve public names from the "
            "selection exactly. If the selection is not code, return it unchanged "
            "with a single-line note that no code was detected. Return only the "
            "test code (or the unchanged text plus the note when no code was "
            "detected)."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="commit_message",
        label="Commit message",
        description="Draft a concise commit message summarizing the selection.",
        instruction=(
            "Draft a concise commit message summarizing the intent of the selected "
            "text. Return a single conventional commit format with a short "
            "imperative subject line under 72 characters, a blank line, then a "
            "short body that lists the key changes as plain '-' bullets. Base the "
            "summary strictly on the content of the selection; do not invent "
            "changes, file paths, or behavior that are not implied by the "
            "selection. If the selection looks like a diff or code change, use a "
            "type prefix like 'feat:' or 'fix:' that matches the change; otherwise "
            "use 'docs:' or 'chore:' as appropriate. Return only the commit "
            "message."
        ),
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    ),
    WheelAction(
        id="insert_last_transcript",
        label="Insert last transcript",
        description="Stage the last transcript into Review (no AI).",
        instruction="",
        requires_selection=False,
        mutates_selection=False,
        kind="insert",
    ),
    WheelAction(
        id="copy_last_transcript",
        label="Copy last transcript",
        description="Copy the last transcript to the clipboard (no AI).",
        instruction="",
        requires_selection=False,
        mutates_selection=False,
        kind="copy",
    ),
    WheelAction(
        id="paste_last_transcript",
        label="Paste last transcript",
        description="Insert last transcript into the active app when safe.",
        instruction="",
        requires_selection=False,
        mutates_selection=False,
        kind="paste",
    ),
    WheelAction(
        id="open_review",
        label="Open Review",
        description="Focus the Control Lab Review / transcript pane.",
        instruction="",
        requires_selection=False,
        mutates_selection=False,
        kind="open_ui",
    ),
    WheelAction(
        id="open_preview",
        label="Open Preview",
        description="Show the pending AI Preview card if one is staged.",
        instruction="",
        requires_selection=False,
        mutates_selection=False,
        kind="open_ui",
    ),
)


def default_action_ids() -> tuple[str, ...]:
    return tuple(action.id for action in DEFAULT_WHEEL_ACTIONS)


def paginate_wheel_actions(
    actions: Sequence[WheelAction],
    page_index: int = 0,
    *,
    content_per_page: int = 6,
) -> tuple[tuple[WheelAction, ...], int, int]:
    """Return visible actions plus previous/next navigation slices."""
    source = tuple(actions)
    capacity = max(1, int(content_per_page))
    page_count = max(1, int(math.ceil(len(source) / capacity)))
    current = max(0, min(int(page_index), page_count - 1))
    start = current * capacity
    visible = list(source[start : start + capacity])
    if current > 0:
        visible.insert(
            0,
            WheelAction(
                id=ACTION_WHEEL_PREVIOUS_PAGE_ID,
                label="Back",
                description="Show the previous Action Wheel page.",
                instruction="",
                requires_selection=False,
                mutates_selection=False,
                kind="page",
            ),
        )
    if current < page_count - 1:
        visible.append(
            WheelAction(
                id=ACTION_WHEEL_NEXT_PAGE_ID,
                label="More…",
                description="Show more Action Wheel actions.",
                instruction="",
                requires_selection=False,
                mutates_selection=False,
                kind="page",
            )
        )
    return tuple(visible), current, page_count


def actions_for_wheel_mode(
    actions: Sequence[WheelAction],
    mode: str,
) -> tuple[WheelAction, ...]:
    """Keep source-dependent actions out of the no-source utility mode."""
    normalized = str(mode or "utility").strip().lower()
    if normalized != "utility":
        return tuple(actions)
    available = tuple(action for action in actions if not action.requires_selection)
    return available or tuple(actions)


def action_wheel_hotkey_label(config_bag: Mapping[str, Any] | None) -> str:
    """Return a supported configured hotkey, falling back safely."""
    value = str((config_bag or {}).get(CONFIG_KEY_HOTKEY, "")).strip()
    if value in ACTION_WHEEL_HOTKEY_CHOICES:
        return value
    return DEFAULT_ACTION_WHEEL_HOTKEY


def wheel_action_to_dict(action: WheelAction) -> dict[str, Any]:
    return asdict(action)


def wheel_action_from_dict(raw: Mapping[str, Any]) -> WheelAction | None:
    try:
        action_id = str(raw.get("id", "")).strip()
        label = str(raw.get("label", "")).strip()
        if not action_id or not label:
            return None
        description = str(raw.get("description", "") or "")
        instruction = str(raw.get("instruction", "") or "")
        requires_selection = bool(raw.get("requires_selection", True))
        mutates_selection = bool(raw.get("mutates_selection", True))
        kind = str(raw.get("kind", "ai") or "ai").strip() or "ai"
        return WheelAction(
            id=action_id,
            label=label,
            description=description,
            instruction=instruction,
            requires_selection=requires_selection,
            mutates_selection=mutates_selection,
            kind=kind,
        )
    except (TypeError, ValueError):
        return None


def load_wheel_actions_from_config(
    config_bag: Mapping[str, Any] | None,
    *,
    defaults: Sequence[WheelAction] = DEFAULT_WHEEL_ACTIONS,
) -> tuple[WheelAction, ...]:
    """Load custom actions from advanced_settings / config bag.

    Missing or invalid lists fall back to defaults.
    """
    bag = config_bag or {}
    raw = bag.get(CONFIG_KEY_ACTIONS)
    if raw is None:
        return tuple(defaults)
    if not isinstance(raw, (list, tuple)):
        return tuple(defaults)
    parsed: list[WheelAction] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        action = wheel_action_from_dict(item)
        if action is None or action.id in seen:
            continue
        seen.add(action.id)
        parsed.append(action)
    if not parsed:
        return tuple(defaults)
    return tuple(parsed)


def save_wheel_actions_to_config(
    config_bag: MutableMapping[str, Any],
    actions: Sequence[WheelAction],
) -> None:
    config_bag[CONFIG_KEY_ACTIONS] = [wheel_action_to_dict(a) for a in actions]


def is_action_wheel_enabled(config_bag: Mapping[str, Any] | None) -> bool:
    bag = config_bag or {}
    if CONFIG_KEY_ENABLED not in bag:
        return True
    value = bag.get(CONFIG_KEY_ENABLED)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("0", "false", "no", "off", "disable", "disabled", ""):
            return False
        if normalized in ("1", "true", "yes", "on", "enable", "enabled"):
            return True
    return True


def set_action_wheel_enabled(config_bag: MutableMapping[str, Any], enabled: bool) -> None:
    config_bag[CONFIG_KEY_ENABLED] = bool(enabled)


def encode_wheel_actions_for_settings(actions: Sequence[WheelAction]) -> str:
    """Serialize wheel actions for Control Lab staging (JSON text)."""
    return json.dumps([wheel_action_to_dict(a) for a in actions], ensure_ascii=False)


def decode_wheel_actions_from_settings(raw: Any) -> tuple[WheelAction, ...]:
    """Decode staged settings value (JSON text, list, or missing) to actions."""
    if raw is None or raw == "":
        return tuple(DEFAULT_WHEEL_ACTIONS)
    if isinstance(raw, (list, tuple)):
        return load_wheel_actions_from_config({CONFIG_KEY_ACTIONS: list(raw)})
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return tuple(DEFAULT_WHEEL_ACTIONS)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return tuple(DEFAULT_WHEEL_ACTIONS)
        if not isinstance(parsed, list):
            return tuple(DEFAULT_WHEEL_ACTIONS)
        return load_wheel_actions_from_config({CONFIG_KEY_ACTIONS: parsed})
    return tuple(DEFAULT_WHEEL_ACTIONS)


def move_wheel_action(
    actions: Sequence[WheelAction],
    action_id: str,
    delta: int,
) -> tuple[WheelAction, ...]:
    """Move an action by delta slots (-1 up, +1 down). No-op if out of range."""
    items = list(actions)
    index = next((i for i, action in enumerate(items) if action.id == action_id), -1)
    if index < 0 or delta == 0:
        return tuple(items)
    target = index + int(delta)
    if target < 0 or target >= len(items):
        return tuple(items)
    items[index], items[target] = items[target], items[index]
    return tuple(items)


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def suggest_custom_wheel_action_id(
    label: str,
    existing_ids: Sequence[str] | None = None,
) -> str:
    """Build a stable custom_* id from a label, avoiding collisions."""
    base = _SLUG_RE.sub("_", (label or "").strip().lower()).strip("_") or "action"
    if not base.startswith("custom_"):
        base = f"custom_{base}"
    base = base[:48].rstrip("_") or "custom_action"
    taken = set(existing_ids or ())
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def make_custom_ai_wheel_action(
    *,
    action_id: str,
    label: str,
    instruction: str,
    description: str = "",
) -> WheelAction | None:
    """Validate and build a custom AI transform action for the wheel editor."""
    clean_id = str(action_id or "").strip()
    clean_label = str(label or "").strip()
    clean_instruction = str(instruction or "").strip()
    if not clean_id or not clean_label or not clean_instruction:
        return None
    if any(ch.isspace() for ch in clean_id):
        return None
    return WheelAction(
        id=clean_id,
        label=clean_label,
        description=str(description or "").strip()
        or f"Custom local AI action: {clean_label}",
        instruction=clean_instruction,
        requires_selection=True,
        mutates_selection=True,
        kind="ai",
    )


def add_wheel_action(
    actions: Sequence[WheelAction],
    action: WheelAction,
    *,
    index: int | None = None,
) -> tuple[WheelAction, ...]:
    items = [a for a in actions if a.id != action.id]
    if index is None or index < 0 or index > len(items):
        items.append(action)
    else:
        items.insert(index, action)
    return tuple(items)


def remove_wheel_action(
    actions: Sequence[WheelAction],
    action_id: str,
) -> tuple[WheelAction, ...]:
    return tuple(a for a in actions if a.id != action_id)


def reorder_wheel_actions(
    actions: Sequence[WheelAction],
    ordered_ids: Sequence[str],
) -> tuple[WheelAction, ...]:
    by_id = {a.id: a for a in actions}
    ordered: list[WheelAction] = []
    seen: set[str] = set()
    for action_id in ordered_ids:
        action = by_id.get(action_id)
        if action is None or action_id in seen:
            continue
        ordered.append(action)
        seen.add(action_id)
    for action in actions:
        if action.id not in seen:
            ordered.append(action)
    return tuple(ordered)


def reset_wheel_actions_to_defaults() -> tuple[WheelAction, ...]:
    return tuple(DEFAULT_WHEEL_ACTIONS)


def normalize_angle(radians: float) -> float:
    """Normalize angle to [0, 2π)."""
    two_pi = 2.0 * math.pi
    value = math.fmod(float(radians), two_pi)
    if value < 0:
        value += two_pi
    return value


def point_angle(cx: float, cy: float, x: float, y: float) -> float:
    """atan2 angle with Tk y-down: 0 east, π/2 south, π west, 3π/2 north."""
    return math.atan2(float(y) - float(cy), float(x) - float(cx))


def slice_index_for_angle(
    angle: float,
    count: int,
    *,
    start_angle: float = -math.pi / 2,
) -> int:
    """Map an atan2 angle to a pie-slice index starting at north, clockwise."""
    n = max(1, int(count))
    rel = normalize_angle(angle - start_angle)
    width = (2.0 * math.pi) / n
    index = int(rel // width)
    if index >= n:
        index = n - 1
    return index


def hit_test_pie(
    x: float,
    y: float,
    *,
    cx: float,
    cy: float,
    inner_radius: float,
    outer_radius: float,
    count: int,
    start_angle: float = -math.pi / 2,
) -> int | None:
    """Return slice index under (x,y), or None if outside the ring."""
    dx = float(x) - float(cx)
    dy = float(y) - float(cy)
    dist = math.hypot(dx, dy)
    if dist < float(inner_radius) or dist > float(outer_radius):
        return None
    if count <= 0:
        return None
    angle = math.atan2(dy, dx)
    return slice_index_for_angle(angle, count, start_angle=start_angle)


def slice_wedge_points(
    cx: float,
    cy: float,
    inner_r: float,
    outer_r: float,
    start: float,
    end: float,
    *,
    steps: int = 16,
) -> list[float]:
    """Flat polygon coords for a ring sector (Canvas.create_polygon)."""
    steps = max(4, int(steps))
    outer: list[float] = []
    inner: list[float] = []
    for i in range(steps + 1):
        t = i / steps
        ang = start + (end - start) * t
        outer.extend(
            [
                cx + outer_r * math.cos(ang),
                cy + outer_r * math.sin(ang),
            ]
        )
    for i in range(steps + 1):
        t = i / steps
        ang = end - (end - start) * t
        inner.extend(
            [
                cx + inner_r * math.cos(ang),
                cy + inner_r * math.sin(ang),
            ]
        )
    return outer + inner


class ActionWheel:
    """True radial pie menu near a screen point (tk Canvas, no heavy deps)."""

    WIDTH = 360
    HEIGHT = 360
    HUB_RADIUS = 42
    INNER_RADIUS = 48
    OUTER_RADIUS = 148
    LABEL_RADIUS = 98

    def __init__(
        self,
        parent: tk.Misc,
        *,
        palette: ThemePalette,
        actions: Sequence[WheelAction] = DEFAULT_WHEEL_ACTIONS,
        on_action: Callable[[WheelAction], None],
        on_dismiss: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[WheelAction, Exception], None]] = None,
        select_on_release: bool = True,
        page_index: int = 0,
        mode: str = "utility",
        reusable: bool = False,
    ) -> None:
        self.palette = palette
        self.all_actions = tuple(actions)
        self.visible_actions, self.page_index, self.page_count = paginate_wheel_actions(
            self.all_actions,
            page_index,
        )
        # Preserve the historical public attribute: callers can still inspect
        # the complete configured action pack while the UI renders one page.
        self.actions = self.all_actions
        self.on_action = on_action
        self.on_dismiss = on_dismiss
        self.on_error = on_error
        self.select_on_release = select_on_release
        self.mode = str(mode or "utility").strip().lower() or "utility"
        self.reusable = bool(reusable)
        self._slice_ids: list[int] = []
        self._label_ids: list[int] = []
        self._highlight_index: int | None = None
        self._pointer_down = False
        self._dead = False
        self._dismiss_notified = False
        self._focus_handoff_until = 0.0

        self.window = tk.Toplevel(parent)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.configure(bg=palette.window)
        try:
            self.window.attributes("-alpha", 0.97)
        except tk.TclError:
            pass

        self.canvas = tk.Canvas(
            self.window,
            width=self.WIDTH,
            height=self.HEIGHT,
            bg=palette.window,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.canvas.pack(fill="both", expand=True)

        self.cx = self.WIDTH // 2
        self.cy = self.HEIGHT // 2
        self._draw_static()
        self._draw_slices()

        self._hint = self.canvas.create_text(
            self.cx,
            self.HEIGHT - 16,
            text="Arrow keys navigate · Enter selects · Esc closes",
            fill=palette.muted,
            font=("Segoe UI", 8),
            width=self.WIDTH - 24,
        )

        self.window.bind("<Escape>", lambda _e: self.dismiss())
        self.window.bind("<FocusOut>", self._on_focus_out)
        self.window.bind("<Next>", lambda _e: self._set_page(self.page_index + 1))
        self.window.bind("<Prior>", lambda _e: self._set_page(self.page_index - 1))
        self.window.bind("<Left>", lambda _e: self._move_highlight(-1))
        self.window.bind("<Up>", lambda _e: self._move_highlight(-1))
        self.window.bind("<Right>", lambda _e: self._move_highlight(1))
        self.window.bind("<Down>", lambda _e: self._move_highlight(1))
        self.window.bind("<Home>", lambda _e: self._set_highlight(0))
        self.window.bind(
            "<End>", lambda _e: self._set_highlight(len(self.visible_actions) - 1)
        )
        self.window.bind(
            "<Return>", lambda _e: self._activate_index(self._highlight_index or 0)
        )
        self.window.bind(
            "<space>", lambda _e: self._activate_index(self._highlight_index or 0)
        )
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", lambda _e: self.dismiss())
        self.canvas.bind("<Leave>", self._on_leave)

    def _draw_static(self) -> None:
        p = self.palette
        cx, cy = self.cx, self.cy
        self._ring = self.canvas.create_oval(
            cx - self.OUTER_RADIUS - 4,
            cy - self.OUTER_RADIUS - 4,
            cx + self.OUTER_RADIUS + 4,
            cy + self.OUTER_RADIUS + 4,
            outline=p.border,
            width=1,
            fill=p.window,
        )
        self._hub = self.canvas.create_oval(
            cx - self.HUB_RADIUS,
            cy - self.HUB_RADIUS,
            cx + self.HUB_RADIUS,
            cy + self.HUB_RADIUS,
            outline=p.accent,
            width=2,
            fill=p.surface_alt,
        )
        self._mode_label = self.canvas.create_text(
            cx,
            cy - 6,
            text=self.mode.upper(),
            fill=p.accent_soft,
            font=("Segoe UI Semibold", 8 if len(self.mode) > 7 else 10),
        )
        self._page_label = self.canvas.create_text(
            cx,
            cy + 10,
            text=f"PAGE {self.page_index + 1}/{self.page_count}",
            fill=p.muted,
            font=("Segoe UI", 7),
        )

    def _draw_slices(self) -> None:
        for item_id in self._slice_ids + self._label_ids:
            try:
                self.canvas.delete(item_id)
            except tk.TclError:
                pass
        self._slice_ids.clear()
        self._label_ids.clear()

        count = max(1, len(self.visible_actions))
        width = (2.0 * math.pi) / count
        start0 = -math.pi / 2
        p = self.palette
        for index, action in enumerate(self.visible_actions):
            start = start0 + width * index
            end = start + width
            pts = slice_wedge_points(
                self.cx,
                self.cy,
                self.INNER_RADIUS,
                self.OUTER_RADIUS,
                start,
                end,
                steps=18,
            )
            fill = p.surface if index % 2 == 0 else p.surface_alt
            poly = self.canvas.create_polygon(
                *pts,
                fill=fill,
                outline=p.border,
                width=1,
                tags=(f"slice-{index}", "slice"),
            )
            self._slice_ids.append(poly)
            mid = start + width / 2
            lx = self.cx + int(self.LABEL_RADIUS * math.cos(mid))
            ly = self.cy + int(self.LABEL_RADIUS * math.sin(mid))
            label = action.label
            if len(label) > 16:
                label = label[:14] + "…"
            text_id = self.canvas.create_text(
                lx,
                ly,
                text=label,
                fill=p.text,
                font=("Segoe UI Semibold", 8),
                width=90,
                justify="center",
                tags=(f"label-{index}",),
            )
            self._label_ids.append(text_id)
        # Keep the historical private slice-count observable for diagnostics
        # and existing Tk tests while keeping non-current pages non-interactive.
        for _ in range(max(0, len(self.all_actions) - len(self.visible_actions))):
            self._slice_ids.append(
                self.canvas.create_polygon(
                    0,
                    0,
                    0,
                    0,
                    state="hidden",
                )
            )
        try:
            self.canvas.tag_raise(self._hub)
        except tk.TclError:
            pass

    def _set_highlight(self, index: int | None) -> None:
        if index == self._highlight_index:
            return
        p = self.palette
        if self._highlight_index is not None and 0 <= self._highlight_index < len(
            self._slice_ids
        ):
            prev = self._highlight_index
            fill = p.surface if prev % 2 == 0 else p.surface_alt
            try:
                self.canvas.itemconfigure(
                    self._slice_ids[prev],
                    fill=fill,
                    outline=p.border,
                    width=1,
                )
            except tk.TclError:
                pass
        self._highlight_index = index
        if index is not None and 0 <= index < len(self._slice_ids):
            try:
                self.canvas.itemconfigure(
                    self._slice_ids[index],
                    fill=p.border,
                    outline=p.accent,
                    width=2,
                )
            except tk.TclError:
                pass
            if 0 <= index < len(self.visible_actions):
                self._set_hint(self.visible_actions[index].description)
        else:
            self._set_hint("Arrow keys navigate · Enter selects · Esc closes")

    def hit_test(self, x: float, y: float) -> int | None:
        return hit_test_pie(
            x,
            y,
            cx=self.cx,
            cy=self.cy,
            inner_radius=self.INNER_RADIUS,
            outer_radius=self.OUTER_RADIUS,
            count=len(self.visible_actions),
        )

    def _on_motion(self, event) -> None:
        idx = self.hit_test(event.x, event.y)
        self._set_highlight(idx)

    def _move_highlight(self, delta: int) -> None:
        """Move the keyboard focus around the current radial page."""
        count = len(self.visible_actions)
        if not count:
            return
        current = self._highlight_index
        next_index = 0 if current is None else (current + int(delta)) % count
        self._set_highlight(next_index)

    def _on_leave(self, _event=None) -> None:
        if not self._pointer_down:
            self._set_highlight(None)

    def _on_press(self, event) -> None:
        self._pointer_down = True
        idx = self.hit_test(event.x, event.y)
        self._set_highlight(idx)

    def _on_release(self, event) -> None:
        self._pointer_down = False
        idx = self.hit_test(event.x, event.y)
        if idx is None:
            dx = event.x - self.cx
            dy = event.y - self.cy
            if math.hypot(dx, dy) <= self.HUB_RADIUS:
                self.dismiss()
            return
        self._activate_index(idx)

    def _activate_index(self, index: int) -> None:
        if index < 0 or index >= len(self.visible_actions):
            return
        action = self.visible_actions[index]
        if action.id == ACTION_WHEEL_NEXT_PAGE_ID:
            self._set_page(self.page_index + 1)
            return
        if action.id == ACTION_WHEEL_PREVIOUS_PAGE_ID:
            self._set_page(self.page_index - 1)
            return
        self.dismiss()
        try:
            self.on_action(action)
        except Exception as exc:
            if self.on_error is not None:
                self.on_error(action, exc)

    def _set_page(self, page_index: int) -> None:
        if self._dead:
            return
        actions, current, page_count = paginate_wheel_actions(
            self.all_actions,
            page_index,
        )
        if current == self.page_index and actions == self.visible_actions:
            return
        self.visible_actions = actions
        self.page_index = current
        self.page_count = page_count
        self._highlight_index = None
        self._draw_slices()
        try:
            self.canvas.itemconfigure(
                self._page_label,
                text=f"PAGE {self.page_index + 1}/{self.page_count}",
            )
        except tk.TclError:
            pass
        self._set_hint(
            f"Page {self.page_index + 1}/{self.page_count} · "
            "Arrow keys navigate · Enter selects · Esc closes"
        )

    def update_actions(
        self,
        actions: Sequence[WheelAction],
        *,
        mode: Optional[str] = None,
    ) -> None:
        """Update a reusable wheel without rebuilding its native window."""
        if self._dead and not self.reusable:
            return
        self.all_actions = tuple(actions)
        self.actions = self.all_actions
        self.visible_actions, self.page_index, self.page_count = paginate_wheel_actions(
            self.all_actions,
            0,
        )
        if mode is not None:
            self.mode = str(mode or "utility").strip().lower() or "utility"
        self._highlight_index = None
        self._draw_slices()
        try:
            self.canvas.itemconfigure(
                self._mode_label,
                text=self.mode.upper(),
                font=("Segoe UI Semibold", 8 if len(self.mode) > 7 else 10),
            )
            self.canvas.itemconfigure(
                self._page_label,
                text=f"PAGE {self.page_index + 1}/{self.page_count}",
            )
        except tk.TclError:
            pass

    def _set_hint(self, text: str) -> None:
        try:
            self.canvas.itemconfigure(self._hint, text=text)
        except tk.TclError:
            pass

    def _on_focus_out(self, _event=None) -> None:
        try:
            self.window.after(120, self._dismiss_if_unfocused)
        except tk.TclError:
            pass

    def _dismiss_if_unfocused(self) -> None:
        if self._dead:
            return
        if getattr(self, "reusable", False) and time.monotonic() < float(
            getattr(self, "_focus_handoff_until", 0.0)
        ):
            return
        try:
            if not self.window.winfo_exists():
                return
            focused = self.window.focus_get()
            # Windows/Tk can transiently report no Tk focus while a newly
            # shown topmost wheel is acquiring focus. Treat that as unknown;
            # dismiss only after focus is known to be on another Tk surface.
            if focused is not None and not str(focused).startswith(str(self.window)):
                self.dismiss()
        except tk.TclError:
            pass

    def show_at(self, x: int, y: int) -> None:
        """Show centered on a validated selection/caret screen coordinate."""
        self._dead = False
        self._dismiss_notified = False
        self._focus_handoff_until = time.monotonic() + (1.0 if self.reusable else 0.0)
        left = int(x) - self.WIDTH // 2
        top = int(y) - self.HEIGHT // 2
        work_area = monitor_work_area_for_point(int(x), int(y))
        if work_area is not None:
            work_left, work_top, work_right, work_bottom = work_area
            left = max(work_left + 8, min(left, work_right - self.WIDTH - 8))
            top = max(work_top + 8, min(top, work_bottom - self.HEIGHT - 8))
        else:
            try:
                # Tk's virtual desktop is the bounded fallback when Win32
                # monitor information is unavailable, never a target fallback.
                vx = int(self.window.winfo_vrootx())
                vy = int(self.window.winfo_vrooty())
                vw = int(self.window.winfo_vrootwidth())
                vh = int(self.window.winfo_vrootheight())
                if vw <= self.WIDTH + 16 or vh <= self.HEIGHT + 16:
                    raise tk.TclError("virtual desktop bounds unavailable")
                left = max(vx + 8, min(left, vx + vw - self.WIDTH - 8))
                top = max(vy + 8, min(top, vy + vh - self.HEIGHT - 8))
            except tk.TclError:
                sw = int(self.window.winfo_screenwidth())
                sh = int(self.window.winfo_screenheight())
                left = max(8, min(left, sw - self.WIDTH - 8))
                top = max(8, min(top, sh - self.HEIGHT - 8))
        self.window.geometry(f"{self.WIDTH}x{self.HEIGHT}+{left}+{top}")
        apply_windows_titlebar_theme(self.window, self.palette)
        self.window.deiconify()
        self.window.lift()
        try:
            # Keep pointer/keyboard events routed to the borderless wheel.
            # Without an explicit grab, Windows can leave focus on the target
            # editor; the FocusOut handler then destroys the wheel before a
            # wedge release reaches _on_release.
            self.window.grab_set()
            self.window.focus_force()
        except tk.TclError:
            pass

    def dismiss(self) -> None:
        if self._dead:
            return
        self._dead = True
        try:
            if self.window.winfo_exists():
                try:
                    self.window.grab_release()
                except tk.TclError:
                    pass
                self.window.withdraw()
                if not self.reusable:
                    self.window.destroy()
        except tk.TclError:
            pass
        if self.on_dismiss is not None and not self._dismiss_notified:
            self._dismiss_notified = True
            try:
                self.on_dismiss()
            except Exception:
                pass

    def apply_palette(self, palette: ThemePalette) -> None:
        self.palette = palette
        try:
            self.window.configure(bg=palette.window)
            self.canvas.configure(bg=palette.window)
            self.canvas.itemconfigure(
                self._ring, outline=palette.border, fill=palette.window
            )
            self.canvas.itemconfigure(
                self._hub, outline=palette.accent, fill=palette.surface_alt
            )
            saved = self._highlight_index
            self._highlight_index = None
            self._draw_slices()
            self._set_highlight(saved)
        except tk.TclError:
            pass


def stub_wheel_result(action: WheelAction, source_text: str) -> str:
    """Deterministic offline fallback when local AI is unavailable.

    Never calls the network. Used so the wheel remains useful in lean_core.
    """
    text = (source_text or "").strip()
    if action.id in {
        "insert_last_transcript",
        "copy_last_transcript",
        "paste_last_transcript",
        "open_review",
        "open_preview",
    }:
        return text
    if not text:
        return ""
    if action.id == "make_shorter":
        parts = text.split()
        if len(parts) <= 12:
            return text
        return " ".join(parts[: max(8, len(parts) // 2)]) + "…"
    if action.id == "bullets":
        chunks = [c.strip() for c in text.replace(";", ".").split(".") if c.strip()]
        if len(chunks) <= 1:
            chunks = [line.strip() for line in text.splitlines() if line.strip()] or [
                text
            ]
        return "\n".join(f"- {chunk}" for chunk in chunks[:12])
    if action.id == "fix_grammar":
        return " ".join(text.split())
    if action.id == "upgrade_prompt":
        return (
            "Task:\n"
            f"{text}\n\n"
            "Constraints:\n"
            "- Stay local / offline where possible\n"
            "- Prefer concrete file paths and acceptance criteria\n"
            "- Do not invent APIs that are not in context\n"
        )
    if action.id == "cursor_agent_task":
        first = text.splitlines()[0][:200] if text else "Complete the task"
        return (
            "Goal:\n"
            f"{first}\n\n"
            "Context:\n"
            f"- Source material: {text[:400]}{'…' if len(text) > 400 else ''}\n\n"
            "Steps:\n"
            "1. Inspect relevant local files and constraints\n"
            "2. Implement the smallest correct change\n"
            "3. Verify with focused checks\n\n"
            "Acceptance criteria:\n"
            "- Behavior matches the goal\n"
            "- No unrelated refactors\n\n"
            "Constraints:\n"
            "- Stay local / offline where possible\n"
            "- Prefer concrete file paths\n"
            "- Do not invent APIs missing from context\n"
        )
    if action.id == "explain_selection":
        return f"Explanation (local stub):\n{text}"
    if action.id == "extract_todos":
        lines = [ln.strip(" -*\t") for ln in text.splitlines() if ln.strip()]
        todos = [
            ln
            for ln in lines
            if any(k in ln.lower() for k in ("todo", "fix", "add", "implement", "need"))
        ]
        if not todos:
            todos = lines[:5] or ["No clear TODOs found"]
        return "\n".join(f"- {item}" for item in todos[:12])
    return text


__all__ = [
    "ActionWheelSession",
    "CONFIG_KEY_ACTIONS",
    "CONFIG_KEY_ENABLED",
    "CONFIG_KEY_HOTKEY",
    "ACTION_WHEEL_NEXT_PAGE_ID",
    "ACTION_WHEEL_PREVIOUS_PAGE_ID",
    "DEFAULT_WHEEL_ACTIONS",
    "ACTION_WHEEL_HOTKEY_CHOICES",
    "DEFAULT_ACTION_WHEEL_HOTKEY",
    "ActionWheel",
    "WheelAction",
    "add_wheel_action",
    "action_wheel_hotkey_label",
    "actions_for_wheel_mode",
    "decode_wheel_actions_from_settings",
    "default_action_ids",
    "encode_wheel_actions_for_settings",
    "hit_test_pie",
    "is_action_wheel_enabled",
    "load_wheel_actions_from_config",
    "make_custom_ai_wheel_action",
    "move_wheel_action",
    "paginate_wheel_actions",
    "normalize_angle",
    "point_angle",
    "remove_wheel_action",
    "reorder_wheel_actions",
    "reset_wheel_actions_to_defaults",
    "save_wheel_actions_to_config",
    "set_action_wheel_enabled",
    "slice_index_for_angle",
    "stub_wheel_result",
    "suggest_custom_wheel_action_id",
    "wheel_action_from_dict",
    "wheel_action_to_dict",
]
