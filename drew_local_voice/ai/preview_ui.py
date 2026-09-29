"""Trusted-code Tk Preview Card for local AI drafts."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Mapping, Optional

from drew_local_voice.theme import ThemePalette, apply_windows_titlebar_theme
from drew_local_voice.window_geometry import monitor_work_area_for_point
from drew_local_voice.scrolling import (
    WheelDeltaAccumulator,
    install_scroll_bindings,
    uninstall_scroll_bindings,
)

from .preview import PreviewDraft
from .transformation_controls import (
    AMPLIFICATION_CHOICES,
    STYLE_CHOICES,
    TransformationControls,
    build_modifier_contract,
)


class AIPreviewCard:
    """Preview surface for full drafts and immediate Action Wheel transforms.

    ``compact=True`` intentionally decouples card visibility from provider
    completion.  The card can render a loading shell with no draft, then bind
    one immutable ``PreviewDraft`` when the provider returns.  This keeps the
    UI responsive without weakening PreviewGate or approval semantics.
    """

    def __init__(
        self,
        parent: tk.Misc,
        draft: Optional[PreviewDraft],
        *,
        palette: ThemePalette,
        on_stage: Callable[[str, str], None],
        on_replace: Callable[[str, str], None],
        on_copy: Callable[[str, str], None],
        on_retry: Callable[[str, str], None],
        on_cancel: Callable[[str], None],
        on_retry_modified: Optional[
            Callable[[str, str, Mapping[str, Any]], None]
        ] = None,
        on_variant_retry: Optional[
            Callable[[str, str, Mapping[str, Any]], None]
        ] = None,
        compact: bool = False,
        request_id: str = "",
        action_label: str = "AI transformation",
        anchor: Optional[tuple[int, int]] = None,
    ) -> None:
        self.parent = parent
        self.draft = draft
        self.palette = palette
        self.anchor = (
            (int(anchor[0]), int(anchor[1]))
            if anchor is not None and len(anchor) >= 2
            else None
        )
        self.on_stage = on_stage
        self.on_replace = on_replace
        self.on_copy = on_copy
        self.on_retry = on_retry
        self.on_cancel = on_cancel
        self.on_retry_modified = on_retry_modified
        self.on_variant_retry = on_variant_retry
        self.compact = bool(compact)
        self.controls = TransformationControls()
        self._custom_amplification = ""
        self._adaptive_ready = False
        self.request_id = str(
            request_id or (draft.transaction_id if draft is not None else "")
        )
        self.action_label = str(action_label or "AI transformation").strip()
        self._error = False
        self._compact_scroll_canvas: Optional[tk.Canvas] = None
        self._compact_scroll_bindtag: Optional[str] = None
        self._compact_scroll_window_id: Optional[int] = None
        self._compact_wheel_accumulator = WheelDeltaAccumulator()
        self.window = tk.Toplevel(parent)
        self.window.withdraw()
        self.window.title(
            f"GlitchScribe — Transform: {self.action_label}"
            if self.compact
            else "GlitchScribe — Local AI Preview"
        )
        if self.compact:
            # Dedicated result window: small enough to stay out of the way,
            # large enough for a readable multi-line result.
            self.window.geometry("720x560")
            self.window.minsize(560, 420)
        else:
            self.window.geometry("860x700")
            self.window.minsize(720, 560)
        self.window.configure(bg=palette.window)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._cancel)

        outer_pad_x, outer_pad_y = ((18, 14) if self.compact else (22, 20))
        outer = tk.Frame(
            self.window,
            bg=palette.window,
            padx=outer_pad_x,
            pady=outer_pad_y,
        )
        outer.pack(fill="both", expand=True)
        if self.compact:
            outer.configure(width=720, height=560)
            outer.pack_propagate(False)
            self._build_compact(outer)
            self._compact_scroll_bindtag = install_scroll_bindings(
                self.window,
                self._on_compact_mousewheel,
                button4_handler=self._on_compact_button4,
                button5_handler=self._on_compact_button5,
            )
        else:
            self._build_full(outer)

    def _build_full(self, outer: tk.Frame) -> None:
        tk.Label(
            outer,
            text="AI PREVIEW CARD",
            bg=self.palette.window,
            fg=self.palette.text,
            font=("Segoe UI Semibold", 18),
            anchor="w",
        ).pack(fill="x")
        draft = self.draft
        assert draft is not None
        tk.Label(
            outer,
            text=(
                f"{draft.provider_label}  •  Context: {', '.join(draft.context_lenses)}  •  "
                f"Fingerprint: {draft.fingerprint[:12]}"
            ),
            bg=self.palette.window,
            fg=self.palette.accent_soft,
            font=("Segoe UI", 10),
            anchor="w",
        ).pack(fill="x", pady=(3, 14))
        tk.Label(
            outer,
            text=(
                "Nothing is applied automatically. Replace Selection is one-time "
                "and rechecks the exact private selection."
            ),
            bg=self.palette.surface_alt,
            fg=self.palette.text,
            padx=12,
            pady=9,
            anchor="w",
        ).pack(fill="x", pady=(0, 14))

        self._label(outer, "INSTRUCTION")
        instruction = tk.Label(
            outer,
            text=draft.instruction,
            bg=self.palette.surface,
            fg=self.palette.text,
            padx=12,
            pady=9,
            anchor="w",
            justify="left",
            wraplength=790,
        )
        instruction.pack(fill="x", pady=(4, 12))
        self.instruction = instruction

        panes = tk.PanedWindow(
            outer,
            orient="vertical",
            sashwidth=6,
            bg=self.palette.border,
            bd=0,
        )
        panes.pack(fill="both", expand=True)
        original_frame, self.original_text, _original_heading = self._text_panel(
            panes,
            "ORIGINAL — PRIVATE SELECTED TEXT",
            draft.original,
        )
        result_title = (
            "LOCAL AI RESULT — ALREADY GOOD (NO CHANGE)"
            if draft.unchanged
            else "LOCAL AI RESULT — REVIEW BEFORE USE"
        )
        result_frame, self.result_text, self.result_heading = self._text_panel(
            panes,
            result_title,
            draft.result,
        )
        panes.add(original_frame, minsize=120)
        panes.add(result_frame, minsize=160)
        self._build_actions(outer)

    def _build_compact(self, outer: tk.Frame) -> None:
        """Build the result-first transformation window.

        This surface intentionally contains no settings/adaptive-controls panel:
        generation state, the text result, and safe review actions are the only
        things shown while a transformation is running.
        """
        header = tk.Frame(outer, bg=self.palette.window)
        header.pack(fill="x", pady=(0, 4))
        tk.Label(
            header,
            text="TRANSFORM",
            bg=self.palette.window,
            fg=self.palette.text,
            font=("Segoe UI Semibold", 16),
            anchor="w",
        ).pack(side="left")
        self.preview_badge = tk.Label(
            header,
            text="PREVIEW • NOT APPLIED",
            bg=self.palette.surface_alt,
            fg=self.palette.accent_soft,
            font=("Segoe UI Semibold", 8),
            padx=8,
            pady=4,
        )
        self.preview_badge.pack(side="right")
        tk.Label(
            outer,
            text=self.action_label,
            bg=self.palette.window,
            fg=self.palette.accent_soft,
            font=("Segoe UI", 11),
            anchor="w",
        ).pack(fill="x", pady=(0, 6))
        self.status_label = tk.Label(
            outer,
            text="Preparing preview…",
            bg=self.palette.surface_alt,
            fg=self.palette.text,
            padx=10,
            pady=6,
            anchor="w",
        )
        self.status_label.pack(fill="x", pady=(0, 8))
        result_frame, self.result_text, self.result_heading = self._text_panel(
            outer,
            "RESULT",
            "Waiting for the transformation…" if self.draft is None else self.draft.result,
        )
        result_frame.pack(fill="both", expand=True, pady=(0, 8))
        self._build_pass_controls(outer)
        self._build_actions(outer)
        self.original_text = None
        self.instruction = None
        if self.draft is None:
            self._set_compact_controls(ready=False)
            self._set_pass_controls(ready=False)
        else:
            self.set_draft(self.draft)

    def _build_adaptive_controls(self, outer: tk.Frame) -> None:
        """Build stage-aware modifiers without hiding the approval actions."""
        panel = tk.LabelFrame(
            outer,
            text=" ADAPTIVE CONTROLS ",
            bg=self.palette.window,
            fg=self.palette.muted,
            bd=1,
            relief="groove",
            padx=8,
            pady=4,
        )
        panel.pack(fill="x", pady=(0, 3))
        self._adaptive_widgets: list[tk.Widget] = []
        self._execution_buttons: dict[str, tk.Button] = {}
        self.adaptive_summary = tk.Label(
            panel,
            text="Choose a modifier before rerunning; nothing is applied automatically.",
            bg=self.palette.window,
            fg=self.palette.accent_soft,
            anchor="w",
            justify="left",
        )
        self.adaptive_summary.pack(fill="x", pady=(0, 3))

        prompt_row = tk.Frame(panel, bg=self.palette.window)
        prompt_row.pack(fill="x", pady=(0, 3))
        tk.Label(
            prompt_row,
            text="Prompt strength",
            bg=self.palette.window,
            fg=self.palette.text,
            anchor="w",
        ).pack(side="left")
        self._amplification_buttons: dict[float, tk.Button] = {}
        for value in AMPLIFICATION_CHOICES:
            button = tk.Button(
                prompt_row,
                text=f"{value:g}×",
                command=lambda selected=value: self._set_amplification(selected),
                relief="flat",
                padx=7,
                pady=2,
                cursor="hand2",
            )
            button.pack(side="left", padx=(7 if value == 1 else 3, 0))
            self._amplification_buttons[value] = button
            self._adaptive_widgets.append(button)
        self.custom_amplification_entry = tk.Entry(
            prompt_row,
            width=5,
            relief="flat",
            bg=self.palette.field,
            fg=self.palette.text,
            insertbackground=self.palette.accent,
        )
        self.custom_amplification_entry.insert(0, "custom")
        self.custom_amplification_entry.bind("<Return>", self._commit_custom_amplification)
        self.custom_amplification_entry.pack(side="left", padx=(7, 0))
        self._adaptive_widgets.append(self.custom_amplification_entry)

        style_row = tk.Frame(panel, bg=self.palette.window)
        style_row.pack(fill="x", pady=(0, 2))
        tk.Label(
            style_row,
            text="Style",
            bg=self.palette.window,
            fg=self.palette.text,
            anchor="w",
        ).pack(side="left")
        self.style_var = tk.StringVar(value=STYLE_CHOICES[0])
        self.style_menu = tk.OptionMenu(
            style_row,
            self.style_var,
            *STYLE_CHOICES,
            command=self._set_style,
        )
        self.style_menu.configure(
            relief="flat",
            bg=self.palette.surface_alt,
            fg=self.palette.text,
            activebackground=self.palette.border,
            activeforeground=self.palette.text,
            highlightthickness=0,
            padx=5,
            pady=1,
        )
        self.style_menu.pack(side="left", padx=(35, 0))
        self._adaptive_widgets.append(self.style_menu)

        length_row = tk.Frame(panel, bg=self.palette.window)
        length_row.pack(fill="x", pady=(0, 2))
        tk.Label(
            length_row,
            text="Length",
            bg=self.palette.window,
            fg=self.palette.text,
            anchor="w",
        ).pack(side="left")
        self.length_value_var = tk.StringVar(value="1.0×")
        self.length_scale = tk.Scale(
            length_row,
            from_=50,
            to=200,
            resolution=5,
            orient="horizontal",
            showvalue=False,
            command=self._set_length,
            bg=self.palette.window,
            fg=self.palette.text,
            troughcolor=self.palette.surface_alt,
            activebackground=self.palette.accent,
            highlightthickness=0,
            bd=0,
            length=180,
        )
        self.length_scale.set(100)
        self.length_scale.pack(side="left", padx=(27, 4))
        self.length_value_label = tk.Label(
            length_row,
            textvariable=self.length_value_var,
            bg=self.palette.window,
            fg=self.palette.accent_soft,
            width=5,
            anchor="w",
        )
        self.length_value_label.pack(side="left")
        self._adaptive_widgets.append(self.length_scale)

        execution_row = tk.Frame(panel, bg=self.palette.window)
        execution_row.pack(fill="x", pady=(2, 0))
        tk.Label(
            execution_row,
            text="Execution",
            bg=self.palette.window,
            fg=self.palette.text,
            anchor="w",
        ).pack(side="left")
        for key, label in (
            ("modified", "Rerun modified"),
            ("parallel", "Parallel variants"),
            ("batch", "Batch retry"),
        ):
            button = tk.Button(
                execution_row,
                text=label,
                command=lambda selected=key: self._dispatch_modified(selected),
                relief="flat",
                padx=6,
                pady=2,
                cursor="hand2",
            )
            button.pack(side="left", padx=(7 if key == "modified" else 3, 0))
            self._execution_buttons[key] = button
            self._adaptive_widgets.append(button)
        self._refresh_adaptive_summary()
        self._set_adaptive_controls(enabled=False)

    def _set_amplification(self, value: float) -> None:
        self.controls = TransformationControls(
            amplification=value,
            style=self.controls.style,
            length_scale=self.controls.length_scale,
            execution=self.controls.execution,
        ).normalized()
        self._refresh_adaptive_summary()

    def _commit_custom_amplification(self, _event=None) -> str:
        try:
            value = float(self.custom_amplification_entry.get().strip())
        except (TypeError, ValueError):
            value = 1.0
        self._set_amplification(value)
        return "break"

    def _set_style(self, value: str) -> None:
        self.controls = TransformationControls(
            amplification=self.controls.amplification,
            style=value,
            length_scale=self.controls.length_scale,
            execution=self.controls.execution,
        ).normalized()
        self._refresh_adaptive_summary()

    def _set_length(self, value: str) -> None:
        self.controls = TransformationControls(
            amplification=self.controls.amplification,
            style=self.controls.style,
            length_scale=float(value) / 100.0,
            execution=self.controls.execution,
        ).normalized()
        self.length_value_var.set(f"{self.controls.length_scale:g}×")
        self._refresh_adaptive_summary()

    def _refresh_adaptive_summary(self) -> None:
        if not hasattr(self, "adaptive_summary"):
            return
        contract = build_modifier_contract(self.controls)
        self.adaptive_summary.configure(
            text=(
                f"{contract['amplification']:g}× strength  ·  {contract['style']}  ·  "
                f"{contract['length_scale']:g}× length  ·  "
                f"{contract['execution']} preview"
            )
        )
        for value, button in getattr(self, "_amplification_buttons", {}).items():
            selected = abs(float(value) - contract["amplification"]) < 0.01
            button.configure(
                bg=self.palette.accent if selected else self.palette.surface_alt,
                fg=self.palette.on_accent if selected else self.palette.text,
                activebackground=self.palette.accent_hover if selected else self.palette.border,
                activeforeground=self.palette.on_accent if selected else self.palette.text,
            )

    def _set_adaptive_controls(self, *, enabled: bool) -> None:
        self._adaptive_ready = bool(enabled)
        state = "normal" if enabled else "disabled"
        for widget in getattr(self, "_adaptive_widgets", ()):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
        if enabled:
            self._refresh_adaptive_summary()

    def _dispatch_modified(self, execution: str) -> None:
        if not self._adaptive_ready or self.draft is None:
            return
        self.controls = TransformationControls(
            amplification=self.controls.amplification,
            style=self.controls.style,
            length_scale=self.controls.length_scale,
            execution=execution,
        ).normalized()
        contract = build_modifier_contract(self.controls)
        callback = self.on_variant_retry if execution in {"parallel", "batch"} else self.on_retry_modified
        if callback is None:
            callback = self.on_retry_modified or self.on_retry
        if callback is self.on_retry:
            self.on_retry(self.draft.transaction_id, self.draft.fingerprint)
        else:
            callback(self.draft.transaction_id, self.draft.fingerprint, contract)

    def _build_pass_controls(self, outer: tk.Frame) -> None:
        """Keep bounded 1×/2×/3× rerun choices visible in compact mode."""
        panel = tk.Frame(outer, bg=self.palette.window)
        panel.pack(fill="x", pady=(0, 5))
        tk.Label(
            panel,
            text="RUN AGAIN",
            bg=self.palette.window,
            fg=self.palette.muted,
            font=("Segoe UI Semibold", 8),
            anchor="w",
        ).pack(side="left", padx=(0, 8))
        self._pass_buttons: dict[int, tk.Button] = {}
        choices = (
            (1, "1×  RUN", "current"),
            (2, "2×  VARIANTS", "parallel"),
            (3, "3×  VARIANTS", "batch"),
        )
        for count, label, execution in choices:
            button = tk.Button(
                panel,
                text=label,
                command=lambda selected=count, mode=execution: self._run_passes(selected, mode),
                bg=self.palette.surface_alt,
                fg=self.palette.text,
                activebackground=self.palette.border,
                activeforeground=self.palette.text,
                relief="flat",
                padx=7,
                pady=4,
                cursor="arrow",
            )
            button.pack(side="left", padx=(0, 5))
            self._pass_buttons[count] = button
        tk.Label(
            panel,
            text="review only",
            bg=self.palette.window,
            fg=self.palette.accent_soft,
            font=("Segoe UI", 8),
        ).pack(side="right")

    def _run_passes(self, count: int, execution: str) -> None:
        if self.draft is None:
            return
        count = max(1, min(3, int(count)))
        if count == 1:
            self.on_retry(self.draft.transaction_id, self.draft.fingerprint)
            return
        callback = self.on_variant_retry or self.on_retry_modified or self.on_retry
        contract = {"execution": execution, "requested_passes": count}
        if callback is self.on_retry:
            self.on_retry(self.draft.transaction_id, self.draft.fingerprint)
        else:
            callback(self.draft.transaction_id, self.draft.fingerprint, contract)

    def _set_pass_controls(self, *, ready: bool) -> None:
        for count, button in getattr(self, "_pass_buttons", {}).items():
            enabled = bool(ready and self.draft is not None and not self._error)
            button.configure(
                state=("normal" if enabled else "disabled"),
                cursor=("hand2" if enabled else "arrow"),
                fg=(self.palette.text if enabled else self.palette.muted),
            )

    def _build_actions(self, outer: tk.Frame) -> None:
        actions = tk.Frame(outer, bg=self.palette.window)
        actions.pack(fill="x", pady=(2, 0))
        self.action_buttons: dict[str, tk.Button] = {}
        for key, display_label, callback, accent in (
            ("Stage in Review", "Stage" if self.compact else "Stage in Review", self._stage, True),
            ("Replace Selection", "Replace" if self.compact else "Replace Selection", self._replace, False),
            ("Copy", "Copy", self._copy, False),
            ("Retry", "Retry", self._retry, False),
            ("Cancel", "Cancel", self._cancel, False),
        ):
            button = tk.Button(
                actions,
                text=display_label,
                command=callback,
                bg=self.palette.accent if accent else self.palette.surface_alt,
                fg=self.palette.on_accent if accent else self.palette.text,
                activebackground=(
                    self.palette.accent_hover if accent else self.palette.border
                ),
                activeforeground=self.palette.on_accent if accent else self.palette.text,
                relief="flat",
                padx=7 if self.compact else 14,
                pady=6 if self.compact else 8,
                cursor="hand2",
            )
            side = (
                "right" if self.compact and key in {"Retry", "Cancel"} else "left"
            )
            button.pack(side=side, padx=(0, 8), pady=(0, 1))
            self.action_buttons[key] = button
        if not self.compact:
            draft = self.draft
            assert draft is not None
            if draft.unchanged:
                self.action_buttons["Stage in Review"].configure(
                    state="disabled", cursor="arrow"
                )
            if draft.unchanged or not draft.selection_identity:
                self.action_buttons["Replace Selection"].configure(
                    state="disabled", cursor="arrow"
                )

    def _set_compact_controls(self, *, ready: bool) -> None:
        """Enable only actions supported by the currently bound draft."""
        draft = self.draft if ready else None
        stage = self.action_buttons.get("Stage in Review")
        replace = self.action_buttons.get("Replace Selection")
        copy = self.action_buttons.get("Copy")
        retry = self.action_buttons.get("Retry")
        if stage is not None:
            stage.configure(
                state=("normal" if draft is not None and not draft.unchanged else "disabled"),
                cursor=("hand2" if draft is not None and not draft.unchanged else "arrow"),
            )
        if replace is not None:
            enabled = bool(draft is not None and not draft.unchanged and draft.selection_identity)
            replace.configure(
                state=("normal" if enabled else "disabled"),
                cursor=("hand2" if enabled else "arrow"),
            )
        if copy is not None:
            copy.configure(
                state=("normal" if draft is not None else "disabled"),
                cursor=("hand2" if draft is not None else "arrow"),
            )
        if retry is not None:
            retry.configure(
                state=("normal" if draft is not None and not self._error else "disabled"),
                cursor=("hand2" if draft is not None and not self._error else "arrow"),
            )
        cancel = self.action_buttons.get("Cancel")
        if cancel is not None:
            cancel.configure(state="normal", cursor="hand2")

    @staticmethod
    def _set_text(widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", str(content))
        widget.configure(state="disabled")

    def set_partial_text(self, text: str) -> None:
        """Render streamed output while keeping approval controls disabled."""
        if not self.compact:
            return
        self._error = False
        self._set_text(self.result_text, str(text or ""))
        self.status_label.configure(text="Generating · live result")
        self.result_heading.configure(text="RESULT · LIVE")
        badge = getattr(self, "preview_badge", None)
        if badge is not None:
            badge.configure(text="LIVE PREVIEW", fg=self.palette.accent_soft)
        self._set_adaptive_controls(enabled=False)
        self._set_compact_controls(ready=False)
        self._set_pass_controls(ready=False)

    def set_draft(self, draft: PreviewDraft) -> None:
        """Bind a completed immutable result to an already-visible compact card."""
        self.draft = draft
        self.request_id = draft.transaction_id
        self._error = False
        self._set_text(self.result_text, draft.result)
        if self.compact:
            self.result_heading.configure(
                text="RESULT · NO CHANGE" if draft.unchanged else "RESULT · READY"
            )
            self.status_label.configure(
                text="No change needed" if draft.unchanged else "Ready · review before use"
            )
            badge = getattr(self, "preview_badge", None)
            if badge is not None:
                badge.configure(
                    text="NO CHANGE" if draft.unchanged else "READY • NOT APPLIED",
                    fg=self.palette.warning if draft.unchanged else self.palette.success,
                )
            self._set_adaptive_controls(enabled=False)
            self._set_compact_controls(ready=True)
            self._set_pass_controls(ready=True)

    def set_error(self, message: str) -> None:
        """Show a concise transformation failure without exposing provider internals."""
        self._error = True
        if self.compact:
            self.status_label.configure(text="Unavailable · nothing applied")
            self.result_heading.configure(text="RESULT · UNAVAILABLE")
            badge = getattr(self, "preview_badge", None)
            if badge is not None:
                badge.configure(text="FAILED • NOT APPLIED", fg=self.palette.danger)
            self._set_adaptive_controls(enabled=False)
            self._set_text(
                self.result_text,
                message.strip() or "The transformation could not be completed.",
            )
            self._set_compact_controls(ready=False)
            self._set_pass_controls(ready=False)

    def _label(self, parent: tk.Misc, text: str) -> None:
        tk.Label(
            parent,
            text=text,
            bg=self.palette.window,
            fg=self.palette.muted,
            font=("Segoe UI Semibold", 9),
            anchor="w",
        ).pack(fill="x")

    def _text_panel(
        self,
        parent: tk.Misc,
        title: str,
        content: str,
    ) -> tuple[tk.Frame, tk.Text, tk.Label]:
        compact = bool(self.compact)
        frame = tk.Frame(
            parent,
            bg=self.palette.surface,
            padx=10,
            pady=5 if compact else 9,
        )
        heading = tk.Label(
            frame,
            text=title,
            bg=self.palette.surface,
            fg=self.palette.muted,
            font=("Segoe UI Semibold", 9),
            anchor="w",
        )
        heading.pack(fill="x", pady=(0, 2 if compact else 5))
        body = tk.Frame(frame, bg=self.palette.surface)
        body.pack(fill="both", expand=True)
        text = tk.Text(
            body,
            wrap="word",
            bg=self.palette.field,
            fg=self.palette.text,
            insertbackground=self.palette.accent,
            selectbackground=self.palette.accent,
            relief="flat",
            padx=10,
            pady=4 if compact else 8,
        )
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.insert("1.0", content)
        text.configure(state="disabled")
        text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        if compact:
            self.result_scrollbar = scrollbar
        return frame, text, heading

    def matches_request(self, request_id: str) -> bool:
        return bool(self.request_id and self.request_id == str(request_id))

    def _scroll_compact(self, units: int) -> str:
        target = self._compact_scroll_canvas or getattr(self, "result_text", None)
        if target is not None:
            try:
                target.yview_scroll(int(units), "units")
            except tk.TclError:
                pass
        return "break"

    def _on_compact_mousewheel(self, event: tk.Event) -> str:
        units = self._compact_wheel_accumulator.consume(
            int(getattr(event, "delta", 0) or 0)
        )
        return self._scroll_compact(units)

    def _on_compact_button4(self, _event: tk.Event) -> str:
        return self._scroll_compact(
            self._compact_wheel_accumulator.consume_discrete(-1)
        )

    def _on_compact_button5(self, _event: tk.Event) -> str:
        return self._scroll_compact(
            self._compact_wheel_accumulator.consume_discrete(1)
        )

    def _fit_window_size(
        self,
        width: int,
        height: int,
        work_area: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        screen_x, screen_y, screen_right, screen_bottom = work_area
        available_width = max(520, int(screen_right - screen_x - 24))
        available_height = max(520, int(screen_bottom - screen_y - 24))
        if self.compact:
            return (
                min(max(560, int(width)), min(720, available_width)),
                min(max(420, int(height)), min(560, available_height)),
            )
        return (
            min(max(720, int(width)), available_width),
            min(max(560, int(height)), available_height),
        )

    def show(self) -> None:
        if self.anchor is not None:
            self._place_near_anchor()
        else:
            self._center_on_parent()
        apply_windows_titlebar_theme(self.window, self.palette)
        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()

    def _place_near_anchor(self, gap: int = 18) -> None:
        """Place the card beside the captured selection/caret, not the root window."""
        try:
            self.window.update_idletasks()
            width = max(560 if self.compact else 720, int(self.window.winfo_width()))
            height = max(420 if self.compact else 560, int(self.window.winfo_height()))
            anchor_x, anchor_y = self.anchor
            left = int(anchor_x) - width // 2
            top = int(anchor_y) + int(gap)
            work_area = monitor_work_area_for_point(int(anchor_x), int(anchor_y))
            if work_area is not None:
                screen_x, screen_y, screen_right, screen_bottom = work_area
                screen_width = screen_right - screen_x
                screen_height = screen_bottom - screen_y
            else:
                screen_x = int(self.window.winfo_vrootx())
                screen_y = int(self.window.winfo_vrooty())
                screen_width = int(self.window.winfo_vrootwidth())
                screen_height = int(self.window.winfo_vrootheight())
            width, height = self._fit_window_size(
                width,
                height,
                (screen_x, screen_y, screen_x + screen_width, screen_y + screen_height),
            )
            if top + height > screen_y + screen_height - 12:
                top = int(anchor_y) - height - int(gap)
            left = max(screen_x + 12, min(left, screen_x + screen_width - width - 12))
            top = max(screen_y + 12, min(top, screen_y + screen_height - height - 12))
            self.window.geometry(f"{width}x{height}+{left}+{top}")
        except (tk.TclError, TypeError, ValueError):
            # A missing/invalid anchor must never produce an arbitrary geometry.
            self._center_on_parent()

    def _center_on_parent(self) -> None:
        """Place the review card on the same monitor as Control Lab."""
        try:
            self.window.update_idletasks()
            self.parent.update_idletasks()
            width = max(560 if self.compact else 720, int(self.window.winfo_width()))
            height = max(420 if self.compact else 560, int(self.window.winfo_height()))
            parent_width = max(1, int(self.parent.winfo_width()))
            parent_height = max(1, int(self.parent.winfo_height()))
            parent_x = int(self.parent.winfo_rootx())
            parent_y = int(self.parent.winfo_rooty())
            left = parent_x + (parent_width - width) // 2
            top = parent_y + (parent_height - height) // 2
            screen_x = int(self.window.winfo_vrootx())
            screen_y = int(self.window.winfo_vrooty())
            screen_width = int(self.window.winfo_vrootwidth())
            screen_height = int(self.window.winfo_vrootheight())
            width, height = self._fit_window_size(
                width,
                height,
                (screen_x, screen_y, screen_x + screen_width, screen_y + screen_height),
            )
            left = max(screen_x + 12, min(left, screen_x + screen_width - width - 12))
            top = max(screen_y + 12, min(top, screen_y + screen_height - height - 12))
            self.window.geometry(f"{width}x{height}+{left}+{top}")
        except tk.TclError:
            pass

    def close(self) -> None:
        if self._compact_scroll_bindtag is not None:
            uninstall_scroll_bindings(self.window, self._compact_scroll_bindtag)
            self._compact_scroll_bindtag = None
        if self.window.winfo_exists():
            self.window.destroy()

    def _stage(self) -> None:
        if self.draft is None:
            return
        self.on_stage(self.draft.transaction_id, self.draft.fingerprint)
        self.close()

    def _copy(self) -> None:
        if self.draft is None:
            return
        self.on_copy(self.draft.transaction_id, self.draft.fingerprint)

    def _replace(self) -> None:
        if self.draft is None:
            return
        self.on_replace(self.draft.transaction_id, self.draft.fingerprint)
        self.close()

    def _retry(self) -> None:
        if self.draft is None:
            return
        self.on_retry(self.draft.transaction_id, self.draft.fingerprint)
        self.close()

    def _cancel(self) -> None:
        self.on_cancel(self.request_id or (self.draft.transaction_id if self.draft else ""))
        self.close()
