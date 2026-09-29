from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from dictation_features import (
    parse_custom_replacements,
    remove_custom_replacement,
    upsert_custom_replacement,
)

from .preview_model import (
    PREVIEW_SAMPLE_TEXTS,
    PREVIEW_STATE_LABELS,
    BubblePreviewModel,
    preview_setting_keys,
)
from .preview_widget import PreviewWidget
from .scrolling import (
    WheelDeltaAccumulator,
    install_scroll_bindings,
    uninstall_scroll_bindings,
)
from .settings import (
    EffectRecipe,
    SettingKind,
    SettingSpec,
    SettingsRegistry,
    built_in_effect_recipes,
)
from .theme import (
    ThemePalette,
    apply_ttk_theme,
    apply_windows_titlebar_theme,
    build_palette,
)


SECTION_ORDER = (
    "General",
    "Interface",
    "Speech Bubble",
    "Corrections",
    "Performance Lab",
    "Recognition",
    "Local AI",
    "Streaming",
    "Typing",
    "Endpointing",
    "Finalization",
    "Audio",
    "Commands",
    "Privacy",
    "Plugins",
    "Compatibility",
)

SETTINGS_TAB_ORDER = (
    "General",
    "Recognition",
    "Streaming",
    "Dictation",
    "Commands",
    "Appearance",
    "System",
)

_SETTINGS_TAB_BY_SECTION = {
    "General": "General",
    "Performance Lab": "General",
    "Recognition": "Recognition",
    "Local AI": "Recognition",
    "Streaming": "Streaming",
    "Audio": "Streaming",
    "Typing": "Dictation",
    "Endpointing": "Dictation",
    "Finalization": "Dictation",
    "Commands": "Commands",
    "Corrections": "Commands",
    "Interface": "Appearance",
    "Speech Bubble": "Appearance",
    "Privacy": "System",
    "Plugins": "System",
    "Compatibility": "System",
}


def settings_tab_for_section(section: str) -> str:
    return _SETTINGS_TAB_BY_SECTION.get(section, "System")


def wheel_scroll_units(delta: int) -> int:
    """Translate Windows wheel/trackpad deltas into responsive vertical units."""
    if not delta:
        return 0
    direction = -1 if delta > 0 else 1
    return direction * max(1, round(abs(delta) / 120))


def snap_slider_value(
    value: float,
    minimum: float,
    maximum: float,
    step: float,
) -> float:
    """Clamp and snap a slider value to its canonical setting increment."""
    bounded = min(max(float(value), float(minimum)), float(maximum))
    increment = max(float(step), 1e-9)
    snapped = float(minimum) + round((bounded - float(minimum)) / increment) * increment
    return round(min(max(snapped, float(minimum)), float(maximum)), 10)


def _format_slider_value(value: float, step: float) -> str:
    if float(step).is_integer():
        return str(int(round(value)))
    decimals = max(1, len(f"{float(step):.10f}".rstrip("0").split(".")[1]))
    return f"{value:.{decimals}f}"


def slider_step_for_spec(spec: SettingSpec) -> float | None:
    """Return a stable slider increment for every bounded numeric setting."""
    if (
        spec.kind is not SettingKind.NUMBER
        or spec.minimum is None
        or spec.maximum is None
        or spec.maximum <= spec.minimum
    ):
        return None
    if spec.step is not None:
        return float(spec.step)

    def decimal_places(value: float) -> int:
        text = f"{float(value):.10f}".rstrip("0").rstrip(".")
        return len(text.rsplit(".", 1)[1]) if "." in text else 0

    places = max(
        decimal_places(float(spec.default)),
        decimal_places(float(spec.minimum)),
        decimal_places(float(spec.maximum)),
    )
    step = 10.0 ** -places if places else 1.0
    span = float(spec.maximum) - float(spec.minimum)
    while span / step > 10_000:
        step *= 10.0
    return min(step, span)


class SchemaSettingsDialog:
    """Scrollable, schema-generated settings editor.

    Extensions add SettingSpec objects to SettingsRegistry; this dialog renders
    them automatically without plugin-specific UI code.
    """

    def __init__(
        self,
        parent: tk.Misc,
        registry: SettingsRegistry,
        values: dict[str, Any],
        on_save: Callable[[dict[str, Any]], None],
        *,
        title: str = "Glitch Factor — GlitchScribe Control Lab",
        palette: ThemePalette | None = None,
        palette_provider: Callable[[], ThemePalette] | None = None,
        effect_recipes: tuple[EffectRecipe, ...] | None = None,
        embedded: bool = False,
        excluded_keys: tuple[str, ...] = (),
        values_provider: Callable[[], dict[str, Any]] | None = None,
        on_stage_changed: Callable[[dict[str, Any] | None], None] | None = None,
    ) -> None:
        self.parent = parent
        self.registry = registry
        self.values = dict(values)
        self.on_save = on_save
        self.palette = palette or build_palette("dark", "violet")
        self.palette_provider = palette_provider
        self.effect_recipes = effect_recipes or built_in_effect_recipes()
        self.embedded = bool(embedded)
        self.excluded_keys = frozenset(excluded_keys)
        self.values_provider = values_provider
        self.on_stage_changed = on_stage_changed
        self.recipes_by_title = {
            recipe.title: recipe for recipe in self.effect_recipes
        }
        self.variables: dict[str, tk.Variable] = {}
        self.canvases: dict[str, tk.Canvas] = {}
        self._wheel_delta_accumulator = WheelDeltaAccumulator()
        self._scroll_bindtag: str | None = None
        # GlitchScribe Control Lab: discoverable attributes for the live preview
        # layout. ``preview_model`` and ``preview_widget`` are populated when the
        # dialog builds a non-embedded two-pane body. Embedded dialogs keep the
        # historical single-pane form so existing consumers (tray, plugin
        # settings panels) are unaffected.
        self.preview_model: BubblePreviewModel | None = None
        self.preview_widget: PreviewWidget | None = None
        self.preview_pane: ttk.Frame | None = None
        self.settings_pane: ttk.Frame | None = None
        self.control_lab_body: ttk.Frame | None = None
        # Cached set of preview-only keys derived from the canonical registry;
        # used to route every schema-generated control through a single
        # ``on_preview_value_changed`` path without per-widget special cases.
        self._preview_keys: frozenset[str] = frozenset(
            preview_setting_keys(self.registry)
        )
        self._stage_keys: frozenset[str] = frozenset(
            {
                "recognition.model",
                "recognition.device",
                "recognition.compute_type",
            }
        )
        # Right-pane preview controls (sample selector, custom sample text,
        # state selector). Populated only on the non-embedded dialog.
        self.sample_var: tk.StringVar | None = None
        self.custom_sample_text: tk.Text | None = None
        self.state_var: tk.StringVar | None = None
        # Suppresses ``on_preview_value_changed`` recursion while the dialog
        # itself writes into ``self.variables`` (e.g. revert/save/reset).
        self._suppress_preview_traces = False
        self.window = parent if self.embedded else tk.Toplevel(parent)
        if not self.embedded:
            self.window.title(title)
            self.window.geometry("1180x720")
            self.window.minsize(980, 560)
            self.window.configure(bg=self.palette.window)
            self.window.transient(parent)
            self.window.protocol("WM_DELETE_WINDOW", self.close)
        apply_ttk_theme(self.window.winfo_toplevel(), self.palette)
        self._build()
        apply_windows_titlebar_theme(self.window.winfo_toplevel(), self.palette)

    def _build(self) -> None:
        heading = ttk.Frame(
            self.window,
            style="SettingsHeader.TFrame",
            padding=(20, 16, 20, 12),
        )
        heading.pack(fill="x")
        ttk.Label(
            heading,
            text="GLITCH FACTOR  //  GLITCHSCRIBE",
            style="SettingGuidance.TLabel",
            font=("Segoe UI Semibold", 8),
        ).pack(anchor="w")
        ttk.Label(
            heading,
            text="CONTROL LAB",
            style="Title.TLabel",
            font=("Segoe UI Semibold", 17),
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(
            heading,
            text=(
                f"{len(self.registry.to_schema())} local options  •  "
                "Mouse wheel, trackpad, scrollbar, or Page Up / Page Down"
            ),
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        if self.embedded:
            heading.pack_forget()

        recipe_panel = ttk.Frame(
            self.window,
            style="SettingCard.TFrame",
            padding=(14, 10),
        )
        recipe_panel.pack(fill="x", padx=18, pady=(0, 10))
        recipe_panel.columnconfigure(1, weight=1)
        ttk.Label(
            recipe_panel,
            text="EFFECT RECIPE",
            style="SettingTitle.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.recipe_var = tk.StringVar(value=self.effect_recipes[0].title)
        self.recipe_box = ttk.Combobox(
            recipe_panel,
            textvariable=self.recipe_var,
            values=tuple(recipe.title for recipe in self.effect_recipes),
            state="readonly",
            width=30,
        )
        self.recipe_box.grid(row=0, column=1, sticky="ew")
        self.recipe_box.bind("<<ComboboxSelected>>", self._update_recipe_outcome)
        ttk.Button(
            recipe_panel,
            text="STAGE RECIPE",
            style="Primary.TButton",
            command=self.stage_selected_recipe,
        ).grid(row=0, column=2, sticky="e", padx=(10, 0))
        self.recipe_outcome_var = tk.StringVar(
            value=f"TARGET EFFECT  •  {self.effect_recipes[0].outcome}"
        )
        ttk.Label(
            recipe_panel,
            textvariable=self.recipe_outcome_var,
            wraplength=800,
            style="SettingGuidance.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        specs_by_section: dict[str, list[SettingSpec]] = {}
        for row in self.registry.to_schema():
            spec = self.registry.get(row["key"])
            if spec.key in self.excluded_keys:
                continue
            specs_by_section.setdefault(spec.section, []).append(spec)

        section_rank = {section: index for index, section in enumerate(SECTION_ORDER)}
        ordered_sections = sorted(
            specs_by_section.items(),
            key=lambda item: (section_rank.get(item[0], len(section_rank)), item[0]),
        )
        self.notebook = None
        self.canvas = None
        forms: dict[str, tk.Frame] = {}
        if self.embedded:
            form = ttk.Frame(
                self.window,
                style="SettingsPage.TFrame",
                padding=(18, 4, 18, 12),
            )
            form.pack(fill="x", expand=True)
            forms["All"] = form
        else:
            # GlitchScribe Control Lab: persistent two-pane body. The
            # left pane owns the existing tabbed settings notebook and
            # its scrollable pages; the right pane hosts the embedded
            # ``PreviewWidget`` that mirrors staged Speech Bubble values.
            body = ttk.Frame(self.window, style="SettingsPage.TFrame")
            body.pack(fill="both", expand=True, padx=18, pady=(0, 12))
            self.control_lab_body = body
            body.columnconfigure(0, weight=1)
            body.columnconfigure(1, weight=0)
            body.rowconfigure(0, weight=1)

            settings_pane = ttk.Frame(body, style="SettingsPage.TFrame")
            settings_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
            self.settings_pane = settings_pane

            preview_pane = ttk.Frame(
                body,
                style="SettingCard.TFrame",
                padding=(14, 12),
            )
            preview_pane.grid(row=0, column=1, sticky="nsew")
            preview_pane.columnconfigure(0, weight=0)
            self.preview_pane = preview_pane

            ttk.Label(
                preview_pane,
                text="LIVE BUBBLE PREVIEW",
                style="SettingTitle.TLabel",
            ).grid(row=0, column=0, sticky="w", pady=(0, 8))
            ttk.Label(
                preview_pane,
                text=(
                    "Mirrors staged Speech Bubble settings. Save Settings to "
                    "publish them to the live overlay."
                ),
                style="SettingGuidance.TLabel",
                wraplength=320,
                justify="left",
            ).grid(row=1, column=0, sticky="ew", pady=(0, 10))

            notebook = ttk.Notebook(settings_pane)
            notebook.pack(fill="both", expand=True)
            self.notebook = notebook
            available_sections = {section for section, _specs in ordered_sections}
            for tab_name in SETTINGS_TAB_ORDER:
                if not any(
                    settings_tab_for_section(section) == tab_name
                    for section in available_sections
                ):
                    continue
                page = ttk.Frame(notebook, style="SettingsPage.TFrame")
                notebook.add(page, text=tab_name.upper())
                canvas = tk.Canvas(
                    page,
                    bg=self.palette.surface,
                    highlightthickness=0,
                    borderwidth=0,
                    takefocus=True,
                )
                scrollbar = ttk.Scrollbar(
                    page,
                    orient="vertical",
                    command=canvas.yview,
                )
                canvas.pack(side="left", fill="both", expand=True)
                scrollbar.pack(side="right", fill="y")
                form = ttk.Frame(
                    canvas,
                    style="SettingsPage.TFrame",
                    padding=(12, 10),
                )
                form.bind(
                    "<Configure>",
                    lambda _event, current=canvas: current.configure(
                        scrollregion=current.bbox("all")
                    ),
                )
                window_id = canvas.create_window((0, 0), window=form, anchor="nw")
                canvas.bind(
                    "<Configure>",
                    lambda event, current=canvas, item=window_id: current.itemconfigure(
                        item, width=event.width
                    ),
                )
                canvas.configure(yscrollcommand=scrollbar.set)
                forms[tab_name] = form
                self.canvases[str(page)] = canvas

            # Embedded PreviewWidget on the right. The model is derived from
            # the canonical settings values + registry so the staged Speech
            # Bubble keys feed the spec deterministically; the generated
            # controls below route staged values through the same model.
            preview_model = BubblePreviewModel.from_values(
                self.values,
                registry=self.registry,
            )
            self.preview_model = preview_model
            self.preview_widget = PreviewWidget(
                preview_pane,
                registry=self.registry,
            )
            self.preview_widget.grid(
                row=2,
                column=0,
                sticky="n",
                pady=(4, 0),
            )
            self.preview_widget.refresh(preview_model)
            # Right-pane sample / state / revert controls. They live next to
            # the embedded preview so Drew can test short, long,
            # punctuation, rapid, and custom synthetic phrases plus the
            # preview state label without touching the microphone or
            # production overlay.
            self._build_preview_controls(preview_pane)

        next_rows: dict[str, int] = {}
        for section, specs in ordered_sections:
            tab_name = "All" if self.embedded else settings_tab_for_section(section)
            form = forms[tab_name]
            next_row = next_rows.get(tab_name, 0)
            section_frame = ttk.Frame(
                form,
                style="SettingsPage.TFrame",
                padding=(0, 4, 0, 12),
            )
            section_frame.grid(row=next_row, column=0, sticky="ew", pady=(0, 8))
            form.columnconfigure(0, weight=1)
            section_frame.columnconfigure(0, weight=1)
            ttk.Label(
                section_frame,
                text=section.upper(),
                style="Title.TLabel",
                font=("Segoe UI Semibold", 13),
            ).grid(row=0, column=0, sticky="w", padx=(4, 0), pady=(0, 5))
            for index, spec in enumerate(specs):
                self._add_setting(section_frame, spec, index + 1)
            next_rows[tab_name] = next_row + 1

        self.window.bind("<MouseWheel>", self._on_mousewheel)
        self.window.bind("<Button-4>", lambda event: self._on_button_scroll(event, -1))
        self.window.bind("<Button-5>", lambda event: self._on_button_scroll(event, 1))
        self.window.bind("<Prior>", lambda event: self._scroll_pages(event, -1))
        self.window.bind("<Next>", lambda event: self._scroll_pages(event, 1))
        self.window.bind("<Control-Home>", lambda event: self._scroll_edge(event, 0.0))
        self.window.bind("<Control-End>", lambda event: self._scroll_edge(event, 1.0))

        buttons = ttk.Frame(
            self.window,
            style="SettingsHeader.TFrame",
            padding=(18, 0, 18, 16),
        )
        buttons.pack(fill="x")
        ttk.Button(
            buttons,
            text="RESET STAGED DEFAULTS",
            style="Secondary.TButton",
            command=self.reset_defaults,
        ).pack(side="left")
        ttk.Button(
            buttons,
            text="RESET & SAVE DEFAULTS",
            style="Secondary.TButton",
            command=lambda: self.reset_defaults(persist=True),
        ).pack(side="left", padx=(8, 0))
        if not self.embedded:
            ttk.Button(
                buttons,
                text="CLOSE",
                style="Secondary.TButton",
                command=self.close,
            ).pack(side="right", padx=(8, 0))
        ttk.Button(
            buttons,
            text="SAVE SETTINGS",
            style="Primary.TButton",
            command=self.save,
        ).pack(side="right")
        if not self.embedded:
            self._scroll_bindtag = install_scroll_bindings(
                self.window,
                self._on_mousewheel,
                button4_handler=lambda event: self._on_button_scroll(event, -1),
                button5_handler=lambda event: self._on_button_scroll(event, 1),
            )

    def _current_canvas(self) -> tk.Canvas | None:
        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            return canvas
        notebook = getattr(self, "notebook", None)
        if notebook is not None:
            return self.canvases.get(notebook.select())
        return None

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        accumulator = getattr(self, "_wheel_delta_accumulator", None)
        if accumulator is None:
            accumulator = WheelDeltaAccumulator()
            self._wheel_delta_accumulator = accumulator
        canvas = self._current_canvas()
        if canvas is None:
            accumulator.reset()
            return None
        units = accumulator.consume(
            int(getattr(event, "delta", 0))
        )
        if not units:
            # Consume the event so a child ttk control cannot reinterpret a
            # partial precision-touchpad delta as a selection change.
            return "break"
        canvas.yview_scroll(units, "units")
        return "break"

    def _on_button_scroll(self, _event: tk.Event, units: int) -> str | None:
        canvas = self._current_canvas()
        if canvas is None:
            self._wheel_delta_accumulator.reset()
            return None
        canvas.yview_scroll(
            self._wheel_delta_accumulator.consume_discrete(units),
            "units",
        )
        return "break"

    def _scroll_pages(self, _event: tk.Event, pages: int) -> str | None:
        canvas = self._current_canvas()
        if canvas is None:
            return None
        canvas.yview_scroll(pages, "pages")
        return "break"

    def _scroll_edge(self, _event: tk.Event, fraction: float) -> str | None:
        canvas = self._current_canvas()
        if canvas is None:
            return None
        canvas.yview_moveto(fraction)
        return "break"

    def _add_setting(self, parent: ttk.Frame, spec: SettingSpec, row: int) -> None:
        container = ttk.Frame(
            parent,
            style="SettingCard.TFrame",
            padding=(14, 11),
        )
        container.grid(row=row, column=0, sticky="ew", pady=4)
        parent.columnconfigure(0, weight=1)
        container.columnconfigure(0, weight=3, uniform="setting_columns")
        container.columnconfigure(1, weight=2, uniform="setting_columns")
        value = self.values.get(spec.key, spec.default)
        slider_step = slider_step_for_spec(spec)
        label = spec.label + ("  •  ADVANCED" if spec.advanced else "")
        ttk.Label(container, text=label, style="SettingTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        if spec.description:
            ttk.Label(
                container,
                text=f"WHAT IT DOES  •  {spec.description}",
                wraplength=420,
                style="SettingDescription.TLabel",
            ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        guidance = spec.guidance
        if spec.kind is SettingKind.NUMBER and (
            spec.minimum is not None or spec.maximum is not None
        ):
            guidance = f"{guidance}  Range: {spec.minimum:g}–{spec.maximum:g}."
        ttk.Label(
            container,
            text=f"SUGGESTED USE  •  {guidance}",
            wraplength=440,
            style="SettingGuidance.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(3, 0))

        if spec.kind is SettingKind.BOOLEAN:
            variable: tk.Variable = tk.BooleanVar(value=bool(value))
            control = ttk.Checkbutton(
                container,
                variable=variable,
                style="Setting.TCheckbutton",
            )
        elif spec.kind is SettingKind.CHOICE:
            variable = tk.StringVar(value=str(value))
            control = ttk.Combobox(
                container,
                textvariable=variable,
                values=[str(choice) for choice in spec.choices],
                state="readonly",
                width=30,
            )
        elif spec.presentation == "correction_editor":
            variable = tk.StringVar(value=str(value))
            control = self._build_correction_editor(container, variable)
        elif (
            spec.kind is SettingKind.NUMBER
            and slider_step is not None
        ):
            snapped = snap_slider_value(
                float(value),
                spec.minimum,
                spec.maximum,
                slider_step,
            )
            variable = tk.DoubleVar(value=snapped)
            control = ttk.Frame(container, style="SettingCard.TFrame")
            control.columnconfigure(0, weight=1)
            scale = ttk.Scale(
                control,
                from_=spec.minimum,
                to=spec.maximum,
                variable=variable,
                orient="horizontal",
                length=240,
                command=lambda raw, current=variable, current_spec=spec: current.set(
                    snap_slider_value(
                        float(raw),
                        float(current_spec.minimum),
                        float(current_spec.maximum),
                        float(slider_step),
                    )
                ),
            )
            scale.grid(row=0, column=0, sticky="ew")
            displayed = tk.StringVar(
                value=_format_slider_value(snapped, slider_step)
            )
            ttk.Label(
                control,
                textvariable=displayed,
                style="SettingTitle.TLabel",
                width=7,
                anchor="e",
            ).grid(row=0, column=1, sticky="e", padx=(10, 0))

            def update_display(
                *_args,
                current=variable,
                current_display=displayed,
                current_spec=spec,
            ) -> None:
                try:
                    current_display.set(
                        _format_slider_value(
                            float(current.get()),
                            float(slider_step),
                        )
                    )
                except (tk.TclError, TypeError, ValueError):
                    return

            variable.trace_add("write", update_display)
        else:
            if spec.kind is SettingKind.STRING_LIST and not isinstance(value, str):
                value = ", ".join(str(item) for item in value)
            variable = tk.StringVar(value=str(value))
            control = ttk.Entry(container, textvariable=variable, width=38)
        if spec.presentation == "correction_editor":
            control.grid(
                row=3,
                column=0,
                columnspan=2,
                sticky="ew",
                pady=(12, 0),
            )
        else:
            control.grid(row=0, column=1, rowspan=3, sticky="ew", padx=(18, 0))
        self.variables[spec.key] = variable
        # GlitchScribe Control Lab: every schema-generated control routes
        # preview-relevant changes through a single generic path. Non-preview
        # keys are ignored by ``on_preview_value_changed`` so this trace is
        # cheap and uniform across booleans, choices, sliders, text entries,
        # and the corrections editor.
        if self._is_preview_key(spec.key):
            variable.trace_add(
                "write",
                lambda *_args, current_key=spec.key, current_var=variable: self._trace_preview_variable(
                    current_key, current_var
                ),
            )
        if spec.key in self._stage_keys:
            variable.trace_add(
                "write",
                lambda *_args, current_key=spec.key, current_var=variable: self._trace_staged_variable(
                    current_key, current_var
                ),
            )

    # ------------------------------------------------------------------
    # GlitchScribe Control Lab: live preview wiring
    # ------------------------------------------------------------------

    def _is_preview_key(self, key: str) -> bool:
        return key in self._preview_keys

    def _trace_preview_variable(self, key: str, variable: tk.Variable) -> None:
        """Safe bridge from a ``trace_add("write", ...)`` callback to the preview router.

        ``tk.DoubleVar.get()`` (and other Tk variable ``get`` methods) raise
        ``tk.TclError`` when the underlying value is malformed mid-edit — for
        example, a half-typed numeric entry whose text cannot be parsed as a
        float. Reading the value inside the trace lambda would let that error
        escape the dialog's fail-closed contract. This helper swallows only
        the expected malformed-value exceptions and routes a readable value to
        :meth:`on_preview_value_changed`; anything else propagates so genuine
        bugs stay visible.
        """
        try:
            value = variable.get()
        except (tk.TclError, TypeError, ValueError):
            # Malformed intermediate input: leave the staged preview model and
            # the widget untouched. The next valid keystroke will fire a fresh
            # trace write.
            return
        self.on_preview_value_changed(key, value)

    def _notify_stage_changed(self) -> None:
        """Publish only valid staged runtime values to the owning app."""
        callback = getattr(self, "on_stage_changed", None)
        if callback is None:
            return
        staged: dict[str, Any] = {}
        for key in self._stage_keys:
            variable = self.variables.get(key)
            if variable is None:
                continue
            try:
                staged[key] = self.registry.validate(key, variable.get())
            except (KeyError, TypeError, ValueError, tk.TclError):
                continue
        callback(staged)

    def _trace_staged_variable(
        self,
        key: str,
        variable: tk.Variable,
    ) -> None:
        if self._suppress_preview_traces:
            return
        try:
            variable.get()
        except (tk.TclError, TypeError, ValueError):
            return
        self._notify_stage_changed()

    def _refresh_preview_widget(self) -> None:
        """Push the current staged model into the embedded ``PreviewWidget``."""
        widget = getattr(self, "preview_widget", None)
        model = getattr(self, "preview_model", None)
        if widget is None or model is None:
            return
        try:
            widget.refresh(model)
        except (AttributeError, TypeError, ValueError, tk.TclError):
            # The widget must never crash the settings dialog. A failing
            # refresh leaves the previous frame in place; the next staged
            # value will retry on the next trace write.
            return

    def on_preview_value_changed(self, key: str, value: Any) -> None:
        """Single generic routing path for every schema-generated control.

        Preview-only keys are validated and staged into the existing
        :class:`BubblePreviewModel`; the widget is refreshed immediately. Keys
        that are not preview-relevant are ignored: their canonical value flows
        exclusively through ``save()``. Invalid intermediate input (e.g. a
        half-typed number) never mutates the model or the preview.
        """
        if self._suppress_preview_traces:
            return
        if not self._is_preview_key(key):
            return
        model = self.preview_model
        if model is None:
            return
        spec = self.registry.get(key)
        try:
            validated = self.registry.validate(key, value)
        except (KeyError, TypeError, ValueError):
            try:
                validated = self.registry.validate(key, spec.default)
            except (KeyError, TypeError, ValueError):
                return
        try:
            model.stage_setting(key, validated)
        except (KeyError, TypeError, ValueError):
            return
        self._refresh_preview_widget()
        # Palette keys drive the dialog's own palette as well.
        if key in ("ui.theme", "ui.accent_color") and self.palette_provider is not None:
            self._apply_palette_silently(self.palette_provider())

    def _apply_palette_silently(self, palette: ThemePalette) -> None:
        """Refresh theme tokens without rewriting control variables.

        ``apply_palette`` must never re-enter the preview trace; we only need
        the TTK theme/canvas colors and an immediate preview refresh.
        """
        prior = self._suppress_preview_traces
        self._suppress_preview_traces = True
        try:
            self.apply_palette(palette)
        finally:
            self._suppress_preview_traces = prior

    def _build_preview_controls(self, parent: ttk.Frame) -> None:
        """Create the right-pane sample/state/revert controls."""
        sample_keys = list(PREVIEW_SAMPLE_TEXTS.keys())
        self.sample_var = tk.StringVar(
            value=self.preview_model.sample_key if self.preview_model else sample_keys[0]
        )
        self.state_var = tk.StringVar(
            value=self.preview_model.state_label
            if self.preview_model
            else PREVIEW_STATE_LABELS[0]
        )

        controls = ttk.Frame(parent, style="SettingCard.TFrame", padding=(0, 12, 0, 0))
        controls.grid(row=3, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(
            controls,
            text="PREVIEW SAMPLE",
            style="SettingTitle.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        sample_box = ttk.Combobox(
            controls,
            textvariable=self.sample_var,
            values=sample_keys,
            state="readonly",
            width=14,
        )
        sample_box.grid(row=0, column=1, sticky="ew")
        sample_box.bind("<<ComboboxSelected>>", self._on_sample_changed)

        ttk.Label(
            controls,
            text="CUSTOM SAMPLE",
            style="SettingTitle.TLabel",
        ).grid(row=1, column=0, sticky="nw", pady=(8, 0))
        custom_frame = ttk.Frame(controls, style="SettingCard.TFrame")
        custom_frame.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        custom_frame.columnconfigure(0, weight=1)
        self.custom_sample_text = tk.Text(
            custom_frame,
            height=3,
            width=28,
            wrap="word",
            font=("Segoe UI", 9),
            highlightthickness=0,
            borderwidth=0,
        )
        self.custom_sample_text.grid(row=0, column=0, sticky="ew")
        if self.preview_model is not None and self.preview_model.sample_key == "custom":
            self.custom_sample_text.insert("1.0", self.preview_model.sample_text)
        # Synchronise the model without writing back into the variable.
        self.custom_sample_text.bind(
            "<KeyRelease>",
            lambda _event: self._on_custom_sample_text_changed(),
        )
        self._custom_sample_status_var = tk.StringVar(
            value=(
                f"Bounded to {BubblePreviewModel.MAX_SAMPLE_CHARACTERS} characters; "
                "fail-closed."
            )
        )
        ttk.Label(
            controls,
            textvariable=self._custom_sample_status_var,
            style="SettingGuidance.TLabel",
            wraplength=280,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        ttk.Label(
            controls,
            text="PREVIEW STATE",
            style="SettingTitle.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=(10, 4))
        state_box = ttk.Combobox(
            controls,
            textvariable=self.state_var,
            values=list(PREVIEW_STATE_LABELS),
            state="readonly",
            width=18,
        )
        state_box.grid(row=3, column=1, sticky="ew")
        state_box.bind("<<ComboboxSelected>>", self._on_state_changed)

        actions = ttk.Frame(controls, style="SettingCard.TFrame")
        actions.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        ttk.Button(
            actions,
            text="REVERT STAGED",
            style="Secondary.TButton",
            command=self.revert_staged,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(
            actions,
            text="RESET PREVIEW",
            style="Secondary.TButton",
            command=self.reset_preview,
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def _on_sample_changed(self, _event: tk.Event | None = None) -> None:
        model = self.preview_model
        sample_var = self.sample_var
        custom_text = self.custom_sample_text
        if model is None or sample_var is None:
            return
        previous = self._suppress_preview_traces
        self._suppress_preview_traces = True
        try:
            new_text = model.select_sample(sample_var.get())
            if custom_text is not None and model.sample_key != "custom":
                custom_text.delete("1.0", "end")
            elif custom_text is not None and model.sample_key == "custom":
                custom_text.delete("1.0", "end")
                custom_text.insert("1.0", new_text)
        finally:
            self._suppress_preview_traces = previous
        self._refresh_preview_widget()

    def _on_state_changed(self, _event: tk.Event | None = None) -> None:
        model = self.preview_model
        state_var = self.state_var
        if model is None or state_var is None:
            return
        model.set_state_label(state_var.get())
        self._refresh_preview_widget()

    def _on_custom_sample_text_changed(self) -> None:
        text_widget = self.custom_sample_text
        model = self.preview_model
        sample_var = self.sample_var
        if text_widget is None or model is None or sample_var is None:
            return
        raw = text_widget.get("1.0", "end-1c")
        previous = self._suppress_preview_traces
        self._suppress_preview_traces = True
        try:
            new_text = model.set_sample_text(raw)
            sample_var.set(model.sample_key)
            # Truncate the visible buffer to the bounded length without
            # re-entering the KeyRelease handler.
            if len(raw) > model.MAX_SAMPLE_CHARACTERS:
                text_widget.delete("1.0", "end")
                text_widget.insert("1.0", new_text)
        finally:
            self._suppress_preview_traces = previous
        self._refresh_preview_widget()

    def _baseline_values(self) -> dict[str, Any]:
        """Return the last-saved canonical values for revert/reset."""
        if self.values_provider is not None:
            try:
                return dict(self.values_provider())
            except (KeyError, TypeError, ValueError):
                pass
        return dict(self.values)

    def revert_staged(self) -> None:
        """Restore the last-saved values into controls and the preview.

        This never invokes ``on_save``: the production overlay and the
        persisted configuration are untouched.
        """
        self._restore_baseline_into_controls()
        model = self.preview_model
        if model is not None:
            model.revert()
        # Re-sync the right-pane sample/state widgets with the restored model.
        if model is not None and self.sample_var is not None:
            self.sample_var.set(model.sample_key)
        if model is not None and self.state_var is not None:
            self.state_var.set(model.state_label)
        if (
            model is not None
            and self.custom_sample_text is not None
            and model.sample_key == "custom"
        ):
            self.custom_sample_text.delete("1.0", "end")
            self.custom_sample_text.insert("1.0", model.sample_text)
        elif self.custom_sample_text is not None:
            self.custom_sample_text.delete("1.0", "end")
        self._refresh_preview_widget()
        self._notify_stage_changed()

    def reset_preview(self) -> None:
        """``RESET PREVIEW`` is an alias for ``revert_staged``.

        Both actions restore the saved baseline; neither mutates persisted
        configuration or invokes ``on_save``.
        """
        self.revert_staged()

    def _restore_baseline_into_controls(self) -> None:
        """Reset every generated control to the last-saved baseline."""
        baseline = self._baseline_values()
        previous = self._suppress_preview_traces
        self._suppress_preview_traces = True
        try:
            for key, variable in self.variables.items():
                if key not in baseline:
                    continue
                spec = self.registry.get(key)
                value = baseline[key]
                if spec.kind is SettingKind.STRING_LIST and not isinstance(value, str):
                    value = ", ".join(str(item) for item in value)
                try:
                    variable.set(value)
                except (tk.TclError, ValueError):
                    continue
        finally:
            self._suppress_preview_traces = previous


    def _build_correction_editor(
        self,
        parent: ttk.Frame,
        mapping_variable: tk.StringVar,
    ) -> ttk.Frame:
        editor = ttk.Frame(parent, style="SettingCard.TFrame")
        editor.columnconfigure(0, weight=1)
        editor.columnconfigure(1, weight=1)
        self.correction_heard_var = tk.StringVar(value="")
        self.correction_intended_var = tk.StringVar(value="")
        self.correction_status_var = tk.StringVar(
            value="Corrections remain local. Add or update one, then choose Save Settings."
        )

        ttk.Label(
            editor,
            text="GLITCHSCRIBE HEARD",
            style="SettingTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            editor,
            text="REPLACE IT WITH",
            style="SettingTitle.TLabel",
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))
        heard_entry = ttk.Entry(
            editor,
            textvariable=self.correction_heard_var,
            width=34,
        )
        heard_entry.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        intended_entry = ttk.Entry(
            editor,
            textvariable=self.correction_intended_var,
            width=34,
        )
        intended_entry.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(12, 0),
            pady=(4, 0),
        )
        add_button = ttk.Button(
            editor,
            text="ADD / UPDATE CORRECTION",
            style="Primary.TButton",
            command=self.add_or_update_correction,
        )
        add_button.grid(row=1, column=2, sticky="e", padx=(12, 0), pady=(4, 0))
        heard_entry.bind("<Return>", lambda _event: self.add_or_update_correction())
        intended_entry.bind("<Return>", lambda _event: self.add_or_update_correction())

        table_frame = ttk.Frame(editor, style="SettingCard.TFrame")
        table_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.correction_tree = ttk.Treeview(
            table_frame,
            columns=("heard", "intended"),
            show="headings",
            height=7,
            selectmode="browse",
        )
        self.correction_tree.heading("heard", text="GlitchScribe heard")
        self.correction_tree.heading("intended", text="Write instead")
        self.correction_tree.column("heard", width=300, minwidth=120, stretch=True)
        self.correction_tree.column("intended", width=330, minwidth=120, stretch=True)
        self.correction_tree.grid(row=0, column=0, sticky="nsew")
        correction_scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.correction_tree.yview,
        )
        correction_scrollbar.grid(row=0, column=1, sticky="ns")
        self.correction_tree.configure(yscrollcommand=correction_scrollbar.set)
        self.correction_tree.bind("<<TreeviewSelect>>", self._load_selected_correction)

        actions = ttk.Frame(editor, style="SettingCard.TFrame")
        actions.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(
            actions,
            text="REMOVE SELECTED",
            style="Secondary.TButton",
            command=self.remove_selected_correction,
        ).pack(side="left")
        ttk.Button(
            actions,
            text="CLEAR FIELDS",
            style="Secondary.TButton",
            command=self.clear_correction_fields,
        ).pack(side="left", padx=(8, 0))
        ttk.Label(
            actions,
            textvariable=self.correction_status_var,
            style="SettingGuidance.TLabel",
            wraplength=510,
        ).pack(side="right", fill="x", expand=True, padx=(12, 0))

        self.correction_mapping_var = mapping_variable
        mapping_variable.trace_add(
            "write",
            lambda *_args: self._refresh_correction_tree(),
        )
        self._refresh_correction_tree()
        return editor

    def _refresh_correction_tree(self) -> None:
        tree = getattr(self, "correction_tree", None)
        mapping = getattr(self, "correction_mapping_var", None)
        if tree is None or mapping is None:
            return
        for item in tree.get_children():
            tree.delete(item)
        for heard, intended in parse_custom_replacements(mapping.get()).items():
            tree.insert("", "end", values=(heard, intended))

    def _promote_correction_vocabulary(self, intended: str) -> None:
        variable = self.variables.get("recognition.custom_vocabulary")
        if variable is None:
            return
        items = [item.strip() for item in str(variable.get()).split(",") if item.strip()]
        if intended.casefold() not in {item.casefold() for item in items}:
            items.append(intended)
            variable.set(", ".join(items))

    def add_or_update_correction(self) -> None:
        mapping = self.variables.get("commands.custom_replacements")
        if mapping is None:
            mapping = getattr(self, "correction_mapping_var", None)
        if mapping is None:
            return
        heard = str(self.correction_heard_var.get())
        intended = str(self.correction_intended_var.get())
        try:
            mapping.set(upsert_custom_replacement(mapping.get(), heard, intended))
        except ValueError as exc:
            self.correction_status_var.set(str(exc))
            return
        self._promote_correction_vocabulary(intended.strip())
        self.correction_heard_var.set("")
        self.correction_intended_var.set("")
        self._refresh_correction_tree()
        self.correction_status_var.set(
            "Correction staged locally. Choose Save Settings to use it from now on."
        )

    def _load_selected_correction(self, _event: tk.Event | None = None) -> None:
        selection = self.correction_tree.selection()
        if not selection:
            return
        values = self.correction_tree.item(selection[0], "values")
        if len(values) >= 2:
            self.correction_heard_var.set(values[0])
            self.correction_intended_var.set(values[1])
            self.correction_status_var.set(
                "Selected correction loaded. Edit either field, then choose Add / Update."
            )

    def remove_selected_correction(self) -> None:
        selection = self.correction_tree.selection()
        if not selection:
            self.correction_status_var.set("Select a correction to remove.")
            return
        values = self.correction_tree.item(selection[0], "values")
        if not values:
            return
        mapping = self.variables.get("commands.custom_replacements")
        if mapping is None:
            mapping = self.correction_mapping_var
        mapping.set(remove_custom_replacement(mapping.get(), str(values[0])))
        self.clear_correction_fields()
        self._refresh_correction_tree()
        self.correction_status_var.set(
            "Correction removed from the staged list. Choose Save Settings to confirm."
        )

    def clear_correction_fields(self) -> None:
        self.correction_heard_var.set("")
        self.correction_intended_var.set("")

    def _update_recipe_outcome(self, _event: tk.Event | None = None) -> None:
        recipe = self.recipes_by_title.get(self.recipe_var.get())
        if recipe is not None:
            self.recipe_outcome_var.set(f"TARGET EFFECT  •  {recipe.outcome}")

    def stage_selected_recipe(self) -> None:
        recipe = self.recipes_by_title.get(self.recipe_var.get())
        if recipe is None:
            return
        previous = getattr(self, "_suppress_preview_traces", False)
        self._suppress_preview_traces = True
        try:
            for key, value in recipe.changes:
                variable = self.variables.get(key)
                if variable is not None:
                    try:
                        variable.set(value)
                    except (tk.TclError, ValueError):
                        continue
        finally:
            self._suppress_preview_traces = previous
        # Push the staged recipe values into the preview model in one pass
        # so the embedded widget reflects the full effect of the recipe.
        preview_model = getattr(self, "preview_model", None)
        preview_keys = getattr(self, "_preview_keys", frozenset())
        registry = getattr(self, "registry", None)
        if preview_model is not None and registry is not None:
            for key, value in recipe.changes:
                if key in preview_keys:
                    try:
                        preview_model.stage_setting(
                            key, registry.validate(key, value)
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
        self._refresh_preview_widget()
        self.recipe_outcome_var.set(
            f"STAGED  •  {recipe.outcome}  Review the staged values, then choose Save Settings."
        )

    def _raw_value(self, spec: SettingSpec) -> Any:
        value = self.variables[spec.key].get()
        if spec.kind is SettingKind.NUMBER:
            text = str(value).strip()
            if not text:
                raise ValueError(f"{spec.label} cannot be empty")
            return float(text)
        return value

    def save(self) -> None:
        updated: dict[str, Any] = (
            dict(self.values_provider())
            if self.values_provider is not None
            else dict(self.values)
        )
        try:
            for key in self.variables:
                spec = self.registry.get(key)
                updated[spec.key] = self.registry.validate(
                    spec.key,
                    self._raw_value(spec),
                )
        except (KeyError, TypeError, ValueError) as exc:
            messagebox.showerror("Invalid setting", str(exc), parent=self.window)
            return
        try:
            self.on_save(updated)
        except (KeyError, TypeError, ValueError) as exc:
            messagebox.showerror("Settings not saved", str(exc), parent=self.window)
            return
        # Persisted successfully. Refresh controls without re-entering the
        # preview trace, then update the dialog baseline and accept the
        # staged preview as the new baseline.
        previous = self._suppress_preview_traces
        self._suppress_preview_traces = True
        try:
            for key, value in updated.items():
                if key not in self.variables:
                    continue
                spec = self.registry.get(key)
                if spec.kind is SettingKind.STRING_LIST and not isinstance(value, str):
                    value = ", ".join(str(item) for item in value)
                self.variables[key].set(value)
        finally:
            self._suppress_preview_traces = previous
        self.values = dict(updated)
        if self.preview_model is not None:
            self.preview_model.accept_staged()
        self._notify_stage_changed()
        if self.palette_provider is not None:
            self._apply_palette_silently(self.palette_provider())
        else:
            self._refresh_preview_widget()
        messagebox.showinfo("Settings saved", "Settings were saved locally.", parent=self.window)

    def reset_defaults(self, *, persist: bool = False) -> None:
        title = "Reset settings" if persist else "Reset staged settings"
        message = (
            "Reset every setting to its default value and save immediately?"
            if persist
            else "Reset every staged setting to its default value?"
        )
        if not messagebox.askyesno(
            title,
            message,
            parent=self.window,
        ):
            return
        previous = self._suppress_preview_traces
        self._suppress_preview_traces = True
        try:
            for key in self.variables:
                spec = self.registry.get(key)
                value = spec.default
                if spec.kind is SettingKind.STRING_LIST and not isinstance(value, str):
                    value = ", ".join(str(item) for item in value)
                self.variables[spec.key].set(value)
        finally:
            self._suppress_preview_traces = previous
        if persist:
            # Reuse the normal validation, persistence, runtime-apply, and
            # preview-baseline path.  This keeps reset behavior identical to a
            # user editing every field and choosing Save Settings.
            self.save()
            return
        # Reflect the new default values in the live preview; do not call
        # ``on_save`` or touch the persisted configuration.
        if self.preview_model is not None:
            for spec in self.registry.to_schema():
                if spec["key"] in self._preview_keys and spec["key"] in self.variables:
                    try:
                        self.preview_model.stage_setting(
                            spec["key"],
                            self.registry.validate(
                                spec["key"], self.variables[spec["key"]].get()
                            ),
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
        self._refresh_preview_widget()
        self._notify_stage_changed()

    def apply_palette(self, palette: ThemePalette) -> None:
        self.palette = palette
        top_level = self.window.winfo_toplevel()
        apply_ttk_theme(top_level, palette)
        apply_windows_titlebar_theme(top_level, palette)
        if not self.embedded:
            self.window.configure(bg=palette.window)
        for canvas in self.canvases.values():
            canvas.configure(bg=palette.surface)
        # GlitchScribe Control Lab: the palette also drives the embedded
        # preview so a theme/accent change is visible immediately.
        self._refresh_preview_widget()

    def show(self) -> None:
        if self.embedded:
            return
        if self.palette_provider is not None:
            self.apply_palette(self.palette_provider())
        else:
            apply_windows_titlebar_theme(self.window, self.palette)
        self.window.deiconify()
        self.window.after_idle(
            lambda: apply_windows_titlebar_theme(self.window, self.palette)
        )
        self.window.lift()
        self.window.focus_force()

    def close(self) -> None:
        if not self.embedded:
            callback = getattr(self, "on_stage_changed", None)
            if callback is not None:
                callback(None)
            uninstall_scroll_bindings(self.window, self._scroll_bindtag)
            self._scroll_bindtag = None
            self.window.destroy()
