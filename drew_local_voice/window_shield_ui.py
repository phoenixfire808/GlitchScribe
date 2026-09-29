"""Tkinter UI for the Window Shield manager."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .window_shield import (
    PlatformAdapter,
    ShieldConfig,
    WindowShield,
    load_config,
    save_config,
)


class WindowShieldApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        adapter: PlatformAdapter,
        config_path: Path | None = None,
    ) -> None:
        self.root = root
        self.config_path = config_path
        self.adapter = adapter
        self.config = load_config(path=config_path, adapter=adapter)
        self.shield = WindowShield(config=self.config, adapter=adapter)
        self._event_items: dict[str, object] = {}
        self._closing = False

        root.title("Window Shield")
        root.geometry("1040x650")
        root.minsize(780, 460)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.status_var = tk.StringVar()
        self.enabled_var = tk.BooleanVar(value=not self.shield.is_paused)
        self.title_match_var = tk.BooleanVar(value=self.config.use_title_keyword_match)
        self.interval_var = tk.StringVar(value=str(self.config.poll_interval_seconds))

        self._build_ui()
        self._refresh_rules()
        self._poll()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="WINDOW SHIELD", font=("Segoe UI", 18, "bold")).pack(side="left")
        ttk.Label(header, text="Visible, reversible desktop window control", foreground="#666").pack(side="left", padx=(12, 0), pady=(7, 0))
        ttk.Checkbutton(header, text="Shield enabled", variable=self.enabled_var, command=self._toggle_enabled).pack(side="right")
        ttk.Label(outer, textvariable=self.status_var).pack(anchor="w", pady=(8, 10))

        rules = ttk.LabelFrame(outer, text="Matching rules", padding=8)
        rules.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(rules, text="Match terminal/shell words in window titles", variable=self.title_match_var, command=self._apply_rules).grid(row=0, column=0, sticky="w")
        ttk.Label(rules, text="Poll seconds:").grid(row=0, column=1, padx=(24, 4), sticky="e")
        ttk.Entry(rules, textvariable=self.interval_var, width=7).grid(row=0, column=2, sticky="w")
        ttk.Button(rules, text="Apply", command=self._apply_rules).grid(row=0, column=3, padx=(6, 0))
        ttk.Label(rules, text="Default action is hide; processes are never terminated.", foreground="#666").grid(row=1, column=0, columnspan=4, sticky="w", pady=(7, 0))

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)
        events_frame = ttk.LabelFrame(body, text="Blocked windows (newest first)", padding=8)
        rules_frame = ttk.LabelFrame(body, text="Process rules", padding=8)
        body.add(events_frame, weight=4)
        body.add(rules_frame, weight=1)

        columns = ("time", "process", "title", "pid", "hwnd", "matched")
        self.events_tree = ttk.Treeview(events_frame, columns=columns, show="headings", selectmode="browse")
        headings = {"time": "Time", "process": "Process", "title": "Window title", "pid": "PID", "hwnd": "HWND", "matched": "Rule"}
        widths = {"time": 160, "process": 125, "title": 300, "pid": 65, "hwnd": 100, "matched": 80}
        for key in columns:
            self.events_tree.heading(key, text=headings[key])
            self.events_tree.column(key, width=widths[key], anchor="w")
        yscroll = ttk.Scrollbar(events_frame, orient="vertical", command=self.events_tree.yview)
        self.events_tree.configure(yscrollcommand=yscroll.set)
        self.events_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        events_frame.rowconfigure(0, weight=1)
        events_frame.columnconfigure(0, weight=1)

        ttk.Label(rules_frame, text="Blocked executables", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.blocked_list = tk.Listbox(rules_frame, height=8, exportselection=False)
        self.blocked_list.pack(fill="both", expand=True, pady=(4, 8))
        self.blocked_entry = ttk.Entry(rules_frame)
        self.blocked_entry.pack(fill="x")
        ttk.Button(rules_frame, text="Add blocked process", command=self._add_blocked).pack(fill="x", pady=(4, 10))

        ttk.Label(rules_frame, text="Allowlisted executables", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.allowlist = tk.Listbox(rules_frame, height=6, exportselection=False)
        self.allowlist.pack(fill="both", expand=True, pady=(4, 8))
        self.allow_entry = ttk.Entry(rules_frame)
        self.allow_entry.pack(fill="x")
        ttk.Button(rules_frame, text="Add allowlisted process", command=self._add_allowlisted).pack(fill="x", pady=(4, 4))
        ttk.Button(rules_frame, text="Remove selected allowlist item", command=self._remove_allowlisted).pack(fill="x")

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(actions, text="Unblock selected", command=self._unblock_selected).pack(side="left")
        ttk.Button(actions, text="Allow selected process", command=self._allow_selected_process).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Restore all hidden", command=self._restore_all).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Clear event log", command=self._clear_log).pack(side="right")

    def _persist(self) -> None:
        save_config(self.shield.config, path=self.config_path)

    def _toggle_enabled(self) -> None:
        if self.enabled_var.get():
            self.shield.resume()
        else:
            self.shield.pause()
        self._persist()
        self._update_status()

    def _apply_rules(self) -> None:
        try:
            interval = max(0.05, min(10.0, float(self.interval_var.get())))
        except ValueError:
            messagebox.showerror("Invalid interval", "Poll seconds must be between 0.05 and 10.")
            return
        self.config = self.shield.config
        self.config.poll_interval_seconds = interval
        self.config.use_title_keyword_match = bool(self.title_match_var.get())
        self.shield.update_config(config=self.config)
        self.interval_var.set(str(interval))
        self._persist()

    def _refresh_rules(self) -> None:
        self.blocked_list.delete(0, "end")
        for item in self.shield.config.blocked_executables:
            self.blocked_list.insert("end", item)
        self.allowlist.delete(0, "end")
        for item in self.shield.config.allowlist_executables:
            self.allowlist.insert("end", item)

    def _add_blocked(self) -> None:
        value = self.blocked_entry.get().strip().casefold()
        if not value:
            return
        config = self.shield.config
        if value not in config.blocked_executables:
            config.blocked_executables.append(value)
            self.shield.update_config(config=config)
            self._persist()
        self.blocked_entry.delete(0, "end")
        self._refresh_rules()

    def _add_allowlisted(self) -> None:
        value = self.allow_entry.get().strip().casefold()
        if not value:
            return
        config = self.shield.config
        if value not in config.allowlist_executables:
            config.allowlist_executables.append(value)
            self.shield.update_config(config=config)
            self._persist()
        self.allow_entry.delete(0, "end")
        self._refresh_rules()

    def _remove_allowlisted(self) -> None:
        selection = self.allowlist.curselection()
        if not selection:
            return
        value = self.allowlist.get(selection[0])
        config = self.shield.config
        config.allowlist_executables = [item for item in config.allowlist_executables if item != value]
        self.shield.update_config(config=config)
        self._persist()
        self._refresh_rules()

    def _selected_event(self):
        selection = self.events_tree.selection()
        if not selection:
            return None
        return self._event_items.get(selection[0])

    def _unblock_selected(self) -> None:
        event = self._selected_event()
        if event is None:
            return
        self.shield.unblock_hwnd(event.hwnd)
        self._update_status()

    def _allow_selected_process(self) -> None:
        event = self._selected_event()
        if event is None:
            return
        config = self.shield.config
        if event.process_name not in config.allowlist_executables:
            config.allowlist_executables.append(event.process_name)
            self.shield.update_config(config=config)
            self._persist()
        self.shield.unblock_hwnd(event.hwnd)
        self._refresh_rules()
        self._update_status()

    def _restore_all(self) -> None:
        restored = self.shield.restore_all()
        self.status_var.set(f"Restored {restored} hidden window(s).")

    def _clear_log(self) -> None:
        self.shield.clear_log()
        self._refresh_events()
        self._update_status()

    def _refresh_events(self) -> None:
        for item in self.events_tree.get_children():
            self.events_tree.delete(item)
        self._event_items.clear()
        for index, event in enumerate(reversed(self.shield.events.snapshot())):
            iid = f"event-{index}-{event.hwnd}-{event.pid}"
            self._event_items[iid] = event
            self.events_tree.insert(
                "", "end", iid=iid,
                values=(event.timestamp, event.process_name, event.title or "(untitled)", event.pid, event.hwnd, event.matched_by),
            )

    def _update_status(self) -> None:
        self.status_var.set(f"{self.shield.status}  •  events: {len(self.shield.events)}  •  adapter: {self.adapter.name}")

    def _poll(self) -> None:
        if self._closing:
            return
        try:
            self.shield.tick()
            self._refresh_events()
            self._update_status()
        except Exception as exc:
            self.status_var.set(f"Scan error: {exc}")
        self.root.after(max(50, int(self.shield.config.poll_interval_seconds * 1000)), self._poll)

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.shield.restore_all()
        self._persist()
        self.root.destroy()


def launch(*, config_path: Path | None = None, adapter: PlatformAdapter | None = None) -> None:
    from .window_shield import build_default_adapter

    root = tk.Tk()
    WindowShieldApp(root, adapter=adapter or build_default_adapter(), config_path=config_path)
    root.mainloop()


__all__ = ["WindowShieldApp", "launch"]
