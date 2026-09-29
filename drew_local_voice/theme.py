from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os
import tkinter as tk
from tkinter import ttk
from typing import Optional


THEME_CHOICES = ("system", "dark", "light")
ACCENT_CHOICES = ("violet", "cyan", "blue", "emerald", "rose")

DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWA_TEXT_COLOR = 36


@dataclass(frozen=True)
class ThemePalette:
    mode: str
    window: str
    surface: str
    surface_alt: str
    field: str
    text: str
    muted: str
    border: str
    accent: str
    accent_hover: str
    accent_soft: str
    on_accent: str
    success: str
    warning: str
    danger: str
    meter_trough: str
    bubble: str


def system_uses_light_theme() -> bool:
    if os.name != "nt":
        return False
    try:
        import winreg

        path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
            value, _kind = winreg.QueryValueEx(key, "AppsUseLightTheme")
        return bool(value)
    except (OSError, ValueError):
        return False


def build_palette(
    theme: str,
    accent_color: str,
    *,
    system_uses_light: Optional[bool] = None,
) -> ThemePalette:
    requested_theme = str(theme).strip().casefold()
    if requested_theme not in THEME_CHOICES:
        requested_theme = "dark"
    if requested_theme == "system":
        use_light = (
            system_uses_light_theme()
            if system_uses_light is None
            else bool(system_uses_light)
        )
        mode = "light" if use_light else "dark"
    else:
        mode = requested_theme

    accent_name = str(accent_color).strip().casefold()
    if accent_name not in ACCENT_CHOICES:
        accent_name = "violet"
    dark_accents = {
        "violet": ("#8067F8", "#927BFF", "#B8A9FF"),
        "cyan": ("#16B8CF", "#2BC9DF", "#78E3F1"),
        "blue": ("#4F8DF7", "#6A9FFF", "#9FC1FF"),
        "emerald": ("#20B981", "#35CA92", "#82E2BC"),
        "rose": ("#F05270", "#FA6B86", "#FFA3B4"),
    }
    light_accents = {
        "violet": ("#6D4DE8", "#5C3DD2", "#684BC4"),
        "cyan": ("#087F94", "#066C7E", "#08758A"),
        "blue": ("#2563D9", "#1D4FC0", "#245CC4"),
        "emerald": ("#087B59", "#066747", "#087052"),
        "rose": ("#D62F52", "#BC2343", "#C72B4B"),
    }
    accent, accent_hover, accent_soft = (
        light_accents[accent_name] if mode == "light" else dark_accents[accent_name]
    )

    if mode == "light":
        return ThemePalette(
            mode=mode,
            window="#F2F5FA",
            surface="#FFFFFF",
            surface_alt="#E9EEF6",
            field="#F8FAFD",
            text="#172033",
            muted="#657086",
            border="#D4DCE9",
            accent=accent,
            accent_hover=accent_hover,
            accent_soft=accent_soft,
            on_accent="#FFFFFF",
            success="#087B59",
            warning="#9A6500",
            danger="#C62D4D",
            meter_trough="#DCE3EE",
            bubble="#FFFFFF",
        )
    return ThemePalette(
        mode=mode,
        window="#090B10",
        surface="#11141C",
        surface_alt="#191E29",
        field="#0D1017",
        text="#F3F5FA",
        muted="#99A3B5",
        border="#2B3242",
        accent=accent,
        accent_hover=accent_hover,
        accent_soft=accent_soft,
        on_accent="#FFFFFF",
        success="#36D49A",
        warning="#F6C760",
        danger="#FA718C",
        meter_trough="#242B39",
        bubble="#151A24",
    )


def _windows_colorref(hex_color: str) -> int:
    value = str(hex_color).strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit color, got {hex_color!r}")
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return red | (green << 8) | (blue << 16)


def apply_windows_titlebar_theme(
    window: tk.Misc,
    palette: ThemePalette,
    *,
    user32=None,
    dwmapi=None,
) -> bool:
    """Match a decorated Tk window's native Windows caption to its palette."""
    if os.name != "nt" and (user32 is None or dwmapi is None):
        return False
    try:
        window.update_idletasks()
        if user32 is None:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.GetParent.argtypes = [wintypes.HWND]
            user32.GetParent.restype = wintypes.HWND
        if dwmapi is None:
            dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
            dwmapi.DwmSetWindowAttribute.argtypes = [
                wintypes.HWND,
                wintypes.DWORD,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            dwmapi.DwmSetWindowAttribute.restype = ctypes.c_long

        tk_handle = int(window.winfo_id())
        parent_handle = int(user32.GetParent(tk_handle) or 0)
        hwnd = parent_handle or tk_handle

        def set_attribute(attribute: int, value: ctypes._SimpleCData) -> bool:
            try:
                result = dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    attribute,
                    ctypes.byref(value),
                    ctypes.sizeof(value),
                )
                return int(result) == 0
            except (AttributeError, OSError, TypeError, ValueError):
                return False

        dark_value = ctypes.c_int(1 if palette.mode == "dark" else 0)
        immersive_applied = set_attribute(DWMWA_USE_IMMERSIVE_DARK_MODE, dark_value)
        if not immersive_applied:
            immersive_applied = set_attribute(
                DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1,
                dark_value,
            )

        color_applied = False
        for attribute, color in (
            (DWMWA_CAPTION_COLOR, palette.surface),
            (DWMWA_TEXT_COLOR, palette.text),
            (DWMWA_BORDER_COLOR, palette.border),
        ):
            color_applied = set_attribute(
                attribute,
                ctypes.c_uint32(_windows_colorref(color)),
            ) or color_applied
        return immersive_applied or color_applied
    except (AttributeError, OSError, tk.TclError, TypeError, ValueError):
        return False


def apply_ttk_theme(root: tk.Misc, palette: ThemePalette) -> ttk.Style:
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(bg=palette.window)
    root.option_add("*TCombobox*Listbox.background", palette.field)
    root.option_add("*TCombobox*Listbox.foreground", palette.text)
    root.option_add("*TCombobox*Listbox.selectBackground", palette.accent)
    root.option_add("*TCombobox*Listbox.selectForeground", palette.on_accent)
    style.configure(
        ".",
        background=palette.window,
        foreground=palette.text,
        font=("Segoe UI", 10),
    )
    style.configure("TFrame", background=palette.window)
    style.configure("TLabel", background=palette.window, foreground=palette.text)
    style.configure(
        "TButton",
        background=palette.surface_alt,
        foreground=palette.text,
        bordercolor=palette.border,
        lightcolor=palette.surface_alt,
        darkcolor=palette.surface_alt,
        padding=(12, 8),
        font=("Segoe UI Semibold", 9),
    )
    style.map(
        "TButton",
        background=[("active", palette.border), ("pressed", palette.accent)],
        foreground=[("pressed", palette.on_accent)],
    )
    style.configure(
        "TCheckbutton",
        background=palette.surface,
        foreground=palette.text,
        indicatorbackground=palette.field,
        indicatorforeground=palette.accent,
        padding=(2, 3),
    )
    style.map(
        "TCheckbutton",
        background=[("active", palette.surface)],
        indicatorbackground=[("selected", palette.accent)],
    )
    style.configure(
        "TEntry",
        fieldbackground=palette.field,
        foreground=palette.text,
        bordercolor=palette.border,
        insertcolor=palette.text,
        padding=8,
    )
    style.configure(
        "TCombobox",
        fieldbackground=palette.field,
        background=palette.surface_alt,
        foreground=palette.text,
        arrowcolor=palette.accent,
        bordercolor=palette.border,
        padding=7,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", palette.field)],
        foreground=[("readonly", palette.text)],
        selectbackground=[("readonly", palette.field)],
        selectforeground=[("readonly", palette.text)],
    )
    style.configure(
        "TNotebook",
        background=palette.window,
        bordercolor=palette.border,
        tabmargins=(2, 6, 2, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background=palette.surface_alt,
        foreground=palette.muted,
        padding=(14, 9),
        font=("Segoe UI Semibold", 9),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", palette.surface), ("active", palette.border)],
        foreground=[("selected", palette.accent), ("active", palette.text)],
    )
    style.configure(
        "Vertical.TScrollbar",
        background=palette.surface_alt,
        troughcolor=palette.window,
        bordercolor=palette.window,
        arrowcolor=palette.muted,
        width=14,
    )

    style.configure("App.TFrame", background=palette.window)
    style.configure("Panel.TFrame", background=palette.surface)
    style.configure("SettingsHeader.TFrame", background=palette.window)
    style.configure("SettingsPage.TFrame", background=palette.surface)
    style.configure("SettingCard.TFrame", background=palette.surface_alt)
    style.configure(
        "Title.TLabel",
        background=palette.window,
        foreground=palette.text,
        font=("Segoe UI Semibold", 22),
    )
    style.configure(
        "Sub.TLabel",
        background=palette.window,
        foreground=palette.muted,
        font=("Segoe UI", 10),
    )
    style.configure(
        "Panel.TLabel",
        background=palette.surface,
        foreground=palette.text,
        font=("Segoe UI Semibold", 10),
    )
    style.configure(
        "SettingTitle.TLabel",
        background=palette.surface_alt,
        foreground=palette.text,
        font=("Segoe UI Semibold", 10),
    )
    style.configure(
        "SettingDescription.TLabel",
        background=palette.surface_alt,
        foreground=palette.muted,
        font=("Segoe UI", 9),
    )
    style.configure(
        "SettingGuidance.TLabel",
        background=palette.surface_alt,
        foreground=palette.accent_soft,
        font=("Segoe UI", 9),
    )
    style.configure(
        "Setting.TCheckbutton",
        background=palette.surface_alt,
        foreground=palette.text,
    )
    style.map("Setting.TCheckbutton", background=[("active", palette.surface_alt)])
    style.configure(
        "Primary.TButton",
        font=("Segoe UI Semibold", 11),
        padding=(18, 11),
        background=palette.accent,
        foreground=palette.on_accent,
        bordercolor=palette.accent_hover,
    )
    style.map(
        "Primary.TButton",
        background=[("active", palette.accent_hover), ("disabled", palette.border)],
    )
    style.configure(
        "Secondary.TButton",
        font=("Segoe UI Semibold", 9),
        padding=(11, 7),
        background=palette.surface_alt,
        foreground=palette.text,
        bordercolor=palette.border,
    )
    style.map("Secondary.TButton", background=[("active", palette.border)])
    style.configure(
        "Ready.Horizontal.TProgressbar",
        troughcolor=palette.meter_trough,
        background=palette.accent,
        lightcolor=palette.accent,
        darkcolor=palette.accent,
    )
    return style
