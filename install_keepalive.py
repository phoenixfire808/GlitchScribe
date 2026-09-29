from __future__ import annotations

import getpass
import json
from datetime import datetime, timedelta
from pathlib import Path

import win32com.client
import win32api

TASK_NAME = "Drew Local Voice Always Running"
PROJECT_DIR = Path(__file__).resolve().parent
LAUNCHER = PROJECT_DIR / "keepalive_launcher.py"
APP_LAUNCHER = PROJECT_DIR / "launch_glitchscribe.py"
PYTHONW = PROJECT_DIR / ".venv" / "Scripts" / "pythonw.exe"
ICON = PROJECT_DIR / "assets" / "drew-local-voice-tray.ico"

TASK_TRIGGER_DAILY = 2
TASK_TRIGGER_LOGON = 9
TASK_ACTION_EXEC = 0
TASK_CREATE_OR_UPDATE = 6
TASK_LOGON_INTERACTIVE_TOKEN = 3
TASK_INSTANCES_IGNORE_NEW = 2


def install_shortcuts() -> list[str]:
    if not APP_LAUNCHER.is_file():
        raise FileNotFoundError(f"GlitchScribe launcher not found: {APP_LAUNCHER}")
    if not ICON.is_file():
        raise FileNotFoundError(f"GlitchScribe icon not found: {ICON}")
    shell = win32com.client.Dispatch("WScript.Shell")
    locations = [
        Path(shell.SpecialFolders("Desktop")) / "GlitchScribe.lnk",
        Path(shell.SpecialFolders("Programs")) / "GlitchScribe.lnk",
    ]
    for path in locations:
        shortcut = shell.CreateShortcut(str(path))
        shortcut.TargetPath = str(PYTHONW)
        shortcut.Arguments = f'"{APP_LAUNCHER}"'
        shortcut.WorkingDirectory = str(PROJECT_DIR)
        shortcut.IconLocation = f"{ICON},0"
        shortcut.Description = "Open GlitchScribe local speech-to-text"
        shortcut.Save()
    return [str(path) for path in locations]


def install() -> dict[str, object]:
    if not PYTHONW.exists():
        raise FileNotFoundError(f"Stable Python runtime not found: {PYTHONW}")
    if not LAUNCHER.exists():
        raise FileNotFoundError(f"Keepalive launcher not found: {LAUNCHER}")

    service = win32com.client.Dispatch("Schedule.Service")
    service.Connect()
    root = service.GetFolder("\\")
    definition = service.NewTask(0)
    definition.RegistrationInfo.Description = (
        "Keeps GlitchScribe running invisibly, starts it at sign-in, and checks every minute."
    )
    definition.RegistrationInfo.Author = getpass.getuser()

    principal = definition.Principal
    principal.UserId = win32api.GetUserNameEx(2)
    principal.LogonType = TASK_LOGON_INTERACTIVE_TOKEN

    settings = definition.Settings
    settings.Enabled = True
    settings.Hidden = True
    settings.StartWhenAvailable = True
    settings.DisallowStartIfOnBatteries = False
    settings.StopIfGoingOnBatteries = False
    settings.AllowDemandStart = True
    settings.AllowHardTerminate = True
    settings.ExecutionTimeLimit = "PT0S"
    settings.MultipleInstances = TASK_INSTANCES_IGNORE_NEW

    logon = definition.Triggers.Create(TASK_TRIGGER_LOGON)
    logon.Enabled = True
    logon.UserId = principal.UserId

    daily = definition.Triggers.Create(TASK_TRIGGER_DAILY)
    daily.Enabled = True
    daily.StartBoundary = (datetime.now() + timedelta(seconds=15)).strftime("%Y-%m-%dT%H:%M:%S")
    daily.DaysInterval = 1
    daily.Repetition.Interval = "PT1M"
    daily.Repetition.Duration = "P1D"
    daily.Repetition.StopAtDurationEnd = False

    action = definition.Actions.Create(TASK_ACTION_EXEC)
    action.Path = str(PYTHONW)
    action.Arguments = f'"{LAUNCHER}"'
    action.WorkingDirectory = str(PROJECT_DIR)

    registered = root.RegisterTaskDefinition(
        TASK_NAME,
        definition,
        TASK_CREATE_OR_UPDATE,
        None,
        None,
        TASK_LOGON_INTERACTIVE_TOKEN,
    )
    registered.Run("")
    shortcuts = install_shortcuts()
    return {
        "task": TASK_NAME,
        "pythonw": str(PYTHONW),
        "launcher": str(LAUNCHER),
        "enabled": bool(registered.Enabled),
        "shortcuts": shortcuts,
    }


if __name__ == "__main__":
    print(json.dumps(install(), indent=2))
