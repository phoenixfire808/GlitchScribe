from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import local_voice
from restart_glitchscribe import (
    build_helper_command,
    build_launch_command,
    build_taskkill_command,
    launcher_identity_is_expected,
)


class FakeRoot:
    def __init__(self) -> None:
        self.after_calls: list[tuple[int, object]] = []

    def after(self, delay: int, callback):
        self.after_calls.append((delay, callback))
        return "after-id"


class FakeButton:
    def __init__(self) -> None:
        self.configured: list[dict[str, object]] = []

    def configure(self, **kwargs):
        self.configured.append(kwargs)


def make_app() -> local_voice.LocalVoiceApp:
    app = local_voice.LocalVoiceApp.__new__(local_voice.LocalVoiceApp)
    app.root = FakeRoot()
    app.recording = False
    app.busy = False
    app._restart_requested = False
    app.restart_button = FakeButton()
    app.status_calls: list[tuple[str, str]] = []
    app.close_calls = 0
    app._set_status = lambda message, kind: app.status_calls.append((message, kind))
    app.close = lambda: setattr(app, "close_calls", app.close_calls + 1)
    return app


def test_build_launch_command_uses_project_venv_and_keepalive_launcher():
    with TemporaryDirectory() as temporary_directory:
        tmp_path = Path(temporary_directory)
        command = build_launch_command(tmp_path)

    assert command == [
        str(tmp_path / ".venv" / "Scripts" / "pythonw.exe"),
        str(tmp_path / "keepalive_launcher.py"),
    ]


def test_build_helper_command_carries_exact_launcher_and_app_pids():
    with TemporaryDirectory() as temporary_directory:
        tmp_path = Path(temporary_directory)
        command = build_helper_command(tmp_path, launcher_pid=11452, app_pid=15168)

    assert command[1] == str(tmp_path / "restart_glitchscribe.py")
    assert command[2:] == [
        "--launcher-pid",
        "11452",
        "--app-pid",
        "15168",
        "--project-dir",
        str(tmp_path),
    ]


def test_taskkill_targets_only_launcher_without_descendant_tree():
    assert build_taskkill_command(11452) == [
        "taskkill.exe",
        "/PID",
        "11452",
        "/F",
    ]


def test_launcher_identity_accepts_relative_script_when_executable_is_project():
    with TemporaryDirectory() as temporary_directory:
        project = Path(temporary_directory)
        output = (
            f"ExecutablePath : {project / '.venv' / 'Scripts' / 'pythonw.exe'}\\n"
            "CommandLine : pythonw.exe ./keepalive_launcher.py\\n"
        )

        def runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

        assert launcher_identity_is_expected(11452, project, runner=runner)


def test_restart_button_starts_detached_helper_and_schedules_graceful_close():
    app = make_app()
    with TemporaryDirectory() as temporary_directory:
        tmp_path = Path(temporary_directory)
        with (
            patch.object(local_voice, "SOURCE_DIR", tmp_path),
            patch.object(local_voice.os, "getppid", return_value=11452),
            patch.object(local_voice.os, "getpid", return_value=15168),
            patch.object(local_voice.subprocess, "Popen") as popen,
        ):
            app._restart_application()

        command = popen.call_args.args[0]
        assert command == [
            str(sys.executable),
            str(tmp_path / "restart_glitchscribe.py"),
            "--launcher-pid",
            "11452",
            "--app-pid",
            "15168",
            "--project-dir",
            str(tmp_path),
        ]
    assert popen.call_args.kwargs["stdin"] is local_voice.subprocess.DEVNULL
    assert popen.call_args.kwargs["stdout"] is local_voice.subprocess.DEVNULL
    assert popen.call_args.kwargs["stderr"] is local_voice.subprocess.DEVNULL
    assert app.root.after_calls and app.root.after_calls[0][0] == 100
    assert app.restart_button.configured[-1] == {
        "text": "RESTARTING…",
        "state": "disabled",
    }
    assert app.status_calls[-1] == ("Reloading GlitchScribe…", "working")


def test_restart_button_requires_confirmation_when_dictation_is_active():
    app = make_app()
    app.recording = True
    with (
        patch.object(local_voice.messagebox, "askyesno", return_value=False),
        patch.object(local_voice.subprocess, "Popen") as popen,
    ):
        app._restart_application()

    popen.assert_not_called()
    assert app.close_calls == 0
    assert app._restart_requested is False
