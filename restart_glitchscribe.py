"""Safe in-app restart handoff for GlitchScribe on Windows.

The GUI starts this module detached. It waits for the current app to finish its
normal shutdown, stops only the exact launcher PID, and starts the canonical
project launcher. Keeping the helper independent of ``local_voice`` prevents
model/UI imports during the restart window.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
APP_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "DrewLocalVoice"
HEALTH_LOG = APP_DIR / "keepalive.jsonl"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
CREATE_BREAKAWAY_FROM_JOB = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)


def _as_project_dir(project_dir: str | Path | None) -> Path:
    return Path(project_dir or PROJECT_DIR).expanduser().resolve()


def build_launch_command(project_dir: str | Path | None = None) -> list[str]:
    """Build the canonical detached GlitchScribe launch command."""
    project = _as_project_dir(project_dir)
    return [
        str(project / ".venv" / "Scripts" / "pythonw.exe"),
        str(project / "keepalive_launcher.py"),
    ]


def build_helper_command(
    project_dir: str | Path | None,
    *,
    launcher_pid: int,
    app_pid: int,
) -> list[str]:
    """Build the command used by the GUI to start this detached helper."""
    project = _as_project_dir(project_dir)
    return [
        sys.executable,
        str(project / "restart_glitchscribe.py"),
        "--launcher-pid",
        str(int(launcher_pid)),
        "--app-pid",
        str(int(app_pid)),
        "--project-dir",
        str(project),
    ]


def build_taskkill_command(pid: int) -> list[str]:
    """Target exactly one launcher process; do not recurse into its tree."""
    return ["taskkill.exe", "/PID", str(int(pid)), "/F"]


def is_process_running(
    pid: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bool:
    """Return whether Windows currently reports the supplied PID."""
    if int(pid) <= 0:
        return False
    result = runner(
        ["tasklist.exe", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        creationflags=CREATE_NO_WINDOW,
        check=False,
    )
    return result.returncode == 0 and f'"{int(pid)}"' in str(result.stdout or "")


def launcher_identity_is_expected(
    pid: int,
    project_dir: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bool:
    """Verify a PID is this project's launcher, including relative args."""
    project = _as_project_dir(project_dir)
    result = runner(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                f'(Get-CimInstance Win32_Process -Filter "ProcessId = {int(pid)}" '
                "| Select-Object ExecutablePath,CommandLine | Format-List | Out-String)"
            ),
        ],
        capture_output=True,
        text=True,
        creationflags=CREATE_NO_WINDOW,
        timeout=10,
        check=False,
    )
    identity = str(result.stdout or "").casefold()
    expected_project = str(project).casefold().replace("/", "\\")
    return (
        result.returncode == 0
        and "keepalive_launcher.py" in identity
        and expected_project in identity
    )


def wait_for_process_exit(
    pid: int,
    *,
    timeout_seconds: float,
    is_running: Callable[[int], bool] = is_process_running,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """Wait boundedly for a PID to exit without touching unrelated processes."""
    deadline = monotonic() + max(0.0, float(timeout_seconds))
    while is_running(int(pid)):
        if monotonic() >= deadline:
            return False
        sleep(min(0.1, max(0.0, deadline - monotonic())))
    return True


def _write_health(event: str, **fields: object) -> None:
    try:
        HEALTH_LOG.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.time(),
            "event": event,
            "pid": os.getpid(),
            "role": "restart_helper",
            **fields,
        }
        with HEALTH_LOG.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except OSError:
        # Restart must not fail merely because diagnostics cannot be written.
        pass


def _detached_flags() -> int:
    return DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB


def helper_creation_flags() -> int:
    """Return the detached Windows flags used for the helper process."""
    return _detached_flags()


def _launch_new_instance(project_dir: Path) -> subprocess.Popen:
    command = build_launch_command(project_dir)
    kwargs = {
        "cwd": str(project_dir),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "creationflags": _detached_flags(),
    }
    try:
        return subprocess.Popen(command, **kwargs)
    except OSError:
        # Some Windows task hosts do not permit breakaway from their job.
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        return subprocess.Popen(command, **kwargs)


def _stop_exact_launcher(
    launcher_pid: int,
    *,
    project_dir: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> subprocess.CompletedProcess | None:
    if not is_process_running(launcher_pid, runner=runner):
        return None
    if not launcher_identity_is_expected(
        launcher_pid,
        project_dir,
        runner=runner,
    ):
        _write_health(
            "restart_launcher_not_verified",
            launcher_pid=int(launcher_pid),
        )
        return None
    return runner(
        build_taskkill_command(launcher_pid),
        capture_output=True,
        text=True,
        creationflags=CREATE_NO_WINDOW,
        timeout=15,
        check=False,
    )


def restart_from_helper(
    *,
    launcher_pid: int,
    app_pid: int,
    project_dir: str | Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    is_running: Callable[[int], bool] = is_process_running,
    launch: Callable[[Path], subprocess.Popen] = _launch_new_instance,
) -> int:
    """Perform the bounded restart sequence and return a process-style code."""
    project = _as_project_dir(project_dir)
    _write_health(
        "restart_helper_started",
        launcher_pid=int(launcher_pid),
        app_pid=int(app_pid),
        project_dir=str(project),
    )

    if not wait_for_process_exit(
        app_pid,
        timeout_seconds=90.0,
        is_running=is_running,
    ):
        _write_health("restart_app_exit_timeout", app_pid=int(app_pid))
        return 2
    _write_health("restart_app_exited", app_pid=int(app_pid))

    try:
        result = _stop_exact_launcher(
            launcher_pid,
            project_dir=project,
            runner=runner,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _write_health(
            "restart_launcher_stop_failed",
            launcher_pid=int(launcher_pid),
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        return 3

    if is_running(launcher_pid):
        _write_health(
            "restart_launcher_stop_timeout",
            launcher_pid=int(launcher_pid),
            returncode=(None if result is None else result.returncode),
        )
        return 4
    _write_health(
        "restart_launcher_stopped",
        launcher_pid=int(launcher_pid),
        returncode=(None if result is None else result.returncode),
    )

    try:
        process = launch(project)
    except (OSError, subprocess.SubprocessError) as exc:
        _write_health(
            "restart_launch_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        return 5
    _write_health("restart_launch_started", new_pid=int(getattr(process, "pid", 0) or 0))
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restart GlitchScribe safely")
    parser.add_argument("--launcher-pid", required=True, type=int)
    parser.add_argument("--app-pid", required=True, type=int)
    parser.add_argument("--project-dir", default=str(PROJECT_DIR))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return restart_from_helper(
        launcher_pid=args.launcher_pid,
        app_pid=args.app_pid,
        project_dir=args.project_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
