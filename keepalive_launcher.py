from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MUTEX_NAME = r"Local\DrewLocalVoice.Instance.v1"
ERROR_ALREADY_EXISTS = 183
HEALTH_LOG = Path(os.environ.get("LOCALAPPDATA", PROJECT_DIR)) / "DrewLocalVoice" / "keepalive.jsonl"


def _write_health(event: str, **fields: object) -> None:
    HEALTH_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.time(), "event": event, "pid": os.getpid(), **fields}
    with HEALTH_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _acquire_instance_mutex() -> tuple[int, bool]:
    if os.name != "nt":
        raise RuntimeError("Drew Local Voice keepalive requires Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    ctypes.set_last_error(0)
    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(handle), ctypes.get_last_error() == ERROR_ALREADY_EXISTS


def main() -> int:
    handle, already_running = _acquire_instance_mutex()
    kernel32 = ctypes.windll.kernel32
    if already_running:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        return 0
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, str(PROJECT_DIR))
    _write_health("app_start")
    try:
        from local_voice import main as run_local_voice

        run_local_voice()
        _write_health("app_exit")
        return 0
    except BaseException as exc:
        _write_health("app_crash", error_type=type(exc).__name__, error=str(exc)[:500])
        return 1
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


if __name__ == "__main__":
    raise SystemExit(main())
