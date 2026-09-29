"""Compatibility entry point for the unified Window Shield tool."""

from tools.window_management.window_shield import *  # noqa: F401,F403
from tools.window_management.window_shield import main


if __name__ == "__main__":
    raise SystemExit(main())
