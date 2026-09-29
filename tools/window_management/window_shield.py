"""Launch the standalone Window Shield manager."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from drew_local_voice.window_shield import build_default_adapter, load_config


def self_test() -> int:
    """Read-only smoke test: enumerate visible windows without hiding anything."""
    adapter = build_default_adapter()
    windows = adapter.enum_visible_windows()
    print(json.dumps({
        "adapter": adapter.name,
        "visible_top_level_windows": len(windows),
        "processes": sorted({window.process_name for window in windows}),
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safely hide and manage unwanted Windows")
    parser.add_argument("--self-test", action="store_true", help="enumerate visible windows without changing them")
    parser.add_argument("--config", type=Path, help="use an alternate JSON configuration path")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if os.name != "nt":
        print("Window Shield requires Windows for real window control.")
        return 2
    from drew_local_voice.window_shield_ui import launch

    # Load once here so a malformed configuration fails closed before the UI is
    # created; the UI loads the same path again for its editable state.
    load_config(path=args.config, adapter=build_default_adapter())
    launch(config_path=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
