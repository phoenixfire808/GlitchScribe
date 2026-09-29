"""Explicitly provision a local CrisperWhisper 2.0 checkpoint.

This is intentionally a user-invoked setup command. GlitchScribe never calls
Hugging Face download helpers during normal startup or model selection.

Example:
    python prepare_crisperwhisper_model.py --size small
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

SIZES = ("small", "medium", "turbo", "large")
DEFAULT_ROOT = (
    Path(os.environ.get("LOCALAPPDATA", Path.home()))
    / "DrewLocalVoice"
    / "models"
    / "crisperwhisper"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    output = args.output or (DEFAULT_ROOT / f"CrisperWhisper2.0_{args.size}")
    output = Path(os.path.expandvars(os.path.expanduser(str(output))))
    output.parent.mkdir(parents=True, exist_ok=True)
    repo_id = f"nyralabs/CrisperWhisper2.0_{args.size}"

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required; install the project requirements first"
        ) from exc

    print(f"Downloading {repo_id} to {output}")
    resolved = snapshot_download(
        repo_id=repo_id,
        revision=args.revision,
        local_dir=str(output),
    )
    print(f"Ready: {resolved}")
    print("Model weights are governed by Nyra Health's separate non-commercial research license.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
