"""Tool registry for multi-agent system. Populated by agents.py at import time."""

from __future__ import annotations

from typing import Callable

# Global registry: tool_name -> callable(text, context) -> str
TOOL_REGISTRY: dict[str, Callable[[str, str], str]] = {}
