"""Compatibility import for the unified draft-only call-response app.

The implementation lives in ``apps.call_response``. This root shim preserves
existing imports and the historical ``python call_response.py`` entry point.
"""

from apps.call_response.call_response import *  # noqa: F401,F403
from apps.call_response.call_response import main


if __name__ == "__main__":
    main()
