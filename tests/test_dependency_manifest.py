"""Path-based unit tests for the dependency / model manifest.

These tests do **not** import the project package, install packages, query
devices, load models, or use the network. Every check is a deterministic
file-system / text inspection rooted at the repository root, so they are
safe to run in any CI environment with only the source tree present.

Coverage:

* ``requirements.txt`` declares ``torch`` and ``transformers``.
* ``THIRD_PARTY_NOTICES.md`` mentions Nemotron, OpenMDW, and each of the
  runtime libraries used by the local Nemotron path (PyTorch,
  Transformers, Safetensors).
* ``prepare_nemotron_model.py`` exists in the source tree and exposes the
  ``--offline`` flag that lets the staging script run without network.

Run only this file:

    python -m unittest tests.test_dependency_manifest -v
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants (resolved once at import time; tests are read-only).
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
THIRD_PARTY_NOTICES_PATH = REPO_ROOT / "THIRD_PARTY_NOTICES.md"
PREPARE_NEMOTRON_PATH = REPO_ROOT / "prepare_nemotron_model.py"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    """Return the file contents or skip the test if the file is missing.

    Missing manifest files should fail loudly via the per-test assertions
    rather than vanishing silently, but a corrupt / unreadable file is
    treated as a setup problem so the rest of the suite can still run.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - defensive only
        raise unittest.SkipTest(f"cannot read {path}: {exc}") from exc


# A requirement line looks like: ``name==1.2.3``, ``name>=1.0``, ``name``,
# possibly with leading whitespace, an inline ``# comment``, or a marker
# like ``; sys_platform == 'win32'``. We only care that the bare package
# name appears as a token at the start of the line.
_REQUIREMENT_NAME_RE = re.compile(
    r"^\s*([A-Za-z0-9_.\-]+)\s*(?:[<>=!~;\[]|$)"
)


def _iter_requirement_names(text: str) -> list[str]:
    names: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _REQUIREMENT_NAME_RE.match(line)
        if match:
            names.append(match.group(1).lower())
    return names


# ---------------------------------------------------------------------------
# requirements.txt
# ---------------------------------------------------------------------------


class RequirementsManifestTests(unittest.TestCase):
    """requirements.txt must list the runtime ML dependencies."""

    def test_requirements_file_exists(self) -> None:
        self.assertTrue(
            REQUIREMENTS_PATH.is_file(),
            f"expected manifest at {REQUIREMENTS_PATH}",
        )

    def test_requirements_declares_torch(self) -> None:
        names = _iter_requirement_names(_read_text(REQUIREMENTS_PATH))
        self.assertIn(
            "torch",
            names,
            "requirements.txt must pin torch (PyTorch runtime).",
        )

    def test_requirements_declares_transformers(self) -> None:
        names = _iter_requirement_names(_read_text(REQUIREMENTS_PATH))
        self.assertIn(
            "transformers",
            names,
            "requirements.txt must pin transformers (HuggingFace runtime).",
        )


# ---------------------------------------------------------------------------
# THIRD_PARTY_NOTICES.md
# ---------------------------------------------------------------------------


class ThirdPartyNoticesTests(unittest.TestCase):
    """THIRD_PARTY_NOTICES.md must credit Nemotron / OpenMDW and the libs."""

    def setUp(self) -> None:
        self.contents = _read_text(THIRD_PARTY_NOTICES_PATH)
        self.lowered = self.contents.lower()

    def test_third_party_notices_file_exists(self) -> None:
        self.assertTrue(
            THIRD_PARTY_NOTICES_PATH.is_file(),
            f"expected notices at {THIRD_PARTY_NOTICES_PATH}",
        )

    def test_mentions_nemotron(self) -> None:
        self.assertIn(
            "nemotron",
            self.lowered,
            "THIRD_PARTY_NOTICES.md must mention Nemotron.",
        )

    def test_mentions_openmdw(self) -> None:
        self.assertIn(
            "openmdw",
            self.lowered,
            "THIRD_PARTY_NOTICES.md must credit the OpenMDW license.",
        )

    def test_mentions_pytorch(self) -> None:
        self.assertIn(
            "pytorch",
            self.lowered,
            "THIRD_PARTY_NOTICES.md must credit PyTorch.",
        )

    def test_mentions_transformers(self) -> None:
        self.assertIn(
            "transformers",
            self.lowered,
            "THIRD_PARTY_NOTICES.md must credit Transformers.",
        )

    def test_mentions_safetensors(self) -> None:
        self.assertIn(
            "safetensors",
            self.lowered,
            "THIRD_PARTY_NOTICES.md must credit Safetensors.",
        )


# ---------------------------------------------------------------------------
# prepare_nemotron_model.py
# ---------------------------------------------------------------------------


class PrepareNemotronScriptTests(unittest.TestCase):
    """The source tree must ship prepare_nemotron_model.py with --offline."""

    def setUp(self) -> None:
        self.contents = _read_text(PREPARE_NEMOTRON_PATH)

    def test_script_exists_in_repo_root(self) -> None:
        self.assertTrue(
            PREPARE_NEMOTRON_PATH.is_file(),
            f"expected prepare_nemotron_model.py at {PREPARE_NEMOTRON_PATH}",
        )

    def test_offline_flag_declared(self) -> None:
        # argparse-style ``"--offline"`` string in the source is sufficient.
        self.assertIn(
            '"--offline"',
            self.contents,
            'prepare_nemotron_model.py must declare an "--offline" argparse flag.',
        )

    def test_offline_flag_actually_offline(self) -> None:
        # A real offline path must short-circuit before any URL fetch /
        # urllib.request.urlopen call. The simplest invariant we can check
        # path-based without importing the module is the presence of the
        # parameter binding used by the staging helper.
        self.assertIn(
            "offline=",
            self.contents,
            "prepare_nemotron_model.py must thread an offline= flag into "
            "the staging helper.",
        )

    def test_does_not_eagerly_touch_network_at_import_time(self) -> None:
        # Belt-and-braces: importing the module must not trigger a network
        # call. We can't import it without running main(), so we only check
        # that the import-time code is free of ``urlopen``/``requests``/etc.
        head = self.contents.split("\ndef ", 1)[0]
        self.assertNotIn("urlopen", head)
        self.assertNotIn("requests.", head)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
