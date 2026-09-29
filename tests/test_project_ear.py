from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import Mock, patch

from local_voice import LocalVoiceApp


class ProjectEarTests(unittest.TestCase):
    def test_registry_accepts_only_bounded_relative_metadata(self):
        self.assertIsNotNone(importlib.util.find_spec("drew_local_voice.project_ear"))
        from drew_local_voice.project_ear import InventoryRejected, ProjectEarRegistry

        registry = ProjectEarRegistry(max_files=3)
        snapshot = registry.ingest(
            {
                "protocol": 1,
                "client_id": "vscode-window-1",
                "workspace_names": ["GlitchScribe"],
                "relative_paths": [
                    "local_voice.py",
                    "drew_local_voice/context.py",
                    "tests/test_core.py",
                    "fourth/is/truncated.py",
                ],
            },
            clock=lambda: 123.0,
        )
        self.assertEqual(snapshot.workspace_names, ("GlitchScribe",))
        self.assertEqual(
            snapshot.relative_paths,
            ("local_voice.py", "drew_local_voice/context.py", "tests/test_core.py"),
        )
        self.assertEqual(snapshot.received_file_count, 4)
        self.assertTrue(snapshot.truncated)
        self.assertEqual(snapshot.observed_at, 123.0)
        self.assertEqual(registry.snapshot, snapshot)

        for unsafe in (
            ["C:/private/file.py"],
            ["/absolute/file.py"],
            ["../escape.py"],
            ["safe/../../escape.py"],
            ["bad\x00name.py"],
        ):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(InventoryRejected):
                    registry.ingest(
                        {
                            "protocol": 1,
                            "client_id": "vscode-window-1",
                            "workspace_names": ["GlitchScribe"],
                            "relative_paths": unsafe,
                        }
                    )

    def test_registry_v2_requires_separate_metadata_approvals(self):
        import drew_local_voice.project_ear as project_ear

        self.assertTrue(hasattr(project_ear, "ActiveFileMetadata"))
        InventoryRejected = project_ear.InventoryRejected
        ProjectEarRegistry = project_ear.ProjectEarRegistry

        payload = {
            "protocol": 2,
            "client_id": "vscode-window-1",
            "workspace_names": ["GlitchScribe"],
            "relative_paths": ["src/main.py"],
            "metadata_approvals": {
                "active_file": True,
                "diagnostics": True,
            },
            "active_file": {
                "relative_path": "src/main.py",
                "language_id": "python",
            },
            "diagnostics": [
                {
                    "relative_path": "src/main.py",
                    "severity": 1,
                    "code": "E0602",
                    "source": "pylint",
                    "start_line": 7,
                    "start_character": 3,
                    "end_line": 7,
                    "end_character": 12,
                }
            ],
        }
        snapshot = ProjectEarRegistry().ingest(payload, clock=lambda: 123.0)

        self.assertEqual(snapshot.protocol, 2)
        self.assertEqual(snapshot.active_file.relative_path, "src/main.py")
        self.assertEqual(snapshot.active_file.language_id, "python")
        self.assertEqual(snapshot.diagnostics[0].code, "E0602")
        self.assertNotIn("message", snapshot.diagnostics[0].__dict__)

        payload["metadata_approvals"] = {
            "active_file": False,
            "diagnostics": True,
        }
        with self.assertRaises(InventoryRejected):
            ProjectEarRegistry().ingest(payload)

    def test_loopback_server_requires_token_and_removes_rendezvous(self):
        import drew_local_voice.project_ear as project_ear

        ProjectEarRegistry = project_ear.ProjectEarRegistry
        ProjectEarServer = getattr(project_ear, "ProjectEarServer", None)
        self.assertIsNotNone(ProjectEarServer)

        with tempfile.TemporaryDirectory() as folder:
            rendezvous = Path(folder) / "project-ear.json"
            registry = ProjectEarRegistry(max_files=10)
            server = ProjectEarServer(
                registry,
                rendezvous,
                token_factory=lambda: "fixed-test-token",
            )
            server.start()
            try:
                descriptor = json.loads(rendezvous.read_text(encoding="utf-8"))
                self.assertEqual(descriptor["host"], "127.0.0.1")
                self.assertEqual(descriptor["token"], "fixed-test-token")
                endpoint = f"http://127.0.0.1:{descriptor['port']}/v1/workspace-inventory"
                body = json.dumps(
                    {
                        "protocol": 1,
                        "client_id": "vscode-test",
                        "workspace_names": ["Project"],
                        "relative_paths": ["src/main.js"],
                    }
                ).encode("utf-8")
                with self.assertRaises(HTTPError) as unauthorized:
                    urlopen(Request(endpoint, data=body, method="PUT"), timeout=2)
                self.assertEqual(unauthorized.exception.code, 401)

                request = Request(
                    endpoint,
                    data=body,
                    method="PUT",
                    headers={
                        "Authorization": "Bearer fixed-test-token",
                        "Content-Type": "application/json",
                    },
                )
                with urlopen(request, timeout=2) as response:
                    accepted = json.loads(response.read().decode("utf-8"))
                self.assertEqual(response.status, 202)
                self.assertEqual(accepted["file_count"], 1)
                self.assertEqual(registry.snapshot.relative_paths, ("src/main.js",))
            finally:
                server.stop()
            self.assertFalse(rendezvous.exists())

    def test_app_owns_project_ear_server_lifecycle(self):
        self.assertTrue(hasattr(LocalVoiceApp, "_start_project_ear"))
        self.assertTrue(hasattr(LocalVoiceApp, "_stop_project_ear"))
        app = LocalVoiceApp.__new__(LocalVoiceApp)
        server = Mock()
        with patch("local_voice.ProjectEarServer", return_value=server) as server_type:
            app._start_project_ear()
        server_type.assert_called_once()
        server.start.assert_called_once_with()
        self.assertIs(app.project_ear_server, server)
        self.assertIsNotNone(app.project_ear_registry)

        app._stop_project_ear()
        server.stop.assert_called_once_with()
        self.assertIsNone(app.project_ear_server)


if __name__ == "__main__":
    unittest.main()
