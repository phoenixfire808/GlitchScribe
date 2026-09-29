from __future__ import annotations

import importlib.util
import importlib
import json
import tempfile
import unittest
from pathlib import Path


class WorkflowTests(unittest.TestCase):
    def test_workflow_module_is_available(self):
        self.assertIsNotNone(importlib.util.find_spec("drew_local_voice.workflows"))

    def test_snippet_store_seeds_resolves_and_persists_local_json(self):
        workflows = importlib.import_module("drew_local_voice.workflows")
        snippet_type = getattr(workflows, "Snippet", None)
        store_type = getattr(workflows, "SnippetStore", None)
        built_ins = getattr(workflows, "built_in_snippets", None)
        self.assertIsNotNone(snippet_type)
        self.assertIsNotNone(store_type)
        self.assertIsNotNone(built_ins)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "snippets.json"
            store = store_type(path, defaults=built_ins())
            store.initialize()

            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["version"], 1)
            self.assertIn("__main__", store.resolve("Python Main.").content)

            store.upsert(
                snippet_type(
                    name="Docker Compose Up",
                    content="docker compose up -d",
                    description="Start the local stack in detached mode.",
                )
            )
            reloaded = store_type(path, defaults=built_ins())
            self.assertEqual(
                reloaded.resolve("docker compose up").content,
                "docker compose up -d",
            )

    def test_prompt_compiler_builds_a_bounded_implementation_brief(self):
        workflows = importlib.import_module("drew_local_voice.workflows")
        compiler_type = getattr(workflows, "PromptCompiler", None)
        self.assertIsNotNone(compiler_type)

        compiled = compiler_type().compile(
            "Add SHA256 validation to API responses",
        )

        self.assertEqual(compiled.template_id, "implementation_brief")
        self.assertEqual(compiled.intent, "Add SHA256 validation to API responses")
        self.assertIn("IMPLEMENTATION BRIEF", compiled.text)
        self.assertIn("Add SHA256 validation to API responses", compiled.text)
        self.assertIn("focused regression coverage", compiled.text)
        self.assertIn("Do not execute external or public actions", compiled.text)
        with self.assertRaises(ValueError):
            compiler_type().compile("  ")


if __name__ == "__main__":
    unittest.main()
