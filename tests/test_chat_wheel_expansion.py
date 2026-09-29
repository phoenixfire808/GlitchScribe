from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from drew_local_voice.ai.openrouter_models import (
    eligible_free_text_models,
    parse_model_catalog,
    probe_model_latencies,
    select_fastest_free,
    solid_text_model_options,
)
from drew_local_voice.ai.transformation_profiles import (
    build_transformation_spec,
    validate_transformation_result,
)


class OpenRouterCatalogTests(unittest.TestCase):
    def _payload(self):
        def model(model_id, *, name="Model", prompt="0", completion="0", modalities=("text",), context=262144, max_completion=32768, parameters=("max_tokens",)):
            return {
                "id": model_id,
                "name": name,
                "context_length": context,
                "pricing": {"prompt": prompt, "completion": completion},
                "supported_parameters": list(parameters),
                "input_modalities": list(modalities),
                "output_modalities": ["text"],
                "top_provider": {
                    "context_length": context,
                    "max_completion_tokens": max_completion,
                },
            }

        return {
            "data": [
                model("google/gemma-4-31b-it:free", name="Gemma"),
                model("inclusionai/ling-3.0-tiny:free", name="Ling Tiny"),
                model("openrouter/free", name="Router"),
                model("google/lyria-3-pro-preview", name="Audio Preview"),
                model("cohere/north-mini-code:free", name="Code", modalities=("text",)),
                model("paid/model", prompt="0.000001", completion="0.000002"),
            ]
        }

    def test_live_shape_uses_top_level_modalities_and_excludes_non_free_zero_price_ids(self):
        models = parse_model_catalog(self._payload())
        eligible = eligible_free_text_models(models)
        ids = {model.model_id for model in eligible}
        self.assertIn("google/gemma-4-31b-it:free", ids)
        self.assertIn("inclusionai/ling-3.0-tiny:free", ids)
        self.assertNotIn("openrouter/free", ids)
        self.assertNotIn("google/lyria-3-pro-preview", ids)
        self.assertNotIn("paid/model", ids)

    def test_measured_selection_beats_metadata_order_and_ignores_invalid_measurements(self):
        models = parse_model_catalog(self._payload())
        selection = select_fastest_free(
            models,
            measured_latencies={
                "google/gemma-4-31b-it:free": 0.8,
                "inclusionai/ling-3.0-tiny:free": "invalid",
            },
        )
        self.assertEqual(selection.route, "measured")
        self.assertEqual(selection.model.model_id, "google/gemma-4-31b-it:free")
        self.assertEqual(selection.measured_latency_seconds, 0.8)

    def test_solid_options_are_compact_and_exclude_non_transform_variants(self):
        models = parse_model_catalog(self._payload())
        options = solid_text_model_options(models, max_models=4)
        ids = [model.model_id for model in options]
        self.assertLessEqual(len(ids), 4)
        self.assertIn("google/gemma-4-31b-it:free", ids)
        self.assertIn("paid/model", ids)
        self.assertNotIn("google/lyria-3-pro-preview", ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_probe_is_bounded_and_uses_synthetic_context_only(self):
        models = parse_model_catalog(self._payload())
        captured = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit=-1):
                return json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode()

        def opener(request, *, timeout):
            captured.append((request, timeout))
            return Response()

        measured = probe_model_latencies(
            models,
            api_key="sk-test",
            max_candidates=3,
            opener=opener,
        )
        self.assertEqual(len(captured), 3)
        self.assertEqual(len(measured), 3)
        self.assertTrue(all("selected_text" not in request.data.decode() for request, _ in captured))
        self.assertTrue(all(request.full_url.endswith("/chat/completions") for request, _ in captured))


class TransformationProfileTests(unittest.TestCase):
    def test_expand_profile_requests_material_growth(self):
        spec = build_transformation_spec(
            "expand_full_draft",
            "Expand into full draft",
            "Expand the note.",
            80,
        )
        self.assertTrue(spec.requires_expansion)
        self.assertGreaterEqual(spec.minimum_output_characters, 320)
        self.assertIn("substantially longer", spec.instruction)
        self.assertFalse(validate_transformation_result("short note", "still short", spec)[0])

    def test_cursor_agent_task_has_exact_five_section_contract(self):
        spec = build_transformation_spec(
            "cursor_agent_task",
            "Cursor agent task",
            "ignored legacy instruction",
            100,
        )
        self.assertEqual(
            spec.action_id,
            "cursor_agent_task",
        )
        for heading in (
            "Goal:",
            "Context:",
            "Steps:",
            "Acceptance criteria:",
            "Constraints:",
        ):
            self.assertIn(heading, spec.instruction)
        good = (
            "Goal: Build the feature and make the result dependable for the user.\n\n"
            "Context:\n"
            "- The selected note is the only source material.\n"
            "- Preserve its intent and distinguish assumptions from known facts.\n"
            "- The result must be easy for another agent to implement.\n\n"
            "Steps:\n"
            "1. Inspect the selected requirement and identify the concrete outcome.\n"
            "2. Build the smallest complete implementation that satisfies the request.\n"
            "3. Verify the behavior with focused checks and preserve rollback safety.\n"
            "4. Report any unresolved assumption without inventing a fact.\n\n"
            "Acceptance criteria:\n"
            "- The requested behavior is implemented end to end.\n"
            "- The result is clear, testable, and reviewable.\n"
            "- No selected-text meaning is silently dropped.\n\n"
            "Constraints:\n"
            "- Use selected text only as source context.\n"
            "- Do not add unsupported facts, credentials, or external actions."
        )
        self.assertTrue(validate_transformation_result("short note", good, spec)[0])
        self.assertFalse(validate_transformation_result("short note", "```\n" + good + "\n```", spec)[0])
        self.assertFalse(validate_transformation_result("short note", "Goal: only", spec)[0])


if __name__ == "__main__":
    unittest.main()
