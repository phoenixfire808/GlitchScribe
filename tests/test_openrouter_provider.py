from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit=-1):
        return json.dumps(self.payload).encode("utf-8")


class _StreamingResponse:
    def __init__(self, lines):
        self.lines = [line.encode("utf-8") for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def readline(self, _limit=-1):
        return self.lines.pop(0) if self.lines else b""

    def close(self):
        return None


class OpenRouterProviderTests(unittest.TestCase):
    def test_openrouter_streaming_emits_chunks_and_returns_complete_response(self):
        from drew_local_voice.ai.providers import AIRequest, OpenRouterProvider

        captured = {}
        chunks = []

        def opener(request, *, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _StreamingResponse(
                [
                    'data: {"choices":[{"delta":{"content":"FAST"}}]}\n\n',
                    ': keep-alive\n\n',
                    'data: {"choices":[{"delta":{"content":"_PATH"}}]}\n\n',
                    "data: [DONE]\n\n",
                ]
            )

        provider = OpenRouterProvider(
            api_key="sk-test-only",
            model="test/model",
            timeout_seconds=9.0,
            opener=opener,
        )
        response = provider.complete_stream(
            AIRequest(
                instruction="Return the smoke sentinel.",
                context="synthetic test context",
                max_output_characters=100,
            ),
            on_text=chunks.append,
        )

        self.assertTrue(captured["payload"]["stream"])
        self.assertEqual(captured["timeout"], 9.0)
        self.assertEqual(chunks, ["FAST", "_PATH"])
        self.assertEqual(response.text, "FAST_PATH")
    def test_openrouter_completion_uses_authenticated_api_path_and_metadata(self):
        from drew_local_voice.ai.providers import AIRequest, OpenRouterProvider

        captured = {}

        def opener(request, *, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.headers)
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _Response(
                {
                    "id": "gen-test",
                    "model": "nvidia/nemotron-3-nano-30b-a3b:free",
                    "choices": [{"message": {"content": "OPENROUTER_OK"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                }
            )

        provider = OpenRouterProvider(
            "https://openrouter.ai/api/v1",
            api_key="sk-test-only",
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            timeout_seconds=9.0,
            opener=opener,
        )
        response = provider.complete(
            AIRequest(
                instruction="Return the smoke sentinel.",
                context="synthetic test context",
                max_output_characters=100,
            )
        )

        self.assertEqual(
            captured["url"],
            "https://openrouter.ai/api/v1/chat/completions",
        )
        headers = {key.casefold(): value for key, value in captured["headers"].items()}
        self.assertEqual(headers["authorization"], "Bearer sk-test-only")
        self.assertEqual(headers["http-referer"], "https://drewlocalvoice.local")
        self.assertEqual(headers["x-title"], "GlitchScribe")
        self.assertEqual(captured["timeout"], 9.0)
        self.assertEqual(
            captured["payload"]["model"],
            "nvidia/nemotron-3-nano-30b-a3b:free",
        )
        self.assertEqual(response.text, "OPENROUTER_OK")
        self.assertEqual(response.provider_label, "OpenRouter • nvidia/nemotron-3-nano-30b-a3b:free")
        self.assertEqual(response.prompt_tokens, 7)
        self.assertEqual(response.completion_tokens, 2)

    def test_openrouter_model_discovery_uses_authenticated_models_path(self):
        from drew_local_voice.ai.providers import OpenRouterProvider

        captured = {}

        def opener(request, *, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.headers)
            captured["timeout"] = timeout
            return _Response(
                {
                    "data": [
                        {"id": "nvidia/nemotron-3-nano-30b-a3b:free"},
                        {"id": "other/model"},
                    ]
                }
            )

        provider = OpenRouterProvider(
            api_key="sk-test-only",
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            opener=opener,
        )
        self.assertEqual(
            provider.list_models(timeout_seconds=4.0),
            ("nvidia/nemotron-3-nano-30b-a3b:free", "other/model"),
        )
        self.assertEqual(captured["url"], "https://openrouter.ai/api/v1/models")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer sk-test-only")
        self.assertEqual(captured["timeout"], 4.0)

    def test_openrouter_requires_key_and_rejects_untrusted_endpoint(self):
        from drew_local_voice.ai.providers import OpenRouterProvider

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                OpenRouterProvider(model="nvidia/nemotron-3-nano-30b-a3b:free")
        with self.assertRaises(ValueError):
            OpenRouterProvider(
                "https://example.com/api/v1",
                api_key="sk-test-only",
                model="nvidia/nemotron-3-nano-30b-a3b:free",
            )


if __name__ == "__main__":
    unittest.main()