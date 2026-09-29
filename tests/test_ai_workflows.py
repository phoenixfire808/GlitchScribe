from __future__ import annotations

import importlib.util
import importlib
import json
import socket
import threading
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from drew_local_voice.theme import build_palette


class AIWorkflowTests(unittest.TestCase):
    def test_compact_preview_card_can_show_partial_stream(self):
        from drew_local_voice.ai.preview_ui import AIPreviewCard

        card = AIPreviewCard.__new__(AIPreviewCard)
        card.compact = True
        card.result_text = Mock()
        card.result_heading = Mock()
        card.status_label = Mock()
        card._error = False
        card._set_adaptive_controls = Mock()
        card._set_compact_controls = Mock()

        card.set_partial_text("partial result")

        card.result_text.insert.assert_called_once_with("1.0", "partial result")
        card.status_label.configure.assert_called_once_with(
            text="Generating · live result"
        )
        card.result_heading.configure.assert_called_once_with(
            text="RESULT · LIVE"
        )
        card._set_adaptive_controls.assert_called_once_with(enabled=False)
        card._set_compact_controls.assert_called_once_with(ready=False)

    def test_transform_card_is_scheduled_before_provider_preparation(self):
        import inspect
        from local_voice import LocalVoiceApp

        source = inspect.getsource(LocalVoiceApp._handle_action_wheel_choice)
        self.assertLess(
            source.index("show_transform_card()"),
            source.index("transform_profiles = _lazy_import"),
        )
        self.assertLess(
            source.index("show_transform_card()"),
            source.index('providers = _lazy_import("drew_local_voice.ai.providers")'),
        )

    def test_first_partial_visible_is_logged_without_content(self):
        import local_voice
        from local_voice import LocalVoiceApp

        app = LocalVoiceApp.__new__(LocalVoiceApp)
        card = Mock()
        card.compact = True
        card.matches_request.return_value = True
        app._ai_preview_card = card
        app._active_ai_transform_action_id = "make_shorter"
        app._active_ai_transform_started_at = 1.0
        app._ai_visible_partial_requests = set()
        with patch.object(local_voice, "append_runtime_event") as event:
            app._update_ai_transformation_partial("partial text", "request-1")
        card.set_partial_text.assert_called_once_with("partial text")
        names = [call.args[0] for call in event.call_args_list]
        self.assertIn("ai_transformation_first_partial_visible", names)
        receipt = next(
            call.args[0:2]
            for call in event.call_args_list
            if call.args and call.args[0] == "ai_transformation_first_partial_visible"
        )
        self.assertNotIn("partial text", repr(receipt))

    def test_microphone_ring_geometry_is_centered_and_pulses(self):
        from drew_local_voice.microphone_ring import ring_phase_geometry

        first = ring_phase_geometry(112, 0.0)
        second = ring_phase_geometry(112, 1.5)
        self.assertEqual(first["center"], (56.0, 56.0))
        self.assertEqual(len(first["outer"]), 4)
        self.assertGreater(first["arc_extent"], 170.0)
        self.assertNotEqual(first["outer"], second["outer"])

        self.assertIsNotNone(importlib.util.find_spec("drew_local_voice.ai"))

    def test_local_provider_is_loopback_only_and_parses_bounded_response(self):
        self.assertIsNotNone(
            importlib.util.find_spec("drew_local_voice.ai.providers")
        )
        providers = importlib.import_module("drew_local_voice.ai.providers")
        request_type = getattr(providers, "AIRequest", None)
        provider_type = getattr(providers, "LocalOpenAIProvider", None)
        self.assertIsNotNone(request_type)
        self.assertIsNotNone(provider_type)

        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit=-1):
                return json.dumps(
                    {
                        "choices": [
                            {"message": {"content": "A concise explanation."}}
                        ],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                    }
                ).encode("utf-8")

        def opener(request, *, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

        provider = provider_type(
            "http://127.0.0.1:8080",
            model="qwen3-coder-local",
            timeout_seconds=7.5,
            opener=opener,
        )
        response = provider.complete(
            request_type(
                instruction="Explain this selected error.",
                context="ValueError: invalid token",
                max_output_characters=13,
            )
        )

        payload = json.loads(captured["request"].data.decode("utf-8"))
        self.assertEqual(captured["request"].full_url, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(captured["timeout"], 7.5)
        self.assertEqual(payload["model"], "qwen3-coder-local")
        self.assertNotIn("think", payload)
        self.assertIn("untrusted selected text", payload["messages"][0]["content"])
        self.assertIn("ValueError: invalid token", payload["messages"][1]["content"])
        self.assertEqual(response.text, "A concise exp")
        self.assertTrue(response.truncated)
        self.assertEqual(response.provider_label, "Local • qwen3-coder-local")
        self.assertEqual(response.prompt_tokens, 12)
        self.assertEqual(response.completion_tokens, 4)

        ollama_captured = {}

        def ollama_opener(request, *, timeout):
            ollama_captured["request"] = request
            ollama_captured["timeout"] = timeout
            return Response()

        ollama_provider = provider_type(
            "http://127.0.0.1:11434",
            model="local-lfm",
            opener=ollama_opener,
        )
        ollama_provider.complete(
            request_type(
                instruction="Make this shorter.",
                context="A local sample.",
            )
        )
        ollama_payload = json.loads(
            ollama_captured["request"].data.decode("utf-8")
        )
        self.assertIs(ollama_payload["think"], False)
        self.assertEqual(ollama_captured["timeout"], 30.0)

        for endpoint in (
            "https://api.openai.com",
            "http://192.168.1.5:8080",
            "http://example.com",
            "file:///tmp/model",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(ValueError):
                    provider_type(endpoint, model="unsafe")

    def test_local_provider_cancellation_closes_active_response(self):
        providers = importlib.import_module("drew_local_voice.ai.providers")
        signal_type = getattr(providers, "CancellationSignal", None)
        cancelled_type = getattr(providers, "ProviderCancelled", None)
        self.assertIsNotNone(signal_type)
        self.assertIsNotNone(cancelled_type)

        read_started = threading.Event()
        response_closed = threading.Event()

        class BlockingResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit=-1):
                read_started.set()
                response_closed.wait(2)
                raise OSError("response closed")

            def close(self):
                response_closed.set()

        provider = providers.LocalOpenAIProvider(
            "http://127.0.0.1:8080",
            model="qwen3-coder-local",
            opener=lambda _request, *, timeout: BlockingResponse(),
        )
        signal = signal_type()
        result = {}

        def run():
            try:
                provider.complete(
                    providers.AIRequest(
                        instruction="Explain.",
                        context="synthetic error",
                    ),
                    cancellation=signal,
                )
            except Exception as exc:
                result["exception"] = exc

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(read_started.wait(1))
        self.assertTrue(signal.cancel())
        worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertTrue(response_closed.is_set())
        self.assertIsInstance(result.get("exception"), cancelled_type)

    def test_local_provider_reports_timeout_distinctly(self):
        providers = importlib.import_module("drew_local_voice.ai.providers")
        timeout_type = getattr(providers, "ProviderTimeout", None)
        self.assertIsNotNone(timeout_type)

        provider = providers.LocalOpenAIProvider(
            "http://127.0.0.1:8080",
            model="qwen3-coder-local",
            opener=lambda _request, *, timeout: (_ for _ in ()).throw(
                socket.timeout("timed out")
            ),
        )
        with self.assertRaises(timeout_type):
            provider.complete(
                providers.AIRequest(
                    instruction="Explain.",
                    context="synthetic error",
                )
            )

    def test_local_provider_accepts_a_trusted_edit_mode_system_contract(self):
        providers = importlib.import_module("drew_local_voice.ai.providers")
        self.assertIn("system_prompt", providers.AIRequest.__dataclass_fields__)
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit=-1):
                return b'{"choices":[{"message":{"content":"Short result."}}]}'

        def opener(request, *, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return Response()

        provider = providers.LocalOpenAIProvider(
            "http://127.0.0.1:8080",
            model="qwen3-coder-local",
            opener=opener,
        )
        provider.complete(
            providers.AIRequest(
                instruction="Make this shorter.",
                context="This is selected text that should become shorter.",
                system_prompt="Transform only the selected text and return only replacement text.",
                context_label="selected_text",
            )
        )

        messages = captured["payload"]["messages"]
        self.assertEqual(
            messages[0]["content"],
            "Transform only the selected text and return only replacement text.",
        )
        self.assertIn("<selected_text>", messages[1]["content"])
        self.assertIn("</selected_text>", messages[1]["content"])

    def test_request_timeout_overrides_provider_default_within_bounds(self):
        providers = importlib.import_module("drew_local_voice.ai.providers")
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit=-1):
                return b'{"choices":[{"message":{"content":"Clean text."}}]}'

        def opener(_request, *, timeout):
            captured["timeout"] = timeout
            return Response()

        provider = providers.LocalOpenAIProvider(
            "http://127.0.0.1:8080",
            model="qwen3-coder-local",
            timeout_seconds=45.0,
            opener=opener,
        )
        self.assertIn("timeout_seconds", providers.AIRequest.__dataclass_fields__)
        provider.complete(
            providers.AIRequest(
                instruction="Clean this.",
                context="raw text",
                timeout_seconds=9.0,
            )
        )

        self.assertEqual(captured["timeout"], 9.0)

    def test_preview_gate_binds_exact_payload_and_consumes_once(self):
        self.assertIsNotNone(importlib.util.find_spec("drew_local_voice.ai.preview"))
        preview = importlib.import_module("drew_local_voice.ai.preview")
        draft_type = getattr(preview, "PreviewDraft", None)
        gate_type = getattr(preview, "PreviewGate", None)
        self.assertIsNotNone(draft_type)
        self.assertIsNotNone(gate_type)

        draft = draft_type.create(
            original="ValueError: invalid token",
            instruction="Explain the selected error.",
            result="The parser rejected an invalid token.",
            provider_label="Local • qwen3-coder-local",
            context_lenses=("Selected Text",),
            target_window=88,
            clock=lambda: 456.0,
            id_factory=lambda: "preview-1",
        )
        self.assertTrue(hasattr(draft, "selection_identity"))
        bound = draft_type.create(
            original=draft.original,
            instruction=draft.instruction,
            result=draft.result,
            provider_label=draft.provider_label,
            context_lenses=draft.context_lenses,
            target_window=draft.target_window,
            selection_identity="control-1",
            clock=lambda: 456.0,
            id_factory=lambda: "preview-1",
        )
        moved = draft_type.create(
            original=draft.original,
            instruction=draft.instruction,
            result=draft.result,
            provider_label=draft.provider_label,
            context_lenses=draft.context_lenses,
            target_window=draft.target_window,
            selection_identity="control-2",
            clock=lambda: 456.0,
            id_factory=lambda: "preview-1",
        )
        self.assertNotEqual(bound.fingerprint, moved.fingerprint)
        changed = draft_type.create(
            original=draft.original,
            instruction=draft.instruction,
            result=draft.result + " Changed.",
            provider_label=draft.provider_label,
            context_lenses=draft.context_lenses,
            target_window=draft.target_window,
            clock=lambda: 456.0,
            id_factory=lambda: "preview-1",
        )
        self.assertNotEqual(draft.fingerprint, changed.fingerprint)

        gate = gate_type()
        gate.stage(draft)
        self.assertIsNone(gate.claim("preview-1", changed.fingerprint))
        self.assertEqual(gate.claim("preview-1", draft.fingerprint), draft)
        self.assertIsNone(gate.claim("preview-1", draft.fingerprint))

    def test_preview_marks_whitespace_only_result_unchanged(self):
        preview = importlib.import_module("drew_local_voice.ai.preview")
        draft = preview.PreviewDraft.create(
            original="value = 1\r\n",
            instruction="Simplify selection.",
            result="value = 1\n\n",
            provider_label="Local • qwen3-coder-local",
            context_lenses=("Selected Text",),
            target_window=88,
            clock=lambda: 456.0,
            id_factory=lambda: "preview-unchanged",
        )
        self.assertTrue(hasattr(draft, "unchanged"))
        self.assertTrue(draft.unchanged)

        root = tk.Tk()
        root.withdraw()
        try:
            card = importlib.import_module(
                "drew_local_voice.ai.preview_ui"
            ).AIPreviewCard(
                root,
                draft,
                palette=build_palette("dark", "violet"),
                on_stage=Mock(),
                on_replace=Mock(),
                on_copy=Mock(),
                on_retry=Mock(),
                on_cancel=Mock(),
            )
            self.assertEqual(
                card.action_buttons["Stage in Review"].cget("state"),
                "disabled",
            )
            self.assertIn("ALREADY GOOD", card.result_heading.cget("text"))
            card.close()
        finally:
            root.destroy()

    def test_preview_card_renders_all_explicit_actions_and_binds_stage(self):
        self.assertIsNotNone(importlib.util.find_spec("drew_local_voice.ai.preview_ui"))
        preview = importlib.import_module("drew_local_voice.ai.preview")
        preview_ui = importlib.import_module("drew_local_voice.ai.preview_ui")
        card_type = getattr(preview_ui, "AIPreviewCard", None)
        self.assertIsNotNone(card_type)
        self.assertTrue(hasattr(card_type, "_replace"))
        draft = preview.PreviewDraft.create(
            original="selected error",
            instruction="Explain the selected error.",
            result="A local explanation.",
            provider_label="Local • qwen3-coder-local",
            context_lenses=("Selected Text",),
            target_window=88,
            clock=lambda: 456.0,
            id_factory=lambda: "preview-ui-1",
        )
        root = tk.Tk()
        root.withdraw()
        on_stage = Mock()
        on_replace = Mock()
        try:
            card = card_type(
                root,
                draft,
                palette=build_palette("dark", "violet"),
                on_stage=on_stage,
                on_replace=on_replace,
                on_copy=Mock(),
                on_retry=Mock(),
                on_cancel=Mock(),
            )
            self.assertEqual(
                set(card.action_buttons),
                {"Stage in Review", "Replace Selection", "Copy", "Retry", "Cancel"},
            )
            self.assertIn("selected error", card.original_text.get("1.0", "end"))
            self.assertIn("A local explanation.", card.result_text.get("1.0", "end"))
            card._stage()
            on_stage.assert_called_once_with(draft.transaction_id, draft.fingerprint)
            self.assertFalse(card.window.winfo_exists())

            bound = preview.PreviewDraft.create(
                original="selected error",
                instruction="Explain the selected error.",
                result="A local explanation.",
                provider_label="Local • qwen3-coder-local",
                context_lenses=("Selected Text",),
                target_window=88,
                selection_identity="control-1",
            )
            replacement_card = card_type(
                root,
                bound,
                palette=build_palette("dark", "violet"),
                on_stage=Mock(),
                on_replace=on_replace,
                on_copy=Mock(),
                on_retry=Mock(),
                on_cancel=Mock(),
            )
            replacement_card._replace()
            on_replace.assert_called_once_with(
                bound.transaction_id,
                bound.fingerprint,
            )
            self.assertFalse(replacement_card.window.winfo_exists())
        finally:
            root.destroy()
    def test_compact_transformation_card_is_immediate_and_result_only(self):
        preview = importlib.import_module("drew_local_voice.ai.preview")
        preview_ui = importlib.import_module("drew_local_voice.ai.preview_ui")
        root = tk.Tk()
        root.withdraw()
        on_cancel = Mock()
        try:
            card = preview_ui.AIPreviewCard(
                root,
                None,
                palette=build_palette("dark", "violet"),
                on_stage=Mock(),
                on_replace=Mock(),
                on_copy=Mock(),
                on_retry=Mock(),
                on_cancel=on_cancel,
                compact=True,
                request_id="wheel-request-1",
                action_label="Make shorter",
            )
            self.assertTrue(card.compact)
            self.assertIsNone(card.original_text)
            self.assertIsNone(card.instruction)
            self.assertFalse(hasattr(card, "adaptive_summary"))
            self.assertIn("NOT APPLIED", card.preview_badge.cget("text"))
            self.assertTrue(hasattr(card, "result_scrollbar"))
            self.assertLessEqual(int(card.window.winfo_width()), 720)
            self.assertIn("Waiting", card.result_text.get("1.0", "end"))
            self.assertEqual(card.action_buttons["Stage in Review"].cget("state"), "disabled")
            self.assertEqual(card.action_buttons["Copy"].cget("state"), "disabled")
            self.assertEqual(card.action_buttons["Cancel"].cget("state"), "normal")

            draft = preview.PreviewDraft.create(
                original="selected error",
                instruction="Make the selected text shorter.",
                result="Shorter result.",
                provider_label="Provider metadata is not rendered in compact mode",
                context_lenses=("Action Wheel", "Make shorter"),
                target_window=88,
                selection_identity="control-1",
                id_factory=lambda: "wheel-request-1",
            )
            card.set_draft(draft)
            self.assertEqual(card.request_id, "wheel-request-1")
            self.assertIn("Shorter result.", card.result_text.get("1.0", "end"))
            self.assertEqual(card.action_buttons["Stage in Review"].cget("state"), "normal")
            self.assertEqual(card.action_buttons["Replace Selection"].cget("state"), "normal")
            card._cancel()
            on_cancel.assert_called_once_with("wheel-request-1")
            self.assertFalse(card.window.winfo_exists())
        finally:
            root.destroy()
    def test_compact_preview_exposes_visible_bounded_pass_controls(self):
        preview = importlib.import_module("drew_local_voice.ai.preview")
        preview_ui = importlib.import_module("drew_local_voice.ai.preview_ui")
        root = tk.Tk()
        root.withdraw()
        on_retry = Mock()
        on_variant_retry = Mock()
        try:
            card = preview_ui.AIPreviewCard(
                root,
                None,
                palette=build_palette("dark", "violet"),
                on_stage=Mock(),
                on_replace=Mock(),
                on_copy=Mock(),
                on_retry=on_retry,
                on_cancel=Mock(),
                on_variant_retry=on_variant_retry,
                compact=True,
                request_id="pass-request-1",
                action_label="Make clearer",
            )
            self.assertEqual(set(card._pass_buttons), {1, 2, 3})
            self.assertTrue(all(button.cget("state") == "disabled" for button in card._pass_buttons.values()))
            draft = preview.PreviewDraft.create(
                original="selected text",
                instruction="Make the selected text clearer.",
                result="Clear selected text.",
                provider_label="Local test provider",
                context_lenses=("Action Wheel",),
                target_window=12,
                selection_identity="control-1",
                action_id="make_clearer",
                id_factory=lambda: "pass-request-1",
            )
            card.set_draft(draft)
            self.assertTrue(all(button.cget("state") == "normal" for button in card._pass_buttons.values()))
            card._run_passes(2, "parallel")
            on_variant_retry.assert_called_once_with(
                "pass-request-1",
                draft.fingerprint,
                {"execution": "parallel", "requested_passes": 2},
            )
            card._run_passes(1, "current")
            on_retry.assert_called_once_with("pass-request-1", draft.fingerprint)
            card.close()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
