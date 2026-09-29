"""Fake-driven unit tests for ``drew_local_voice.resource_ledger``.

These tests do **not** require the live app, model runtimes, GPU, network,
microphone, or filesystem. Every probe is replaced with a deterministic
fake; no test calls ``build_resource_ledger`` with ``probe_enabled=True``
except where the probe behaviour itself is being verified, and those
tests always inject their own fake.

The tests are authored for the project's standard
``python -m unittest discover -s tests`` runner. They follow the same
isolation style used by the runtime resource contract: import the
module under test lazily via a helper so a missing / broken module does
not cascade.
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import sys
import unittest
from typing import Any, Iterable, List, Optional, Sequence, Tuple


def _ensure_module() -> Any:
    """Resolve the module under test without forcing a package import."""
    module_name = "drew_local_voice.resource_ledger"
    try:
        return importlib.import_module(module_name)
    except Exception:
        # Fall back to direct file import. Useful when the package
        # ``__init__`` is partially broken during scaffolding.
        candidate = "drew_local_voice/resource_ledger.py"
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            raise unittest.SkipTest(f"{module_name} not importable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _module_under_test() -> Any:
    """Cached module accessor; caches on the test class for speed."""
    cache_key = "_rl_module_cache"
    cached = getattr(_ModuleCache, cache_key, None)
    if cached is not None:
        return cached
    module = _ensure_module()
    setattr(_ModuleCache, cache_key, module)
    return module


@dataclasses.dataclass
class _ModuleCache:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _primary_asr_input(
    rl: Any,
    *,
    identifier: str = "nemotron-3.5-asr-streaming-0.6b",
    source: str = "huggingface",
    backend: str = "transformers-rnnt",
    device: str = "cuda",
    dtype: str = "float16",
    is_loaded: bool = True,
    artifact_bytes: int = 638_000_000,
    estimated_vram_mb: int = 2304,
    runtime_vram_mb: Optional[int] = None,
) -> Any:
    return rl.ModelInput(
        purpose="primary_asr",
        identifier=identifier,
        source=source,
        backend=backend,
        device=device,
        dtype=dtype,
        is_loaded=is_loaded,
        artifact_bytes=artifact_bytes,
        estimated_vram_mb=estimated_vram_mb,
        runtime_vram_mb=runtime_vram_mb,
    )


def _aux_input(
    rl: Any,
    *,
    identifier: str = "qwen2.5-3b-instruct-q4_k_m.gguf",
    source: str = "local_path",
    backend: str = "llama.cpp",
    device: str = "cpu",
    dtype: str = "q4_k_m",
    is_loaded: bool = False,
    artifact_bytes: int = 2_100_000_000,
) -> Any:
    return rl.ModelInput(
        purpose="aux_local_llm",
        identifier=identifier,
        source=source,
        backend=backend,
        device=device,
        dtype=dtype,
        is_loaded=is_loaded,
        artifact_bytes=artifact_bytes,
    )


def _plugin_input(
    rl: Any,
    *,
    name: str,
    origin: str = "builtin",
    enabled: bool = True,
    loaded: bool = True,
    failure_type: Optional[str] = None,
) -> Any:
    return rl.PluginInput(
        name=name,
        origin=origin,
        enabled=enabled,
        loaded=loaded,
        failure_type=failure_type,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


class FormatHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rl = _module_under_test()

    def test_format_bytes_known_units(self) -> None:
        self.assertEqual(self.rl.format_bytes(0), "0 B")
        self.assertEqual(self.rl.format_bytes(900), "900 B")
        self.assertEqual(self.rl.format_bytes(2048), "2 KiB")
        self.assertEqual(self.rl.format_bytes(5 * 1024 * 1024), "5 MiB")
        self.assertEqual(
            self.rl.format_bytes(2 * 1024 * 1024 * 1024), "2 GiB"
        )

    def test_format_bytes_unknown(self) -> None:
        self.assertEqual(self.rl.format_bytes(None), "unknown")
        self.assertEqual(self.rl.format_bytes(-1), "unknown")
        # Garbage input is also reported as unknown, not as 0.
        self.assertEqual(self.rl.format_bytes("not a number"), "unknown")

    def test_format_mib_integer_only(self) -> None:
        self.assertEqual(self.rl.format_mib(0), "0 MiB")
        self.assertEqual(self.rl.format_mib(2304), "2304 MiB")
        self.assertEqual(self.rl.format_mib(None), "unknown")
        self.assertEqual(self.rl.format_mib(-1), "unknown")

    def test_bytes_to_mib_truncates(self) -> None:
        self.assertEqual(self.rl.bytes_to_mib(0), 0)
        self.assertEqual(self.rl.bytes_to_mib(2 * 1024 * 1024), 2)
        self.assertEqual(
            self.rl.bytes_to_mib(2 * 1024 * 1024 + 512 * 1024), 2
        )
        self.assertIsNone(self.rl.bytes_to_mib(None))
        self.assertIsNone(self.rl.bytes_to_mib(-5))


# ---------------------------------------------------------------------------
# Builder + record structure
# ---------------------------------------------------------------------------


class ResourceLedgerBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rl = _module_under_test()

    def test_primary_only_ledger_has_deterministic_row_order(self) -> None:
        ledger = self.rl.build_resource_ledger(
            primary_asr=_primary_asr_input(self.rl),
        )
        roles = [r.role for r in ledger.records]
        self.assertEqual(
            roles,
            [
                self.rl.ROLE_PRIMARY_ASR,
                self.rl.ROLE_PROCESS,
                self.rl.ROLE_NETWORK,
                self.rl.ROLE_DEVICE,
            ],
        )
        asr = ledger.primary_asr()
        self.assertIsNotNone(asr)
        self.assertEqual(asr.path_or_repository, "nemotron-3.5-asr-streaming-0.6b")
        self.assertEqual(asr.load_state, "loaded")
        self.assertEqual(asr.source, "huggingface")
        self.assertEqual(asr.backend, "transformers-rnnt")

    def test_plugins_are_sorted_by_origin_then_name(self) -> None:
        # Inserted in a non-sorted order; expect deterministic sorted order.
        ledger = self.rl.build_resource_ledger(
            primary_asr=_primary_asr_input(self.rl),
            plugins=[
                _plugin_input(self.rl, name="zeta", origin="user_directory"),
                _plugin_input(self.rl, name="alpha", origin="builtin"),
                _plugin_input(
                    self.rl,
                    name="broken",
                    origin="entry_point",
                    loaded=False,
                    failure_type="ImportError",
                ),
                _plugin_input(self.rl, name="beta", origin="builtin"),
            ],
        )
        plugin_rows = ledger.plugin_rows()
        names = [p.path_or_repository for p in plugin_rows]
        origins = [p.plugin_origin for p in plugin_rows]
        self.assertEqual(
            list(zip(origins, names)),
            [
                ("builtin", "alpha"),
                ("builtin", "beta"),
                ("entry_point", "broken"),
                ("user_directory", "zeta"),
            ],
        )
        broken = plugin_rows[2]
        self.assertEqual(broken.load_state, "failed")

    def test_auxiliary_model_row_appears_after_primary(self) -> None:
        ledger = self.rl.build_resource_ledger(
            primary_asr=_primary_asr_input(self.rl),
            auxiliary_model=_aux_input(self.rl),
        )
        roles = [r.role for r in ledger.records]
        self.assertEqual(
            roles[:2],
            [self.rl.ROLE_PRIMARY_ASR, self.rl.ROLE_AUXILIARY_MODEL],
        )
        aux = ledger.auxiliary_model()
        self.assertIsNotNone(aux)
        self.assertEqual(aux.path_or_repository, "qwen2.5-3b-instruct-q4_k_m.gguf")
        self.assertEqual(aux.load_state, "not_loaded")
        self.assertEqual(aux.source, "local_path")
        self.assertEqual(aux.backend, "llama.cpp")

    def test_no_auxiliary_when_not_supplied(self) -> None:
        ledger = self.rl.build_resource_ledger(
            primary_asr=_primary_asr_input(self.rl),
        )
        self.assertIsNone(ledger.auxiliary_model())

    def test_probe_disabled_by_default(self) -> None:
        # Even with a probe that would otherwise return data, default
        # ``probe_enabled=False`` must ignore it.
        sentinel = object()  # any non-int value would crash int()
        ledger = self.rl.build_resource_ledger(
            primary_asr=_primary_asr_input(self.rl),
            process_rss_probe=lambda: sentinel,  # type: ignore[arg-type, return-value]
            probe_enabled=False,
        )
        proc = ledger.process_row()
        self.assertIsNotNone(proc)
        self.assertIsNone(proc.process_rss_bytes)

    def test_probe_enabled_consumes_injected_probe(self) -> None:
        ledger = self.rl.build_resource_ledger(
            primary_asr=_primary_asr_input(self.rl),
            process_rss_probe=lambda: 256 * 1024 * 1024,
            probe_enabled=True,
        )
        proc = ledger.process_row()
        self.assertIsNotNone(proc)
        self.assertEqual(proc.process_rss_bytes, 256 * 1024 * 1024)

    def test_probe_that_raises_is_swallowed(self) -> None:
        def boom() -> Optional[int]:
            raise RuntimeError("host probe failed")

        ledger = self.rl.build_resource_ledger(
            primary_asr=_primary_asr_input(self.rl),
            process_rss_probe=boom,
            probe_enabled=True,
        )
        proc = ledger.process_row()
        self.assertIsNotNone(proc)
        self.assertIsNone(proc.process_rss_bytes)

    def test_probe_runtime_vram_only_when_loaded(self) -> None:
        # When not loaded, runtime VRAM probe should not fire.
        called: List[str] = []

        def vram_probe(_identifier: str) -> Optional[int]:
            called.append("called")
            return 1900

        # Loaded → probe fires.
        ledger = self.rl.build_resource_ledger(
            primary_asr=_primary_asr_input(
                self.rl, is_loaded=True, runtime_vram_mb=None
            ),
            runtime_vram_probe=vram_probe,
            probe_enabled=True,
        )
        self.assertEqual(called, ["called"])
        self.assertEqual(ledger.primary_asr().runtime_vram_mb, 1900)

        # Not loaded → probe should NOT fire.
        called.clear()
        ledger = self.rl.build_resource_ledger(
            primary_asr=_primary_asr_input(
                self.rl, is_loaded=False, runtime_vram_mb=None
            ),
            runtime_vram_probe=vram_probe,
            probe_enabled=True,
        )
        self.assertEqual(called, [])
        self.assertIsNone(ledger.primary_asr().runtime_vram_mb)

    def test_summary_is_json_serializable(self) -> None:
        ledger = self.rl.build_resource_ledger(
            primary_asr=_primary_asr_input(self.rl),
            auxiliary_model=_aux_input(self.rl),
            plugins=[_plugin_input(self.rl, name="core")],
        )
        rendered = json.dumps(ledger.summary())
        self.assertIsInstance(rendered, str)
        self.assertIn("primary_asr", rendered)
        self.assertIn("auxiliary_model", rendered)

    def test_record_is_immutable(self) -> None:
        ledger = self.rl.build_resource_ledger(
            primary_asr=_primary_asr_input(self.rl),
        )
        record = ledger.primary_asr()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            record.load_state = "mutated"  # type: ignore[misc]

    def test_ledger_preserves_unknown_artifact_bytes(self) -> None:
        ledger = self.rl.build_resource_ledger(
            primary_asr=_primary_asr_input(
                self.rl, artifact_bytes=-1, estimated_vram_mb=None
            ),
        )
        asr = ledger.primary_asr()
        self.assertEqual(asr.artifact_bytes, -1)
        self.assertIsNone(asr.estimated_vram_mb)


# ---------------------------------------------------------------------------
# Auxiliary deny policy
# ---------------------------------------------------------------------------


class AuxiliaryDenyPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rl = _module_under_test()

    def test_offline_policy_denies_all_requests(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(policy="offline_only"),
            requested_endpoint="http://127.0.0.1:8080",
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_TRANSPORT_DENIED)
        self.assertFalse(reason.allowed)
        self.assertIn("offline-only", reason.message)
        self.assertEqual(reason.detail["policy"], "offline_only")

    def test_unknown_policy_is_disabled(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(policy="something_else"),
            requested_endpoint="http://127.0.0.1:8080",
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_POLICY_DISABLED)
        self.assertFalse(reason.allowed)

    def test_loopback_endpoint_in_allowlist_is_allowed(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(
                policy="loopback_only",
                allowed_loopback_endpoints=("http://127.0.0.1:8080",),
            ),
            requested_endpoint="http://127.0.0.1:8080",
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_ALLOWED)
        self.assertTrue(reason.allowed)

    def test_loopback_endpoint_not_in_allowlist(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(
                policy="loopback_only",
                allowed_loopback_endpoints=("http://127.0.0.1:8081",),
            ),
            requested_endpoint="http://127.0.0.1:8080",
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_ENDPOINT_NOT_ALLOWLISTED)
        self.assertFalse(reason.allowed)

    def test_non_loopback_endpoint_is_denied(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(
                policy="loopback_only",
                allowed_loopback_endpoints=("http://127.0.0.1:8080",),
            ),
            requested_endpoint="http://evil.example.com",
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_ENDPOINT_NOT_LOOPBACK)
        self.assertFalse(reason.allowed)
        self.assertEqual(reason.detail["host"], "evil.example.com")

    def test_empty_endpoint_is_denied(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(policy="loopback_only"),
            requested_endpoint="",
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_ENDPOINT_NOT_ALLOWLISTED)
        self.assertFalse(reason.allowed)

    def test_remote_path_is_denied(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(policy="allow_remote"),
            requested_local_path="https://example.com/model.gguf",
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_PATH_NOT_LOCAL)
        self.assertFalse(reason.allowed)

    def test_loopback_only_policy_denies_local_paths(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(
                policy="loopback_only",
                allowed_local_paths=("C:/models/qwen.gguf",),
            ),
            requested_local_path="C:/models/qwen.gguf",
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_PATH_NOT_ALLOWLISTED)
        self.assertFalse(reason.allowed)

    def test_local_path_in_allowlist_is_allowed(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(
                policy="allow_remote",
                allowed_local_paths=("C:/models/qwen.gguf",),
            ),
            requested_local_path="C:/models/qwen.gguf",
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_ALLOWED)
        self.assertTrue(reason.allowed)

    def test_model_identifier_not_in_allowlist(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(
                policy="allow_remote",
                allowed_model_identifiers=("qwen2.5-3b-instruct-q4_k_m.gguf",),
            ),
            requested_model_identifier="different-model.gguf",
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_MODEL_NOT_ALLOWLISTED)
        self.assertFalse(reason.allowed)

    def test_model_identifier_allowlist_match_is_case_insensitive(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(
                policy="allow_remote",
                allowed_model_identifiers=("Qwen2.5-3B-Instruct-Q4_K_M.GGUF",),
            ),
            requested_model_identifier="qwen2.5-3b-instruct-q4_k_m.gguf",
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_ALLOWED)
        self.assertTrue(reason.allowed)

    def test_no_request_without_declared_aux_model_denies(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(policy="loopback_only"),
            auxiliary_declared=None,
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_MODEL_NOT_DECLARED)
        self.assertFalse(reason.allowed)

    def test_no_request_with_declared_aux_model_allows(self) -> None:
        declared = _aux_input(self.rl)
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(policy="loopback_only"),
            auxiliary_declared=declared,
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_ALLOWED)
        self.assertTrue(reason.allowed)

    def test_payload_classification_not_ok_is_denied(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(policy="loopback_only"),
            requested_endpoint="http://127.0.0.1:8080",
            payload_classification="contains_secret",
        )
        self.assertEqual(reason.code, self.rl.AUX_DENY_PAYLOAD_DENIED)
        self.assertFalse(reason.allowed)
        self.assertEqual(
            reason.detail["payload_classification"], "contains_secret"
        )

    def test_deny_reason_has_known_code(self) -> None:
        # Every code we surface must be in the published enum.
        sample_codes = [
            self.rl.AUX_DENY_ALLOWED,
            self.rl.AUX_DENY_POLICY_DISABLED,
            self.rl.AUX_DENY_ENDPOINT_NOT_LOOPBACK,
            self.rl.AUX_DENY_ENDPOINT_NOT_ALLOWLISTED,
            self.rl.AUX_DENY_PATH_NOT_LOCAL,
            self.rl.AUX_DENY_PATH_NOT_ALLOWLISTED,
            self.rl.AUX_DENY_MODEL_NOT_ALLOWLISTED,
            self.rl.AUX_DENY_MODEL_NOT_DECLARED,
            self.rl.AUX_DENY_TRANSPORT_DENIED,
            self.rl.AUX_DENY_PAYLOAD_DENIED,
        ]
        for code in sample_codes:
            self.assertIn(code, self.rl.KNOWN_AUX_DENY_CODES)

    def test_evaluate_auxiliary_deny_never_raises(self) -> None:
        # Garbage input must not raise; it must return a structured deny.
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(policy=None),  # type: ignore[arg-type]
            requested_endpoint="   ",
            requested_local_path="http://example.com",
            requested_model_identifier=None,
        )
        self.assertFalse(reason.allowed)
        self.assertIn(reason.code, self.rl.KNOWN_AUX_DENY_CODES)

    def test_deny_reason_as_dict_is_plain_mapping(self) -> None:
        reason = self.rl.evaluate_auxiliary_deny(
            self.rl.AuxiliaryPolicyConfig(policy="offline_only"),
            requested_endpoint="http://127.0.0.1:8080",
        )
        mapping = reason.as_dict()
        self.assertEqual(
            set(mapping.keys()), {"code", "allowed", "message", "detail"}
        )
        self.assertFalse(mapping["allowed"])
        # JSON-serialisable.
        json.dumps(mapping)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


class RenderHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rl = _module_under_test()
        self.ledger = self.rl.build_resource_ledger(
            primary_asr=_primary_asr_input(self.rl),
            auxiliary_model=_aux_input(self.rl),
            plugins=[
                _plugin_input(self.rl, name="core.commands", origin="builtin"),
                _plugin_input(self.rl, name="user.lexicon", origin="user_directory"),
            ],
        )

    def test_render_table_deterministic(self) -> None:
        text1 = self.rl.render_ledger_table(self.ledger, indent="  ")
        text2 = self.rl.render_ledger_table(self.ledger, indent="  ")
        self.assertEqual(text1, text2)
        self.assertIn("role=primary_asr", text1)
        self.assertIn("role=auxiliary_model", text1)
        self.assertIn("role=plugin", text1)
        self.assertIn("role=process", text1)
        self.assertIn("role=network", text1)
        self.assertIn("role=device", text1)

    def test_render_summary_contains_each_section(self) -> None:
        text = self.rl.render_ledger_summary(self.ledger)
        self.assertIn("primary_asr:", text)
        self.assertIn("auxiliary_model:", text)
        self.assertIn("plugins:", text)
        self.assertIn("process:", text)
        self.assertIn("network_policy:", text)
        self.assertIn("compute_device:", text)


# ---------------------------------------------------------------------------
# Import safety: no subprocess / network / UI / model imports at import
# ---------------------------------------------------------------------------


class ImportSafetyTests(unittest.TestCase):
    def test_module_does_not_import_forbidden_modules_at_import(self) -> None:
        forbidden = {"subprocess", "socket", "urllib", "tkinter", "torch", "transformers"}
        # Reload the module so we observe the actual import-time imports.
        module_name = "drew_local_voice.resource_ledger"
        for mod_name in list(sys.modules):
            if mod_name == module_name or mod_name.startswith(module_name + "."):
                del sys.modules[mod_name]
        # Snapshot sys.modules BEFORE the import so we only flag modules
        # that the import actually adds. Some forbidden names (notably
        # ``urllib`` and ``urllib.parse``) are preloaded by Python's
        # startup site initialization on 3.11+ and are present in
        # ``sys.modules`` before our code runs at all — they are not a
        # transitive import introduced by ``resource_ledger``.
        before = set(sys.modules)
        importlib.import_module(module_name)
        after = set(sys.modules)
        added = after - before
        leaked = sorted(
            name
            for name in added
            if name.split(".")[0] in forbidden
        )
        self.assertEqual(
            leaked,
            [],
            msg=f"resource_ledger transitively imported: {leaked}",
        )


# ---------------------------------------------------------------------------
# Determinism: equal inputs → equal output
# ---------------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rl = _module_under_test()

    def test_equal_inputs_produce_equal_ledgers(self) -> None:
        inputs = dict(
            primary_asr=_primary_asr_input(self.rl),
            auxiliary_model=_aux_input(self.rl),
            plugins=[
                _plugin_input(self.rl, name="a", origin="builtin"),
                _plugin_input(self.rl, name="b", origin="user_directory"),
            ],
        )
        ledger1 = self.rl.build_resource_ledger(**inputs)
        ledger2 = self.rl.build_resource_ledger(**inputs)
        self.assertEqual(ledger1.rows(), ledger2.rows())
        self.assertEqual(
            json.dumps(ledger1.summary(), sort_keys=True),
            json.dumps(ledger2.summary(), sort_keys=True),
        )

    def test_unequal_plugin_order_produces_equal_rows(self) -> None:
        kwargs1 = dict(
            primary_asr=_primary_asr_input(self.rl),
            plugins=[
                _plugin_input(self.rl, name="a", origin="builtin"),
                _plugin_input(self.rl, name="b", origin="user_directory"),
            ],
        )
        kwargs2 = dict(
            primary_asr=_primary_asr_input(self.rl),
            plugins=[
                _plugin_input(self.rl, name="b", origin="user_directory"),
                _plugin_input(self.rl, name="a", origin="builtin"),
            ],
        )
        ledger1 = self.rl.build_resource_ledger(**kwargs1)
        ledger2 = self.rl.build_resource_ledger(**kwargs2)
        self.assertEqual(ledger1.rows(), ledger2.rows())


if __name__ == "__main__":
    unittest.main()
