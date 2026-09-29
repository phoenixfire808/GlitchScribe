from __future__ import annotations

import unittest
import importlib.util


class ModelLabTests(unittest.TestCase):
    def test_rtx2070_candidate_plan_fails_closed_on_live_headroom(self):
        self.assertIsNotNone(importlib.util.find_spec("drew_local_voice.model_lab"))
        import drew_local_voice.model_lab as model_lab

        report = model_lab.evaluate_speech_backends(
            gpu=model_lab.GPULabSnapshot(
                index=1,
                name="NVIDIA GeForce RTX 2070 SUPER",
                compute_capability="7.5",
                total_vram_mb=8192,
                used_vram_mb=7530,
                free_vram_mb=460,
            ),
            runtime=model_lab.RuntimeCapabilities(
                ctranslate2_version="4.8.1",
                ctranslate2_cuda_compute_types=(
                    "float16",
                    "float32",
                    "int8",
                    "int8_float16",
                    "int8_float32",
                ),
                onnxruntime_version="1.28.0",
                onnxruntime_providers=(
                    "AzureExecutionProvider",
                    "CPUExecutionProvider",
                ),
                sherpa_onnx_version="",
                nemo_version="",
            ),
        )

        by_id = {candidate.candidate_id: candidate for candidate in report.candidates}
        base = by_id["faster-whisper-base-en"]
        self.assertTrue(base.hardware_compatible)
        self.assertTrue(base.runtime_ready)
        self.assertEqual(base.status, "blocked_headroom")
        self.assertEqual(base.required_free_vram_mb, 2560)

        parakeet = by_id["sherpa-parakeet-tdt-0.6b-v3-int8"]
        self.assertTrue(parakeet.hardware_compatible)
        self.assertFalse(parakeet.runtime_ready)
        self.assertEqual(parakeet.status, "blocked_headroom")
        self.assertIn("sherpa-onnx runtime is not installed", parakeet.blockers)
        self.assertIn("CUDAExecutionProvider is unavailable", parakeet.blockers)
        self.assertFalse(report.any_model_load_allowed)
        self.assertEqual(report.decision, "no_go_preserve_existing_workloads")

    def test_candidate_becomes_eligible_only_with_runtime_and_headroom(self):
        self.assertIsNotNone(importlib.util.find_spec("drew_local_voice.model_lab"))
        import drew_local_voice.model_lab as model_lab

        report = model_lab.evaluate_speech_backends(
            gpu=model_lab.GPULabSnapshot(
                index=1,
                name="NVIDIA GeForce RTX 2070 SUPER",
                compute_capability="7.5",
                total_vram_mb=8192,
                used_vram_mb=2048,
                free_vram_mb=6144,
            ),
            runtime=model_lab.RuntimeCapabilities(
                ctranslate2_version="4.8.1",
                ctranslate2_cuda_compute_types=("int8_float16",),
                onnxruntime_version="1.28.0",
                onnxruntime_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                sherpa_onnx_version="1.12.40",
                nemo_version="",
            ),
        )

        by_id = {candidate.candidate_id: candidate for candidate in report.candidates}
        self.assertEqual(by_id["faster-whisper-base-en"].status, "eligible")
        self.assertEqual(
            by_id["sherpa-parakeet-tdt-0.6b-v3-int8"].status,
            "eligible",
        )
        self.assertTrue(report.any_model_load_allowed)
        self.assertEqual(report.decision, "eligible_candidates_require_explicit_benchmark")


if __name__ == "__main__":
    unittest.main()
