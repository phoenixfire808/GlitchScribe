import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from drew_local_voice.doctor import (
    CheckResult,
    DoctorReport,
    build_doctor_report,
    idle_unload_due,
    model_appears_loadable,
    normalize_idle_unload_minutes,
    parse_cli_args,
    parse_hotkey_label,
    run_doctor,
    tail_text_file,
)


class IdleUnloadTimerTests(unittest.TestCase):
    def test_normalize_defaults_and_clamps(self):
        self.assertEqual(normalize_idle_unload_minutes(None), 10.0)
        self.assertEqual(normalize_idle_unload_minutes("bad"), 10.0)
        self.assertEqual(normalize_idle_unload_minutes(-3), 0.0)
        self.assertEqual(normalize_idle_unload_minutes(0), 0.0)
        self.assertEqual(normalize_idle_unload_minutes(15), 15.0)
        self.assertEqual(normalize_idle_unload_minutes(99999), 24 * 60)

    def test_due_only_after_idle_window_when_loaded(self):
        self.assertFalse(
            idle_unload_due(
                last_activity_monotonic=100.0,
                now_monotonic=100.0 + 9 * 60,
                idle_unload_minutes=10,
                model_loaded=True,
            )
        )
        self.assertTrue(
            idle_unload_due(
                last_activity_monotonic=100.0,
                now_monotonic=100.0 + 10 * 60,
                idle_unload_minutes=10,
                model_loaded=True,
            )
        )

    def test_not_due_when_disabled_busy_or_unloaded(self):
        self.assertFalse(
            idle_unload_due(
                last_activity_monotonic=0.0,
                now_monotonic=3600.0,
                idle_unload_minutes=0,
                model_loaded=True,
            )
        )
        self.assertFalse(
            idle_unload_due(
                last_activity_monotonic=0.0,
                now_monotonic=3600.0,
                idle_unload_minutes=10,
                model_loaded=False,
            )
        )
        self.assertFalse(
            idle_unload_due(
                last_activity_monotonic=0.0,
                now_monotonic=3600.0,
                idle_unload_minutes=10,
                model_loaded=True,
                recording=True,
            )
        )
        self.assertFalse(
            idle_unload_due(
                last_activity_monotonic=0.0,
                now_monotonic=3600.0,
                idle_unload_minutes=10,
                model_loaded=True,
                busy=True,
            )
        )


class DoctorPureTests(unittest.TestCase):
    def test_hotkey_parse(self):
        ok, _ = parse_hotkey_label("Ctrl + Alt + Space")
        self.assertTrue(ok)
        ok, _ = parse_hotkey_label("F8")
        self.assertTrue(ok)
        ok, detail = parse_hotkey_label("Ctrl+Shift+Z")
        self.assertFalse(ok)
        self.assertIn("unsupported", detail)

    def test_model_path_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "ct2-model"
            model_dir.mkdir()
            ok, detail = model_appears_loadable(
                str(model_dir),
                path_exists=lambda p: Path(p).exists(),
            )
            self.assertTrue(ok)
            self.assertIn("local path", detail)

    def test_report_exit_codes(self):
        report = DoctorReport()
        report.add(CheckResult("a", True, "ok", critical=True))
        self.assertEqual(report.exit_code(), 0)
        report.add(CheckResult("b", False, "nope", critical=True))
        self.assertEqual(report.exit_code(), 1)
        soft = DoctorReport()
        soft.add(CheckResult("c", False, "warn", critical=False))
        self.assertEqual(soft.exit_code(), 0)

    def test_build_report_includes_mic_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(
                '{"model_name":"tiny.en","hotkey":"Ctrl + Alt + Space","device_index":0}',
                encoding="utf-8",
            )
            import json

            data = json.loads(cfg.read_text(encoding="utf-8"))
            devices = [
                {"index": 0, "name": "Mic", "api": "Windows WASAPI", "label": "[0] Mic"}
            ]
            report = build_doctor_report(
                config_path=cfg,
                config_data=data,
                devices=devices,
                default_device_index=0,
                model_name="tiny.en",
                hotkey="Ctrl + Alt + Space",
                cuda_devices=(),
                runtime_events_path=Path(tmp) / "runtime-events.jsonl",
                faster_whisper_importable=True,
                selftest=True,
            )
            text = report.format_text()
            self.assertIn("microphone.devices", text)
            self.assertIn("RESULT:", text)
            names = {c.name for c in report.checks}
            self.assertIn("hotkey.label", names)
            self.assertIn("selftest.mode", names)

    def test_tail_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.txt"
            path.write_text("a\nb\nc\nd\n", encoding="utf-8")
            self.assertEqual(tail_text_file(path, max_lines=2), "c\nd")

    def test_parse_cli_args(self):
        d, s, c = parse_cli_args(["doctor"])
        self.assertTrue(d and c and not s)
        d, s, c = parse_cli_args(["--selftest"])
        self.assertTrue(d and s and c)
        d, s, c = parse_cli_args([])
        self.assertFalse(c)

    def test_run_doctor_with_injected_probes(self):
        stream = io.StringIO()
        code = run_doctor(
            selftest=True,
            stream=stream,
            list_devices=lambda: [
                {"index": 1, "name": "Headset", "api": "WASAPI", "label": "h"}
            ],
            choose_default=lambda devices: 1,
            probe_gpu=lambda: (
                SimpleNamespace(index=0, name="Test GPU", free_vram_mb=8000),
            ),
        )
        out = stream.getvalue()
        self.assertIn("GlitchScribe", out)
        self.assertIn("microphone", out.casefold())
        self.assertIn(code, (0, 1))


if __name__ == "__main__":
    unittest.main()
