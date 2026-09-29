"""Path-based, isolated tests for the local update-request queue.

These tests cover the durable, atomic, reviewable, non-executable behavior
of ``drew_local_voice.update_requests.UpdateRequestStore``. They use
``tempfile.TemporaryDirectory`` so nothing touches the real APP_DIR, Git, or
the network.
"""

from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path


def _load_module():
    return importlib.import_module("drew_local_voice.update_requests")


class UpdateRequestStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_module()
        self._tempdir = tempfile.TemporaryDirectory(prefix="upd-req-")
        self.addCleanup(self._tempdir.cleanup)
        self.folder = Path(self._tempdir.name)
        self.path = self.folder / "update_requests.json"

    # ----- import contract -----

    def test_module_exposes_required_api(self):
        self.assertTrue(hasattr(self.module, "UpdateRequestStore"))
        self.assertTrue(hasattr(self.module, "UpdateRequest"))
        self.assertTrue(hasattr(self.module, "UpdateRequestError"))

    def test_store_import_has_no_side_effects(self):
        # Importing the module a second time must not create files on disk.
        sentinel = self.folder / "sentinel.json"
        self.assertFalse(sentinel.exists())
        importlib.import_module("drew_local_voice.update_requests")
        self.assertFalse(sentinel.exists())

    # ----- submit() / pending() -----

    def test_submit_returns_pending_request_with_unique_id(self):
        store = self.module.UpdateRequestStore(self.path)
        request = store.submit("  add dark-mode toggle  ")
        self.assertEqual(request.status, "pending")
        self.assertTrue(request.request_id)
        self.assertEqual(request.text, "add dark-mode toggle")
        self.assertTrue(request.created_at)
        self.assertTrue(request.is_pending)
        self.assertFalse(request.is_ready)
        # File should now exist and be valid JSON.
        self.assertTrue(self.path.is_file())

    def test_submit_normalizes_line_endings_and_strips_control_chars(self):
        store = self.module.UpdateRequestStore(self.path)
        # Inject CRLF, lone CR, NUL byte, and BOM-ish control char; the store
        # must scrub these before persisting.
        request = store.submit(
            "line one\r\nline two\rline three\x00line four\x07"
        )
        # CRLF and lone CR normalize to LF; \x00 and \x07 are stripped.
        self.assertEqual(request.text, "line one\nline two\nline threeline four")
        # Verify the on-disk file too.
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], 1)
        self.assertEqual(len(payload["requests"]), 1)
        self.assertEqual(
            payload["requests"][0]["text"],
            "line one\nline two\nline threeline four",
        )

    def test_submit_rejects_empty_or_whitespace_only(self):
        store = self.module.UpdateRequestStore(self.path)
        for bad in ("", "   ", "\n\n", "\t  \n"):
            with self.assertRaises(self.module.UpdateRequestError):
                store.submit(bad)
        # No file should have been written after any failed submission.
        self.assertFalse(self.path.exists())

    def test_submit_rejects_overlong_text(self):
        store = self.module.UpdateRequestStore(self.path)
        with self.assertRaises(self.module.UpdateRequestError):
            store.submit("a" * (self.module.MAX_TEXT_CHARS + 1))

    def test_submit_rejects_non_string_text(self):
        store = self.module.UpdateRequestStore(self.path)
        with self.assertRaises(self.module.UpdateRequestError):
            store.submit(None)  # type: ignore[arg-type]

    def test_pending_returns_pending_only_in_insertion_order(self):
        store = self.module.UpdateRequestStore(self.path)
        first = store.submit("first")
        second = store.submit("second")
        third = store.submit("third")
        store.mark_ready(second.request_id)
        pending = store.pending()
        self.assertEqual(
            [r.request_id for r in pending],
            [first.request_id, third.request_id],
        )
        self.assertEqual(pending[0].text, "first")
        self.assertEqual(pending[1].text, "third")

    def test_pending_on_missing_file_is_empty_tuple(self):
        store = self.module.UpdateRequestStore(self.path)
        self.assertEqual(store.pending(), ())

    # ----- mark_ready() -----

    def test_mark_ready_flips_status_and_persists(self):
        store = self.module.UpdateRequestStore(self.path)
        request = store.submit("queue for review")
        updated = store.mark_ready(request.request_id)
        self.assertEqual(updated.status, "ready")
        self.assertEqual(updated.request_id, request.request_id)
        self.assertEqual(updated.text, request.text)
        # Reload from disk via a fresh store to confirm persistence.
        reloaded = self.module.UpdateRequestStore(self.path)
        pending = reloaded.pending()
        self.assertEqual(pending, ())
        all_requests = reloaded.all()
        self.assertEqual(len(all_requests), 1)
        self.assertEqual(all_requests[0].status, "ready")

    def test_mark_ready_unknown_id_raises(self):
        store = self.module.UpdateRequestStore(self.path)
        with self.assertRaises(self.module.UpdateRequestError):
            store.mark_ready("does-not-exist")

    def test_mark_ready_already_ready_raises(self):
        store = self.module.UpdateRequestStore(self.path)
        request = store.submit("only once")
        store.mark_ready(request.request_id)
        with self.assertRaises(self.module.UpdateRequestError):
            store.mark_ready(request.request_id)

    def test_mark_ready_rejects_empty_id(self):
        store = self.module.UpdateRequestStore(self.path)
        with self.assertRaises(self.module.UpdateRequestError):
            store.mark_ready("")

    # ----- clear() -----

    def test_clear_removes_request_and_persists(self):
        store = self.module.UpdateRequestStore(self.path)
        keep = store.submit("keep me")
        drop = store.submit("drop me")
        store.clear(drop.request_id)
        reloaded = self.module.UpdateRequestStore(self.path)
        ids = [r.request_id for r in reloaded.all()]
        self.assertEqual(ids, [keep.request_id])

    def test_clear_unknown_id_raises(self):
        store = self.module.UpdateRequestStore(self.path)
        store.submit("something")
        with self.assertRaises(self.module.UpdateRequestError):
            store.clear("missing-id")

    # ----- atomic write -----

    def test_atomic_write_leaves_no_temp_files_on_success(self):
        store = self.module.UpdateRequestStore(self.path)
        store.submit("atomic please")
        leftovers = [
            p for p in self.folder.iterdir()
            if p.name != self.path.name and p.suffix == ".tmp"
        ]
        self.assertEqual(leftovers, [], f"unexpected temp files: {leftovers}")

    def test_corrupt_store_file_is_rejected_not_silently_dropped(self):
        self.path.write_text("{ this is not json", encoding="utf-8")
        store = self.module.UpdateRequestStore(self.path)
        with self.assertRaises(self.module.UpdateRequestError):
            store.pending()

    def test_unsupported_schema_is_rejected(self):
        self.path.write_text(
            json.dumps({"schema": 999, "requests": []}), encoding="utf-8"
        )
        store = self.module.UpdateRequestStore(self.path)
        with self.assertRaises(self.module.UpdateRequestError):
            store.pending()

    # ----- export_markdown() -----

    def test_export_markdown_writes_non_executable_handoff(self):
        store = self.module.UpdateRequestStore(self.path)
        request = store.submit("review-only intent body")
        out_path = self.folder / "review_handoff.md"
        returned = store.export_markdown(out_path)
        self.assertEqual(returned, out_path)
        self.assertTrue(out_path.is_file())

        text = out_path.read_text(encoding="utf-8")
        # Must clearly label itself as non-executable and as a review handoff.
        self.assertIn("REVIEW HANDOFF", text)
        self.assertIn("NOT EXECUTABLE", text)
        self.assertIn("do not execute", text.lower())
        # The queued body must appear inside a fenced code block, not as a
        # top-level instruction.
        self.assertIn("```text", text)
        self.assertIn("review-only intent body", text)
        # The request id should be referenced so a reviewer can trace it.
        self.assertIn(request.request_id, text)

    def test_export_markdown_handles_empty_queue(self):
        store = self.module.UpdateRequestStore(self.path)
        out_path = self.folder / "empty_handoff.md"
        store.export_markdown(out_path)
        text = out_path.read_text(encoding="utf-8")
        self.assertIn("REVIEW HANDOFF", text)
        self.assertIn("Pending requests: **0**", text)
        self.assertIn("No pending requests.", text)

    def test_export_markdown_atomic_no_temp_leftovers(self):
        store = self.module.UpdateRequestStore(self.path)
        store.submit("x")
        out_path = self.folder / "atomic_handoff.md"
        store.export_markdown(out_path)
        leftovers = [
            p for p in self.folder.iterdir()
            if p.name != out_path.name and p.suffix == ".tmp"
        ]
        self.assertEqual(leftovers, [], f"unexpected temp files: {leftovers}")

    # ----- safety / non-executable -----

    def test_store_never_writes_outside_target_directory(self):
        outside = self.folder.parent / "outside.json"
        store = self.module.UpdateRequestStore(outside)
        store.submit("contained")
        self.assertTrue(outside.is_file())
        # No sibling files should appear in the parent directory from our store.
        siblings = [
            p for p in self.folder.parent.iterdir()
            if p.name == "outside.json"
        ]
        self.assertEqual(len(siblings), 1)
        outside.unlink(missing_ok=True)

    def test_persisted_record_contains_only_safe_fields(self):
        store = self.module.UpdateRequestStore(self.path)
        store.submit("safe payload only")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        record = payload["requests"][0]
        allowed = {"request_id", "created_at", "text", "status"}
        self.assertEqual(set(record.keys()), allowed)


class UpdateRequestDataclassTests(unittest.TestCase):
    def test_invalid_status_rejected(self):
        m = _load_module()
        with self.assertRaises(m.UpdateRequestError):
            m.UpdateRequest(
                request_id="abc",
                created_at="2026-08-05T00:00:00+00:00",
                text="x",
                status="bogus",
            )

    def test_empty_request_id_rejected(self):
        m = _load_module()
        with self.assertRaises(m.UpdateRequestError):
            m.UpdateRequest(
                request_id="",
                created_at="2026-08-05T00:00:00+00:00",
                text="x",
                status="pending",
            )


if __name__ == "__main__":
    unittest.main()
