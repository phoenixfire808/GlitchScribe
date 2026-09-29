"""Local, side-effect-free coding workflow primitives for GlitchScribe."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


def _normalized_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold()).rstrip(".,!?;:")


@dataclass(frozen=True)
class Snippet:
    name: str
    content: str
    description: str = ""

    def __post_init__(self) -> None:
        if not _normalized_name(self.name):
            raise ValueError("snippet name cannot be empty")
        if not self.content:
            raise ValueError("snippet content cannot be empty")
        if len(self.name) > 120 or len(self.content) > 20_000:
            raise ValueError("snippet exceeds the local size limit")

    @property
    def normalized_name(self) -> str:
        return _normalized_name(self.name)


class SnippetStore:
    """Versioned local JSON store with atomic writes and deterministic lookup."""

    def __init__(self, path: Path | str, *, defaults: Iterable[Snippet] = ()) -> None:
        self.path = Path(path)
        self.defaults = tuple(defaults)

    def initialize(self) -> None:
        if not self.path.exists():
            self.save(self.defaults)

    def load(self) -> tuple[Snippet, ...]:
        if not self.path.exists():
            return self.defaults
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported snippet-store format")
        records = payload.get("snippets")
        if not isinstance(records, list):
            raise ValueError("snippet-store records must be a list")
        return tuple(
            Snippet(
                name=str(record.get("name", "")),
                content=str(record.get("content", "")),
                description=str(record.get("description", "")),
            )
            for record in records
            if isinstance(record, dict)
        )

    def save(self, snippets: Iterable[Snippet]) -> None:
        records = tuple(snippets)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "snippets": [asdict(snippet) for snippet in records],
        }
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{self.path.stem}-",
                suffix=".tmp",
                dir=self.path.parent,
                delete=False,
            ) as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = Path(handle.name)
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def resolve(self, name: str) -> Optional[Snippet]:
        key = _normalized_name(name)
        return next(
            (snippet for snippet in self.load() if snippet.normalized_name == key),
            None,
        )

    def upsert(self, snippet: Snippet) -> None:
        records = list(self.load())
        for index, current in enumerate(records):
            if current.normalized_name == snippet.normalized_name:
                records[index] = snippet
                break
        else:
            records.append(snippet)
        self.save(records)


def built_in_snippets() -> tuple[Snippet, ...]:
    return (
        Snippet(
            name="Python Main",
            content='if __name__ == "__main__":\n    main()',
            description="Python module entry-point guard.",
        ),
        Snippet(
            name="TODO",
            content="TODO: ",
            description="Short local implementation reminder.",
        ),
        Snippet(
            name="Markdown Code Fence",
            content="```text\n\n```",
            description="A neutral fenced Markdown block.",
        ),
    )


@dataclass(frozen=True)
class CompiledPrompt:
    template_id: str
    intent: str
    text: str


class PromptCompiler:
    """Deterministically turns spoken intent into a reviewable coding brief."""

    template_id = "implementation_brief"

    def compile(self, intent: str) -> CompiledPrompt:
        normalized_intent = re.sub(r"\s+", " ", str(intent).strip())
        if not normalized_intent:
            raise ValueError("prompt intent cannot be empty")
        if len(normalized_intent) > 4_000:
            raise ValueError("prompt intent exceeds the local size limit")
        text = (
            "IMPLEMENTATION BRIEF\n\n"
            "GOAL\n"
            f"{normalized_intent}\n\n"
            "WORKING RULES\n"
            "- Inspect the relevant files and trace existing call paths before editing.\n"
            "- Preserve existing behavior outside the requested scope.\n"
            "- Add focused regression coverage for every behavior change.\n"
            "- Prefer a small integrated implementation over disconnected scaffolding.\n"
            "- Do not execute external or public actions without explicit approval.\n\n"
            "ACCEPTANCE\n"
            "- Exercise the implementation through its real runtime path.\n"
            "- Run focused verification and report exact pass, fail, or blocked evidence.\n"
            "- Retain user control over destructive, external, credentialed, or public actions."
        )
        return CompiledPrompt(self.template_id, normalized_intent, text)
