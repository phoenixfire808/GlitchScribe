"""Local vector store using LanceDB — stored under .glitchscribe/vector_db/."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from drew_local_voice.embedding import EmbeddingResult, create_embedding_provider


@dataclass
class VectorEntry:
    """One document/chunk stored in the vector database."""

    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""


class VectorStore:
    """LanceDB-backed local vector store.

    Stores documents locally under .glitchscribe/vector_db/ — no external server needed.
    """

    def __init__(
        self,
        db_path: str | None = None,
        embedding_provider: str = "ollama",
        embedding_model: str = "nomic-embed-text",
        table_name: str = "documents",
    ) -> None:
        if db_path is None:
            base = Path.home() / ".glitchscribe"
            db_path = str(base / "vector_db")

        self.db_path = db_path
        self._provider = create_embedding_provider(
            provider=embedding_provider,
            model=embedding_model,
        )
        self._table_name = table_name
        self._table: Any = None  # LanceDB table — loaded lazily

    def _get_table(self) -> Any:
        """Lazily open or create the LanceDB table."""
        if self._table is not None:
            return self._table

        try:
            import lancedb
        except ImportError:
            raise RuntimeError(
                "lancedb not installed. Install with: pip install lancedb"
            )

        os.makedirs(self.db_path, exist_ok=True)
        db = lancedb.connect(self.db_path)

        import pyarrow as pa

        try:
            table = db.open_table(self._table_name)
        except Exception:
            # Create the table with a generic 768-dim vector schema
            # Actual dimension is detected and corrected on first insert
            schema_def = pa.schema([
                ("id", pa.string()),
                ("doc_id", pa.string()),
                ("text", pa.string()),
                ("vector", pa.list_(pa.float32(), 768)),
                ("metadata", pa.string()),
            ])
            db.create_table(self._table_name, schema=schema_def)
            table = db.open_table(self._table_name)

        self._table = table
        return self._table

    def add_text(
        self,
        text: str,
        doc_id: str = "",
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 512,
    ) -> list[str]:
        """Add a document, chunked automatically. Returns list of entry IDs."""
        import json

        # Chunk the text
        chunks = self._chunk_text(text, chunk_size)

        # Generate embeddings for all chunks
        embeddings = self._provider.embed_batch(chunks)

        # Build entries
        entries: list[VectorEntry] = []
        for chunk, emb in zip(chunks, embeddings):
            entry_id = uuid.uuid4().hex[:16]
            entries.append(VectorEntry(
                id=entry_id,
                text=chunk,
                embedding=emb.embedding,
                metadata=metadata or {},
                doc_id=doc_id,
            ))

        # Write to LanceDB
        table = self._get_table()

        # Detect actual embedding dimension from first entry
        dim = len(entries[0].embedding) if entries else 768

        schema = pa.schema([
            ("id", pa.string()),
            ("doc_id", pa.string()),
            ("text", pa.string()),
            ("vector", pa.list_(pa.float32(), dim)),
            ("metadata", pa.string()),
        ])

        rows = [{
            "id": e.id,
            "doc_id": e.doc_id,
            "text": e.text,
            "vector": e.embedding,
            "metadata": json.dumps(e.metadata),
        } for e in entries]

        table.add(__import__("pyarrow").table(rows, schema=schema))
        return [e.id for e in entries]

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_id: str | None = None,
        recency_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ) -> list[dict]:
        """Search for most similar chunks.

        Args:
            query: text to search for
            top_k: number of results to return
            doc_id: optional filter by document ID
            recency_weight: blend toward recency (0.0–1.0) — currently a hint for future ranking
            semantic_weight: blend toward semantic similarity (0.0–1.0)

        Returns list of result dicts with {id, text, score, metadata}.
        """
        table = self._get_table()
        emb = self._provider.embed(query)

        try:
            # LanceDB Python API
            results = (
                table.search(emb.embedding)
                .limit(top_k)
                .to_list()
            )
        except Exception:
            # Fallback: scan + cosine
            return self._naive_search(emb.embedding, top_k, doc_id)

        processed = []
        for r in results:
            if doc_id and r.get("doc_id") != doc_id:
                continue
            processed.append({
                "id": r.get("id", ""),
                "text": r.get("text", ""),
                "score": r.get("score", 0.0),
                "metadata": {},
            })
        return processed

    def _naive_search(
        self,
        query_vector: list[float],
        top_k: int,
        doc_id: str | None = None,
    ) -> list[dict]:
        """Naive cosine similarity fallback when LanceDB query fails."""
        table = self._get_table()
        try:
            all_rows = table.to_list()
        except Exception:
            return []

        def cosine(a: list[float], b: list[float]) -> float:
            import math
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        scored = []
        for row in all_rows:
            if doc_id and row.get("doc_id") != doc_id:
                continue
            vec = row.get("vector", [])
            score = cosine(query_vector, vec) if vec else 0.0
            scored.append({
                "id": row.get("id", ""),
                "text": row.get("text", ""),
                "score": score,
                "metadata": {},
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """Simple fixed-size chunking with overlap."""
        words = text.split()
        chunks: list[str] = []
        for i in range(0, len(words), chunk_size // 5):  # ~5 chars per word estimate
            chunk = " ".join(words[i:i + chunk_size // 5])
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks if chunks else [text[:chunk_size]]

    def count(self) -> int:
        """Return total number of entries in the store."""
        try:
            table = self._get_table()
            return len(table.to_list())
        except Exception:
            return 0

    def delete_doc(self, doc_id: str) -> int:
        """Delete all entries for a document. Returns count of deleted entries."""
        try:
            table = self._get_table()
            # LanceDB delete by query
            table.delete(f'doc_id = "{doc_id}"')
            return 0  # deletion count not easily available
        except Exception:
            return 0
