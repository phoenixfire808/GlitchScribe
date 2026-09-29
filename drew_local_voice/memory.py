"""Memory scopes: ShortTerm (per-session), LongTerm (SQLite), Entity (relationships), User (preferences).

All data stays local — no cloud, no telemetry.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from drew_local_voice.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Memory store base
# ---------------------------------------------------------------------------


class MemoryError(Exception):
    """Base memory error."""
    pass


class MemoryStore:
    """Base class for scoped memory implementations."""

    def __init__(self, scope: str) -> None:
        self.scope = scope

    def remember(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    def recall(self, key: str) -> str | None:
        raise NotImplementedError

    def forget(self, key: str) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search across stored memory."""
        return []


# ---------------------------------------------------------------------------
# Short-term memory (in-process, session-scoped)
# ---------------------------------------------------------------------------


@dataclass
class ShortTermMemory(MemoryStore):
    """Per-session in-memory store. Cleared on app restart.

    Scope: single GlitchScribe session.
    """

    _store: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)

    def __init__(self) -> None:
        super().__init__("short_term")
        self._store = {}

    def remember(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        self._store[key] = (str(value), dict(metadata or {}))

    def recall(self, key: str) -> str | None:
        entry = self._store.get(key)
        return entry[0] if entry else None

    def forget(self, key: str) -> None:
        self._store.pop(key, None)

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Long-term memory (SQLite, cross-session)
# ---------------------------------------------------------------------------


@dataclass
class LongTermMemory(MemoryStore):
    """Persistent cross-session memory stored in SQLite.

    Scope: all GlitchScribe sessions across restarts.
    Data lives in .glitchscribe/memory.db.
    """

    _conn: sqlite3.Connection = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _vector: Optional[VectorStore] = field(default=None, init=False, repr=False)

    def __init__(
        self,
        db_path: str | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        super().__init__("long_term")
        if db_path is None:
            db_path = str(Path.home() / ".glitchscribe" / "memory.db")

        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._vector = vector_store
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            self._conn.commit()

    def remember(
        self,
        key: str,
        value: str,
        metadata: dict[str, Any] | None = None,
        index_vector: bool = True,
    ) -> None:
        now = time.time()
        meta_json = json.dumps(dict(metadata or {}))
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO memory (key, value, metadata, created_at, updated_at, access_count)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    metadata=excluded.metadata,
                    updated_at=excluded.updated_at,
                    access_count=memory.access_count+1
                """,
                (key, str(value), meta_json, now, now),
            )
            self._conn.commit()

        if index_vector and self._vector:
            try:
                self._vector.add_text(
                    text=f"{key}: {value}",
                    doc_id=f"lt_{key}",
                    metadata={"scope": "long_term", "key": key},
                )
            except Exception:
                pass  # Never let vector indexing break memory

    def recall(self, key: str) -> str | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT value FROM memory WHERE key=?",
                (key,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            # Increment access count
            self._conn.execute(
                "UPDATE memory SET access_count=access_count+1 WHERE key=?",
                (key,),
            )
            self._conn.commit()
            return row[0]

    def forget(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memory WHERE key=?", (key,))
            self._conn.commit()
        if self._vector:
            try:
                self._vector.delete_doc(f"lt_{key}")
            except Exception:
                pass

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search using the vector store, falling back to LIKE."""
        if self._vector:
            try:
                results = self._vector.search(query, top_k=top_k, doc_id=None)
                return [
                    {
                        "key": r["metadata"].get("key", r["id"]),
                        "text": r["text"],
                        "score": r["score"],
                    }
                    for r in results
                ]
            except Exception:
                pass

        # Fallback: LIKE search
        with self._lock:
            cursor = self._conn.execute(
                "SELECT key, value, updated_at FROM memory WHERE value LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (f"%{query}%", top_k),
            )
            return [
                {"key": row[0], "text": row[1], "updated_at": row[2]}
                for row in cursor.fetchall()
            ]

    def keys(self) -> list[str]:
        with self._lock:
            cursor = self._conn.execute("SELECT key FROM memory ORDER BY updated_at DESC")
            return [r[0] for r in cursor.fetchall()]

    def close(self) -> None:
        if self._conn:
            self._conn.close()


# ---------------------------------------------------------------------------
# Entity memory (relationship graph)
# ---------------------------------------------------------------------------


@dataclass
class EntityMemory(MemoryStore):
    """Tracks entities (people, projects, concepts) and their relationships.

    Scope: entire GlitchScribe lifetime.
    Stored in SQLite: .glitchscribe/entity_memory.db
    """

    def __init__(self, db_path: str | None = None) -> None:
        super().__init__("entity")
        if db_path is None:
            db_path = str(Path.home() / ".glitchscribe" / "entity_memory.db")

        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT DEFAULT 'concept',
                description TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at REAL NOT NULL,
                FOREIGN KEY (from_id) REFERENCES entities(id),
                FOREIGN KEY (to_id) REFERENCES entities(id)
            )
        """)
        self._conn.commit()

    def add_entity(
        self,
        name: str,
        entity_type: str = "concept",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        entity_id = uuid.uuid4().hex[:16]
        now = time.time()
        meta_json = json.dumps(dict(metadata or {}))
        self._conn.execute(
            "INSERT OR REPLACE INTO entities (id, name, entity_type, description, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (entity_id, name, entity_type, description, meta_json, now),
        )
        self._conn.commit()
        return entity_id

    def relate(
        self,
        from_name: str,
        to_name: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        """Create or update a relationship between two entities by name."""
        now = time.time()
        # Get or create entities
        from_id = self._get_or_create(from_name)
        to_id = self._get_or_create(to_name)
        rel_id = uuid.uuid4().hex[:16]
        self._conn.execute(
            "INSERT OR REPLACE INTO relations (id, from_id, to_id, relation_type, weight, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (rel_id, from_id, to_id, relation_type, weight, now),
        )
        self._conn.commit()

    def _get_or_create(self, name: str) -> str:
        row = self._conn.execute("SELECT id FROM entities WHERE name=?", (name,)).fetchone()
        if row:
            return row[0]
        return self.add_entity(name)

    def get_related(self, name: str, relation_type: str | None = None) -> list[dict]:
        from_id = self._get_or_create(name)
        if relation_type:
            rows = self._conn.execute(
                "SELECT e.name, e.entity_type, r.relation_type, r.weight FROM relations r JOIN entities e ON e.id=r.to_id WHERE r.from_id=? AND r.relation_type=? ORDER BY r.weight DESC",
                (from_id, relation_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT e.name, e.entity_type, r.relation_type, r.weight FROM relations r JOIN entities e ON e.id=r.to_id WHERE r.from_id=? ORDER BY r.weight DESC",
                (from_id,),
            ).fetchall()
        return [
            {"name": r[0], "type": r[1], "relation": r[2], "weight": r[3]}
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()

    def remember(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        self.add_entity(key, entity_type="concept", description=value, metadata=metadata)

    def recall(self, key: str) -> str | None:
        row = self._conn.execute("SELECT description FROM entities WHERE name=?", (key,)).fetchone()
        return row[0] if row else None

    def forget(self, key: str) -> None:
        entity_id = self._get_or_create(key)
        self._conn.execute("DELETE FROM relations WHERE from_id=? OR to_id=?", (entity_id, entity_id))
        self._conn.execute("DELETE FROM entities WHERE id=?", (entity_id,))
        self._conn.commit()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        rows = self._conn.execute(
            "SELECT name, entity_type, description FROM entities WHERE name LIKE ? OR description LIKE ? LIMIT ?",
            (f"%{query}%", f"%{query}%", top_k),
        ).fetchall()
        return [{"name": r[0], "type": r[1], "description": r[2]} for r in rows]


# ---------------------------------------------------------------------------
# User preferences memory
# ---------------------------------------------------------------------------


@dataclass
class UserMemory(MemoryStore):
    """Stores user preferences, hotkey configs, and UI settings.

    Scope: per-user, across sessions.
    """

    _store: dict[str, str] = field(default_factory=dict)
    _path: Path = field(default_factory=Path.home() / ".glitchscribe" / "user_memory.json")

    def __init__(self, path: str | None = None) -> None:
        super().__init__("user")
        if path:
            self._path = Path(path)
        else:
            self._path = Path.home() / ".glitchscribe" / "user_memory.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._store = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._store = {}

    def _save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=2, ensure_ascii=False)

    def remember(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        self._store[key] = str(value)
        self._save()

    def recall(self, key: str) -> str | None:
        return self._store.get(key)

    def forget(self, key: str) -> None:
        self._store.pop(key, None)
        self._save()

    def all(self) -> dict[str, str]:
        return dict(self._store)


# ---------------------------------------------------------------------------
# Composite memory — all scopes in one
# ---------------------------------------------------------------------------


@dataclass
class CompositeMemory:
    """Unified memory facade providing all four scopes."""

    short_term: ShortTermMemory = field(default_factory=ShortTermMemory)
    long_term: LongTermMemory = field(default_factory=lambda: LongTermMemory())
    entity: EntityMemory = field(default_factory=EntityMemory)
    user: UserMemory = field(default_factory=UserMemory)

    def remember(
        self,
        key: str,
        value: str,
        scopes: tuple[str, ...] = ("short_term", "long_term"),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for scope_name in scopes:
            store = getattr(self, scope_name, None)
            if store is not None:
                try:
                    store.remember(key, value, metadata=metadata)
                except Exception:
                    pass

    def recall(
        self,
        key: str,
        scopes: tuple[str, ...] = ("short_term", "long_term", "user"),
    ) -> str | None:
        for scope_name in scopes:
            store = getattr(self, scope_name, None)
            if store is not None:
                value = store.recall(key)
                if value is not None:
                    return value
        return None

    def search(self, query: str, scopes: tuple[str, ...] = ("long_term", "entity")) -> list[dict]:
        results = []
        for scope_name in scopes:
            store = getattr(self, scope_name, None)
            if store is not None:
                try:
                    results.extend(store.search(query))
                except Exception:
                    pass
        return results

    def close(self) -> None:
        try:
            self.long_term.close()
        except Exception:
            pass
        try:
            self.entity.close()
        except Exception:
            pass
