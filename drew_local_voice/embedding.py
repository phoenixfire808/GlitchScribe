"""Local embedding provider: Ollama endpoint or HuggingFace sentence-transformers fallback."""

from __future__ import annotations

import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any


@dataclass
class EmbeddingResult:
    """Result of a single embedding call."""

    embedding: list[float]
    model: str
    provider: str  # "ollama" | "huggingface"


class EmbeddingError(Exception):
    """Embedding provider failure."""
    pass


class OllamaEmbeddingProvider:
    """Local embedding via Ollama /api/embeddings endpoint.

    Requires Ollama running at base_url with an embedding model loaded
    (e.g., nomic-embed-text, mxbai-embed-large).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "nomic-embed-text",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> EmbeddingResult:
        """Generate a single embedding vector for text."""
        payload = {"model": self.model, "prompt": text}
        body = __import__("json").dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = __import__("json").loads(resp.read())
                return EmbeddingResult(
                    embedding=data["embeddings"],
                    model=self.model,
                    provider="ollama",
                )
        except urllib.error.URLError as exc:
            raise EmbeddingError(f"Ollama embedding failed: {exc}") from exc
        except TimeoutError as exc:
            raise EmbeddingError(f"Ollama embedding timed out after {self.timeout}s") from exc
        except (KeyError, ValueError) as exc:
            raise EmbeddingError(f"Invalid embedding response: {exc}") from exc

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for a batch of texts."""
        return [self.embed(t) for t in texts]


class HuggingFaceEmbeddingProvider:
    """HuggingFace sentence-transformers fallback — no network call, runs locally."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self._model: Any = None  # loaded lazily

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise EmbeddingError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def embed(self, text: str) -> EmbeddingResult:
        model = self._load()
        embedding = model.encode(text, convert_to_numpy=True).tolist()
        return EmbeddingResult(
            embedding=embedding,
            model=self.model_name,
            provider="huggingface",
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        model = self._load()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [
            EmbeddingResult(embedding=embeddings[i].tolist(), model=self.model_name, provider="huggingface")
            for i in range(len(texts))
        ]


def create_embedding_provider(
    provider: str = "ollama",
    **kwargs: Any,
) -> "OllamaEmbeddingProvider | HuggingFaceEmbeddingProvider":
    """Factory: create the appropriate embedding provider.

    Args:
        provider: "ollama" (default) or "huggingface"
        **kwargs: forwarded to the provider constructor
    """
    if provider == "ollama":
        return OllamaEmbeddingProvider(**kwargs)
    elif provider == "huggingface":
        return HuggingFaceEmbeddingProvider(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider!r}")
