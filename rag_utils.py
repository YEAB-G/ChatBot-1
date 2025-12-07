from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class IndexedChunk:
    id: int
    source_name: str
    text: str


class RAGIndex:
    """Simple in-memory vector store for RAG."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self.chunks: List[IndexedChunk] = []
        self.embeddings: np.ndarray | None = None

    def _ensure_model(self) -> None:
        if self.model is None:
            # Download model on first use
            self.model = SentenceTransformer(self.model_name)

    def clear(self) -> None:
        self.chunks = []
        self.embeddings = None

    def add_document(
        self,
        text: str,
        source_name: str,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ) -> None:
        """Split text into chunks, embed them, and store."""
        self._ensure_model()
        new_chunks_text = chunk_text(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        start_id = len(self.chunks)
        new_chunks: List[IndexedChunk] = []
        for i, chunk in enumerate(new_chunks_text):
            new_chunks.append(
                IndexedChunk(
                    id=start_id + i,
                    source_name=source_name,
                    text=chunk,
                )
            )

        if not new_chunks:
            return

        texts_to_embed = [c.text for c in new_chunks]
        new_embs = self.model.encode(
            texts_to_embed,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if self.embeddings is None:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])

        self.chunks.extend(new_chunks)

    def is_empty(self) -> bool:
        return self.embeddings is None or len(self.chunks) == 0

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k most similar chunks for the query."""
        if self.is_empty():
            return []

        self._ensure_model()
        query_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ query_emb  # cosine similarity

        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            ch = self.chunks[int(idx)]
            score = float(scores[int(idx)])
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "source_name": ch.source_name,
                    "text": ch.text,
                }
            )
        return results


def chunk_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> List[str]:
    """Simple character-based chunking with overlap."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunk = cleaned[start:end]
        chunks.append(chunk.strip())
        start += max(1, chunk_size - chunk_overlap)

    return [c for c in chunks if c]
