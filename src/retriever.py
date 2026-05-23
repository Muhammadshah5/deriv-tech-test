"""Hybrid retrieval: BM25 (lexical) + sentence-transformer dense embeddings.

Hybrid scoring with min-max normalisation:

    score = alpha * bm25_norm + (1 - alpha) * dense_norm

If `sentence-transformers` is unavailable, we degrade gracefully to BM25 only
so the pipeline still runs from a clean checkout.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


class HybridRetriever:
    def __init__(self, chunks: list[dict], alpha: float = 0.5, model_name: str | None = None):
        self.chunks = chunks
        self.alpha = alpha
        self.texts = [c["text"] for c in chunks]
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])

        # Dense embeddings — optional. Falls back to BM25-only if unavailable.
        self._embedder = None
        self._doc_embeddings = None
        try:
            from sentence_transformers import SentenceTransformer

            name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self._embedder = SentenceTransformer(name)
            self._doc_embeddings = self._embedder.encode(
                self.texts, normalize_embeddings=True, convert_to_numpy=True
            )
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"[retriever] dense embeddings disabled: {exc!r}")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        bm25_scores = np.array(self.bm25.get_scores(_tokenize(query)), dtype=float)

        if self._embedder is not None and self._doc_embeddings is not None:
            q_emb = self._embedder.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            )[0]
            dense_scores = self._doc_embeddings @ q_emb  # cosine since both normalised
            hybrid = self.alpha * _minmax(bm25_scores) + (1 - self.alpha) * _minmax(dense_scores)
        else:
            hybrid = _minmax(bm25_scores)

        order = np.argsort(-hybrid)
        results = []
        for rank, idx in enumerate(order[:top_k], start=1):
            c = self.chunks[idx]
            results.append(
                {
                    "rank": rank,
                    "chunk_id": c["chunk_id"],
                    "doc_title": c["doc_title"],
                    "score": float(hybrid[idx]),
                    "chunk_text": c["text"],
                }
            )
        return results
