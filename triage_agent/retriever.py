"""
Retrieval over past resolved tickets.

Why TF-IDF instead of embeddings: for a small, single-team ticket
corpus (hundreds to low-thousands of tickets), TF-IDF cosine
similarity is fast, requires no API calls or vector DB, and is fully
explainable -- you can point to the exact overlapping terms that drove
a match. Embeddings would matter more at larger scale or for
paraphrase-heavy queries; documented here as a deliberate,
scale-appropriate choice rather than defaulting to whatever's
trendiest.
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TicketRetriever:
    def __init__(self, data_path: str | Path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.tickets: list[dict] = json.load(f)

        self._corpus = [f"{t['title']} {t['description']}" for t in self.tickets]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._corpus)

    def find_similar(self, query_text: str, top_k: int = 2, min_similarity: float = 0.12) -> list[dict]:
        """
        Return up to top_k past tickets similar to query_text, filtered
        by a minimum similarity threshold so weak matches are excluded
        rather than returned as false precedent.
        """
        query_vec = self._vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        ranked = sorted(
            ((score, ticket) for score, ticket in zip(scores, self.tickets)),
            key=lambda pair: pair[0],
            reverse=True,
        )

        return [ticket for score, ticket in ranked[:top_k] if score >= min_similarity]
