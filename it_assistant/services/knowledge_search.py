"""BM25 keyword search over the local IT knowledge base.

BM25 is used deliberately instead of embeddings: the knowledge base is small, the
ranking is explainable to an evaluator, and it needs no extra model or API key.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from it_assistant.data.models import KnowledgeArticle
from it_assistant.services.text import tokenize


@dataclass(frozen=True)
class SearchHit:
    article: KnowledgeArticle
    score: float


class KnowledgeSearchIndex:
    def __init__(self, articles: list[KnowledgeArticle], k1: float = 1.5, b: float = 0.75) -> None:
        self._articles = articles
        self._k1 = k1
        self._b = b
        # Title and tags are repeated so that they weigh more than body text.
        self._docs = [
            tokenize(f"{a.title} {a.title} {' '.join(a.tags)} {' '.join(a.tags)} {a.category} {a.content}")
            for a in articles
        ]
        self._doc_freqs = [Counter(doc) for doc in self._docs]
        self._avg_len = (sum(len(d) for d in self._docs) / len(self._docs)) if self._docs else 0.0
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(set(doc))
        n = len(self._docs)
        self._idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[SearchHit]:
        terms = tokenize(query)
        if not terms or not self._docs:
            return []
        hits = []
        for article, doc, freqs in zip(self._articles, self._docs, self._doc_freqs):
            score = 0.0
            for term in terms:
                tf = freqs.get(term)
                if not tf:
                    continue
                norm = tf * (self._k1 + 1) / (
                    tf + self._k1 * (1 - self._b + self._b * len(doc) / self._avg_len)
                )
                score += self._idf.get(term, 0.0) * norm
            if score > min_score:
                hits.append(SearchHit(article=article, score=round(score, 3)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
