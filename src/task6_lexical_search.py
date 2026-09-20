"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


CORPUS: list[dict] = []


import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Tách từ tiếng Việt/tiếng Anh thành các token chữ-số (lowercase)."""
    return re.findall(r"[\w]+", text.lower())


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    return _BM25([_tokenize(item["content"]) for item in corpus])


class _BM25:
    """BM25 tự cài (idf luôn dương), không phụ thuộc thư viện ngoài."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_len = [len(document) for document in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size else 0.0
        self.doc_freqs = [Counter(document) for document in corpus]
        df: Counter[str] = Counter()
        for frequencies in self.doc_freqs:
            df.update(frequencies.keys())
        self.idf = {
            term: math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for index in range(self.corpus_size):
            frequencies = self.doc_freqs[index]
            for term in query:
                if term not in self.idf:
                    continue
                freq = frequencies.get(term, 0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_len[index] / self.avgdl
                )
                if denominator:
                    scores[index] += self.idf[term] * numerator / denominator
        return scores


_BM25_CACHE: tuple[list[dict], _BM25] | None = None


def load_corpus() -> list[dict]:
    """Nạp CORPUS từ cùng nguồn chunks của Task 4 (lazy, chỉ đọc một lần).

    Task 6 phải chạy trên **cùng corpus chunks với Task 5**; nếu để CORPUS rỗng
    thì lexical_search luôn trả [] và "hybrid" thực chất chỉ còn dense.
    Test có thể monkeypatch CORPUS để tiêm corpus giả — khi đó không nạp lại.
    """
    global CORPUS
    if not CORPUS:
        from .task4_chunking_indexing import (  # noqa: PLC0415 - tránh vòng import
            chunk_documents,
            load_documents,
        )

        CORPUS = chunk_documents(load_documents())
    return CORPUS


def _get_bm25(corpus: list[dict]) -> _BM25:
    """Trả BM25 index đã cache, chỉ build lại khi corpus đổi.

    Tránh tokenize lại toàn bộ corpus ở mỗi query (evaluation chạy 18 câu hỏi
    x 2 config sẽ rất chậm nếu build mỗi lần).
    """
    global _BM25_CACHE
    if _BM25_CACHE is None or _BM25_CACHE[0] is not corpus:
        _BM25_CACHE = (corpus, build_bm25_index(corpus))
    return _BM25_CACHE[1]


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    corpus = load_corpus()
    if not corpus:
        return []
    bm25 = _get_bm25(corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )
    results = []
    for index in ranked_indices:
        if len(results) >= top_k:
            break
        if scores[index] <= 0:
            continue
        item = corpus[index]
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": float(scores[index]),
                "metadata": item["metadata"],
                "retrieval_method": "bm25",
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
