"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau: cosine
similarity nằm trong [0, 1] và phản ánh độ gần ngữ nghĩa, còn RRF score chỉ
phản ánh thứ hạng (với k=60 và 2 list thì trần chỉ khoảng 0.033).
"""

import os

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


load_dotenv(override=True)


def _threshold_from_env(default: float = 0.3) -> float:
    """Đọc SCORE_THRESHOLD từ .env, giữ default khi chưa hiệu chỉnh."""
    raw = os.getenv("SCORE_THRESHOLD", "").strip()
    try:
        return float(raw)
    except ValueError:
        return default


# Hiệu chỉnh bằng query in-domain và out-of-domain — xem RESULT.md (calibration).
SCORE_THRESHOLD = _threshold_from_env()
DEFAULT_TOP_K = 5

# Hằng số RRF của Task 7: RRF(d) = sum(1 / (RRF_K + rank)), rank bắt đầu từ 1.
# Pipeline dùng đúng default k=60 của rerank_rrf (không override).
RRF_K = 60


def _normalize(results: list[dict]) -> list[dict]:
    """Bù lại field metadata bị vectorstore làm rụng.

    ChromaDB không lưu được giá trị None nên ``url`` biến mất khỏi metadata sau
    khi round-trip, làm kết quả vi phạm contract (``metadata.url`` phải là str
    hoặc None). Khôi phục ở biên của pipeline để Task 10 luôn nhận đúng schema.
    """
    for item in results:
        metadata = item.get("metadata")
        if isinstance(metadata, dict) and "url" not in metadata:
            metadata["url"] = None
    return results


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    # Lấy dư gấp đôi để RRF có đủ ứng viên giao nhau giữa hai ranked list.
    try:
        dense = _normalize(semantic_search(query, top_k=top_k * 2))
    except Exception:  # noqa: BLE001 - vectorstore chưa index thì vẫn chạy tiếp
        dense = []

    try:
        sparse = _normalize(lexical_search(query, top_k=top_k * 2))
    except Exception:  # noqa: BLE001 - BM25 corpus rỗng không được chặn pipeline
        sparse = []

    if use_reranking:
        # RRF chỉ được fuse đúng một lần trong toàn pipeline, với k=RRF_K (60)
        # là default của rerank_rrf.
        hybrid = rerank_rrf([dense, sparse], top_k=top_k)
    else:
        hybrid = dense[:top_k]

    # Quyết định fallback dựa trên cosine score GỐC của dense, không phải RRF.
    best_dense_score = dense[0]["score"] if dense else 0.0
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return _normalize(fallback)
        except Exception:  # noqa: BLE001 - provider ngoài lỗi không được crash UI
            pass

    return hybrid[:top_k]


if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)
