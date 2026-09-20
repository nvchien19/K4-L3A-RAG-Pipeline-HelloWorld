"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.

Thiết kế:
  - PageIndex SDK chỉ nhận PDF, nên Markdown chuẩn hoá được convert sang PDF
    tạm bằng fpdf2 trước khi upload.
  - Mapping ``source -> doc_id`` được cache ở ``.pageindex_cache.json`` để chạy
    lại không upload lại (upload tốn thời gian và quota).
  - Mọi lỗi mạng/quota/thiếu key đều trả list rỗng thay vì raise, để Task 9 còn
    fallback về hybrid result.
"""

import json
import os
import re
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_PATH = Path(__file__).parent.parent / ".pageindex_cache.json"
PDF_CACHE_DIR = Path(__file__).parent.parent / ".pageindex_pdf"

# PageIndex là dịch vụ ngoài: chặn cứng thời gian chờ để UI không treo.
REQUEST_TIMEOUT = 60
POLL_TIMEOUT = 300


def _load_cache() -> dict:
    """Đọc mapping source -> document ID đã upload."""
    if not CACHE_PATH.exists():
        return {}
    try:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return cache if isinstance(cache, dict) else {}


def _save_cache(cache: dict) -> None:
    """Ghi lại cache document IDs."""
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _strip_markdown(text: str) -> str:
    """Bỏ cú pháp Markdown cơ bản để PDF dễ đọc."""
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    return text


def _markdown_to_pdf(path: Path) -> Path | None:
    """Convert một file Markdown sang PDF tạm, trả None nếu không convert được.

    PageIndex chỉ nhận PDF. Dùng font Unicode nếu có; nếu không thì hạ về
    latin-1 để ít nhất phần chữ không dấu vẫn upload được.
    """
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PDF_CACHE_DIR / f"{path.stem}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_mtime >= path.stat().st_mtime:
        return pdf_path

    try:
        from fpdf import FPDF  # noqa: PLC0415 - dep optional
    except ImportError:
        return None

    text = _strip_markdown(path.read_text(encoding="utf-8"))
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"
    if font_path.exists():
        pdf.add_font("body", "", str(font_path))
        pdf.set_font("body", size=11)
    else:
        # Không có font Unicode: bỏ dấu để fpdf không vỡ khi encode latin-1.
        pdf.set_font("Helvetica", size=11)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()

    try:
        pdf.multi_cell(0, 6, text)
        pdf.output(str(pdf_path))
    except Exception:  # noqa: BLE001 - font/encoding lỗi thì bỏ qua file này
        return None
    return pdf_path


def _get_client():
    """Khởi tạo PageIndex client, trả None nếu thiếu key hoặc chưa cài SDK."""
    if not PAGEINDEX_API_KEY:
        return None
    try:
        from pageindex import PageIndexClient  # noqa: PLC0415 - dep optional
    except ImportError:
        return None
    try:
        return PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except Exception:  # noqa: BLE001 - key sai định dạng
        return None


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    client = _get_client()
    if client is None:
        print("PageIndex chưa sẵn sàng (thiếu PAGEINDEX_API_KEY hoặc SDK).")
        return

    cache = _load_cache()
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        key = path.relative_to(STANDARDIZED_DIR).as_posix()
        if key in cache:
            continue

        pdf_path = _markdown_to_pdf(path)
        if pdf_path is None:
            print(f"Bỏ qua {key}: không convert được sang PDF.")
            continue

        try:
            response = client.submit_document(str(pdf_path))
        except Exception as error:  # noqa: BLE001 - lỗi mạng/quota
            print(f"Upload {key} thất bại: {error}")
            continue

        doc_id = _extract_doc_id(response)
        if not doc_id:
            print(f"Upload {key}: response không có document ID.")
            continue

        cache[key] = {
            "doc_id": doc_id,
            "source": path.name,
            "title": path.stem,
            "doc_type": "legal" if "legal" in path.parts else "news",
        }
        print(f"Đã upload {key} -> {doc_id}")

    _save_cache(cache)
    print(f"Cache có {len(cache)} document.")


def _extract_doc_id(response: object) -> str:
    """Lấy document ID từ response, chấp nhận nhiều kiểu field của SDK."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("doc_id", "document_id", "id"):
            value = response.get(key)
            if isinstance(value, str) and value:
                return value
        return ""
    for key in ("doc_id", "document_id", "id"):
        value = getattr(response, key, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _as_dict(node: object) -> dict:
    """Chuẩn hoá một node trả về thành dict."""
    if isinstance(node, dict):
        return node
    return {
        key: getattr(node, key)
        for key in ("node_id", "title", "text", "content", "relevant_content", "score")
        if hasattr(node, key)
    }


def _node_text(node: dict) -> str:
    """Lấy phần nội dung của node, thử lần lượt các tên field SDK dùng."""
    for key in ("relevant_content", "text", "content", "node_text", "summary"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _iter_nodes(payload: object):
    """Duyệt các retrieved node trong response bất kể SDK bọc ở tầng nào."""
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_nodes(item)
        return
    if not isinstance(payload, dict):
        payload = _as_dict(payload)
    if not payload:
        return
    for key in ("retrieved_nodes", "nodes", "results", "sources"):
        nested = payload.get(key)
        if isinstance(nested, list):
            for item in nested:
                yield from _iter_nodes(item)
            return
    if _node_text(payload):
        yield payload


def _run_retrieval(client, doc_id: str, query: str) -> object | None:
    """Submit query rồi poll tới khi có kết quả, trả None nếu lỗi/quá hạn.

    SDK gọi ``requests`` không có timeout nên phải tự chặn bằng deadline ở
    vòng poll, tránh treo UI khi PageIndex chậm.
    """
    try:
        submitted = client.submit_query(doc_id=doc_id, query=query)
    except Exception:  # noqa: BLE001 - lỗi mạng/quota/key sai
        return None

    retrieval_id = ""
    if isinstance(submitted, dict):
        retrieval_id = submitted.get("retrieval_id") or submitted.get("id") or ""
    if not retrieval_id:
        return None

    deadline = time.monotonic() + POLL_TIMEOUT
    delay = 1.0
    while time.monotonic() < deadline:
        try:
            payload = client.get_retrieval(retrieval_id)
        except Exception:  # noqa: BLE001 - lỗi mạng giữa chừng
            return None
        status = payload.get("status") if isinstance(payload, dict) else None
        if status in {"completed", "success", "done", None}:
            return payload
        if status in {"failed", "error", "cancelled"}:
            return None
        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)
    return None


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    client = _get_client()
    cache = _load_cache()
    if client is None or not cache:
        return []

    results: list[dict] = []
    for key, entry in cache.items():
        if not isinstance(entry, dict) or not entry.get("doc_id"):
            continue

        response = _run_retrieval(client, entry["doc_id"], query)
        if response is None:
            continue

        for index, node in enumerate(_iter_nodes(response)):
            content = _node_text(node)
            if not content:
                continue
            score = node.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                # API không trả score: gán giảm dần theo rank để giữ contract.
                score = 1.0 / (index + 1)
            results.append(
                {
                    "id": f"{key}::pageindex-{node.get('node_id', index)}",
                    "content": content,
                    "score": float(score),
                    "metadata": {
                        "source": entry.get("source", key),
                        "title": node.get("title") or entry.get("title", key),
                        "doc_type": entry.get("doc_type", "legal"),
                        "url": None,
                        "chunk_index": index,
                    },
                    "retrieval_method": "pageindex",
                }
            )

    # Khử trùng ID rồi sort giảm dần theo contract SearchResult.
    unique: dict[str, dict] = {}
    for item in results:
        if item["id"] not in unique or item["score"] > unique[item["id"]]["score"]:
            unique[item["id"]] = item
    return sorted(unique.values(), key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    upload_documents()
