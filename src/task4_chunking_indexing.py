"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().

Provider embedding được chọn bằng biến EMBEDDING_PROVIDER trong .env. Các giá
trị hợp lệ (xem EMBEDDING_PROVIDER_ALIASES bên dưới):
  - "local" / "sentence_transformers" / "sentence-transformers" / "st"
                           : sentence-transformers + EMBEDDING_MODEL (vd BAAI/bge-m3)
  - "openai"               : OpenAI text-embedding-3-small/large
  - "gemini"               : Google Gemini embedding (gemini-embedding-001, 3072
                             chiều) — chạy qua API nên không cần tải model về máy
  - fallback offline       : embedding hash xác định (EMBEDDING_DIM chiều) để vẫn
                             chạy pipeline demo khi chưa cài model nặng.

CẢNH BÁO: fallback hash KHÔNG có ngữ nghĩa, chỉ dùng để pipeline không chết khi
thiếu model. Nếu đang chạy fallback thì kết quả retrieval và ngưỡng
SCORE_THRESHOLD đo được đều không đại diện cho cấu hình production. Dùng
describe_embedding_backend() để biết chắc backend nào đang chạy.

Chunking ưu tiên langchain_text_splitters; nếu chưa cài thì dùng bộ tách nội bộ.
"""

import hashlib
import math
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024  # chỉ áp dụng cho hash fallback; model thật tự quyết chiều

# Số text gửi trong một request embed của Gemini. API giới hạn 100 contents mỗi
# lần gọi, để 50 cho an toàn khi chunk dài.
GEMINI_EMBED_BATCH = 50

# Free tier giới hạn 100 request embed mỗi phút. Khi chạm trần, đợi rồi thử lại
# thay vì rơi xuống hash fallback (sẽ ghi vector vô nghĩa vào index).
GEMINI_EMBED_MAX_RETRY = 12
GEMINI_EMBED_RETRY_DELAY = 30.0

# Chuẩn hoá tên provider: .env của các thành viên viết nhiều kiểu khác nhau
# ("sentence_transformers", "sentence-transformers", "local"...). Trước đây code
# chỉ so khớp "openai"/"gemini" rồi coi MỌI giá trị còn lại là local, nên một
# giá trị gõ sai cũng lặng lẽ rơi vào nhánh local thay vì báo lỗi.
EMBEDDING_PROVIDER_ALIASES = {
    "local": "local",
    "sentence_transformers": "local",
    "sentence-transformers": "local",
    "st": "local",
    "openai": "openai",
    "gemini": "gemini",
    "google": "gemini",
}

_RAW_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
EMBEDDING_PROVIDER = EMBEDDING_PROVIDER_ALIASES.get(_RAW_EMBEDDING_PROVIDER, "local")

# Ghi lại backend thực sự đã dùng ở lần embed gần nhất: "sentence_transformers",
# "openai", "gemini" hoặc "hash_fallback". None nghĩa là chưa embed lần nào.
_ACTIVE_EMBEDDING_BACKEND: str | None = None


def describe_embedding_backend() -> str:
    """Cho biết backend embedding đang thực sự chạy (không phải chỉ cấu hình).

    Cần thiết vì embed_texts() nuốt lỗi và tự fallback sang hash: nhìn .env thì
    tưởng đang chạy BAAI/bge-m3 trong khi thực tế là vector hash vô nghĩa.
    """
    if _ACTIVE_EMBEDDING_BACKEND is None:
        return f"chưa embed lần nào (cấu hình: {EMBEDDING_PROVIDER}/{EMBEDDING_MODEL})"
    if _ACTIVE_EMBEDDING_BACKEND == "hash_fallback":
        return (
            "hash_fallback — KHÔNG có ngữ nghĩa, chỉ để pipeline chạy được. "
            f"Cấu hình mong muốn là {EMBEDDING_PROVIDER}/{EMBEDDING_MODEL}."
        )
    return f"{_ACTIVE_EMBEDDING_BACKEND} ({EMBEDDING_MODEL})"


COLLECTION_NAME = "rag_documents"


_SENTENCE_TRANSFORMER = None

# Windows không tạo được symlink nên huggingface_hub in cảnh báo mỗi lần nạp
# model. Cache vẫn chạy bình thường (chỉ tốn thêm dung lượng), cảnh báo này chỉ
# gây nhiễu output nên tắt đi.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _get_sentence_transformer(factory):
    """Nạp SentenceTransformer một lần rồi cache lại.

    Trước đây model được khởi tạo trong MỖI lần gọi embed_texts(). Với bge-m3
    (~2.2GB) thì mỗi query sẽ nạp lại model từ đĩa, chậm tới mức không dùng nổi
    cho UI hay vòng lặp evaluation.
    """
    global _SENTENCE_TRANSFORMER
    if _SENTENCE_TRANSFORMER is None:
        _SENTENCE_TRANSFORMER = factory(EMBEDDING_MODEL)
    return _SENTENCE_TRANSFORMER


def _fallback_embedding(text: str) -> list[float]:
    """Embedding xác định, không cần model: tf của token hash → L2 chuẩn hóa."""
    vector = [0.0] * EMBEDDING_DIM
    for token in re.findall(r"\w+", text.lower()):
        index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % EMBEDDING_DIM
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _embed_gemini_batch(client, batch: list[str]):
    """Gọi embed_content, tự đợi và thử lại khi dính rate limit (HTTP 429).

    Free tier chỉ cho 100 request/phút. Nếu để lỗi 429 rơi xuống hash fallback
    thì index sẽ lẫn vector vô nghĩa mà pipeline vẫn báo thành công — đúng loại
    lỗi âm thầm đã khiến cả corpus phải index lại một lần rồi. Ở đây chờ theo
    retryDelay server trả về rồi thử lại; hết số lần thử mới ném lỗi ra ngoài.
    """
    import time  # noqa: PLC0415 - chỉ cần ở nhánh này

    last_error: Exception | None = None
    for attempt in range(GEMINI_EMBED_MAX_RETRY):
        try:
            return client.models.embed_content(model=EMBEDDING_MODEL, contents=batch)
        except Exception as error:  # noqa: BLE001 - phân loại bằng nội dung lỗi
            message = str(error)
            if "429" not in message and "RESOURCE_EXHAUSTED" not in message:
                raise
            last_error = error
            delay = _parse_retry_delay(message) or GEMINI_EMBED_RETRY_DELAY
            print(
                f"[embed] Dính giới hạn tốc độ Gemini, đợi {delay:.0f}s rồi thử lại "
                f"(lần {attempt + 1}/{GEMINI_EMBED_MAX_RETRY}).",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Vẫn bị giới hạn tốc độ sau {GEMINI_EMBED_MAX_RETRY} lần thử: {last_error}"
    )


def _parse_retry_delay(message: str) -> float | None:
    """Lấy số giây server đề nghị đợi từ thông điệp lỗi 429."""
    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", message)
    if not match:
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", message)
    return float(match.group(1)) + 2 if match else None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed một lô văn bản, dispatch theo EMBEDDING_PROVIDER.

    Override provider trong .env. Khi provider yêu cầu model/thư viện chưa
    được cài, tự động fallback về embedding hash để pipeline vẫn chạy được.
    """
    global _ACTIVE_EMBEDDING_BACKEND

    if not texts:
        return []

    reason = ""
    if EMBEDDING_PROVIDER == "openai":
        try:
            from openai import OpenAI  # noqa: PLC0415 - dep optional

            client = OpenAI()
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            _ACTIVE_EMBEDDING_BACKEND = "openai"
            return [item.embedding for item in response.data]
        except Exception as error:  # noqa: BLE001 - fallback khi thiếu key/package
            reason = f"{type(error).__name__}: {error}"
    elif EMBEDDING_PROVIDER == "gemini":
        try:
            from google import genai  # noqa: PLC0415 - dep optional

            client = genai.Client()
            vectors: list[list[float]] = []
            # Gọi theo lô: API nhận nhiều contents trong một request và mất gần
            # đúng bằng thời gian gọi một text. Vòng lặp từng text như trước
            # khiến index 471 chunks tốn ~471 round-trip (vài phút) thay vì ~30s.
            for start in range(0, len(texts), GEMINI_EMBED_BATCH):
                batch = texts[start : start + GEMINI_EMBED_BATCH]
                response = _embed_gemini_batch(client, batch)
                vectors.extend(item.values for item in response.embeddings)
            _ACTIVE_EMBEDDING_BACKEND = "gemini"
            return vectors
        except Exception as error:  # noqa: BLE001 - fallback khi thiếu key/package
            reason = f"{type(error).__name__}: {error}"
    else:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - dep optional

            model = _get_sentence_transformer(SentenceTransformer)
            _ACTIVE_EMBEDDING_BACKEND = "sentence_transformers"
            return model.encode(texts).tolist()
        except Exception as error:  # noqa: BLE001 - chưa cài sentence-transformers
            reason = f"{type(error).__name__}: {error}"

    # Fallback phải ồn ào: im lặng ở đây từng khiến cả nhóm tưởng đang chạy
    # bge-m3 trong khi thực tế toàn bộ index là vector hash vô nghĩa.
    if _ACTIVE_EMBEDDING_BACKEND != "hash_fallback":
        print(
            f"[CẢNH BÁO] Không dùng được provider embedding '{EMBEDDING_PROVIDER}' "
            f"({EMBEDDING_MODEL}) nên chuyển sang hash fallback. Lý do: {reason}",
            file=sys.stderr,
        )
    _ACTIVE_EMBEDDING_BACKEND = "hash_fallback"
    return [_fallback_embedding(text) for text in texts]


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb  # noqa: PLC0415 - cài trong môi trường lab

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        doc_type = "legal" if "legal" in path.parts else "news"
        documents.append(
            {
                "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
                "content": path.read_text(encoding="utf-8"),
                "metadata": {
                    "source": path.name,
                    "title": path.stem,
                    "doc_type": doc_type,
                    "url": None,
                },
            }
        )
    return documents


def _fallback_split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Tách văn bản theo chiều dài + ranh giới đoạn, xác định và không rỗng.

    Tương đương RecursiveCharacterTextSplitter: ưu tiên ngắt ở "\n\n", "\n",
    ". " rồi " ", cuối cùng cắt cứng theo ký tự. Kết quả không quá
    chunk_size để tránh vượt ràng buộc chia chunk.
    """
    hard_max = chunk_size - overlap - 2
    parts: list[str] = [text]

    for separator in ("\n\n", "\n", ". ", " "):
        if not any(len(part) > hard_max for part in parts):
            break
        next_parts: list[str] = []
        for part in parts:
            if len(part) <= hard_max:
                next_parts.append(part)
            elif separator == "":
                next_parts.extend(
                    part[index : index + hard_max] for index in range(0, len(part), hard_max)
                )
            else:
                split_part = [piece for piece in part.split(separator) if piece]
                if len(split_part) == 1:
                    next_parts.extend(
                        part[index : index + hard_max]
                        for index in range(0, len(part), hard_max)
                    )
                else:
                    next_parts.extend(split_part)
        parts = next_parts

    chunks: list[str] = []
    current = ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + len(part) + 2 <= chunk_size:
            current += "\n\n" + part
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + "\n\n" + part).strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_text = splitter.split_text
    except ImportError:
        split_text = _fallback_split_text

    chunks = []
    for document in documents:
        for index, text in enumerate(split_text(document["content"])):
            if not text.strip():
                continue
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": text,
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def _to_storable_metadata(metadata: dict) -> dict:
    """Chuyển metadata sang dạng ChromaDB lưu được.

    ChromaDB chỉ nhận str/int/float/bool: giá trị None bị loại bỏ âm thầm, làm
    ``url`` biến mất khỏi metadata sau khi đọc lại và vi phạm contract
    (``metadata.url`` phải là str hoặc None). Mã hoá None thành chuỗi rỗng để
    ``_from_stored_metadata`` khôi phục đúng kiểu ban đầu.
    """
    storable = {}
    for key, value in metadata.items():
        storable[key] = "" if value is None else value
    return storable


def from_stored_metadata(metadata: dict) -> dict:
    """Khôi phục metadata đọc từ ChromaDB về đúng contract.

    Task 5 dùng hàm này để trả lại ``url=None`` thay vì chuỗi rỗng, và bù key
    ``url`` cho dữ liệu đã index bằng phiên bản cũ.
    """
    restored = dict(metadata)
    restored["url"] = restored.get("url") or None
    return restored


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    collection = get_collection()
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[_to_storable_metadata(chunk["metadata"]) for chunk in chunks],
    )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()