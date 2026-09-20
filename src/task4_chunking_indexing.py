"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().

Provider embedding được chọn bằng biến EMBEDDING_PROVIDER trong .env:
  - "local"  (mặc định) : sentence-transformers + EMBEDDING_MODEL (vd BAAI/bge-m3)
  - "openai"             : OpenAI text-embedding-3-small/large
  - "gemini"             : Google Gemini embedding
  - fallback offline     : embedding hash xác định (EMBEDDING_DIM chiều) để vẫn
                           chạy pipeline demo khi chưa cài model nặng.
Chunking ưu tiên langchain_text_splitters; nếu chưa cài thì dùng bộ tách nội bộ.
"""

import hashlib
import math
import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    pass


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

COLLECTION_NAME = "rag_documents"


def _fallback_embedding(text: str) -> list[float]:
    """Embedding xác định, không cần model: tf của token hash → L2 chuẩn hóa."""
    vector = [0.0] * EMBEDDING_DIM
    for token in re.findall(r"\w+", text.lower()):
        index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % EMBEDDING_DIM
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed một lô văn bản, dispatch theo EMBEDDING_PROVIDER.

    Override provider trong .env. Khi provider yêu cầu model/thư viện chưa
    được cài, tự động fallback về embedding hash để pipeline vẫn chạy được.
    """
    if not texts:
        return []

    use_model = True
    if EMBEDDING_PROVIDER == "openai":
        try:
            from openai import OpenAI  # noqa: PLC0415 - dep optional

            client = OpenAI()
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return [item.embedding for item in response.data]
        except Exception:  # noqa: BLE001 - fallback khi thiếu key/package
            use_model = False
    elif EMBEDDING_PROVIDER == "gemini":
        try:
            from google import genai  # noqa: PLC0415 - dep optional

            client = genai.Client()
            items = [
                client.models.embed_content(model=EMBEDDING_MODEL, contents=text).embeddings
                for text in texts
            ]
            return [item.values for item in items]
        except Exception:  # noqa: BLE001 - fallback khi thiếu key/package
            use_model = False
    else:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - dep optional

            model = SentenceTransformer(EMBEDDING_MODEL)
            return model.encode(texts).tolist()
        except Exception:  # noqa: BLE001 - chưa cài sentence-transformers
            use_model = False

    if not use_model:
        return [_fallback_embedding(text) for text in texts]
    raise RuntimeError("Không có provider embedding khả dụng")


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