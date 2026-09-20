"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation. Nếu thiếu evidence, hãy từ chối xác minh."""

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

DEFAULT_LLM_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash",
    "anthropic": "claude-3-5-haiku-latest",
}


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        source = metadata.get("source", "unknown")
        title = metadata.get("title", source)
        url = metadata.get("url")
        source_label = f"{source} | URL: {url}" if url else source
        parts.append(
            f"[Document {index} | ID: {chunk['id']} | Title: {title} | "
            f"Source: {source_label}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = (os.getenv("LLM_PROVIDER") or LLM_PROVIDER).strip().lower()
    # Provider phải được kiểm tra TRƯỚC khi resolve model: nếu không, provider lạ
    # đi kèm LLM_MODEL có giá trị sẽ lọt qua và rơi xuống nhánh import sai ở dưới,
    # raise ImportError thay vì ValueError như contract mô tả.
    if provider not in DEFAULT_LLM_MODELS:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
    model = (os.getenv("LLM_MODEL") or LLM_MODEL).strip() or DEFAULT_LLM_MODELS[provider]

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or None)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return (response.choices[0].message.content or "").strip()

    if provider == "gemini":
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or None)
        response = client.models.generate_content(
            model=model,
            contents=f"{system_prompt}\n\n{user_message}",
            config={"temperature": TEMPERATURE, "top_p": TOP_P},
        )
        return (response.text or "").strip()

    if provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or None)
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=1024,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "text", None)
        ).strip()

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


PROVIDER_ERROR_NOTE = (
    "Không gọi được mô hình sinh câu trả lời (lỗi provider hoặc hết quota). "
    "Đây KHÔNG phải do thiếu dữ liệu trong corpus."
)


def _safe_refusal(error: str = "") -> dict:
    """Trả về refusal an toàn, kèm lý do thật để UI không báo nhầm.

    Trước đây mọi lỗi đều thành cùng một thông báo "không tìm được nguồn", nên
    lỗi quota 429 của provider bị hiểu nhầm thành "corpus không có dữ liệu" —
    rất khó chẩn đoán vì cả hai trông giống hệt nhau trên giao diện.
    """
    result = {
        "answer": SAFE_REFUSAL,
        "sources": [],
        "retrieval_source": "none",
    }
    if error:
        result["error"] = error
    return result


def _retrieval_source(chunks: list[dict]) -> str:
    """Quy retrieval_method của chunk về RetrievalSource của GenerationResult.

    Contract chỉ cho phép hybrid|pageindex|none. Task 9 có thể trả về chunk gắn
    nhãn "dense"/"bm25" (ví dụ khi gọi với use_reranking=False): đó vẫn là có
    nguồn thật, nên quy về "hybrid" chứ không phải "none" — trả "none" sẽ làm
    generate_with_citation từ chối trả lời dù context hoàn toàn hợp lệ.

    Ngoại lệ: chunk có cờ ``low_confidence`` (dense dưới ngưỡng và PageIndex
    fallback không khả dụng) được quy về "none" để UI hiện cảnh báo thay vì
    trình bày kết quả yếu như thể nó đáng tin.
    """
    if not chunks:
        return "none"
    # Task 9 gắn low_confidence khi dense score dưới SCORE_THRESHOLD mà PageIndex
    # fallback không dùng được: coi như không có nguồn đủ tin cậy.
    if any(chunk.get("low_confidence") for chunk in chunks):
        return "none"
    method = chunks[0].get("retrieval_method")
    if method == "pageindex":
        return "pageindex"
    if method in {"hybrid", "dense", "bm25"}:
        return "hybrid"
    return "none"


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception as error:  # noqa: BLE001 - pipeline lỗi thì UI vẫn trả refusal an toàn
        return _safe_refusal(f"Lỗi retrieval: {type(error).__name__}: {error}")

    retrieval_source = _retrieval_source(chunks)
    if retrieval_source == "none":
        return _safe_refusal()

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Trả lời bằng tiếng Việt. Trích dẫn bằng nhãn [Document n] tương ứng."
    )
    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception as error:  # noqa: BLE001 - UI phải trả safe refusal khi provider lỗi
        detail = str(error)
        if "RESOURCE_EXHAUSTED" in detail or "429" in detail:
            detail = (
                "Hết quota API (429 RESOURCE_EXHAUSTED). Free tier giới hạn theo "
                "từng model/ngày — đổi LLM_MODEL trong .env sang model Gemini khác "
                "hoặc chờ quota reset."
            )
        return _safe_refusal(f"{PROVIDER_ERROR_NOTE} Chi tiết: {detail}")

    if not answer.strip():
        return _safe_refusal()

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
