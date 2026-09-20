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
    "gemini": "gemini-1.5-flash",
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
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    model = os.getenv("LLM_MODEL", LLM_MODEL).strip() or DEFAULT_LLM_MODELS.get(provider)
    if not model:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

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


def _safe_refusal() -> dict:
    return {
        "answer": SAFE_REFUSAL,
        "sources": [],
        "retrieval_source": "none",
    }


def _retrieval_source(chunks: list[dict]) -> str:
    if not chunks:
        return "none"
    method = chunks[0].get("retrieval_method")
    if method in {"hybrid", "pageindex"}:
        return method
    return "none"


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception:  # noqa: BLE001 - pipeline lỗi thì UI vẫn trả refusal an toàn
        return _safe_refusal()

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
    except Exception:  # noqa: BLE001 - UI phải trả safe refusal khi provider lỗi
        return _safe_refusal()

    if not answer.strip():
        return _safe_refusal()

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
