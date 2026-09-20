"""
Streamlit UI cho Tourism RAG Chatbot.

File này chỉ lo phần hiển thị. Toàn bộ logic retrieve/generate nằm ở src/,
UI không được tự ý đổi kết quả — chỉ trình bày lại cho dễ đọc.
"""

import os
import re

import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="Tourism RAG Chatbot",
    page_icon="🧭",
    layout="wide",
)

# --- Hằng số hiển thị -------------------------------------------------------

# Giải thích ý nghĩa từng retrieval_source + màu badge tương ứng.
RETRIEVAL_SOURCE_INFO = {
    "hybrid": (
        "#1a7f37",
        "#dafbe1",
        "Kết hợp dense (vector) + BM25 (từ khoá), hợp nhất bằng RRF.",
    ),
    "pageindex": (
        "#bc4c00",
        "#fff1e5",
        "Dense score thấp hơn ngưỡng nên pipeline đã chuyển sang PageIndex.",
    ),
    "none": (
        "#57606a",
        "#f6f8fa",
        "Không tìm được nguồn phù hợp nên bot từ chối trả lời.",
    ),
}

DOC_TYPE_LABEL = {"legal": "Văn bản luật", "news": "Tin tức"}

SAMPLE_QUESTIONS = [
    "Quyền của khách du lịch theo Luật Du lịch 2017 là gì?",
    "Điều kiện kinh doanh dịch vụ lữ hành quốc tế?",
    "Khách quốc tế đến Việt Nam 8 tháng 2026 đạt bao nhiêu?",
    "Có những loại cơ sở lưu trú du lịch nào?",
]

# WORKAROUND (chỉ ở tầng UI): corpus dính chuỗi "Đang theo dõi" — nút "Following"
# bị scrape nhầm từ web nguồn, lặp 1118 lần và nhiễm 53.5% số chunk.
# Ở đây chỉ lọc khi HIỂN THỊ cho demo đỡ rối; chuỗi rác VẪN nằm trong embedding
# và BM25 index nên vẫn đang làm giảm chất lượng retrieval.
# Fix gốc thuộc Task 3/4: làm sạch data/standardized/legal/*.md, re-index,
# rồi BẮT BUỘC chạy lại scripts/calibrate_threshold.py vì ngưỡng sẽ đổi.
_BOILERPLATE_RE = re.compile(r"(?:Đang theo dõi\s*)+")


def clean_for_display(text: str) -> str:
    """Bỏ boilerplate rác khỏi nội dung chunk trước khi hiển thị."""
    return _BOILERPLATE_RE.sub("", text or "").strip()


st.markdown(
    """
    <style>
    .source-badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: .2px;
    }
    .sample-hint { color: #57606a; font-size: 0.86rem; margin-bottom: .35rem; }
    div[data-testid="stExpander"] details { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_source_badge(retrieval_source: str) -> None:
    """Badge màu + tooltip cho retrieval_source."""
    color, background, tooltip = RETRIEVAL_SOURCE_INFO.get(
        retrieval_source, RETRIEVAL_SOURCE_INFO["none"]
    )
    st.markdown(
        f'<span class="source-badge" style="color:{color};background:{background};" '
        f'title="{tooltip}">Nguồn truy xuất: {retrieval_source}</span>',
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict], key_prefix: str) -> None:
    """Hiển thị danh sách nguồn: tiêu đề phân biệt được, metadata dạng cột."""
    if not sources:
        return

    st.markdown(f"**Nguồn tham khảo ({len(sources)})**")
    for index, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        title = metadata.get("title") or metadata.get("source") or source.get("id")
        score = source.get("score")
        chunk_index = metadata.get("chunk_index")

        # (10) Tiêu đề kèm số chunk + score để phân biệt các chunk cùng một file.
        label = f"Nguồn {index}: {title}"
        if chunk_index is not None:
            label += f" · chunk {chunk_index}"
        if isinstance(score, (int, float)):
            label += f" · score {score:.4f}"

        with st.expander(label):
            left, right = st.columns([3, 2])
            with left:
                st.caption("Tệp nguồn")
                st.markdown(f"`{metadata.get('source', 'không rõ')}`")

                doc_type = metadata.get("doc_type")
                if doc_type:
                    st.caption("Loại tài liệu")
                    st.markdown(DOC_TYPE_LABEL.get(doc_type, doc_type))

                # (2) url=None thì ghi rõ "(không có link)", có thì render link bấm được.
                url = metadata.get("url")
                st.caption("Liên kết")
                st.markdown(f"[{url}]({url})" if url else "_(không có link)_")

            with right:
                # (1)(3) Thay raw dict bằng metric + progress, score làm tròn 4 số.
                if isinstance(score, (int, float)):
                    st.metric("Điểm liên quan", f"{score:.4f}")
                    st.progress(min(max(float(score), 0.0), 1.0))
                st.caption("Phương pháp")
                st.markdown(f"`{source.get('retrieval_method', 'không rõ')}`")
                st.caption("Mã chunk")
                st.markdown(f"`{source.get('id', 'không rõ')}`")

            st.divider()
            st.markdown(clean_for_display(source.get("content", "")))


def render_assistant_message(message: dict, key_prefix: str) -> None:
    """Vẽ một lượt trả lời của bot (dùng chung cho lịch sử và lượt mới)."""
    retrieval_source = message.get("retrieval_source", "none")
    st.markdown(message["content"])
    render_source_badge(retrieval_source)

    # (8) Giải thích khi bot từ chối, tránh hiểu nhầm là app hỏng.
    # Lỗi provider (vd hết quota 429) phải hiện riêng: nếu gộp chung với "không
    # tìm được nguồn" thì người dùng sẽ tưởng corpus thiếu dữ liệu.
    error = message.get("error")
    if error:
        st.error(
            f"**Không tạo được câu trả lời.** {error}",
            icon="🚫",
        )
    elif retrieval_source == "none":
        st.warning(
            "Bot từ chối trả lời vì không truy xuất được nguồn đủ tin cậy trong "
            "corpus (luật + tin tức du lịch): độ tương đồng của đoạn khớp nhất "
            f"nằm dưới ngưỡng `SCORE_THRESHOLD={os.getenv('SCORE_THRESHOLD', 'mặc định')}` "
            "đã hiệu chỉnh. Đây là hành vi có chủ đích để tránh bịa thông tin, "
            "không phải lỗi ứng dụng. Hãy thử diễn đạt lại câu hỏi, tăng "
            "**Số chunks** ở thanh bên, hoặc hỏi nội dung nằm trong corpus.",
            icon="⚠️",
        )

    render_sources(message.get("sources", []), key_prefix)


# --- Trạng thái phiên -------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# --- Thanh bên --------------------------------------------------------------

with st.sidebar:
    st.title("🧭 Tourism RAG")
    st.caption("Luật và tin tức du lịch Việt Nam")
    st.divider()

    top_k = st.slider("Số chunks", 3, 10, 5)
    st.caption(
        "Số đoạn văn bản được lấy từ corpus để làm ngữ cảnh cho mô hình. "
        "Nhiều hơn thì bao quát rộng hơn nhưng dễ lẫn nội dung nhiễu."
    )

    st.divider()
    st.caption("Cấu hình đang dùng")
    # Chỉ hiện provider/model. TUYỆT ĐỐI không in API key ra giao diện.
    st.markdown(
        f"- Provider: `{os.getenv('LLM_PROVIDER', 'chưa đặt')}`\n"
        f"- Model: `{os.getenv('LLM_MODEL') or 'mặc định theo provider'}`\n"
        f"- Ngưỡng fallback: `{os.getenv('SCORE_THRESHOLD', 'mặc định')}`"
    )

    st.divider()
    if st.button("🗑️ Xoá hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()


# --- Khu vực chính ----------------------------------------------------------

st.title("Tourism RAG Chatbot")
st.caption("Hỏi đáp dựa trên corpus luật và tin tức du lịch của nhóm")

# (9) Câu hỏi mẫu bấm được — chỉ hiện khi chưa có hội thoại cho đỡ chiếm chỗ.
if not st.session_state.messages:
    st.markdown('<p class="sample-hint">Thử nhanh một câu hỏi mẫu:</p>', unsafe_allow_html=True)
    columns = st.columns(len(SAMPLE_QUESTIONS))
    for column, question in zip(columns, SAMPLE_QUESTIONS):
        with column:
            if st.button(question, use_container_width=True, key=f"sample-{question}"):
                st.session_state.pending_query = question
                st.rerun()

for position, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message, key_prefix=f"history-{position}")
        else:
            st.markdown(message["content"])

query = st.chat_input("Nhập câu hỏi...") or st.session_state.pending_query
st.session_state.pending_query = None

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và tạo câu trả lời..."):
            result = generate_with_citation(query, top_k)

        message = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "retrieval_source": result.get("retrieval_source", "none"),
            "error": result.get("error", ""),
        }
        render_assistant_message(message, key_prefix=f"live-{len(st.session_state.messages)}")

    st.session_state.messages.append(message)
