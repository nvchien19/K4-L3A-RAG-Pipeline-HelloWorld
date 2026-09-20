import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="Tourism RAG Chatbot",
    page_icon="",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Tourism RAG")
    st.caption("Luật và tin tức du lịch Việt Nam")
    top_k = st.slider("Số chunks", 3, 10, 5)

st.title("Tourism RAG Chatbot")
st.caption("Hỏi đáp dựa trên corpus luật và tin tức du lịch của nhóm")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            retrieval_source = message.get("retrieval_source", "none")
            sources = message.get("sources", [])
            st.caption(f"retrieval_source: {retrieval_source}")
            for index, source in enumerate(sources, 1):
                metadata = source.get("metadata", {})
                title = metadata.get("title") or metadata.get("source") or source.get("id")
                with st.expander(f"Nguồn {index}: {title}"):
                    st.markdown(source.get("content", ""))
                    st.write(
                        {
                            "id": source.get("id"),
                            "source": metadata.get("source"),
                            "url": metadata.get("url"),
                            "retrieval_method": source.get("retrieval_method"),
                            "score": source.get("score"),
                        }
                    )

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và tạo câu trả lời..."):
            result = generate_with_citation(query, top_k)
        answer = result["answer"]
        sources = result.get("sources", [])
        retrieval_source = result.get("retrieval_source", "none")
        st.markdown(answer)
        st.caption(f"retrieval_source: {retrieval_source}")

        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            title = metadata.get("title") or metadata.get("source") or source.get("id")
            with st.expander(f"Nguồn {index}: {title}"):
                st.markdown(source.get("content", ""))
                st.write(
                    {
                        "id": source.get("id"),
                        "source": metadata.get("source"),
                        "url": metadata.get("url"),
                        "retrieval_method": source.get("retrieval_method"),
                        "score": source.get("score"),
                    }
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
        }
    )
