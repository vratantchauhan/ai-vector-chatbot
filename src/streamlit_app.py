import html

import streamlit as st

from src.ingest import ensure_vector_store
from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖"
)

st.title("RAG Chatbot")

st.write(
    "Ask questions from your knowledge base using "
    "FAISS vector search and OpenAI."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

ensure_vector_store()
rag = RAGPipeline()

chat_area = st.container()

with st.form("query_form", clear_on_submit=True):

    query = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Get Answer")

if submitted:

    if query.strip():

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            answer = rag.generate_answer(query)

        st.session_state.messages.append(
            {"role": "user", "content": query.strip()}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    else:
        st.warning("Please enter a question.")

with chat_area:

    for msg in st.session_state.messages:

        if msg["role"] == "user":

            safe = html.escape(msg["content"])

            st.markdown(
                f'<div style="background-color: #ececec; padding: 12px 16px; '
                f'border-radius: 8px; margin-bottom: 12px; white-space: pre-wrap;">'
                f"{safe}</div>",
                unsafe_allow_html=True,
            )

        else:

            st.markdown(msg["content"])
            st.markdown("")
