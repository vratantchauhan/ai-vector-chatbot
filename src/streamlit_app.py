import streamlit as st

from src.ingest import ensure_vector_store
from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="AI Vector Chatbot",
    page_icon="🤖"
)

st.title("AI Vector Chatbot")

st.write(
    "Ask questions from your knowledge base using "
    "FAISS vector search and OpenAI."
)

ensure_vector_store()
rag = RAGPipeline()

query = st.text_input("Ask a question:")

if st.button("Get Answer"):

    if query.strip():

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            answer = rag.generate_answer(query)

            st.subheader("Answer")

            st.write(answer)

    else:
        st.warning("Please enter a question.")