import html
import uuid
from datetime import datetime

import streamlit as st

from src.ingest import ensure_vector_store
from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("RAG Chatbot")

st.write(
    "Ask questions from your knowledge base using "
    "FAISS vector search and OpenAI."
)


def _init_chat_state() -> None:
    if "chats" not in st.session_state:
        st.session_state.chats = {}

    if not st.session_state.chats:
        first_id = str(uuid.uuid4())
        st.session_state.chats[first_id] = {
            "title": "New chat",
            "messages": [],
            "updated": datetime.now().isoformat(),
        }
        st.session_state.active_chat_id = first_id
        return

    active = st.session_state.get("active_chat_id")
    if active not in st.session_state.chats:
        st.session_state.active_chat_id = next(iter(st.session_state.chats))


def _touch_chat(chat_id: str) -> None:
    st.session_state.chats[chat_id]["updated"] = datetime.now().isoformat()


def _new_chat() -> None:
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {
        "title": "New chat",
        "messages": [],
        "updated": datetime.now().isoformat(),
    }
    st.session_state.active_chat_id = new_id


def _sorted_chat_ids() -> list[str]:
    chats = st.session_state.chats
    return sorted(
        chats.keys(),
        key=lambda cid: chats[cid]["updated"],
        reverse=True,
    )


_init_chat_state()
ensure_vector_store()
rag = RAGPipeline()

with st.sidebar:
    st.header("Chats")
    if st.button("➕ New chat", use_container_width=True):
        _new_chat()
        st.rerun()

    st.divider()

    for cid in _sorted_chat_ids():
        chat = st.session_state.chats[cid]
        label = chat["title"]
        if len(label) > 36:
            label = label[:33] + "…"
        is_active = cid == st.session_state.active_chat_id
        prefix = "● " if is_active else "○ "
        if st.button(
            f"{prefix}{label}",
            key=f"select_chat_{cid}",
            use_container_width=True,
        ):
            st.session_state.active_chat_id = cid
            st.rerun()

active_id = st.session_state.active_chat_id
active = st.session_state.chats[active_id]

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

        user_text = query.strip()
        if not active["messages"]:
            active["title"] = (
                user_text[:50] + ("…" if len(user_text) > 50 else "")
            )
        active["messages"].append(
            {"role": "user", "content": user_text}
        )
        active["messages"].append(
            {"role": "assistant", "content": answer}
        )
        _touch_chat(active_id)

    else:
        st.warning("Please enter a question.")

with chat_area:

    for msg in active["messages"]:

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
