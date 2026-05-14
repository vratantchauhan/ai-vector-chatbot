import html
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from src import auth_service
from src import chat_repository
from src.db import init_db, session_scope
from src.ingest import ensure_vector_store
from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

_CSS_FLAG = "_streamlit_app_css_injected"


def _inject_app_css() -> None:
    if st.session_state.get(_CSS_FLAG):
        return
    css_path = Path(__file__).with_suffix(".css")
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )
    st.session_state[_CSS_FLAG] = True


@st.cache_resource
def _ensure_database() -> bool:
    init_db()
    return True


_ensure_database()
_inject_app_css()


def _render_auth_ui() -> None:
    try:
        with st.container(key="ragtitle_auth"):
            st.markdown("# RAG Chatbot")
    except TypeError:
        st.title("RAG Chatbot")
    st.write(
        "Sign in to ask questions from your knowledge base. "
        "Conversations are saved to your account."
    )
    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])
    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input(
                "Password", type="password", key="login_password"
            )
            login_submit = st.form_submit_button("Log in")
        if login_submit:
            with session_scope() as session:
                uid = auth_service.verify_login(
                    session, login_username, login_password
                )
            if uid is None:
                st.error("Invalid username or password.")
            else:
                st.session_state.user_id = uid
                st.session_state.username = login_username.strip()
                st.rerun()
    with signup_tab:
        with st.form("signup_form"):
            su_name = st.text_input("Username", key="signup_username")
            su_pass = st.text_input("Password", type="password", key="signup_password")
            su_pass2 = st.text_input(
                "Confirm password", type="password", key="signup_password2"
            )
            signup_submit = st.form_submit_button("Create account")
        if signup_submit:
            if su_pass != su_pass2:
                st.error("Passwords do not match.")
            else:
                with session_scope() as session:
                    uid, err = auth_service.register(session, su_name, su_pass)
                if err:
                    st.error(err)
                else:
                    st.session_state.user_id = uid
                    st.session_state.username = su_name.strip()
                    st.rerun()


def _reload_chats_from_db() -> None:
    uid = st.session_state.user_id
    with session_scope() as session:
        db_chats = chat_repository.list_chats(session, uid)
    st.session_state.chats = {}
    if not db_chats:
        new_id = str(uuid.uuid4())
        with session_scope() as session:
            chat_repository.create_chat(session, uid, new_id)
        st.session_state.chats[new_id] = {
            "title": "New chat",
            "messages": [],
            "updated": datetime.now().isoformat(),
            "_loaded": True,
        }
        st.session_state.active_chat_id = new_id
        return
    for c in db_chats:
        st.session_state.chats[c.id] = {
            "title": c.title,
            "messages": [],
            "updated": c.updated_at.isoformat(),
            "_loaded": False,
        }
    st.session_state.active_chat_id = db_chats[0].id


def _ensure_chats_bootstrapped() -> None:
    if st.session_state.get("_chat_bootstrap_user") != st.session_state.user_id:
        _reload_chats_from_db()
        st.session_state._chat_bootstrap_user = st.session_state.user_id


def _ensure_messages_loaded(chat_id: str) -> None:
    entry = st.session_state.chats[chat_id]
    if entry.get("_loaded"):
        return
    uid = st.session_state.user_id
    with session_scope() as session:
        msgs = chat_repository.get_messages(session, chat_id, uid)
    entry["messages"] = [] if msgs is None else msgs
    entry["_loaded"] = True


def _touch_chat(chat_id: str) -> None:
    st.session_state.chats[chat_id]["updated"] = datetime.now().isoformat()


def _new_chat() -> None:
    new_id = str(uuid.uuid4())
    uid = st.session_state.user_id
    with session_scope() as session:
        chat_repository.create_chat(session, uid, new_id)
    st.session_state.chats[new_id] = {
        "title": "New chat",
        "messages": [],
        "updated": datetime.now().isoformat(),
        "_loaded": True,
    }
    st.session_state.active_chat_id = new_id


def _sorted_chat_ids() -> list[str]:
    chats = st.session_state.chats
    return sorted(
        chats.keys(),
        key=lambda cid: chats[cid]["updated"],
        reverse=True,
    )


def _logout() -> None:
    for key in (
        "user_id",
        "username",
        "chats",
        "active_chat_id",
        "_chat_bootstrap_user",
    ):
        st.session_state.pop(key, None)


if st.session_state.get("user_id") is None:
    _render_auth_ui()
    st.stop()

_ensure_chats_bootstrapped()

ensure_vector_store()
rag = RAGPipeline()

try:
    with st.container(key="ragtitle"):
        st.markdown("# RAG Chatbot")
except TypeError:
    st.title("RAG Chatbot")

st.write(
    f"Signed in as **{st.session_state.get('username', '')}**. "
    "Ask questions from your knowledge base using FAISS vector search and OpenAI."
)

active_id = st.session_state.active_chat_id
if active_id not in st.session_state.chats:
    st.session_state.active_chat_id = next(iter(_sorted_chat_ids()))
    active_id = st.session_state.active_chat_id

with st.sidebar:
    if st.button("Log out", use_container_width=True):
        _logout()
        st.rerun()

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
_ensure_messages_loaded(active_id)
active = st.session_state.chats[active_id]

chat_area = st.container()

with st.form("query_form", clear_on_submit=True):
    query = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Get Answer")

if submitted:
    if query.strip():
        user_text = query.strip()
        was_empty = len(active["messages"]) == 0

        with st.spinner("Searching documents and generating answer..."):
            answer = rag.generate_answer(query)

        uid = st.session_state.user_id
        with session_scope() as session:
            chat_repository.append_turn(
                session, uid, active_id, user_text, answer
            )

        if was_empty:
            active["title"] = user_text[:50] + (
                "…" if len(user_text) > 50 else ""
            )
        active["messages"].append({"role": "user", "content": user_text})
        active["messages"].append({"role": "assistant", "content": answer})
        active["_loaded"] = True
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
