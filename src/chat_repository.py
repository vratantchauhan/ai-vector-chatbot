from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.models import Chat, Message


def list_chats(session: Session, user_id: int) -> list[Chat]:
    stmt = (
        select(Chat)
        .where(Chat.user_id == user_id)
        .order_by(Chat.updated_at.desc())
    )
    return list(session.scalars(stmt).all())


def get_chat_owned(session: Session, chat_id: str, user_id: int) -> Chat | None:
    return session.scalar(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id)
    )


def get_messages(session: Session, chat_id: str, user_id: int) -> list[dict] | None:
    if get_chat_owned(session, chat_id, user_id) is None:
        return None
    rows = session.scalars(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.position)
    ).all()
    return [{"role": m.role, "content": m.content} for m in rows]


def create_chat(session: Session, user_id: int, chat_id: str) -> None:
    session.add(
        Chat(
            id=chat_id,
            user_id=user_id,
            title="New chat",
            updated_at=datetime.utcnow(),
        )
    )


def append_turn(
    session: Session,
    user_id: int,
    chat_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    chat = get_chat_owned(session, chat_id, user_id)
    if chat is None:
        raise ValueError("Chat not found or access denied")
    prior = session.scalar(
        select(func.count()).select_from(Message).where(Message.chat_id == chat_id)
    )
    if int(prior or 0) == 0:
        chat.title = user_text[:50] + ("…" if len(user_text) > 50 else "")
    max_pos = session.scalar(
        select(func.max(Message.position)).where(Message.chat_id == chat_id)
    )
    start = 0 if max_pos is None else int(max_pos) + 1
    session.add(
        Message(
            chat_id=chat_id,
            role="user",
            content=user_text,
            position=start,
        )
    )
    session.add(
        Message(
            chat_id=chat_id,
            role="assistant",
            content=assistant_text,
            position=start + 1,
        )
    )
    chat.updated_at = datetime.utcnow()
