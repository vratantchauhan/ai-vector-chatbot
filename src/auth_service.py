from __future__ import annotations

import re

import bcrypt
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models import User

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,32}$")
MIN_PASSWORD_LEN = 8


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("ascii")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("ascii"),
        )
    except (ValueError, TypeError):
        return False


def register(
    session: Session,
    username: str,
    password: str,
) -> tuple[int | None, str | None]:
    username = username.strip()
    if not USERNAME_PATTERN.fullmatch(username):
        return (
            None,
            "Username must be 3–32 characters (letters, digits, underscore only).",
        )
    if len(password) < MIN_PASSWORD_LEN:
        return (
            None,
            f"Password must be at least {MIN_PASSWORD_LEN} characters.",
        )
    taken = session.scalar(select(User.id).where(User.username == username))
    if taken is not None:
        return None, "That username is already taken."
    user = User(username=username, password_hash=_hash_password(password))
    session.add(user)
    session.flush()
    return user.id, None


def verify_login(session: Session, username: str, password: str) -> int | None:
    username = username.strip()
    user = session.scalar(select(User).where(User.username == username))
    if user is None:
        return None
    if not _verify_password(password, user.password_hash):
        return None
    return user.id
