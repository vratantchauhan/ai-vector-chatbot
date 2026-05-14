import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.index"
DOCS_PATH = VECTOR_STORE_DIR / "documents.txt"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

TOP_K = int(os.getenv("TOP_K", "3"))

_default_sqlite_path = (PROJECT_ROOT / "app.db").resolve()
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{_default_sqlite_path.as_posix()}",
)