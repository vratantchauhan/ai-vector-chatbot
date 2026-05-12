from __future__ import annotations
import faiss
import numpy as np
from openai import OpenAI
from pathlib import Path
from pypdf import PdfReader
from docx import Document

from src.config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    DOCS_PATH,
    EMBEDDING_MODEL,
    INDEX_PATH,
    OPENAI_API_KEY,
    VECTOR_STORE_DIR,
)

SUPPORTED_SUFFIXES = {".txt", ".pdf", ".docx"}

client = OpenAI(api_key=OPENAI_API_KEY)


def iter_source_files(data_dir: Path):
    if not data_dir.is_dir():
        return
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def newest_source_mtime(data_dir: Path) -> float | None:
    mtimes = [p.stat().st_mtime for p in iter_source_files(data_dir)]
    return max(mtimes) if mtimes else None


def _vector_store_mtime() -> float | None:
    if not INDEX_PATH.exists() or not DOCS_PATH.exists():
        return None
    return min(INDEX_PATH.stat().st_mtime, DOCS_PATH.stat().st_mtime)


def needs_ingestion() -> bool:
    """True when the index is missing, incomplete, or older than source files."""
    store_ts = _vector_store_mtime()
    source_ts = newest_source_mtime(DATA_DIR)

    if store_ts is None:
        # Vector store missing or incomplete: ingest only when data/ has files to embed.
        if source_ts is None:
            return False
        return True

    if source_ts is None:
        return False

    return source_ts > store_ts


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_pdf_file(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def read_docx_file(file_path):
    document = Document(file_path)
    paragraphs = [
        para.text for para in document.paragraphs if para.text.strip()
    ]
    return "\n".join(paragraphs)


def load_documents_from_directory(directory):
    documents = []
    for file_path in iter_source_files(Path(directory)):
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".txt":
                text = read_text_file(file_path)
            elif suffix == ".pdf":
                text = read_pdf_file(file_path)
            elif suffix == ".docx":
                text = read_docx_file(file_path)
            else:
                continue
            if text.strip():
                documents.append({"source": str(file_path), "text": text})
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
    return documents


def chunk_text(text, chunk_size, chunk_overlap):
    words = text.split()
    if not words:
        return []
    stride = max(1, chunk_size - chunk_overlap)
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += stride
    return chunks


def get_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def ingest_vector_store() -> None:
    """Rebuild FAISS index and documents.txt from files under DATA_DIR."""
    raw_documents = load_documents_from_directory(DATA_DIR)
    documents = []
    for doc in raw_documents:
        for chunk in chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            documents.append({"source": doc["source"], "text": chunk})

    print(
        f"Loaded {len(raw_documents)} files, chunked into {len(documents)} chunks"
    )

    if not documents:
        print("No document chunks to embed; skipping index update.")
        if not INDEX_PATH.exists():
            raise RuntimeError(
                "No ingestable content in data/ and no vector store exists. "
                "Add .txt, .pdf, or .docx files under data/."
            )
        return

    document_embeddings = [get_embedding(doc["text"]) for doc in documents]
    print("Embeddings created")

    dimension = len(document_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array(document_embeddings).astype("float32")
    index.add(vectors)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with open(DOCS_PATH, "w", encoding="utf-8") as file:
        for doc in documents:
            file.write(f"Source: {doc['source']}\n")
            file.write(f"Text: {doc['text']}\n")
            file.write("---END---\n")

    print("FAISS index and documents saved successfully")


def ensure_vector_store() -> None:
    """Ingest when the store is missing or source files are newer than the store."""
    if needs_ingestion():
        print("Knowledge base out of date; ingesting documents...")
        ingest_vector_store()


if __name__ == "__main__":
    ingest_vector_store()
