import os
import sys
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from pypdf import PdfReader
from docx import Document

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
VECTOR_STORE_DIR = _PROJECT_ROOT / "vector_store"

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)

def load_documents_from_directory(directory):

    documents = []
    for file_path in Path(directory).rglob("*"):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            if suffix in [".txt", ".pdf", ".docx"]:
                try:
                    if suffix == ".txt":
                        text = read_text_file(file_path)
                    elif suffix == ".pdf":
                        text = read_pdf_file(file_path)
                    elif suffix == ".docx":
                        text = read_docx_file(file_path)
                    else:
                        print(f"Unsupported file type: {suffix}")
                        continue
                    if (text.strip()):
                        documents.append({
                            "source": str(file_path),
                            "text": text
                        })
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue
    return documents

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

raw_documents = load_documents_from_directory(DATA_DIR)

documents = []
for doc in raw_documents:
    text = doc["text"]
    chunks = chunk_text(text)
    for chunk in chunks:
        documents.append({
            "source": doc["source"],
            "text": chunk
        })

print(f"Loaded {len(raw_documents)} files, chunked into {len(documents)} chunks")

if not documents:
    print("No document chunks to embed; exiting without writing index.")
    sys.exit(0)

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

document_embeddings = [get_embedding(doc["text"]) for doc in documents]
print("Embeddings created")

dimension = len(document_embeddings[0])

index = faiss.IndexFlatL2(dimension)

vectors = np.array(document_embeddings).astype("float32")
index.add(vectors)

VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(VECTOR_STORE_DIR / "faiss_index.index"))

with open(VECTOR_STORE_DIR / "documents.txt", "w", encoding="utf-8") as file:
    for doc in documents:
        file.write(f"Source: {doc['source']}\n")
        file.write(f"Text: {doc['text']}\n")
        file.write("---END---\n")

print("FAISS index and documents saved successfully")


