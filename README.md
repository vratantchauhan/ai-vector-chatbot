# AI RAG Chatbot Using Facebooks Vector Similarity Search and OpenAI's Chat Completions API

RAG chatbot: documents under `data/` are chunked and embedded with OpenAI, indexed with FAISS, then queried via a CLI or Streamlit UI.

## Prerequisites

- Python 3.10+ recommended
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

1. Clone the repository and open a terminal in the **project root** (the directory that contains `src/`, `data/`, and `requirements.txt`).

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   On Windows: `venv\Scripts\activate`

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If install fails because of `file://` entries in `requirements.txt` (common on some macOS exports), install the core packages manually, for example:

   ```bash
   pip install openai python-dotenv faiss-cpu numpy pypdf python-docx streamlit
   ```

4. Environment variables

   Change `.env.example` to `.env` in the project root and set your key:

   ```bash
   OPENAI_API_KEY=sk-...
   ```

   Optional variables (see `src/config/settings.py`): `EMBEDDING_MODEL`, `CHAT_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`.

## Ingest documents (build the vector store)

Add `.txt`, `.pdf`, and/or `.docx` files under `data/` (subfolders are allowed). From the **project root**:

```bash
python -m src.ingest
```

This creates `vector_store/faiss_index.index` and `vector_store/documents.txt`. Re-run ingest after changing source files.

## Run the chatbot

Run these from the **project root** so `import src` works.

**CLI**

```bash
python -m src.cli_app
```

Type questions at the prompt; enter `exit` to quit.

**Streamlit**

```bash
python -m streamlit run src/streamlit_app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Layout

| Path | Purpose |
|------|---------|
| `data/` | Source documents for ingestion |
| `vector_store/` | FAISS index and chunk store (generated) |
| `src/ingest.py` | Chunk, embed, write index |
| `src/rag_pipeline.py` | Load index, retrieve context, call OpenAI |
| `src/cli_app.py` | Terminal chat |
| `src/streamlit_app.py` | Web UI |
| `src/config/settings.py` | Paths and env-driven settings |

## Troubleshooting

- **`No module named 'src'`** — Use the commands above from the project root with `python -m src....`, not `python ingest.py` from inside `src/` unless `PYTHONPATH` includes the project root.
- **No or weak answers** — Confirm ingest finished successfully and `vector_store/` contains `faiss_index.index` and `documents.txt`.
- **OpenAI errors** — Check `OPENAI_API_KEY` in `.env` and account billing/limits.
