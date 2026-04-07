# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An offline Retrieval-Augmented Generation (RAG) chatbot that ingests PDFs, extracts text and images, builds a FAISS vector store, and serves a web UI for semantic search queries. The embeddings model runs locally; the LLM is served by a local Jan server.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest PDFs (builds/rebuilds vector store from uploads/)
python ingest.py

# Start the web server
uvicorn app:app --reload

# Test vector DB queries
python test.py                          # Interactive mode
python test.py quick                    # Sanity check
python test.py query "your question"    # Single query
```

## Architecture

### Data Flow
```
uploads/<category>/*.pdf
    → PDFIngestor (ingest.py)
        → Text chunks (500 chars, 50 overlap) via RecursiveCharacterTextSplitter
        → Image extraction + surrounding text context via PyMuPDF (fitz)
    → SentenceTransformerEmbeddings (paraphrase-multilingual-MiniLM-L12-v2, 384-dim)
    → FAISS index (vector_store/index.faiss + index.pkl)
    → manifest.json (tracks ingested files by path + mtime)

User query → POST /ask
    → Language detection via langdetect (optional, graceful fallback)
    → Intent detection: text-only / images-only / both (multilingual keyword matching)
    → FAISS similarity_search_by_vector (k=15, top 5 text chunks used for LLM context)
    → LLM call → Jan server at http://localhost:1337/v1 (OpenAI-compatible API)
    → JSON response with HTML-formatted text + image grid + clickable citations
```

### Key Files

- **`ingest.py`** — `PDFIngestor` class: full pipeline from PDF → FAISS. `full_rebuild()` always deletes the entire vector store and images dir before rebuilding from scratch — no incremental updates. The first ~315 lines are a commented-out older version that used `clip-ViT-B-32` embeddings; the active code starts at line 321.
- **`app.py`** — Fully functional FastAPI app. Routes: `/` and `/home` (file management UI), `/chat` (chat UI, redirects to home if embeddings missing), `/ask` (POST, main query endpoint), `/upload-files`, `/list-files`, `/delete-file`, `/run-ingestion`, `/serve-pdf`, `/pdf-preview`, `/security-audit`, `/analytics`. Upload triggers background ingestion automatically.
- **`test.py`** — CLI tool for querying the vector store directly, useful for verifying ingestion results.
- **`templates/home.html`** — File management and ingestion trigger UI.
- **`templates/index.html`** — Main chat interface.
- **`vector_store/manifest.json`** — Tracks which PDFs are ingested and their timestamps.

### Storage Layout
- `uploads/<category>/` — PDFs organized in named subfolders (each subfolder becomes a category/folder label in citations)
- `images/` — Extracted PDF images named `<pdf_stem>_p<page>_img<idx>.<ext>`
- `vector_store/` — FAISS index files and manifest

### Embeddings Model
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — lightweight (~100 MB), 50+ language support, 384-dimensional vectors. Downloaded on first run and cached by HuggingFace.

## Key Implementation Details

- **LLM integration**: Uses `openai.OpenAI` client pointed at a local Jan server (`http://localhost:1337/v1`). The API key is currently hardcoded in `app.py:95`. Both the server URL and key should be moved to environment variables.
- **Optional dependencies**: `SpeechRecognition`/`pyaudio` (voice input) and `langdetect` (language detection) are imported with try/except — the app runs without them.
- **Query intent**: `analyze_query_intent()` in `app.py` uses multilingual keyword dictionaries for 12 languages to classify queries as `text_only`, `images_only`, or `both`.
- **Image contextual embedding**: Images are embedded via their text description (surrounding page text + caption), not as raw pixels. The description is what gets retrieved by FAISS.
- **Citations**: The `/ask` response includes clickable citation HTML that opens PDFs via `/serve-pdf`. `/serve-pdf` enforces that the path starts with `uploads/` as a path-traversal guard.
- **`bt_print.py`** is an unrelated serial port listener utility (COM3), not part of the RAG pipeline.
