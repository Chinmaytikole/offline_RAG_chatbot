# 🤖 Offline RAG Chatbot

A fully **offline, privacy-first** Retrieval-Augmented Generation (RAG) chatbot that lets you chat with your PDF documents — no internet required after initial setup.

Upload PDFs, build a local vector store, and ask questions in **multiple languages**. Images inside PDFs are also extracted and made searchable. Voice input is supported via offline transcription.

---

## ✨ Features

- 📄 **PDF Ingestion** – Upload PDFs and automatically extract text and images
- 🔍 **Semantic Search** – FAISS vector store with multilingual sentence embeddings
- 🌍 **Multilingual Support** – Ask questions in any language (powered by `paraphrase-multilingual-MiniLM-L12-v2`)
- 🖼️ **Image Retrieval** – Extracts images from PDFs with surrounding context for semantic search
- 🎙️ **Offline Voice Input** – Speech-to-text using a local Whisper model (no internet after first download)
- 🔒 **Fully Local** – All data, embeddings, and LLM inference stay on your machine
- ⚡ **FastAPI Backend** – Fast async REST API with background ingestion tasks
- 🏷️ **Source Citations** – Answers include source document and page references

---

## 🗂️ Project Structure

```
offline_RAG_chatbot/
├── app.py              # FastAPI application (main entry point)
├── ingest.py           # PDF ingestion & FAISS vector store builder
├── install.bat         # One-time setup: venv + dependencies + Whisper model download
├── run.bat             # Start the app (activates venv and launches server)
├── requirements.txt    # Python dependencies
├── templates/          # Jinja2 HTML templates (home.html, index.html)
├── uploads/            # Your PDF files go here (organised in subfolders)
├── vector_store/       # Auto-generated FAISS index (do not commit)
├── images/             # Auto-extracted PDF images (do not commit)
└── whisper_model/      # Cached Whisper model — downloaded by install.bat (do not commit)
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python **3.10 – 3.12** (Python 3.13 is not yet supported — `audioop` is removed)
- A running **local LLM server** compatible with the OpenAI API (e.g. [Jan](https://jan.ai/) or [LM Studio](https://lmstudio.ai/)) listening on `http://localhost:1337/v1`

### 2. Clone the Repository

```bash
git clone https://github.com/<your-username>/offline_RAG_chatbot.git
cd offline_RAG_chatbot
```

### 3. Install (one time)

Double-click **`install.bat`** or run it from a terminal:

```bat
install.bat
```

This will:
- Create a Python virtual environment in `.venv/`
- Install all dependencies from `requirements.txt`
- Download the Whisper `base` speech model (~145 MB) from Hugging Face into `whisper_model/`

> **Note:** An internet connection is required for this step. After setup, the app runs fully offline.

### 4. Start the App

Double-click **`run.bat`** or run it from a terminal:

```bat
run.bat
```

Open your browser at **http://127.0.0.1:8000**

---

## 📖 Usage

### Step 1 – Upload PDFs

1. Go to the **Home** page (`/`)
2. Enter a folder/topic name (e.g. `biology`, `company-docs`)
3. Select one or more PDF files and click **Upload**
4. Ingestion starts automatically in the background — this may take a few minutes for large files

### Step 2 – Chat

1. Once ingestion is complete, click **Go to Chat** (`/chat`)
2. Type your question in any language, or use the microphone button for voice input
3. The chatbot retrieves relevant text chunks and images, then generates an answer using your local LLM

### Step 3 – Manage Files

- Use the Home page to **list** or **delete** uploaded PDFs
- Deleting a file automatically triggers a re-ingestion of the remaining documents

---

## 🎙️ Offline Voice Input (Whisper)

Voice transcription uses **faster-whisper** with a local `base` Whisper model.

- The model (~145 MB) is downloaded automatically by `install.bat` and cached in `whisper_model/`
- Transcription works **completely offline** with no further downloads needed
- The model is cached locally to avoid Windows symlink permission errors

---

## ⚙️ Configuration

Key paths and settings are defined at the top of `app.py`:

| Variable | Default | Description |
|---|---|---|
| `DB_FAISS_PATH` | `vector_store` | Where the FAISS index is stored |
| `IMAGE_DIR` | `images` | Extracted PDF images |
| `UPLOADS_PATH` | `uploads` | Uploaded PDF files |
| LLM base URL | `http://localhost:1337/v1` | Your local LLM server URL |

To change the **embedding model**, update `model_name` in both `app.py` and `ingest.py`.

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Web server |
| `langchain` + `langchain-community` | LLM orchestration & document loading |
| `faiss` | Vector similarity search |
| `sentence-transformers` | Multilingual text embeddings |
| `PyMuPDF` (`fitz`) | PDF text & image extraction |
| `faster-whisper` | Offline speech-to-text |
| `Pillow` | Image processing |
| `langdetect` | Language detection |
| `openai` | OpenAI-compatible client for local LLM |

---

## 🔒 Privacy

- No data is ever sent to external servers
- PDF content, embeddings, and chat history remain on your local machine
- The only external network call is the **one-time** Whisper model download from Hugging Face

---

## ⚠️ Known Issues

- **Python 3.13**: The `audioop` module used by `SpeechRecognition` was removed. Use Python 3.10–3.12.
- **Large PDFs**: Ingestion time scales with PDF size and image count. The vector store is rebuilt from scratch on every ingestion.
- **Whisper model**: Requires internet during `install.bat` to download the model. After that, fully offline.

---

## 📝 License

This project is for personal/educational use. See individual dependency licenses for their respective terms.
