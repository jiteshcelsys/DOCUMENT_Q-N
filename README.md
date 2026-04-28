# Document Q&A System

A full-stack RAG (Retrieval-Augmented Generation) application that lets you upload documents and ask questions about their content using natural language.

## Architecture

```
Upload File → Loader → Chunking → Embeddings → FAISS Vector DB
                                                      ↓
User Question → Embed → Similarity Search → Groq LLM → Answer
```

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector Store | FAISS (persisted to disk) |
| RAG Framework | LangChain (LCEL) |
| Frontend | React + Vite |

## Supported File Types

- PDF (`.pdf`)
- Word Documents (`.docx`)
- Plain Text (`.txt`)

## Project Structure

```
Document Q&A System/
├── main.py              # FastAPI backend
├── pyproject.toml       # Python dependencies
├── .env                 # API keys (never commit this)
├── uploads/             # Uploaded documents (auto-created)
├── faiss_index/         # Vector store (auto-created)
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── App.css
    │   ├── api.js               # API calls to backend
    │   └── components/
    │       ├── Uploader.jsx     # Drag & drop file upload
    │       └── Chat.jsx         # Q&A chat interface
    └── vite.config.js           # Dev proxy config
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) package manager
- Groq API key — get one free at [console.groq.com](https://console.groq.com)

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd "Document Q&A System"
```

### 2. Set up environment

Create a `.env` file in the root:

```
GROQ_API_KEY=gsk_...
```

### 3. Install Python dependencies

```bash
uv sync
```

### 4. Install frontend dependencies

```bash
cd frontend
npm install
```

## Running the App

Open **two terminals**:

**Terminal 1 — Backend:**
```bash
uv run python main.py
```

**Terminal 2 — Frontend:**
```bash
cd frontend
node node_modules/vite/bin/vite.js
```

Then open [http://localhost:5173](http://localhost:5173) in your browser.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a document (PDF, DOCX, TXT) |
| `POST` | `/ask` | Ask a question about uploaded documents |
| `DELETE` | `/index` | Clear the vector store |
| `GET` | `/health` | Check if the index is loaded |

### Example — Upload

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

### Example — Ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?", "k": 4}'
```

## How It Works

1. **Upload** — A document is parsed by a LangChain loader, split into overlapping chunks (1000 chars, 150 overlap), and embedded using a local HuggingFace model.
2. **Index** — Chunk vectors are stored in a FAISS index, saved to disk so they persist across restarts.
3. **Ask** — Your question is embedded, the top-k most similar chunks are retrieved, and Groq's LLaMA model generates a grounded answer citing only the document content.
