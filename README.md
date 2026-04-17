# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and ask questions about them. Built with Claude, ChromaDB, Sentence Transformers, and Streamlit.

---

## Architecture

```
User Question
      │
      ▼
Sentence Transformer (all-MiniLM-L6-v2)
      │  embed question
      ▼
ChromaDB  ──► Top-3 relevant chunks + source metadata
      │
      ▼
Claude claude-sonnet-4-20250514
  (system prompt + context + question)
      │
      ▼
Answer with cited source(s)
```

---

## Project Structure

```
RAG chatbot/
├── app.py              # Streamlit UI
├── rag_engine.py       # Embedding, storage, retrieval, and LLM call
├── pdf_processor.py    # PDF parsing and text chunking
├── requirements.txt    # Python dependencies
├── .env                # API key (not committed)
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python **3.10+**
- An [Anthropic API key](https://console.anthropic.com/)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tip:** Use a virtual environment:
> ```bash
> python -m venv .venv
> .venv\Scripts\activate        # Windows
> source .venv/bin/activate     # macOS/Linux
> pip install -r requirements.txt
> ```

### 3. Configure your API key

Open `.env` and replace the placeholder:

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Run the app

```bash
streamlit run app.py
```

The UI opens automatically at `http://localhost:8501`.

---

## Usage

1. **Upload a PDF** — Click *Browse files* in the sidebar and select one or more PDFs.
2. **Index** — Click the **➕ Index** button next to each file. A spinner shows progress; a success message confirms how many chunks were stored.
3. **Ask questions** — Type in the chat input at the bottom.  
   The app retrieves the top-3 most relevant chunks, sends them as context to Claude, and displays the answer with source citations.
4. **Inspect context** — Expand *Retrieved context chunks* below any answer to see exactly what text was passed to Claude.
5. **Remove a document** — Click 🗑️ next to a filename in the sidebar.

---

## Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Embedding model | `rag_engine.py → EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| Claude model | `rag_engine.py → CLAUDE_MODEL` | `claude-sonnet-4-20250514` |
| Chunks retrieved | `rag_engine.py → TOP_K` | `3` |
| Chunk size (chars) | `pdf_processor.py → chunk_text(chunk_size=...)` | `500` |
| Chunk overlap | `pdf_processor.py → chunk_text(chunk_overlap=...)` | `100` |
| ChromaDB path | `rag_engine.py → persist_directory` | `./chroma_store` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ANTHROPIC_API_KEY is not set` | Add your key to `.env` |
| `No extractable text found` | The PDF may be image-based; use an OCR tool first |
| Slow first run | Sentence Transformer model downloads on first launch (~90 MB) |
| ChromaDB `duckdb` error | Upgrade: `pip install --upgrade chromadb` |
