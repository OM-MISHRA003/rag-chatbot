"""
RAG Chatbot — Streamlit frontend
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (after page config) ────────────────────────────────────────
from pdf_processor import process_pdf
from rag_engine import RAGEngine

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- Chat bubbles (theme-aware using CSS variables) ---- */
    .chat-bubble {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        max-width: 85%;
        line-height: 1.55;
        font-size: 0.95rem;
    }

    /* Dark theme bubbles */
    [data-theme="dark"] .user-bubble,
    .user-bubble {
        background: #1e3a5f;
        color: #e8f0fe;
        margin-left: auto;
        border-bottom-right-radius: 2px;
    }
    [data-theme="dark"] .assistant-bubble,
    .assistant-bubble {
        background: #1c2333;
        color: #d4d9e8;
        border-bottom-left-radius: 2px;
        border: 1px solid #2a2f3d;
    }
    [data-theme="dark"] .source-tag,
    .source-tag {
        display: inline-block;
        background: #0d3349;
        color: #7ec8e3;
        border: 1px solid #1a5f7a;
        border-radius: 6px;
        padding: 0.15rem 0.55rem;
        font-size: 0.75rem;
        margin-right: 0.3rem;
        margin-top: 0.4rem;
    }

    /* Light theme overrides */
    @media (prefers-color-scheme: light) {
        .user-bubble {
            background: #dbeafe;
            color: #1e3a5f;
            margin-left: auto;
            border-bottom-right-radius: 2px;
        }
        .assistant-bubble {
            background: #f1f5f9;
            color: #1e293b;
            border-bottom-left-radius: 2px;
            border: 1px solid #cbd5e1;
        }
        .source-tag {
            background: #e0f2fe;
            color: #0369a1;
            border: 1px solid #7dd3fc;
        }
    }

    /* Streamlit light theme class override */
    [data-testid="stAppViewContainer"][class*="light"] .user-bubble,
    .stApp[data-theme="light"] .user-bubble {
        background: #dbeafe;
        color: #1e3a5f;
    }
    .stApp[data-theme="light"] .assistant-bubble {
        background: #f1f5f9;
        color: #1e293b;
        border: 1px solid #cbd5e1;
    }
    .stApp[data-theme="light"] .source-tag {
        background: #e0f2fe;
        color: #0369a1;
        border: 1px solid #7dd3fc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state initialisation ─────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "engine": None,
        "engine_error": None,
        "chat_history": [],       # list of {"role": ..., "content": ..., "sources": [...]}
        "indexed_sources": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ── RAG engine singleton ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine() -> RAGEngine:
    # Support both Streamlit Cloud secrets and local .env
    api_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
    return RAGEngine(api_key=api_key)


def get_engine() -> RAGEngine | None:
    if st.session_state.engine is None and st.session_state.engine_error is None:
        try:
            st.session_state.engine = load_engine()
            st.session_state.indexed_sources = (
                st.session_state.engine.get_indexed_sources()
            )
        except EnvironmentError as e:
            st.session_state.engine_error = str(e)
    return st.session_state.engine


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 RAG Chatbot")
    st.markdown("---")

    # API key warning
    engine = get_engine()
    if st.session_state.engine_error:
        st.error(f"⚠️ {st.session_state.engine_error}")
        st.stop()

    # PDF uploader
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supports multi-page PDFs. Re-uploading the same file updates it.",
    )

    if uploaded_files:
        for uploaded in uploaded_files:
            already_indexed = uploaded.name in st.session_state.indexed_sources
            btn_label = (
                f"🔄 Re-index  {uploaded.name}"
                if already_indexed
                else f"➕ Index  {uploaded.name}"
            )
            if st.button(btn_label, key=f"index_{uploaded.name}"):
                with st.spinner(f"Processing {uploaded.name} …"):
                    try:
                        file_bytes = uploaded.read()
                        chunks = process_pdf(file_bytes, uploaded.name)
                        count = engine.ingest_chunks(chunks)
                        st.session_state.indexed_sources = (
                            engine.get_indexed_sources()
                        )
                        st.success(
                            f"✅ Indexed **{uploaded.name}** — {count} chunks"
                        )
                    except ValueError as e:
                        st.error(f"❌ {e}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")

    st.markdown("---")

    # Indexed sources list
    st.markdown("### 📚 Indexed Documents")
    if st.session_state.indexed_sources:
        for src in st.session_state.indexed_sources:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📎 `{src}`")
            if col2.button("🗑️", key=f"del_{src}", help=f"Remove {src}"):
                removed = engine.delete_source(src)
                st.session_state.indexed_sources = engine.get_indexed_sources()
                st.success(f"Removed {removed} chunks from **{src}**")
                st.rerun()
        st.markdown(
            f"<small>Total chunks: **{engine.total_chunks()}**</small>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No documents indexed yet.")

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        "<small style='color:#555'>Powered by Groq (Llama 3.1) + ChromaDB + Sentence Transformers</small>",
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## 💬 Chat with your documents")

if not st.session_state.indexed_sources:
    st.info(
        "👈 Upload and index a PDF from the sidebar to get started.",
        icon="📄",
    )

# Render chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-bubble user-bubble">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        sources_html = "".join(
            f'<span class="source-tag">📎 {s}</span>'
            for s in msg.get("sources", [])
        )
        st.markdown(
            f'<div class="chat-bubble assistant-bubble">'
            f'🤖 {msg["content"]}'
            f'{"<br>" + sources_html if sources_html else ""}'
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input(
    "Ask a question about your documents…",
    disabled=not st.session_state.indexed_sources,
)

if question:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("Searching documents and generating answer…"):
        try:
            response = engine.answer(
                question,
                chat_history=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[:-1]  # exclude current
                ],
            )
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                }
            )

            # Optionally show retrieved chunks in an expander
            if response.retrieved_chunks:
                with st.expander("🔍 Retrieved context chunks", expanded=False):
                    for i, chunk in enumerate(response.retrieved_chunks, 1):
                        st.markdown(
                            f"**Chunk {i}** — `{chunk.source}` p.{chunk.page} "
                            f"(similarity: {chunk.score:.2f})"
                        )
                        st.markdown(
                            f"<div style='background:#161b27;padding:0.6rem;"
                            f"border-radius:8px;border:1px solid #2a2f3d;"
                            f"font-size:0.85rem;color:#aab;'>{chunk.text}</div>",
                            unsafe_allow_html=True,
                        )

        except Exception as e:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"⚠️ Error generating response: {e}",
                    "sources": [],
                }
            )

    st.rerun()
