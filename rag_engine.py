import os
from dataclasses import dataclass
from typing import List, Tuple

from groq import Groq
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from pdf_processor import TextChunk

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_documents"
TOP_K = 3
GROQ_MODEL = "llama-3.1-8b-instant"
MAX_CONTEXT_CHARS = 6000


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float


@dataclass
class ChatResponse:
    answer: str
    sources: List[str]
    retrieved_chunks: List[RetrievedChunk]


class RAGEngine:
    def __init__(self, api_key: str | None = None) -> None:
        api_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
        if not api_key or api_key == "your_api_key_here":
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to your .env file or Streamlit secrets."
            )

        self._client = Groq(api_key=api_key)
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Persistent ChromaDB stored in a local directory
        self._chroma = chromadb.PersistentClient(path="./chroma_store")
        self._collection = self._chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_chunks(self, chunks: List[TextChunk]) -> int:
        """Embed and store chunks in ChromaDB. Returns number added."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()

        ids = [
            f"{c.source}__p{c.page}__c{c.chunk_index}" for c in chunks
        ]
        metadatas = [
            {"source": c.source, "page": c.page, "chunk_index": c.chunk_index}
            for c in chunks
        ]

        # ChromaDB upsert avoids duplicate errors on re-upload
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
        """Return the top-k most relevant chunks for a query."""
        count = self._collection.count()
        if count == 0:
            return []

        k = min(top_k, count)
        query_embedding = self._embedder.encode([query], show_progress_bar=False).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Cosine distance → similarity score (1 - distance)
            score = round(1.0 - dist, 4)
            retrieved.append(
                RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    page=meta.get("page", 0),
                    score=score,
                )
            )

        return retrieved

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def answer(self, question: str, chat_history: List[dict] | None = None) -> ChatResponse:
        """Retrieve context, call Claude, return structured response."""
        retrieved = self.retrieve(question)

        if not retrieved:
            return ChatResponse(
                answer=(
                    "I don't have any documents loaded yet. "
                    "Please upload a PDF first."
                ),
                sources=[],
                retrieved_chunks=[],
            )

        # Build context block (cap total characters to stay within token limits)
        context_parts = []
        total_chars = 0
        for chunk in retrieved:
            entry = (
                f"[Source: {chunk.source}, Page {chunk.page}]\n{chunk.text}"
            )
            if total_chars + len(entry) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(entry)
            total_chars += len(entry)

        context_block = "\n\n---\n\n".join(context_parts)
        unique_sources = list(dict.fromkeys(c.source for c in retrieved))

        system_prompt = (
            "You are a helpful assistant that answers questions strictly based on "
            "the provided document context. If the answer is not contained in the "
            "context, say so clearly. Always cite the source document name(s) at "
            "the end of your answer."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for turn in chat_history[-6:]:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({
            "role": "user",
            "content": f"Context from uploaded documents:\n\n{context_block}\n\nQuestion: {question}"
        })

        response = self._client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=1024,
        )
        answer_text = response.choices[0].message.content.strip()

        return ChatResponse(
            answer=answer_text,
            sources=unique_sources,
            retrieved_chunks=retrieved,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_indexed_sources(self) -> List[str]:
        """Return a list of unique source filenames currently in the DB."""
        count = self._collection.count()
        if count == 0:
            return []
        results = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "") for m in results["metadatas"]}
        return sorted(s for s in sources if s)

    def delete_source(self, filename: str) -> int:
        """Remove all chunks belonging to a specific source file."""
        results = self._collection.get(
            where={"source": filename},
            include=["metadatas"],
        )
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def total_chunks(self) -> int:
        return self._collection.count()
