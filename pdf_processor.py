import io
import re
from dataclasses import dataclass, field
from typing import List

import PyPDF2


@dataclass
class TextChunk:
    text: str
    source: str
    page: int
    chunk_index: int


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[dict]:
    """Extract raw text per page from PDF bytes."""
    pages = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"text": text, "page": page_num, "source": filename})
    except Exception as e:
        raise ValueError(f"Failed to parse PDF '{filename}': {e}")
    return pages


def _clean_text(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    pages: List[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[TextChunk]:
    """
    Split page text into overlapping chunks of approximately `chunk_size`
    characters, respecting sentence/word boundaries where possible.
    """
    chunks: List[TextChunk] = []

    for page_data in pages:
        raw = _clean_text(page_data["text"])
        source = page_data["source"]
        page_num = page_data["page"]

        if not raw:
            continue

        start = 0
        chunk_index = 0

        while start < len(raw):
            end = start + chunk_size

            if end >= len(raw):
                segment = raw[start:]
            else:
                # Try to break at a sentence boundary (. ! ?)
                boundary = _find_break(raw, end)
                segment = raw[start:boundary]
                end = boundary

            segment = segment.strip()
            if segment:
                chunks.append(
                    TextChunk(
                        text=segment,
                        source=source,
                        page=page_num,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            start = end - chunk_overlap
            if start >= len(raw):
                break

    return chunks


def _find_break(text: str, pos: int, window: int = 80) -> int:
    """
    Search backwards from `pos` within `window` characters for a sentence or
    word boundary. Falls back to `pos` if none found.
    """
    search_start = max(0, pos - window)
    segment = text[search_start:pos]

    # Prefer sentence boundary
    for sep in (". ", "! ", "? ", "\n"):
        idx = segment.rfind(sep)
        if idx != -1:
            return search_start + idx + len(sep)

    # Fall back to word boundary
    idx = segment.rfind(" ")
    if idx != -1:
        return search_start + idx + 1

    return pos


def process_pdf(file_bytes: bytes, filename: str) -> List[TextChunk]:
    """Full pipeline: extract → clean → chunk."""
    pages = extract_text_from_pdf(file_bytes, filename)
    if not pages:
        raise ValueError(f"No extractable text found in '{filename}'.")
    return chunk_text(pages)
