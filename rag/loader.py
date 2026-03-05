"""
Loads a PDF file, splits it into overlapping chunks, and returns a list of
LangChain Document objects — each carrying the source page number in metadata.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# Constants
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_TOTAL_CHARS = 100


def load_and_split(pdf_path: str | Path, original_filename: str = None) -> list[Document]:
    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix!r}")

    # 1. Load pages 
    loader = PyPDFLoader(str(pdf_path))
    pages: list[Document] = loader.load()          

    # 2. Scanned-PDF guard 
    total_text = "".join(p.page_content for p in pages)
    if len(total_text.strip()) < MIN_TOTAL_CHARS:
        raise ValueError(
            f"Extracted text is too short ({len(total_text.strip())} chars). "
            "This PDF may be scanned or image-only and cannot be processed. "
            "Please use a text-based PDF or run OCR first."
        )

    # 3. Split into chunks 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],# Keep sentence/paragraph boundaries where possible
        length_function=len,
    )

    chunks: list[Document] = splitter.split_documents(pages)

    # 4. Guarantee `page` key in every chunk's metadata ------------------
    # PyPDFLoader already sets metadata["page"], but we normalise here in
    # case a future loader omits it.
    for chunk in chunks:
        chunk.metadata.setdefault("page", 0)
        chunk.metadata["source"] = original_filename if original_filename else str(pdf_path)

    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m rag.loader <path/to/file.pdf>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"\nLoading: {path}\n{'─' * 50}")

    try:
        docs = load_and_split(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print(f"✓ Total chunks produced: {len(docs)}\n")
    print("── First 3 chunks ──────────────────────────────────")
    for i, doc in enumerate(docs[:3]):
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"\n[Chunk {i}]  page={page}")
        print(f"  text preview : {preview!r}")
        print(f"  char length  : {len(doc.page_content)}")
        print(f"  metadata     : {doc.metadata}")
