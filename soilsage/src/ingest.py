"""Step 1: PDFs -> cleaned text -> overlapping chunks (with source + page metadata).

Run: python -m src.ingest
Output: data/processed/chunks.jsonl  (one JSON per line)
"""
import json
import re
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from tqdm import tqdm

from src.config import (CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_FILE,
                        MIN_CHUNK_CHARS, PROC_DIR, RAW_DIR)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)         # join hyphenated line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)   # single newline -> space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove headers/footers that repeat on many pages."""
    if len(pages) < 4:
        return pages
    counts = Counter()
    for p in pages:
        lines = {l.strip() for l in p.split("\n") if l.strip()}
        counts.update(lines)
    threshold = max(3, int(0.4 * len(pages)))
    common = {l for l, c in counts.items() if c >= threshold and len(l) < 120}
    return ["\n".join(l for l in p.split("\n") if l.strip() not in common) for p in pages]


def load_pdf(path):
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    return strip_repeated_lines(pages)


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RAW_DIR}. Add files (see data/SOURCES.md).")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    n_chunks, skipped = 0, []
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for pdf in tqdm(pdfs, desc="PDFs"):
            try:
                pages = load_pdf(pdf)
            except Exception as e:
                skipped.append((pdf.name, str(e)))
                continue
            if sum(len(p.strip()) for p in pages) < 500:
                skipped.append((pdf.name, "no extractable text (scanned? needs OCR)"))
                continue
            for page_no, raw in enumerate(pages, start=1):
                text = clean_text(raw)
                if len(text) < MIN_CHUNK_CHARS:
                    continue
                for i, chunk in enumerate(splitter.split_text(text)):
                    if len(chunk) < MIN_CHUNK_CHARS:
                        continue
                    rec = {
                        "id": f"{pdf.stem}_p{page_no}_c{i}",
                        "text": chunk,
                        "source": pdf.name,
                        "page": page_no,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_chunks += 1

    print(f"\nDone: {len(pdfs) - len(skipped)} PDFs -> {n_chunks} chunks -> {CHUNKS_FILE}")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")


if __name__ == "__main__":
    main()
