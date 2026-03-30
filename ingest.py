"""
ingest.py

Processes research paper PDFs and ingests them directly into LanceDB.
One script does everything: PDF extraction → chunking → embedding → storage.

Usage:
    python ingest.py --input_dir ./papers

Dependencies:
    pip install pymupdf lancedb sentence-transformers
"""

import re
import uuid
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import fitz  # PyMuPDF
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_URI = "shoeknowledge_lancedb"
TABLE_NAME = "research_chunks"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

embedding_model = (
    get_registry()
    .get("sentence-transformers")
    .create(name=EMBEDDING_MODEL, device="cpu")
)


class ChunkSchema(LanceModel):
    chunk_id: str
    text: str = embedding_model.SourceField()
    vector: Vector(embedding_model.ndims()) = embedding_model.VectorField()
    paper_title: str
    authors: str
    year: int
    doi: str
    section: str
    chunk_index: int


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    text: str
    paper_title: str
    authors: str
    year: int
    doi: str
    section: str
    chunk_index: int


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*abstract\s*$", re.IGNORECASE), "abstract"),
    (re.compile(r"^\s*introduction\s*$", re.IGNORECASE), "introduction"),
    (re.compile(r"^\s*(materials?\s+and\s+)?methods?\s*$", re.IGNORECASE), "methods"),
    (re.compile(r"^\s*(results?(\s+and\s+discussion)?)\s*$", re.IGNORECASE), "results"),
    (re.compile(r"^\s*discussion\s*$", re.IGNORECASE), "discussion"),
    (re.compile(r"^\s*conclusion[s]?\s*$", re.IGNORECASE), "conclusion"),
    (re.compile(r"^\s*references?\s*$", re.IGNORECASE), "references"),
    (re.compile(r"^\s*acknowledgements?\s*$", re.IGNORECASE), "acknowledgements"),
    (re.compile(r"^\s*limitations?\s*$", re.IGNORECASE), "limitations"),
    (re.compile(r"^\s*(statistical\s+)?analysis\s*$", re.IGNORECASE), "methods"),
    (re.compile(r"^\s*participants?\s*$", re.IGNORECASE), "methods"),
    (re.compile(r"^\s*(study\s+)?design\s*$", re.IGNORECASE), "methods"),
    (re.compile(r"^\s*\d+[\.\s]+introduction\s*$", re.IGNORECASE), "introduction"),
    (re.compile(r"^\s*\d+[\.\s]+methods?\s*$", re.IGNORECASE), "methods"),
    (re.compile(r"^\s*\d+[\.\s]+results?\s*$", re.IGNORECASE), "results"),
    (re.compile(r"^\s*\d+[\.\s]+discussion\s*$", re.IGNORECASE), "discussion"),
    (re.compile(r"^\s*\d+[\.\s]+conclusion[s]?\s*$", re.IGNORECASE), "conclusion"),
]

SKIP_SECTIONS = {"references", "acknowledgements"}


def detect_section(line: str) -> str | None:
    stripped = line.strip()
    if len(stripped) == 0 or len(stripped) > 60:
        return None
    for pattern, canonical in SECTION_PATTERNS:
        if pattern.match(stripped):
            return canonical
    return None


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_year_from_text(text: str) -> int:
    matches = re.findall(r"\b(19[89]\d|20[012]\d)\b", text)
    if matches:
        from collections import Counter
        return int(Counter(matches).most_common(1)[0][0])
    return 0


def extract_doi_from_text(text: str) -> str:
    match = re.search(r"(10\.\d{4,9}/[^\s]+)", text)
    return match.group(1).rstrip(".,;)") if match else ""


def extract_title_from_first_page(page_text: str) -> str:
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]

    skip_patterns = [
        re.compile(r"^\d"),
        re.compile(r"\d+\s*,\s*\d+"),
        re.compile(r"[A-Z][a-z]+\s+\d+\s*[,\*]"),       # "Name 1," author+affiliation
        re.compile(r"and\s+[A-Z][a-z]+\s+\d+"),          # "and Name 1"
        re.compile(r"@"),
        re.compile(r"university|institute|department|school|college|hospital|center",
                   re.IGNORECASE),
        re.compile(r"received|accepted|published|doi:|copyright", re.IGNORECASE),
        re.compile(r"\*\s*correspondence", re.IGNORECASE),
        re.compile(r"^(article|review|editorial|letter|comment)$", re.IGNORECASE),
    ]

    def is_likely_title(line: str) -> bool:
        if len(line) < 15:
            return False
        for pattern in skip_patterns:
            if pattern.search(line):
                return False
        return True

    candidates = [l for l in lines[:25] if is_likely_title(l)]

    if not candidates:
        return lines[0] if lines else "Unknown Title"

    # Titles in academic papers often split across two lines in bold headers
    if len(candidates) >= 2:
        first = candidates[0]
        second = candidates[1]
        combined = f"{first} {second}"
        if not first.endswith(".") and len(combined) < 200:
            return combined

    return candidates[0]


def extract_authors_from_first_page(page_text: str) -> str:
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]

    # Updated pattern handles "Name Number," format (MDPI and similar journals)
    author_pattern = re.compile(
        r"^([A-Z][a-z]+\s+[A-Z][a-z]+[\s\d\*,†‡§]+){1,}(and\s+)?[A-Z][a-z]+"
    )

    for line in lines[1:20]:
        if author_pattern.match(line):
            cleaned = re.sub(r"\s*[\d\*†‡§]+", "", line)
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip().rstrip(",*")
            return cleaned

    return "Authors not detected"



# ---------------------------------------------------------------------------
# PDF extraction and chunking
# ---------------------------------------------------------------------------

def extract_text_blocks(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    blocks = []
    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                line_text = ""
                is_bold = False
                font_size = 0.0
                for span in line["spans"]:
                    line_text += span["text"]
                    font_size = max(font_size, span["size"])
                    if "bold" in span["font"].lower():
                        is_bold = True
                line_text = line_text.strip()
                if line_text:
                    blocks.append({
                        "text": line_text,
                        "is_bold": is_bold,
                        "font_size": round(font_size, 1),
                        "page": page_num,
                    })
    doc.close()
    return blocks


def estimate_body_font_size(blocks: list[dict]) -> float:
    from collections import Counter
    sizes = [b["font_size"] for b in blocks if b["font_size"] > 0]
    if not sizes:
        return 10.0
    return Counter(sizes).most_common(1)[0][0]


def is_likely_header(block: dict, body_font_size: float) -> bool:
    text = block["text"].strip()
    if len(text) > 80:
        return False
    return block["is_bold"] or block["font_size"] > body_font_size + 0.5


def build_section_chunks(
    blocks: list[dict],
    body_font_size: float,
    min_chunk_words: int = 30,
    max_chunk_words: int = 300,
) -> list[tuple[str, str]]:
    current_section = "preamble"
    current_paragraph: list[str] = []
    results: list[tuple[str, str]] = []

    def flush_paragraph():
        text = " ".join(current_paragraph).strip()
        if len(text.split()) >= min_chunk_words:
            results.append((current_section, text))
        current_paragraph.clear()

    for block in blocks:
        text = block["text"]

        detected = detect_section(text)
        if detected:
            flush_paragraph()
            current_section = detected
            continue

        if is_likely_header(block, body_font_size) and len(text.split()) <= 8:
            flush_paragraph()
            continue

        if current_section in SKIP_SECTIONS:
            continue

        current_paragraph.append(text)
        if len(" ".join(current_paragraph).split()) >= max_chunk_words:
            flush_paragraph()

    flush_paragraph()
    return results


def process_pdf(pdf_path: Path) -> list[Chunk]:
    print(f"  Processing: {pdf_path.name}")
    blocks = extract_text_blocks(pdf_path)
    if not blocks:
        print(f"    ⚠ No text extracted, skipping.")
        return []

    first_page_text = "\n".join(b["text"] for b in blocks if b["page"] == 0)
    full_text_sample = "\n".join(b["text"] for b in blocks[:200])

    paper_title = extract_title_from_first_page(first_page_text)
    authors = extract_authors_from_first_page(first_page_text)
    year = extract_year_from_text(full_text_sample)
    doi = extract_doi_from_text(full_text_sample)

    body_font_size = estimate_body_font_size(blocks)
    section_chunks = build_section_chunks(blocks, body_font_size)

    chunks = [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            text=text,
            paper_title=paper_title,
            authors=authors,
            year=year,
            doi=doi,
            section=section,
            chunk_index=idx,
        )
        for idx, (section, text) in enumerate(section_chunks)
    ]

    print(f"    ✓ {len(chunks)} chunks | sections: {sorted(set(c.section for c in chunks))}")
    return chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest(input_dir: Path) -> None:
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s)\n")

    # Extract and chunk all PDFs
    all_chunks = []
    for pdf_path in pdf_files:
        chunks = process_pdf(pdf_path)
        all_chunks.extend(asdict(c) for c in chunks)

    if not all_chunks:
        print("No chunks extracted. Check your PDFs.")
        return

    print(f"\nTotal chunks extracted: {len(all_chunks)}")

    # Connect to LanceDB
    db = lancedb.connect(DB_URI)

    if TABLE_NAME in db.table_names():
        print(f"Table '{TABLE_NAME}' exists — dropping and recreating.")
        db.drop_table(TABLE_NAME)

    print("Creating table and embedding chunks (this may take a minute)...")
    table = db.create_table(TABLE_NAME, schema=ChunkSchema)
    table.add(all_chunks)
    print(f"✓ Embedded and inserted {len(all_chunks)} chunks")

    print("Building full-text search index...")
    table.create_fts_index("text", replace=True)
    print("✓ FTS index built")

    print(f"\nDone. '{TABLE_NAME}' is ready in '{DB_URI}'")

    # Section distribution summary
    from collections import Counter
    section_counts = Counter(c["section"] for c in all_chunks)
    print("\nChunks per section:")
    for section, count in sorted(section_counts.items()):
        print(f"  {section}: {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process research PDFs and ingest into LanceDB."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("./papers"),
        help="Directory containing PDF files (default: ./papers)",
    )
    args = parser.parse_args()
    ingest(args.input_dir)
