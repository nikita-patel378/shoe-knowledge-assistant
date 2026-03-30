# Shoe Knowledge Assistant

A research-backed Q&A system that answers running biomechanics questions using evidence from peer-reviewed papers. Built as a portfolio project to demonstrate practical RAG (Retrieval-Augmented Generation) engineering with a local-first, cost-effective stack.

---

## What It Does

Ask a question about running shoes, biomechanics, or injury prevention and the assistant retrieves the most relevant passages from a corpus of 6 research papers, then synthesizes a nuanced, cited answer grounded strictly in the research — no hallucinations, no marketing hype.

**Example questions:**
- "Does heel-to-toe drop affect knee pain?"
- "How do carbon fiber plates affect running economy?"
- "What is the relationship between stack height and injury risk?"
- "Should a beginner runner use maximalist shoes?"

**Example output:**
```
Running in shoes with large drops significantly increases peak knee extension
moment, which may elevate patellofemoral stress in runners with knee weakness [1].
However, larger drop shoes can be advantageous for runners prone to ankle injuries,
as they reduce ankle eversion moment during the standing phase [2].

Citations:
  [1] Biomechanical Analysis of Running in Shoes with Different Heel-to-Toe Drops (2021)
  [2] Acute Effects of Heel-to-Toe Drop and Speed on Running Biomechanics (2021)

References:
  [1] Masen Zhang et al. (2021). DOI: 10.3390/app112412144
  [2] ...
```

---

## Architecture

```
PDF Research Papers
       ↓
   ingest.py
   PyMuPDF → section-aware chunking → LanceDB
   (auto-embeds with BAAI/bge-base-en-v1.5)
       ↓
   LanceDB
   Hybrid search (vector + full-text) + per-paper diversity filter
       ↓
   query.py
   BAML + IBM Granite → cited prose answer
```

**Key design decisions:**
- **Section-aware chunking** — chunks are tagged by paper section (Abstract, Results, Discussion, etc.) so retrieval can be filtered by section type
- **Hybrid search** — combines vector similarity and full-text keyword search via LanceDB's built-in RRF fusion, important for domain-specific terms like "heel-to-toe drop" or "patellofemoral stress"
- **Per-paper diversity limit** — retrieval caps at 2 chunks per paper so answers draw from multiple sources rather than one dominant paper
- **BAML for orchestration** — manages the IBM Granite client and prompt, returning structured `ResponseWithCitations` output with inline citation numbers

---

## Tech Stack

| Component | Tool |
|---|---|
| PDF extraction | PyMuPDF |
| Vector database | LanceDB (local, embedded) |
| Embedding model | BAAI/bge-base-en-v1.5 (local, via sentence-transformers) |
| Search | LanceDB hybrid search (vector + FTS) |
| LLM orchestration | BAML |
| LLM | IBM Granite (local, via Ollama) |

---

## Research Papers

The knowledge base is built from 6 peer-reviewed papers covering:
- Heel-to-toe drop and lower extremity biomechanics
- Stack height and running style
- Carbon fiber plate effects on running economy
- Cushioning and impact forces
- Footwear and injury risk

---

## Project Structure

```
shoe-knowledge-assistant/
├── papers/                  # PDF research papers
├── baml_src/
│   └── rag.baml             # BAML function and client config
├── baml_client/             # Auto-generated BAML client
├── ingest.py                # PDF processing + LanceDB ingestion
├── query.py                 # Hybrid search + BAML synthesis
└── shoeknowledge_lancedb/   # LanceDB database (auto-created)
```

---

## Setup

**Install dependencies:**
```bash
pip install pymupdf lancedb sentence-transformers baml-py
```


**Ingest research papers:**
```bash
python ingest.py --input_dir ./papers
```

**Ask a question:**
```bash
# Single query
python query.py --query "How does stack height affect injury risk?"

# Interactive mode
python query.py
```

---

