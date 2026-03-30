"""
query.py

Hybrid search over LanceDB research chunks, then synthesizes a
prose answer with inline citations using Granite via BAML.

Usage:
    python query.py
    python query.py --query "Does heel-to-toe drop affect knee pain?"
    python query.py --top_k 8 --sections results discussion

Dependencies:
    pip install lancedb sentence-transformers
    (plus baml-py and your baml_client generated from baml-cli generate)
"""

import argparse

import lancedb

from baml_client import b  # generated BAML client


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_URI = "shoeknowledge_lancedb"
TABLE_NAME = "research_chunks"
DEFAULT_TOP_K = 6

# Sections to search across — you can narrow this per query if needed
DEFAULT_SECTIONS = ["abstract", "introduction", "results", "discussion", "conclusion"]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    sections: list[str] | None = None,
    max_chunks_per_paper: int = 2,
) -> list[dict]:
    """
    Hybrid search (vector + FTS) against the research chunks table.
    Applies a per-paper diversity limit so no single paper dominates
    the context window regardless of how relevant it is.
    """
    db = lancedb.connect(DB_URI)
    table = db.open_table(TABLE_NAME)

    # Fetch more than top_k upfront so that after diversity filtering
    # we still end up with top_k results across different papers.
    fetch_k = top_k * max_chunks_per_paper * 2

    search = (
        table
        .search(query, query_type="hybrid")
        .limit(fetch_k)
    )

    if sections:
        section_list = ", ".join(f"'{s}'" for s in sections)
        search = search.where(f"section IN ({section_list})", prefilter=True)

    all_results = search.to_list()

    # Apply per-paper diversity limit — keep only the top N chunks
    # per paper title, preserving the original relevance ranking order.
    seen: dict[str, int] = {}  # paper_title -> count of chunks taken
    diverse_results = []

    for chunk in all_results:
        title = chunk["paper_title"]
        count = seen.get(title, 0)
        if count < max_chunks_per_paper:
            diverse_results.append(chunk)
            seen[title] = count + 1
        if len(diverse_results) >= top_k:
            break

    return diverse_results


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Formats retrieved chunks into a numbered context string for the prompt.
    Also returns the ordered list so we can print a references section.

    Context looks like:
        [1] Smith et al. (2023) | Section: results
        Carbon plates significantly reduce loading rate in recreational runners...

        [2] Jones et al. (2021) | Section: discussion
        ...
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[{i}] {chunk['authors']} ({chunk['year']}) "
            f"| \"{chunk['paper_title']}\" "
            f"| Section: {chunk['section']}"
        )
        lines.append(f"{header}\n{chunk['text']}")

    context = "\n\n---\n\n".join(lines)
    return context, chunks


# ---------------------------------------------------------------------------
# BAML prompt (rag.baml)
# ---------------------------------------------------------------------------
#
# Make sure your baml/rag.baml contains:
#
#   function AnswerWithCitations(question: string, context: string) -> string {
#     client GraniteClient
#     prompt #"
#       You are a running biomechanics research assistant.
#       Answer using only the provided research context.
#       Cite sources inline using their numbers, e.g. [1], [2].
#       Do not make up any information not present in the context.
#       If the context doesn't contain an answer, say so clearly.
#
#       QUESTION: {{ question }}
#
#       RESEARCH CONTEXT:
#       {{ context }}
#     "#
#   }
#
# ---------------------------------------------------------------------------

def answer(query: str, top_k: int = DEFAULT_TOP_K, sections: list[str] | None = None) -> None:
    print(f"\n🔍 Query: {query}\n")

    # 1. Retrieve
    chunks = retrieve(query, top_k=top_k, sections=sections)
    if not chunks:
        print("No relevant chunks found.")
        return

    # 2. Format context with citation numbers
    context, ordered_chunks = format_context(chunks)

    # 3. Call Granite via BAML
    print("Generating answer...\n")
    response = b.AnswerWithCitations(question=query, context=context)

    # 4. Print answer
    print("=" * 60)
    print(response)
    print("=" * 60)

    # 5. Print references
    print("\nReferences:")
    for i, chunk in enumerate(ordered_chunks, start=1):
        doi_str = f" | DOI: {chunk['doi']}" if chunk.get("doi") else ""
        print(
            f"  [{i}] {chunk['authors']} ({chunk['year']}). "
            f"\"{chunk['paper_title']}\"{doi_str}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the running research knowledge base."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Question to ask (if omitted, enters interactive mode)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        default=DEFAULT_SECTIONS,
        help="Sections to search (default: abstract introduction results discussion conclusion)",
    )
    args = parser.parse_args()

    if args.query:
        answer(args.query, top_k=args.top_k, sections=args.sections)
    else:
        # Interactive mode
        print("Running Research Assistant (type 'quit' to exit)\n")
        while True:
            query = input("Ask a question: ").strip()
            if query.lower() in {"quit", "exit", "q"}:
                break
            if query:
                answer(query, top_k=args.top_k, sections=args.sections)
