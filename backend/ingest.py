"""CLI: ingest one or more PDFs into the vector store.

Usage:
    python ingest.py path/to/file1.pdf path/to/file2.pdf
    python ingest.py ./data/pdfs/*.pdf
"""
import sys
from pathlib import Path

from rag import ingest_pdf


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python ingest.py <pdf> [<pdf> ...]")
        return 1
    for arg in argv:
        p = Path(arg)
        if not p.exists():
            print(f"  ! not found: {p}")
            continue
        try:
            res = ingest_pdf(p)
            print(f"  ✓ {res['filename']}: {res['chunks']} chunks")
        except Exception as e:
            print(f"  ! failed: {p}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
