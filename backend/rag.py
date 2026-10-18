"""RAG layer: chunking, embedding, ChromaDB read/write.

Uses BAAI/bge-m3 by default – it handles English and Urdu (and 100+ others)
in the same embedding space, so a question in Urdu can retrieve chunks
written in English and vice versa.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_MODEL,
    TOP_K,
)


# Lazy singletons – first call loads the embedding model into memory.
_embedder: SentenceTransformer | None = None
_client: chromadb.PersistentClient | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[rag] loading embedding model: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_collection():
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed(texts: List[str]) -> List[List[float]]:
    return _get_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(parts)


def _chunk(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "۔", ".", "!", "?", ";", ":", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def _doc_id(filename: str) -> str:
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:16]


def ingest_pdf(path: Path, original_name: str | None = None) -> Dict[str, Any]:
    """Read a PDF, chunk, embed, upsert into ChromaDB.

    Returns: {"doc_id": str, "filename": str, "chunks": int}
    """
    name = original_name or path.name
    doc_id = _doc_id(name)
    text = _read_pdf(path)
    if not text.strip():
        raise ValueError(f"No extractable text in {name}. Is it a scanned PDF?")
    chunks = _chunk(text)
    if not chunks:
        raise ValueError(f"PDF produced 0 chunks: {name}")

    coll = _get_collection()
    # Remove any prior version of this doc.
    try:
        coll.delete(where={"doc_id": doc_id})
    except Exception:
        pass

    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "filename": name, "chunk_index": i}
        for i in range(len(chunks))
    ]
    embeddings = _embed(chunks)
    coll.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)

    return {"doc_id": doc_id, "filename": name, "chunks": len(chunks)}


def list_documents() -> List[Dict[str, Any]]:
    coll = _get_collection()
    res = coll.get(include=["metadatas"])
    seen: Dict[str, Dict[str, Any]] = {}
    for meta in res.get("metadatas", []) or []:
        if not meta:
            continue
        d_id = meta.get("doc_id")
        if d_id and d_id not in seen:
            seen[d_id] = {"doc_id": d_id, "filename": meta.get("filename", "")}
            seen[d_id]["chunks"] = 1
        elif d_id:
            seen[d_id]["chunks"] += 1
    return list(seen.values())


def delete_document(doc_id: str) -> int:
    coll = _get_collection()
    try:
        before = coll.count()
        coll.delete(where={"doc_id": doc_id})
        after = coll.count()
        return before - after
    except Exception:
        return 0


def query_kb(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Return the top-k most relevant chunks for the question."""
    if not question.strip():
        return []
    coll = _get_collection()
    if coll.count() == 0:
        return []
    q_emb = _embed([question])[0]
    res = coll.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )
    out: List[Dict[str, Any]] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for d, m, dist in zip(docs, metas, dists):
        out.append({"text": d, "metadata": m, "distance": float(dist)})
    return out


def build_rag_context(question: str, top_k: int = TOP_K) -> str:
    """Assemble retrieved chunks into a single context string for the LLM."""
    chunks = query_kb(question, top_k=top_k)
    if not chunks:
        return ""
    blocks = []
    for i, c in enumerate(chunks, 1):
        src = c["metadata"].get("filename", "?")
        blocks.append(f"[{i}] (source: {src})\n{c['text']}")
    return "\n\n".join(blocks)
