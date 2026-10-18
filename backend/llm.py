"""LLM wrapper around Ollama, with RAG-aware system prompt."""
from __future__ import annotations

from typing import AsyncIterator, Iterator, List, Dict

import ollama

from config import OLLAMA_HOST, OLLAMA_MODEL


SYSTEM_PROMPT_TEMPLATE = """You are a helpful voice assistant answering questions about the user's documents.

Rules:
- Use ONLY the information in the CONTEXT below. If the answer is not in the context, say so honestly in the user's language.
- Reply in {language_name}. Keep answers short and conversational – this is a voice interface, no markdown, no bullet lists, no code blocks.
- Cite the source filename inline when useful, e.g. "According to <filename>, ...".
- Never invent facts.

CONTEXT:
{context}
"""

LANG_NAMES = {"en": "English", "ur": "Urdu (اردو)"}


def build_messages(
    question: str,
    context: str,
    language: str,
    history: List[Dict[str, str]] | None = None,
) -> List[Dict[str, str]]:
    sys = SYSTEM_PROMPT_TEMPLATE.format(
        language_name=LANG_NAMES.get(language, "English"),
        context=context or "(no documents indexed yet)",
    )
    msgs: List[Dict[str, str]] = [{"role": "system", "content": sys}]
    if history:
        msgs.extend(history[-6:])  # keep last 3 turns
    msgs.append({"role": "user", "content": question})
    return msgs


_client: ollama.Client | None = None


def _client_inst() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=OLLAMA_HOST)
    return _client


def stream_reply(messages: List[Dict[str, str]]) -> Iterator[str]:
    """Stream tokens from Ollama as they arrive."""
    stream = _client_inst().chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=True,
        options={"temperature": 0.3},
    )
    for chunk in stream:
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            yield piece
