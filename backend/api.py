"""HTTP + WebSocket server.

HTTP:
    POST  /api/upload    multipart "files"    – ingest one or more PDFs
    GET   /api/documents                       – list ingested docs
    DELETE /api/documents/{doc_id}             – remove a doc from the KB
    POST  /api/ask       JSON {question, language} – text-only Q&A (no voice)

WebSocket:
    /ws/voice?lang=en|ur      – the real-time voice pipeline
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import llm
import rag
from agent_ws import handle_voice_ws
from config import PDF_DIR

app = FastAPI(title="Voice Knowledge Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    language: str = "en"


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            results.append({"filename": f.filename, "ok": False, "error": "Only .pdf files are supported"})
            continue
        dest = Path(PDF_DIR) / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        try:
            res = rag.ingest_pdf(dest, original_name=f.filename)
            results.append({"filename": f.filename, "ok": True, **res})
        except Exception as e:
            results.append({"filename": f.filename, "ok": False, "error": str(e)})
    return {"results": results}


@app.get("/api/documents")
def list_docs():
    return {"documents": rag.list_documents()}


@app.delete("/api/documents/{doc_id}")
def delete_doc(doc_id: str):
    removed = rag.delete_document(doc_id)
    return {"removed_chunks": removed}


@app.post("/api/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "question is empty")
    lang = "ur" if req.language.startswith("ur") else "en"
    context = rag.build_rag_context(req.question)
    messages = llm.build_messages(req.question, context, lang)
    text = "".join(llm.stream_reply(messages))
    return {"answer": text, "language": lang}


@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    await handle_voice_ws(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
