"""Real-time voice agent WebSocket handler.

Wire protocol (one connection per voice session):

  Client → Server:
    binary frames  : raw little-endian int16 PCM, mono, 16 000 Hz
    text  (JSON)   : {"type": "config", "language": "en"|"ur"}
                     {"type": "interrupt"}      – user clicked "stop speaking"
                     {"type": "ping"}

  Server → Client:
    text  (JSON)   : {"type": "user_transcript", "text": "..."}
                     {"type": "assistant_partial", "text": "..."}
                     {"type": "assistant_final",   "text": "..."}
                     {"type": "tts_start", "sample_rate": 22050}
                     {"type": "tts_end"}
                     {"type": "error", "message": "..."}
    binary frames  : raw little-endian int16 PCM, mono, at sample_rate
                     (TTS audio – queue and play in order)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import traceback
from typing import List, Dict

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

import llm
import rag
import stt
import tts
from vad import StreamingVAD


# Make sure debug lines actually print under uvicorn.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [agent] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")


# Sentence boundaries for English + Urdu.
_SENT_RE = re.compile(r"(.+?[\.!\?؟۔]+)\s+", re.DOTALL)


def _split_sentences(buf: str) -> tuple[list[str], str]:
    """Return (complete_sentences, leftover)."""
    out: list[str] = []
    while True:
        m = _SENT_RE.match(buf)
        if not m:
            break
        out.append(m.group(1).strip())
        buf = buf[m.end():]
    return out, buf


class VoiceSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.language: str = "en"
        self.history: List[Dict[str, str]] = []
        self.vad = StreamingVAD()
        self._tts_task: asyncio.Task | None = None
        self._gen_task: asyncio.Task | None = None
        # Background STT tasks (fire-and-forget). We keep references so the
        # event loop doesn't garbage-collect them mid-flight.
        self._stt_tasks: set[asyncio.Task] = set()
        self._closed = False
        # Debug counters.
        self._frames_in = 0
        self._samples_in = 0
        self._last_log = time.time()

    def _tts_active(self) -> bool:
        """True iff there's an in-flight reply / TTS playback to interrupt."""
        return (
            (self._gen_task is not None and not self._gen_task.done())
            or (self._tts_task is not None and not self._tts_task.done())
        )

    # ---------------- send helpers ----------------
    async def send_json(self, obj: dict) -> None:
        if self._closed:
            return
        try:
            await self.ws.send_text(json.dumps(obj, ensure_ascii=False))
        except Exception:
            self._closed = True

    async def send_audio(self, data: bytes) -> None:
        if self._closed:
            return
        try:
            await self.ws.send_bytes(data)
        except Exception:
            self._closed = True

    # ---------------- audio in ----------------
    async def handle_audio(self, raw: bytes) -> None:
        # int16 LE → float32 in [-1, 1]
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        self._frames_in += 1
        self._samples_in += pcm.size

        # Heartbeat every ~2 seconds so we know audio is flowing.
        now = time.time()
        if now - self._last_log > 2.0:
            rms = float(np.sqrt(np.mean(pcm ** 2))) if pcm.size else 0.0
            log.info(
                "audio in: %d frames, %.1f s, last-rms=%.3f, in_speech=%s",
                self._frames_in, self._samples_in / 16000.0, rms, self.vad._in_speech
            )
            self._last_log = now

        for kind, audio in self.vad.push(pcm):
            if kind == "start":
                # Only treat this as a barge-in if a reply is actively
                # being generated or spoken. Otherwise it's just the user
                # talking during silence → no need to cancel anything, and
                # we'll wait for STT to confirm it isn't background noise.
                if self._tts_active():
                    log.info("VAD: speech_start (barge-in → cancelling reply)")
                    await self._cancel_reply()
                else:
                    log.info("VAD: speech_start")
            elif kind == "end" and audio is not None and audio.size > 0:
                log.info("VAD: speech_end → utterance %.2f s", audio.size / 16000.0)
                # Fire-and-forget: don't block the receive loop while STT runs.
                # If we awaited here, audio frames pile up during STT and the
                # VAD then fires spurious `speech_start` events on the backlog,
                # which would cancel the reply we're about to produce.
                task = asyncio.create_task(self._handle_utterance(audio))
                self._stt_tasks.add(task)
                task.add_done_callback(self._stt_tasks.discard)

    async def _cancel_reply(self) -> None:
        for t in (self._gen_task, self._tts_task):
            if t and not t.done():
                t.cancel()
        self._gen_task = None
        self._tts_task = None

    async def _handle_utterance(self, audio: np.ndarray) -> None:
        loop = asyncio.get_running_loop()
        log.info("STT: transcribing %.2f s of audio (lang=%s) …", audio.size / 16000.0, self.language)
        t0 = time.time()
        try:
            text = await loop.run_in_executor(
                None, lambda: stt.transcribe(audio, language=self.language)
            )
        except Exception as e:
            log.exception("STT failed")
            await self.send_json({"type": "error", "message": f"STT error: {e}"})
            return
        text = (text or "").strip()
        log.info("STT: done in %.2f s → %r", time.time() - t0, text)
        if not text:
            log.info("STT returned empty text; ignoring utterance")
            return
        # Now that we've confirmed real speech, cancel any prior in-flight
        # reply (covers the case where TTS hadn't started yet so the
        # speech_start handler skipped the cancel).
        await self._cancel_reply()
        await self.send_json({"type": "user_transcript", "text": text})
        # Kick off the reply pipeline.
        self._gen_task = asyncio.create_task(self._respond(text))

    # ---------------- response pipeline ----------------
    async def _respond(self, question: str) -> None:
        loop = asyncio.get_running_loop()
        try:
            log.info("RAG: retrieving context for %r", question[:80])
            t0 = time.time()
            context = await loop.run_in_executor(
                None, lambda: rag.build_rag_context(question)
            )
            log.info("RAG: %d chars of context in %.2f s", len(context or ""), time.time() - t0)
            messages = llm.build_messages(
                question=question,
                context=context,
                language=self.language,
                history=self.history,
            )
            log.info("LLM: streaming reply…")

            # Sentence queue feeds the TTS task.
            sent_q: asyncio.Queue[str | None] = asyncio.Queue()
            self._tts_task = asyncio.create_task(self._tts_consumer(sent_q))

            full_reply = ""
            buf = ""

            def _gen():
                # Synchronous generator running in executor; we'll iterate here.
                return llm.stream_reply(messages)

            # Pump tokens from the sync generator without blocking the loop.
            it = await loop.run_in_executor(None, _gen)

            def _next_token(iterator):
                try:
                    return next(iterator)
                except StopIteration:
                    return None

            while True:
                token = await loop.run_in_executor(None, _next_token, it)
                if token is None:
                    break
                full_reply += token
                buf += token
                await self.send_json({"type": "assistant_partial", "text": token})
                sentences, buf = _split_sentences(buf)
                for s in sentences:
                    await sent_q.put(s)

            if buf.strip():
                await sent_q.put(buf.strip())
            await sent_q.put(None)  # close

            log.info("LLM: done, %d chars", len(full_reply))
            await self.send_json({"type": "assistant_final", "text": full_reply})
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": full_reply})

            if self._tts_task:
                try:
                    await self._tts_task
                except asyncio.CancelledError:
                    pass
        except asyncio.CancelledError:
            log.info("response cancelled (interrupted)")
        except Exception as e:
            log.error("response pipeline crashed: %s\n%s", e, traceback.format_exc())
            await self.send_json({"type": "error", "message": str(e)})

    async def _tts_consumer(self, q: asyncio.Queue) -> None:
        loop = asyncio.get_running_loop()
        announced = False
        log.info("TTS consumer started (lang=%s)", self.language)
        try:
            while True:
                sentence = await q.get()
                if sentence is None:
                    log.info("TTS consumer: end-of-stream sentinel")
                    break
                if not announced:
                    try:
                        sr = tts.sample_rate(self.language)
                    except Exception as e:
                        log.exception("TTS sample_rate() failed")
                        await self.send_json({"type": "error", "message": f"TTS init: {e}"})
                        return
                    log.info("TTS: sending tts_start sr=%d", sr)
                    await self.send_json({"type": "tts_start", "sample_rate": sr})
                    announced = True

                log.info("TTS: synthesizing sentence (%d chars)", len(sentence))
                # Generate chunks in a thread, ship them as they arrive.
                def _gen_chunks(text: str):
                    try:
                        return list(tts.synthesize_stream(text, language=self.language))
                    except Exception as e:
                        log.exception("TTS synthesize_stream raised")
                        return []

                chunks = await loop.run_in_executor(None, _gen_chunks, sentence)
                total = sum(len(c) for c in chunks)
                log.info("TTS: got %d chunks, %d bytes for sentence", len(chunks), total)
                for ch in chunks:
                    await self.send_audio(ch)
            if announced:
                await self.send_json({"type": "tts_end"})
            log.info("TTS consumer finished (announced=%s)", announced)
        except asyncio.CancelledError:
            log.info("TTS consumer cancelled")
            await self.send_json({"type": "tts_end"})
            raise
        except Exception:
            log.exception("TTS consumer crashed")
            raise


async def handle_voice_ws(ws: WebSocket) -> None:
    await ws.accept()
    session = VoiceSession(ws)
    # Optional language query param, e.g. /ws/voice?lang=ur
    lang = ws.query_params.get("lang", "en")
    session.language = "ur" if lang.startswith("ur") else "en"
    log.info("WS open: language=%s client=%s", session.language, ws.client)

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                log.info("WS disconnect (client closed)")
                break
            if "bytes" in msg and msg["bytes"] is not None:
                await session.handle_audio(msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                try:
                    obj = json.loads(msg["text"])
                except Exception:
                    continue
                t = obj.get("type")
                if t == "config":
                    new_lang = obj.get("language", "en")
                    session.language = "ur" if new_lang.startswith("ur") else "en"
                    session.history = []
                elif t == "interrupt":
                    await session._cancel_reply()
                elif t == "ping":
                    await session.send_json({"type": "pong"})
    except WebSocketDisconnect:
        log.info("WS disconnect (WebSocketDisconnect)")
    except Exception as e:
        log.exception("WS handler crashed: %s", e)
    finally:
        await session._cancel_reply()
        log.info("WS closed: total %d frames, %.1f s of audio", session._frames_in, session._samples_in / 16000.0)
