"""Streaming TTS.

Two backends are supported, selected by TTS_ENGINE in .env:

    TTS_ENGINE=edge   →  Microsoft Edge neural voices (cloud, no API key,
                          very natural quality, requires internet)
    TTS_ENGINE=piper  →  Local Piper standalone binary (offline, robotic)

Both expose the same `synthesize_stream(text, language) -> Iterator[bytes]`
yielding 16-bit little-endian mono PCM at the rate returned by
`sample_rate(language)`. The rest of the agent is engine-agnostic.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Iterator

from config import (
    TTS_ENGINE,
    EDGE_VOICE_EN,
    EDGE_VOICE_UR,
    PIPER_VOICE_EN,
    PIPER_VOICE_UR,
    PIPER_EXE,
    ESPEAK_DATA_DIR,
)

log = logging.getLogger("tts")


# ============================================================
# Edge TTS backend
# ============================================================

# Edge TTS streams MP3 at 24 kHz mono natively. We decode to int16 PCM and
# announce 24000 Hz to the client; the frontend's playback context handles it.
_EDGE_SAMPLE_RATE = 24000


def _edge_voice(language: str) -> str:
    return EDGE_VOICE_UR if language == "ur" else EDGE_VOICE_EN


async def _edge_collect_mp3(text: str, voice: str) -> bytes:
    import edge_tts  # imported lazily so a missing dep only breaks edge mode
    buf = io.BytesIO()
    comm = edge_tts.Communicate(text, voice)
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def _decode_mp3_to_pcm(mp3_bytes: bytes, target_rate: int) -> bytes:
    """Decode MP3 → 16-bit mono PCM at target_rate. Returns raw PCM bytes."""
    import miniaudio
    decoded = miniaudio.decode(
        mp3_bytes,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=target_rate,
    )
    # decoded.samples is an array.array of int16; bytes(...) gives LE int16.
    return bytes(decoded.samples)


def _edge_synthesize_stream(text: str, language: str) -> Iterator[bytes]:
    if not text.strip():
        log.info("edge-tts: empty text, skipping")
        return
    voice = _edge_voice(language)
    log.info("edge-tts: voice=%s lang=%s text=%r", voice, language, text[:80])

    try:
        mp3_bytes = asyncio.run(_edge_collect_mp3(text, voice))
    except Exception as e:
        log.exception("edge-tts synthesis failed")
        raise RuntimeError(f"edge-tts synthesis failed: {e}") from e

    if not mp3_bytes:
        log.warning("edge-tts returned 0 bytes (no audio for voice=%s)", voice)
        return

    try:
        pcm = _decode_mp3_to_pcm(mp3_bytes, _EDGE_SAMPLE_RATE)
    except Exception as e:
        log.exception("MP3 decode failed (mp3=%d bytes)", len(mp3_bytes))
        raise RuntimeError(f"MP3 decode failed: {e}") from e

    duration = len(pcm) / 2 / _EDGE_SAMPLE_RATE
    log.info("edge-tts: %d MP3 bytes → %d PCM bytes (%.2f s)", len(mp3_bytes), len(pcm), duration)

    # Yield in 4 KiB chunks to keep network frames small.
    chunk_size = 4096
    for i in range(0, len(pcm), chunk_size):
        yield pcm[i:i + chunk_size]


# ============================================================
# Piper backend (local fallback)
# ============================================================

_voice_meta: dict[str, dict] = {}


def _piper_voice_path(language: str) -> str:
    return PIPER_VOICE_UR if language == "ur" else PIPER_VOICE_EN


def _resolve(p: str) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = (Path(__file__).parent / path).resolve()
    return str(path)


def _piper_meta(language: str) -> dict:
    voice = _resolve(_piper_voice_path(language))
    if not os.path.exists(voice):
        raise FileNotFoundError(
            f"Piper voice file missing: {voice}. "
            f"Run `python download_voices.py` and check PIPER_VOICE_{language.upper()} in .env."
        )
    if voice not in _voice_meta:
        with open(voice + ".json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        _voice_meta[voice] = {
            "voice_path": voice,
            "sample_rate": cfg.get("audio", {}).get("sample_rate", 22050),
        }
    return _voice_meta[voice]


def _piper_synthesize_stream(text: str, language: str) -> Iterator[bytes]:
    if not text.strip():
        return

    meta = _piper_meta(language)
    piper_exe = _resolve(PIPER_EXE)
    espeak_data = _resolve(ESPEAK_DATA_DIR)

    if not os.path.exists(piper_exe):
        raise FileNotFoundError(f"piper.exe not found at {piper_exe}.")

    cmd = [
        piper_exe,
        "--model", meta["voice_path"],
        "--output-raw",
        "--espeak_data", espeak_data,
    ]

    log.info("piper: %r (%d chars, lang=%s)", text[:80], len(text), language)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(Path(piper_exe).parent),
    )
    total = 0
    try:
        proc.stdin.write(text.encode("utf-8"))
        proc.stdin.close()
        chunk_size = 4096
        while True:
            chunk = proc.stdout.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            yield chunk
    finally:
        try: proc.stdout.close()
        except Exception: pass
        try: rc = proc.wait(timeout=5)
        except Exception as e: rc = f"wait-failed:{e}"
        stderr_data = b""
        if proc.stderr is not None:
            try:
                stderr_data = proc.stderr.read() or b""
                proc.stderr.close()
            except Exception:
                pass
        if total == 0 or (isinstance(rc, int) and rc != 0):
            log.warning("piper: %d bytes, rc=%s, stderr=%r", total, rc,
                        stderr_data.decode("utf-8", errors="replace")[:500])
        else:
            log.info("piper: %d PCM bytes emitted (rc=%s)", total, rc)


# ============================================================
# Public API – dispatches based on TTS_ENGINE
# ============================================================

def sample_rate(language: str = "en") -> int:
    if TTS_ENGINE == "edge":
        return _EDGE_SAMPLE_RATE
    return _piper_meta(language)["sample_rate"]


def synthesize_stream(text: str, language: str = "en") -> Iterator[bytes]:
    if TTS_ENGINE == "edge":
        yield from _edge_synthesize_stream(text, language)
    elif TTS_ENGINE == "piper":
        yield from _piper_synthesize_stream(text, language)
    else:
        raise ValueError(f"Unknown TTS_ENGINE={TTS_ENGINE!r}; expected 'edge' or 'piper'.")
