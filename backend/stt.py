"""Speech-to-text using faster-whisper."""
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        print(
            f"[stt] loading whisper: model={WHISPER_MODEL} "
            f"device={WHISPER_DEVICE} compute_type={WHISPER_COMPUTE_TYPE}"
        )
        _model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _model


def transcribe(pcm16k: np.ndarray, language: str | None = None) -> str:
    """Transcribe a 16kHz mono float32 numpy array.

    `language` should be an ISO-639-1 code: "en", "ur" (Urdu), or None for auto.
    """
    if pcm16k.size == 0:
        return ""
    if pcm16k.dtype != np.float32:
        pcm16k = pcm16k.astype(np.float32)
    # Whisper expects values in [-1, 1].
    if np.abs(pcm16k).max() > 1.0:
        pcm16k = pcm16k / 32768.0

    segments, _info = get_model().transcribe(
        pcm16k,
        language=language,
        vad_filter=False,           # we run our own VAD upstream
        beam_size=1,                # latency-first; bump to 5 for accuracy
        condition_on_previous_text=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()
