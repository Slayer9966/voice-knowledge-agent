"""Streaming voice-activity detection using Silero VAD."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

# silero-vad ships a small PyTorch model.
from silero_vad import load_silero_vad, VADIterator

SAMPLE_RATE = 16000
WINDOW = 512  # samples per VAD step at 16kHz (~32ms)


class StreamingVAD:
    """Wraps silero-vad's VADIterator for our 16k mono float32 stream.

    Feed PCM in arbitrary chunk sizes via `push`; it returns events:
        ("start",  None)            – speech starts
        ("end",    np.ndarray)      – speech ends, ndarray = full utterance audio
        (None,     None)            – no event this step
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_ms: int = 600,
        speech_pad_ms: int = 200,
    ):
        self._model = load_silero_vad()
        self._iter = VADIterator(
            self._model,
            threshold=threshold,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self._buf = np.zeros(0, dtype=np.float32)
        self._utt: list[np.ndarray] = []
        self._in_speech = False

    def reset(self) -> None:
        self._iter.reset_states()
        self._buf = np.zeros(0, dtype=np.float32)
        self._utt = []
        self._in_speech = False

    def push(self, pcm: np.ndarray) -> list[Tuple[str, Optional[np.ndarray]]]:
        events: list[Tuple[str, Optional[np.ndarray]]] = []
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        self._buf = np.concatenate([self._buf, pcm])

        while self._buf.size >= WINDOW:
            window = self._buf[:WINDOW]
            self._buf = self._buf[WINDOW:]

            # VADIterator expects a torch tensor.
            res = self._iter(torch.from_numpy(window), return_seconds=False)

            if self._in_speech:
                self._utt.append(window.copy())

            if res is not None:
                if "start" in res and not self._in_speech:
                    self._in_speech = True
                    self._utt = [window.copy()]
                    events.append(("start", None))
                elif "end" in res and self._in_speech:
                    self._in_speech = False
                    audio = (
                        np.concatenate(self._utt)
                        if self._utt
                        else np.zeros(0, dtype=np.float32)
                    )
                    self._utt = []
                    events.append(("end", audio))

        return events
