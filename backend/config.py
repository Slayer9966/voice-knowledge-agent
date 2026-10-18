"""Central config loaded from environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env is optional – if present, it is loaded; otherwise OS env is used.
try:
    load_dotenv()
except Exception:
    pass


def _env(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default


# --- Ollama ---
OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# --- Whisper ---
WHISPER_MODEL = _env("WHISPER_MODEL", "small")
WHISPER_DEVICE = _env("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = _env("WHISPER_COMPUTE_TYPE", "int8")

# --- Embeddings ---
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-m3")

# --- Storage ---
DATA_DIR = Path(_env("DATA_DIR", "./data")).resolve()
CHROMA_DIR = Path(_env("CHROMA_DIR", str(DATA_DIR / "chroma"))).resolve()
PDF_DIR = Path(_env("PDF_DIR", str(DATA_DIR / "pdfs"))).resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

# --- TTS engine ---
# "edge"  → Microsoft Edge neural TTS (cloud, no API key, very natural)
# "piper" → local Piper standalone binary (offline, lower quality)
TTS_ENGINE = _env("TTS_ENGINE", "edge").lower()

# --- Edge TTS voices (used when TTS_ENGINE=edge) ---
# Browse all voices: `edge-tts --list-voices`
# Recommended:
#   English: en-US-JennyNeural, en-US-AriaNeural, en-GB-SoniaNeural
#   Urdu:    ur-PK-UzmaNeural (female), ur-PK-AsadNeural (male)
EDGE_VOICE_EN = _env("EDGE_VOICE_EN", "en-US-JennyNeural")
EDGE_VOICE_UR = _env("EDGE_VOICE_UR", "ur-PK-UzmaNeural")

# --- Piper voices (used when TTS_ENGINE=piper) ---
PIPER_VOICE_EN = _env("PIPER_VOICE_EN", "./voices/en_US-amy-medium.onnx")
PIPER_VOICE_UR = _env("PIPER_VOICE_UR", "./voices/ur_PK-fasih-medium.onnx")

# --- Piper standalone Windows binary ---
PIPER_EXE = _env("PIPER_EXE", "./piper/piper.exe")
ESPEAK_DATA_DIR = _env("ESPEAK_DATA_DIR", "./piper/espeak-ng-data")

# --- RAG ---
TOP_K = int(_env("TOP_K", "4"))
CHUNK_SIZE = int(_env("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "120"))

COLLECTION_NAME = "knowledge_base"
