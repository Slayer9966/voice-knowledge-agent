"""Download Piper voices for English + Urdu into ./voices/.

Run once after `pip install -r requirements.txt`:
    python download_voices.py
"""
import os
import urllib.request
from pathlib import Path

VOICES = {
    # English – Amy, US, medium quality
    "en_US-amy-medium.onnx":      "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
    "en_US-amy-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    # Urdu – Fasih, Pakistan, medium quality (rhasspy renamed "gyan" -> "fasih")
    "ur_PK-fasih-medium.onnx":      "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/fasih/medium/ur_PK-fasih-medium.onnx",
    "ur_PK-fasih-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/fasih/medium/ur_PK-fasih-medium.onnx.json",
}

out_dir = Path(__file__).parent / "voices"
out_dir.mkdir(exist_ok=True)

for name, url in VOICES.items():
    dest = out_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ already have {name}")
        continue
    print(f"  ↓ {name}")
    urllib.request.urlretrieve(url, dest)
print(f"\nDone. Voices saved to {out_dir.resolve()}")
