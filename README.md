# 🎙️ Voice Knowledge Agent

A bilingual voice assistant for asking questions about your own PDF documents. Upload files, speak in English or Urdu, and hear answers based on the retrieved passages. The application runs as a React frontend with a FastAPI backend and a local document index.

## 🚀 Features

### Document knowledge base

- Upload and remove PDFs from the browser, or ingest them from the command line.
- Extract and split text, including Urdu sentence boundaries.
- Embed passages with multilingual BGE-M3 and store them in a local ChromaDB collection.
- Retrieve relevant passages for each question and pass them to an Ollama-hosted language model.

### Voice conversation

- Capture microphone audio in the browser and stream it to the backend over a WebSocket.
- Detect speech turns with Silero VAD and transcribe them with faster-whisper.
- Answer in the selected language and start speech playback sentence by sentence.
- Interrupt a spoken reply, switch between English and Urdu, and ask follow-up questions.
- Choose Edge neural voices or local Piper speech synthesis.

## 🛠️ Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | React, TypeScript, Vite, Web Audio API, AudioWorklet |
| API and transport | FastAPI, WebSockets |
| Speech input | Silero VAD, faster-whisper |
| Retrieval | pypdf, LangChain text splitters, BGE-M3, ChromaDB |
| Language model | Ollama (Qwen 2.5 7B Instruct by default) |
| Speech output | Edge TTS by default; Piper as a local option |

## 🧠 How It Works

```text
PDFs → text extraction → chunks → BGE-M3 embeddings → ChromaDB
                                                       │
Microphone → AudioWorklet → WebSocket → VAD → Whisper  │
                                               │       │
                                               └→ retrieval
                                                      │
                                                 Ollama reply
                                                      │
                                         sentence-by-sentence TTS
                                                      │
                                             WebSocket → speaker
```

The model receives retrieved document passages and is instructed to answer only from that context. The backend keeps the last three conversation turns for follow-up questions.

## 📁 Project Structure

```text
voice-kb-agent/
├── backend/
│   ├── api.py              # HTTP endpoints and voice WebSocket
│   ├── agent_ws.py         # real-time conversation pipeline
│   ├── rag.py              # PDF ingestion and retrieval
│   ├── llm.py              # Ollama prompt and streamed responses
│   ├── stt.py              # faster-whisper transcription
│   ├── tts.py              # Edge TTS and Piper backends
│   ├── vad.py              # speech turn detection
│   ├── config.py           # environment settings
│   ├── ingest.py           # optional PDF ingestion CLI
│   ├── download_voices.py # Piper voice model downloader
│   └── requirements.txt
└── frontend/
    ├── src/App.tsx         # document and conversation UI
    ├── src/lib/voice.ts    # browser audio and WebSocket client
    ├── src/worklets/       # microphone capture worklet
    └── package.json
```

## ⚙️ Setup Instructions

The commands below use **Windows PowerShell**. You need Python 3.10 or 3.11, Node.js 18 or later, and [Ollama](https://ollama.com/download). Allow several gigabytes of free disk space for the language, embedding, and speech models. An NVIDIA GPU is optional.

### 1. Prepare the language model

Start Ollama, then download the default model:

```powershell
ollama pull qwen2.5:7b-instruct
```

If you choose another model, set `OLLAMA_MODEL` in `backend/.env` to the model name you downloaded.

### 2. Start the backend

From the project root:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe api.py
```

The API starts at `http://localhost:8000`. The first transcription and document ingestion may download the configured Whisper and BGE-M3 models.

If Python 3.11 is not installed, replace `py -3.11` with your installed Python 3.10 command.

### 3. Start the frontend

In a second PowerShell window, from the project root:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. Vite forwards `/api` and `/ws` requests to the backend on port 8000.

### 4. Use the agent

1. Upload one or more text-based PDFs in the **Knowledge Base** panel.
2. Select **English** or **Urdu** and choose a microphone if needed.
3. Click **Start**, grant microphone access, and ask a question about the documents.
4. Use **Interrupt** to stop a reply or **Stop** to end the session.

You can also ingest PDFs from a terminal inside `backend`:

```powershell
.\.venv\Scripts\python.exe ingest.py .\path\to\document.pdf
```

## 🔊 Speech Output Options

`TTS_ENGINE=edge` is the default. It uses Microsoft Edge neural voices and **requires an internet connection for speech synthesis**. Document storage, retrieval, transcription, and Ollama generation run locally after their models are downloaded.

For offline speech output on Windows, set `TTS_ENGINE=piper` in `backend/.env`, run `python download_voices.py` with the backend virtual environment, and place the Piper Windows executable plus its supporting files in `backend/piper/`. The configured English and Urdu voice models belong in `backend/voices/`. These downloaded binaries and models are intentionally excluded from Git.

## 📌 Notes

- Scanned or image-only PDFs need OCR before ingestion; the current pipeline extracts selectable text with `pypdf`.
- Uploaded PDFs, the ChromaDB index, voice models, Piper binaries, virtual environments, and local `.env` files are excluded from the repository.
- The API is designed for local use and does not include user authentication. Do not expose it directly to the public internet.
- Performance depends on the selected models and hardware; no fixed latency is promised.

## 📜 License

Licensed under the [MIT License](LICENSE) — use, modify, and distribute the code under its terms.

## 🙋 Author

**Syed Muhammad Faizan Ali**  
📍 Pakistan  
📧 [faizandev666@gmail.com](mailto:faizandev666@gmail.com)  
🔗 [GitHub](https://github.com/Slayer9966) | [LinkedIn](https://www.linkedin.com/in/faizanali-ml/)
