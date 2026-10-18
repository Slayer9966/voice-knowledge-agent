import { useEffect, useRef, useState } from "react";
import { VoiceSession, listMics, type VoiceEvent, type VoiceDebugInfo, type MicDevice } from "./lib/voice";

type Doc = { doc_id: string; filename: string; chunks: number };
type Turn = { role: "user" | "assistant"; text: string };
type Lang = "en" | "ur";

export default function App() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [uploading, setUploading] = useState(false);
  const [lang, setLang] = useState<Lang>("en");
  const [status, setStatus] = useState<string>("idle");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [debug, setDebug] = useState<VoiceDebugInfo | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [showDebug, setShowDebug] = useState(true);
  const [mics, setMics] = useState<MicDevice[]>([]);
  const [micId, setMicId] = useState<string>("");      // "" = browser default
  const sessionRef = useRef<VoiceSession | null>(null);

  async function refreshMics() {
    try {
      const list = await listMics();
      setMics(list);
    } catch (e) {
      console.error("listMics failed", e);
    }
  }

  useEffect(() => {
    void refreshMics();
    if (navigator.mediaDevices?.addEventListener) {
      const handler = () => void refreshMics();
      navigator.mediaDevices.addEventListener("devicechange", handler);
      return () => navigator.mediaDevices.removeEventListener("devicechange", handler);
    }
  }, []);

  function appendLog(line: string) {
    const ts = new Date().toLocaleTimeString();
    setLogLines((prev) => {
      const next = [...prev, `${ts}  ${line}`];
      // Keep last 80 lines.
      return next.length > 80 ? next.slice(next.length - 80) : next;
    });
  }

  // Keep a reference to the in-progress assistant turn so partial tokens
  // append to it instead of creating new turns.
  const partialRef = useRef<string>("");

  useEffect(() => { void refresh(); }, []);

  async function refresh() {
    const r = await fetch("/api/documents");
    const j = await r.json();
    setDocs(j.documents || []);
  }

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const fd = new FormData();
    for (const f of Array.from(files)) fd.append("files", f);
    setUploading(true);
    try {
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      const j = await r.json();
      console.log("upload result", j);
      await refresh();
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function onDelete(doc_id: string) {
    await fetch(`/api/documents/${doc_id}`, { method: "DELETE" });
    await refresh();
  }

  function onEvent(ev: VoiceEvent) {
    if (ev.type === "user_transcript") {
      setTurns((t) => [...t, { role: "user", text: ev.text }]);
      partialRef.current = "";
      setTurns((t) => [...t, { role: "assistant", text: "" }]);
    } else if (ev.type === "assistant_partial") {
      partialRef.current += ev.text;
      setTurns((t) => {
        const copy = [...t];
        for (let i = copy.length - 1; i >= 0; i--) {
          if (copy[i].role === "assistant") {
            copy[i] = { ...copy[i], text: partialRef.current };
            break;
          }
        }
        return copy;
      });
    } else if (ev.type === "assistant_final") {
      partialRef.current = "";
    } else if (ev.type === "error") {
      console.error("agent error:", ev.message);
    }
  }

  async function startVoice() {
    if (sessionRef.current) return;
    setLogLines([]);
    appendLog(`browser: ${navigator.userAgent.slice(0, 80)}`);
    appendLog(`origin: ${location.protocol}//${location.host}  secureContext=${window.isSecureContext}`);
    const s = new VoiceSession({
      language: lang,
      onEvent,
      onStatus: setStatus,
      onDebug: setDebug,
      onLog: appendLog,
      deviceId: micId || undefined,
    });
    sessionRef.current = s;
    try {
      await s.start();
      // Now that permission is granted, mic labels become readable.
      void refreshMics();
    } catch (err: any) {
      console.error(err);
      const msg = err?.message || String(err);
      setStatus("error: " + msg);
      appendLog("FATAL: " + msg);
      sessionRef.current = null;
    }
  }

  async function stopVoice() {
    await sessionRef.current?.stop();
    sessionRef.current = null;
  }

  function changeLang(l: Lang) {
    setLang(l);
    sessionRef.current?.setLanguage(l);
  }

  function interrupt() {
    sessionRef.current?.interrupt();
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", height: "100vh" }}>
      {/* Sidebar – knowledge base */}
      <aside style={{ borderRight: "1px solid #1f2329", padding: 20, overflow: "auto" }}>
        <h2 style={{ marginTop: 0 }}>Knowledge Base</h2>
        <label
          style={{
            display: "inline-block",
            padding: "8px 14px",
            background: "#2c5cff",
            borderRadius: 8,
            cursor: "pointer",
            opacity: uploading ? 0.6 : 1,
          }}
        >
          {uploading ? "Uploading…" : "+ Upload PDFs"}
          <input
            type="file"
            accept="application/pdf"
            multiple
            disabled={uploading}
            onChange={onUpload}
            style={{ display: "none" }}
          />
        </label>

        <h3 style={{ marginTop: 28, fontSize: 14, opacity: 0.7 }}>Indexed</h3>
        {docs.length === 0 && (
          <p style={{ opacity: 0.6, fontSize: 14 }}>
            Nothing yet. Upload some PDFs to get started.
          </p>
        )}
        <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
          {docs.map((d) => (
            <li
              key={d.doc_id}
              style={{
                padding: "10px 12px",
                marginBottom: 6,
                background: "#13171c",
                borderRadius: 8,
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 8,
              }}
            >
              <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {d.filename}
                <div style={{ fontSize: 12, opacity: 0.5 }}>{d.chunks} chunks</div>
              </div>
              <button
                onClick={() => onDelete(d.doc_id)}
                style={{
                  background: "transparent",
                  border: "1px solid #3a3f47",
                  color: "#e6e8eb",
                  padding: "4px 8px",
                  borderRadius: 6,
                  cursor: "pointer",
                }}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      </aside>

      {/* Main – voice chat */}
      <main style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "16px 24px",
            borderBottom: "1px solid #1f2329",
          }}
        >
          <h1 style={{ margin: 0, fontSize: 20 }}>Voice Knowledge Agent</h1>
          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontSize: 12, opacity: 0.6 }}>{status}</span>
            <select
              value={micId}
              onChange={(e) => setMicId(e.target.value)}
              disabled={!!sessionRef.current}
              title="Microphone (set before clicking Start)"
              style={{
                background: "#13171c",
                color: "#e6e8eb",
                border: "1px solid #2a2f37",
                padding: "6px 10px",
                borderRadius: 6,
                maxWidth: 240,
              }}
            >
              <option value="">Default mic</option>
              {mics.map((m) => (
                <option key={m.deviceId} value={m.deviceId}>{m.label || m.deviceId.slice(0, 8)}</option>
              ))}
            </select>
            <select
              value={lang}
              onChange={(e) => changeLang(e.target.value as Lang)}
              style={{
                background: "#13171c",
                color: "#e6e8eb",
                border: "1px solid #2a2f37",
                padding: "6px 10px",
                borderRadius: 6,
              }}
            >
              <option value="en">English</option>
              <option value="ur">Urdu (اردو)</option>
            </select>
            {!sessionRef.current ? (
              <button
                onClick={startVoice}
                style={{
                  background: "#2cba5f",
                  color: "white",
                  border: 0,
                  padding: "8px 14px",
                  borderRadius: 8,
                  cursor: "pointer",
                }}
              >
                Start
              </button>
            ) : (
              <>
                <button
                  onClick={interrupt}
                  style={{
                    background: "#5a5f67",
                    color: "white",
                    border: 0,
                    padding: "8px 14px",
                    borderRadius: 8,
                    cursor: "pointer",
                  }}
                >
                  Interrupt
                </button>
                <button
                  onClick={stopVoice}
                  style={{
                    background: "#c83b3b",
                    color: "white",
                    border: 0,
                    padding: "8px 14px",
                    borderRadius: 8,
                    cursor: "pointer",
                  }}
                >
                  Stop
                </button>
              </>
            )}
          </div>
        </header>

        <div style={{ flex: 1, overflow: "auto", padding: 24 }}>
          {turns.length === 0 && (
            <p style={{ opacity: 0.5, textAlign: "center", marginTop: 80 }}>
              Click <strong>Start</strong> and ask a question about your documents.
              <br /> The agent listens continuously — just talk.
            </p>
          )}
          {turns.map((t, i) => (
            <div
              key={i}
              style={{
                margin: "12px 0",
                display: "flex",
                justifyContent: t.role === "user" ? "flex-end" : "flex-start",
              }}
            >
              <div
                dir={lang === "ur" ? "rtl" : "ltr"}
                style={{
                  maxWidth: "70%",
                  padding: "10px 14px",
                  background: t.role === "user" ? "#2c5cff" : "#1a1f26",
                  borderRadius: 12,
                  whiteSpace: "pre-wrap",
                  lineHeight: 1.5,
                }}
              >
                {t.text || (t.role === "assistant" ? "…" : "")}
              </div>
            </div>
          ))}
        </div>

        {/* ----- Debug panel ----- */}
        <div style={{ borderTop: "1px solid #1f2329", background: "#0c0f13", padding: "10px 16px", fontSize: 12, fontFamily: "ui-monospace, monospace" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <button
              onClick={() => setShowDebug((v) => !v)}
              style={{ background: "transparent", border: "1px solid #2a2f37", color: "#e6e8eb", padding: "3px 8px", borderRadius: 4, cursor: "pointer", fontSize: 11 }}
            >
              {showDebug ? "▼" : "▶"} debug
            </button>
            {/* Mic level meter — always visible. */}
            <div style={{ display: "flex", alignItems: "center", gap: 6, flex: "0 0 220px" }}>
              <span style={{ opacity: 0.6 }}>mic</span>
              <div style={{ flex: 1, height: 8, background: "#1a1f26", borderRadius: 4, overflow: "hidden" }}>
                <div
                  style={{
                    height: "100%",
                    width: `${Math.min(100, (debug?.lastRms ?? 0) * 400)}%`,
                    background: (debug?.lastRms ?? 0) > 0.02 ? "#2cba5f" : "#3a3f47",
                    transition: "width 80ms linear",
                  }}
                />
              </div>
              <span style={{ opacity: 0.6, width: 60 }}>
                {debug ? (debug.lastRms * 100).toFixed(1) + "%" : "—"}
              </span>
            </div>
            <span style={{ opacity: 0.7 }}>
              ws: <b style={{ color: debug?.wsState === "open" ? "#2cba5f" : "#c83b3b" }}>{debug?.wsState ?? "—"}</b>
              {" · "}frames: <b>{debug?.framesSent ?? 0}</b>
              {" · "}peak: <b>{debug ? (debug.peakRms * 100).toFixed(0) + "%" : "—"}</b>
              {" · "}TTS chunks: <b>{debug?.ttsChunksReceived ?? 0}</b>
            </span>
          </div>
          {showDebug && (
            <>
              {debug?.micLabel && (
                <div style={{ marginTop: 6, opacity: 0.6 }}>
                  mic device: {debug.micLabel} {debug.micSampleRate ? `@ ${debug.micSampleRate} Hz` : ""}
                </div>
              )}
              <pre
                style={{
                  marginTop: 6,
                  maxHeight: 140,
                  overflow: "auto",
                  background: "#13171c",
                  padding: 8,
                  borderRadius: 4,
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                }}
              >
                {logLines.length === 0 ? "(no log yet — click Start)" : logLines.join("\n")}
              </pre>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
