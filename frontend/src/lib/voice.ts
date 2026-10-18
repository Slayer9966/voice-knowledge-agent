// Voice session: opens a WebSocket, captures mic at 16kHz, streams binary PCM,
// receives JSON events + binary TTS chunks, and plays them back smoothly.

// Vite resolves this to a URL for the worklet file at build/dev time.
import captureWorkletUrl from "../worklets/capture-processor.js?url";

export type VoiceEvent =
  | { type: "user_transcript"; text: string }
  | { type: "assistant_partial"; text: string }
  | { type: "assistant_final"; text: string }
  | { type: "tts_start"; sample_rate: number }
  | { type: "tts_end" }
  | { type: "pong" }
  | { type: "error"; message: string };

export interface VoiceDebugInfo {
  framesSent: number;
  bytesSent: number;
  lastRms: number;        // 0..1 — current mic loudness
  peakRms: number;        // session-wide peak
  wsState: string;        // "connecting" | "open" | "closing" | "closed"
  micSampleRate: number;
  micLabel: string;
  ttsChunksReceived: number;
}

export interface VoiceSessionOptions {
  language: "en" | "ur";
  onEvent: (e: VoiceEvent) => void;
  onStatus?: (s: string) => void;
  onDebug?: (info: VoiceDebugInfo) => void;
  onLog?: (line: string) => void;     // human-readable log lines for on-screen panel
  deviceId?: string;                  // mic deviceId — optional; default if omitted
}

export interface MicDevice {
  deviceId: string;
  label: string;
}

/**
 * List available microphone input devices. Labels are populated only after
 * the user has granted mic permission at least once in this origin.
 */
export async function listMics(): Promise<MicDevice[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices
    .filter((d) => d.kind === "audioinput")
    .map((d) => ({ deviceId: d.deviceId, label: d.label || "(permission needed)" }));
}

const log = (cb: ((s: string) => void) | undefined, msg: string) => {
  console.log("[voice]", msg);
  cb?.(msg);
};

export class VoiceSession {
  private ws: WebSocket | null = null;
  private ctx: AudioContext | null = null;
  private mic: MediaStreamAudioSourceNode | null = null;
  private worklet: AudioWorkletNode | null = null;
  private stream: MediaStream | null = null;

  // Playback
  private playCtx: AudioContext | null = null;
  private playSampleRate = 22050;
  private nextStartTime = 0;

  // Debug counters
  private debug: VoiceDebugInfo = {
    framesSent: 0,
    bytesSent: 0,
    lastRms: 0,
    peakRms: 0,
    wsState: "closed",
    micSampleRate: 0,
    micLabel: "",
    ttsChunksReceived: 0,
  };
  private debugTimer: number | null = null;

  constructor(private opts: VoiceSessionOptions) {}

  private emitDebug() {
    this.debug.wsState = this.wsStateName();
    this.opts.onDebug?.({ ...this.debug });
  }

  private wsStateName(): string {
    if (!this.ws) return "closed";
    return ["connecting", "open", "closing", "closed"][this.ws.readyState] || "?";
  }

  async start(): Promise<void> {
    log(this.opts.onLog, `start() — protocol=${location.protocol} host=${location.host}`);

    // ----- Pre-flight: secure context + getUserMedia availability -----
    if (typeof navigator === "undefined" || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const reason = !window.isSecureContext
        ? `Microphone access requires HTTPS (or localhost). This page is on ${location.protocol}//${location.host}, which the browser treats as insecure, so navigator.mediaDevices is undefined. Open the site through the Cloudflare HTTPS URL instead of the LAN IP.`
        : `navigator.mediaDevices.getUserMedia is unavailable in this browser.`;
      log(this.opts.onLog, "ERROR: " + reason);
      throw new Error(reason);
    }
    log(this.opts.onLog, `secureContext=${window.isSecureContext} userAgent=${navigator.userAgent.slice(0, 60)}…`);

    // ----- WebSocket -----
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${location.host}/ws/voice?lang=${this.opts.language}`;
    log(this.opts.onLog, `opening WebSocket: ${url}`);
    this.ws = new WebSocket(url);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen  = () => { log(this.opts.onLog, "ws: open"); this.opts.onStatus?.("connected"); this.emitDebug(); };
    this.ws.onclose = (e) => { log(this.opts.onLog, `ws: close code=${e.code} reason=${e.reason || "(none)"}`); this.opts.onStatus?.("disconnected"); this.emitDebug(); };
    this.ws.onerror = () => { log(this.opts.onLog, "ws: error"); this.opts.onStatus?.("error"); this.emitDebug(); };
    this.ws.onmessage = (ev) => this.onMessage(ev);

    await new Promise<void>((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("WebSocket timeout (5s)")), 5000);
      this.ws!.addEventListener("open",  () => { clearTimeout(t); resolve(); }, { once: true });
      this.ws!.addEventListener("error", () => { clearTimeout(t); reject(new Error("WebSocket error during connect")); }, { once: true });
    });
    log(this.opts.onLog, "ws connected");

    // ----- Mic capture -----
    log(this.opts.onLog, `requesting mic permission… deviceId=${this.opts.deviceId || "(default)"}`);
    try {
      const audioConstraints: MediaTrackConstraints = {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      };
      if (this.opts.deviceId) {
        audioConstraints.deviceId = { exact: this.opts.deviceId };
      }
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
    } catch (err: any) {
      const msg = `getUserMedia failed: ${err?.name || ""} ${err?.message || err}`;
      log(this.opts.onLog, "ERROR: " + msg);
      throw new Error(msg);
    }
    const track = this.stream.getAudioTracks()[0];
    const settings = track?.getSettings() || {};
    this.debug.micLabel = track?.label || "(unknown mic)";
    this.debug.micSampleRate = (settings as any).sampleRate || 0;
    log(this.opts.onLog, `mic granted: "${this.debug.micLabel}" sampleRate=${this.debug.micSampleRate || "?"}`);

    this.ctx = new AudioContext();
    log(this.opts.onLog, `AudioContext sampleRate=${this.ctx.sampleRate} state=${this.ctx.state}`);
    if (this.ctx.state === "suspended") {
      try { await this.ctx.resume(); log(this.opts.onLog, "AudioContext resumed"); } catch {}
    }

    await this.ctx.audioWorklet.addModule(captureWorkletUrl);
    log(this.opts.onLog, "audio worklet module loaded");

    this.mic = this.ctx.createMediaStreamSource(this.stream);
    this.worklet = new AudioWorkletNode(this.ctx, "capture-processor", {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: { targetRate: 16000 },
    });
    this.worklet.port.onmessage = (ev: MessageEvent<ArrayBuffer>) => {
      const buf = ev.data;
      // Compute RMS (mic level) so we can show "is mic alive" visually.
      const i16 = new Int16Array(buf);
      let sumSq = 0;
      for (let i = 0; i < i16.length; i++) {
        const s = i16[i] / 32768;
        sumSq += s * s;
      }
      const rms = i16.length ? Math.sqrt(sumSq / i16.length) : 0;
      this.debug.lastRms = rms;
      if (rms > this.debug.peakRms) this.debug.peakRms = rms;
      this.debug.framesSent++;
      this.debug.bytesSent += buf.byteLength;

      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(buf);
      } else if (this.debug.framesSent % 25 === 0) {
        log(this.opts.onLog, `WARN: ws not open (${this.wsStateName()}), dropping frame ${this.debug.framesSent}`);
      }
    };
    this.mic.connect(this.worklet);
    log(this.opts.onLog, "mic → worklet pipeline connected, streaming…");

    // Periodic debug emit so UI updates the meter ~5×/sec.
    this.debugTimer = window.setInterval(() => this.emitDebug(), 200);

    // Playback context.
    this.playCtx = new AudioContext();
    this.nextStartTime = this.playCtx.currentTime;
    log(this.opts.onLog, `playback ctx sampleRate=${this.playCtx.sampleRate} state=${this.playCtx.state}`);
    if (this.playCtx.state === "suspended") {
      try { await this.playCtx.resume(); } catch {}
    }

    this.opts.onStatus?.("listening");
    this.emitDebug();
  }

  setLanguage(lang: "en" | "ur") {
    this.opts.language = lang;
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "config", language: lang }));
    }
  }

  interrupt() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "interrupt" }));
    }
    this.flushPlayback();
  }

  async stop() {
    log(this.opts.onLog, "stop()");
    if (this.debugTimer) { clearInterval(this.debugTimer); this.debugTimer = null; }
    this.flushPlayback();
    try { this.worklet?.disconnect(); } catch {}
    try { this.mic?.disconnect(); } catch {}
    this.stream?.getTracks().forEach((t) => t.stop());
    try { await this.ctx?.close(); } catch {}
    try { this.ws?.close(); } catch {}
    this.ws = null;
    this.opts.onStatus?.("stopped");
    this.emitDebug();
  }

  // -------- internal --------
  private onMessage(ev: MessageEvent) {
    if (typeof ev.data === "string") {
      let obj: VoiceEvent;
      try { obj = JSON.parse(ev.data); } catch { return; }
      log(this.opts.onLog, `ws ← event: ${obj.type}` + ("text" in obj && obj.text ? ` "${String(obj.text).slice(0, 60)}"` : ""));
      if (obj.type === "tts_start") {
        this.playSampleRate = obj.sample_rate;
        this.nextStartTime = this.playCtx?.currentTime ?? 0;
      }
      this.opts.onEvent(obj);
    } else if (ev.data instanceof ArrayBuffer) {
      this.debug.ttsChunksReceived++;
      this.enqueuePcm(ev.data);
    }
  }

  private enqueuePcm(buf: ArrayBuffer) {
    if (!this.playCtx) return;
    const int16 = new Int16Array(buf);
    if (int16.length === 0) return;
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

    const audioBuf = this.playCtx.createBuffer(1, float32.length, this.playSampleRate);
    audioBuf.copyToChannel(float32, 0);

    const src = this.playCtx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(this.playCtx.destination);

    const now = this.playCtx.currentTime;
    const startAt = Math.max(now + 0.02, this.nextStartTime);
    src.start(startAt);
    this.nextStartTime = startAt + audioBuf.duration;
  }

  private flushPlayback() {
    if (!this.playCtx) return;
    // Cheapest "stop everything": close and recreate context.
    try { this.playCtx.close(); } catch {}
    this.playCtx = new AudioContext();
    this.nextStartTime = this.playCtx.currentTime;
  }
}
