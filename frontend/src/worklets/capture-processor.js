// AudioWorklet: capture mic, downsample to 16kHz, post int16 PCM chunks.

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = (options.processorOptions && options.processorOptions.targetRate) || 16000;
    this.inRate = sampleRate; // global in worklet scope
    this.ratio = this.inRate / this.targetRate;
    this._acc = []; // accumulated input samples (mono float32)
    this._idx = 0;  // virtual position in input stream
    this._chunkMs = 40; // emit every ~40ms
    this._chunkSamplesOut = Math.round(this.targetRate * this._chunkMs / 1000);
    this._outBuf = new Float32Array(this._chunkSamplesOut);
    this._outPos = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];
    if (!ch) return true;

    // Push into accumulator, then resample by linear interpolation.
    const start = this._acc.length;
    for (let i = 0; i < ch.length; i++) this._acc.push(ch[i]);

    // Generate output samples at target rate.
    while (true) {
      const srcPos = this._idx * this.ratio;
      const i0 = Math.floor(srcPos);
      const i1 = i0 + 1;
      if (i1 >= this._acc.length) break;
      const t = srcPos - i0;
      const sample = this._acc[i0] * (1 - t) + this._acc[i1] * t;
      this._outBuf[this._outPos++] = sample;
      this._idx++;

      if (this._outPos >= this._chunkSamplesOut) {
        // Convert float32 → int16 LE.
        const int16 = new Int16Array(this._outPos);
        for (let k = 0; k < this._outPos; k++) {
          let s = Math.max(-1, Math.min(1, this._outBuf[k]));
          int16[k] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage(int16.buffer, [int16.buffer]);
        this._outPos = 0;
      }
    }

    // Trim accumulator (keep only what's still needed).
    const consumed = Math.floor(this._idx * this.ratio);
    if (consumed > 1024) {
      this._acc = this._acc.slice(consumed);
      this._idx -= consumed / this.ratio; // adjust virtual idx
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
