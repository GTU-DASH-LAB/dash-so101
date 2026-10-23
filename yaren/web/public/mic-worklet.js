// Captures the microphone and resamples it to 16 kHz mono (linear interpolation), posting
// Float32Array chunks of 512 samples (32 ms, Silero VAD's frame) to the main thread.
class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.t = 0;
    this.prev = 0;
    this.out = new Float32Array(512);
    this.n = 0;
  }
  process(inputs) {
    const x = inputs[0] && inputs[0][0];
    if (!x || x.length === 0) return true;
    let t = this.t;
    while (t < x.length - 1) {
      const i = Math.floor(t);
      const f = t - i;
      const a = i < 0 ? this.prev : x[i];
      const b = x[i + 1];
      this.out[this.n++] = a + (b - a) * f;
      if (this.n === this.out.length) {
        this.port.postMessage(this.out.slice(0));
        this.n = 0;
      }
      t += this.ratio;
    }
    this.t = t - x.length;
    this.prev = x[x.length - 1];
    return true;
  }
}
registerProcessor("mic-capture", MicCapture);
