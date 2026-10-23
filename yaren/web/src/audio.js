// Microphone capture (16 kHz int16 frames for the server's VAD) and playback of Yaren's speech.

export class Mic {
  constructor(onFrame) {
    this.onFrame = onFrame; // (Int16Array, rms) => void
    this.level = 0;
    this.stream = null;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.ctx = new AudioContext();
    await this.ctx.audioWorklet.addModule("/mic-worklet.js");
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "mic-capture");
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    src.connect(this.node).connect(mute).connect(this.ctx.destination);
    this.node.port.onmessage = (e) => {
      const f = e.data;
      let s = 0;
      const pcm = new Int16Array(f.length);
      for (let i = 0; i < f.length; i++) {
        const v = Math.max(-1, Math.min(1, f[i]));
        s += v * v;
        pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
      }
      this.level = Math.sqrt(s / f.length);
      this.onFrame(pcm, this.level);
    };
  }

  // A MediaRecorder on the same microphone, for cloning a voice.
  recorder() {
    const types = ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus", ""];
    const mimeType = types.find((t) => t === "" || MediaRecorder.isTypeSupported(t));
    return new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
  }

  stop() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close();
  }
}

export class Player {
  constructor(onPlaying) {
    this.onPlaying = onPlaying; // (bool) => void
    this.ctx = new AudioContext();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.connect(this.ctx.destination);
    this.data = new Uint8Array(this.analyser.fftSize);
    this.sources = new Set();
    this.playhead = 0;
  }

  resume() {
    return this.ctx.resume();
  }

  play(int16, sampleRate) {
    const n = int16.length;
    if (!n) return;
    const buf = this.ctx.createBuffer(1, n, sampleRate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < n; i++) ch[i] = int16[i] / 32768;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.analyser);
    const start = Math.max(this.ctx.currentTime + 0.03, this.playhead);
    src.start(start);
    this.playhead = start + buf.duration;
    const wasIdle = this.sources.size === 0;
    this.sources.add(src);
    if (wasIdle) this.onPlaying(true);
    src.onended = () => {
      this.sources.delete(src);
      if (this.sources.size === 0) {
        this.playhead = 0;
        this.onPlaying(false);
      }
    };
  }

  stop() {
    const had = this.sources.size > 0;
    for (const s of this.sources) {
      s.onended = null;
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
    }
    this.sources.clear();
    this.playhead = 0;
    if (had) this.onPlaying(false);
  }

  level() {
    if (this.sources.size === 0) return 0;
    this.analyser.getByteTimeDomainData(this.data);
    let s = 0;
    for (let i = 0; i < this.data.length; i++) {
      const v = (this.data[i] - 128) / 128;
      s += v * v;
    }
    return Math.sqrt(s / this.data.length);
  }
}
