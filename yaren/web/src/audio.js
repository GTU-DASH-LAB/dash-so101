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

  // A phone that locked its screen can end the track or suspend the context: wake() checks both.
  get dead() {
    const t = this.stream?.getAudioTracks()[0];
    return !t || t.readyState === "ended";
  }

  resume() {
    return this.ctx?.state === "running" ? Promise.resolve() : this.ctx?.resume();
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
    this.gain = this.ctx.createGain(); // ducking: quieter while someone may be talking over her
    this.gain.connect(this.analyser);
    this.data = new Uint8Array(this.analyser.fftSize);
    this.sources = new Set();
    this.playhead = 0;
  }

  resume() {
    return this.ctx.state === "running" ? Promise.resolve() : this.ctx.resume();
  }

  duck(on) {
    const g = this.gain.gain;
    g.cancelScheduledValues(this.ctx.currentTime);
    g.setTargetAtTime(on ? 0.25 : 1, this.ctx.currentTime, 0.05);
  }

  play(int16, sampleRate) {
    const n = int16.length;
    if (!n) return;
    const buf = this.ctx.createBuffer(1, n, sampleRate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < n; i++) ch[i] = int16[i] / 32768;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.gain);
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
    this.gain.gain.cancelScheduledValues(this.ctx.currentTime);
    this.gain.gain.value = 1;
    if (had) this.onPlaying(false);
  }

  // A dance beat: not her voice, so it is never ducked, never stopped by audio_stop and never counts as
  // her speaking. onStart fires when its first sample should be audible: the arm starts dancing then.
  playMusic(int16, sampleRate, onStart) {
    this.stopMusic();
    if (!this.musicGain) {
      this.musicGain = this.ctx.createGain();
      this.musicGain.gain.value = 0.7;
      this.musicGain.connect(this.ctx.destination);
    }
    const buf = this.ctx.createBuffer(1, int16.length, sampleRate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < int16.length; i++) ch[i] = int16[i] / 32768;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.musicGain);
    const at = this.ctx.currentTime + 0.05;
    src.start(at);
    this.music = src;
    src.onended = () => { if (this.music === src) this.music = null; };
    const latency = this.ctx.outputLatency || this.ctx.baseLatency || 0;
    setTimeout(onStart, Math.max(0, (at - this.ctx.currentTime + latency) * 1000));
  }

  stopMusic() {
    if (!this.music) return;
    this.music.onended = null;
    try {
      this.music.stop();
    } catch {
      /* already stopped */
    }
    this.music = null;
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
