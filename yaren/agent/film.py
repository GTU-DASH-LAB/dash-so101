"""Clips for the reels. robot_server.py films the wrist camera (run/film/*.mp4, picture only, wall-clock
timed); this module records what was heard meanwhile - Yaren's voice exactly as it went to the browser, the
app's microphone, a dance beat - and writes the finished clip with its sound to videos_for_editing/.
"""

import asyncio
import json
import logging
import time
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("yaren.film")
CLIPS = Path(__file__).resolve().parent.parent / "videos_for_editing"
SR = 24000            # the soundtrack's rate (Chatterbox's)
MIC_SR = 16000        # the app sends 512-sample frames at 16 kHz (web/public/mic-worklet.js)
MIC_GAIN = 0.9
MIC_GAP_S = 0.5       # a microphone frame this much later than the stream expects: it was off in between


def resample(x, sr_from, sr_to):
    if sr_from == sr_to or len(x) == 0:
        return x.astype(np.float32)
    n = int(round(len(x) * sr_to / sr_from))
    return np.interp(np.arange(n) * (sr_from / sr_to), np.arange(len(x)), x).astype(np.float32)


def place(out, x, i):
    """Mix x into out starting at sample i (either may stick out at the ends)."""
    a, b = max(0, i), min(len(out), i + len(x))
    if b > a:
        out[a:b] += x[a - i:b - i]


class Soundtrack:
    """Audio collected with wall-clock times (time.time(), like robot_server.py's frames) while filming."""

    def __init__(self):
        self.speech = []        # (start, float32 at its rate, rate): her sentences, one after another
        self.sounds = []        # (start, float32, rate): a dance beat, where it started
        self.mic = []           # (arrival, int16 bytes at MIC_SR)
        self.cuts = []          # when her speech was cut off (barge-in, STOP, hush)
        self.playhead = 0.0     # like the browser's Player: a sentence starts when the one before it ends

    def add_speech(self, audio, sr, now):
        start = max(now + 0.03, self.playhead)
        self.speech.append((start, np.asarray(audio, np.float32), sr))
        self.playhead = start + len(audio) / sr

    def add_sound(self, audio, sr, start):
        self.sounds.append((start, np.asarray(audio, np.float32), sr))

    def cut(self, now):
        self.cuts.append(now)
        self.playhead = 0.0

    def add_mic(self, pcm, now):
        self.mic.append((now, bytes(pcm)))

    def render(self, t0, seconds):
        """The track for a clip whose first frame was at t0 (float32 at SR, -1..1)."""
        out = np.zeros(max(0, int(round(seconds * SR))), np.float32)
        for start, x, sr in self.speech:
            x = resample(x, sr, SR)
            cut = next((c for c in self.cuts if start <= c < start + len(x) / SR), None)
            if cut is not None:
                x = x[:int((cut - start) * SR)]
            place(out, x, int(round((start - t0) * SR)))
        for start, x, sr in self.sounds:
            place(out, resample(x, sr, SR), int(round((start - t0) * SR)))
        # The microphone is one continuous stream: each frame goes right after the one before it, unless it
        # arrived clearly later than that (the microphone was off). Arrival marks a frame's end.
        cursor = None
        for at, pcm in self.mic:
            x = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768
            dur = len(x) / MIC_SR
            if cursor is None or at - dur - cursor > MIC_GAP_S:
                cursor = at - dur
            place(out, resample(x, MIC_SR, SR) * MIC_GAIN, int(round((cursor - t0) * SR)))
            cursor += dur
        peak = float(np.abs(out).max()) if len(out) else 0.0
        if peak > 0.99:
            out *= 0.99 / peak
        return out


def write_wav(path, audio):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def clip_name(t0, kind, label=None):
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(t0 or time.time()))
    return "_".join(x for x in (stamp, kind, label) if x)


async def finish_clip(raw, track, name, meta=None, out_dir=CLIPS):
    """raw: robot_server.py's /film stop result. Writes out_dir/<name>.mp4 with the sound, and a .json next
    to it; the picture-only file is removed. Returns the clip's file name."""
    src = Path(raw["file"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dst, k = out_dir / f"{name}.mp4", 1
    while dst.exists():
        k += 1
        dst = out_dir / f"{name}-{k}.mp4"
    audio = None
    if track is not None and raw.get("t0"):
        audio = await asyncio.get_running_loop().run_in_executor(None, track.render, raw["t0"], raw["seconds"])
    wav = src.with_suffix(".wav")
    if audio is not None and len(audio) and float(np.abs(audio).max()) > 1e-4:
        write_wav(wav, audio)
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-i", str(wav), "-map", "0:v", "-map", "1:a",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(dst)]
    else:
        audio = None
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    wav.unlink(missing_ok=True)
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg: {err.decode(errors='replace').strip()[-300:]}")
    info = {**(meta or {}), "file": dst.name, "seconds": raw.get("seconds"), "frames": raw.get("frames"),
            "dropped_frames": raw.get("dropped"), "sound": audio is not None,
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(raw.get("t0") or time.time()))}
    dst.with_suffix(".json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    src.unlink(missing_ok=True)
    log.info("clip %s: %.1f s%s", dst.name, raw.get("seconds") or 0, " with sound" if audio is not None else "")
    return dst.name


def relabel(name, label, out_dir=CLIPS):
    """A clip's label decided after it was written (a test verdict that came late): renames the .mp4 and its
    .json (the part after the last "_" of the name) and returns the new file name."""
    src = out_dir / name
    stem = src.stem.rsplit("_", 1)[0]
    dst, k = out_dir / f"{stem}_{label}.mp4", 1
    while dst.exists():
        k += 1
        dst = out_dir / f"{stem}_{label}-{k}.mp4"
    src.rename(dst)
    meta = src.with_suffix(".json")
    if meta.exists():
        info = json.loads(meta.read_text())
        info.update(file=dst.name, label=label)
        dst.with_suffix(".json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
        meta.unlink()
    return dst.name
