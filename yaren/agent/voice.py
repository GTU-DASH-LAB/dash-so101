"""Yaren's voice: Chatterbox Multilingual (Resemble AI, MIT, 23 languages incl. Turkish) on the GPU.

One voice speaks both languages: a voice is a set of Chatterbox conditionals made from a short
reference recording, computed once and reused for every sentence. Measured on the RTX 5090
(2026-10-23): ~5x faster than real time (RTF 0.20-0.23), 3.4 GB peak VRAM. Round-tripping the output
through Whisper, the built-in voice at cfg_weight 0.5 was the most intelligible in both languages
(character error 0.024 Turkish, 0.038 English), so it is the default.
"""

import json
import logging
import re
import shutil
import subprocess
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("yaren.voice")

TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
TR_WORDS = {"ve", "bir", "bu", "ama", "evet", "hayır", "tamam", "kanki", "şimdi", "için", "ben", "sen",
            "değil", "var", "yok", "çok", "ne", "mi", "mı", "mu", "mü", "da", "de", "ile", "gibi",
            "kalem", "kalemi", "kase", "kaseye", "deniyorum", "hadi", "olur", "oldu", "abi"}
EN_WORDS = {"the", "and", "you", "i", "is", "it", "to", "a", "of", "in", "that", "what", "okay", "yes",
            "no", "pen", "bowl", "buddy", "let's", "i'm", "can", "will", "sure", "done", "try"}


def guess_lang(text, default):
    if any(c in TR_CHARS for c in text):
        return "tr"
    words = re.findall(r"[\w']+", text.lower())
    tr = sum(w in TR_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    if tr > en:
        return "tr"
    if en > tr:
        return "en"
    return default


def _num_words(m, lang):
    from num2words import num2words

    s = m.group(0)
    try:
        if re.fullmatch(r"\d+[.,]\d+", s):
            a, b = re.split(r"[.,]", s)
            sep = " virgül " if lang == "tr" else " point "
            return num2words(int(a), lang=lang) + sep + " ".join(num2words(int(d), lang=lang) for d in b)
        return num2words(int(s), lang=lang)
    except Exception:  # noqa: BLE001
        return s


def speakable(text, lang):
    """What the TTS should read: no markdown, emoji or symbols; numbers spelled out."""
    t = re.sub(r"[*_#`>\[\]{}|~^]", " ", text)
    t = "".join(c for c in t if unicodedata.category(c)[0] != "S" or c in "°%")
    t = t.replace("°", " derece" if lang == "tr" else " degrees")
    t = t.replace("%", " yüzde " if lang == "tr" else " percent")
    t = re.sub(r"\d+(?:[.,]\d+)?", lambda m: _num_words(m, lang), t)
    return re.sub(r"\s+", " ", t).strip()


class Voice:
    def __init__(self, model_dir, ref_dir, custom_dir, device="cuda"):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        t = time.time()
        self.tts = ChatterboxMultilingualTTS.from_local(model_dir, device)
        self.sr = self.tts.sr
        self.lock = threading.Lock()
        self.custom_dir = Path(custom_dir)
        self.custom_dir.mkdir(parents=True, exist_ok=True)
        self.voices = {"yaren": {"name": "Yaren (default)", "conds": self.tts.conds, "kind": "builtin"}}
        refs = {"deniz": ("tr_m.flac", "Deniz - Turkish male sample"),
                "ada": ("en_f1.flac", "Ada - English female sample")}
        for vid, (fname, name) in refs.items():
            p = Path(ref_dir) / fname
            if p.exists():
                self._add(vid, name, p, "sample")
        for meta in sorted(self.custom_dir.glob("*.json")):
            info = json.loads(meta.read_text())
            wav = meta.with_suffix(".wav")
            if wav.exists():
                self._add(meta.stem, info.get("name", meta.stem), wav, "custom")
        self.synth("Merhaba.", "tr", "yaren")  # warm-up (the first call is ~5x slower)
        log.info("Chatterbox ready in %.1f s, voices: %s", time.time() - t, list(self.voices))

    def _add(self, vid, name, wav_path, kind):
        with self.lock:
            saved = self.tts.conds
            self.tts.prepare_conditionals(str(wav_path), exaggeration=0.5)
            conds, self.tts.conds = self.tts.conds, saved
        self.voices[vid] = {"name": name, "conds": conds, "kind": kind}

    def list(self):
        return [{"id": k, "name": v["name"], "kind": v["kind"]} for k, v in self.voices.items()]

    def add_recording(self, src_path, name):
        """Clone a voice from a recording (any format ffmpeg reads). Returns its id."""
        vid = "custom_" + time.strftime("%Y%m%d_%H%M%S")
        wav = self.custom_dir / f"{vid}.wav"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_path), "-ac", "1",
                            "-ar", str(self.sr), "-t", "20", str(wav)], capture_output=True, text=True)
        if r.returncode != 0 or not wav.exists():
            raise ValueError(f"could not read the recording: {r.stderr.strip()[:300]}")
        import soundfile as sf

        audio, _ = sf.read(wav)
        if len(audio) < self.sr * 3:
            wav.unlink()
            raise ValueError("recording too short: at least 3 s of speech is needed (5-10 s is best)")
        self._add(vid, name, wav, "custom")
        (self.custom_dir / f"{vid}.json").write_text(json.dumps({"name": name}, ensure_ascii=False))
        return vid

    def delete(self, vid):
        if self.voices.get(vid, {}).get("kind") != "custom":
            raise ValueError("only recorded voices can be deleted")
        del self.voices[vid]
        for ext in (".wav", ".json"):
            p = self.custom_dir / f"{vid}{ext}"
            if p.exists():
                p.unlink()

    @torch.inference_mode()
    def synth(self, text, lang, voice_id="yaren"):
        """float32 mono audio at self.sr, or None when there is nothing to say."""
        lang = lang if lang in ("tr", "en") else "en"
        t = speakable(text, lang)
        if not re.search(r"\w", t):
            return None
        v = self.voices.get(voice_id) or self.voices["yaren"]
        with self.lock:
            self.tts.conds = v["conds"]
            wav = self.tts.generate(t, language_id=lang, cfg_weight=0.5, exaggeration=0.5, temperature=0.8)
        audio = trim_tail(wav.squeeze(0).float().cpu().numpy(), self.sr, len(t))
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 0.99:
            audio = audio * (0.99 / peak)
        return audio


def trim_tail(a, sr, n_chars, gap_s=0.35, rate=16.0):
    """Chatterbox often keeps going after the sentence: a pause, then breaths, a repeat or babble
    ("Tamam, durdum. Duri."; one 82-character sentence came out 13.9 s long instead of ~5 s). Cut at the
    first pause >= gap_s that starts after 60% of the expected speech time (~16 characters/s), and never
    allow more than 1.5 s + 1 s per 8 characters. Checked with Whisper on 24 samples: the tail went, the
    sentence stayed."""
    fr = int(sr * 0.02)
    end = min(len(a), int((1.5 + n_chars / 8) * sr))
    rms = np.array([np.sqrt(np.mean(a[i:i + fr] ** 2)) for i in range(0, end - fr + 1, fr)])
    if len(rms):
        speech = rms > max(0.008, 0.08 * float(np.percentile(rms, 95)))
        need, run = int(gap_s / 0.02), 0
        for i, sp in enumerate(speech):
            run = 0 if sp else run + 1
            if run == need and (i - need + 1) * 0.02 >= 0.6 * n_chars / rate:
                end = min(end, int(((i - need + 1) * 0.02 + 0.12) * sr))
                break
    if end >= len(a):
        return a
    out = a[:end].copy()
    f = min(len(out), int(0.03 * sr))
    out[len(out) - f:] *= np.linspace(1, 0, f, dtype=out.dtype)
    return out


def ffmpeg_ok():
    return shutil.which("ffmpeg") is not None
