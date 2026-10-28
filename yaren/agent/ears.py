"""Yaren's ears: voice activity detection (Silero VAD, CPU, one per connection) and speech recognition
(Whisper large-v3-turbo, GPU, shared).

Measured on the RTX 5090 (2026-10-23): ~50 ms to transcribe a 4-6 s utterance, 1.6 GB of VRAM, and the
language (Turkish / English) detected correctly on every test sentence.
"""

import logging
import re
import threading
import time

import numpy as np
import torch

log = logging.getLogger("yaren.ears")
SR = 16000
FRAME = 512                   # Silero VAD's frame at 16 kHz (32 ms)

# Whisper's well-known hallucinations on noise/silence (Turkish ones come from subtitled training data).
HALLUCINATIONS = [
    "altyazı", "izlediğiniz için teşekkür", "abone olmayı unutmayın", "abone ol", "beğenmeyi unutmayın",
    "thank you for watching", "thanks for watching", "subtitles by", "amara.org", "please subscribe",
]  # not "teşekkürler": people do say thanks to a robot friend, and VAD already gates silence out


class Segmenter:
    """Cuts a 16 kHz int16 stream into utterances with Silero VAD.

    feed() returns events: ("start",) when speech begins, ("end", float32 audio) when it ends after
    `silence_ms` of silence, ("drop",) for blips shorter than `min_speech_ms`.
    """

    def __init__(self, threshold=0.5, neg_threshold=0.35, min_speech_ms=250, silence_ms=550,
                 pre_roll_ms=300, max_s=25.0):
        from silero_vad import load_silero_vad

        self.model = load_silero_vad()
        self.th, self.neg = threshold, neg_threshold
        self.min_speech = int(min_speech_ms / 32)
        self.silence = int(silence_ms / 32)
        self.pre_roll = int(pre_roll_ms / 32)
        self.max_frames = int(max_s * SR / FRAME)
        self.buf = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self):
        self.model.reset_states()
        self.ring = []            # frames before speech starts (pre-roll)
        self.speech = []          # frames of the current utterance
        self.in_speech = False
        self.voiced = 0
        self.quiet = 0
        self.level = 0.0

    def feed(self, pcm: bytes, forced=False):
        """forced=True (push-to-talk held): every frame counts as speech, VAD is not consulted."""
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self.buf = np.concatenate([self.buf, x])
        events = []
        while len(self.buf) >= FRAME:
            fr, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            self.level = float(np.sqrt(np.mean(fr * fr)))
            p = 1.0 if forced else float(self.model(torch.from_numpy(fr), SR))
            if not self.in_speech:
                self.ring.append(fr)
                self.ring = self.ring[-self.pre_roll:]
                self.voiced = self.voiced + 1 if p >= self.th else 0
                if self.voiced >= 2:
                    self.in_speech, self.quiet = True, 0
                    self.speech = list(self.ring)
                    self.ring = []
                    events.append(("start",))
            else:
                self.speech.append(fr)
                if p >= self.th:
                    self.voiced += 1
                    self.quiet = 0
                elif p < self.neg:
                    self.quiet += 1
                if (self.quiet >= self.silence and not forced) or len(self.speech) >= self.max_frames:
                    events.append(self._close())
        return events

    def finish(self):
        """End of push-to-talk: close whatever was said."""
        if self.in_speech:
            return [self._close()]
        self.reset()
        return []

    def _close(self):
        audio = np.concatenate(self.speech) if self.speech else np.zeros(0, np.float32)
        voiced = self.voiced
        self.model.reset_states()
        self.in_speech, self.speech, self.voiced, self.quiet = False, [], 0, 0
        if voiced < self.min_speech:
            return ("drop",)
        return ("end", audio)


class Ears:
    def __init__(self, model_dir, device="cuda"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()   # "Both max_new_tokens and max_length..." on every utterance
        t = time.time()
        self.proc = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.float16).to(device).eval()
        self.device = device
        tok = self.proc.tokenizer
        self.lang_ids = {lang: tok.convert_tokens_to_ids(f"<|{lang}|>") for lang in ("tr", "en")}
        self.lock = threading.Lock()
        self.transcribe(np.zeros(SR, np.float32), "auto")  # warm-up
        log.info("Whisper ready in %.1f s", time.time() - t)

    @torch.inference_mode()
    def transcribe(self, audio, lang="auto"):
        """audio: float32 mono 16 kHz. lang: 'tr', 'en' or 'auto' (Turkish vs English only)."""
        t = time.perf_counter()
        with self.lock:
            feats = self.proc(audio, sampling_rate=SR, return_tensors="pt").input_features
            feats = feats.to(self.device, torch.float16)
            enc = self.model.model.encoder(feats)
            if lang not in self.lang_ids:
                start = torch.tensor([[self.model.config.decoder_start_token_id]], device=self.device)
                logits = self.model(encoder_outputs=enc, decoder_input_ids=start).logits[0, -1]
                lang = max(self.lang_ids, key=lambda k: logits[self.lang_ids[k]].item())
            out = self.model.generate(encoder_outputs=enc, language=lang, task="transcribe",
                                      max_new_tokens=160)
        text = self.proc.batch_decode(out, skip_special_tokens=True)[0].strip()
        return {"text": clean_transcript(text), "raw": text, "lang": lang,
                "ms": round((time.perf_counter() - t) * 1000)}


def clean_transcript(text):
    t = text.strip().strip("'\"“”‘’ ")
    low = t.lower()
    if not re.search(r"\w", t):
        return ""
    for h in HALLUCINATIONS:
        if h in low and len(low) < len(h) + 25:
            return ""
    return t
