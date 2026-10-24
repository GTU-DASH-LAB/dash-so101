#!/usr/bin/env python3
"""End-to-end test of a running Yaren, talking to it like the browser does (no microphone needed).

    ~/envs/yaren/bin/python selftest.py [--url ws://127.0.0.1:8080] [--only N]

Use it against ./start.sh --dry (no arm!). Spoken test lines are synthesised once with Chatterbox's
"Deniz" sample voice (a different voice than Yaren's) into run/selftest/*.wav, then streamed in real
time as 16 kHz PCM, followed by silence so the server's VAD ends the turn. For every line it prints
what Yaren heard, which tools she called, what she said, and the latencies:
  heard   end of speech -> transcript (VAD end-of-turn wait + Whisper)
  1st     end of speech -> first audio chunk of the reply
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LINES = [  # (id, language, text, what should happen)
    ("hello_tr", "tr", "Merhaba Yaren, nasılsın?", "chat, no tool"),
    ("task_tr", "tr", "Kalemi alıp kaseye koyar mısın?", "do_task with the trained instruction"),
    ("stop_tr", "tr", "Dur!", "fast-path stop while the arm runs"),
    ("look_en", "en", "What do you see right now?", "look"),
    ("untrained_tr", "tr", "Kalemi kasenin yanına koy.", "do_task, untrained, says she will try"),
    ("home_en", "en", "Okay, go back to your start position please.", "go_home"),
    ("who_en", "en", "Who built you, and what can you do?", "chat, no tool, facts from the persona"),
    ("hi_en", "en", "Say hi to my friend Ahmet!", "gesture hello, greets Ahmet while it plays"),
    ("mouth_tr", "tr", "Ağzını aç.", "move_part gripper open"),
]


def make_audio(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [ln for ln in LINES if not (out_dir / f"{ln[0]}.wav").exists()]
    if not todo:
        return
    import soundfile as sf
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    sys.path.insert(0, str(HERE))
    from agent.voice import trim_tail  # an untrimmed tail would end the "user's" turn late and skew latencies

    tts = ChatterboxMultilingualTTS.from_local(str(Path.home() / "models/yaren/chatterbox"), "cuda")
    ref = Path.home() / "models/yaren/voices_ref/tr_m.flac"
    tts.prepare_conditionals(str(ref))
    for lid, lang, text, _ in todo:
        torch.manual_seed(1)
        wav = tts.generate(text, language_id=lang, cfg_weight=0.5)
        sf.write(out_dir / f"{lid}.wav", trim_tail(wav.squeeze(0).cpu().numpy(), tts.sr, len(text)), tts.sr)
        print("made", lid)


def load16k(path):
    import librosa

    a, _ = librosa.load(path, sr=16000, mono=True)
    return (np.clip(a, -1, 1) * 32767).astype("<i2")


async def run(url, token, only):
    import websockets

    audio_dir = HERE / "run" / "selftest"
    results = []
    async with websockets.connect(f"{url}/ws?token={token}", max_size=8 * 2**20) as ws:
        events = asyncio.Queue()

        async def reader():
            meta = None
            async for msg in ws:
                if isinstance(msg, bytes):
                    await events.put(("bytes", meta, time.perf_counter()))
                else:
                    ev = json.loads(msg)
                    if ev.get("type") == "audio":
                        meta = ev
                    await events.put(("json", ev, time.perf_counter()))

        rtask = asyncio.create_task(reader())
        await ws.send(json.dumps({"type": "settings", "lang": "auto", "handsFree": True, "bargeIn": True}))
        for i, (lid, lang, text, expect) in enumerate(LINES):
            if only is not None and i not in only:
                continue
            while not events.empty():
                events.get_nowait()
            pcm = load16k(audio_dir / f"{lid}.wav")
            chunk = 512
            t_send = time.perf_counter()
            for k in range(0, len(pcm), chunk):
                await ws.send(pcm[k:k + chunk].tobytes())
                await asyncio.sleep(chunk / 16000)
            t_speech_end = time.perf_counter()
            silence = np.zeros(chunk, "<i2").tobytes()
            heard = tools = said = first_audio = t_heard = None
            tools, said, played_s = [], [], 0.0
            llm_first = tts_first = None
            deadline = time.perf_counter() + 60
            idle_since = None
            while time.perf_counter() < deadline:
                await ws.send(silence)        # keep streaming silence like an open mic
                await asyncio.sleep(chunk / 16000)
                while not events.empty():
                    kind, ev, t = events.get_nowait()
                    if kind == "bytes":
                        if first_audio is None:
                            first_audio = t - t_speech_end
                            await ws.send(json.dumps({"type": "playback", "playing": True}))
                        played_s += ev["seconds"]
                        continue
                    typ = ev.get("type")
                    if typ == "user" and heard is None:
                        heard, t_heard = ev, t - t_speech_end
                    elif typ == "tool":
                        tools.append((ev["name"], ev.get("args"), ev.get("result")))
                    elif typ == "say":
                        said.append(ev["text"])
                    elif typ == "llm_stats" and llm_first is None and ev.get("first_sentence_ms") is not None:
                        llm_first = ev["first_sentence_ms"]
                    elif typ == "audio" and tts_first is None:
                        tts_first = ev.get("tts_ms")
                    elif typ == "state":
                        idle_since = t if ev["state"] == "idle" else None
                if first_audio is not None and idle_since is None and played_s and \
                        time.perf_counter() - t_speech_end > first_audio + played_s:
                    await ws.send(json.dumps({"type": "playback", "playing": False}))  # "finished playing"
                    played_s = 0.0
                if heard and idle_since and time.perf_counter() - idle_since > 1.0:
                    break
            r = {"id": lid, "expect": expect, "sent": text, "heard": heard and heard["text"],
                 "lang": heard and heard["lang"], "asr_ms": heard and heard.get("asr_ms"),
                 "heard_s": None if t_heard is None else round(t_heard, 2),
                 "first_audio_s": None if first_audio is None else round(first_audio, 2),
                 "tools": tools, "said": " ".join(said),
                 "llm_first_sentence_ms": llm_first, "tts_first_ms": tts_first}
            results.append(r)
            print(f"\n[{lid}] expect: {expect}\n  sent : {text}\n  heard: {r['heard']} ({r['lang']}, "
                  f"asr {r['asr_ms']} ms, transcript after {r['heard_s']} s)")
            for name, args, res in tools:
                print(f"  tool : {name} {json.dumps(args, ensure_ascii=False)} -> {json.dumps(res, ensure_ascii=False)[:160]}")
            print(f"  said : {r['said']}\n  first audio {r['first_audio_s']} s after the end of speech "
                  f"(LLM first sentence {llm_first} ms, its TTS {tts_first} ms)")
            await asyncio.sleep(1.5 if lid != "task_tr" else 3.0)  # let the dry arm run a little before "dur"
        rtask.cancel()
    out = HERE / "run" / "selftest" / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\nsaved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8080")
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--make-audio-only", action="store_true")
    a = ap.parse_args()
    make_audio(HERE / "run" / "selftest")
    if a.make_audio_only:
        return 0
    token = (HERE / "run" / "token").read_text().strip()
    asyncio.run(run(a.url, token, set(a.only) if a.only else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
