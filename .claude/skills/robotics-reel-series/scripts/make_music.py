"""Synthesise the reel's background music + transition SFX with numpy/scipy only.

Usage (needs numpy + scipy; /usr/bin/python3 has both on this machine):
  python3 make_music.py OUT_DIR [--seconds 64] [--bpm 112] [--transpose 0] [--seed 7]
Defaults reproduce the Day-1 track exactly (112 BPM, A minor, 32 bars, seed 7).

Track: modern electronic / tech, sections scale with the bar count B:
  bars 0-3          intro      pad + filtered arp, riser into bar 4
  bars 4..B/2-1     groove A   four-on-the-floor kick, offbeat hats, clap, bass, arp, sidechained pad
  bars B/2..B/2+3   breakdown  pad + arp only, riser into the second drop
  bars ..B-3        groove B   groove A + 16th hats + higher arp octave
  last 2 bars       outro      pad + final impact, long tail
Mixed to sit UNDER a voice: no lead in 1-4 kHz, arp kept quiet, master -1 dBFS peak.
The script prints every section's start time so you can line the breakdown and the
second drop up with topic changes in the edit (or shift the music item's offset).

SFX (separate files, placed on cuts in the editor): whoosh, impact, riser, pop.
"""
import argparse
import math
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, fftconvolve

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("out", help="output directory for the WAV files")
_ap.add_argument("--seconds", type=float, default=None,
                 help="minimum music length in seconds (rounded up to whole bars, +1 bar). Default: 32 bars")
_ap.add_argument("--bpm", type=float, default=112)
_ap.add_argument("--transpose", type=int, default=0, help="semitones; 0 = A minor, 3 = C minor, -2 = G minor")
_ap.add_argument("--seed", type=int, default=7, help="noise/timbre seed; change it for a different-sounding take")
ARGS = _ap.parse_args() if __name__ == "__main__" else _ap.parse_args(["."])

SR = 48000
BPM = ARGS.bpm
BEAT = 60 / BPM
BAR = 4 * BEAT
NBARS = 32 if ARGS.seconds is None else max(16, math.ceil(ARGS.seconds / BAR) + 1)
BRK = NBARS // 2  # breakdown start bar; second drop at BRK + 4
N = int(NBARS * BAR * SR) + SR * 3  # + tail
rng = np.random.default_rng(ARGS.seed)


def t_of(n):
    return np.arange(n) / SR


def lp(x, f, order=2):
    return sosfilt(butter(order, f, "low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


def midi(m):
    return 440.0 * 2 ** ((m + ARGS.transpose - 69) / 12)


def saw(f, n, phase=0.0):
    ph = (phase + f * t_of(n)) % 1.0
    return 2 * ph - 1


def adsr(n, a, d, s, r):
    env = np.ones(n) * s
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na = min(na, n); env[:na] = np.linspace(0, 1, na)
    nd = min(nd, n - na); env[na:na + nd] = np.linspace(1, s, nd)
    nr = min(nr, n); env[n - nr:] *= np.linspace(1, 0, nr)
    return env


def add(buf, sig, start_s):
    i = int(start_s * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i]


def reverb_ir(seconds=1.8, decay=3.5):
    n = int(seconds * SR)
    ir = rng.standard_normal((2, n)) * np.exp(-decay * t_of(n) / seconds)
    ir[:, : int(0.01 * SR)] *= np.linspace(0, 1, int(0.01 * SR))
    return lp(ir, 6000) * 0.12


IR = reverb_ir()


def verb(mono_or_st, wet):
    st = mono_or_st if mono_or_st.ndim == 2 else np.stack([mono_or_st, mono_or_st])
    out = np.stack([fftconvolve(st[c], IR[c])[: st.shape[1]] for c in range(2)])
    return st + wet * out


# ---------------------------------------------------------------- harmony
# Am(add9)  Fmaj7  C(add9)  G6  -- one chord per bar
CHORDS = [
    [57, 60, 64, 71],  # A3 C4 E4 B4
    [53, 57, 60, 64],  # F3 A3 C4 E4
    [55, 60, 64, 62 + 12],  # G3 C4 E4 D5 -> C(add9) voiced over G
    [55, 59, 62, 64],  # G3 B3 D4 E4
]
ROOTS = [33, 29, 36, 31]  # A1 F1 C2 G1


def section(bar):
    if bar < 4: return "intro"
    if bar < BRK: return "A"
    if bar < BRK + 4: return "break"
    if bar < NBARS - 2: return "B"
    return "outro"


# ---------------------------------------------------------------- stems
pad = np.zeros((2, N)); bass = np.zeros(N); kick = np.zeros(N); hats = np.zeros(N)
clap = np.zeros(N); arp = np.zeros((2, N)); fx = np.zeros((2, N))

# kick
def kick_hit():
    n = int(0.45 * SR); t = t_of(n)
    f = 45 + 85 * np.exp(-t / 0.035)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)
    click = hp(rng.standard_normal(n), 2500) * np.exp(-t / 0.004) * 0.25
    return np.tanh(1.6 * (body + click))

def hat_hit(decay=0.035):
    n = int(0.12 * SR); t = t_of(n)
    return hp(rng.standard_normal(n), 7500, 4) * np.exp(-t / decay)

def clap_hit():
    n = int(0.35 * SR); t = t_of(n); x = np.zeros(n)
    for k, off in enumerate([0, 0.011, 0.022]):
        i = int(off * SR); m = n - i
        x[i:] += rng.standard_normal(m) * np.exp(-t_of(m) / (0.008 if k < 2 else 0.11))
    return bp(x, 900, 3200)

KICK, CLAP = kick_hit(), clap_hit()

# sidechain envelope (ducks pad/bass on each kick)
sc = np.ones(N)
for bar in range(NBARS):
    if section(bar) in ("A", "B"):
        for b in range(4):
            s = int((bar * BAR + b * BEAT) * SR); n = int(BEAT * SR)
            e = 1 - 0.65 * np.exp(-t_of(n) / 0.09)
            sc[s:s + n] = np.minimum(sc[s:s + n], e)

for bar in range(NBARS):
    sec = section(bar); t0 = bar * BAR; ch = CHORDS[bar % 4]; root = ROOTS[bar % 4]

    # pad: 3 detuned saws per note, L/R spread, slow attack
    n = int(BAR * SR) + int(0.6 * SR)
    for m in ch:
        f = midi(m)
        for c, cents in ((0, -9), (1, 9)):
            v = saw(f * 2 ** (cents / 1200), n, rng.random()) + saw(f, n, rng.random()) * 0.7
            pad[c, int(t0 * SR):int(t0 * SR) + n][: N - int(t0 * SR)] += (
                v * adsr(n, 0.35, 0.3, 0.8, 0.6) * 0.05)[: N - int(t0 * SR)]

    # bass: offbeat-pumping eighths (groove), long root note otherwise
    if sec in ("A", "B"):
        for e in range(8):
            n = int(BEAT / 2 * SR); f = midi(root + 12)
            v = (np.sin(2 * np.pi * f * t_of(n)) + 0.35 * lp(saw(f, n), 600)) * adsr(n, 0.005, 0.08, 0.7, 0.04)
            add(bass, v * 0.42, t0 + e * BEAT / 2)
    elif sec in ("intro", "break", "outro"):
        n = int(BAR * SR); f = midi(root + 12)
        add(bass, np.sin(2 * np.pi * f * t_of(n)) * adsr(n, 0.2, 0.2, 0.8, 0.3) * 0.22, t0)

    # drums
    if sec in ("A", "B"):
        for b in range(4):
            add(kick, KICK, t0 + b * BEAT)
            add(hats, hat_hit() * 0.5, t0 + b * BEAT + BEAT / 2)
            if sec == "B":
                for s16 in (0.25, 0.75):
                    add(hats, hat_hit(0.02) * 0.22, t0 + b * BEAT + s16 * BEAT)
        add(clap, CLAP, t0 + BEAT); add(clap, CLAP, t0 + 3 * BEAT)

    # arp: 16th-note pluck through chord tones, quiet, delayed
    if sec != "outro":
        tones = sorted(ch) + [sorted(ch)[1] + 12]
        octave = 12 if sec == "B" else 0
        for s in range(16):
            m = tones[[0, 2, 1, 3, 2, 4, 1, 3][s % 8]] + octave
            n = int(0.22 * SR); f = midi(m)
            v = lp(saw(f, n) + 0.5 * saw(f * 1.004, n), 2600 if sec != "intro" else 1100)
            v *= np.exp(-t_of(n) / 0.07)
            g = 0.055 if sec in ("A", "B") else 0.04
            pan = 0.5 + 0.35 * np.sin(s * 0.9)
            st = np.stack([v * (1 - pan), v * pan]) * g * 2
            i = int((t0 + s * BEAT / 4) * SR)
            j = min(N, i + n); arp[:, i:j] += st[:, : j - i]

# riser into bar 4 and bar 20, impact on bar 4/20/30
def riser(seconds):
    n = int(seconds * SR); t = t_of(n); x = rng.standard_normal(n)
    out = np.zeros(n); seg = int(0.02 * SR)
    for i in range(0, n, seg):  # stepped lowpass sweep 300 Hz -> 9 kHz
        fc = 300 * (30 ** (i / n))
        out[i:i + seg] = lp(x[max(0, i - 2000):i + seg], fc)[-len(out[i:i + seg]):]
    tone = np.sin(2 * np.pi * np.cumsum(200 * 4 ** (t / seconds)) / SR) * 0.25
    return (out + tone) * (t / seconds) ** 2

def impact():
    n = int(2.2 * SR); t = t_of(n)
    boom = np.sin(2 * np.pi * np.cumsum(38 + 60 * np.exp(-t / 0.05)) / SR) * np.exp(-t / 0.5)
    crash = hp(rng.standard_normal(n), 3000) * np.exp(-t / 0.55) * 0.35
    return np.tanh(1.3 * (boom + crash))

R2 = riser(2 * BAR)
for b in (4, BRK + 4):
    add(fx[0], R2 * 0.18, b * BAR - 2 * BAR); add(fx[1], R2 * 0.18, b * BAR - 2 * BAR)
IMP = impact()
for b in (4, BRK + 4, NBARS - 2):
    add(fx[0], IMP * 0.32, b * BAR); add(fx[1], IMP * 0.32, b * BAR)

# ---------------------------------------------------------------- mix
pad = lp(pad, 2200) * sc
bass = lp(bass, 900) * sc
drums = kick * 0.55 + hats * 0.32 + clap * 0.18
mix = verb(pad, 0.45) + verb(arp, 0.6) + np.stack([bass, bass]) + verb(np.stack([drums, drums]), 0.12) + verb(fx, 0.5)
# carve 1.5-4 kHz a little so the voice sits on top
mix = mix - 0.35 * bp(mix, 1500, 4000)
# fade in 1 bar, fade out last 2 s
fi = int(BAR * SR); mix[:, :fi] *= np.linspace(0, 1, fi) ** 1.5
fo = int(2.5 * SR); end = int((NBARS * BAR + 2.0) * SR)
mix = mix[:, :end]; mix[:, -fo:] *= np.linspace(1, 0, fo) ** 2
mix = np.tanh(1.2 * mix / np.max(np.abs(mix))) ; mix *= 0.89 / np.max(np.abs(mix))


def write(path, st):
    st = np.atleast_2d(st)
    if st.shape[0] == 1: st = np.vstack([st, st])
    wavfile.write(path, SR, (st.T * 32767).astype(np.int16))


if __name__ == "__main__":
    import os
    out = ARGS.out
    os.makedirs(out, exist_ok=True)
    bgm = f"{out}/bgm_robotics_{int(round(BPM))}bpm.wav"
    write(bgm, mix)
    print(f"bgm: {bgm}  bars={NBARS}  bar={BAR:.3f}s")
    for name, b in (("intro", 0), ("drop 1 (groove A)", 4), ("breakdown", BRK),
                    ("drop 2 (groove B)", BRK + 4), ("outro", NBARS - 2)):
        print(f"  {name:18s} bar {b:2d}  t={b * BAR:6.2f}s  frame@30fps={round(b * BAR * 30)}")

    # ---- SFX
    def norm(x, peak=0.89): return x * peak / np.max(np.abs(x))
    # whoosh: band-passed noise swept up then down, stereo pan L->R
    n = int(0.9 * SR); t = t_of(n); x = rng.standard_normal(n); w = np.zeros(n); seg = 480
    for i in range(0, n, seg):
        fc = 400 + 5000 * np.sin(np.pi * i / n) ** 2
        w[i:i + seg] = bp(x[max(0, i - 1500):i + seg], fc * 0.6, min(fc * 1.6, 20000))[-len(w[i:i + seg]):]
    w *= np.sin(np.pi * t / t[-1]) ** 2
    write(f"{out}/sfx_whoosh.wav", norm(np.stack([w * (1 - t / t[-1]), w * (t / t[-1])])))
    write(f"{out}/sfx_impact.wav", norm(verb(impact(), 0.4)))
    write(f"{out}/sfx_riser.wav", norm(verb(riser(1.6), 0.4)))
    n = int(0.18 * SR); t = t_of(n)
    pop = np.sin(2 * np.pi * np.cumsum(1400 - 700 * t / t[-1]) / SR) * np.exp(-t / 0.03)
    write(f"{out}/sfx_pop.wav", norm(pop, 0.7))
    print("ok", mix.shape[1] / SR, "s")
