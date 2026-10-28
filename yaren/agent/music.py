"""A dance beat for Yaren, synthesised with numpy in a few ms: drums, bass and a chord stab, at the
tempo and length of the arm's dance plan (motions.plan_dance): a hi-hat intro while she stands up, the
groove for the dance beats, a fading outro while she goes back.
"""

import numpy as np

SR = 24000
# A minor: Am - F - C - G, one chord per bar (4 beats); bass on the root
CHORDS = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]


def _hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def _env(n, attack=0.002, decay=0.15):
    t = np.arange(n) / SR
    return np.minimum(1.0, t / attack) * np.exp(-t / decay)


def _kick(n):
    t = np.arange(n) / SR
    f = 45 + 75 * np.exp(-t / 0.04)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.001, 0.18)


def _snare(n, rng):
    t = np.arange(n) / SR
    return (0.6 * rng.standard_normal(n) + 0.4 * np.sin(2 * np.pi * 190 * t)) * _env(n, 0.001, 0.09)


def _hat(n, rng):
    x = rng.standard_normal(n)
    x = np.diff(x, prepend=0.0)               # crude high-pass
    return x * _env(n, 0.0005, 0.025)


def _tone(freq, n, shape="saw", decay=0.3):
    t = np.arange(n) / SR
    ph = (freq * t) % 1.0
    w = (2 * ph - 1) if shape == "saw" else np.sin(2 * np.pi * ph)
    return w * _env(n, 0.005, decay)


def make_track(bpm, intro_beats, beats, outro_s, seed=7):
    """int16 mono at SR: intro (hats), `beats` of groove, an outro that fades over outro_s."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    total = (intro_beats + beats) * beat + max(outro_s, beat) + 0.3
    out = np.zeros(int(total * SR) + SR)
    nb = int(beat * SR)

    def add(sig, at_s, gain):
        i = int(at_s * SR)
        j = min(len(out), i + len(sig))
        out[i:j] += gain * sig[:j - i]

    for k in range(intro_beats):                       # intro: off-beat hats, a kick on the last beat
        add(_hat(nb // 2, rng), (k + 0.5) * beat, 0.25)
    add(_kick(nb), (intro_beats - 1) * beat, 0.6)
    t0 = intro_beats * beat
    for k in range(int(round(beats))):
        t = t0 + k * beat
        chord = CHORDS[(k // 4) % len(CHORDS)]
        add(_kick(nb), t, 0.9)
        if k % 2 == 1:
            add(_snare(nb, rng), t, 0.45)
        for h in (0.0, 0.5):
            add(_hat(nb // 2, rng), t + h * beat, 0.18 if h else 0.1)
        add(_tone(_hz(chord[0] - 24), nb, "saw", 0.22), t + 0.5 * beat, 0.22)      # off-beat bass
        if k % 4 == 0:
            for m in chord:
                add(_tone(_hz(m), 2 * nb, "sine", 0.5), t, 0.09)                # chord stab per bar
    t_end = t0 + beats * beat                          # outro: the chord rings out
    for m in CHORDS[0]:
        add(_tone(_hz(m), int(max(outro_s, beat) * SR), "sine", max(outro_s, beat) / 2), t_end, 0.12)
    add(_kick(nb), t_end, 0.7)
    out /= max(1e-6, np.abs(out).max()) / 0.8
    fade = int(0.25 * SR)
    out[-fade:] *= np.linspace(1, 0, fade)
    return (out * 32767).astype("<i2")
