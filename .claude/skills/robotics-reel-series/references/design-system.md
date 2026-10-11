# Design system — "Robotics Series" reels

One visual identity across every episode so the series is recognisable in the feed.

## Palette (matches the user's preference: red + blue, works with the white t-shirt)

| Token | Hex | Use |
|---|---|---|
| red | `#FF2D55` | primary accent: title chips, highlight pill on captions, Instagram/GitHub-me buttons, "failed" stamps |
| blue | `#2F6BFF` | secondary accent: subtitle chips, LinkedIn/lab buttons, diagrams, signal pulses |
| navy | `#070B24` | card / chip backgrounds (opacity ~0.82–0.9 over video) |
| navy-deep | `#060A1F` | full-screen explainer backgrounds |
| light-blue text | `#9DB7FF` | mono labels, url prefixes, secondary text on navy |
| white | `#FFFFFF` | all primary text |
| vignette edge | `#03040C` | cinematic vignette shade |

If the user wears something else in a new episode, keep red/blue anyway (it is the series identity)
unless they ask; just check contrast against the footage.

## Type

| Role | Font (exact `search_fonts` name) | Sizes used |
|---|---|---|
| Headlines, chips, big numbers | **Anton** | hero "DAY N" 168; CTA headline 76; chips 52–80; badges 74 |
| Body, names, handles | **Space Grotesk** (700) | names 48–56; handles 31–34 (auto-fit by length) |
| Labels, HUD text, urls | **JetBrains Mono** (700, letter-spacing 2–3) | 19–30 |
| Captions | Noto Sans SC 900 (TikTok Pop preset) | 66 |

Hero title treatment: RGB split `textShadow: "-6px 0 0 red, 6px 0 0 blue"`, a 2-frame ±8 px glitch on entry,
navy chip with a 12 px red left border, whole stack `rotate(-2deg)`.

## Motion language

- Entrances are springs: `spring({frame: frame - delay, fps, config:{damping:11–16, stiffness:150–190}})`.
- Stagger elements by 4–8 frames; alternate slide directions (from left, then from right).
- Subtle life while holding: `pulse = 1 + 0.015–0.03*sin(frame/5)`; arrows nudge `max(0,sin(frame/4))*8px`.
- Exits: fade/lift over the last 6–10 frames (`interpolate(frame,[dur-8,dur],[0,1])`).
- Every graphic entrance gets a whoosh (start 13 frames early); counters/chips get pops; big reveals get an impact.

## Canvas zones (1080x1920)

```
y    0-110   Instagram top header (Reels title / camera icon) — keep text below ~110
y  110-470   TOP BAND: hook title, mission card, section titles   <- watch the robot arm!
y  470-1100  subject zone: face, robot arm, workspace — never cover with text
y 1130-1310  CAPTION BAND (captions text ~y1190-1230) — full-screen MGs leave it empty
y 1300-1610  LOWER-THIRD MGs: name tag, badges, idea card, teaser, diagrams over B-roll
y 1150-1500  END CARD (CTA) — captions for those cards moved up to top=1030
y 1550-1920  Instagram username / caption / audio row — nothing important here
x  950-1080  Instagram action rail (like/comment/share) from ~y1100 down — keep text left of ~x950
```

## "Don't cover" rule (learned the hard way)

The Day-1 hook title (1000x620 at y170, DAY 1 at 230 px) sat right on top of the SO-101 arm, which is
the whole point of the shot. Before placing ANY overlay on talking-head footage:

1. `scripts/frame_grid.sh <talking-head.mp4> grid.jpg <t_start> <t_mid> <t_end>` for the overlay's span
   (source seconds), Read it, and note the bounding boxes (in canvas px) of: face, robot/arm/gripper,
   the object being manipulated, anything the voice mentions.
2. Remember zoom effects scale the footage around (540, focalPointY*1920): an object at y moves to
   `710 + (y-710)*mag` (focalPointY 0.37). With a slow push to 1.15 the arm rises ~30–80 px over the shot.
3. Size the overlay to fit the free area with ≥40 px clearance at the worst frame, then verify with
   `preview_timeline viewerFrames` at entrance, middle and end. Shrinking a title is better than covering the robot.

Day-1 hook fix: everything packed into y112–410 (series tag 24 px, DAY 1 168 px, two subtitle chips side by side
at 52 px) → ≈60 px clear above the gripper at its highest.

## Captions

TikTok Pop preset: white 900-weight text, dark stroke 2.5, soft shadow, active word on a red pill
(`#FF2D55`, radius 10, padding 8x2), phrase pacing, max 1 line / 15 chars / 3 words per card.
Position: anchor top-center, offsetYRatio 0.6, widthRatio 0.87, heightRatio 0.065.
Card overrides: hook `top:1440`, CTA cards `top:1030`.

## Camera / "professionally shot" look (from a phone talking head)

- Cinematic vignette MG on V3 for the whole timeline (strength 0.5, edge `#03040C`).
- Punch-in zooms on V1 on every new idea: slow-push 1.15, instant 1.2, quick-in 1.25 (see chatcut-recipes §9).
- AI voice isolation at strength 80 on every A-roll clip; 4-frame audio crossfades (`smooth_audio`).
- Music ducked automatically under speech; loudness finished to ~-14 LUFS.
- B-roll every ~10 s so the face never sits unbroken for long: teleop footage, demo grid, robot's-eye clips.

## Platform notes

- 9:16, 1080x1920, 30 fps, H.264 + AAC. The same file works for Instagram Reels and LinkedIn.
- ~60 s target (Day 1: 64 s). Hook in the first 2 s, with a sound hit on frame 0.
- Instagram allows max 5 hashtags per post; LinkedIn 3 reads best.
