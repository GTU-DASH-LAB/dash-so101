---
name: robotics-reel-series
description: "Produce the next episode of Fouad's 'Day N of making robotics videos until I get recognized' series: edit raw talking-head + robot footage into a ~60 s 9:16 Instagram/LinkedIn reel with ChatCut MCP (hook, repeat/silence removal, red-blue motion graphics, B-roll from the LeRobot dataset, styled captions, Python-generated music + SFX, zoom punch-ins, voice isolation, follow end card), export and verify it, then write the Instagram caption and LinkedIn post. Triggers on: next episode, Day 2 / Day N reel, edit my robotics video, robotics series reel, make the reel for today, continue the series."
---

# Robotics reel series (Day N)

Reusable end-to-end workflow distilled from **Day 1** (ACT pen-in-bowl, wrist camera only — see
`references/day1-beat-map.md`). Everything here was actually run and verified on this machine.

`<skill>` below = the folder this file is in. Templates, scripts and references are all inside it.

## Who / what

- **Fouad Aladhami** — MSc in AI (Computer Engineering), Gebze Technical University, GTU DASH Lab.
- Socials (end card + copy): Instagram **@engineer__fouad**, LinkedIn **in/fouad-aladhami**,
  GitHub **fouad1233**, lab GitHub **GTU-DASH-LAB**, Hugging Face **fouad1233**.
- Series: "Day N of making robotics videos until I get recognized"; tag `ROBOTICS SERIES · EP.0N`.
- Robot work lives in `~/robotics/dash-so101/<task>/` (LeRobot SO-101 projects; each has `config.env`,
  numbered scripts, `videos_for_editing/` for raw reel footage). Read `references/episodes.md` first —
  it holds what the previous episode promised (Day 1 promised real-robot results).

## Standing preferences (from the user — follow without asking)

- Run **end-to-end without checkpoints**; deliver the finished MP4 + copy. Do it yourself — **no subagents**
  (the user asked for that explicitly).
- ~1 minute, 9:16, strong hook in the first 2 s; remove ALL repeats, false starts and silences.
- Red `#FF2D55` + blue `#2F6BFF` on navy; modern, stylish, "professional camera" look.
- Generate the background music with Python (`scripts/make_music.py`), matched to topic turns.
- Explain the topic with animated graphics and real numbers; never invent stats.
- **Never cover the robot arm, the object or the face with text** (Day-1 complaint).
- Follow buttons at the **bottom**, with Instagram + LinkedIn + both GitHubs.
- Copy must sound human; the thanks line is ONE short sentence.
- Repo rules still apply: no commits/pushes unless asked; log steps to `~/robotics/logs/setup-progress.log`;
  never pipe status-checked commands into head/tail; never `pkill -f`/`pgrep -f` with a self-matching pattern.

## Workflow

### 0. Gather (local, before ChatCut)
1. Find the raw footage: `<task>/videos_for_editing/` (ask only if nothing is there). `ffprobe` each file.
2. Contact sheets to see the footage:
   `ffmpeg -i talk.mp4 -vf "fps=0.1,scale=240:-1,tile=6x2" -frames:v 1 sheet.jpg` (Read it). Note outfit,
   framing, where the robot sits.
3. Real numbers:
   - dataset → `$PY <skill>/scripts/dataset_assets.py stats --root ~/.cache/huggingface/lerobot/<repo>`
     (`PY=~/.local/share/uv/tools/lelab/bin/python`, which has pandas);
   - training → in the task folder `source lib.sh; "$PY" -c 'import train_dashboard as t,json,math; d=t.collect(); tr=d["train"]; n=40; idx=[round(i*(len(tr)-1)/(n-1)) for i in range(n)]; print(json.dumps([round(math.log10(tr[i]["loss"]),3) for i in idx])); print(tr[0]["loss"], tr[-1]["loss"], d["epoch"])'`
     (log10 loss curve for the explainer MG), or `train_dashboard.py --status`;
   - rollout/eval results (for "results" episodes): ask for or find the success count; film/clip the runs.
4. Plan the beat sheet on the arc **hook → who → mission → method → data/proof → twist → problem → idea →
   benefit → cliffhanger/result → CTA**, ~10 s of B-roll or a full-screen explainer every ~10 s.

### 1. B-roll prep (local)
- Demo montage: `$PY <skill>/scripts/dataset_assets.py grid --root R --camera observation.images.wrist --n 21 --secs 9 --out demos_grid.mp4`
- Robot's-eye clip: `... sheet --episode E --out s.jpg` (pick a clean grasp) → `... clip --episode E --secs 14 --out wrist.mp4`
- Rollout / teleop phone footage: use as-is (fit cover, muted, sped up 1.45–2x).

### 2–9. Edit in ChatCut
Follow `references/chatcut-recipes.md` in order — it has the exact calls, parameters and every gotcha:
project + editor tab → upload (helper, ≤4 files/session) → transcript fixes → Script cut (hook, strikes,
`[silence=Xs→Ys]`) → tracks (V1 anchor … A1 follower) → motion graphics from `templates/mg/` → B-roll →
captions (TikTok Pop, red pill) → music + SFX → zoom punch-ins (full override objects!) → voice isolation
per clip (apply, never attach) → `smooth_audio` → caption refresh → frame-by-frame verification.

Before placing any overlay on talking-head footage, run
`<skill>/scripts/frame_grid.sh <talk.mp4> grid.jpg <t0> <t1> <t2>` and keep text out of the face/arm boxes
(`references/design-system.md` has the zones and the zoom math).

### 10. Music
`/usr/bin/python3 <skill>/scripts/make_music.py <scratch>/media --seconds <timeline seconds>` (defaults =
Day-1 track: 112 BPM, A minor, seed 7). It prints the frame of drop 1, breakdown and drop 2 — align them to
topic turns (shift the item or change `--bpm/--seconds`); use `--seed`/`--transpose` so each episode
sounds a bit different while keeping the series sound. Also writes `sfx_whoosh/impact/riser/pop.wav`.

### 11. Export, finish, deliver
Load `chatcut:export`; `submit_export {format:"video", codec:"h264", resolution:"1080p", name:"fouad_robotics_dayN_reel"}`;
run `<skill>/scripts/wait_download.sh fouad_robotics_dayN_reel` in the background and end the turn;
then:
```bash
cp ~/Downloads/fouad_robotics_dayN_reel.mp4 <task>/videos_for_editing/dayN_chatcut_export.mp4
bash <skill>/scripts/loudnorm.sh <task>/videos_for_editing/dayN_chatcut_export.mp4 <task>/videos_for_editing/FINAL_reel_dayN.mp4
bash <skill>/scripts/verify_export.sh <task>/videos_for_editing/FINAL_reel_dayN.mp4 <scratch>/verify 0.5 4 30 58
```
Read `stills.jpg`, check loudness ≈ -14 LUFS, then `SendUserFile` (a still + the MP4; say that files
> 30 MiB only show in the desktop app). Fix-ups: edit in ChatCut, re-export as `_v2`, `_v3`…

### 12. Copy + log
- Write the Instagram caption and LinkedIn post from `references/social-copy.md` (voice rules included).
- Append the episode to `references/episodes.md` (project id, final path, numbers, what Day N+1 must answer).
- Append a line to `~/robotics/logs/setup-progress.log`.

## Motion-graphic templates (`templates/mg/<name>.jsx` + `.json`)

Each `.json` has `properties` (with Day-1 defaults), box `width/height`, `durationInFrames`, `role`, and
`day1Placement` (track, left/top/width/height, frames, zone). Change the text defaults for the new episode,
match `durationInFrames` to the spoken span, then `create_motion_graphic_from_code` + `edit_item`.

| Template | Use | Per-episode edits |
|---|---|---|
| `hook-day-title` | every episode, frame 0 | `label` EP.0N, `title` DAY N (subtitle chips stay = series tagline). Re-check the arm with frame_grid |
| `name-tag` | every episode, ~5 s after the hook | none (shorten to ~3 s after Day 3) |
| `mission-card-pen-to-bowl` | "today's task" | `chip`, `step1`, `step2`; swap the two Lucide icons (`pen`, `soup`) in code |
| `teleoperation-diagram` | when showing demo recording | labels only |
| `dataset-counter-100-demos` | data/proof beat over the demo grid | `target`, `unit`, `chip1`/`chip2` from `stats` (label recording time explicitly!), `foot`, `revealAt` |
| `act-explainer-full-screen` | model explainer (full frame) | text props + replace the `LOSS` array inside the component with the real log10 curve; wrist video on V5 at (96,416,368x276) |
| `badge-only-1-camera` | one punchy key fact | `chip`, `sub`, icon |
| `pip-frame-robot-s-view` | frame around a PiP clip (video on V2 under it) | `label` |
| `two-camera-problem-full-screen` | "why the old way fails" diagram | text + `turnAt`/`failAt`; diagram is Day-1-specific — adapt the SVG for new problems |
| `idea-card-one-camera-enough` | the hypothesis / takeaway | `line1`, `hl`, `rest`, `hlAt` |
| `generalization-full-screen` | benefit beat with a framed clip (V5 180,436 720x540) | text props |
| `will-it-work-teaser` | cliffhanger | `title`, `sub` → "results on DAY N+1" |
| `cta-follow-socials` | every episode, last ~3.5 s, bottom | `follow` → "FOLLOW FOR DAY N+1"; move those caption cards to `top:1030` |
| `cinematic-vignette` | whole timeline on V3 | `durationInFrames` = timeline length |

New beats (results scoreboard, failure montage, before/after) → build new MGs in the same language
(design-system.md) and save them back into `templates/mg/` with a `.json` next to them.

## Done = all of these

- [ ] ~60 s, hook in first 2 s, no repeats/fillers/dead air; numbering consistent (DAY N / DAY N+1 everywhere)
- [ ] every graphic checked at entrance + settle in `preview_timeline`; face/arm/object never covered;
      captions never collide with graphics; nothing important under Instagram's bottom UI/right rail
- [ ] on-screen numbers match `dataset_assets.py stats` / training logs
- [ ] music ducked under voice, SFX on entrances, zooms applied, voice isolated on every clip, `smooth_audio` run
- [ ] exported, loudness-finished (~-14 LUFS), verified stills Read, MP4 sent
- [ ] Instagram caption + LinkedIn post delivered; `episodes.md` + progress log updated
