# Series log — one entry per episode (append when an episode is delivered)

Keep this current: the next episode's hook/teaser/CTA numbering and "previously on" continuity come from here.

## Day 1 — 2026-10-11 — ACT pen-in-bowl with ONLY a wrist camera

- Task folder: `~/robotics/dash-so101/pen_bowl_act/` (raw footage in `videos_for_editing/`:
  `my_video_with_face.mp4` talking head 2m12s 720x1280, `teleoperation_video.mp4` 1080x1920)
- Dataset: `fouad1233/so101_pen_in_bowl_wrist` — 100 episodes, 65,550 frames @30 fps (36.4 min), 1 camera
  `observation.images.wrist`, task "Pick up the pen and put it in the bowl"
- Model: ACT, trained on 2× RTX 5090 (60 epochs, loss 12.6 → 0.09 at edit time, ~800 samples/s)
- ChatCut project: `e54603cf-5c81-4f18-910c-027e3a52fcc6` (https://app.chatcut.io/editor/e54603cf-5c81-4f18-910c-027e3a52fcc6)
- Final: `pen_bowl_act/videos_for_editing/FINAL_reel_day1.mp4` (v3: 64 s, 1080x1920, H.264/AAC, 56.7 MB, -19.8 LUFS — not yet loudness-normalized)
- Music: make_music.py defaults (112 BPM, A minor, seed 7)
- Promised next: "real-robot results" — Day 2 must show the autonomous rollouts of this ACT policy
  (`3_run_autonomous.sh`), ideally with a success count over N tries, and answer "did one wrist camera work?"
- Known issue: dataset HUD chip says "≈ 2 HOURS" (spoken recording time) next to "65,550 FRAMES"; real footage is 36.4 min.
- Copy: Instagram caption + LinkedIn post in `social-copy.md`.
