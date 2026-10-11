# Day 1 beat map — the structure that worked (64 s, 1919 frames @30 fps)

Episode: "ACT pen-in-bowl with ONLY a wrist camera". ChatCut project `e54603cf-5c81-4f18-910c-027e3a52fcc6`.
Final file: `~/robotics/dash-so101/pen_bowl_act/videos_for_editing/FINAL_reel_day1.mp4`.

Narrative arc to reuse every episode: **hook → who I am → today's mission → how (method) → data/proof →
key twist → why it might fail (problem) → the idea → the benefit → cliffhanger → follow CTA.**

| Frames | Spoken (A-roll, V1) | Visual layers | Audio accents | Zoom on V1 |
|---|---|---|---|---|
| 0–142 | "Today is the first day of making robotics videos until I get recognized." | **Hook title** (V4, top band y112–410): `ROBOTICS SERIES · EP.01` / `DAY 1` / `ROBOTICS VIDEOS` + `UNTIL I GET RECOGNIZED`. Captions moved to top 1440. | impact f0 (-8 dB); music intro | slow-push 1.15 (0–149) |
| 149–300 | "So today with this robotic arm, we'll try to train it…" | **Name tag** lower third (40,1320): FOUAD ALADHAMI / MSc in AI · Computer Engineering / Gebze Technical University | — | instant 1.2 on 240–403 (4 clips) |
| 245–403 | "…so it will take the pen and put it in the bowl." | **Mission card** top (50,170): THE MISSION · pen icon → arrow → bowl icon | music drop 1 at f257; whoosh f377 | (cont.) |
| 403–660 | "To do this we use something called teleoperation… using the other robotic arm we are controlling this one." | **Teleop B-roll** full frame (V2, from 6 s, 1.45x, muted) + **Teleoperation diagram** below captions (60,1300): LEADER ARM "I move this one" → pulses → FOLLOWER ARM "copies me live" | whoosh f634 | — |
| 660–875 | "I collected 100 demos… it takes me like two hours of demos. You can see some of them here." | **21-demo grid** B-roll (V2, overscanned) + **Dataset counter** HUD (80,380): MY DATASET 0→100 DEMONSTRATIONS, chips "≈ 2 HOURS" / "65,550 FRAMES" | pops f710, f764; whoosh f849 | — |
| 875–1035 | "Then we'll train an ACT policy — an action chunking transformer policy." | **ACT explainer** full screen: inputs (wrist video on V5 inside the input box) → TRANSFORMER (CVAE) → NEXT 100 ACTIONS; TRAINING ON 2× RTX 5090 · 60 EPOCHS · ~800 SAMPLES/S · real loss curve 12.6 → 0.09 | whoosh f1009 | — |
| 1064–1167 | "This time we just use one camera at the gripper here." | **Badge** ONLY 1 CAMERA / mounted on the gripper (40,1320) + **PiP** robot's view top-right (V2 video 657,184 + V5 frame 650,130) | pop f1064; whoosh f1181 | instant 1.2 |
| 1207–1408 | "Before, we used two cameras… if I change the direction of the second camera, the policy fails." | **Two-camera problem** full screen: arm + 2 cameras, "it learns 3D by fusing BOTH views", camera rotates (turnAt 91), TASK FAILED stamp (failAt 146), lesson chips | impact f1353 on the stamp; music breakdown ≈ f1029–1286, drop 2 f1286 | — |
| 1408–1642 | "So I'm thinking that for a simple task one camera should be enough." | **Idea card** (40,1320): THE IDEA · For a simple task… ONE CAMERA (red highlight at hlAt 150) SHOULD BE ENOUGH | whoosh f1382 | slow-push 1.15 (1465–1642) |
| 1642–1732 | "…and it may generalize more." | **Generalization** full screen: GENERALIZE MORE? + framed robot's-view clip (V5 180,436 720x540) + "the camera rides WITH the gripper" / SAME VIEW · ANYWHERE | whoosh f1616 | — |
| 1732–1814 | "Will it work? I'm not sure, let's see." | **Teaser** (40,1320): WILL IT WORK? ⓘ / real-robot results in PART 2 | pop f1732 | instant 1.2 |
| 1814–1919 | "See you in the next part. Follow me." | **CTA end card** (40,1150 960x380): FOLLOW FOR DAY 2 → + 2x2 buttons Instagram @engineer__fouad, LinkedIn in/fouad-aladhami, GitHub fouad1233 (ME), GitHub GTU-DASH-LAB (OUR LAB). Captions moved to top 1030. | impact f1814 (-10 dB); music outro/fade 2 s | quick-in 1.25 (1814–1860) |

Whole timeline: cinematic vignette (V3), music on A1 (-3 dB, fade in 1 s / out 2 s, ducked under V1).

## Lessons from the Day-1 review (apply next time from the start)

1. **Hook title covered the robot arm** → design the hook inside y112–410 and check with `frame_grid.sh`.
2. **Follow buttons belonged at the bottom with ALL socials** (Instagram + LinkedIn were missing) → the
   `cta-follow-socials` template is the default end card now.
3. **"≈ 2 HOURS" on the dataset HUD came from the spoken line ("it takes me like two hours") — recording
   time, not data length.** The dataset is 36.4 min of footage (65,550 frames @30 fps). On-screen numbers
   must come from `dataset_assets.py stats`; if you show recording time, label it ("≈ 2 H TO RECORD").
4. The teaser said "PART 2" while the CTA said "DAY 2" → keep naming consistent: "DAY N+1".
5. Export came out at -19.8 LUFS → run `loudnorm.sh` before delivering.
