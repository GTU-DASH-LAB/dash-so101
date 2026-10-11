# ChatCut MCP recipes — the exact calls that worked (Day 1)

All ChatCut tools are **deferred**: load them with `ToolSearch` (`select:mcp__plugin_chatcut_chatcut__<name>,...`)
before calling. Before the FIRST ChatCut call in a conversation, invoke the Skill
`chatcut:chatcut-plugin-basics-claude` (the server requires it). Load `chatcut:talking-head-guide`,
`chatcut:create-motion-graphics`, `chatcut:asset-import`, and (at the end) `chatcut:export` when you reach
those stages — they are the plugin's own playbooks; this file is what we learned on top of them.

Frames are at 30 fps. Canvas 1080x1920. Times below are Day-1 values; adapt to the new cut.

---

## 1. Project + editor tab

```
create_project {name:"Robotics Series — Day N <topic> (Reel)", compositionWidth:1080, compositionHeight:1920, fps:30,
                description:"<one line>"}
```
Open the returned `browserHandoff.url` with `mcp__Claude_Browser__preview_start {url}` (the in-app browser) and
**keep that tab open for the whole session** — viewer previews, voice isolation and the local export all run in it.

## 2. Import media

The loopback route (`serve-local-media.mjs` + `import_media action:"from_editor"`) **failed** in the in-app
browser with `ERR_BLOCKED_BY_CLIENT`. Use the upload helper instead (max **4 files per session**):

```
import_media {action:"create_session", projectId}
-> token, endpoint
node ~/.claude/plugins/cache/chatcut-inc/chatcut/<ver>/skills/asset-import/scripts/upload-media.mjs \
     --token '<token>' --endpoint <endpoint> f1 f2 f3 f4 > upload.json 2> upload.err; echo "exit=$?"
```
Read `imports[].result.assetId` from `upload.json` (python `json.load`; never pipe the upload into `tail`).
Uploading 240 MB took a few minutes — run it with `run_in_background`. A fresh `create_session` per batch of 4.
Then `track_progress {target:"transcription"}` on the talking-head asset before any script work.

Things imported on Day 1: talking head, teleop B-roll, demo grid, wrist-cam clip (batch 1); BGM + 3 SFX (batch 2).

## 3. A-roll: hook, repeats, silences (Script workflow)

1. `read_script {showSilence:true}` — rows look like `[s10@3] it takes me like two hours of demos.` with
   `[silence=0.53s]` between.
2. Fix ASR mishearings first (they also fix captions): preview, then apply.
   ```
   manage_transcript {action:"fix", asset:"<talking-head assetId>", content:"FIX [s4]: 11 -> \"pen\"\nFIX [s23]: 4 -> \"ACT\"", preview:true}
   ```
   Typical misses: "ACT" (heard as "act"/"ect"), "pen", "bowl", "SO-101", "LeRobot", "teleoperation".
3. `clean_script {silence:"compress:250"}` — removes um/uh/er/ah and caps silences.
4. `read_script {showSilence:true, refreshCache:true}` then edit the returned `timelineMd`:
   - strike retakes / false starts / repeated words with `~~text~~` (e.g. `[s4@2] ~~together ~~to`,
     `[s13@1] we'll,~~ we'll~~`);
   - drop whole off-topic segments by striking them;
   - compress pauses with the **Unicode arrow**: `[silence=0.27s→0.05s]` (intra-sentence 0.05–0.15 s,
     between sentences ~0.1–0.25 s — reels are punchy);
   - keep the strongest line as the opening hook (Day 1: "Today is the first day of making robotics videos
     until I get recognized.").
5. `apply_script {timelineMd, preview:true}` → check → same call without `preview`.
6. Target ≈ 60–64 s. Day 1 went 2m12s raw → 63 s + 1 s tail (1919 frames).

## 4. Track layout

```
edit_track {action:"create", json:"{\"trackType\":\"video\",\"count\":4}"}
edit_track {action:"create", json:"{\"trackType\":\"audio\",\"count\":2}"}   # + 1 more audio later
```
| Track | Content | Role / name |
|---|---|---|
| V1 | talking head (A-roll) | `role:"anchor"`, name "A-roll" |
| V2 | B-roll (teleop, demo grid, PiP wrist clip) | — |
| V3 | cinematic vignette MG (whole timeline) | — |
| V4 | main motion graphics | — |
| V5 | secondary graphics / video inside full-screen MGs / PiP frame | — |
| A1 | music | `role:"follower"`, name "Music" (auto-ducks under V1; duckDepthDb auto-set ≈ -14 dB) |
| A2 | whooshes + pops | name "SFX", no role |
| A3 | impacts | name "SFX hits", no role |

Higher video track = drawn on top. Use the **track id** (e.g. `a031b2a313`) for effects; aliases can shift.

## 5. Motion graphics

Create from the templates in `templates/mg/` (code + property schema + Day-1 placement):
```
create_motion_graphic_from_code {name, description, width, height, durationInFrames, properties:[...], code}
edit_item {adds:[{type:"motion-graphic", assetId, trackId:"<V4 id>", fromFrame, durationInFrames, left, top, width, height}]}
```
Change per-episode text by editing the property `defaultValue`s before creating (or `propertyOverrides`
on the item). Contract that bit us (validator errors, all fixed in the templates):
- every declared property must be read in the code; read props from `item.props`, never `props.x || "default"`;
- no top-level constants — put arrays like loss curves INSIDE the component;
- root element must be `<div style={rootStyle}>`; no imports/exports; no `Sequence`;
- no inset `boxShadow`, no `backdropFilter`, no `zIndex` (order = DOM order);
- Lucide `<Icon name=... size=... color="#FFFFFF" strokeWidth=.../>` needs an explicit color.
  Verified icon names: `github`, `instagram`, `linkedin`, `camera`, `arrow-right` (plus whatever the templates use);
- fonts must come from `search_fonts`: **Anton**, **Space Grotesk**, **JetBrains Mono** (exact names);
- `edit_asset` can update code but **cannot change width/height** — when the box must change, create a NEW
  asset and swap the item (`deletes:[{id}]` + `adds:[...]` in one `edit_item`);
- item aspect ratio is locked to the asset; resizing an item scales it uniformly.
- A warning "uncertain about style boxShadow (dynamic-style-value)" is non-blocking — ignore.

Full-screen MGs (1080x1920, opaque navy bg) must leave the **caption band y≈1130–1310 empty**; put any
video that should appear "inside" them on V5 at the slot the MG draws (ACT explainer input box
(96,416,368x276) radius 20; Generalization frame (180,436,720x540) radius 26).

## 6. B-roll

```
edit_item {adds:[
  {type:"video", assetId:<teleop>, trackId:<V2>, fromFrame:403, durationInFrames:257, sourceStartFromInSeconds:6,
   playbackRate:1.45, fit:"cover", decibelAdjustment:-60},
  {type:"video", assetId:<demo grid>, trackId:<V2>, fromFrame:660, durationInFrames:215, fit:"cover"},
  {type:"video", assetId:<wrist clip>, trackId:<V2>, fromFrame:1075, durationInFrames:92, sourceStartFromInSeconds:1,
   playbackRate:2, left:657, top:184, width:377, height:283, borderRadius:20}   # PiP, top-right
]}
```
- Mute B-roll audio (`decibelAdjustment:-60`) — teleop motor noise fights the voice.
- Speed up teleop 1.45–2x; it reads as energetic.
- The demo grid had black edges → overscan it slightly: `left:-12, top:-22, width:1104, height:1963`.
- The teleop action fills the top 58% of that frame → its explainer MG goes BELOW the captions (top 1300).
- PiP goes top-right and must not cover the face — check a frame.

## 7. Captions (sequential — each call returns the revision the next needs)

```
edit_captions {action:"enable"}
edit_captions {action:"set_sources", json:"{\"sources\":[{\"trackId\":\"V1\"}]}"}
edit_captions {action:"template", templatePreset:"tiktok"}                       # TikTok Pop
edit_captions {action:"style", json:"{\"sizePx\":66,\"highlightColor\":\"#FFFFFF\",\"highlightBackground\":{\"color\":\"#FF2D55\",\"paddingX\":8,\"paddingY\":2,\"borderRadius\":10},\"pacing\":\"phrase\"}"}
edit_captions {action:"source_layout", json:"{\"sourceId\":\"track:<V1 track id>#original:::\",\"anchor\":\"top-center\",\"offsetYRatio\":0.6,\"widthRatio\":0.87,\"heightRatio\":0.065}"}
edit_captions {action:"refresh"}
read_captions {json:"{\"fromFrame\":0,\"toFrame\":150}"}      # card ids + revision
edit_captions {action:"set_card_layout", json:"{\"cardId\":\"cue:...\",\"layout\":{\"top\":1440},\"revision\":\"fnv1a:...\"}"}
```
- `source_layout` **rejects left/top/width/height** → use anchor + ratios as above (text lands at y≈1190–1230).
- Result: Noto Sans SC 900, 66 px, white with dark stroke, active word on a red `#FF2D55` pill,
  max 3 words / 15 chars per card, one line.
- Per-card overrides: hook cards `top:1440` (title owns the top, keep captions off the chin);
  end-card cards `top:1030` (CTA owns y1150–1500). One `set_card_layout` per card, chaining revisions.
- After ANY timeline edit: `edit_captions {action:"refresh"}` (exits early if already reconciled).

## 8. Music + SFX

Generate with `scripts/make_music.py` (see SKILL.md), upload in one 4-file batch, then:
```
edit_track update V1 {role:"anchor", name:"A-roll"};  A1 {role:"follower", name:"Music"};  A2 {name:"SFX"}
edit_track create {trackType:"audio", name:"SFX hits"}                      # becomes A3
edit_item {adds:[
  {type:"audio", assetId:<bgm>, trackId:"A1", fromFrame:0, durationInFrames:<timeline len>, decibelAdjustment:-3, audioFadeIn:1, audioFadeOut:2},
  {type:"audio", assetId:<impact>, trackId:"A3", fromFrame:0,    decibelAdjustment:-8},    # hook hit
  {type:"audio", assetId:<impact>, trackId:"A3", fromFrame:1353, decibelAdjustment:-10},   # "TASK FAILED" stamp
  {type:"audio", assetId:<impact>, trackId:"A3", fromFrame:1814, decibelAdjustment:-10},   # CTA
  {type:"audio", assetId:<whoosh>, trackId:"A2", fromFrame:<cut-13>, decibelAdjustment:-10}, # x7, one per graphic entrance
  {type:"audio", assetId:<pop>,    trackId:"A2", fromFrame:<pop frame>, decibelAdjustment:-6}  # x4, counter ticks / chips
]}
```
- Whoosh is 27 frames with its peak in the middle → start it **13 frames before** the cut/entrance it marks.
- Pop is 5 frames; impact 66 frames. Keep SFX on tracks WITHOUT a role (they should punch through).
- `audioFadeIn/Out` are **seconds**, not frames.
- Line the music's breakdown and second drop up with topic turns (make_music.py prints their frames; shift
  the BGM item or regenerate with a different `--seconds/--bpm`). Day 1: drop 1 at f257 = mission card,
  breakdown f1029 ≈ the "one camera" section, drop 2 f1286 ≈ "I change the direction".

## 9. Zoom punch-ins (pro-camera feel)

```
browse_library {id:"library:zoom:slow-push"}       # also library:zoom:instant, library:zoom:quick-in
edit_item {adds:[{type:"effect", assetId:"library:zoom:instant", trackId:"<V1 track id>", fromFrame:1064, durationInFrames:103}]}
```
- An effect **attaches to the single V1 clip under `fromFrame` and is clipped to that clip's end.** A span
  crossing cuts needs one effect per clip (Day 1: 240–403 needed four instant zooms at 240/259/275/303).
- Library default magnification 1.4 is far too strong for a talking head. Use **1.15 slow-push, 1.2 instant,
  1.25 quick-in** (focalPointY 0.37 ≈ eyes).
- **Updating `propertyOverrides` REPLACES the whole object** (it wiped `shape` and turned every zoom into a
  plain one). Always send the full set:
  ```
  {id, propertyOverrides:{shape:"instant", direction:"in", focalPointY:0.37, magnification:1.2,
                          easeInDurationInSeconds:0, easeOutDurationInSeconds:0}}
  slow-push: shape:"slow-push", easeInDurationInSeconds:<clip seconds>      quick-in: shape:"spring-focused", easeIn 1.53
  ```
- Day-1 placement: slow-push on the hook clip, instant on the explanation run, instant on "we just use one
  camera", slow-push on the idea section, instant on "will it work", quick-in on the CTA.

## 10. Voice isolation

```
isolate_voice {action:"apply", itemId:"<V1 item>", sourceAssetId:"<talking-head asset>", strength:80}
```
- It applies to ONE item; repeat for every V1 clip (27 on Day 1). Parallel batches of ~8 calls worked fine.
- **Never `attach` one item's derivative to another item.** The derivative only covers that item's source
  range; `attach` checks just the start, so a partial overlap silently mutes the rest of the clip
  (happened on Day 1: fixed with `clear` + `apply`).

## 11. Audio polish + final checks

```
smooth_audio {}                          # 4-frame crossfades on every cut, fades on exposed edges; run LAST
edit_captions {action:"refresh"}
preview_timeline {views:["viewer"], viewerFrames:[...12 frames...]}   # look at every graphic entrance + settle
```
`smooth_audio` reports B-roll with `no_audio` skips — fine.

## 12. Export + delivery

Load the `chatcut:export` skill, then:
```
submit_export {format:"video", codec:"h264", resolution:"1080p", name:"fouad_robotics_dayN_reel"}
```
- It renders **in the open editor tab** (~8–9 min for 64 s) and the browser downloads it to `~/Downloads`.
- Start `scripts/wait_download.sh fouad_robotics_dayN_reel` with `run_in_background`, end the turn, resume on
  the notification. `track_export {action:"status"}` once if the user asks for progress — never poll it.
- Copy to `<task>/videos_for_editing/FINAL_reel_dayN.mp4`, run `verify_export.sh`, then `loudnorm.sh` to
  ~-14 LUFS (Day 1 came out at -19.8 LUFS), Read the stills, then `SendUserFile` with a still + the MP4.
  Files over 30 MiB don't reach the phone/web viewer — say so.
- Re-exports for fixes: bump the name (`_v2`, `_v3`) so the watcher can't pick up an older file.

## Shell gotchas from this session

- `pkill -f` / `ps | grep` patterns matched the invoking shell and **killed it (exit 144)**. List node
  processes with `for p in $(pgrep -x node); do tr '\0' ' ' < /proc/$p/cmdline; echo; done`.
- zsh parsed `"$i:layout"` as a history modifier → put loops in a `bash` script and write `${i}`.
- Never pipe a command whose exit status matters into `head`/`tail`; redirect to a file and check `$?`.
