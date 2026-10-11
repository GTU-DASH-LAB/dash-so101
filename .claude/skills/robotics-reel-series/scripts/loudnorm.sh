#!/usr/bin/env bash
# Final loudness pass for social video: two-pass EBU R128 loudnorm on the audio, video stream copied
# untouched (no re-encode, no quality loss).
#   loudnorm.sh <in.mp4> <out.mp4> [target_LUFS=-14] [true_peak=-2]
# Day 1 exported at -19.8 LUFS from ChatCut (quiet next to other reels); this brings it to ~-14.
set -u
in="$1"; out="$2"; I="${3:--14}"; TP="${4:--2}"; LRA=11
[ "$in" != "$out" ] || { echo "refusing to overwrite the input in place"; exit 2; }
log=$(mktemp) || exit 1
ffmpeg -hide_banner -nostats -i "$in" -vn -af "loudnorm=I=$I:TP=$TP:LRA=$LRA:print_format=json" -f null - > "$log" 2>&1 \
  || { echo "measure pass failed"; cat "$log"; exit 1; }
read -r mi mtp mlra mth off < <(python3 - "$log" <<'EOF'
import json,re,sys
t=open(sys.argv[1]).read(); j=json.loads(t[t.rindex('{'):t.rindex('}')+1])
print(j["input_i"], j["input_tp"], j["input_lra"], j["input_thresh"], j["target_offset"])
EOF
) || { echo "could not parse loudnorm stats"; exit 1; }
echo "measured: I=$mi LUFS  TP=$mtp dBTP  LRA=$mlra"
ffmpeg -v error -y -i "$in" -c:v copy \
  -af "loudnorm=I=$I:TP=$TP:LRA=$LRA:measured_I=$mi:measured_TP=$mtp:measured_LRA=$mlra:measured_thresh=$mth:offset=$off:linear=true,aresample=48000" \
  -c:a aac -b:a 192k -movflags +faststart "$out" || { echo "normalize pass failed"; exit 1; }
rm -f "$log"
ffmpeg -hide_banner -nostats -i "$out" -vn -af ebur128=peak=true -f null - 2>&1 | sed -n '/Summary:/,$p' | grep -E 'I:|Peak:'
echo "wrote $out"
