#!/usr/bin/env bash
# Verify a finished reel MP4 and pull stills for a visual check.
#   verify_export.sh <video.mp4> <out_dir> [t1 t2 ...]   (default stills: 0.5 4 30 62 seconds)
# Prints codec/size/duration, integrated loudness (LUFS) + true peak, and writes
# <out_dir>/stills.jpg (stills side by side) - Read it to look at the real exported pixels.
# Exit status reflects ffprobe/ffmpeg failures (no pipes into head/tail on the checked commands).
set -u
v="$1"; out="$2"; shift 2
times=("$@"); [ ${#times[@]} -gt 0 ] || times=(0.5 4 30 62)
mkdir -p "$out" || exit 1
ffprobe -v error -show_entries format=duration,size,bit_rate:stream=codec_name,width,height,r_frame_rate,sample_rate,channels \
  -of default=nw=1 "$v" || { echo "ffprobe failed"; exit 1; }
ffmpeg -hide_banner -nostats -i "$v" -vn -af ebur128=peak=true -f null - > "$out/loudness.txt" 2>&1 || { echo "loudness pass failed"; exit 1; }
echo "--- loudness (social target ~ -14 LUFS, true peak <= -1 dBTP)"
sed -n '/Summary:/,$p' "$out/loudness.txt" | grep -E 'I:|Peak:'
ins=(); i=0
for t in "${times[@]}"; do
  ffmpeg -v error -y -ss "$t" -i "$v" -frames:v 1 -vf scale=360:-1 "$out/still_$i.jpg" || { echo "still at ${t}s failed"; exit 1; }
  ins+=(-i "$out/still_$i.jpg"); i=$((i+1))
done
if [ "$i" -gt 1 ]; then
  ffmpeg -v error -y "${ins[@]}" -filter_complex "hstack=inputs=$i" "$out/stills.jpg" || exit 1
else
  cp "$out/still_0.jpg" "$out/stills.jpg"
fi
echo "stills: $out/stills.jpg  (times: ${times[*]})"
