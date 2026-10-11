#!/usr/bin/env bash
# Pull source frames at canvas scale (1080x1920) with a 100 px grid, side by side, so you can
# read off exactly where the face, the robot arm and other must-see objects sit BEFORE placing
# a graphic. This is how the Day-1 hook was fixed: the arm's gripper reached y~500 at the start
# of the shot, so the title was packed into y112-410.
#   frame_grid.sh <source.mp4> <out.jpg> t1 [t2 ...]     (times in SOURCE seconds)
# The output panel is scaled to 540 px wide per frame, so every grid line = 100 canvas px.
set -u
src="$1"; out="$2"; shift 2
[ $# -gt 0 ] || { echo "usage: $0 <source.mp4> <out.jpg> t1 [t2 ...]"; exit 2; }
tmp=$(mktemp -d) || exit 1
ins=(); i=0
for t in "$@"; do
  ffmpeg -v error -y -ss "$t" -i "$src" -frames:v 1 \
    -vf "scale=1080:1920,drawgrid=w=100:h=100:t=2:c=red@0.6" "$tmp/f$i.png" || { echo "frame at ${t}s failed"; exit 1; }
  ins+=(-i "$tmp/f$i.png"); i=$((i+1))
done
if [ "$i" -gt 1 ]; then
  ffmpeg -v error -y "${ins[@]}" -filter_complex "hstack=inputs=$i,scale=$((540*i)):-1" "$out" || exit 1
else
  ffmpeg -v error -y -i "$tmp/f0.png" -vf scale=540:-1 "$out" || exit 1
fi
rm -rf "$tmp"
echo "grid sheet: $out  ($i frames; 1 grid cell = 100 canvas px)"
