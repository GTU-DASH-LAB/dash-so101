#!/usr/bin/env bash
# Wait until a ChatCut browser export lands in ~/Downloads and stops growing.
#   wait_download.sh <export-name-without-.mp4> [timeout_s=1200]
# Run it with the Bash tool's run_in_background so the harness wakes you when it exits.
# Prints "FOUND <path> <bytes>" (exit 0) or "TIMEOUT" (exit 1).
# Uses find + stat only: never pgrep/pkill -f (they match this script's own command line).
name="$1"; timeout="${2:-1200}"
[ -n "$name" ] || { echo "usage: $0 <export-name> [timeout_s]"; exit 2; }
end=$(( $(date +%s) + timeout ))
while [ "$(date +%s)" -lt "$end" ]; do
  f=$(find "$HOME/Downloads" -maxdepth 1 -name "${name}*.mp4" -newermt '-3 hours' 2>/dev/null | head -1)
  if [ -n "$f" ] && [ ! -e "$f.crdownload" ] && [ ! -e "$f.part" ]; then
    s1=$(stat -c %s "$f"); sleep 5; s2=$(stat -c %s "$f")
    if [ "$s1" = "$s2" ] && [ "$s1" -gt 0 ]; then echo "FOUND $f $s2"; exit 0; fi
  fi
  sleep 10
done
echo "TIMEOUT"; exit 1
