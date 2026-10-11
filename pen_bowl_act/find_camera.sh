#!/usr/bin/env bash
# ============================================================================
# Find the cameras: lists every USB camera with its stable by-id path, grabs a
# snapshot from each into camera_snapshots/, and says which one config.env uses.
#   ./find_camera.sh
# To change camera: copy its by-id path into "index_or_path" in config.env's
# CAMERAS (only BEFORE the first demo is recorded - never change it mid-dataset).
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/lib.sh"
mkdir -p camera_snapshots
CONFIGURED="$(camera_device)"

found=0
for link in /dev/v4l/by-id/*-video-index0; do
  [ -e "$link" ] || continue
  found=1
  dev="$(readlink -f "$link")"
  name="$(basename "$link" | sed 's/^usb-//; s/-video-index0$//')"
  snap="camera_snapshots/${name}.png"
  "$PY" - "$link" "$snap" > "$RUN_DIR/findcam.out" 2>&1 <<'EOF'
import sys, time, cv2
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=sys.argv[1], width=640, height=480, fps=30, warmup_s=3))
cam.connect(); n, t = 0, time.perf_counter()
while time.perf_counter() - t < 2: img = cam.async_read(timeout_ms=1000); n += 1
cam.disconnect()
cv2.imwrite(sys.argv[2], cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(f"~{n/2:.0f} fps")
EOF
  if [ $? -eq 0 ]; then res="$(tail -1 "$RUN_DIR/findcam.out"), snapshot: $snap"; else res="could not open (in use by another program?)"; fi
  mark=""; [ "$link" = "$CONFIGURED" ] && mark="  ${C_GRN}<- configured as the wrist camera${C_OFF}"
  echo "$link  ($dev)  $res$mark"
done
[ $found -eq 1 ] || die "No USB cameras found. Plug the wrist camera in."
[ -e "$CONFIGURED" ] || warn "The configured camera $CONFIGURED is NOT plugged in."
[ -n "${DISPLAY:-}" ] && xdg-open camera_snapshots >/dev/null 2>&1 &
exit 0
