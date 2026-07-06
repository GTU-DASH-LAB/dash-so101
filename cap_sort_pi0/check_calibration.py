"""
Quick calibration sanity check -- NOT a recalibration.

Disables torque so you can freely move the arm by hand. Move each of the 6
joints slowly through its full physical range, one at a time, then Ctrl+C.
Prints a summary per joint: the min/max it saw, and the largest single-tick
jump -- a large jump (many degrees between two readings 0.1s apart) would
suggest a servo horn slipped relative to its shaft, since a hand move can't
physically produce a huge instantaneous change.
"""
import sys
import time

from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower

FOLLOWER_PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM1"
FOLLOWER_ID = sys.argv[2] if len(sys.argv) > 2 else "my_follower"

robot_cfg = SOFollowerRobotConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, cameras={})
robot = SOFollower(robot_cfg)
robot.connect(calibrate=False)
robot.bus.disable_torque()
print("Torque disabled -- arm is now freely movable by hand.")
print("Move each joint (one at a time) slowly through its full range. Ctrl+C when done.\n")

stats = {}  # name -> {min, max, max_jump, last}

try:
    while True:
        obs = robot.get_observation()
        for k, v in obs.items():
            if not k.endswith(".pos"):
                continue
            name = k.removesuffix(".pos")
            s = stats.setdefault(name, {"min": v, "max": v, "max_jump": 0.0, "last": v})
            s["min"] = min(s["min"], v)
            s["max"] = max(s["max"], v)
            s["max_jump"] = max(s["max_jump"], abs(v - s["last"]))
            s["last"] = v
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\n=== Summary ===")
    for name, s in stats.items():
        span = s["max"] - s["min"]
        flag = "  <-- LARGE JUMP, check this joint" if s["max_jump"] > 15 else ""
        print(f"{name:>14}: min={s['min']:7.1f}  max={s['max']:7.1f}  span={span:6.1f}  max_single_tick_jump={s['max_jump']:5.1f}{flag}")
finally:
    robot.disconnect()
