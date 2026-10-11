#!/usr/bin/env python3
"""Practice teleoperation: the follower copies the leader and the rerun window shows
the wrist camera exactly as the policy will see it. NOTHING is recorded.
Started by 0_teleoperate.sh. Ctrl+C gives you 5 s to lower the arm, then the
motors relax."""

import argparse
import json
import sys
import time

from record_session import align_follower_to_leader


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--follower-port", required=True)
    ap.add_argument("--follower-id", required=True)
    ap.add_argument("--leader-port", required=True)
    ap.add_argument("--leader-id", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SOFollowerRobotConfig
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperators.so_leader import SOLeaderTeleopConfig
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import init_visualization, log_visualization_data

    cams = {n: OpenCVCameraConfig(**{k: v for k, v in s.items() if k != "type"}) for n, s in json.loads(a.cameras).items()}
    robot = make_robot_from_config(SOFollowerRobotConfig(port=a.follower_port, id=a.follower_id, cameras=cams))
    teleop = make_teleoperator_from_config(SOLeaderTeleopConfig(port=a.leader_port, id=a.leader_id))
    try:
        robot.connect()
        teleop.connect()
        init_visualization("rerun", session_name="teleop practice")
        print("Blending the follower onto the leader (2 s) - hold the leader still ...", flush=True)
        align_follower_to_leader(robot, teleop, seconds=2.0, fps=a.fps)
        print("\nTELEOPERATING - nothing is recorded. Ctrl+C to finish.\n", flush=True)
        stop_at = None
        while stop_at is None or time.perf_counter() < stop_at:
            t0 = time.perf_counter()
            try:
                obs = robot.get_observation()
                action = teleop.get_action()
                robot.send_action(action)
                log_visualization_data("rerun", observation=obs, action=action)
            except KeyboardInterrupt:
                if stop_at is None:
                    print("\nLower the arm to its REST pose now - motors relax in 5 s.", flush=True)
                    stop_at = time.perf_counter() + 5
                continue
            precise_sleep(max(0.0, 1 / a.fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"\nSTOPPED: {type(e).__name__}: {e}\n(a cable or the arm power?) Run ./0_teleoperate.sh again.", flush=True)
        return 3
    finally:
        for dev in (robot, teleop):
            try:
                if dev.is_connected:
                    dev.disconnect()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
