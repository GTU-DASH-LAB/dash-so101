"""The full zero-shot loop: instruction -> plan -> ground -> execute on the robot.

    ./so_brain/run.sh "put the red pen on the black mouse pad"

Steps: grab one camera frame; VLM planner extracts (object, destination); OWLv2
grounds them to two image points; the point-conditioned policy is launched via
the battle-tested `lerobot-rollout` with the points encoded in the task string.

Useful flags:
    --dry-run          stop before touching the robot; save annotated preview
    --image PATH       use a saved image instead of the live camera
    --no-llm           skip the VLM, parse the instruction with a regex
    --object/--place   bypass the planner entirely with explicit phrases
"""

import argparse
import os
import subprocess
import sys

import cv2

from so_brain import grounding, planner


def capture_frame(index: int, width: int, height: int) -> str:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    cap.release()  # must release before lerobot-rollout opens the same camera
    if not ok:
        sys.exit(f"could not read from camera {index}")
    path = "so_brain/last_frame.png"
    cv2.imwrite(path, frame)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction", nargs="?", default="")
    parser.add_argument("--model", default=os.environ.get("POINTACT_MODEL", ""))
    parser.add_argument("--image")
    parser.add_argument("--object")
    parser.add_argument("--place")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env = os.environ
    image = args.image or capture_frame(
        int(env["CAMERA_INDEX"]), int(env["CAMERA_W"]), int(env["CAMERA_H"])
    )

    # 1. Understand the language.
    if args.object and args.place:
        goal = {"object": args.object, "destination": args.place}
    elif args.no_llm:
        goal = planner.naive_plan(args.instruction)
    else:
        goal = planner.plan(args.instruction, image_path=image)
    print(f"plan: pick {goal['object']!r} -> place on {goal['destination']!r}")

    # 2. Ground it to pixels (zero-shot, open vocabulary).
    pick = grounding.locate(image, goal["object"])
    place = grounding.locate(image, goal["destination"])
    print(f"pick  point: ({pick[0]:.3f}, {pick[1]:.3f})  score={pick[2]:.2f}")
    print(f"place point: ({place[0]:.3f}, {place[1]:.3f})  score={place[2]:.2f}")

    task = f"pick@{pick[0]:.3f},{pick[1]:.3f} place@{place[0]:.3f},{place[1]:.3f}"

    # Annotated preview: green = pick, blue = place.
    frame = cv2.imread(image)
    h, w = frame.shape[:2]
    cv2.circle(frame, (int(pick[0] * w), int(pick[1] * h)), 12, (0, 255, 0), 3)
    cv2.circle(frame, (int(place[0] * w), int(place[1] * h)), 12, (255, 0, 0), 3)
    cv2.imwrite("so_brain/last_plan.png", frame)
    print(f"task string: {task}   (preview: so_brain/last_plan.png)")

    if args.dry_run:
        return
    if not args.model:
        sys.exit("no policy: pass --model or set POINTACT_MODEL in config.env")

    # 3. Execute with the standard rollout stack.
    cmd = [
        "lerobot-rollout",
        "--strategy.type=base",
        "--robot.type=so101_follower",
        f"--robot.port={env['FOLLOWER_PORT']}",
        f"--robot.id={env['FOLLOWER_ID']}",
        "--robot.cameras={ front: {type: opencv, index_or_path: %s, width: %s, height: %s, fps: %s}}"
        % (env["CAMERA_INDEX"], env["CAMERA_W"], env["CAMERA_H"], env["CAMERA_FPS"]),
        f"--task={task}",
        "--policy.discover_packages_path=point_act",
        f"--policy.path={args.model}",
        f"--policy.device={env.get('POLICY_DEVICE', 'cpu')}",
        "--inference.type=sync",
        "--display_data=true",
    ]
    print("launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
