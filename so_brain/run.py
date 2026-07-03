"""The full zero-shot loop with recovery: instruction -> plan -> ground -> execute
-> verify -> learn from failure -> retry.

    ./so_brain/run.sh "put the red pen on the black mouse pad"

Each attempt: grab a frame; the planner (VLM) extracts (object, destination),
informed by lessons from past failures in episodic memory; the grounder turns the
phrases into two image points; `lerobot-rollout` executes the point-conditioned
policy for --duration seconds. Afterwards the grounder re-detects the object: if it
isn't at the destination, the VLM writes a postmortem of what went wrong, the
attempt is logged to memory, and the loop re-grounds (the object may have moved!)
and tries again.

Useful flags:
    --dry-run          stop before touching the robot; save annotated preview
    --image PATH       use a saved image instead of the live camera
    --no-llm           skip the VLM planner/postmortem, parse with a regex
    --object/--place   bypass the planner with explicit phrases
    --attempts N       max attempts (default 3)   --duration S   seconds per attempt
"""

import argparse
import os
import subprocess
import sys

import cv2

from so_brain import grounding, planner
from so_brain.memory import Memory


def capture_frame(index: int, width: int, height: int, path: str) -> str:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    cap.release()  # must release before lerobot-rollout opens the same camera
    if not ok:
        sys.exit(f"could not read from camera {index}")
    cv2.imwrite(path, frame)
    return path


def save_preview(image: str, pick, place) -> None:
    frame = cv2.imread(image)
    h, w = frame.shape[:2]
    cv2.circle(frame, (int(pick[0] * w), int(pick[1] * h)), 12, (0, 255, 0), 3)
    cv2.circle(frame, (int(place[0] * w), int(place[1] * h)), 12, (255, 0, 0), 3)
    cv2.imwrite("so_brain/last_plan.png", frame)


def verify(after_image: str, obj_phrase: str, place, tol: float = 0.12) -> tuple[str, str]:
    """Detector-based outcome check: is the object now at the destination?"""
    try:
        u, v, _ = grounding.locate(after_image, obj_phrase)
    except LookupError:
        return "unknown", f"{obj_phrase!r} not visible after the attempt"
    dist = ((u - place[0]) ** 2 + (v - place[1]) ** 2) ** 0.5
    if dist <= tol:
        return "success", f"object ended {dist:.2f} from the target point"
    return "fail", f"object ended at ({u:.2f}, {v:.2f}), {dist:.2f} away from the target"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction", nargs="?", default="")
    parser.add_argument("--model", default=os.environ.get("POINTACT_MODEL", ""))
    parser.add_argument("--image")
    parser.add_argument("--object")
    parser.add_argument("--place")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--duration", type=int, default=int(os.environ.get("EPISODE_TIME_S", 25)))
    args = parser.parse_args()

    env = os.environ
    memory = Memory()
    corrections: dict = {}  # CoT diagnosis of the previous attempt -> concrete changes

    def grab(path):
        return capture_frame(int(env["CAMERA_INDEX"]), int(env["CAMERA_W"]), int(env["CAMERA_H"]), path)

    for attempt in range(1, args.attempts + 1):
        image = args.image if (args.image and attempt == 1) else grab("so_brain/last_before.png")

        # 1. Understand the language (with lessons from past failures).
        lessons = memory.lessons(args.instruction or f"{args.object} {args.place}")
        if lessons:
            print(f"lessons from memory:\n{lessons}")
        if args.object and args.place:
            goal = {"object": args.object, "destination": args.place}
        elif args.no_llm:
            goal = planner.naive_plan(args.instruction)
        else:
            goal = planner.plan(args.instruction, image_path=image, lessons=lessons)
            if goal.get("thought"):
                print(f"planner thinking: {goal['thought']}")
            # Apply the previous attempt's diagnosis (never overrides explicit flags).
            goal["object"] = corrections.get("object_phrase", goal["object"])
            goal["destination"] = corrections.get("destination_phrase", goal["destination"])
        print(f"[attempt {attempt}] pick {goal['object']!r} -> place on {goal['destination']!r}")

        # 2. Ground to pixels (re-done every attempt: failed grasps move objects).
        # The pick marker goes on the *grasp part* (handle/neck/...), not the object center.
        grasp_target = corrections.get("grasp_phrase") or goal.get("grasp") or goal["object"]
        try:
            pick = grounding.locate(image, grasp_target)
        except LookupError:
            if grasp_target == goal["object"]:
                raise
            print(f"grasp part {grasp_target!r} not found; falling back to the whole object")
            grasp_target = goal["object"]
            pick = grounding.locate(image, grasp_target)
        if grasp_target != goal["object"]:
            print(f"grasping by: {grasp_target!r}")
        place = grounding.locate(image, goal["destination"])
        if "pick_offset" in corrections:
            du, dv = corrections["pick_offset"]
            pick = (min(1.0, max(0.0, pick[0] + du)), min(1.0, max(0.0, pick[1] + dv)), pick[2])
            print(f"applying diagnosed grasp correction ({du:+.3f}, {dv:+.3f})")
        effort = corrections.get("effort", goal.get("effort"))
        print(f"pick ({pick[0]:.3f}, {pick[1]:.3f}) score={pick[2]:.2f} | place ({place[0]:.3f}, {place[1]:.3f}) score={place[2]:.2f}")
        save_preview(image, pick, place)
        task = f"pick@{pick[0]:.3f},{pick[1]:.3f} place@{place[0]:.3f},{place[1]:.3f}"
        if effort is not None:
            task += f" effort@{effort:.2f}"
            print(f"grip effort: {effort:.2f} (0.5≈fragile, 0.8≈normal, 1.1≈heavy/slippery)")
        print(f"task string: {task}   (preview: so_brain/last_plan.png)")

        if args.dry_run:
            return
        if not args.model:
            sys.exit("no policy: pass --model or set POINTACT_MODEL in config.env")

        # 3. Execute.
        cmd = [
            "lerobot-rollout",
            "--strategy.type=base",
            "--robot.type=so101_follower",
            f"--robot.port={env['FOLLOWER_PORT']}",
            f"--robot.id={env['FOLLOWER_ID']}",
            "--robot.cameras={ front: {type: opencv, index_or_path: %s, width: %s, height: %s, fps: %s}}"
            % (env["CAMERA_INDEX"], env["CAMERA_W"], env["CAMERA_H"], env["CAMERA_FPS"]),
            f"--task={task}",
            f"--duration={args.duration}",
            "--policy.discover_packages_path=point_act",
            f"--policy.path={args.model}",
            f"--policy.device={env.get('POLICY_DEVICE', 'cpu')}",
            "--inference.type=sync",
        ]
        subprocess.run(cmd, check=True)

        # 4. Verify, think about what went wrong, remember.
        after = grab("so_brain/last_after.png")
        outcome, note = verify(after, goal["object"], place)
        cause = "none"
        corrections = {}
        if not args.no_llm:
            diag = planner.diagnose(args.instruction or task, image, after, pick, place, note)
            if diag:
                if diag["thought"]:
                    print(f"diagnosis thinking: {diag['thought']}")
                outcome = "success" if diag["success"] and outcome != "fail" else outcome
                note = f"{note}; VLM: {diag['note']}"
                cause = diag["cause"]
                corrections = {
                    k: diag[k]
                    for k in ("object_phrase", "destination_phrase", "grasp_phrase",
                              "pick_offset", "effort")
                    if k in diag
                }
        print(f"[attempt {attempt}] {outcome}: {note}" + (f" (cause: {cause})" if cause != "none" else ""))
        memory.log(
            instruction=args.instruction,
            object=goal["object"],
            destination=goal["destination"],
            grasp=grasp_target,
            effort=effort,
            pick=[round(pick[0], 3), round(pick[1], 3)],
            place=[round(place[0], 3), round(place[1], 3)],
            outcome=outcome,
            note=note,
            cause=cause,
            attempt=attempt,
        )
        if outcome == "success":
            return
        if corrections:
            print(f"next attempt will apply: {corrections}")
        print("retrying with a fresh look at the scene..." if attempt < args.attempts else "giving up.")


if __name__ == "__main__":
    main()
