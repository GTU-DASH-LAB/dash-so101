"""Language understanding: free-form instruction -> {"object": ..., "destination": ...}.

Uses any OpenAI-compatible chat endpoint (default: local ollama) with an open VLM
such as qwen2.5vl — this is where open-ended language understanding comes from.
The scene image can be attached so the model can disambiguate ("the other pen").

No new dependencies: plain urllib. Offline fallback: naive_plan() regex.
"""

import base64
import json
import os
import re
import urllib.request

ENDPOINT = os.environ.get("SO_BRAIN_LLM_ENDPOINT", "http://localhost:11434/v1")
MODEL = os.environ.get("SO_BRAIN_LLM_MODEL", "qwen2.5vl:7b")

SYSTEM = (
    "You control a tabletop robot arm that can do one thing: pick up an object and place it "
    "somewhere. First think briefly (2-4 sentences) inside <think></think> about what the user "
    "wants, what is visible, WHERE on the object a parallel gripper should grip it, and how "
    "much grip effort its material needs. Then respond with ONLY a JSON object: "
    '{"object": "<short noun phrase of the thing to pick up>", '
    '"destination": "<short noun phrase of where to place it>", '
    '"grasp": "<detector phrase for the exact part to grip — e.g. \'the handle of the mug\', '
    "'the neck of the bottle', 'the handle of the screwdriver'; \"\" if the whole object "
    'grips fine>, '
    '"effort": <grip effort 0.4-1.2: ~0.5 fragile (paper cup, egg), ~0.8 normal, '
    "~1.1 heavy or slippery (steel tool, full bottle)>}. "
    "Phrases must be visually descriptive, suitable for an object detector. "
    'If the instruction is not a pick-and-place request, respond {"error": "<why>"}.'
)

DIAGNOSE_PROMPT = """A robot arm was told: {instruction!r}.
Photo 1 is the table BEFORE its attempt, photo 2 is AFTER.
It aimed its grasp at normalized image point {pick} and its placement at {place}.
An object detector's verdict afterwards: {note}.

First think step by step inside <think></think>: compare the photos; did the object reach
the destination? If not, what is the single most likely cause — the detector found the
wrong object, the wrong destination, the grasp point was badly placed on the object
(wrong part: body instead of handle/neck), the grip effort was wrong (crushed it, or it
slipped from too weak a grip on a heavy/slippery object), the motion/policy itself
failed, or the scene changed mid-attempt?

Then respond with ONLY a JSON object:
{{"success": true/false,
 "note": "<one short sentence: what happened / what went wrong>",
 "cause": "grounding_object" | "grounding_destination" | "grasp_point" | "grip_force" | "policy" | "scene" | "none",
 "object_phrase": "<better detector phrase, ONLY if cause is grounding_object>",
 "destination_phrase": "<better phrase, ONLY if cause is grounding_destination>",
 "grasp_phrase": "<part to grip instead, e.g. 'the handle of the mug', ONLY if cause is grasp_point>",
 "pick_offset": [du, dv],  <-- ONLY if cause is grasp_point and no better part exists; each within ±0.08
 "effort": <0.4-1.2 corrected grip effort, ONLY if cause is grip_force>
}}"""


def _chat(messages: list, max_tokens: int = 700) -> str:
    # ponytail: max_tokens keeps the CoT lightweight — a few sentences, not an essay.
    body = json.dumps(
        {"model": MODEL, "temperature": 0, "max_tokens": max_tokens, "messages": messages}
    ).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def _image_part(image_path: str) -> dict:
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def _extract_json(text: str) -> tuple[dict, str]:
    """Split a think-then-JSON response into (json, thought). Handles nested JSON
    (decompose returns a list of subgoal dicts) via raw_decode from each brace."""
    thought = " ".join(re.findall(r"<think>(.*?)</think>", text, re.DOTALL)).strip()
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            obj, _ = decoder.raw_decode(stripped[match.start():])
            return obj, thought
        except json.JSONDecodeError:
            continue
    raise ValueError(f"planner returned no JSON: {text!r}")


def plan(instruction: str, image_path: str | None = None, lessons: str = "") -> dict:
    if lessons:
        instruction = (
            f"Lessons from this robot's previous failed attempts:\n{lessons}\n\n"
            f"Take them into account. Instruction: {instruction}"
        )
    content: list | str = [{"type": "text", "text": instruction}]
    if image_path:
        content.append(_image_part(image_path))
    result, thought = _extract_json(
        _chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}])
    )
    if "error" in result:
        raise ValueError(f"planner: {result['error']}")
    goal = {"object": result["object"], "destination": result["destination"], "thought": thought}
    if result.get("grasp"):
        goal["grasp"] = str(result["grasp"])
    if result.get("effort") is not None:
        goal["effort"] = _clamp_effort(result["effort"])
    return goal


def _clamp_effort(value) -> float:
    return max(0.3, min(1.3, float(value)))


def diagnose(
    instruction: str, before_path: str, after_path: str, pick, place, note: str
) -> dict | None:
    """CoT failure analysis on before/after photos: what happened, why, and what to
    change next attempt. Returns {"success", "note", "thought", "cause", and optionally
    "object_phrase" / "destination_phrase" / "pick_offset"}, or None when no VLM is
    reachable — diagnosis is best-effort and never blocks recovery."""
    prompt = DIAGNOSE_PROMPT.format(
        instruction=instruction,
        pick=(round(pick[0], 3), round(pick[1], 3)),
        place=(round(place[0], 3), round(place[1], 3)),
        note=note,
    )
    try:
        result, thought = _extract_json(
            _chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            _image_part(before_path),
                            _image_part(after_path),
                        ],
                    }
                ]
            )
        )
        diag = {
            "success": bool(result.get("success")),
            "note": str(result.get("note", "")),
            "cause": str(result.get("cause", "none")),
            "thought": thought[:300],
        }
        for key in ("object_phrase", "destination_phrase", "grasp_phrase"):
            if result.get(key):
                diag[key] = str(result[key])
        if isinstance(result.get("pick_offset"), (list, tuple)) and len(result["pick_offset"]) == 2:
            diag["pick_offset"] = [max(-0.08, min(0.08, float(d))) for d in result["pick_offset"]]
        if result.get("effort") is not None:
            diag["effort"] = _clamp_effort(result["effort"])
        return diag
    except Exception:  # noqa: BLE001
        return None


# ponytail: regex fallback for offline use / simple phrasings; the LLM handles the rest.
NAIVE_RE = re.compile(
    r"(?:put|place|move|bring|drop)\s+(?:the\s+)?(?P<obj>.+?)\s+(?:on(?:to)?|in(?:to)?|to|inside)\s+(?:the\s+)?(?P<dest>.+?)\.?$",
    re.IGNORECASE,
)


def naive_plan(instruction: str) -> dict:
    m = NAIVE_RE.search(instruction.strip())
    if not m:
        raise ValueError(
            f"could not parse {instruction!r} — use an LLM planner or pass --object/--place explicitly"
        )
    return {"object": m["obj"], "destination": m["dest"]}


DECOMPOSE_PROMPT = """You are the high-level planner of a robot arm. The low-level policy was
trained on SHORT atomic commands, each involving one object, phrased like:
"pick up the alphabet soup and place it in the basket" / "open the top drawer of the
cabinet" / "put the black bowl on the stove" / "turn on the stove".

Long instruction: {instruction!r}
{lessons}
Think step by step inside <think></think>: what minimal ordered sequence of atomic
commands completes the long instruction? Keep each command in the training phrasing.

Then respond with ONLY a JSON object:
{{"subgoals": [{{"instruction": "<atomic command>",
                "object": "<detector phrase for the object moved>",
                "destination": "<detector phrase for where it ends up, or \\"\\">",
                "grasp": "<part to grip if non-obvious (handle/neck/rim), else \\"\\">",
                "effort": <grip effort 0.4-1.2, or null for default>}}, ...]}}"""

VERIFY_PROMPT = """Photo of a robot workspace. Has this command been FULLY completed:
{subgoal!r}?
{evidence}
Think briefly inside <think></think>: where is the object now, relative to where the
command wants it? Then respond with ONLY a JSON object:
{{"done": true/false, "note": "<one short sentence>"}}"""


def decompose(instruction: str, image_path: str | None = None, lessons: str = "") -> list[dict]:
    """Long instruction -> ordered atomic subgoals with detector phrases (CoT)."""
    lessons_text = f"Lessons from previous failed attempts at this task:\n{lessons}\n" if lessons else ""
    content: list = [{"type": "text", "text": DECOMPOSE_PROMPT.format(instruction=instruction, lessons=lessons_text)}]
    if image_path:
        content.append(_image_part(image_path))
    result, thought = _extract_json(_chat([{"role": "user", "content": content}], max_tokens=1200))
    subgoals = [s for s in result.get("subgoals", []) if s.get("instruction")]
    if not subgoals:
        raise ValueError(f"decompose returned no subgoals: {result}")
    for s in subgoals:
        s.setdefault("object", ""), s.setdefault("destination", ""), s.setdefault("grasp", "")
        s["effort"] = _clamp_effort(s["effort"]) if s.get("effort") is not None else None
    if thought:
        print(f"decompose thinking: {thought[:300]}")
    return subgoals


def verify_subgoal(subgoal: str, image_path: str, evidence: str = "") -> dict:
    """One CoT VLM call: is this subgoal visibly complete? -> {"done": bool, "note": str}"""
    content = [
        {"type": "text", "text": VERIFY_PROMPT.format(subgoal=subgoal, evidence=evidence)},
        _image_part(image_path),
    ]
    result, _ = _extract_json(_chat([{"role": "user", "content": content}], max_tokens=400))
    return {"done": bool(result.get("done")), "note": str(result.get("note", ""))}


def _selftest():
    """Offline check of the think-then-JSON parsing (no network, no models)."""
    canned = (
        "<think>The user wants the mug moved. A parallel gripper should take the handle, "
        "and ceramic needs normal grip {tricky}.</think>\n"
        '{"object": "a white mug", "destination": "a black mouse pad", '
        '"grasp": "the handle of the white mug", "effort": 0.8}'
    )
    result, thought = _extract_json(canned)
    assert result["object"] == "a white mug" and "tricky" in thought
    assert result["grasp"] == "the handle of the white mug" and result["effort"] == 0.8
    assert _clamp_effort(9) == 1.3 and _clamp_effort(0.1) == 0.3

    diag_reply = (
        "<think>The pen is still at its original spot; the gripper closed on its tip and it "
        "slipped. The grasp point was too far from the center.</think>\n"
        '{"success": false, "note": "pen slipped from a tip grasp", "cause": "grasp_point", '
        '"grasp_phrase": "the middle of the pen", "pick_offset": [0.2, -0.01], "effort": 1.5}'
    )
    import tempfile

    global _chat
    real_chat = _chat
    _chat = lambda messages, max_tokens=700: diag_reply  # noqa: E731
    try:
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            f.write(b"fake png bytes")
            f.flush()
            diag = diagnose(
                "put the pen on the pad", f.name, f.name, (0.7, 0.8), (0.4, 0.8), "0.3 away"
            )
    finally:
        _chat = real_chat
    assert diag["cause"] == "grasp_point" and "slipped" in diag["thought"]
    assert diag["pick_offset"] == [0.08, -0.01], diag["pick_offset"]  # clamped to ±0.08
    assert diag["grasp_phrase"] == "the middle of the pen" and diag["effort"] == 1.3

    # Nested JSON (decompose) survives think-tag stripping and prose around it.
    nested = (
        "<think>Two objects must move, so two picks {order matters}.</think>\n"
        "Here is the plan:\n"
        '{"subgoals": [{"instruction": "pick up the alphabet soup and place it in the basket", '
        '"object": "alphabet soup can", "destination": "basket"}, '
        '{"instruction": "pick up the frying pan and place it on the stove", '
        '"object": "frying pan", "destination": "stove", "grasp": "the handle of the frying pan", '
        '"effort": 1.1}]}'
    )
    result, thought = _extract_json(nested)
    assert len(result["subgoals"]) == 2 and "order matters" in thought
    assert result["subgoals"][1]["grasp"] == "the handle of the frying pan"
    print("planner self-test OK")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)
    instruction = " ".join(sys.argv[1:]) or "put the red pen on the black mouse pad"
    try:
        print(plan(instruction))
    except Exception as e:  # noqa: BLE001 — CLI convenience: show fallback result too
        print(f"LLM planner unavailable ({e}); naive fallback:")
        print(naive_plan(instruction))
