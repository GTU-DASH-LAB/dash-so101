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
    "somewhere. Given the user's instruction (and optionally a photo of the table), respond with "
    'ONLY a JSON object: {"object": "<short noun phrase of the thing to pick up>", '
    '"destination": "<short noun phrase of where to place it>"}. '
    "Phrases must be visually descriptive (color/type), suitable for an object detector. "
    'If the instruction is not a pick-and-place request, respond {"error": "<why>"}.'
)


def plan(instruction: str, image_path: str | None = None) -> dict:
    content: list | str = [{"type": "text", "text": instruction}]
    if image_path:
        b64 = base64.b64encode(open(image_path, "rb").read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    body = json.dumps(
        {
            "model": MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": content},
            ],
        }
    ).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = json.loads(resp.read())["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"planner returned no JSON: {text!r}")
    result = json.loads(match.group())
    if "error" in result:
        raise ValueError(f"planner: {result['error']}")
    return {"object": result["object"], "destination": result["destination"]}


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


if __name__ == "__main__":
    import sys

    instruction = " ".join(sys.argv[1:]) or "put the red pen on the black mouse pad"
    try:
        print(plan(instruction))
    except Exception as e:  # noqa: BLE001 — CLI convenience: show fallback result too
        print(f"LLM planner unavailable ({e}); naive fallback:")
        print(naive_plan(instruction))
