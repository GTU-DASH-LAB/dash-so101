"""Yaren's mind: a local LLM served by Ollama (its own instance on GPU1), native /api/chat with tools."""

import json
import logging
import time

import httpx

log = logging.getLogger("yaren.llm")


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, base_url, model, num_ctx=8192):
        self.model = model
        self.num_ctx = num_ctx
        self.http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(120.0, connect=3.0))

    def _payload(self, messages, tools=None, stream=True, **options):
        p = {"model": self.model, "messages": messages, "stream": stream, "think": False, "keep_alive": -1,
             "options": {"num_ctx": self.num_ctx, "temperature": 0.7, **options}}
        if tools:
            p["tools"] = tools
        return p

    async def stream(self, messages, tools=None):
        """Yields ("text", str) deltas, ("tools", [tool_call, ...]) and finally ("done", stats)."""
        async with self.http.stream("POST", "/api/chat", json=self._payload(messages, tools)) as r:
            if r.status_code != 200:
                raise LLMError(f"Ollama {r.status_code}: {(await r.aread()).decode()[:400]}")
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise LLMError(chunk["error"])
                msg = chunk.get("message") or {}
                if msg.get("content"):
                    yield "text", msg["content"]
                if msg.get("tool_calls"):
                    yield "tools", msg["tool_calls"]
                if chunk.get("done"):
                    yield "done", {k: chunk.get(k) for k in ("prompt_eval_count", "prompt_eval_duration",
                                                             "eval_count", "eval_duration", "total_duration")}

    async def ask(self, messages, max_tokens=8):
        r = await self.http.post("/api/chat", json=self._payload(messages, stream=False, num_predict=max_tokens,
                                                                  temperature=0.0))
        if r.status_code != 200:
            raise LLMError(f"Ollama {r.status_code}: {r.text[:400]}")
        return (r.json().get("message") or {}).get("content", "").strip()

    async def warm(self, system=None, tools=None):
        """Load the model into VRAM and keep it there (cold, the first turn took 5 s: the model load).
        It does not pre-cache the prompt: with qwen3.8 on Ollama 0.40.9 only a request that extends
        the previous one reuses the cache (measured: 1,666 tokens prefill in ~400 ms with or without a
        warm-up; follow-up turns of a conversation ~155 ms)."""
        t = time.time()
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": "Hi"}]
        r = await self.http.post("/api/chat", json=self._payload(msgs, tools, stream=False, num_predict=1))
        if r.status_code != 200:
            raise LLMError(f"Ollama {r.status_code}: {r.text[:400]}")
        log.info("LLM %s loaded in %.1f s", self.model, time.time() - t)

    async def loaded(self):
        try:
            r = await self.http.get("/api/ps", timeout=2.0)
            return [m.get("name") for m in r.json().get("models", [])]
        except Exception:  # noqa: BLE001
            return []
