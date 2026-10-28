"""Yaren's voice server (GPU1): ears, voice and mind for the web app, the robot through robot_server.py.

    python -m agent.server        (run by ../start.sh with the settings from ../config.env)

Serves the React app (web/dist) on http://127.0.0.1:$UI_HTTP_PORT (this machine; the browser allows
the microphone on localhost) and https://<this machine>:$UI_HTTPS_PORT (self-signed certificate, for a
laptop on the same network). Everything except the static app needs the access token printed by
start.sh: the arm should not be drivable by anyone on the network.
"""

import asyncio
import contextlib
import json
import logging
import os
import secrets
import signal
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .body import Body
from .brain import PERSONA, TOOLS, Session
from .ears import Ears
from .llm import LLM
from .memory import STORES, Memory
from .voice import Voice

log = logging.getLogger("yaren.server")
ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "run"


def env(name, default=None, cast=str):
    v = os.environ.get(name, default)
    return cast(v) if v is not None else None


CFG = {
    "llm_url": env("LLM_URL", "http://127.0.0.1:11435"),
    "llm_model": env("LLM_MODEL", "qwen3.8:27b"),
    "llm_ctx": env("LLM_CTX", "8192", int),
    "asr_model": os.path.expanduser(env("ASR_MODEL", "~/models/yaren/whisper-large-v3-turbo")),
    "tts_model": os.path.expanduser(env("TTS_MODEL", "~/models/yaren/chatterbox")),
    "voice_refs": os.path.expanduser(env("VOICE_REFS", "~/models/yaren/voices_ref")),
    "robot_url": env("ROBOT_URL", "http://127.0.0.1:8791"),
    "http_port": env("UI_HTTP_PORT", "8080", int),
    "https_port": env("UI_HTTPS_PORT", "8443", int),
    "default_voice": env("DEFAULT_VOICE", "yaren"),
    "task_timeout": env("TASK_TIMEOUT_S", "30", float),
    "silence_ms": env("END_OF_TURN_MS", "550", int),
    "done_check": env("DONE_CHECK", "1") == "1",
    "done_check_after_s": env("DONE_CHECK_AFTER_S", "5", float),
    "done_check_every_s": env("DONE_CHECK_EVERY_S", "2.0", float),
    "done_check_streak": env("DONE_CHECK_STREAK", "2", int),
    "memory_dir": os.path.expanduser(env("MEMORY_DIR", str(RUN / "memory"))),
}


class Hub:
    """Shared by every connection: the models, the robot client and the latest robot status."""

    def __init__(self):
        self.cfg = CFG
        self.asr_pool = ThreadPoolExecutor(1, thread_name_prefix="asr")
        self.tts_pool = ThreadPoolExecutor(1, thread_name_prefix="tts")
        self.ears = Ears(CFG["asr_model"])
        self.voice = Voice(CFG["tts_model"], CFG["voice_refs"], RUN / "voices")
        self.llm = LLM(CFG["llm_url"], CFG["llm_model"], CFG["llm_ctx"])
        self.body = Body(CFG["robot_url"])
        self.memory = Memory(CFG["memory_dir"])
        self.logged_at = self.memory.last_run_at()   # runs up to here are already in runs.jsonl
        self.robot_status = None
        self.robot_error = None
        self.llm_ready = False
        self.sessions = set()
        self.saved = {}            # browser tab id -> Session.snapshot() of a closed connection

    async def poll_robot(self):
        while True:
            try:
                self.robot_status, self.robot_error = await self.body.status(), None
                self.log_runs(self.robot_status)
            except Exception as e:  # noqa: BLE001
                self.robot_status, self.robot_error = None, f"{type(e).__name__}: {e}"
            msg = {"type": "robot", "status": self.robot_status, "error": self.robot_error, "llm": self.llm_ready}
            for s in list(self.sessions):
                if s.out.qsize() < 100:    # a stalled socket: don't pile up status it will never need
                    s.out.put_nowait(("json", msg))
            await asyncio.sleep(0.5)

    def log_runs(self, status):
        """Every run robot_server.py reports goes into the memory once, whoever started it."""
        for r in status.get("recent_results") or []:
            if (r.get("at") or 0) > self.logged_at:
                self.memory.log_run(r, dry=status.get("dry"))
                self.logged_at = r["at"]

    def memory_view(self):
        dry = bool((self.robot_status or {}).get("dry"))
        return {"entries": self.memory.listing(), "dry": dry, "runs": self.memory.runs(dry)[-20:],
                "usage": {s: self.memory.usage(s) for s in STORES}}

    def health(self):
        return {"ears": True, "voice": True, "llm": self.llm_ready, "llm_model": CFG["llm_model"],
                "robot": self.robot_status is not None, "robot_error": self.robot_error,
                "dry": bool((self.robot_status or {}).get("dry")), "voices": self.voice.list(),
                "task_timeout": CFG["task_timeout"], "done_check": CFG["done_check"]}


def load_token():
    RUN.mkdir(exist_ok=True)
    p = RUN / "token"
    if not p.exists():
        p.write_text(secrets.token_urlsafe(18))
        p.chmod(0o600)
    return p.read_text().strip()


def make_app(hub, token):
    app = FastAPI()
    dist = ROOT / "web" / "dist"

    def check(request_token):
        if not secrets.compare_digest(request_token or "", token):
            raise HTTPException(401, "missing or wrong token (open the link start.sh printed)")

    def req_token(request: Request):
        return request.query_params.get("token") or request.headers.get("x-yaren-token")

    @app.get("/api/health")
    def health(request: Request):
        check(req_token(request))
        return hub.health()

    @app.get("/api/voices")
    def voices(request: Request):
        check(req_token(request))
        return hub.voice.list()

    @app.post("/api/voices")
    async def add_voice(request: Request, file: UploadFile = File(...), name: str = Form("My voice")):
        check(req_token(request))
        data = await file.read()
        if len(data) > 20 * 2**20:
            raise HTTPException(413, "recording too large")
        suffix = Path(file.filename or "rec.webm").suffix or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
        try:
            vid = await asyncio.get_running_loop().run_in_executor(hub.tts_pool, hub.voice.add_recording, f.name, name[:40])
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        finally:
            os.unlink(f.name)
        return {"id": vid, "voices": hub.voice.list()}

    @app.delete("/api/voices/{vid}")
    def del_voice(vid: str, request: Request):
        check(req_token(request))
        try:
            hub.voice.delete(vid)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return hub.voice.list()

    @app.get("/api/memory")
    def memory(request: Request):
        check(req_token(request))
        return hub.memory_view()

    @app.delete("/api/memory/{eid}")
    def forget(eid: str, request: Request):
        check(req_token(request))
        res = hub.memory.forget(eid)
        if "error" in res:
            raise HTTPException(404, res["error"])
        return hub.memory_view()

    @app.get("/api/camera/{name}.jpg")
    async def camera_jpg(name: str, request: Request):
        check(req_token(request))
        try:
            data = await hub.body.frame(name)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"robot server: {e}") from e
        return StreamingResponse(iter([data]), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/camera/{name}.mjpg")
    async def camera_mjpg(name: str, request: Request):
        check(req_token(request))

        async def relay():
            try:
                async with hub.body.stream(name) as r:
                    async for chunk in r.aiter_raw():
                        yield chunk
            except Exception as e:  # noqa: BLE001
                log.info("camera relay ended: %s", e)

        return StreamingResponse(relay(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        if not secrets.compare_digest(websocket.query_params.get("token") or "", token):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        out = asyncio.Queue()
        cid = (websocket.query_params.get("cid") or "")[:64] or None
        session = Session(hub, out, cid)
        hub.sessions.add(session)
        if session.resumed:
            log.info("tab %s reconnected: conversation of %d messages resumed", cid, len(session.history))

        async def writer():
            while True:
                kind, payload = await out.get()
                if kind == "json":
                    await websocket.send_text(json.dumps(payload, ensure_ascii=False))
                else:
                    await websocket.send_bytes(payload)

        wtask = asyncio.create_task(writer())
        session.send(type="hello", health=hub.health(), settings=session.settings,
                     resumed=session.resumed, transcript=session.transcript() if session.resumed else [])
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await session.on_audio(msg["bytes"])
                    continue
                ev = json.loads(msg.get("text") or "{}")
                t = ev.get("type")
                if t == "text" and ev.get("text", "").strip():
                    session.send(type="user", text=ev["text"].strip(), lang=None, typed=True)
                    await session.on_user_text(ev["text"].strip())
                elif t == "settings":
                    session.settings.update({k: v for k, v in ev.items() if k in session.settings})
                    session.send(type="settings", settings=session.settings)
                elif t == "ptt":
                    await session.on_ptt(bool(ev.get("down")))
                elif t == "stop":
                    await session.stop_button()
                elif t == "home":
                    await session.home_button()
                elif t == "guide":
                    await session.guide_button(bool(ev.get("on")), ev.get("level"))
                elif t == "playback":
                    session.on_playback(bool(ev.get("playing")))
                elif t == "hush":
                    await session.interrupt("hush button")
                    session._settle()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            log.exception("websocket error")
        finally:
            hub.sessions.discard(session)
            session.close()
            wtask.cancel()
            if cid:
                now = time.time()
                for k in [k for k, v in hub.saved.items() if now - v["at"] > 3600]:
                    del hub.saved[k]
                hub.saved[cid] = session.snapshot()

    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            f = dist / path
            if path and f.is_file() and dist in f.resolve().parents:
                return FileResponse(f)
            # always revalidate the page itself: a phone holding an old index.html asks for asset
            # files a rebuild has already deleted (assets/ names are content hashes, those may cache)
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})
    else:
        @app.get("/")
        def no_ui():
            return JSONResponse({"error": "web/dist missing: run ./setup.sh (npm run build in web/)"}, 503)

    return app


class _Server(uvicorn.Server):
    """Two servers (http + https) share one process: signals are handled once, in main()."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    token = load_token()
    t = time.time()
    hub = Hub()
    app = make_app(hub, token)
    servers = [_Server(uvicorn.Config(app, host="127.0.0.1", port=CFG["http_port"], lifespan="off",
                                      log_level="warning", ws_max_size=4 * 2**20, timeout_graceful_shutdown=3))]
    cert, key = RUN / "cert.pem", RUN / "key.pem"
    if cert.exists() and key.exists():
        servers.append(_Server(uvicorn.Config(app, host="0.0.0.0", port=CFG["https_port"], lifespan="off",
                                              log_level="warning", ssl_certfile=str(cert), ssl_keyfile=str(key),
                                              ws_max_size=4 * 2**20, timeout_graceful_shutdown=3)))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: [setattr(s, "should_exit", True) for s in servers])
    asyncio.create_task(hub.poll_robot())

    async def warm():
        while not hub.llm_ready:   # retry: the model may still be downloading or Ollama starting
            try:
                await hub.llm.warm(PERSONA, TOOLS)
                hub.llm_ready = True
            except Exception as e:  # noqa: BLE001
                log.error("LLM not available yet (%s); retrying in 10 s", e)
                await asyncio.sleep(10)

    asyncio.create_task(warm())
    log.info("models loaded in %.0f s; serving", time.time() - t)
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())
