"""Client of robot_server.py (the arm + MolmoAct2 process on GPU0)."""

import httpx


class Body:
    def __init__(self, base_url):
        self.base = base_url.rstrip("/")
        self.http = httpx.AsyncClient(base_url=self.base, timeout=httpx.Timeout(5.0, connect=2.0))

    async def status(self):
        r = await self.http.get("/status")
        r.raise_for_status()
        return r.json()

    async def task(self, instruction, max_seconds=None):
        r = await self.http.post("/task", json={"task": instruction, "max_seconds": max_seconds})
        r.raise_for_status()
        return r.json()

    async def stop(self, home=False, reason=None):
        r = await self.http.post("/stop", json={"home": home, "reason": reason})
        r.raise_for_status()
        return r.json()

    async def home(self):
        r = await self.http.post("/home")
        r.raise_for_status()
        return r.json()

    async def frame(self, camera):
        r = await self.http.get(f"/camera/{camera}.jpg")
        r.raise_for_status()
        return r.content

    def stream(self, camera):
        return self.http.stream("GET", f"/camera/{camera}.mjpg", timeout=httpx.Timeout(None, connect=2.0))
