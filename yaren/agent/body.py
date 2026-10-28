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

    async def gesture(self, name):
        r = await self.http.post("/gesture", json={"name": name})
        r.raise_for_status()
        return r.json()

    async def move(self, part, direction, degrees=None):
        r = await self.http.post("/move", json={"part": part, "direction": direction, "degrees": degrees})
        r.raise_for_status()
        return r.json()

    async def guide(self, on=True, level=None):
        r = await self.http.post("/guide", json={"on": on, "level": level})
        r.raise_for_status()
        return r.json()

    async def _post(self, path, **body):
        r = await self.http.post(path, json=body, timeout=httpx.Timeout(10.0, connect=2.0))
        r.raise_for_status()
        return r.json()

    async def teach(self, action, source=None, name=None):
        return await self._post("/teach", action=action, source=source, name=name)

    async def replay(self, name, speed=None):
        return await self._post("/replay", name=name, speed=speed)

    async def mirror(self, on=True):
        return await self._post("/mirror", on=on)

    async def follow(self, on=True):
        return await self._post("/follow", on=on)

    async def pose(self, name, gentle=False):
        return await self._post("/pose", name=name, gentle=gentle)

    async def dance(self, bpm, beats, plan_only=False):
        return await self._post("/dance", bpm=bpm, beats=beats, plan_only=plan_only)

    async def express(self, kind):
        return await self._post("/express", kind=kind)

    async def frame(self, camera, upright=False):
        r = await self.http.get(f"/camera/{camera}.jpg", params={"upright": 1} if upright else None)
        r.raise_for_status()
        return r.content

    async def photo(self, camera):
        """A frame turned so a face in it is upright: (jpeg bytes, a face was found)."""
        r = await self.http.get(f"/camera/{camera}.jpg", params={"upright": 1})
        r.raise_for_status()
        return r.content, r.headers.get("x-face") == "1"

    def stream(self, camera):
        return self.http.stream("GET", f"/camera/{camera}.mjpg", timeout=httpx.Timeout(None, connect=2.0))
