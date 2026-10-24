import { useCallback, useEffect, useRef, useState } from "react";
import { Mic, Player } from "./audio.js";
import { T } from "./i18n.js";

function getToken() {
  const url = new URL(location.href);
  const t = url.searchParams.get("token");
  if (t) {
    try { localStorage.setItem("yarenToken", t); } catch { /* private mode */ }
    url.searchParams.delete("token");
    history.replaceState(null, "", url.pathname + url.search);
    return t;
  }
  try { return localStorage.getItem("yarenToken") || ""; } catch { return ""; }
}

const TOKEN = getToken();
let msgId = 0;

export default function App() {
  const [awake, setAwake] = useState(false);
  const [connected, setConnected] = useState(false);
  const [health, setHealth] = useState(null);
  const [robot, setRobot] = useState(null);
  const [robotErr, setRobotErr] = useState(null);
  const [llmReady, setLlmReady] = useState(false);
  const [settings, setSettings] = useState({ lang: "auto", voice: "yaren", bargeIn: true, handsFree: true, speak: true });
  const [state, setState] = useState("idle");
  const [vad, setVad] = useState(false);
  const [msgs, setMsgs] = useState([]);
  const [micOn, setMicOn] = useState(true);
  const [micErr, setMicErr] = useState(null);
  const [ptt, setPtt] = useState(false);
  const [voices, setVoices] = useState([]);
  const [text, setText] = useState("");
  const [perf, setPerf] = useState({});
  const [camKey, setCamKey] = useState(0);
  const [camOk, setCamOk] = useState(true);
  const [recOpen, setRecOpen] = useState(false);
  const [memOpen, setMemOpen] = useState(false);
  const [memVersion, setMemVersion] = useState(0);

  const ws = useRef(null);
  const mic = useRef(null);
  const player = useRef(null);
  const audioMeta = useRef(null);
  const orb = useRef(null);
  const live = useRef({ micOn: true, ptt: false, handsFree: true });
  const chatEnd = useRef(null);

  const t = T[settings.lang === "en" ? "en" : "tr"];

  const push = useCallback((m) => setMsgs((ms) => [...ms.slice(-200), { id: ++msgId, ...m }]), []);
  const send = useCallback((obj) => {
    if (ws.current?.readyState === 1) ws.current.send(JSON.stringify(obj));
  }, []);

  useEffect(() => { live.current = { micOn, ptt, handsFree: settings.handsFree }; }, [micOn, ptt, settings.handsFree]);
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [msgs]);

  // ---------------------------------------------------------------- websocket
  const onEvent = useCallback((ev) => {
    switch (ev.type) {
      case "hello":
        setHealth(ev.health); setVoices(ev.health.voices); setLlmReady(ev.health.llm);
        setSettings((s) => ({ ...s, ...ev.settings }));
        break;
      case "settings": setSettings((s) => ({ ...s, ...ev.settings })); break;
      case "state": setState(ev.state); break;
      case "vad": setVad(ev.speaking); break;
      case "robot":
        setRobot(ev.status); setRobotErr(ev.error); setLlmReady(ev.llm);
        break;
      case "user":
        push({ role: "user", text: ev.text, lang: ev.lang, typed: ev.typed });
        if (ev.asr_ms != null) setPerf((p) => ({ ...p, asr: ev.asr_ms }));
        break;
      case "asr_empty": break;
      case "say":
        setMsgs((ms) => {
          const last = ms[ms.length - 1];
          if (last && last.role === "assistant" && last.gen === ev.gen && !last.closed) {
            return [...ms.slice(0, -1), { ...last, text: last.text + " " + ev.text }];
          }
          return [...ms.slice(-200), { id: ++msgId, role: "assistant", text: ev.text, gen: ev.gen }];
        });
        break;
      case "audio":
        audioMeta.current = ev;
        setPerf((p) => ({ ...p, tts: ev.tts_ms }));
        break;
      case "audio_stop":
        player.current?.stop();
        setMsgs((ms) => ms.map((m, i) => (i === ms.length - 1 && m.role === "assistant" ? { ...m, closed: true } : m)));
        break;
      case "tool":
        push({ role: "tool", name: ev.name, args: ev.args, result: ev.result });
        setMsgs((ms) => ms.map((m) => (m.role === "assistant" ? { ...m, closed: true } : m)));
        break;
      case "run_end": push({ role: "event", kind: "run_end", ...ev }); break;
      case "done_check": push({ role: "event", kind: "done_check", ...ev }); break;
      case "look": push({ role: "event", kind: "look", camera: ev.camera }); break;
      case "memory": setMemVersion((v) => v + 1); break;
      case "llm_stats":
        setPerf((p) => ({ ...p, first: ev.first_sentence_ms ?? p.first,
          tps: ev.eval_count && ev.eval_duration ? Math.round(ev.eval_count / (ev.eval_duration / 1e9)) : p.tps }));
        break;
      case "error": push({ role: "event", kind: "error", text: ev.message }); break;
      default: break;
    }
  }, [push]);

  useEffect(() => {
    if (!awake || !TOKEN) return undefined;
    let closed = false;
    let retry = null;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const sock = new WebSocket(`${proto}://${location.host}/ws?token=${encodeURIComponent(TOKEN)}`);
      sock.binaryType = "arraybuffer";
      sock.onopen = () => { setConnected(true); };
      sock.onclose = () => {
        setConnected(false);
        player.current?.stop();
        if (!closed) retry = setTimeout(connect, 1500);
      };
      sock.onmessage = (e) => {
        if (typeof e.data === "string") onEvent(JSON.parse(e.data));
        else if (audioMeta.current) player.current?.play(new Int16Array(e.data), audioMeta.current.sr);
      };
      ws.current = sock;
    };
    connect();
    return () => { closed = true; clearTimeout(retry); ws.current?.close(); };
  }, [awake, onEvent]);

  // Push the current settings whenever they change locally.
  const updateSettings = (patch) => {
    setSettings((s) => ({ ...s, ...patch }));
    send({ type: "settings", ...patch });
  };

  // ---------------------------------------------------------------- audio
  const wake = async () => {
    player.current = new Player((playing) => send({ type: "playback", playing }));
    await player.current.resume();
    mic.current = new Mic((pcm) => {
      const l = live.current;
      if (ws.current?.readyState === 1 && l.micOn && (l.handsFree || l.ptt)) ws.current.send(pcm.buffer);
    });
    try {
      await mic.current.start();
    } catch (e) {
      setMicErr(String(e.message || e));
      setMicOn(false);
    }
    setAwake(true);
  };

  // Orb animation: driven by the mic level while listening and the speech level while speaking.
  useEffect(() => {
    if (!awake) return undefined;
    let raf;
    const tick = () => {
      const m = mic.current?.level || 0;
      const p = player.current?.level() || 0;
      const amp = Math.min(1, Math.max(p * 4, live.current.micOn ? m * 6 : 0));
      orb.current?.style.setProperty("--amp", amp.toFixed(3));
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [awake]);

  // ---------------------------------------------------------------- keys
  const pttDown = useCallback(() => {
    if (live.current.ptt) return;
    setPtt(true); live.current.ptt = true;
    player.current?.stop();
    send({ type: "ptt", down: true });
  }, [send]);
  const pttUp = useCallback(() => {
    if (!live.current.ptt) return;
    setPtt(false); live.current.ptt = false;
    send({ type: "ptt", down: false });
  }, [send]);
  const stopAll = useCallback(() => { player.current?.stop(); send({ type: "stop" }); }, [send]);

  useEffect(() => {
    const typing = (e) => ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName);
    const down = (e) => {
      if (e.key === "Escape") { e.preventDefault(); stopAll(); }
      else if (e.code === "Space" && !typing(e) && !e.repeat && !settings.handsFree) { e.preventDefault(); pttDown(); }
    };
    const up = (e) => { if (e.code === "Space" && !typing(e) && !settings.handsFree) { e.preventDefault(); pttUp(); } };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, [stopAll, pttDown, pttUp, settings.handsFree]);

  const submitText = (e) => {
    e.preventDefault();
    const s = text.trim();
    if (!s) return;
    player.current?.stop();
    send({ type: "text", text: s });
    setText("");
  };

  // ---------------------------------------------------------------- render
  if (!TOKEN) {
    return <div className="splash"><Logo /><p className="warn">{t.noToken}</p></div>;
  }
  if (!awake) {
    return (
      <div className="splash">
        <Logo />
        <p className="tag">{t.tagline}</p>
        <button className="wake" onClick={wake}>{t.wake}</button>
        <p className="hint">{t.wakeHint}</p>
      </div>
    );
  }

  const rstate = robot ? robot.state : "offline";
  const camName = robot?.cameras?.[0] || "wrist";
  const progress = robot?.state === "running" && robot.elapsed_s != null ? Math.min(1, robot.elapsed_s / robot.max_seconds) : 0;
  const motion = robot?.state === "moving" ? robot.motion : null;
  const shownState = vad && state !== "speaking" ? "listening" : state;

  return (
    <div className="app">
      <header>
        <Logo small />
        {/* narrow screens (phone/tablet): the big STOP scrolls away, so a copy stays in the header */}
        <button className="stop head-stop" onClick={stopAll}>■ {t.stop}</button>
        <div className="chips">
          <Chip ok={connected} label={t.ears} />
          <Chip ok={connected} label={t.voice} />
          <Chip ok={connected && llmReady} label={t.mind} warn={connected && !llmReady} />
          <Chip ok={!!robot} label={t.body} title={robotErr || ""} />
          {robot?.dry && <span className="badge">{t.dry}</span>}
        </div>
        <div className="perf">
          {perf.asr != null && <span>ASR {perf.asr} ms</span>}
          {perf.first != null && <span>LLM→1. {perf.first} ms</span>}
          {perf.tps != null && <span>{perf.tps} tok/s</span>}
          {perf.tts != null && <span>TTS {perf.tts} ms</span>}
        </div>
      </header>

      {!connected && <div className="banner">{t.disconnected}</div>}

      <main>
        <section className="left">
          <div ref={orb} className={`orb ${shownState}`}>
            <div className="core" />
            <div className="ring r1" />
            <div className="ring r2" />
          </div>
          <div className="who">Yaren</div>
          <div className={`state-label ${shownState}`}>{t.states[shownState] || shownState}</div>

          <div className="controls">
            <button className={`toggle ${micOn ? "on" : ""}`} onClick={() => setMicOn(!micOn)} disabled={!!micErr}>
              {micOn ? "🎙 " + t.mic : "🔇 " + t.micOff}
            </button>
            <div className="seg">
              <button className={settings.handsFree ? "on" : ""} onClick={() => updateSettings({ handsFree: true })}>{t.handsFree}</button>
              <button className={!settings.handsFree ? "on" : ""} onClick={() => updateSettings({ handsFree: false })}>{t.ptt}</button>
            </div>
            {micErr && <div className="warn small">{micErr}</div>}
            {!settings.handsFree && (
              <button className={`ptt ${ptt ? "held" : ""}`} onPointerDown={pttDown} onPointerUp={pttUp} onPointerLeave={pttUp}>
                {t.pttHold}
              </button>
            )}
            <label>{t.lang}
              <select value={settings.lang} onChange={(e) => updateSettings({ lang: e.target.value })}>
                <option value="auto">{t.auto} (TR/EN)</option>
                <option value="tr">Türkçe</option>
                <option value="en">English</option>
              </select>
            </label>
            <label>{t.voiceSel}
              <div className="row">
                <select value={settings.voice} onChange={(e) => updateSettings({ voice: e.target.value })}>
                  {voices.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}
                </select>
                <button className="ghost" onClick={() => setRecOpen(true)}>⏺ {t.record}</button>
              </div>
            </label>
            <button className="ghost mem-btn" onClick={() => setMemOpen(true)}>🧠 {t.memory}</button>
            <label className="check"><input type="checkbox" checked={settings.bargeIn} onChange={(e) => updateSettings({ bargeIn: e.target.checked })} /> {t.bargeIn}</label>
            <label className="check"><input type="checkbox" checked={settings.speak} onChange={(e) => updateSettings({ speak: e.target.checked })} /> {t.speak}</label>
          </div>
        </section>

        <section className="center">
          <div className="cam">
            <div className="cam-title">{t.camera}</div>
            {camOk ? (
              <img
                key={camKey}
                src={`/api/camera/${camName}.mjpg?token=${encodeURIComponent(TOKEN)}&k=${camKey}`}
                alt={t.camera}
                onError={() => { setCamOk(false); setTimeout(() => { setCamOk(true); setCamKey((k) => k + 1); }, 3000); }}
              />
            ) : (
              <div className="nocam">{t.noCamera}</div>
            )}
            <div className={`robot-chip ${rstate}`}>{t.robot[rstate] || rstate}</div>
            {robot?.state === "running" && (
              <div className="task-overlay">
                <div className="task-text">{robot.task}</div>
                <div className="bar"><div style={{ width: `${progress * 100}%` }} /></div>
                <div className="task-time">{robot.elapsed_s?.toFixed(0)} / {robot.max_seconds?.toFixed(0)} s</div>
              </div>
            )}
            {motion && (
              <div className="task-overlay">
                <div className="task-text">{t.gestures[motion.name] || motion.name}</div>
                <div className="bar"><div style={{ width: `${Math.min(1, motion.elapsed_s / Math.max(motion.seconds, 0.1)) * 100}%` }} /></div>
                <div className="task-time">{motion.elapsed_s?.toFixed(1)} / {motion.seconds?.toFixed(1)} s</div>
              </div>
            )}
          </div>
          <div className="robot-info">
            {robot?.last_result && (
              <span>{t.lastRun}: <b>{robot.last_result.task}</b> — {t.ended[robot.last_result.ended] || robot.last_result.ended} ({robot.last_result.seconds} s)</span>
            )}
            {robot?.chunk_ms_last != null && <span>{t.chunk} {robot.chunk_ms_last} ms</span>}
            {robot && <span>{t.tick} p50 {robot.tick_ms_p50} ms</span>}
          </div>
          <div className="actions">
            <button className="stop" onClick={stopAll} title={t.stopHint}>■ {t.stop} <small>{t.stopHint}</small></button>
            <button className="ghost" onClick={() => send({ type: "home" })}>⌂ {t.home}</button>
            <button className="ghost" onClick={() => { player.current?.stop(); send({ type: "hush" }); }}>🤫 {t.hush}</button>
          </div>
        </section>

        <section className="right">
          <div className="chat">
            {msgs.map((m) => <Message key={m.id} m={m} t={t} />)}
            <div ref={chatEnd} />
          </div>
          <form className="composer" onSubmit={submitText}>
            <input value={text} onChange={(e) => setText(e.target.value)} placeholder={t.type} />
            <button type="submit">{t.send}</button>
          </form>
        </section>
      </main>

      {memOpen && <MemoryPanel t={t} version={memVersion} onClose={() => setMemOpen(false)} />}
      {recOpen && (
        <Recorder t={t} mic={mic.current} onClose={() => setRecOpen(false)}
          onSaved={(vid, list) => { setVoices(list); updateSettings({ voice: vid }); setRecOpen(false); }} />
      )}
    </div>
  );
}

function Logo({ small }) {
  return (
    <div className={`logo ${small ? "small" : ""}`}>
      <img src="/yaren.svg" alt="" />
      <span>YAREN</span>
    </div>
  );
}

function Chip({ ok, warn, label, title }) {
  return <span className={`chip ${ok ? "ok" : warn ? "warn" : "bad"}`} title={title}><i />{label}</span>;
}

function Message({ m, t }) {
  const moved = (r) => r?.moved != null ? ` · ${r.moved > 0 ? "+" : ""}${r.moved}${r.unit === "%" ? "%" : "°"}${r.limited_by ? ` (${r.limited_by})` : ""}` : "";
  if (m.role === "user") return <div className="msg user"><div className="bubble">{m.text}</div></div>;
  if (m.role === "assistant") return <div className="msg bot"><img src="/yaren.svg" alt="" /><div className="bubble">{m.text}</div></div>;
  if (m.role === "tool") {
    const icon = { do_task: "🦾", stop_robot: "■", go_home: "⌂", look: "👁", robot_status: "ℹ", remember: "🧠", forget: "🗑", gesture: "👋", move_part: "↕" }[m.name] || "⚙";
    const detail = m.name === "do_task" ? `“${m.args?.instruction}”${m.result?.trained_skill === false ? " · untrained" : ""}`
      : m.name === "remember" ? `“${m.args?.fact}”` : m.name === "forget" ? `“${m.result?.forgotten ?? m.args?.id}”`
      : m.name === "gesture" ? (t.gestures[m.args?.name] || m.args?.name || "")
      : m.name === "move_part" ? `${m.args?.part} ${m.args?.direction}${moved(m.result)}` : "";
    const err = m.result?.error ? ` · ${m.result.error}` : "";
    return <div className="msg tool">{icon} {m.name} {detail}{err}</div>;
  }
  if (m.kind === "run_end") return <div className="msg event">🏁 {m.task} — {t.ended[m.ended] || m.ended} {m.seconds != null ? `(${m.seconds} s)` : ""}</div>;
  if (m.kind === "done_check") return <div className="msg event faint">👁 {m.answer === "yes" ? `✓ ${m.streak}` : "…"}</div>;
  if (m.kind === "look") return <div className="msg event faint">👁 {m.camera}</div>;
  if (m.kind === "error") return <div className="msg event err">⚠ {m.text}</div>;
  return null;
}

function MemoryPanel({ t, version, onClose }) {
  const [mem, setMem] = useState(null);
  const [err, setErr] = useState("");
  const call = useCallback(async (method, path) => {
    try {
      const r = await fetch(path, { method, headers: { "x-yaren-token": TOKEN } });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.statusText);
      setMem(j); setErr("");
    } catch (e) { setErr(String(e.message || e)); }
  }, []);
  useEffect(() => { call("GET", "/api/memory"); }, [call, version]);

  const group = (about, title) => {
    const list = (mem?.entries || []).filter((e) => e.about === about);
    const [used, limit] = mem?.usage?.[about] || [0, 1];
    return (
      <div className="mem-group">
        <div className="mem-head"><b>{title}</b><span className="mem-usage"><i style={{ width: `${Math.min(100, (100 * used) / limit)}%` }} /></span><small>{used}/{limit}</small></div>
        {list.length ? list.map((e) => (
          <div key={e.id} className="mem-entry">
            <span>{e.fact}</span>
            <button className="ghost" title={t.memForget} onClick={() => call("DELETE", `/api/memory/${e.id}`)}>✕</button>
          </div>
        )) : <div className="mem-empty">{t.memEmpty}</div>}
      </div>
    );
  };
  const runs = [...(mem?.runs || [])].reverse().slice(0, 10);
  return (
    <div className="modal" onClick={onClose}>
      <div className="card mem-card" onClick={(e) => e.stopPropagation()}>
        <h3>🧠 {t.memTitle}</h3>
        <p className="hint">{t.memHint}</p>
        {err && <p className="warn">{err}</p>}
        {group("person", t.memPeople)}
        {group("yaren", t.memNotes)}
        <div className="mem-group">
          <div className="mem-head"><b>{mem?.dry ? t.memDryRuns : t.memRuns}</b></div>
          {runs.length ? runs.map((r) => (
            <div key={r.at} className="mem-run">
              <small>{new Date(r.at * 1000).toLocaleString(t.locale, { dateStyle: "medium", timeStyle: "short" })}</small>
              <span>{r.task} — {t.ended[r.ended] || r.ended} ({r.seconds} s)</span>
            </div>
          )) : <div className="mem-empty">{t.memEmpty}</div>}
        </div>
        <button className="ghost" onClick={onClose}>{t.close}</button>
      </div>
    </div>
  );
}

function Recorder({ t, mic, onClose, onSaved }) {
  const [phase, setPhase] = useState("ready"); // ready | rec | named | upload | error
  const [left, setLeft] = useState(10);
  const [name, setName] = useState("");
  const [err, setErr] = useState("");
  const rec = useRef(null);
  const chunks = useRef([]);
  const blob = useRef(null);
  const timer = useRef(null);

  const start = () => {
    if (!mic?.stream) { setErr("microphone not available"); setPhase("error"); return; }
    chunks.current = [];
    rec.current = mic.recorder();
    rec.current.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
    rec.current.onstop = () => { blob.current = new Blob(chunks.current, { type: rec.current.mimeType }); setPhase("named"); };
    rec.current.start();
    setPhase("rec");
    setLeft(10);
    timer.current = setInterval(() => setLeft((l) => {
      if (l <= 1) { clearInterval(timer.current); rec.current?.state === "recording" && rec.current.stop(); return 0; }
      return l - 1;
    }), 1000);
  };
  const finish = () => { clearInterval(timer.current); rec.current?.state === "recording" && rec.current.stop(); };
  const save = async () => {
    setPhase("upload");
    const fd = new FormData();
    const ext = (blob.current.type || "").includes("mp4") ? "m4a" : (blob.current.type || "").includes("ogg") ? "ogg" : "webm";
    fd.append("file", blob.current, `voice.${ext}`);
    fd.append("name", name.trim() || "My voice");
    try {
      const r = await fetch("/api/voices", { method: "POST", body: fd, headers: { "x-yaren-token": TOKEN } });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.statusText);
      onSaved(j.id, j.voices);
    } catch (e) { setErr(String(e.message || e)); setPhase("error"); }
  };
  useEffect(() => () => clearInterval(timer.current), []);

  return (
    <div className="modal" onClick={onClose}>
      <div className="card" onClick={(e) => e.stopPropagation()}>
        <h3>{t.recTitle}</h3>
        <p className="hint">{t.recText}</p>
        <blockquote>{t.recSample}</blockquote>
        {phase === "ready" && <button className="wake" onClick={start}>⏺ {t.recStart}</button>}
        {phase === "rec" && <button className="stop" onClick={finish}>■ {t.recStop} ({left})</button>}
        {phase === "named" && (
          <div className="row">
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder={t.recName} autoFocus />
            <button className="wake" onClick={save}>{t.recSave}</button>
          </div>
        )}
        {phase === "upload" && <p>{t.recUploading}</p>}
        {phase === "error" && <p className="warn">{err}</p>}
        <button className="ghost" onClick={onClose}>{t.cancel}</button>
      </div>
    </div>
  );
}
