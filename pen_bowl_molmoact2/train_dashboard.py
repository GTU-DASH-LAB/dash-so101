#!/usr/bin/env python3
"""Live training dashboard for pen_bowl_molmoact2 (started by ./2_train_local.sh).

    train_dashboard.py --serve     web page at http://DASHBOARD_HOST:DASHBOARD_PORT
    train_dashboard.py --status    one text summary (./2_train_local.sh status)
    train_dashboard.py --png       write run/loss_curve.png

Everything is parsed from files, so it works for a running, finished or crashed
run and survives restarts:
  run/plan.json     total steps / epochs / steps-per-epoch (train_plan.py)
  run/train.log     lerobot's metric lines:
                    "INFO <date time> ... step:2K ... epch:4.31 loss:0.212 grdn:.. lr:.. updt_s:.. data_s:.."
                    and validation lines: "INFO <date time> ... step 2305: eval_loss=0.1834"
  run/watchdog.log  STARTING / RESUMING / CRASHED / FINISHED / STOPPED / GAVE UP
Steps are recovered from `epch` (exact to 2 decimals). lerobot prints `step:` as
"2K", so it cannot be read back exactly.

"Train loss" is MolmoAct2's continuous-action objective (flow matching on the action
expert). "Validation loss" is the same objective on held-out demos. Flow matching
samples random noise levels, so both curves are noisy; read the trend.
"""

import argparse
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
METRIC = re.compile(
    r"INFO (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \S+ step:\S+ smpl:\S+ ep:\S+ epch:([0-9.]+) loss:([0-9.]+)"
    r" grdn:([0-9.]+) lr:(\S+) updt_s:([0-9.]+) data_s:([0-9.]+)"
)
EVAL = re.compile(r"INFO (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \S+ step (\d+): eval_loss=([0-9.]+)")
CKPT = re.compile(r"^\d{6}$")


def env_config() -> dict:
    cfg = {}
    for line in (HERE / "config.env").read_text().splitlines():
        m = re.match(r"^([A-Z_]+)=(.*)$", line)
        if m:
            v = re.sub(r"\s+#.*$", "", m.group(2)).strip().strip("'\"")
            cfg[m.group(1)] = v
    return cfg


def ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


def pid_alive(name: str) -> bool:
    try:
        os.kill(int((RUN / name).read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def gpu_stats() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return []
    gpus = []
    for line in out.strip().splitlines():
        i, util, used, total, temp, power = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(i), "util": float(util), "mem_used": float(used) / 1024,
                     "mem_total": float(total) / 1024, "temp": float(temp), "power": float(power)})
    return gpus


def collect() -> dict:
    cfg = env_config()
    plan = json.loads((RUN / "plan.json").read_text()) if (RUN / "plan.json").exists() else None
    text = (RUN / "train.log").read_text(errors="ignore").replace("\r", "\n") if (RUN / "train.log").exists() else ""
    wlog = (RUN / "watchdog.log").read_text().splitlines() if (RUN / "watchdog.log").exists() else []

    # Last occurrence wins: after a resume, lerobot repeats the steps since the last checkpoint.
    points: dict[float, dict] = {}
    for m in METRIC.finditer(text):
        e = float(m.group(2))
        points[e] = {"t": ts(m.group(1)), "epoch": e, "loss": float(m.group(3)), "grad_norm": float(m.group(4)),
                     "lr": m.group(5), "updt_s": float(m.group(6)), "data_s": float(m.group(7))}
    train = sorted(points.values(), key=lambda p: p["epoch"])
    evals: dict[int, dict] = {}
    for m in EVAL.finditer(text):
        evals[int(m.group(2))] = {"t": ts(m.group(1)), "step": int(m.group(2)), "loss": float(m.group(3))}

    out = {"now": time.time(), "plan": plan, "task": cfg.get("TASK"), "train": train, "val": [], "epoch_train": [],
           "gpus": gpu_stats(), "checkpoints": [], "watchdog": wlog[-8:]}
    if not plan:
        out["state"] = "NOT STARTED"
        return out
    spe, total, frames, eff = plan["steps_per_epoch"], plan["total_steps"], plan["train_frames"], plan["effective_batch"]

    out["val"] = [{"epoch": s / spe, "step": s, "loss": v["loss"]} for s, v in sorted(evals.items())]
    # Mean train loss per epoch: the log points falling inside (k-1, k].
    bins: dict[int, list[float]] = {}
    for p in train:
        bins.setdefault(max(1, math.ceil(p["epoch"] - 1e-9)), []).append(p["loss"])
    out["epoch_train"] = [{"epoch": k, "loss": sum(v) / len(v)} for k, v in sorted(bins.items())]

    step = 0
    if train:
        step = min(total, round(train[-1]["epoch"] * frames / eff))
    if evals:
        step = max(step, max(evals))
    out["step"], out["epoch"] = step, step / spe

    # Speed from wall-clock over recent log points of the current attempt (gap < 10 min).
    recent = [p for p in train if p["t"] > out["now"] - 3600]
    rate = None
    if len(recent) >= 3:
        seg = [recent[-1]]
        for p in reversed(recent[:-1]):
            if seg[-1]["t"] - p["t"] > 600 or len(seg) >= 30:
                break
            seg.append(p)
        if len(seg) >= 2:
            d_ep = seg[0]["epoch"] - seg[-1]["epoch"]
            d_t = seg[0]["t"] - seg[-1]["t"]
            if d_ep > 0 and d_t > 0:
                rate = d_ep * frames / eff / d_t  # steps per second, eval + checkpoints included
    out["steps_per_s"] = rate
    out["samples_per_s"] = rate * eff if rate else None
    out["eta_s"] = (total - step) / rate if rate else None
    starts = [ln for ln in wlog if "STARTING" in ln]
    out["started_at"] = ts(starts[-1][:19]) if starts else None

    last = wlog[-1] if wlog else ""
    if "FINISHED" in last:
        out["state"] = "FINISHED"
    elif "GAVE UP" in last:
        out["state"] = "FAILED"
    elif "STOPPED" in last:
        out["state"] = "STOPPED"
    elif pid_alive("watchdog.pid"):
        out["state"] = "RETRYING" if "RETRYING" in last else ("TRAINING" if train else "STARTING")
    else:
        out["state"] = "STOPPED"

    ckroot = HERE / cfg.get("OUTPUT_DIR", "outputs/train/molmoact2_pen_bowl") / "checkpoints"
    val_by_step = {v["step"]: v["loss"] for v in out["val"]}
    if ckroot.is_dir():
        for d in sorted(ckroot.iterdir()):
            if CKPT.match(d.name):
                s = int(d.name)
                out["checkpoints"].append({"name": d.name, "step": s, "epoch": s / spe, "val": val_by_step.get(s)})
    vals = [c for c in out["checkpoints"] if c["val"] is not None]
    if vals:
        min(vals, key=lambda c: c["val"])["best"] = True
    return out


def hms(s: float | None) -> str:
    if s is None:
        return "-"
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"


def text_status() -> str:
    d = collect()
    p = d["plan"]
    if not p:
        return "No training run yet. Start one with ./2_train_local.sh"
    lines = [f"State:      {d['state']}"]
    pct = 100 * d["step"] / p["total_steps"]
    bar = "#" * int(pct / 100 * 40)
    lines.append(f"Progress:   [{bar:<40}] {pct:5.1f}%   epoch {d['epoch']:.2f}/{p['epochs']}   step {d['step']:,}/{p['total_steps']:,}")
    if d["train"]:
        lines.append(f"Train loss: {d['train'][-1]['loss']:.4f}   (epoch mean {d['epoch_train'][-1]['loss']:.4f})")
    if d["val"]:
        best = min(d["val"], key=lambda v: v["loss"])
        lines.append(f"Val loss:   {d['val'][-1]['loss']:.4f}   (best {best['loss']:.4f} at epoch {best['epoch']:.0f})")
    if d["state"] in ("TRAINING", "STARTING", "RETRYING") and d["eta_s"]:
        fin = datetime.fromtimestamp(d["now"] + d["eta_s"]).strftime("%H:%M")
        lines.append(f"Speed:      {d['samples_per_s']:.0f} samples/s")
        lines.append(f"Remaining:  {hms(d['eta_s'])}   -> finishes ~{fin}")
    for g in d["gpus"]:
        lines.append(f"GPU{g['index']}:       {g['util']:.0f}% busy, {g['mem_used']:.1f}/{g['mem_total']:.0f} GB, {g['temp']:.0f} C")
    if d["checkpoints"]:
        lines.append("Checkpoints (for ./3_run_autonomous.sh <name>):")
        for c in d["checkpoints"]:
            v = f"val {c['val']:.4f}" if c["val"] is not None else ""
            lines.append(f"   {c['name']}  epoch {c['epoch']:5.1f}  {v}{'   <- best val' if c.get('best') else ''}")
    lines.append("Last events: " + (d["watchdog"][-1] if d["watchdog"] else "-"))
    return "\n".join(lines)


def write_png() -> Path | None:
    d = collect()
    if not d["train"]:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=130)
    ax.plot([p["epoch"] for p in d["train"]], [p["loss"] for p in d["train"]], color="#9db7d5", lw=1, label="train loss (every log step)")
    ax.plot([e["epoch"] for e in d["epoch_train"]], [e["loss"] for e in d["epoch_train"]], "o-", color="#2a6fb0", ms=3, lw=2, label="train loss (epoch mean)")
    if d["val"]:
        ax.plot([v["epoch"] for v in d["val"]], [v["loss"] for v in d["val"]], "s-", color="#d9622b", ms=4, lw=2, label="validation L1 (held-out demos)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(f"MolmoAct2 - {d['task']}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = RUN / "loss_curve.png"
    fig.savefig(out)
    plt.close(fig)
    return out


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MolmoAct2 Training</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b2430;--mute:#647084;--line:#e3e7ee;--train:#2a6fb0;--train2:#a9c3e0;--val:#d9622b;--ok:#1f8a4c;--warn:#b7791f;--bad:#c53030}
@media (prefers-color-scheme:dark){:root{--bg:#11151b;--card:#1a2029;--ink:#e6eaf0;--mute:#93a0b3;--line:#2a323e;--train:#6aa6e8;--train2:#33506f;--val:#f08a52}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0}.sub{color:var(--mute);margin:2px 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.k{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}
.small{font-size:13px;color:var(--mute)}
.bar{height:12px;background:var(--line);border-radius:6px;overflow:hidden;margin:10px 0 4px}.bar>div{height:100%;background:var(--train);transition:width .5s}
.state{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:650;font-size:13px;color:#fff;background:var(--mute)}
.TRAINING,.STARTING{background:var(--train)}.FINISHED{background:var(--ok)}.RETRYING{background:var(--warn)}.FAILED{background:var(--bad)}
svg{width:100%;height:auto;display:block}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--mute)}
.sw{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:6px;border-radius:2px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line)}th{color:var(--mute);font-weight:500;font-size:12px}
pre{margin:0;white-space:pre-wrap;font:12px/1.4 ui-monospace,monospace;color:var(--mute)}
button{background:none;border:1px solid var(--line);color:var(--ink);border-radius:6px;padding:3px 10px;cursor:pointer}
</style></head><body><main>
<h1>MolmoAct2 training: <span id="task"></span></h1><div class="sub">Updates every 5 s. Run <code>./2_train_local.sh status</code> for the same numbers in the terminal.</div>
<div class="card"><span id="state" class="state">...</span>
<div class="bar"><div id="bar" style="width:0"></div></div><div class="small" id="prog"></div></div>
<div class="row" style="margin-bottom:14px">
<div class="card"><div class="k">Remaining</div><div class="v" id="eta">-</div><div class="small" id="fin"></div></div>
<div class="card"><div class="k">Train loss</div><div class="v" id="tl">-</div><div class="small" id="tl2"></div></div>
<div class="card"><div class="k">Validation loss</div><div class="v" id="vl">-</div><div class="small" id="vl2"></div></div>
<div class="card"><div class="k">Speed</div><div class="v" id="sp">-</div><div class="small" id="sp2"></div></div>
</div>
<div class="card"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
<b>Loss per epoch</b><button id="scale">log scale</button></div>
<div class="legend"><span><i class="sw" style="background:var(--train2)"></i>train (every log step)</span>
<span><i class="sw" style="background:var(--train)"></i>train, epoch mean</span>
<span><i class="sw" style="background:var(--val)"></i>validation L1 (held-out demos)</span></div>
<svg id="chart" viewBox="0 0 1000 380"></svg>
<div class="small">Train loss = ACT's objective (L1 + 10&times;KL). Validation = L1 action error on demos it never trained on, with the VAE off, as on the robot. Watch each curve's trend; their heights are not comparable.</div></div>
<div class="row"><div class="card"><b>GPUs</b><table id="gpus"></table></div>
<div class="card"><b>Checkpoints</b> <span class="small">(./3_run_autonomous.sh &lt;name&gt;)</span><table id="ck"></table></div></div>
<div class="card"><b>Events</b><pre id="ev"></pre></div>
</main><script>
let logy=true;const $=id=>document.getElementById(id);
$('scale').onclick=()=>{logy=!logy;$('scale').textContent=logy?'log scale':'linear scale';draw(last)};
const hms=s=>{if(s==null)return'-';s=Math.round(s);return s>=3600?`${Math.floor(s/3600)}h ${String(Math.floor(s%3600/60)).padStart(2,'0')}m`:`${Math.floor(s/60)}m ${String(s%60).padStart(2,'0')}s`};
let last=null;
function draw(d){if(!d||!d.plan)return;const W=1000,H=380,L=60,R=15,T=12,B=40,svg=$('chart');
 const all=[...d.train.map(p=>p.loss),...d.val.map(p=>p.loss)].filter(x=>x>0);if(!all.length){svg.innerHTML='<text x="500" y="190" text-anchor="middle" fill="currentColor" opacity=".5">waiting for the first loss values...</text>';return}
 const xmax=d.plan.epochs,f=logy?Math.log10:(x=>x);let lo=f(Math.min(...all)),hi=f(Math.max(...all));if(hi-lo<1e-6){hi+=.5;lo-=.5}const pad=(hi-lo)*.06;lo-=pad;hi+=pad;
 const X=e=>L+(W-L-R)*e/xmax,Y=v=>T+(H-T-B)*(1-(f(v)-lo)/(hi-lo));let s='';
 const cs=getComputedStyle(document.documentElement),c=n=>cs.getPropertyValue(n).trim();
 for(let i=0;i<=5;i++){const yv=lo+(hi-lo)*i/5,y=T+(H-T-B)*(1-i/5),lab=logy?Math.pow(10,yv):yv;s+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="${c('--line')}"/><text x="${L-6}" y="${y+4}" text-anchor="end" font-size="11" fill="${c('--mute')}">${lab<.01?lab.toExponential(1):lab.toFixed(3)}</text>`}
 const step=xmax>40?10:xmax>15?5:xmax>6?2:1;for(let e=0;e<=xmax;e+=step){const x=X(e);s+=`<line x1="${x}" x2="${x}" y1="${T}" y2="${H-B}" stroke="${c('--line')}"/><text x="${x}" y="${H-B+16}" text-anchor="middle" font-size="11" fill="${c('--mute')}">${e}</text>`}
 s+=`<text x="${(W+L)/2}" y="${H-6}" text-anchor="middle" font-size="12" fill="${c('--mute')}">epoch</text>`;
 const path=(pts,col,w)=>pts.length?`<polyline fill="none" stroke="${col}" stroke-width="${w}" points="${pts.map(p=>X(p.epoch)+','+Y(p.loss)).join(' ')}"/>`:'';
 s+=path(d.train,c('--train2'),1.2)+path(d.epoch_train,c('--train'),2.5)+path(d.val,c('--val'),2.5);
 d.epoch_train.forEach(p=>s+=`<circle cx="${X(p.epoch)}" cy="${Y(p.loss)}" r="3" fill="${c('--train')}"><title>epoch ${p.epoch}: train ${p.loss.toFixed(4)}</title></circle>`);
 d.val.forEach(p=>s+=`<rect x="${X(p.epoch)-3.5}" y="${Y(p.loss)-3.5}" width="7" height="7" fill="${c('--val')}"><title>epoch ${p.epoch.toFixed(1)}: val ${p.loss.toFixed(4)}</title></rect>`);
 svg.innerHTML=s}
async function tick(){try{const d=await (await fetch('/api/status')).json();last=d;$('task').textContent=d.task||'';
 $('state').textContent=d.state;$('state').className='state '+d.state;
 if(d.plan){const p=d.plan,pct=100*d.step/p.total_steps;$('bar').style.width=pct+'%';
  $('prog').textContent=`${pct.toFixed(1)}%  -  epoch ${d.epoch.toFixed(2)} of ${p.epochs}  -  step ${d.step.toLocaleString()} of ${p.total_steps.toLocaleString()}  -  ${p.train_episodes} train / ${p.val_episodes} validation demos  -  batch ${p.batch_size_per_gpu} x ${p.num_gpus} GPUs`;
  const running=['TRAINING','STARTING','RETRYING'].includes(d.state);
  $('eta').textContent=running?hms(d.eta_s):(d.state==='FINISHED'?'done':'-');
  $('fin').textContent=running&&d.eta_s?'finishes ~'+new Date((d.now+d.eta_s)*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}):(d.started_at?'started '+new Date(d.started_at*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}):'');
  if(d.train.length){$('tl').textContent=d.train[d.train.length-1].loss.toFixed(4);const et=d.epoch_train;$('tl2').textContent=et.length?`epoch ${et[et.length-1].epoch} mean ${et[et.length-1].loss.toFixed(4)}`:''}
  if(d.val.length){$('vl').textContent=d.val[d.val.length-1].loss.toFixed(4);const b=d.val.reduce((a,v)=>v.loss<a.loss?v:a);$('vl2').textContent=`best ${b.loss.toFixed(4)} at epoch ${b.epoch.toFixed(0)}`}
  $('sp').textContent=d.samples_per_s?Math.round(d.samples_per_s)+' /s':'-';$('sp2').textContent=d.samples_per_s?'samples per second, both GPUs':''}
 $('gpus').innerHTML='<tr><th>GPU</th><th>busy</th><th>memory</th><th>temp</th><th>power</th></tr>'+d.gpus.map(g=>`<tr><td>${g.index}</td><td>${g.util.toFixed(0)}%</td><td>${g.mem_used.toFixed(1)} / ${g.mem_total.toFixed(0)} GB</td><td>${g.temp.toFixed(0)} C</td><td>${g.power.toFixed(0)} W</td></tr>`).join('');
 $('ck').innerHTML=d.checkpoints.length?'<tr><th>name</th><th>epoch</th><th>val loss</th><th></th></tr>'+d.checkpoints.map(k=>`<tr><td>${k.name}</td><td>${k.epoch.toFixed(1)}</td><td>${k.val!=null?k.val.toFixed(4):''}</td><td>${k.best?'best val':''}</td></tr>`).join(''):'<tr><td class="small">none yet</td></tr>';
 $('ev').textContent=d.watchdog.join('\n');draw(d)}catch(e){$('state').textContent='dashboard cannot read the run'}}
tick();setInterval(tick,5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/status"):
            body, ctype = json.dumps(collect()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--serve", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--png", action="store_true")
    a = ap.parse_args()
    if a.status:
        print(text_status())
    elif a.png:
        p = write_png()
        print(p or "no loss values yet")
    else:
        cfg = env_config()
        host, port = cfg.get("DASHBOARD_HOST", "127.0.0.1"), int(cfg.get("DASHBOARD_PORT", "8787"))
        ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
