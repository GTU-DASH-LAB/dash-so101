#!/usr/bin/env python3
"""
Standalone monitor for the pi0.5 tshirt-fold LoRA training run.

Runs as its own detached process (launched via setsid from start_training.sh,
independent of any terminal or agent session). Every EMAIL_INTERVAL_S it
emails a loss-vs-epoch graph plus a remaining-time estimate, and sends one
final email when the run reaches a terminal state.

Progress/ETA come from lerobot's own stdout metric lines, which look like:

    step:1K smpl:19.2K ep:7.3 epch:0.16 loss:0.412 grdn:1.203 lr:1.0e-04 \
        updt_s:2.150 data_s:0.021

`step:` is abbreviated by format_big_number (1200 -> "1K"), so it is NOT used
for progress. `epch:` is an exact 2-decimal float and `updt_s:` is exact
seconds-per-optimizer-step, so those two drive the step count and the ETA
respectively. Checkpoint directories are used as an independent cross-check.
"""

import base64
import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_ENV = HERE / "config.env"
TRAIN_LOG = HERE / "train.log"
WATCHDOG_LOG = HERE / "watchdog.log"
METRICS_JSONL = HERE / "loss_history.jsonl"
GRAPH_PNG = HERE / "loss_curve.png"
MONITOR_LOG = HERE / "monitor.log"
T0_FILE = HERE / "monitor_t0"

SCRAPE_INTERVAL_S = 5 * 60
EMAIL_INTERVAL_S = 1 * 60 * 60
TAIL_BYTES = 400_000

RESEND_KEYFILE = Path.home() / ".config" / "resend" / "api_key"
RESEND_TO = os.environ.get("RESEND_TO", "fouadiadhami@gmail.com")
RESEND_FROM = os.environ.get("RESEND_FROM", "onboarding@resend.dev")

CKPT_DIR_RE = re.compile(r"^\d{6}$")
# A metric line must have at least epch+loss to be usable.
EPCH_RE = re.compile(r"\bepch:([0-9]*\.?[0-9]+)")
LOSS_RE = re.compile(r"\bloss:([0-9]*\.?[0-9]+)")
UPDT_RE = re.compile(r"\bupdt_s:([0-9]*\.?[0-9]+)")
GRDN_RE = re.compile(r"\bgrdn:([0-9]*\.?[0-9]+)")


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with open(MONITOR_LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def read_config_env() -> dict:
    cfg: dict[str, str] = {}
    for m in re.finditer(r"^([A-Z_]+)=(.*)$", CONFIG_ENV.read_text(), re.MULTILINE):
        key, val = m.group(1), m.group(2).strip()
        # strip trailing inline comment, then quotes
        val = re.sub(r"\s+#.*$", "", val).strip().strip('"').strip("'")

        def repl(mo, _cfg=cfg):
            name = mo.group(1)
            return _cfg.get(name, os.environ.get(name, mo.group(0)))

        cfg[key] = re.sub(r"\$\{([A-Za-z_]+)\}", repl, val)
    return cfg


def total_frames(cfg: dict) -> int | None:
    info = Path(cfg["DATASET_ROOT"]) / "meta" / "info.json"
    try:
        return int(json.loads(info.read_text())["total_frames"])
    except Exception:
        return None


def tail_text(path: Path, nbytes: int = TAIL_BYTES) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        return f.read().decode("utf-8", errors="ignore")


def parse_latest_metrics(text: str) -> dict | None:
    """Newest usable metric line, plus a mean updt_s over the tail window."""
    latest = None
    updts: list[float] = []
    for line in text.splitlines():
        epch, loss = EPCH_RE.search(line), LOSS_RE.search(line)
        if not (epch and loss):
            continue
        updt = UPDT_RE.search(line)
        grdn = GRDN_RE.search(line)
        if updt:
            updts.append(float(updt.group(1)))
        latest = {
            "epochs": float(epch.group(1)),
            "loss": float(loss.group(1)),
            "updt_s": float(updt.group(1)) if updt else None,
            "grad_norm": float(grdn.group(1)) if grdn else None,
        }
    if latest is None:
        return None
    # Average over the recent window smooths out per-step jitter; ignore the
    # first few (warmup steps include CUDA autotuning and are not representative).
    window = updts[-50:] if len(updts) > 5 else updts
    latest["updt_s_avg"] = sum(window) / len(window) if window else None
    return latest


def parse_all_metrics(text: str) -> list[dict]:
    """Every usable metric line, oldest first, de-duplicated on `epch`.

    The graph is built from this rather than from the 5-minute polling samples:
    lerobot emits a metric line every `log_freq` (200) steps, which at a few
    seconds/step is roughly every 7-13 minutes -- so polling every 5 minutes
    would record the same line two or three times and miss nothing in between.
    Parsing the log directly gives the exact curve at the logger's own
    resolution, independent of how often this monitor happens to wake up.
    Over a 20k-step run that is only ~100 lines, so re-reading is cheap.
    """
    out: list[dict] = []
    seen: set[float] = set()
    for line in text.splitlines():
        epch, loss = EPCH_RE.search(line), LOSS_RE.search(line)
        if not (epch and loss):
            continue
        e = float(epch.group(1))
        if e in seen:
            continue
        seen.add(e)
        out.append({"epochs": e, "loss": float(loss.group(1))})
    return out


def steps_from_epochs(epochs: float, cfg: dict, frames: int | None) -> int | None:
    """epochs = (steps * effective_batch) / total_frames  ->  invert it."""
    if not frames:
        return None
    try:
        eff = int(cfg["BATCH_SIZE"]) * int(cfg["NUM_GPUS"])
    except (KeyError, ValueError):
        return None
    return int(round(epochs * frames / eff))


def checkpoint_step(output_dir: Path) -> int | None:
    root = output_dir / "checkpoints"
    if not root.is_dir():
        return None
    steps = [int(c.name) for c in root.iterdir() if c.is_dir() and CKPT_DIR_RE.match(c.name)]
    return max(steps) if steps else None


def training_finished() -> str | None:
    text = tail_text(WATCHDOG_LOG, 100_000)
    if "Training finished successfully" in text:
        return "success"
    if "Reached MAX_RETRIES" in text:
        return "gave_up"
    return None


def load_history() -> list[dict]:
    if not METRICS_JSONL.exists():
        return []
    out = []
    for line in METRICS_JSONL.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def render_graph(history: list[dict], epochs_target: float, steps_target: int) -> bool:
    """Loss vs epoch. A secondary top axis carries the equivalent step number,
    since epochs and steps are just a fixed rescaling of each other here
    (epoch = step * effective_batch / total_frames) and the run's end condition
    is expressed in steps."""
    pts = [(h["epochs"], h["loss"]) for h in history
           if h.get("loss") is not None and h.get("epochs") is not None]
    if len(pts) < 2:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    ax.plot(xs, ys, linewidth=1.4, color="#2b6cb0", label="loss")
    if len(ys) >= 10:  # running mean to make the trend legible through the noise
        w = max(3, len(ys) // 20)
        sm = [sum(ys[max(0, i - w + 1):i + 1]) / len(ys[max(0, i - w + 1):i + 1])
              for i in range(len(ys))]
        ax.plot(xs, sm, linewidth=2.0, color="#dd6b20", label=f"running mean ({w})")
    ax.legend(loc="upper right")
    ax.set_xlim(0, epochs_target)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("pi0.5 t-shirt fold — LoRA fine-tune")
    ax.grid(True, alpha=0.3)

    if epochs_target > 0:
        top = ax.secondary_xaxis(
            "top",
            functions=(lambda e: e * steps_target / epochs_target,
                       lambda s: s * epochs_target / steps_target),
        )
        top.set_xlabel(f"Training step (of {steps_target:,})")

    fig.tight_layout()
    fig.savefig(GRAPH_PNG)
    plt.close(fig)
    return True


def send_email(subject: str, body: str, attach_png: bool) -> bool:
    if not RESEND_KEYFILE.exists():
        log(f"ERROR: no Resend API key at {RESEND_KEYFILE}")
        return False
    payload = {
        "from": RESEND_FROM,
        "to": [RESEND_TO],
        "subject": subject,
        "text": body,
    }
    if attach_png and GRAPH_PNG.exists():
        payload["attachments"] = [{
            "filename": "loss_curve.png",
            "content": base64.b64encode(GRAPH_PNG.read_bytes()).decode("ascii"),
        }]
    # Key goes in a header via stdin-fed config, never on the command line.
    proc = subprocess.run(
        ["curl", "-sS", "-w", "\n%{http_code}", "-X", "POST",
         "https://api.resend.com/emails",
         "-H", f"Authorization: Bearer {RESEND_KEYFILE.read_text().strip()}",
         "-H", "Content-Type: application/json",
         "--data-binary", "@-"],
        input=json.dumps(payload), capture_output=True, text=True,
    )
    out = proc.stdout.strip()
    code = out.rsplit("\n", 1)[-1] if "\n" in out else ""
    if code == "200":
        log(f"Email sent: {subject}")
        return True
    log(f"Email FAILED (http={code}): {out[:400]}")
    return False


def human_hours(h: float) -> str:
    if h < 1:
        return f"{h * 60:.0f} minutes"
    d, rem = divmod(h, 24)
    return f"{h:.1f} hours" + (f" (~{d:.0f}d {rem:.0f}h)" if d >= 1 else "")


def build_report(cfg: dict, m: dict | None, step: int | None,
                 ckpt: int | None, t0: float, epochs_target: float = 0.0) -> str:
    target = int(cfg["STEPS"])
    lines = []
    if step is not None:
        pct = 100.0 * step / target
        lines.append(f"Progress:  step {step:,} / {target:,}  ({pct:.1f}%)")
    else:
        lines.append(f"Progress:  no metric lines yet (target {target:,} steps)")
    if m:
        lines.append(f"Loss:      {m['loss']:.4f}" +
                     (f"   grad-norm {m['grad_norm']:.3f}" if m.get("grad_norm") else ""))
        lines.append(f"Epochs:    {m['epochs']:.2f}"
                     + (f" / {epochs_target:.2f}" if epochs_target else ""))
    if m and m.get("updt_s_avg") and step is not None:
        rate = m["updt_s_avg"]
        remaining_h = (target - step) * rate / 3600.0
        finish = time.time() + remaining_h * 3600
        lines.append(f"Speed:     {rate:.2f} s/step  ({3600 / rate:.0f} steps/hour)")
        lines.append(f"Remaining: {human_hours(remaining_h)}")
        lines.append(f"ETA:       {time.strftime('%a %d %b, %H:%M', time.localtime(finish))}")
    lines.append(f"Elapsed:   {human_hours((time.time() - t0) / 3600.0)}")
    lines.append(f"Checkpoint: {'step ' + format(ckpt, ',') if ckpt else 'none written yet'}"
                 f"  (every {int(cfg['SAVE_FREQ']):,} steps)")
    lines.append("")
    lines.append(f"GPUs: {cfg['NUM_GPUS']}x RTX 5090, per-GPU batch {cfg['BATCH_SIZE']} "
                 f"(effective {int(cfg['BATCH_SIZE']) * int(cfg['NUM_GPUS'])})")
    lines.append(f"Output: {HERE / cfg['OUTPUT_DIR']}")
    return "\n".join(lines)


def main() -> None:
    cfg = read_config_env()
    od = cfg["OUTPUT_DIR"]
    output_dir = Path(od) if od.startswith("/") else (HERE / od).resolve()
    frames = total_frames(cfg)
    target = int(cfg["STEPS"])
    eff_batch = int(cfg["BATCH_SIZE"]) * int(cfg["NUM_GPUS"])
    epochs_target = (target * eff_batch / frames) if frames else 0.0

    # Stable t0 across monitor restarts.
    if T0_FILE.exists():
        t0 = float(T0_FILE.read_text().strip())
    else:
        t0 = time.time()
        T0_FILE.write_text(str(t0))

    log(f"Monitor up. target={target} frames={frames} output={output_dir} "
        f"email every {EMAIL_INTERVAL_S // 3600}h to {RESEND_TO}")

    # last_email = 0 makes the first loop iteration fire immediately: that
    # startup email is deliberate, so the Resend path is proven working within
    # minutes of launch instead of failing silently and only being discovered
    # six hours later when the first real report never arrives.
    last_email = 0.0
    first = True
    while True:
        full_log = tail_text(TRAIN_LOG, 50_000_000)
        m = parse_latest_metrics(full_log)
        step = steps_from_epochs(m["epochs"], cfg, frames) if m else None
        ckpt = checkpoint_step(output_dir)
        if step is None and ckpt is not None:
            step = ckpt

        # Kept as a durable record of the run; the graph is drawn from the log.
        if m:
            with open(METRICS_JSONL, "a") as f:
                f.write(json.dumps({"t": time.time(), "step": step,
                                    "loss": m["loss"], "epochs": m["epochs"]}) + "\n")

        finished = training_finished()
        if finished or (time.time() - last_email) >= EMAIL_INTERVAL_S:
            has_graph = render_graph(parse_all_metrics(full_log), epochs_target, target)
            report = build_report(cfg, m, step, ckpt, t0, epochs_target)

            # "STARTED" only when there is genuinely nothing to report yet --
            # otherwise a monitor restart mid-run would mislabel a progress
            # email as a fresh start.
            if first and not finished and step is None:
                subject = "pi0.5 t-shirt fold — training STARTED"
                body = (f"Training has started on the workstation. You'll get an update "
                        f"every hour, and one final email when "
                        f"it finishes.\n\n{report}\n")
            elif finished == "success":
                subject = "pi0.5 t-shirt fold — training COMPLETE"
                body = f"Training finished successfully.\n\n{report}\n"
            elif finished == "gave_up":
                subject = "pi0.5 t-shirt fold — training STOPPED (watchdog gave up)"
                body = (f"The watchdog exhausted its retries and stopped.\n\n{report}\n\n"
                        f"Check {WATCHDOG_LOG} on the workstation.\n")
            else:
                subject = "pi0.5 t-shirt fold — training progress"
                body = f"{report}\n"

            send_email(subject, body, attach_png=has_graph)
            last_email = time.time()
            first = False

            if finished:
                log(f"Terminal state ({finished}); monitor exiting.")
                return

        time.sleep(SCRAPE_INTERVAL_S)


if __name__ == "__main__":
    main()
