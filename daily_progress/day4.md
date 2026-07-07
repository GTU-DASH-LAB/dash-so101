# Day 4 — Record Ball-Pickup Demos & Fine-Tune pi0 on the GX10

**Date:** 2026-07-02

---

## English

### Overview

New task, new pipeline: **`ball_pickup_pi0`** — *"Pick up the tennis ball and place it in the white
cup."* Unlike the pencil-pickup task (Days 2–3), which needed a Kaggle GPU notebook, this one is
fine-tuned on a much bigger model, **pi0** (~4B parameters), and runs **entirely on this machine** —
an Nvidia **GX10 with 121GB of unified CPU+GPU memory** — record, train, and (eventually) run, no
cloud round-trip.

Today covered:
1. Recording **15 teleoperated demonstrations** of ball → cup with both cameras.
2. Two real blockers before training would even start: a **camera name mismatch** and an **OOM** at
   the first batch size we tried.
3. Adding a **crash watchdog** before committing to an unattended overnight run.
4. Kicking off the real fine-tune: **10,000 steps**, which took **~11.5 hours**, finishing overnight.

### Part 1 — A new self-contained pipeline

`ball_pickup_pi0/` follows the same numbered-script convention as the other tasks
(`find_camera.sh` → `1_record.sh` → `2_train_local.sh` → `3_run_autonomous.sh`), with every setting
centralized in `config.env` so record/train/run always agree on cameras, task string, and ports.

Base model: `lerobot/pi0_base`, **full fine-tune** (no LoRA yet — that comes later, in a different
task folder). Task string, used byte-for-byte identical at record and inference time:

```text
TASK="Pick up the tennis ball and place it in the white cup"
```

### Part 2 — Record 15 demonstrations

Teleop needs **both** arms connected (leader mirrors into the follower). First snag of the day: the
leader/follower ports had swapped since the last session — `/dev/ttyACM0` / `/dev/ttyACM1` aren't
fixed identities, just USB enumeration order, so a replug (or reboot) can flip which one is which.
Re-checked with `lerobot-find-port` and fixed `FOLLOWER_PORT`/`LEADER_PORT` in `config.env` before
recording.

Cameras — the same two used across this project, kept **identical** between record and run:

```text
top   = /dev/video0   (overhead workspace view)
wrist = /dev/video2   (gripper-mounted, rotated 180 + mirrored to match reality)
```

Recording settings, bumped up a bit from earlier tasks since ball-pickup is a slightly longer motion
than a flat pick-and-place:

```text
NUM_EPISODES=15
EPISODE_TIME_S=40      # seconds per demonstration
RESET_TIME_S=10        # pause to reset the scene between demos
FPS=30
```

```bash
./1_record.sh
# = lerobot-record --robot.type=so101_follower --robot.cameras="$CAMERAS" \
#     --teleop.type=so101_leader --dataset.repo_id="$DATASET_REPO" \
#     --dataset.single_task="$TASK" --dataset.num_episodes=15 \
#     --dataset.episode_time_s=40 --dataset.reset_time_s=10 --dataset.push_to_hub=true
```

Dataset saved locally under `~/.cache/huggingface/lerobot/fouad1233/so101_ball_pickup`, then pushed
to <https://huggingface.co/datasets/fouad1233/so101_ball_pickup>.

### Part 3 — Blocker 1: camera feature mismatch

First attempt at `./2_train_local.sh` failed immediately with a `Feature mismatch` error.
`lerobot/pi0_base`'s declared `input_features` expect
`observation.images.{base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb}` — pi0's own naming — but our
dataset's cameras are named `top`/`wrist`. `make_policy`'s strict feature check only runs when no
rename map is given.

Fix — remap the dataset's columns to pi0's names at training time:

```bash
--rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
```

The missing third slot (`right_wrist_0_rgb`) is fine — pi0 masks out declared-but-absent image keys
internally, it just doesn't get a third camera.

### Part 4 — Blocker 2: OOM at the first batch size

With the feature mismatch fixed, training crashed again — this time from memory. fp32 (the
`lerobot-train` default) with `BATCH_SIZE=16` reliably OOMs on this GX10, using **113GB+** before
crashing (out of 121GB unified memory — note `nvidia-smi` reports memory as `N/A` on this chip since
it's not a discrete-VRAM GPU; `free -h` is what actually shows usage here).

`--policy.use_amp` looked like the obvious fix but turned out to be a **no-op** in this lerobot
version — it's validated on the CLI but never actually passed through to `Accelerate`'s
`mixed_precision` setting. The real switch is an environment variable that `Accelerator()` reads
directly:

```bash
export ACCELERATE_MIXED_PRECISION=bf16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduces fragmentation-related OOM risk
```

With that plus dropping to `BATCH_SIZE=8`, training reached a steady state of **~78GB, ~4.05s/step
(~2 samples/sec)** — though the first couple of warmup steps (CUDA kernel autotuning) spike memory
hard enough to briefly dip into swap (~10GB observed) before settling. Both env vars are now wired
permanently into `2_train_local.sh`.

### Part 5 — Add a crash watchdog before the real run

Before committing to an unattended ~11h run, added `watchdog_train.sh`. The reason: a crashed
`lerobot-train` can leave **orphaned dataloader worker processes** holding tens of GB, which would
make an immediate retry OOM again even with the now-correct settings (this exact failure mode had
already cost a machine-wide slowdown once — 114GB used / 10GB swapped — while iterating on the
settings above).

`watchdog_train.sh` can either attach to an already-running `lerobot-train` PID or start fresh; on any
non-zero exit before reaching `STEPS` it kills leftover workers, waits for memory to settle, and
resumes from the last checkpoint (up to 10 retries), logging every attempt to `watchdog.log`:

```bash
./watchdog_train.sh              # start (or resume) a run under the watchdog
```

### Part 6 — The real run: 10,000 steps overnight

Final settings: `STEPS=10000`, `SAVE_FREQ=1000` (checkpoint roughly every hour, deliberately frequent
given how tight the memory margin is).

Kicked off in the afternoon; checkpoints landed on schedule right up to the last one:

| Checkpoint | Timestamp |
|---|---|
| 001000 | 2026-07-02 15:14 |
| 002000 | 2026-07-02 16:26 |
| 005000 | 2026-07-02 19:52 |
| 008000 | 2026-07-02 23:18 |
| 010000 (last) | 2026-07-03 ~01:35 |

**Total: about 11.5 hours** for 10,000 steps at ~2 samples/sec — right in line with the estimate.
Model pushed to <https://huggingface.co/fouad1233/pi0_ball_pickup> once training finished
(`PUSH_MODEL_TO_HUB=true`).

### Gotchas summary (Day 4)

| # | Error | Fix |
|---|-------|-----|
| 1 | Leader/follower ports swapped after replug | re-check with `lerobot-find-port`, fix `config.env` |
| 2 | `Feature mismatch` (top/wrist vs base_0_rgb/left_wrist_0_rgb) | `--rename_map` at train time |
| 3 | fp32 + `batch_size=16` OOM (113GB+) | `ACCELERATE_MIXED_PRECISION=bf16` + `batch_size=8` (~78GB) |
| 4 | Crashed run leaves orphaned workers, retry OOMs again | `watchdog_train.sh`: kill orphans, wait, resume |

### Next — Day 5

- Fix `3_run_autonomous.sh` to use the identical `--rename_map` (a checkpoint's declared feature names
  don't change even though `rename_map` only touched the dataset at train time).
- Run the trained policy autonomously and see how it actually performs picking up the ball and placing
  it in the cup.

---

## Türkçe

### Genel Bakış

Yeni görev, yeni hat: **`ball_pickup_pi0`** — *"Tenis topunu al ve beyaz bardağın içine koy."*
Kalem-alma görevinden (2–3. Gün) farklı olarak — o bir Kaggle GPU not defteri gerektirmişti — bu görev
çok daha büyük bir modelde, **pi0**'da (~4B parametre) fine-tune ediliyor ve **tamamen bu makinede**
çalışıyor: 121GB birleşik CPU+GPU belleğe sahip bir Nvidia **GX10** — kayıt, eğitim ve (sonunda)
çalıştırma, bulutla gidip gelme yok.

Bugün şunlar oldu:
1. Her iki kamerayla top → bardak görevinin **15 teleoperasyon gösterimini** kaydetmek.
2. Eğitim başlamadan önce iki gerçek engel: bir **kamera adı uyuşmazlığı** ve denediğimiz ilk batch
   boyutunda bir **OOM**.
3. Gece boyu insansız bir çalışmaya karar vermeden önce bir **çökme gözcüsü (watchdog)** eklemek.
4. Asıl fine-tune'u başlatmak: **10.000 adım**, bu da **~11,5 saat** sürdü ve gece bitti.

### Bölüm 1 — Yeni, kendi kendine yeten bir hat

`ball_pickup_pi0/`, diğer görevlerle aynı numaralandırılmış betik kuralını izliyor
(`find_camera.sh` → `1_record.sh` → `2_train_local.sh` → `3_run_autonomous.sh`); tüm ayarlar
`config.env`'de merkezileştirilmiş, böylece kayıt/eğitim/çalıştırma kameralar, görev metni ve
portlar konusunda her zaman uyumlu.

Temel model: `lerobot/pi0_base`, **tam fine-tune** (henüz LoRA yok — o daha sonra, başka bir görev
klasöründe geliyor). Görev metni, kayıt ve çıkarım anında harfi harfine aynı kullanılıyor:

```text
TASK="Pick up the tennis ball and place it in the white cup"
```

### Bölüm 2 — 15 gösterim kaydet

Teleoperasyon **her iki** kolun da bağlı olmasını gerektirir (lider takipçiye yansır). Günün ilk
sürprizi: lider/takipçi portları son oturumdan beri yer değiştirmişti — `/dev/ttyACM0` / `/dev/ttyACM1`
sabit kimlikler değil, sadece USB numaralandırma sırası; bir fişi çekip takmak (veya yeniden başlatma)
hangisinin hangisi olduğunu değiştirebiliyor. `lerobot-find-port` ile yeniden kontrol edip
kayıttan önce `config.env`'deki `FOLLOWER_PORT`/`LEADER_PORT`'u düzelttik.

Kameralar — bu projede kullanılan aynı ikili, kayıt ve çalıştırma arasında **aynı** tutuluyor:

```text
top   = /dev/video0   (üstten çalışma alanı görünümü)
wrist = /dev/video2   (tutucuya monteli, gerçekliğe uysun diye 180 döndürülmüş + aynalanmış)
```

Kayıt ayarları, top-alma düz bir al-götürden biraz daha uzun bir hareket olduğu için önceki
görevlere göre biraz artırıldı:

```text
NUM_EPISODES=15
EPISODE_TIME_S=40      # gösterim başına saniye
RESET_TIME_S=10        # gösterimler arası sahneyi sıfırlama molası
FPS=30
```

```bash
./1_record.sh
# = lerobot-record --robot.type=so101_follower --robot.cameras="$CAMERAS" \
#     --teleop.type=so101_leader --dataset.repo_id="$DATASET_REPO" \
#     --dataset.single_task="$TASK" --dataset.num_episodes=15 \
#     --dataset.episode_time_s=40 --dataset.reset_time_s=10 --dataset.push_to_hub=true
```

Veri seti yerelde `~/.cache/huggingface/lerobot/fouad1233/so101_ball_pickup` altında kaydedildi, sonra
<https://huggingface.co/datasets/fouad1233/so101_ball_pickup>'a gönderildi.

### Bölüm 3 — Engel 1: kamera özellik uyuşmazlığı

`./2_train_local.sh`'ın ilk denemesi hemen bir `Feature mismatch` hatasıyla başarısız oldu.
`lerobot/pi0_base`'in bildirdiği `input_features`, pi0'ın kendi adlandırmasıyla
`observation.images.{base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb}` bekliyor — ama bizim veri
setimizin kameraları `top`/`wrist` adında. `make_policy`'nin sıkı özellik kontrolü yalnızca yeniden
adlandırma haritası verilmediğinde çalışıyor.

Çözüm — veri setinin sütunlarını eğitim anında pi0'ın adlarına yeniden eşle:

```bash
--rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
```

Eksik üçüncü slot (`right_wrist_0_rgb`) sorun değil — pi0, bildirilmiş ama mevcut olmayan görüntü
anahtarlarını içeride otomatik maskeliyor, sadece üçüncü bir kamera almıyor.

### Bölüm 4 — Engel 2: ilk batch boyutunda OOM

Özellik uyuşmazlığı düzeltildikten sonra eğitim yine çöktü — bu sefer bellekten. fp32
(`lerobot-train`'in varsayılanı) ile `BATCH_SIZE=16`, bu GX10'da güvenilir şekilde OOM veriyor,
çökmeden önce **113GB+** kullanıyor (121GB birleşik bellekten — not: bu çip ayrık VRAM'li bir GPU
olmadığı için `nvidia-smi` belleği `N/A` olarak raporluyor; burada gerçek kullanımı gösteren
`free -h`).

`--policy.use_amp` bariz çözüm gibi görünüyordu ama bu lerobot sürümünde **etkisiz** (no-op) çıktı —
CLI'de doğrulanıyor ama gerçekte `Accelerate`'in `mixed_precision` ayarına hiç geçirilmiyor. Gerçek
anahtar, `Accelerator()`'ın doğrudan okuduğu bir ortam değişkeni:

```bash
export ACCELERATE_MIXED_PRECISION=bf16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # parçalanma kaynaklı OOM riskini azaltır
```

Bununla birlikte `BATCH_SIZE=8`'e düşürünce eğitim **~78GB, ~4,05sn/adım (~2 örnek/sn)** kararlı
duruma ulaştı — yine de ilk birkaç ısınma adımı (CUDA kernel autotuning) belleği yerleşmeden önce
kısaca swap'a (~10GB gözlendi) düşecek kadar sıçratıyor. Her iki ortam değişkeni de artık
`2_train_local.sh`'a kalıcı olarak gömülü.

### Bölüm 5 — Asıl çalışmadan önce bir çökme gözcüsü ekle

İnsansız ~11 saatlik bir çalışmaya karar vermeden önce `watchdog_train.sh` eklendi. Sebep: çökmüş bir
`lerobot-train`, onlarca GB tutan **yetim dataloader işçi süreçleri** bırakabiliyor; bu da doğru
ayarlarla bile ani bir yeniden denemenin tekrar OOM vermesine yol açardı (bu tam hata modu, yukarıdaki
ayarlar üzerinde çalışırken bir kez zaten makine genelinde bir yavaşlamaya mal olmuştu — 114GB
kullanım / 10GB swap).

`watchdog_train.sh` ya çalışmakta olan bir `lerobot-train` PID'ine bağlanabilir ya da sıfırdan
başlayabilir; `STEPS`'e ulaşmadan herhangi bir sıfır-olmayan çıkışta kalan işçileri öldürür, bellek
yerleşene kadar bekler ve son checkpoint'ten devam eder (10 denemeye kadar), her denemeyi
`watchdog.log`'a kaydeder:

```bash
./watchdog_train.sh              # gözcü altında bir çalışma başlat (veya devam ettir)
```

### Bölüm 6 — Asıl çalışma: gece boyu 10.000 adım

Son ayarlar: `STEPS=10000`, `SAVE_FREQ=1000` (belleğin ne kadar dar olduğu düşünülürse kasıtlı sık,
yaklaşık her saat bir checkpoint).

Öğleden sonra başlatıldı; checkpoint'ler sonuncusuna kadar zamanında geldi:

| Checkpoint | Zaman damgası |
|---|---|
| 001000 | 2026-07-02 15:14 |
| 002000 | 2026-07-02 16:26 |
| 005000 | 2026-07-02 19:52 |
| 008000 | 2026-07-02 23:18 |
| 010000 (son) | 2026-07-03 ~01:35 |

**Toplam: ~11,5 saat**, 10.000 adım, ~2 örnek/sn — tahminle tam uyumlu. Eğitim bitince model
<https://huggingface.co/fouad1233/pi0_ball_pickup>'a gönderildi (`PUSH_MODEL_TO_HUB=true`).

### Dikkat özeti (4. Gün)

| # | Hata | Çözüm |
|---|------|-------|
| 1 | Fişi çekip takınca lider/takipçi portları yer değiştirdi | `lerobot-find-port` ile yeniden kontrol et, `config.env`'i düzelt |
| 2 | `Feature mismatch` (top/wrist vs base_0_rgb/left_wrist_0_rgb) | eğitim anında `--rename_map` |
| 3 | fp32 + `batch_size=16` OOM (113GB+) | `ACCELERATE_MIXED_PRECISION=bf16` + `batch_size=8` (~78GB) |
| 4 | Çöken çalışma yetim işçi bırakıyor, yeniden deneme tekrar OOM veriyor | `watchdog_train.sh`: yetimleri öldür, bekle, devam et |

### Sıradaki — 5. Gün

- `3_run_autonomous.sh`'ı aynı `--rename_map` ile kullanacak şekilde düzelt (bir checkpoint'in
  bildirdiği özellik adları, `rename_map` yalnızca eğitim anında veri setine dokunmuş olsa bile
  değişmiyor).
- Eğitilen politikayı otonom çalıştır ve topu alıp bardağa koymada gerçekte ne kadar iyi olduğunu gör.
