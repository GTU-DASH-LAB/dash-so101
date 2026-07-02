# Day 3 — Run the Fine-Tuned SmolVLA Policy Autonomously

**Date:** 2026-07-01

---

## English

### Overview

Day 2 ended with the Kaggle notebook fine-tuning **SmolVLA** on the 50-episode pencil-pickup dataset
and pushing the result to `fouad1233/smolvla_pencil_pickup` on the Hugging Face Hub. Today was about
closing the loop: pull that model down and let it drive the **follower** arm by itself, no leader, no
human in the loop — `3_run_autonomous.sh` (Stage 4 of the pipeline).

It took three fixes before the arm actually moved on its own, and one more to make the control loop
stable. Each is a real gotcha worth keeping:

1. **System disk full** → `lerobot-rollout` couldn't download the ~2GB base VLM weights.
2. **Camera name mismatch** → the policy was trained expecting `camera1`, the robot's live camera is
   named `front`.
3. **Control loop way under target FPS** → SmolVLA inference on Apple Silicon (MPS) is too slow to run
   inline at 30Hz.

After all three, the arm **picks up the pen and places it on the mouse pad on its own**, driven purely
by the fine-tuned policy reading the camera + joint state.

### Part 1 — Recap: how the model was trained (Day 2)

Quick reminder of the training path, since today's fixes only make sense in that context:

- 50 teleoperated demos → dataset `fouad1233/so101_pencil_pickup` (43,211 frames, 1 camera named `front`).
- Fine-tuned on a **Kaggle GPU notebook** (`2_kaggle_finetune_smolvla.ipynb`) using both T4 GPUs via
  `accelerate launch --multi_gpu`, 20,000 steps, effective batch size 16.
- `smolvla_base` expects **3 cameras** (`camera1/2/3`); our dataset only has one (`front`). Fixed at
  train time with:
  ```text
  --rename_map={"observation.images.front": "observation.images.camera1"}
  ```
  With `empty_cameras=0` (default) the model just uses the one camera we give it and ignores the other
  two slots.
- Final checkpoint pushed to `fouad1233/smolvla_pencil_pickup`.

That `rename_map` detail from training is exactly what came back to bite Stage 4 today (Part 3 below).

### Part 2 — Blocker 1: disk full downloading the base VLM

First run of `./3_run_autonomous.sh` crashed **while downloading model weights**, not while running:

```text
model.safetensors:  23%|##       | 463M/2.03G [00:30<01:43, 15.2MB/s]
RuntimeError: Task error: File reconstruction error: IO Error: No space left on device (os error 28)
OSError: Can't load the model for 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'.
```

`lerobot-rollout` needs to pull down the **base VLM** (`SmolVLM2-500M-Video-Instruct`, ~2GB) that
SmolVLA is built on, before it can load our fine-tuned weights on top. The whole disk
(`/System/Volumes/Data`) was down to **976MB free** out of 228GB — same class of `ENOSPC` problem as
Day 2, just system-wide this time instead of just the repo.

Freed space by clearing **regenerable caches only** (nothing that holds real data):

```bash
rm -rf ~/.cache/huggingface/hub/*                                   # redownloadable HF models — 397M
rm -rf ~/Library/Application\ Support/Code/CachedData                # VS Code build cache — 102M
rm -rf ~/Library/Application\ Support/Code/CachedExtensionVSIXs      # VS Code ext installer cache — 455M
rm -rf ~/Library/Developer/Xcode/DerivedData                         # Xcode build cache — 156M
npm cache clean --force                                              # ~740M garbage-collectable
```

That took free space from 976MB → 3.4GB, enough to finish the 2GB download. Left alone: two much
bigger items (a 7GB+ VS Code app-support folder and a 7GB+ WhatsApp `Message` folder) because those
hold real data, not caches.

### Part 3 — Blocker 2: camera name mismatch

With disk space fixed, the model loaded but refused to connect to the robot:

```text
ValueError: Visual feature mismatch between policy and robot hardware.
Policy expects: {'observation.images.camera2', 'observation.images.camera3', 'observation.images.camera1'}
Robot provides: {'observation.images.front'}
```

This is the direct consequence of the Day-2 training-time rename: the policy's weights are keyed to
`camera1`, but the live robot camera is still named `front` (whatever we pass in
`--robot.cameras="{ front: {...} }"`). `lerobot-rollout` does a strict name-match check between policy
and hardware — and only **skips that check when `--rename_map` is passed**, applying the same rename
at inference time as we used at training time.

Fix — pass the identical `rename_map` used in the training notebook:

```bash
--rename_map='{"observation.images.front": "observation.images.camera1"}'
```

Added permanently to `3_run_autonomous.sh`. Missing `camera2`/`camera3` are fine — same as training,
the model just ignores the two camera slots it doesn't get data for.

### Part 4 — Blocker 3: control loop can't keep up with 30Hz

Robot connected and the policy started driving it — but spammed this warning the whole time:

```text
WARNING  Record loop is running slower (0.6 Hz) than the target FPS (30.0 Hz). Dataset frames might be
dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up
2) Policy inference taking too long 3) CPU starvation
```

Root cause: by default `lerobot-rollout` uses `--inference.type=sync` — the **synchronous** inference
engine, which calls the policy **inline, once per control tick**, and the control loop blocks until
that call returns. SmolVLA is a ~450M-parameter VLM + action-expert; a forward pass on Apple Silicon
MPS takes far longer than the 33ms budget a 30Hz loop allows, so almost every tick missed its deadline.
Not a bug — the sync engine is just the wrong tool for a slow model on this hardware.

**Fix — `--inference.type=rtc`** (Real-Time Chunking): an *asynchronous* inference backend that runs
the policy in a **background thread**, decoupled from the control loop:

- The background thread keeps calling the policy and pushing predicted action **chunks** onto a queue.
- The control loop, running at full target FPS, just pops the next queued action each tick and sends
  it to the robot — no waiting on the model.
- If the queue runs low (inference falling behind), RTC blends the newly-predicted chunk with the
  tail of the one still executing, so the robot doesn't jerk or stall waiting for a fresh chunk —
  that's the "chunking" part: it works in overlapping chunks of actions instead of one action at a time.

In short: sync = "compute one action, then move," RTC = "keep a pipeline of upcoming actions full while
computing the next batch in the background." That's exactly what a slow-inference / high-control-rate
mismatch needs.

Also parameterized the device instead of hardcoding `mps`, since this repo will eventually run on a
CUDA Linux box too — added `POLICY_DEVICE` to `config.env` / `config.env.example` (`mps` on this Mac,
`cuda` on an Nvidia machine, `cpu` as a fallback), and `3_run_autonomous.sh` now reads
`--policy.device="$POLICY_DEVICE"`.

### Part 5 — Running it for real

```bash
cd pencil_pickup_vla
./3_run_autonomous.sh
```

Under the hood, the final command:

```bash
lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
  --robot.cameras="{ front: {type: opencv, index_or_path: $CAMERA_INDEX, width: $CAMERA_W, height: $CAMERA_H, fps: $CAMERA_FPS}}" \
  --task="$TASK" \
  --policy.path="$MODEL_REPO" \
  --policy.device="$POLICY_DEVICE" \
  --inference.type=rtc \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --display_data=true
```

Result: the follower **autonomously picked up the pen and placed it on the black mouse pad**, driven
only by the camera feed + fine-tuned SmolVLA policy. No FPS warnings, stable control.

### Gotchas summary (Day 3)

| # | Error | Fix |
|---|-------|-----|
| 1 | `No space left on device` downloading base VLM | clear regenerable caches (HF hub, VS Code, Xcode, npm) system-wide |
| 2 | `Visual feature mismatch` (camera1/2/3 vs front) | `--rename_map` at inference, same as training |
| 3 | Control loop stuck at <1Hz vs 30Hz target | `--inference.type=rtc` (async background inference) |

### Next — Day 4

- Run several more autonomous trials and count the actual success rate.
- If it's inconsistent, record more/varied demos (different pen positions, lighting) rather than
  training longer on the same 50 episodes.
- Try running the same policy on the CUDA Linux machine (`POLICY_DEVICE=cuda`) and compare inference
  speed / RTC queue behavior against the Mac.

---

## Türkçe

### Genel Bakış

2. Gün, **SmolVLA**'nın 50 bölümlük kalem-alma veri setinde Kaggle not defterinde fine-tune edilmesi
ve sonucun Hugging Face Hub'da `fouad1233/smolvla_pencil_pickup` olarak yayınlanmasıyla bitmişti.
Bugünün konusu döngüyü kapatmaktı: o modeli indirip **takipçi** kolu kendi başına, lidersiz ve
insan müdahalesi olmadan sürdürmesini sağlamak — `3_run_autonomous.sh` (hattın 4. Aşaması).

Kol kendi başına hareket etmeden önce üç düzeltme, kontrol döngüsünü stabil hale getirmek için de bir
düzeltme daha gerekti. Her biri not almaya değer gerçek bir tuzak:

1. **Sistem diski dolu** → `lerobot-rollout` ~2GB'lık temel VLM ağırlıklarını indiremedi.
2. **Kamera adı uyuşmazlığı** → model `camera1` bekleyecek şekilde eğitilmişti, robotun canlı kamerası
   `front` adında.
3. **Kontrol döngüsü hedef FPS'in çok altında** → Apple Silicon'da (MPS) SmolVLA çıkarımı 30Hz'de
   satır içi (inline) çalıştırmak için çok yavaş.

Üç düzeltmeden sonra kol, yalnızca fine-tune edilmiş politikanın kamera + eklem durumunu okumasıyla
**kalemi kendi başına alıp mouse pad'in üzerine koyuyor**.

### Bölüm 1 — Hatırlatma: model nasıl eğitildi (2. Gün)

Bugünün düzeltmeleri ancak bu bağlamda anlam kazandığı için eğitim yolunun kısa bir hatırlatması:

- 50 teleoperasyon gösterimi → veri seti `fouad1233/so101_pencil_pickup` (43.211 kare, `front` adlı 1 kamera).
- **Kaggle GPU not defterinde** (`2_kaggle_finetune_smolvla.ipynb`) her iki T4 GPU da
  `accelerate launch --multi_gpu` ile kullanılarak fine-tune edildi, 20.000 adım, efektif batch
  boyutu 16.
- `smolvla_base` **3 kamera** bekler (`camera1/2/3`); bizim veri setimizde yalnızca bir tane var
  (`front`). Eğitim anında şununla düzeltildi:
  ```text
  --rename_map={"observation.images.front": "observation.images.camera1"}
  ```
  `empty_cameras=0` (varsayılan) ile model yalnızca verdiğimiz tek kamerayı kullanır, diğer iki slotu
  yok sayar.
- Nihai checkpoint `fouad1233/smolvla_pencil_pickup`'a gönderildi.

Eğitimdeki bu `rename_map` ayrıntısı, bugün tam olarak 4. Aşamayı ısıran şey oldu (aşağıda Bölüm 3).

### Bölüm 2 — Engel 1: temel VLM indirilirken disk doldu

`./3_run_autonomous.sh`'ın ilk çalıştırması **model ağırlıkları indirilirken** çöktü, çalışırken değil:

```text
model.safetensors:  23%|##       | 463M/2.03G [00:30<01:43, 15.2MB/s]
RuntimeError: Task error: File reconstruction error: IO Error: No space left on device (os error 28)
OSError: Can't load the model for 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'.
```

`lerobot-rollout`, kendi fine-tune ağırlıklarımızı üzerine yüklemeden önce SmolVLA'nın üzerine
kurulduğu **temel VLM**'i (`SmolVLM2-500M-Video-Instruct`, ~2GB) indirmesi gerekiyor. Diskin tamamı
(`/System/Volumes/Data`) 228GB'ın **976MB'a** kadar düşmüştü — 2. Gündekiyle aynı sınıftan bir
`ENOSPC` sorunu, bu sefer yalnızca depo değil tüm sistem genelinde.

Yalnızca **yeniden üretilebilir önbellekler** temizlenerek yer açıldı (gerçek veri tutan hiçbir şeye
dokunulmadı):

```bash
rm -rf ~/.cache/huggingface/hub/*                                   # yeniden indirilebilir HF modelleri — 397M
rm -rf ~/Library/Application\ Support/Code/CachedData                # VS Code derleme önbelleği — 102M
rm -rf ~/Library/Application\ Support/Code/CachedExtensionVSIXs      # VS Code eklenti kurulum önbelleği — 455M
rm -rf ~/Library/Developer/Xcode/DerivedData                         # Xcode derleme önbelleği — 156M
npm cache clean --force                                              # ~740M çöp toplanabilir
```

Bu, boş alanı 976MB'tan → 3.4GB'a çıkardı; 2GB'lık indirmeyi bitirmeye yetti. İki çok daha büyük öğeye
(7GB+ bir VS Code uygulama-desteği klasörü ve 7GB+ bir WhatsApp `Message` klasörü) dokunulmadı; çünkü
bunlar önbellek değil gerçek veri tutuyor.

### Bölüm 3 — Engel 2: kamera adı uyuşmazlığı

Disk alanı düzeltildikten sonra model yüklendi ama robota bağlanmayı reddetti:

```text
ValueError: Visual feature mismatch between policy and robot hardware.
Policy expects: {'observation.images.camera2', 'observation.images.camera3', 'observation.images.camera1'}
Robot provides: {'observation.images.front'}
```

Bu, 2. Gündeki eğitim-anı yeniden adlandırmasının doğrudan sonucu: politikanın ağırlıkları `camera1`
adına bağlı, ama canlı robot kamerası hâlâ `front` adında (`--robot.cameras="{ front: {...} }"` ile ne
verirsek). `lerobot-rollout`, politika ile donanım arasında sıkı bir ad-eşleşme kontrolü yapıyor — ve bu
kontrolü yalnızca **`--rename_map` verildiğinde atlıyor**, eğitim anında kullandığımız aynı
yeniden adlandırmayı çıkarım anında da uygulayarak.

Çözüm — eğitim not defterinde kullanılan **aynı** `rename_map`'i ver:

```bash
--rename_map='{"observation.images.front": "observation.images.camera1"}'
```

`3_run_autonomous.sh`'a kalıcı olarak eklendi. Eksik `camera2`/`camera3` sorun değil — eğitimdeki
gibi, model yalnızca veri almadığı iki kamera slotunu yok sayıyor.

### Bölüm 4 — Engel 3: kontrol döngüsü 30Hz'e yetişemiyor

Robot bağlandı ve politika onu sürmeye başladı — ama sürekli şu uyarıyı yağdırdı:

```text
WARNING  Record loop is running slower (0.6 Hz) than the target FPS (30.0 Hz). Dataset frames might be
dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up
2) Policy inference taking too long 3) CPU starvation
```

Kök neden: `lerobot-rollout` varsayılan olarak `--inference.type=sync` kullanır — **eşzamanlı
(synchronous)** çıkarım motoru, politikayı **satır içi, her kontrol adımında bir kez** çağırır ve
kontrol döngüsü bu çağrı dönene kadar bloke olur. SmolVLA ~450M parametreli bir VLM + eylem-uzmanı;
Apple Silicon MPS'te bir ileri geçiş (forward pass), 30Hz döngünün izin verdiği 33ms bütçesinden çok
daha uzun sürüyor, bu yüzden neredeyse her adım süresini kaçırıyordu. Bu bir hata değil — sync motoru
bu donanımdaki yavaş model için yanlış araç.

**Çözüm — `--inference.type=rtc`** (Real-Time Chunking / Gerçek-Zamanlı Parçalama): politikayı kontrol
döngüsünden ayrık bir **arka plan iş parçacığında** çalıştıran *asenkron* bir çıkarım arka ucu:

- Arka plan iş parçacığı politikayı çağırmaya devam eder ve tahmin edilen eylem **parçalarını**
  (chunk) bir kuyruğa iter.
- Tam hedef FPS'te çalışan kontrol döngüsü, her adımda yalnızca kuyruktaki bir sonraki eylemi çeker ve
  robota gönderir — modeli beklemeden.
- Kuyruk azalırsa (çıkarım geride kalıyorsa), RTC yeni tahmin edilen parçayı hâlâ çalışmakta olan
  parçanın kuyruğuyla harmanlar; böylece robot yeni bir parça beklerken sarsılmaz veya durmaz — "parçalama"
  kısmı tam olarak bu: tek seferde bir eylem yerine örtüşen eylem parçalarıyla çalışır.

Kısacası: sync = "bir eylem hesapla, sonra hareket et," RTC = "bir sonraki grubu arka planda
hesaplarken yaklaşan eylemler kuyruğunu dolu tut." Yavaş-çıkarım / yüksek-kontrol-hızı uyumsuzluğunun
tam olarak ihtiyacı bu.

Ayrıca cihazı `mps` olarak sabit kodlamak yerine parametrik hale getirdik, çünkü bu depo bir gün CUDA'lı
bir Linux makinede de çalışacak — `config.env` / `config.env.example`'a `POLICY_DEVICE` eklendi (bu
Mac'te `mps`, bir Nvidia makinede `cuda`, yedek olarak `cpu`), ve `3_run_autonomous.sh` artık
`--policy.device="$POLICY_DEVICE"` okuyor.

### Bölüm 5 — Gerçekten çalıştırmak

```bash
cd pencil_pickup_vla
./3_run_autonomous.sh
```

Arka planda çalışan nihai komut:

```bash
lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
  --robot.cameras="{ front: {type: opencv, index_or_path: $CAMERA_INDEX, width: $CAMERA_W, height: $CAMERA_H, fps: $CAMERA_FPS}}" \
  --task="$TASK" \
  --policy.path="$MODEL_REPO" \
  --policy.device="$POLICY_DEVICE" \
  --inference.type=rtc \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --display_data=true
```

Sonuç: takipçi, yalnızca kamera görüntüsü + fine-tune edilmiş SmolVLA politikası tarafından sürülerek
**kalemi kendi başına aldı ve siyah mouse pad'in üzerine koydu**. FPS uyarısı yok, stabil kontrol.

### Dikkat özeti (3. Gün)

| # | Hata | Çözüm |
|---|------|-------|
| 1 | Temel VLM indirilirken `No space left on device` | sistem genelinde yeniden üretilebilir önbellekleri temizle (HF hub, VS Code, Xcode, npm) |
| 2 | `Visual feature mismatch` (camera1/2/3 vs front) | çıkarım anında da eğitimdekiyle aynı `--rename_map` |
| 3 | Kontrol döngüsü 30Hz hedefine karşı <1Hz'de takılı | `--inference.type=rtc` (asenkron arka plan çıkarımı) |

### Sıradaki — 4. Gün

- Birkaç otonom deneme daha çalıştır ve gerçek başarı oranını say.
- Tutarsızsa, aynı 50 bölümde daha uzun eğitmek yerine daha fazla/çeşitli gösterim kaydet (farklı kalem
  konumları, aydınlatma).
- Aynı politikayı CUDA'lı Linux makinede çalıştır (`POLICY_DEVICE=cuda`) ve çıkarım hızını / RTC kuyruk
  davranışını Mac ile karşılaştır.
