# Day 2 — Calibrate, Teleoperate, Record Data & Fine-Tune SmolVLA on Kaggle

**Date:** 2026-06-25

---

## English

### Overview

Today was the big end-to-end day. Starting from the two assembled SO-101 arms, we:

1. **Calibrated** both arms (leader + follower).
2. **Teleoperated** the follower with the leader to make sure the pair moves correctly.
3. **Recorded** a dataset of **50 teleoperated demonstrations** of a pick-and-place task
   ("pick up the pen and place it on the black mouse pad").
4. **Pushed** that dataset to the Hugging Face Hub (and *recovered* it after the first push failed).
5. **Fine-tuned SmolVLA** on the dataset using a **Kaggle GPU notebook** — fighting through a
   chain of **6 errors** before training finally started, then switching it to use **both T4 GPUs**.

Reused identifiers from Day 1 (`config.env` stores all of these):

- `follower = /dev/tty.usbmodem5B421352311`  (id `my_follower`)
- `leader   = /dev/tty.usbmodem5B420006171`  (id `my_leader`)
- HF username: `fouad1233`
- Dataset: `fouad1233/so101_pencil_pickup`  →  fine-tuned model: `fouad1233/smolvla_pencil_pickup`

The final dataset: **50 episodes, 43,211 frames, 30 fps, 1 camera (`front`)**.

---

### Part 1 — Calibrate both arms

Calibration moves every joint through its full range so LeRobot learns each motor's min/max. It is
saved under an `--*.id`, which we then reuse everywhere (teleop, record, run). Run **one arm at a time**.

```bash
source .venv/bin/activate

# Follower (the robot arm that does the work)
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B421352311 --robot.id=my_follower

# Leader (the small arm I move by hand)
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/tty.usbmodem5B420006171 --teleop.id=my_leader
```

> 🔑 **Same rule as Day 1:** `follower` is a `--robot.type`, `leader` is a `--teleop.type`. The `id`
> you pick here (`my_follower` / `my_leader`) must be reused in every later command, or the arm will
> ask to be calibrated again.

### Part 2 — Teleoperate (sanity check)

Before recording, confirm the follower mirrors the leader. Move the leader by hand; the follower should
copy it in real time.

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B421352311 --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/tty.usbmodem5B420006171 --teleop.id=my_leader
```

### Part 3 — Pick the camera

The dataset needs a camera. List the connected cameras and grab a preview frame from each, then set
`CAMERA_INDEX` in `config.env`.

```bash
cd pencil_pickup_vla
./find_camera.sh            # = lerobot-find-cameras opencv
```

> 📷 **Keep it identical.** The camera, its index, and its physical position must be the **same** at
> record time and at inference time, otherwise the policy sees a different world than it trained on.

### Part 4 — Install the local extras & log in to Hugging Face

Recording needs a few more LeRobot extras than Day 1 (datasets + live viz + the SmolVLA runtime).

```bash
cd pencil_pickup_vla
./setup_local.sh           # pip install -e "../lerobot[feetech,dataset,viz,smolvla]"
huggingface-cli login      # needed so the dataset can be pushed to the Hub
```

- `feetech` = SO-101 motors · `dataset` = recording (+ `av`) · `viz` = live camera view (rerun) ·
  `smolvla` = run the fine-tuned policy locally.

### Part 5 — Record the dataset (50 episodes)

All record settings live in `config.env` (task text, episode count, timings, camera, ports). The task
string **must be byte-for-byte identical** at record and inference time.

```bash
# config.env (key values)
TASK="Pick up the pen and place it on the black mouse pad"
NUM_EPISODES=50            # 50 varied demos is a good start for one task
EPISODE_TIME_S=30          # seconds per demonstration
RESET_TIME_S=10            # pause to reset the scene between demos
CAMERA_INDEX=0; CAMERA_W=640; CAMERA_H=480; CAMERA_FPS=30

./1_record_dataset.sh
```

Under the hood the script runs:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --robot.cameras="{ front: {type: opencv, index_or_path: $CAMERA_INDEX, width: $CAMERA_W, height: $CAMERA_H, fps: $CAMERA_FPS}}" \
  --teleop.type=so101_leader  --teleop.port=$LEADER_PORT  --teleop.id=$LEADER_ID \
  --display_data=true \
  --dataset.repo_id=$DATASET_REPO \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes=$NUM_EPISODES \
  --dataset.fps=$CAMERA_FPS \
  --dataset.episode_time_s=$EPISODE_TIME_S \
  --dataset.reset_time_s=$RESET_TIME_S \
  --dataset.push_to_hub=true
```

**Keyboard controls during recording:**

- `Right Arrow` → end the current episode early, go to the next.
- `Left Arrow`  → cancel & re-record the current episode.
- `Escape`      → stop and save the dataset.

> 💾 **Your data is saved locally first.** Episodes are written to
> `~/.cache/huggingface/lerobot/<repo_id>/` *as you record*, and only pushed to the Hub at the very
> end. So even if the Hub push fails, the data is not lost.

### Part 6 — Recover & upload the dataset (the first push failed)

The end-of-recording Hub push failed (the HF username was still the placeholder at first). The episodes
were safe locally, so we uploaded the local folder directly:

```bash
./recover_push_dataset.sh          # auto-finds the newest local dataset, pushes to $DATASET_REPO
```

…which essentially does:

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("fouad1233/so101_pencil_pickup", repo_type="dataset", exist_ok=True)
api.upload_folder(folder_path=SRC, repo_id="fouad1233/so101_pencil_pickup", repo_type="dataset")
```

Dataset: [https://huggingface.co/datasets/fouad1233/so101_pencil_pickup](https://huggingface.co/datasets/fouad1233/so101_pencil_pickup)

> ⚠️ **Gotcha (bit us later on Kaggle):** `upload_folder` copies the files but **does not create the
> codebase-version tag** (`v3.0`) that LeRobot's own `push_to_hub` would. LeRobot needs that tag to
> resolve the dataset version. We fix this in Part 8, blocker #4.

### Part 7 — Kaggle: upload the training notebook

Training is too heavy for the Mac, so it runs on a **free Kaggle GPU**. The notebook
`2_kaggle_finetune_smolvla.ipynb` is uploaded straight from the CLI (no copy-paste). GPU + Internet are
pre-enabled via `kernel-metadata.json`.

```bash
cd pencil_pickup_vla

# Authenticate once (Kaggle CLI 2.x) — pick one:
kaggle auth login                          # browser, nothing to paste (recommended)
# or:  export KAGGLE_API_TOKEN="KGAT_..."  # your token, for this shell only

./push_notebook_to_kaggle.sh               # = kaggle kernels push -p .
```

Notebook: [https://www.kaggle.com/code/fouad1233/smolvla-pencil-pickup](https://www.kaggle.com/code/fouad1233/smolvla-pencil-pickup)

The notebook's 6 steps: (1) `!nvidia-smi`, (2) install LeRobot, (3) HF login via the `HF_TOKEN` Kaggle
secret, (4) settings, (5) fine-tune, (6) push the model to the Hub.

```python
# Cell 2 — HF login uses a Kaggle Secret (Add-ons -> Secrets -> add HF_TOKEN)
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
login(UserSecretsClient().get_secret("HF_TOKEN"))
```

> 🔐 **On Kaggle, do this once:** Add-ons → Secrets → add `HF_TOKEN`; Session options → Accelerator →
> **GPU T4 x2**; then **Run All**. Revoke any API token you exported once you're done.

### Part 8 — The Kaggle error chain (6 blockers we fixed)

Training did not start on the first try. Each error below is paired with its fix. This is the real meat
of Day 2.

**Blocker 1 — `ImportError: 'av' is required to decode videos`.**
The base install lacked the video/dataset deps. Fix: install the right extras (notebook Cell 4):

```python
!pip install -q "lerobot[smolvla,dataset] @ git+https://github.com/huggingface/lerobot.git"
```

**Blocker 2 — `CUDA error: ... sm_60 is not compatible`.**
Kaggle had assigned an old **P100** GPU (compute capability `sm_60`), which the modern PyTorch wheel no
longer supports. Fix: in Session options pick **GPU T4 x2** (Turing, `sm_75`).

**Blocker 3 — `ValueError: 'repo_id' argument missing`.**
`smolvla_base` ships with `push_to_hub=True`, so `lerobot-train` demanded a `repo_id` to auto-push the
result. We push the model ourselves in Cell 6, so we just disable the auto-push:

```text
--policy.push_to_hub=false
```

**Blocker 4 — `RevisionNotFoundError` (hidden under a confusing `HfHubHTTPError ... missing 'response'`).**
This is the consequence of the Part 6 gotcha: the dataset had **no `v3.0` version tag**. LeRobot even
prints the fix. We ran it locally:

```python
from huggingface_hub import HfApi
HfApi().create_tag("fouad1233/so101_pencil_pickup", tag="v3.0", repo_type="dataset")
```

Verified it resolves: `episodes: 50 | frames: 43211 | fps: 30 | camera: observation.images.front`.

**Blocker 5 — `lerobot-train` crashed while reading video frames (torchcodec / ffmpeg mismatch).**
`torchcodec` (the default decoder) is fragile about the ffmpeg version on Kaggle. Fix: switch to the
rock-solid `pyav` backend (installed by the `dataset` extra):

```text
--dataset.video_backend=pyav
```

**Blocker 6 — `Feature mismatch ... Missing: camera1, camera2, camera3 / Extra: observation.images.front`.**
`smolvla_base` was pretrained expecting **3 cameras** named `camera1/2/3`; our dataset has **1** camera
named `front`. Fix: rename it to one of the expected names. With `empty_cameras=0` (the default), the
model simply uses the one camera we provide and ignores the other two slots.

```text
--rename_map={"observation.images.front": "observation.images.camera1"}
```

After blocker 6, training started:
`Effective batch size: 16 x 1 = 16 · num_learnable_params=100M / total=450M`.

> 🧹 **Side gotcha — the Mac disk filled to 100%** mid-session (`ENOSPC`), which blocked even shell
> output and a `git push`. Fix: clear regenerable caches (NOT the dataset):
>
> ```bash
> rm -rf ~/Library/Caches/Homebrew ~/.cache/huggingface/hub      # freed ~7.3 GB
> ```
>
> The recovered dataset under `~/.cache/huggingface/lerobot/` was left untouched.

### Part 9 — Use both T4 GPUs (multi-GPU)

The single-GPU run was `~2.57 s/step × 20000 = ~14 h`, which would hit **Kaggle's 12 h session limit**
before finishing. Kaggle gives **two** T4s, so we parallelize with HF `accelerate` (LeRobot's documented
multi-GPU path). Final training cell:

```python
import subprocess, shutil

train_args = [
    "--policy.path=lerobot/smolvla_base",
    f"--dataset.repo_id={DATASET_REPO}",
    "--dataset.video_backend=pyav",
    '--rename_map={"observation.images.front": "observation.images.camera1"}',
    f"--batch_size={BATCH_SIZE}",          # PER-GPU; effective = BATCH_SIZE x NUM_GPUS
    f"--steps={STEPS}",
    f"--save_freq={SAVE_FREQ}",
    f"--output_dir={OUTPUT_DIR}",
    "--job_name=smolvla_pencil",
    "--policy.device=cuda",
    "--policy.push_to_hub=false",
    "--wandb.enable=false",
]

if NUM_GPUS and NUM_GPUS > 1:
    cmd = ["accelerate", "launch", "--multi_gpu",
           f"--num_processes={NUM_GPUS}",
           shutil.which("lerobot-train")] + train_args
else:
    cmd = ["lerobot-train"] + train_args

subprocess.run(cmd, check=True)
```

Settings used: `NUM_GPUS = 2`, `BATCH_SIZE = 8` (per-GPU) → **effective batch 8 × 2 = 16** (same as the
single-GPU run, just split across both T4s), `STEPS = 20000`, `SAVE_FREQ = 5000`.

> ✅ **Confirmation both GPUs are used:** the log changes from `Effective batch size: 16 x 1 = 16` to
> **`8 x 2 = 16`** — that `x 2` is the world size. ETA dropped from ~14 h to **~8–10 h**, inside the
> 12 h limit.

> 🖥️ **For long runs, don't babysit the tab.** Use **Save Version → "Save & Run All (Commit)"** so the
> notebook runs server-side in the background (and Cell 6 uploads the model automatically at the end).

### Part 10 — Push the fine-tuned model to the Hub

The last notebook cell uploads the final checkpoint to the Hub:

```python
from huggingface_hub import HfApi
from pathlib import Path
ckpt = Path(OUTPUT_DIR) / "checkpoints" / "last" / "pretrained_model"
api = HfApi()
api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=str(ckpt), repo_id=MODEL_REPO, repo_type="model")
# -> https://huggingface.co/fouad1233/smolvla_pencil_pickup
```

### Gotchas summary (the 6 blockers, at a glance)

| # | Error                                      | Fix                                              |
| - | ------------------------------------------ | ------------------------------------------------ |
| 1 | `'av' is required` to decode videos      | install`lerobot[smolvla,dataset]`              |
| 2 | CUDA`sm_60 not compatible` (P100)        | use**GPU T4 x2**                           |
| 3 | `'repo_id' argument missing`             | `--policy.push_to_hub=false`                   |
| 4 | `RevisionNotFoundError` (no version tag) | `create_tag(..., "v3.0", repo_type="dataset")` |
| 5 | torchcodec / ffmpeg crash on video         | `--dataset.video_backend=pyav`                 |
| 6 | camera name mismatch                       | `--rename_map front→camera1`                  |
| + | Mac disk full (`ENOSPC`)                 | clear Homebrew + HF hub caches                   |

### Next — Day 3

- Wait for the Kaggle run to finish (~8–10 h) and confirm `fouad1233/smolvla_pencil_pickup` is on the Hub.
- Pull the model locally and **run the policy autonomously** on the follower (`3_run_autonomous.sh`),
  using the **same camera and the same task string** as recording.
- Evaluate: how often does it actually pick the pen and place it on the pad? Decide whether to record
  more/varied demos or train longer.

---

## Türkçe

### Genel Bakış

Bugün uçtan uca büyük gündü. Monte edilmiş iki SO-101 kolundan başlayarak:

1. İki kolu da (lider + takipçi) **kalibre ettik**.
2. Lider ile takipçiyi **teleoperasyonla** kontrol edip çiftin doğru hareket ettiğini doğruladık.
3. Bir al-götür görevinin ("kalemi al ve siyah mouse pad'in üzerine koy") **50 teleoperasyon
   gösterimini** içeren bir veri seti **kaydettik**.
4. Bu veri setini Hugging Face Hub'a **yükledik** (ilk yükleme başarısız olunca *kurtardık*).
5. Veri seti üzerinde **Kaggle GPU not defteri** kullanarak **SmolVLA'yı fine-tune ettik** — eğitim
   nihayet başlamadan önce **6 hatadan** oluşan bir zincirle boğuştuk, sonra **iki T4 GPU'yu** birden
   kullanacak şekilde değiştirdik.
6. Günden yeniden kullanılan kimlikler (hepsi `config.env` içinde):

- `follower = /dev/tty.usbmodem5B421352311`  (id `my_follower`)
- `leader   = /dev/tty.usbmodem5B420006171`  (id `my_leader`)
- HF kullanıcı adı: `fouad1233`
- Veri seti: `fouad1233/so101_pencil_pickup`  →  fine-tune model: `fouad1233/smolvla_pencil_pickup`

Nihai veri seti: **50 bölüm (episode), 43.211 kare, 30 fps, 1 kamera (`front`)**.

---

### Bölüm 1 — İki kolu kalibre et

Kalibrasyon her eklemi tam hareket aralığında gezdirir; böylece LeRobot her motorun min/maks değerini
öğrenir. Bir `--*.id` altına kaydedilir ve bunu her yerde (teleop, kayıt, çalıştırma) tekrar kullanırız.
**Her seferinde tek kol** çalıştır.

```bash
source .venv/bin/activate

# Takipçi (işi yapan robot kol)
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B421352311 --robot.id=my_follower

# Lider (elimle hareket ettirdiğim küçük kol)
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/tty.usbmodem5B420006171 --teleop.id=my_leader
```

> 🔑 **1. Gündeki kuralın aynısı:** `follower` bir `--robot.type`, `leader` bir `--teleop.type`'tır.
> Burada seçtiğin `id` (`my_follower` / `my_leader`) sonraki her komutta tekrar kullanılmalı; aksi
> halde kol yeniden kalibrasyon ister.

### Bölüm 2 — Teleoperasyon (kontrol testi)

Kayıttan önce takipçinin lideri taklit ettiğini doğrula. Lideri elinle hareket ettir; takipçi gerçek
zamanlı kopyalamalı.

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B421352311 --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/tty.usbmodem5B420006171 --teleop.id=my_leader
```

### Bölüm 3 — Kamerayı seç

Veri seti bir kamera gerektirir. Bağlı kameraları listele, her birinden önizleme karesi al, sonra
`config.env` içinde `CAMERA_INDEX` değerini ayarla.

```bash
cd pencil_pickup_vla
./find_camera.sh            # = lerobot-find-cameras opencv
```

> 📷 **Aynı kalmalı.** Kamera, indeksi ve fiziksel konumu kayıt anında ve çıkarım (inference) anında
> **aynı** olmalı; yoksa politika, eğitildiğinden farklı bir dünya görür.

### Bölüm 4 — Yerel ekleri kur & Hugging Face'e giriş yap

Kayıt, 1. Güne göre birkaç ek LeRobot bağımlılığı daha gerektirir (datasets + canlı viz + SmolVLA çalışma zamanı).

```bash
cd pencil_pickup_vla
./setup_local.sh           # pip install -e "../lerobot[feetech,dataset,viz,smolvla]"
huggingface-cli login      # veri setini Hub'a göndermek için gerekli
```

- `feetech` = SO-101 motorları · `dataset` = kayıt (+ `av`) · `viz` = canlı kamera görünümü (rerun) ·
  `smolvla` = fine-tune edilmiş politikayı yerelde çalıştırmak.

### Bölüm 5 — Veri setini kaydet (50 bölüm)

Tüm kayıt ayarları `config.env` içinde (görev metni, bölüm sayısı, süreler, kamera, portlar). Görev metni
kayıt ve çıkarım anında **harfi harfine aynı** olmalı.

```bash
# config.env (önemli değerler)
TASK="Pick up the pen and place it on the black mouse pad"
NUM_EPISODES=50            # bir görev için 50 çeşitli gösterim iyi bir başlangıç
EPISODE_TIME_S=30          # gösterim başına saniye
RESET_TIME_S=10            # gösterimler arası sahneyi sıfırlama molası
CAMERA_INDEX=0; CAMERA_W=640; CAMERA_H=480; CAMERA_FPS=30

./1_record_dataset.sh
```

Betik arka planda şunu çalıştırır:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --robot.cameras="{ front: {type: opencv, index_or_path: $CAMERA_INDEX, width: $CAMERA_W, height: $CAMERA_H, fps: $CAMERA_FPS}}" \
  --teleop.type=so101_leader  --teleop.port=$LEADER_PORT  --teleop.id=$LEADER_ID \
  --display_data=true \
  --dataset.repo_id=$DATASET_REPO \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes=$NUM_EPISODES \
  --dataset.fps=$CAMERA_FPS \
  --dataset.episode_time_s=$EPISODE_TIME_S \
  --dataset.reset_time_s=$RESET_TIME_S \
  --dataset.push_to_hub=true
```

**Kayıt sırasında klavye kontrolleri:**

- `Sağ Ok` → mevcut bölümü erken bitir, sonrakine geç.
- `Sol Ok`  → mevcut bölümü iptal et & yeniden kaydet.
- `Escape`  → durdur ve veri setini kaydet.

> 💾 **Verin önce yerelde kaydedilir.** Bölümler kaydederken `~/.cache/huggingface/lerobot/<repo_id>/`
> altına yazılır ve yalnızca en sonda Hub'a gönderilir. Yani Hub gönderimi başarısız olsa bile veri kaybolmaz.

### Bölüm 6 — Veri setini kurtar & yükle (ilk gönderim başarısız oldu)

Kayıt sonundaki Hub gönderimi başarısız oldu (HF kullanıcı adı başta hâlâ yer tutucuydu). Bölümler yerelde
güvendeydi; yerel klasörü doğrudan yükledik:

```bash
./recover_push_dataset.sh          # en yeni yerel veri setini otomatik bulur, $DATASET_REPO'ya gönderir
```

…ki bu özünde şudur:

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("fouad1233/so101_pencil_pickup", repo_type="dataset", exist_ok=True)
api.upload_folder(folder_path=SRC, repo_id="fouad1233/so101_pencil_pickup", repo_type="dataset")
```

Veri seti: [https://huggingface.co/datasets/fouad1233/so101_pencil_pickup](https://huggingface.co/datasets/fouad1233/so101_pencil_pickup)

> ⚠️ **Dikkat (sonradan Kaggle'da bizi yaktı):** `upload_folder` dosyaları kopyalar ama LeRobot'un kendi
> `push_to_hub`'ının oluşturacağı **kod-tabanı sürüm etiketini** (`v3.0`) oluşturmaz. LeRobot veri seti
> sürümünü çözmek için bu etikete ihtiyaç duyar. Bunu Bölüm 8, engel #4'te düzeltiyoruz.

### Bölüm 7 — Kaggle: eğitim not defterini yükle

Eğitim Mac için fazla ağır, bu yüzden **ücretsiz Kaggle GPU**'sunda çalışır. `2_kaggle_finetune_smolvla.ipynb`
not defteri doğrudan CLI'den yüklenir (kopyala-yapıştır yok). GPU + İnternet, `kernel-metadata.json` ile
önceden etkin.

```bash
cd pencil_pickup_vla

# Bir kez kimlik doğrula (Kaggle CLI 2.x) — birini seç:
kaggle auth login                          # tarayıcı, yapıştıracak bir şey yok (önerilir)
# veya:  export KAGGLE_API_TOKEN="KGAT_..."  # token'ın, yalnızca bu kabuk için

./push_notebook_to_kaggle.sh               # = kaggle kernels push -p .
```

Not defteri: [https://www.kaggle.com/code/fouad1233/smolvla-pencil-pickup](https://www.kaggle.com/code/fouad1233/smolvla-pencil-pickup)

Not defterinin 6 adımı: (1) `!nvidia-smi`, (2) LeRobot kurulumu, (3) `HF_TOKEN` Kaggle gizli anahtarıyla
HF girişi, (4) ayarlar, (5) fine-tune, (6) modeli Hub'a gönder.

```python
# Hücre 2 — HF girişi bir Kaggle Secret kullanır (Add-ons -> Secrets -> HF_TOKEN ekle)
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
login(UserSecretsClient().get_secret("HF_TOKEN"))
```

> 🔐 **Kaggle'da bunu bir kez yap:** Add-ons → Secrets → `HF_TOKEN` ekle; Session options → Accelerator →
> **GPU T4 x2**; sonra **Run All**. İşin bitince dışa aktardığın API token'ını iptal et (revoke).

### Bölüm 8 — Kaggle hata zinciri (düzelttiğimiz 6 engel)

Eğitim ilk denemede başlamadı. Aşağıdaki her hata, çözümüyle eşleştirilmiştir. 2. Günün asıl özü budur.

**Engel 1 — `ImportError: 'av' is required to decode videos`.**
Temel kurulumda video/dataset bağımlılıkları eksikti. Çözüm: doğru ekleri kur (Hücre 4):

```python
!pip install -q "lerobot[smolvla,dataset] @ git+https://github.com/huggingface/lerobot.git"
```

**Engel 2 — `CUDA error: ... sm_60 is not compatible`.**
Kaggle eski bir **P100** GPU (compute capability `sm_60`) atamıştı; modern PyTorch wheel'i artık
desteklemiyor. Çözüm: Session options'ta **GPU T4 x2** seç (Turing, `sm_75`).

**Engel 3 — `ValueError: 'repo_id' argument missing`.**
`smolvla_base` `push_to_hub=True` ile gelir, bu yüzden `lerobot-train` sonucu otomatik göndermek için bir
`repo_id` istedi. Modeli kendimiz Hücre 6'da gönderiyoruz, o yüzden otomatik göndermeyi kapatıyoruz:

```text
--policy.push_to_hub=false
```

**Engel 4 — `RevisionNotFoundError` (kafa karıştırıcı bir `HfHubHTTPError ... missing 'response'` altında gizli).**
Bu, Bölüm 6 dikkat notunun sonucu: veri setinin **`v3.0` sürüm etiketi yoktu**. LeRobot çözümü bile
yazdırıyor. Yerelde çalıştırdık:

```python
from huggingface_hub import HfApi
HfApi().create_tag("fouad1233/so101_pencil_pickup", tag="v3.0", repo_type="dataset")
```

Çözüldüğünü doğruladık: `episodes: 50 | frames: 43211 | fps: 30 | camera: observation.images.front`.

**Engel 5 — `lerobot-train` video kareleri okurken çöktü (torchcodec / ffmpeg uyumsuzluğu).**
`torchcodec` (varsayılan çözücü) Kaggle'da ffmpeg sürümü konusunda kırılgandır. Çözüm: sağlam `pyav`
backend'ine geç (`dataset` ekiyle gelir):

```text
--dataset.video_backend=pyav
```

**Engel 6 — `Feature mismatch ... Missing: camera1, camera2, camera3 / Extra: observation.images.front`.**
`smolvla_base` `camera1/2/3` adlı **3 kamera** bekleyecek şekilde önceden eğitilmiş; bizim veri setimizde
`front` adlı **1** kamera var. Çözüm: onu beklenen adlardan birine yeniden adlandır. `empty_cameras=0`
(varsayılan) ile model yalnızca verdiğimiz tek kamerayı kullanır, diğer iki slotu yok sayar.

```text
--rename_map={"observation.images.front": "observation.images.camera1"}
```

Engel 6'dan sonra eğitim başladı:
`Effective batch size: 16 x 1 = 16 · num_learnable_params=100M / total=450M`.

> 🧹 **Yan dikkat — Mac diski %100 doldu** (`ENOSPC`); bu, kabuk çıktısını ve bir `git push`'u bile
> engelledi. Çözüm: yeniden üretilebilir önbellekleri temizle (veri setini DEĞİL):
>
> ```bash
> rm -rf ~/Library/Caches/Homebrew ~/.cache/huggingface/hub      # ~7.3 GB açtı
> ```
>
> `~/.cache/huggingface/lerobot/` altındaki kurtarılan veri setine dokunulmadı.

### Bölüm 9 — İki T4 GPU'yu birden kullan (multi-GPU)

Tek GPU çalışması `~2.57 sn/adım × 20000 = ~14 saat` idi; bu da **Kaggle'ın 12 saatlik oturum
limitine** bitmeden takılırdı. Kaggle **iki** T4 verir, bu yüzden HF `accelerate` ile paralelleştiriyoruz
(LeRobot'un belgelenmiş multi-GPU yolu). Nihai eğitim hücresi:

```python
import subprocess, shutil

train_args = [
    "--policy.path=lerobot/smolvla_base",
    f"--dataset.repo_id={DATASET_REPO}",
    "--dataset.video_backend=pyav",
    '--rename_map={"observation.images.front": "observation.images.camera1"}',
    f"--batch_size={BATCH_SIZE}",          # GPU BAŞINA; efektif = BATCH_SIZE x NUM_GPUS
    f"--steps={STEPS}",
    f"--save_freq={SAVE_FREQ}",
    f"--output_dir={OUTPUT_DIR}",
    "--job_name=smolvla_pencil",
    "--policy.device=cuda",
    "--policy.push_to_hub=false",
    "--wandb.enable=false",
]

if NUM_GPUS and NUM_GPUS > 1:
    cmd = ["accelerate", "launch", "--multi_gpu",
           f"--num_processes={NUM_GPUS}",
           shutil.which("lerobot-train")] + train_args
else:
    cmd = ["lerobot-train"] + train_args

subprocess.run(cmd, check=True)
```

Kullanılan ayarlar: `NUM_GPUS = 2`, `BATCH_SIZE = 8` (GPU başına) → **efektif batch 8 × 2 = 16** (tek GPU
çalışmasıyla aynı, sadece iki T4'e bölünmüş), `STEPS = 20000`, `SAVE_FREQ = 5000`.

> ✅ **İki GPU'nun kullanıldığının kanıtı:** log `Effective batch size: 16 x 1 = 16`'dan
> **`8 x 2 = 16`**'ya değişir — o `x 2` dünya boyutudur (world size). ETA ~14 saatten **~8–10 saate**
> düştü, 12 saat limitinin içinde.

> 🖥️ **Uzun çalışmalarda sekmeyi başında bekleme.** **Save Version → "Save & Run All (Commit)"**
> kullan; not defteri arka planda sunucu tarafında çalışır (ve Hücre 6 modeli sonunda otomatik yükler).

### Bölüm 10 — Fine-tune edilmiş modeli Hub'a gönder

Son not defteri hücresi nihai checkpoint'i Hub'a yükler:

```python
from huggingface_hub import HfApi
from pathlib import Path
ckpt = Path(OUTPUT_DIR) / "checkpoints" / "last" / "pretrained_model"
api = HfApi()
api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=str(ckpt), repo_id=MODEL_REPO, repo_type="model")
# -> https://huggingface.co/fouad1233/smolvla_pencil_pickup
```

### Dikkat özeti (6 engel, tek bakışta)

| # | Hata                                            | Çözüm                                         |
| - | ----------------------------------------------- | ------------------------------------------------ |
| 1 | Video çözmek için`'av'` gerekli            | `lerobot[smolvla,dataset]` kur                 |
| 2 | CUDA`sm_60 not compatible` (P100)             | **GPU T4 x2** kullan                       |
| 3 | `'repo_id' argument missing`                  | `--policy.push_to_hub=false`                   |
| 4 | `RevisionNotFoundError` (sürüm etiketi yok) | `create_tag(..., "v3.0", repo_type="dataset")` |
| 5 | videoda torchcodec / ffmpeg çökmesi           | `--dataset.video_backend=pyav`                 |
| 6 | kamera adı uyuşmazlığı                     | `--rename_map front→camera1`                  |
| + | Mac diski dolu (`ENOSPC`)                     | Homebrew + HF hub önbelleklerini temizle        |

### Sıradaki — 3. Gün

- Kaggle çalışmasının bitmesini bekle (~8–10 saat) ve `fouad1233/smolvla_pencil_pickup`'ın Hub'da
  olduğunu doğrula.
- Modeli yerele indir ve politikayı takipçi üzerinde **otonom çalıştır** (`3_run_autonomous.sh`),
  kayıttakiyle **aynı kamera ve aynı görev metnini** kullanarak.
- Değerlendir: kalemi gerçekten ne sıklıkla alıp pad'in üzerine koyuyor? Daha fazla/çeşitli gösterim
  kaydetmeye mi yoksa daha uzun eğitmeye mi karar ver.
