# Day 5 — First Autonomous Test: Picks Up the Ball, Won't Let Go

**Date:** 2026-07-03

---

## English

### Overview

Woke up to a finished checkpoint — `pi0_ball_pickup` had trained the full 10,000 steps overnight and
was already pushed to the Hub. Before it could actually run on the arm, one more fix was needed
(same class of bug as the pencil-pickup task back on Day 3), and then came the real test: does it
pick up the tennis ball and place it in the white cup?

Short answer: **partially**. It reliably finds and grasps the ball, and it reliably drives it over to
the white cup — but it doesn't cleanly *release* it there. Instead, it keeps pressing/pushing the
ball downward as if trying to force it further into the cup, rather than opening the gripper and
backing off. Sometimes that ends with the ball in the cup anyway (just shoved in, not placed), other
times the pushing knocks it against the rim instead.

### Part 1 — One more fix before it would run at all

`3_run_autonomous.sh` failed on the first attempt with the same shape of error the pencil task hit on
Day 3:

```text
ValueError: Visual feature mismatch between policy and robot hardware.
```

Reason: a checkpoint's declared `input_features` keep the **base model's** original camera names
(`base_0_rgb`/`left_wrist_0_rgb`/`right_wrist_0_rgb`) permanently — `--rename_map` only remapped the
**dataset's** columns at training time (Day 4, Part 3), it doesn't rename the model's own feature
slots. So the exact same rename map used for training has to be passed again at inference time, to
match our robot's actual camera names (`top`/`wrist`):

```bash
--rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
```

Added permanently to `3_run_autonomous.sh` (alongside a `POLICY_SOURCE` toggle to run from either the
local checkpoint or the published Hub model). While in there, also built a persistent FastAPI + web UI
(`4_run_ui.sh`) that loads pi0 once and keeps it warm across runs — driven by lerobot's real
`RTCInferenceEngine`, the same async engine `--inference.type=rtc` uses on the CLI, so motion matches
exactly either way.

### Part 2 — Running it for real

```bash
./3_run_autonomous.sh
```

```bash
lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower --robot.cameras="$CAMERAS" \
  --task="$TASK" \
  --policy.path="$POLICY_PATH" --policy.device=cuda \
  --inference.type=rtc \
  --rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --display_data=true
```

`--inference.type=rtc` runs the policy in a background thread, decoupled from the control loop, so a
~4B-param model's inference time doesn't stall the arm — same reasoning as the pencil task's Day-3
fix, just already wired in from the start this time.

One thing worth flagging: the script's own header comment says *"`--robot.max_relative_target` caps
how far each joint may move per step"* — but the actual command doesn't pass that flag at all for
this task. The comment is stale; there's currently no per-step motion clamp on `ball_pickup_pi0`,
unlike a later task (`cap_sort_pi0`) where the same missing clamp caused visibly unstable motion and
had to be added back. Noted for follow-up, not yet confirmed as related to today's release problem.

### Part 3 — What actually happened

Across several trials:

- **Localizing + grasping the ball:** consistently good. The arm finds the tennis ball on the table
  and closes the gripper on it.
- **Carrying it to the cup:** consistently good. It moves the ball over the white cup reliably.
- **Releasing it into the cup:** this is where it falls apart. Instead of opening the gripper and
  lifting away once positioned over the cup, the policy keeps driving the gripper *downward*, pressing
  the ball into the cup like it's trying to force it in deeper, rather than letting go. It never
  settles into a clean "open gripper, retract" ending the way the demonstrations presumably did.

The task is *usually* nominally satisfied — the ball ends up in the cup — but only because it got
pushed there, not placed there. Occasionally the pushing motion instead knocks the ball against the
cup's rim.

### Ideas for Day 6 (not yet tried, just hypotheses)

- **15 demos may not show the release clearly or consistently enough.** If the release action varied
  a lot episode-to-episode during teleop (how fast the gripper opened, how far it retracted before
  the next reset), the model may not have learned a confident, repeatable release — "keep pushing" can
  look like a locally-safe way to hedge when the training signal for "let go now" was noisy.
  Recording more episodes with a deliberate, exaggerated, identical release motion each time seems
  like the first thing to try.
- **The release is a short slice of a 40-second episode.** It may simply be underrepresented in the
  training data relative to the reach/grasp/carry portion. Longer training or more targeted data
  (shorter episodes focused specifically on the placement phase) might help.
- **No `--robot.max_relative_target` clamp on this task** (Part 2). Worth testing with a small clamp
  added, mainly to rule in/out whether jittery fine-motor control near the cup is compounding the
  "won't let go" behavior — though the core symptom (pushing instead of releasing) looks like a
  learned-behavior issue, not a motion-smoothness one.

---

## Türkçe

### Genel Bakış

Bitmiş bir checkpoint ile uyandık — `pi0_ball_pickup` gece boyunca tam 10.000 adımı eğitmişti ve
zaten Hub'a gönderilmişti. Kolda gerçekten çalışabilmesi için bir düzeltme daha gerekti (3. Gündeki
kalem-alma göreviyle aynı sınıftan bir hata), ve ardından asıl test geldi: tenis topunu alıp beyaz
bardağa koyuyor mu?

Kısa cevap: **kısmen**. Topu bulup tutmakta tutarlı, topu beyaz bardağın üzerine götürmekte de
tutarlı — ama orada temiz bir şekilde **bırakmıyor**. Bunun yerine, sanki daha derine zorlamaya
çalışıyormuş gibi topu aşağı doğru bastırmaya devam ediyor, tutucuyu açıp geri çekilmek yerine.
Bazen bu, topun yine de bardağa girmesiyle sonuçlanıyor (yerleştirilmiş değil, itilmiş), bazen de
itme hareketi topu bardağın kenarına çarptırıyor.

### Bölüm 1 — Çalışmadan önce bir düzeltme daha

`3_run_autonomous.sh`, ilk denemede 3. Gündeki kalem-alma görevinin karşılaştığıyla aynı şekilde bir
hatayla başarısız oldu:

```text
ValueError: Visual feature mismatch between policy and robot hardware.
```

Sebep: bir checkpoint'in bildirdiği `input_features`, **temel modelin** özgün kamera adlarını
(`base_0_rgb`/`left_wrist_0_rgb`/`right_wrist_0_rgb`) kalıcı olarak koruyor — `--rename_map` yalnızca
**veri setinin** sütunlarını eğitim anında yeniden eşlemişti (4. Gün, Bölüm 3), modelin kendi özellik
slotlarını yeniden adlandırmıyor. Yani eğitimde kullanılan aynı yeniden adlandırma haritası, robotun
gerçek kamera adlarıyla (`top`/`wrist`) eşleşmesi için çıkarım anında da tekrar verilmeli:

```bash
--rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
```

`3_run_autonomous.sh`'a kalıcı olarak eklendi (yerel checkpoint veya yayınlanmış Hub modelinden
çalıştırmak arasında geçiş yapan bir `POLICY_SOURCE` seçeneğiyle birlikte). Bu arada, pi0'ı bir kez
yükleyip çalışmalar arasında sıcak tutan kalıcı bir FastAPI + web arayüzü de (`4_run_ui.sh`)
kuruldu — lerobot'un gerçek `RTCInferenceEngine`'i tarafından yönetiliyor, CLI'de
`--inference.type=rtc`'nin kullandığı aynı asenkron motor, böylece hareket her iki durumda da
birebir aynı.

### Bölüm 2 — Gerçekten çalıştırmak

```bash
./3_run_autonomous.sh
```

```bash
lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower --robot.cameras="$CAMERAS" \
  --task="$TASK" \
  --policy.path="$POLICY_PATH" --policy.device=cuda \
  --inference.type=rtc \
  --rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --display_data=true
```

`--inference.type=rtc`, politikayı kontrol döngüsünden ayrık bir arka plan iş parçacığında çalıştırır,
böylece ~4B parametreli bir modelin çıkarım süresi kolu durdurmaz — 3. Gündeki kalem görevinin
düzeltmesiyle aynı mantık, bu sefer yalnızca başından beri gömülü.

Not edilmeye değer bir şey: betiğin kendi başlık yorumu *"`--robot.max_relative_target`, her eklemin
adım başına ne kadar hareket edebileceğini sınırlar"* diyor — ama gerçek komut bu görev için bu
bayrağı hiç geçirmiyor. Yorum güncel değil; şu anda `ball_pickup_pi0`'da adım başına hareket sınırı
yok, sonraki bir görevin (`cap_sort_pi0`) aynı eksik sınırlamanın görünür şekilde kararsız harekete
yol açtığı ve geri eklenmesi gerektiği durumun aksine. Takip için not edildi, bugünün bırakma
sorunuyla ilgili olduğu henüz doğrulanmadı.

### Bölüm 3 — Gerçekte ne oldu

Birkaç denemede:

- **Topu bulma + tutma:** tutarlı şekilde iyi. Kol masadaki tenis topunu buluyor ve tutucuyu üzerine kapatıyor.
- **Bardağa taşıma:** tutarlı şekilde iyi. Topu güvenilir şekilde beyaz bardağın üzerine götürüyor.
- **Bardağa bırakma:** işte burada dağılıyor. Bardağın üzerinde konumlandıktan sonra tutucuyu açıp
  uzaklaşmak yerine, politika tutucuyu **aşağı** doğru sürmeye devam ediyor, sanki bırakmak yerine
  topu daha derine zorlamaya çalışıyormuş gibi topu bardağa bastırıyor. Gösterimlerin muhtemelen
  yaptığı gibi temiz bir "tutucuyu aç, geri çek" bitişine hiç oturmuyor.

Görev *genellikle* nominal olarak yerine getiriliyor — top bardağa giriyor — ama sadece oraya
itildiği için, yerleştirildiği için değil. Bazen itme hareketi topu bardağın kenarına çarptırıyor.

### 6. Gün için fikirler (henüz denenmedi, yalnızca varsayım)

- **15 gösterim, bırakmayı yeterince net veya tutarlı göstermiyor olabilir.** Teleoperasyon sırasında
  bırakma hareketi bölümden bölüme çok değiştiyse (tutucu ne kadar hızlı açıldı, bir sonraki
  sıfırlamadan önce ne kadar geri çekildi), model güvenli, tekrarlanabilir bir bırakma öğrenmemiş
  olabilir — "itmeye devam et", "şimdi bırak" için eğitim sinyali gürültülüyse yerel olarak güvenli
  görünen bir seçenek gibi davranabilir. Her seferinde kasıtlı, abartılı, özdeş bir bırakma
  hareketiyle daha fazla bölüm kaydetmek denenecek ilk şey gibi görünüyor.
- **Bırakma, 40 saniyelik bir bölümün kısa bir dilimi.** Yaklaşma/tutma/taşıma kısmına göre eğitim
  verisinde yeterince temsil edilmiyor olabilir. Daha uzun eğitim veya daha hedefli veri (özellikle
  yerleştirme aşamasına odaklanan daha kısa bölümler) yardımcı olabilir.
- **Bu görevde `--robot.max_relative_target` sınırı yok** (Bölüm 2). Bardağın yakınındaki titrek
  ince-motor kontrolünün "bırakmıyor" davranışını büyütüp büyütmediğini ayırt etmek için küçük bir
  sınırla test etmeye değer — yine de temel belirti (bırakmak yerine itmek) hareket-pürüzsüzlüğü
  değil, öğrenilmiş-davranış sorunu gibi görünüyor.
