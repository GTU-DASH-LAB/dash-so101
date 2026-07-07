# dash-so101

My journey building and learning with the **SO-101** robotic arms and Vision-Language-Action (VLA) models.

## Contents

- `daily_progress/` — a daily diary of the project (`day1.md`, `day2.md`, …), English + Türkçe.
- `lerobot/` — the [LeRobot](https://github.com/huggingface/lerobot) library, included as a **git submodule**.

## Setup

Clone with the submodule:

```bash
git clone --recurse-submodules https://github.com/GTU-DASH-LAB/dash-so101.git
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### Python environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e "./lerobot[feetech]"
```

## Updating the LeRobot submodule

```bash
cd lerobot
git checkout main
git pull
cd ..
git add lerobot
git commit -m "Update lerobot submodule"
```

## Contributors

- **[Fouad Aladhami](https://github.com/fouad1233)** — project owner/maintainer: SO-101 hardware, data
  collection, fine-tuning pipelines, and daily progress notes.
