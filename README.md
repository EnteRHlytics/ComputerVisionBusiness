# Desk Companion

Local computer-vision desk tracker. Uses two webcams (laptop + Pixel 7 Pro as a
webcam) to read your **facial emotion**, **posture**, and **activity**, logging
trends over time. Built to grow into a wellbeing / productivity-analytics product.

**Privacy:** 100% local. No video or frames ever leave your machine or get written
to disk — only derived numbers (emotion label, posture angle, focus score) go into
a local SQLite database (`deskcompanion.db`, gitignored). The only network access
is a one-time download of the open-source FER+ emotion model (~34MB) on first run.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full design.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Edit `config.yaml`: set camera indices under `cameras:` (check with
`ls /dev/video*`; usually the laptop webcam is `0`).

### Pixel 7 Pro as second webcam

Connect the phone by USB, enable **Developer options → USB preferences →
Webcam** (Android 14+ native webcam mode). It shows up as a new `/dev/videoN`
device — add it to `config.yaml`:

```yaml
cameras:
  laptop: 0
  pixel: 2
```

## Run

```bash
# 1. (once) calibrate your upright posture, ~5 seconds
.venv/bin/python -m deskcompanion.calibrate

# 2. start tracking (q in preview window or Ctrl-C to stop)
.venv/bin/python -m deskcompanion.runner

# 3. dashboard
.venv/bin/streamlit run deskcompanion/dashboard.py
```

## Self-checks

Each module ships an assert-based check:

```bash
.venv/bin/python -m deskcompanion.store
.venv/bin/python -m deskcompanion.capture          # needs a camera
.venv/bin/python -m deskcompanion.analyzers.emotion
.venv/bin/python -m deskcompanion.analyzers.posture
.venv/bin/python -m deskcompanion.analyzers.attention
.venv/bin/python -m deskcompanion.analyzers.activity
```

## Signals logged

One flat table: `metrics(session_id, ts, source, signal, value, label)`.

| signal | value | label |
|---|---|---|
| `emotion` | confidence 0–1 | neutral/happy/sad/angry/… |
| `posture_angle` | degrees | neck / shoulder_tilt / slouching |
| `presence` | 0 or 1 | |
| `gaze` | 0–1 (0.5 = at screen) | |
| `focus` | 0–1 | |
| `activity` | 1 | working / distracted / away |
