# Desk Companion — Local CV Activity & Emotion Tracker

## Context

A computer-vision system that watches you at your desk using two cameras (laptop
webcam + Pixel 7 Pro connected as a webcam), figures out **what you're doing**,
reads your **facial emotion**, checks your **posture**, and logs it all so you
can track trends over time. Modular enough to grow into a sellable product for
**worker wellbeing** and **productivity analytics**.

Hard constraints:
- **100% local.** No video or frames leave the machine. Only derived numbers are stored.
- **Both cameras are standard webcams** — one capture code path, no phone SDK.
- Keep it modular so wellbeing / productivity / measurement features bolt on later.

## Approach (all local / CPU)

One capture-and-analyze loop feeds a generic metrics table in SQLite. A separate
Streamlit app reads that table for dashboards. No cloud, no frame storage.

**Stack:**
- `opencv-python` — camera capture (both webcams are device indices).
- `mediapipe` — face mesh, pose, and gaze landmarks. Covers posture
  (shoulder/neck angles), attention (gaze + eye-open ratio), and face-presence.
- Emotion: `fer` (or a small ONNX FER model) behind a swappable `EmotionAnalyzer`.
- Activity: **rule-based first** from existing signals (present + gaze-on-screen +
  hands-near-keyboard → "working"; face gone → "away").
- `sqlite3` (stdlib) — local storage.
- `streamlit` — dashboard.

Everything downstream of capture is a flat metric stream
`(session_id, ts, source, signal, value, label)`. Emotion tracking,
productivity, and wellbeing are all different queries over that one table.

### Data model
```
metrics(id, session_id, ts, source, signal, value REAL, label TEXT)
   signal ∈ {emotion, activity, posture_angle, gaze, presence, focus}
sessions(id, started_at, ended_at, note)
```
Frames are **never** written to disk.

### Module layout
```
deskcompanion/
  capture.py      # opens N webcams, yields frames per source
  analyzers/
    base.py       # Analyzer protocol: analyze(frame) -> list[Metric]
    emotion.py    # FER, swappable
    posture.py    # mediapipe pose -> neck/shoulder angles
    attention.py  # mediapipe face mesh -> gaze + eye-open -> focus score
    activity.py   # rule-based state from the other analyzers' outputs
  store.py        # sqlite writer, Metric dataclass
  runner.py       # main loop: capture -> analyzers -> store, ~2 Hz
  dashboard.py    # streamlit app over sqlite
  config.yaml     # camera indices, sample rate, which analyzers on
```

## Build order (tracked as GitHub issues #1–#8)

1. **Project scaffold + capture** — `requirements.txt`, `config.yaml`,
   `capture.py` opening both webcams by index, `runner.py` showing live frames
   with FPS. Foundation for all others.
2. **SQLite store + metric model** — `store.py`, `Metric` dataclass, schema,
   session start/stop, write/read self-check.
3. **Emotion analyzer** — `analyzers/emotion.py` behind the `Analyzer` protocol,
   wired into `runner.py`, emotions logged to SQLite. Swappable backend.
4. **Posture analyzer** — MediaPipe pose → neck/shoulder angle metrics +
   calibration step for neutral posture.
5. **Attention analyzer** — MediaPipe face mesh → gaze + eye-open ratio → `focus`
   score, logged.
6. **Activity classifier (rule-based)** — combine presence/gaze/posture/hands
   into states (working / on a call / away / distracted). Rules in config.
7. **Streamlit dashboard** — timeline of emotion, focus, posture, activity;
   break-reminder + poor-posture callouts (wellbeing hook).
8. **Packaging + run docs** — one command to start capture, one for the
   dashboard; README with Pixel-as-webcam setup and privacy statement.

Business features (wellbeing scoring, productivity report export, multi-day
trends) are later issues — queries and views over the same metric stream, no new
capture code.

## Verification
- **Capture:** both webcams show live feed, FPS logged; cover a camera → feed drops cleanly.
- **Emotion:** smile/frown → rows land in SQLite with matching labels.
- **Posture:** slouch → neck angle metric crosses calibrated threshold.
- **Attention:** look away → focus score drops.
- **Activity:** leave desk → "away"; return → "working".
- **Dashboard:** `streamlit run dashboard.py` renders charts from a real session.
- **Privacy check:** confirm no image files are written during a run.
- Each analyzer ships one `assert`-based self-check.

## Deliberately skipped (add when needed)
- Cloud sync / remote dashboard — 100% local by choice.
- Video action-recognition model for activity — rules cover v1.
- Body-measurement feature — parked; MediaPipe pose already gives the landmarks.
- Auth / multi-tenant — only when sold as a hosted product.
