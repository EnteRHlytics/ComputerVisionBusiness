"""Streamlit dashboard over the local sqlite metric stream.
Run: streamlit run deskcompanion/dashboard.py"""
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

CFG = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())

st.set_page_config(page_title="Desk Companion", layout="wide")
st.title("Desk Companion")

db = sqlite3.connect(CFG["db_path"])
sessions = pd.read_sql("SELECT * FROM sessions ORDER BY started_at DESC", db)
if sessions.empty:
    st.info("No sessions yet — run `python -m deskcompanion.runner` first.")
    st.stop()

sessions["label"] = pd.to_datetime(sessions.started_at, unit="s").dt.strftime("%Y-%m-%d %H:%M")
sid = st.selectbox("Session", sessions.id, format_func=lambda i:
                   f"#{i} — {sessions.set_index('id').loc[i, 'label']}")
df = pd.read_sql("SELECT * FROM metrics WHERE session_id=?", db, params=(sid,))
if df.empty:
    st.info("Session has no metrics.")
    st.stop()
df["time"] = pd.to_datetime(df.ts, unit="s")

# --- wellbeing callouts ---
slouch = df[(df.signal == "posture_angle") & (df.label == "slouching")]
activity = df[df.signal == "activity"]
face_size = df[df.signal == "face_size"]
cols = st.columns(4)
if not face_size.empty:
    too_close_pct = 100 * (face_size.value > CFG["too_close_face_size"]).mean()
    cols[3].metric("Too close to screen", f"{too_close_pct:.0f}% of session")
    if too_close_pct > 20:
        st.warning("You spend a lot of time leaning into the screen — sit back to reduce eye strain.")
if not slouch.empty:
    pct = 100 * slouch.value.mean()
    cols[0].metric("Slouching", f"{pct:.0f}% of session")
    if pct > 30:
        st.warning("Poor posture for a big chunk of this session — adjust your chair/screen.")
if not activity.empty:
    working = activity[activity.label == "working"]
    cols[1].metric("Working", f"{100 * len(working) / len(activity):.0f}% of samples")
    span_min = (df.ts.max() - df.ts.min()) / 60
    cols[2].metric("Session length", f"{span_min:.0f} min")
    if span_min > 50 and (activity.tail(100).label == "working").mean() > 0.8:
        st.warning("You've been at it a while — take a short break.")

# --- timelines ---
focus = df[df.signal == "focus"]
if not focus.empty:
    st.subheader("Focus")
    st.line_chart(focus.set_index("time").value)

posture = df[(df.signal == "posture_angle") & (df.label != "slouching")]
if not posture.empty:
    st.subheader("Posture angles (°)")
    st.line_chart(posture.pivot_table(index="time", columns="label", values="value"))

emotion = df[df.signal == "emotion"]
if not emotion.empty:
    st.subheader("Emotion")
    left, right = st.columns([2, 1])
    left.scatter_chart(emotion.assign(emotion=emotion.label), x="time", y="emotion")
    right.bar_chart(emotion.label.value_counts())

face = df[df.signal.isin(["face_x", "face_y"])]
if not face.empty:
    st.subheader("Face tracking")
    pos = face.pivot_table(index="time", columns="signal", values="value")
    left, right = st.columns([2, 1])
    left.line_chart(pos)  # x/y drift over time — see where the face is going
    # position map in camera coords (y flipped so up on chart = up in reality)
    right.scatter_chart(pos.assign(face_y=1 - pos.face_y), x="face_x", y="face_y")
    if not face_size.empty:
        st.caption(f"Screen distance (face width fraction; above "
                   f"{CFG['too_close_face_size']} = too close)")
        st.line_chart(face_size.set_index("time").value)

if not activity.empty:
    st.subheader("Activity")
    left, right = st.columns([2, 1])
    left.scatter_chart(activity.assign(state=activity.label), x="time", y="state")
    right.bar_chart(activity.label.value_counts())
