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
cols = st.columns(3)
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

if not activity.empty:
    st.subheader("Activity")
    left, right = st.columns([2, 1])
    left.scatter_chart(activity.assign(state=activity.label), x="time", y="state")
    right.bar_chart(activity.label.value_counts())
