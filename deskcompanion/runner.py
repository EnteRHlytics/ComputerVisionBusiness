"""Main loop: capture -> analyzers -> sqlite, throttled to sample_hz.
Run: python -m deskcompanion.runner   (q in preview window or Ctrl-C to stop)"""
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import yaml

from .capture import Cameras
from .store import Metric, Store
from .analyzers.activity import ActivityClassifier
from .analyzers.attention import AttentionAnalyzer
from .analyzers.emotion import EmotionAnalyzer
from .analyzers.posture import PostureAnalyzer

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def build_analyzers(cfg: dict) -> list:
    on = cfg["analyzers"]
    analyzers = []
    if on.get("emotion"):
        analyzers.append(EmotionAnalyzer())
    if on.get("posture"):
        pn = cfg["posture_neutral"]
        analyzers.append(PostureAnalyzer(pn["neck_angle"], pn["slouch_margin"]))
    if on.get("attention"):
        analyzers.append(AttentionAnalyzer())
    return analyzers


def main():
    cfg = load_config()
    cams = Cameras(cfg["cameras"])
    store = Store(cfg["db_path"])
    store.start_session()
    analyzers = build_analyzers(cfg)
    attention = next((a for a in analyzers if isinstance(a, AttentionAnalyzer)), None)
    activity = ActivityClassifier(**cfg["activity_rules"]) if cfg["analyzers"].get("activity") else None
    interval = 1.0 / cfg["sample_hz"]
    print(f"[runner] session {store.session_id} started — {len(cams.caps)} camera(s), "
          f"{cfg['sample_hz']} Hz. Ctrl-C to stop.")
    frames_done = 0
    t_report = time.time()
    trails: dict[str, deque] = defaultdict(lambda: deque(maxlen=30))  # face trail per camera
    try:
        while True:
            tick = time.time()
            for source, frame in cams.read():
                metrics: list[Metric] = []
                for a in analyzers:
                    metrics.extend(a.analyze(source, frame))
                if activity:
                    latest = {m.signal: m.value for m in metrics}
                    metrics.append(activity.classify(source, latest))
                if metrics:
                    store.write(metrics, ts=tick)
                if cfg.get("show_preview"):
                    label = " | ".join(f"{m.signal}:{m.label or round(m.value, 2)}"
                                       for m in metrics if m.signal in ("emotion", "activity", "focus"))
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    h, w = frame.shape[:2]
                    # face bounding box: green normally, red + warning when too close
                    if attention and source in attention.last_box:
                        bx, by, bw, bh = attention.last_box[source]
                        too_close = bw > cfg["too_close_face_size"]
                        color = (0, 0, 255) if too_close else (0, 255, 0)
                        cv2.rectangle(frame, (int(bx * w), int(by * h)),
                                      (int((bx + bw) * w), int((by + bh) * h)), color, 2)
                        if too_close:
                            cv2.putText(frame, "TOO CLOSE TO SCREEN", (10, h - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # face tracking trail: dot at current face center, fading line behind it
                    vals = {m.signal: m.value for m in metrics}
                    if "face_x" in vals:
                        trails[source].append((int(vals["face_x"] * w), int(vals["face_y"] * h)))
                    trail = trails[source]
                    for i in range(1, len(trail)):
                        cv2.line(frame, trail[i - 1], trail[i], (0, 128 + 4 * i, 255 - 8 * i), 2)
                    if trail and "face_x" in vals:
                        cv2.circle(frame, trail[-1], 6, (0, 255, 255), -1)
                    cv2.imshow(f"deskcompanion:{source}", frame)
                frames_done += 1
            if cfg.get("show_preview") and cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if time.time() - t_report > 10:
                print(f"[runner] {frames_done / (time.time() - t_report):.1f} FPS analyzed")
                frames_done, t_report = 0, time.time()
            time.sleep(max(0.0, interval - (time.time() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        store.close()
        cams.release()
        cv2.destroyAllWindows()
        print("[runner] session ended cleanly")


if __name__ == "__main__":
    main()
