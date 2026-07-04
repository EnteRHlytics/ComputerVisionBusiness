"""Main loop: capture -> analyzers -> sqlite, throttled to sample_hz.
Run: python -m deskcompanion.runner   (q in preview window or Ctrl-C to stop)"""
import time
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
    activity = ActivityClassifier(**cfg["activity_rules"]) if cfg["analyzers"].get("activity") else None
    interval = 1.0 / cfg["sample_hz"]
    print(f"[runner] session {store.session_id} started — {len(cams.caps)} camera(s), "
          f"{cfg['sample_hz']} Hz. Ctrl-C to stop.")
    frames_done = 0
    t_report = time.time()
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
