"""Posture calibration: sit upright, run this for ~5s, it writes your neutral
neck angle into config.yaml. Run: python -m deskcompanion.calibrate"""
import statistics
import time

import yaml

from .capture import Cameras
from .analyzers.posture import PostureAnalyzer
from .runner import CONFIG_PATH, load_config


def main(seconds: float = 5.0):
    cfg = load_config()
    cams = Cameras(cfg["cameras"])
    analyzer = PostureAnalyzer()
    print(f"[calibrate] sit upright, capturing for {seconds:.0f}s ...")
    angles = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        for source, frame in cams.read():
            for m in analyzer.analyze(source, frame):
                if m.label == "neck":
                    angles.append(m.value)
    cams.release()
    if not angles:
        print("[calibrate] no pose detected — check camera and lighting")
        return
    neutral = statistics.median(angles)
    cfg["posture_neutral"]["neck_angle"] = round(neutral, 1)
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[calibrate] neutral neck angle = {neutral:.1f}° written to config.yaml "
          f"({len(angles)} samples)")


if __name__ == "__main__":
    main()
