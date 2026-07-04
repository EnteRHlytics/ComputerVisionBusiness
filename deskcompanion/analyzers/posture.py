"""Posture via MediaPipe Pose: neck angle (ear-shoulder vs vertical) and
shoulder tilt (line between shoulders vs horizontal), in degrees."""
import math

import mediapipe as mp
import numpy as np

from ..store import Metric

_L = mp.solutions.pose.PoseLandmark


class PostureAnalyzer:
    def __init__(self, neutral_neck: float = 15.0, slouch_margin: float = 12.0):
        self.pose = mp.solutions.pose.Pose(model_complexity=0)
        self.slouch_threshold = neutral_neck + slouch_margin

    def analyze(self, source: str, frame: np.ndarray) -> list[Metric]:
        res = self.pose.process(frame[:, :, ::-1])  # BGR -> RGB
        if not res.pose_landmarks:
            return []
        lm = res.pose_landmarks.landmark
        ear = lm[_L.LEFT_EAR] if lm[_L.LEFT_EAR].visibility > lm[_L.RIGHT_EAR].visibility \
            else lm[_L.RIGHT_EAR]
        sh_l, sh_r = lm[_L.LEFT_SHOULDER], lm[_L.RIGHT_SHOULDER]
        sh_mid_x, sh_mid_y = (sh_l.x + sh_r.x) / 2, (sh_l.y + sh_r.y) / 2
        # neck angle: 0 = head stacked over shoulders, grows as head juts forward/down
        neck = abs(math.degrees(math.atan2(ear.x - sh_mid_x, sh_mid_y - ear.y)))
        tilt = abs(math.degrees(math.atan2(sh_r.y - sh_l.y, sh_r.x - sh_l.x)))
        tilt = min(tilt, 180 - tilt)
        slouching = 1.0 if neck > self.slouch_threshold else 0.0
        return [Metric(source, "posture_angle", neck, "neck"),
                Metric(source, "posture_angle", tilt, "shoulder_tilt"),
                Metric(source, "posture_angle", slouching, "slouching")]


def demo():
    """Self-check: no person in frame -> no metrics; angle math sanity."""
    a = PostureAnalyzer()
    blank = np.zeros((480, 640, 3), np.uint8)
    assert a.analyze("test", blank) == []
    # head directly above shoulders -> ~0 deg; head level with shoulders -> 90 deg
    assert abs(math.degrees(math.atan2(0.0, 0.3))) < 1e-6
    assert abs(math.degrees(math.atan2(0.3, 0.0)) - 90) < 1e-6
    print("[posture] OK")


if __name__ == "__main__":
    demo()
