"""Attention via MediaPipe Face Mesh (iris refinement on): presence, gaze
centering, eye-open ratio -> single focus score in [0, 1]. Also emits the
face center (nose tip, normalized 0-1) as face_x/face_y for tracking."""
import mediapipe as mp
import numpy as np

from ..store import Metric

# face mesh landmark indices
L_EYE = dict(inner=133, outer=33, top=159, bottom=145, iris=468)
R_EYE = dict(inner=362, outer=263, top=386, bottom=374, iris=473)


def _eye_signals(lm, eye):
    corner_a, corner_b = lm[eye["outer"]], lm[eye["inner"]]
    width = abs(corner_b.x - corner_a.x) or 1e-6
    open_ratio = abs(lm[eye["top"]].y - lm[eye["bottom"]].y) / width
    gaze = (lm[eye["iris"]].x - min(corner_a.x, corner_b.x)) / width  # 0.5 = centered
    return open_ratio, gaze


class AttentionAnalyzer:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

    def analyze(self, source: str, frame: np.ndarray) -> list[Metric]:
        res = self.mesh.process(frame[:, :, ::-1])
        if not res.multi_face_landmarks:
            return [Metric(source, "presence", 0.0),
                    Metric(source, "focus", 0.0)]
        lm = res.multi_face_landmarks[0].landmark
        open_l, gaze_l = _eye_signals(lm, L_EYE)
        open_r, gaze_r = _eye_signals(lm, R_EYE)
        eye_open = (open_l + open_r) / 2          # ~0.3 open, <0.12 closed
        gaze = (gaze_l + gaze_r) / 2              # 0.5 = looking at screen
        eyes_open = min(eye_open / 0.25, 1.0)
        gaze_centered = max(0.0, 1.0 - abs(gaze - 0.5) / 0.25)
        focus = eyes_open * gaze_centered
        nose = lm[1]  # nose tip = face center, normalized image coords
        return [Metric(source, "presence", 1.0),
                Metric(source, "gaze", gaze),
                Metric(source, "focus", focus),
                Metric(source, "face_x", nose.x),
                Metric(source, "face_y", nose.y)]


def demo():
    """Self-check: empty frame -> presence 0, focus 0."""
    a = AttentionAnalyzer()
    blank = np.zeros((480, 640, 3), np.uint8)
    out = {m.signal: m.value for m in a.analyze("test", blank)}
    assert out == {"presence": 0.0, "focus": 0.0}, out
    print("[attention] OK")


if __name__ == "__main__":
    demo()
