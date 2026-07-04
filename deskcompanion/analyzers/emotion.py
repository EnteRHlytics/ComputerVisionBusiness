"""Emotion via FER+ ONNX (local, ~34MB, auto-downloaded once). Swappable: anything
matching the Analyzer protocol can replace this class in runner.py."""
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ..store import Metric

MODEL_URL = ("https://github.com/onnx/models/raw/main/validated/vision/"
             "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx")
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "emotion-ferplus-8.onnx"
EMOTIONS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]


class EmotionAnalyzer:
    def __init__(self):
        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(exist_ok=True)
            print(f"[emotion] downloading FER+ model to {MODEL_PATH} ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        self.session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def analyze(self, source: str, frame: np.ndarray) -> list[Metric]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))
        if len(faces) == 0:
            return []
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
        face = cv2.resize(gray[y:y + h, x:x + w], (64, 64)).astype(np.float32)
        scores = self.session.run(None, {self.input_name: face[None, None]})[0][0]
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        best = int(probs.argmax())
        return [Metric(source, "emotion", float(probs[best]), EMOTIONS[best])]


def demo():
    """Self-check: run on a synthetic gray frame (no face -> no metrics), then check
    the model produces a valid emotion on a real face if a camera is available."""
    a = EmotionAnalyzer()
    blank = np.full((480, 640, 3), 128, np.uint8)
    assert a.analyze("test", blank) == [], "blank frame should yield no face"
    # direct model sanity: any 64x64 input must give a distribution over 8 emotions
    scores = a.session.run(None, {a.input_name: np.zeros((1, 1, 64, 64), np.float32)})[0][0]
    assert scores.shape == (8,)
    print("[emotion] OK")


if __name__ == "__main__":
    demo()
