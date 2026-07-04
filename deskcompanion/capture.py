"""Camera capture: N webcams by device index, one code path for all of them."""
import time

import cv2


class Cameras:
    def __init__(self, cameras: dict[str, int]):
        self.caps: dict[str, cv2.VideoCapture] = {}
        for name, idx in cameras.items():
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                print(f"[capture] WARNING: camera '{name}' (index {idx}) failed to open, skipping")
                continue
            self.caps[name] = cap
        if not self.caps:
            raise RuntimeError("no cameras opened — check indices in config.yaml")

    def read(self):
        """Yield (source_name, frame_bgr) for every camera that returned a frame."""
        for name, cap in self.caps.items():
            ok, frame = cap.read()
            if ok:
                yield name, frame

    def release(self):
        for cap in self.caps.values():
            cap.release()


def demo():
    """Self-check: open camera 0, grab 10 frames, print FPS."""
    cams = Cameras({"laptop": 0})
    t0 = time.time()
    n = 0
    for _ in range(10):
        for _name, frame in cams.read():
            assert frame.ndim == 3 and frame.shape[2] == 3, "expected BGR frame"
            n += 1
    cams.release()
    assert n > 0, "no frames captured"
    print(f"[capture] OK — {n} frames, {n / (time.time() - t0):.1f} FPS")


if __name__ == "__main__":
    demo()
