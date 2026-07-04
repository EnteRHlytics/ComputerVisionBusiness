"""Rule-based activity state from the other analyzers' latest outputs.
States: working | distracted | away. Consumes metrics, not frames."""
import time

from ..store import Metric


class ActivityClassifier:
    def __init__(self, away_after_s: float = 5.0, distracted_focus: float = 0.4):
        self.away_after_s = away_after_s
        self.distracted_focus = distracted_focus
        self.last_present: dict[str, float] = {}  # source -> ts last seen

    def classify(self, source: str, latest: dict[str, float], now: float | None = None) -> Metric:
        """latest: {signal: value} from this tick. Returns one activity metric."""
        now = now if now is not None else time.time()
        if latest.get("presence", 0.0) > 0:
            self.last_present[source] = now
        seen = self.last_present.get(source)
        if seen is None or now - seen > self.away_after_s:
            state = "away"
        elif latest.get("focus", 1.0) < self.distracted_focus:
            state = "distracted"
        else:
            state = "working"
        return Metric(source, "activity", 1.0, state)


def demo():
    """Self-check: present+focused -> working; low focus -> distracted; gone -> away."""
    c = ActivityClassifier(away_after_s=5, distracted_focus=0.4)
    assert c.classify("cam", {"presence": 1.0, "focus": 0.9}, now=100.0).label == "working"
    assert c.classify("cam", {"presence": 1.0, "focus": 0.1}, now=101.0).label == "distracted"
    assert c.classify("cam", {"presence": 0.0}, now=102.0).label == "working"  # grace period
    assert c.classify("cam", {"presence": 0.0}, now=110.0).label == "away"
    assert c.classify("new", {"presence": 0.0}, now=50.0).label == "away"  # never seen
    print("[activity] OK")


if __name__ == "__main__":
    demo()
