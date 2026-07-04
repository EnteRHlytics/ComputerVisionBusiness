"""SQLite metric store. Only derived numbers land here — never frames."""
import sqlite3
import time
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY,
  started_at REAL NOT NULL,
  ended_at REAL,
  note TEXT
);
CREATE TABLE IF NOT EXISTS metrics(
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES sessions(id),
  ts REAL NOT NULL,
  source TEXT NOT NULL,
  signal TEXT NOT NULL,
  value REAL,
  label TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_session_ts ON metrics(session_id, ts);
"""


@dataclass
class Metric:
    source: str      # camera name
    signal: str      # emotion | activity | posture_angle | gaze | presence | focus
    value: float | None = None
    label: str | None = None


class Store:
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self.db.executescript(SCHEMA)
        self.session_id: int | None = None

    def start_session(self, note: str = "") -> int:
        cur = self.db.execute(
            "INSERT INTO sessions(started_at, note) VALUES(?, ?)", (time.time(), note))
        self.db.commit()
        self.session_id = cur.lastrowid
        return self.session_id

    def end_session(self):
        if self.session_id is not None:
            self.db.execute("UPDATE sessions SET ended_at=? WHERE id=?",
                            (time.time(), self.session_id))
            self.db.commit()
            self.session_id = None

    def write(self, metrics: list[Metric], ts: float | None = None):
        assert self.session_id is not None, "start_session() first"
        ts = ts if ts is not None else time.time()
        self.db.executemany(
            "INSERT INTO metrics(session_id, ts, source, signal, value, label) VALUES(?,?,?,?,?,?)",
            [(self.session_id, ts, m.source, m.signal, m.value, m.label) for m in metrics])
        self.db.commit()

    def close(self):
        self.end_session()
        self.db.close()


def demo():
    """Self-check: write and read back a metric in a temp db."""
    s = Store(":memory:")
    s.start_session("test")
    s.write([Metric("laptop", "emotion", 0.9, "happy"),
             Metric("laptop", "focus", 0.7)])
    rows = s.db.execute("SELECT signal, value, label FROM metrics ORDER BY id").fetchall()
    assert rows == [("emotion", 0.9, "happy"), ("focus", 0.7, None)], rows
    s.end_session()
    ended = s.db.execute("SELECT ended_at FROM sessions").fetchone()[0]
    assert ended is not None
    print("[store] OK")


if __name__ == "__main__":
    demo()
