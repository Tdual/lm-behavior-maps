"""On-disk memo for scores.

Scoring is the only expensive step and it is perfectly reproducible, so it is
cached from the start rather than bolted on. Re-running a map after adding one
agent, changing the embedding, or fixing an analysis bug then costs nothing for
the agents that did not change.

sqlite is used rather than a pile of files: lookups stay fast at a few million
rows, and a partly-written run leaves a usable database behind.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
from pathlib import Path

DEFAULT_DIR = Path(os.environ.get("LM_BEHAVIOR_CACHE", ".lm_behavior_cache"))


def _key(fingerprint: str, system: str | None, body: str, action: str) -> bytes:
    h = hashlib.blake2b(digest_size=20)
    for part in (fingerprint, system or "", body, action):
        h.update(str(len(part)).encode())
        h.update(b"\x00")
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.digest()


class ScoreCache:
    """Maps (fingerprint, system, body, action) to one float."""

    def __init__(self, path: str | os.PathLike | None = None, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            self.conn = None
            return
        p = Path(path) if path is not None else DEFAULT_DIR / "scores.sqlite3"
        p.parent.mkdir(parents=True, exist_ok=True)
        self.path = p
        self.conn = sqlite3.connect(str(p))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS scores (k BLOB PRIMARY KEY, v BLOB)")
        self.conn.commit()

    def get(self, fingerprint: str, system: str | None, body: str, action: str) -> float | None:
        if not self.enabled or self.conn is None:
            return None
        row = self.conn.execute(
            "SELECT v FROM scores WHERE k=?", (_key(fingerprint, system, body, action),)
        ).fetchone()
        return struct.unpack("<d", row[0])[0] if row else None

    def put_many(self, fingerprint: str, system: str | None, items) -> None:
        """items: iterable of (body, action, value)."""
        if not self.enabled or self.conn is None:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO scores (k, v) VALUES (?, ?)",
            [
                (_key(fingerprint, system, b, a), struct.pack("<d", float(v)))
                for b, a, v in items
            ],
        )
        self.conn.commit()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
