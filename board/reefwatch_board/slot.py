"""A single-item, newest-wins hand-off between two threads."""
from __future__ import annotations

import threading
import time


class NewestSlot:
    """`put` never blocks: an item nobody has taken yet is replaced and counted in `dropped`.
    `get` blocks until an item is there (or the timeout / `close`) and takes it."""

    def __init__(self):
        self.cond = threading.Condition()
        self.item = None
        self.posted = 0
        self.taken = 0
        self.dropped = 0
        self.closed = False

    def put(self, item):
        """-> the item this one replaced (also counted in `dropped`), or None."""
        with self.cond:
            old = self.item
            if old is not None:
                self.dropped += 1
            self.item = item
            self.posted += 1
            self.cond.notify()
            return old

    def get(self, timeout=None):
        """-> the newest item, or None on timeout / close."""
        end = None if timeout is None else time.monotonic() + timeout
        with self.cond:
            while self.item is None and not self.closed:
                left = None if end is None else end - time.monotonic()
                if left is not None and left <= 0:
                    return None
                self.cond.wait(left)
            item, self.item = self.item, None
            if item is not None:
                self.taken += 1
            return item

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()
